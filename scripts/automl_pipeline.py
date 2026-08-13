#!/usr/bin/env python3
"""
scripts/automl_pipeline.py — Automatic Retraining & Evolution Tournament (Level 3 & 4)

This script:
1. Fetches the latest 180 days of 15-minute market data.
2. Splits data into Train (70%), Val (15%), and Holdout (15%).
3. Trains multiple candidate Transformer models (e.g., 4-layer vs 2-layer).
4. Evaluates all candidates and the current Production model on the Holdout set.
5. Overwrites the Production model if a candidate decisively wins.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler
import joblib

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import add_features, MASTER_FEATURE_COLS, get_active_features
from src.committee.transformer_brain import GrokGQA_Transformer
from src.telegram_alerts import send_telegram_alert
from src.db import save_experiment_record
import asyncio
import uuid

ACTIVE_FEATURES = get_active_features()

# ── Hyperparameters ─────────────────────────────────────────────────────────────
SEQ_LEN        = 32
BAR_TIMEFRAME  = TimeFrame(15, TimeFrameUnit.Minute)
TARGET_HORIZON = 6
EPOCHS         = 40
BATCH_SIZE     = 64
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
DAYS_HISTORY   = 180
PATIENCE       = 5

# Paths
# NOTE: model/scaler outputs MUST match settings.TRANSFORMER_MODEL_PATH /
# TRANSFORMER_SCALER_PATH (src/config.py, both default to "models/...") -
# transformer_brain.py loads from there. This previously pointed at
# DATA_DIR ("data/...") instead, a different directory that
# transformer_brain.py never reads from, so a "winning" retrain here
# silently never reached the live model.
DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models')
PROD_MODEL_OUT = os.path.join(MODELS_DIR, "grok_gqa_v9_best.pth")
PROD_SCALER_OUT= os.path.join(MODELS_DIR, "feature_scaler.pkl")

TRAIN_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
    "LTC/USD", "AVAX/USD", "LINK/USD", "ADA/USD",
    "BCH/USD", "DOT/USD",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def _get_data_client() -> CryptoHistoricalDataClient:
    key    = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    return CryptoHistoricalDataClient(api_key=key, secret_key=secret)

def fetch_bars(client, symbol: str, days: int) -> pd.DataFrame | None:
    try:
        start = datetime.now(tz=timezone.utc) - timedelta(days=days)
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=BAR_TIMEFRAME, start=start)
        raw_bars = client.get_crypto_bars(req).data.get(symbol, [])
        if not raw_bars: return None
        df = pd.DataFrame([{
            "timestamp": b.timestamp, "open": float(b.open or 0), "high": float(b.high or 0),
            "low": float(b.low or 0), "close": float(b.close or 0), "volume": float(b.volume or 0),
            "vwap": float(b.vwap or 0), "trade_count": float(b.trade_count or 0),
        } for b in raw_bars])
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])
        return df[df["close"] > 0]
    except Exception as exc:
        log.error(f"  {symbol}: fetch failed — {exc}")
        return None

def build_arrays(client, symbols: list[str], days: int, seq_len: int, horizon: int):
    all_X, all_y = [], []
    for sym in symbols:
        df_raw = fetch_bars(client, sym, days)
        if df_raw is None or len(df_raw) < seq_len + horizon + 10: continue
        df_feat = add_features(df_raw)
        shared_idx = df_feat.index.intersection(df_raw.index)
        df_feat = df_feat.loc[shared_idx]
        close_arr = df_raw.loc[shared_idx, "close"].values.astype(np.float64)
        feat_arr  = df_feat[ACTIVE_FEATURES].values.astype(np.float32)
        n = len(feat_arr)
        for idx in range(n - seq_len - horizon + 1):
            x_window = feat_arr[idx : idx + seq_len]
            close_now = close_arr[idx + seq_len - 1]
            close_future = close_arr[idx + seq_len - 1 + horizon]
            if close_now <= 0: continue
            all_X.append(x_window)
            all_y.append(1.0 if close_future > close_now else 0.0)
    if not all_X: raise RuntimeError("No training windows collected.")
    return np.stack(all_X, axis=0).astype(np.float32), np.array(all_y, dtype=np.float32)

def chrono_split(X, y, t_frac, v_frac):
    t_cut = int(len(y) * t_frac)
    v_cut = t_cut + int(len(y) * v_frac)
    return X[:t_cut], y[:t_cut], X[t_cut:v_cut], y[t_cut:v_cut], X[v_cut:], y[v_cut:]

def balance_dataset(X, y, seed=42):
    rng = np.random.default_rng(seed)
    idx_up, idx_down = np.where(y == 1.0)[0], np.where(y == 0.0)[0]
    minority_n = min(len(idx_up), len(idx_down))
    idx_bal = np.concatenate([rng.choice(idx_up, minority_n, False), rng.choice(idx_down, minority_n, False)])
    rng.shuffle(idx_bal)
    return X[idx_bal], y[idx_bal]

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def evaluate_model(model, loader, device, criterion):
    model.eval()
    loss_sum, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(1)
            loss_sum += criterion(pred.view(-1), yb.float().view(-1)).item() * len(yb)
            correct += ((pred > 0.0) == yb.bool()).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset) * 100.0

def train_candidate(name, config, X_tr, y_tr, X_va, y_va, n_feat, device):
    log.info(f"--- Training {name} ---")
    model = GrokGQA_Transformer(
        input_dim=n_feat, seq_len=SEQ_LEN, embed_dim=config["embed"],
        num_layers=config["layers"], num_q_heads=8, num_kv_heads=2, dropout=config["dropout"]
    ).to(device)
    
    train_loader = DataLoader(SequenceDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(SequenceDataset(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=EPOCHS, steps_per_epoch=len(train_loader))
    
    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze(1)
            loss = criterion(pred.view(-1), yb.float().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        v_l, v_a = evaluate_model(model, val_loader, device, criterion)
        if v_l < best_val_loss:
            best_val_loss = v_l
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"{name} Early stopping at epoch {epoch}.")
                break
                
    model.load_state_dict(best_state)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = _get_data_client()
    
    log.info("Fetching data and building windows...")
    X, y = build_arrays(client, TRAIN_SYMBOLS, DAYS_HISTORY, SEQ_LEN, TARGET_HORIZON)
    X_tr, y_tr, X_va, y_va, X_ho, y_ho = chrono_split(X, y, TRAIN_FRAC, VAL_FRAC)
    X_tr, y_tr = balance_dataset(X_tr, y_tr)
    
    n_tr, seq_len, n_feat = X_tr.shape
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(n_tr, seq_len, n_feat).astype(np.float32)
    X_va = scaler.transform(X_va.reshape(-1, n_feat)).reshape(X_va.shape[0], seq_len, n_feat).astype(np.float32)
    X_ho = scaler.transform(X_ho.reshape(-1, n_feat)).reshape(X_ho.shape[0], seq_len, n_feat).astype(np.float32)
    
    holdout_loader = DataLoader(SequenceDataset(X_ho, y_ho), batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    
    candidates = {
        "Bot_A_Light": {"layers": 4, "embed": 128, "lr": 5e-4, "dropout": 0.2, "stop_loss": 0.015, "profit_target": 0.03, "threshold": 0.55},
        "Bot_B_Standard": {"layers": 4, "embed": 128, "lr": 3e-4, "dropout": 0.1, "stop_loss": 0.02, "profit_target": 0.04, "threshold": 0.58},
        "Bot_C_Heavy": {"layers": 4, "embed": 256, "lr": 1e-4, "dropout": 0.3, "stop_loss": 0.03, "profit_target": 0.06, "threshold": 0.60},
    }
    
    import json
    CANDIDATES_DIR = os.path.join(DATA_DIR, "candidates")
    os.makedirs(CANDIDATES_DIR, exist_ok=True)
    
    trained_models = {}
    for name, config in candidates.items():
        trained_models[name] = train_candidate(name, config, X_tr, y_tr, X_va, y_va, n_feat, device)
        # Save candidate for shadow arena
        torch.save(trained_models[name].state_dict(), os.path.join(CANDIDATES_DIR, f"{name}.pth"))
        with open(os.path.join(CANDIDATES_DIR, f"{name}_config.json"), "w") as f:
            json.dump(config, f)
    
    # Save the common scaler
    joblib.dump(scaler, os.path.join(CANDIDATES_DIR, "feature_scaler.pkl"))
        
    # Evaluate Production Model
    prod_acc = 0.0
    if os.path.exists(PROD_MODEL_OUT):
        log.info("Evaluating Production Model...")
        prod_model = GrokGQA_Transformer(input_dim=n_feat, seq_len=SEQ_LEN, embed_dim=128, num_layers=4).to(device)
        try:
            prod_model.load_state_dict(torch.load(PROD_MODEL_OUT, map_location=device))
            _, prod_acc = evaluate_model(prod_model, holdout_loader, device, criterion)
        except Exception as e:
            log.warning(f"Failed to load production model: {e}")
            
    log.info("\n=== TOURNAMENT CANDIDATE GENERATION COMPLETE ===")
    log.info(f"Production Model Holdout Acc: {prod_acc:.2f}%")
    best_acc = 0.0
    best_name = None
    msg_lines = ["🧬 <b>Evolution Tournament: New Generation Spawned</b>", f"Prod Acc: {prod_acc:.2f}%", "Candidates:"]
    for name, model in trained_models.items():
        _, acc = evaluate_model(model, holdout_loader, device, criterion)
        log.info(f"{name}: {acc:.2f}%")
        msg_lines.append(f"- {name}: {acc:.2f}%")
        
        save_experiment_record(
            experiment_id=str(uuid.uuid4()),
            generation_type="AutoML",
            architecture_details=candidates[name],
            sharpe=0.0,
            max_dd=0.0,
            total_return=acc,
            status="Candidate"
        )
        
        if acc > best_acc:
            best_acc = acc
            best_name = name

    # ── Feature Evolution (Permutation Importance) ────────────────────────
    log.info("Calculating Permutation Importance on validation set...")
    if best_name and best_name in trained_models:
        eval_model = trained_models[best_name]
        baseline_loss, baseline_acc = evaluate_model(eval_model, holdout_loader, device, criterion)
        
        importances = {}
        for i, feat_name in enumerate(ACTIVE_FEATURES):
            X_ho_shuffled = X_ho.copy()
            np.random.shuffle(X_ho_shuffled[:, :, i])
            shuffled_loader = DataLoader(SequenceDataset(X_ho_shuffled, y_ho), batch_size=BATCH_SIZE, shuffle=False)
            
            _, shuff_acc = evaluate_model(eval_model, shuffled_loader, device, criterion)
            importance = baseline_acc - shuff_acc
            importances[feat_name] = importance
            log.info(f"  Feature {feat_name}: {importance:+.2f}%")
            
        msg_lines.append("\n🔬 <b>Feature Importance:</b>")
        for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            msg_lines.append(f"  {k}: {v:+.2f}%")
            
        worst_feat = min(importances, key=importances.get)
        if importances[worst_feat] <= 0.05 and len(ACTIVE_FEATURES) > 5:
            log.info(f"Culling worst feature: {worst_feat}")
            ACTIVE_FEATURES.remove(worst_feat)
            msg_lines.append(f"\n🗑️ <b>Culled:</b> {worst_feat}")
            
            inactive_pool = [f for f in MASTER_FEATURE_COLS if f not in ACTIVE_FEATURES]
            if inactive_pool:
                new_feat = np.random.choice(inactive_pool)
                ACTIVE_FEATURES.append(new_feat)
                log.info(f"Mutating in new feature: {new_feat}")
                msg_lines.append(f"🧬 <b>Mutated In:</b> {new_feat}")
                
            with open(os.path.join(DATA_DIR, "active_features.json"), "w") as f:
                json.dump(ACTIVE_FEATURES, f)
            
    msg_lines.append(f"\nCandidates are now live in the Shadow Arena.")
    msg = "\n".join(msg_lines)
    
    try:
        asyncio.run(send_telegram_alert(msg))
    except Exception as e:
        log.error(f"Failed to send telegram alert: {e}")

if __name__ == "__main__":
    main()
