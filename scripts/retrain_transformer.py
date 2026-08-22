#!/usr/bin/env python3
"""
Continuous Transformer Replay Training

Loads the JSONL dataset containing 32-step tensor inputs and true trade outcomes
recorded during actual backtesting/live trading. Continuously fine-tunes the 
Grok GQA model using experience replay.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import asyncio
import joblib
import polars as pl
import yfinance as yf
import traceback
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.committee.transformer_brain import GrokGQA_Transformer, set_ml_predictor_override, reset_ml_predictor
from src.backtest import run_backtest
from src.logging_config import get_logger, set_correlation_id

logger = get_logger("transformer_replay")

HISTORICAL_BUFFER_PATH = "data/historical_experiences.jsonl"
LIVE_BUFFER_PATH = "data/live_experiences.jsonl"
MODEL_PATH = "models/grok_gqa_v9_best.pth"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
MIN_SAMPLES = 50  # Minimum samples required for meaningful training
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
PATIENCE = 5  # Early stopping patience

class ReplayBufferDataset(Dataset):
    def __init__(self, data_paths, max_size_per_file=100000):
        self.samples = []
        self.labels = []
        for data_path in data_paths:
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    lines = f.readlines()
                    # Keep only the last `max_size_per_file` trades
                    for line in lines[-max_size_per_file:]:
                        try:
                            record = json.loads(line)
                            tensor_state = np.array(record["tensor"], dtype=np.float32)
                            label = float(record["label"])
                            self.samples.append((tensor_state, label))
                            self.labels.append(label)
                        except Exception as e:
                            pass
                             
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)
    
    def get_label_distribution(self):
        """Return label distribution for monitoring class balance."""
        return Counter(self.labels)

def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
    return total_loss / len(loader.dataset)

def fetch_validation_bars(symbol: str = "BTC-USD", days: int = 90) -> pl.DataFrame:
    """Fetch a recent, real historical window for walk-forward-style validation.
    This is intentionally a separate, more recent slice than what's used for
    training/holdout-loss evaluation, to give an independent read on real
    simulated trading performance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1h")
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "t", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    df["t"] = df["t"].astype(str)
    return pl.from_pandas(df)


async def run_real_validation(champion_model, challenger_model, scaler, device, input_dim,
                               symbol: str = "BTC-USD", days: int = 90):
    """Run both models through a real simulated backtest (via the full 5-brain
    committee) over the same real historical window, and return their results
    for direct comparison. This replaces fake/random validation with an actual
    measured outcome."""
    val_bars = fetch_validation_bars(symbol=symbol, days=days)
    logger.info(f"Fetched {len(val_bars)} real validation bars for {symbol} ({days}d)")

    set_ml_predictor_override(champion_model, scaler, device, input_dim)
    champ_result = await run_backtest(symbol=symbol, bars=val_bars, use_committee=True)
    reset_ml_predictor()

    set_ml_predictor_override(challenger_model, scaler, device, input_dim)
    challenger_result = await run_backtest(symbol=symbol, bars=val_bars, use_committee=True)
    reset_ml_predictor()

    return champ_result, challenger_result


def retrain_model() -> int:
    correlation_id = set_correlation_id()
    logger.info("Initializing Transformer Replay Retraining (Dual-Buffer NAS)", correlation_id=correlation_id)
    
    try:
        paths = [HISTORICAL_BUFFER_PATH, LIVE_BUFFER_PATH]
        dataset = ReplayBufferDataset(paths, max_size_per_file=100000)
        
        # Check minimum samples for meaningful training
        if len(dataset) < MIN_SAMPLES:
            logger.warning(f"Not enough data in replay buffer ({len(dataset)} samples). Minimum required: {MIN_SAMPLES}. Waiting for more trades.")
            return 0
            
        # Log label distribution
        label_dist = dataset.get_label_distribution()
        logger.info(f"Label distribution: {dict(label_dist)}")
        
        # Check for class imbalance
        total = len(dataset)
        pos_ratio = label_dist.get(1.0, 0) / total
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            logger.warning(f"Severe class imbalance detected: positive ratio = {pos_ratio:.2f}. Consider label smoothing or class weights.")
        
        # Use temporal split (last 15% for validation) instead of random split for time series
        val_size = max(int(0.15 * len(dataset)), 1)
        train_size = len(dataset) - val_size
        
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"Loaded {len(dataset)} trade experiences. Split: {train_size} train, {val_size} holdout (temporal).")
        
        # Infer input shape from dataset
        sample_x, _ = dataset[0]
        seq_len, input_dim = sample_x.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load and Evaluate Champion
        champion_loss = float("inf")
        config_path = os.path.join(os.path.dirname(MODEL_PATH), "transformer_config.json")
        
        champ_layers = 4
        champ_embed = 128
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                arch = json.load(f)
                champ_layers = arch.get("num_layers", 4)
                champ_embed = arch.get("embed_dim", 128)
                
        if os.path.exists(MODEL_PATH):
            champ_q_heads = arch.get("num_q_heads", 8) if os.path.exists(config_path) else 8
            champ_kv_heads = arch.get("num_kv_heads", 2) if os.path.exists(config_path) else 2
            champion_model = GrokGQA_Transformer(input_dim=input_dim, num_layers=champ_layers, embed_dim=champ_embed, num_q_heads=champ_q_heads, num_kv_heads=champ_kv_heads, seq_len=seq_len).to(device)
            try:
                champion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                champion_loss = evaluate_loss(champion_model, val_loader, nn.BCEWithLogitsLoss(), device)
                logger.info(f"Champion [{champ_layers}L, {champ_embed}D] Baseline Holdout Loss: {champion_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load Champion: {e}")
                
        # 2. Define NAS Candidates
        candidates = [
            {"num_layers": 2, "embed_dim": 64, "num_q_heads": 8, "num_kv_heads": 2},
            {"num_layers": 4, "embed_dim": 128, "num_q_heads": 8, "num_kv_heads": 2},
            {"num_layers": 6, "embed_dim": 256, "num_q_heads": 8, "num_kv_heads": 2}
        ]
        
        best_challenger_loss = float("inf")
        best_challenger_arch = None
        best_challenger_state = None
        
        criterion = nn.BCEWithLogitsLoss()
        
        # 3. Train all NAS candidates with early stopping
        for arch in candidates:
            L = arch["num_layers"]
            E = arch["embed_dim"]
            logger.info(f"Training NAS Candidate [{L}L, {E}D]")
            
            model = GrokGQA_Transformer(input_dim=input_dim, num_layers=L, embed_dim=E, num_q_heads=arch.get("num_q_heads", 8), num_kv_heads=arch.get("num_kv_heads", 2), seq_len=seq_len).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
            
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    
                    optimizer.step()
                    total_loss += loss.item() * len(batch_y)
                    
                train_loss = total_loss / len(train_dataset)
                val_loss = evaluate_loss(model, val_loader, criterion, device)
                scheduler.step(val_loss)
                
                logger.info(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        logger.info(f"  Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                        break
            
            # Load best state for this candidate
            if best_state is not None:
                model.load_state_dict(best_state)
                cand_loss = best_val_loss
            else:
                cand_loss = evaluate_loss(model, val_loader, criterion, device)
            
            logger.info(f"Candidate [{L}L, {E}D] Best Holdout Loss: {cand_loss:.4f}")
            
            if cand_loss < best_challenger_loss:
                best_challenger_loss = cand_loss
                best_challenger_arch = arch
                best_challenger_state = best_state
                
        # 4. Champion vs Best Challenger (Holdout Loss)
        logger.info(f"Best Challenger [{best_challenger_arch['num_layers']}L, {best_challenger_arch['embed_dim']}D] Loss: {best_challenger_loss:.4f}")
        
        if best_challenger_loss < champion_loss:
            logger.info("Challenger beat Champion on Holdout Loss. Running real walk-forward validation before promoting...")

            challenger_model = GrokGQA_Transformer(
                input_dim=input_dim,
                num_layers=best_challenger_arch["num_layers"],
                embed_dim=best_challenger_arch["embed_dim"],
                num_q_heads=best_challenger_arch.get("num_q_heads", 8),
                num_kv_heads=best_challenger_arch.get("num_kv_heads", 2),
                seq_len=seq_len,
            ).to(device)
            challenger_model.load_state_dict(best_challenger_state)
            challenger_model.eval()
            champion_model.eval()

            scaler_path = os.path.join(os.path.dirname(MODEL_PATH), "feature_scaler.pkl")
            val_scaler = joblib.load(scaler_path)

            try:
                champ_result, challenger_result = asyncio.run(
                    run_real_validation(champion_model, challenger_model, val_scaler, device, input_dim)
                )
                logger.info(f"Real Validation - Champion:   return={champ_result.total_return_pct:.2f}% "
                            f"sharpe={champ_result.sharpe:.3f} trades={champ_result.n_trades} "
                            f"win_rate={champ_result.win_rate:.1f}%")
                logger.info(f"Real Validation - Challenger: return={challenger_result.total_return_pct:.2f}% "
                            f"sharpe={challenger_result.sharpe:.3f} trades={challenger_result.n_trades} "
                            f"win_rate={challenger_result.win_rate:.1f}%")

                challenger_wins_validation = challenger_result.total_return_pct > champ_result.total_return_pct

                if challenger_wins_validation:
                    os.makedirs("models", exist_ok=True)
                    torch.save(best_challenger_state, MODEL_PATH)
                    import json
                    with open(config_path, "w") as f:
                        json.dump(best_challenger_arch, f)
                    logger.info(f"Challenger PASSED real validation (holdout loss AND simulated backtest). "
                                f"Promoted. Saved new NAS architecture to {config_path}")
                else:
                    logger.info("Challenger beat Champion on holdout loss but FAILED real validation "
                                "(worse simulated return). Vetoing promotion, keeping current Champion.")
            except Exception as e:
                logger.error("Real validation failed to run", error=str(e), traceback=traceback.format_exc())
                logger.info("Vetoing promotion as a precaution (cannot confirm the challenger is actually better).")
        else:
            logger.info("All Challengers FAILED to beat Champion on Loss. Discarding new weights.")
        return 0

    except Exception as e:
        logger.error("Transformer retraining failed", error=str(e), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(retrain_model())
