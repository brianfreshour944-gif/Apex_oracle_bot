#!/usr/bin/env python3
"""
Continuous Transformer Replay Training

Loads the JSONL dataset containing 32-step tensor inputs and true trade outcomes
recorded during actual backtesting/live trading. Continuously fine-tunes the 
Grok GQA model using experience replay.
"""

import asyncio
import datetime
import json
import os
import sys
import traceback
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import model_store
from src.backtest import run_backtest
from src.committee.transformer_brain import GrokGQA_Transformer, reset_ml_predictor, set_ml_predictor_override
from src.live_buffer import read_jsonl_tolerant
from src.logging_config import get_logger, set_correlation_id
from src.walkforward import run_walkforward_validation

logger = get_logger("transformer_replay")

HISTORICAL_BUFFER_PATH = "data/historical_experiences.jsonl"
LIVE_BUFFER_PATH = "data/live_experiences.jsonl"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
MIN_SAMPLES = 50  # Minimum samples required for meaningful training
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
PATIENCE = 5  # Early stopping patience

def _parse_entry_time(value):
    if not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=datetime.UTC)


class ReplayBufferDataset(Dataset):
    def __init__(self, data_paths, max_size_per_file=100000):
        self.samples = []
        self.labels = []
        # (sort_key, tensor, label). Records are ordered chronologically before
        # the train/holdout split below: the historical buffer is written one
        # symbol at a time (all BTC, then ETH, then SOL), so taking its tail
        # as the "temporal" holdout actually held out one whole coin -- train
        # on BTC+ETH, test on SOL -- rather than the most recent trades.
        # Live-buffer records carry no timestamp; they are the newest trades
        # and keep their file order after every timestamped record.
        staged = []
        file_index = 0
        for data_path in data_paths:
            if not os.path.exists(data_path):
                continue
            # Tolerant decoding lives in src.live_buffer (BOM/NUL stripping,
            # per-line counting, [LIVE_BUFFER] warnings). It was extracted so
            # the same reader backs this trainer and scripts/repair_live_buffer.py
            # -- previously the tolerant logic was inline here only, so nothing
            # else could see or fix a bad buffer. A per-file failure still keeps
            # one bad file from losing the others.
            records, _stats = read_jsonl_tolerant(
                data_path, max_lines=max_size_per_file, buffer_name=os.path.basename(data_path)
            )
            for line_index, record in enumerate(records):
                try:
                    tensor_state = np.array(record["tensor"], dtype=np.float32)
                    label = float(record["label"])
                    ts = _parse_entry_time(record.get("entry_time"))
                    key = (0, ts.timestamp(), 0, 0) if ts else (1, 0.0, file_index, line_index)
                    staged.append((key, tensor_state, label))
                except Exception as e:
                    # A structurally valid JSON object that still can't be
                    # turned into a (tensor, label) pair -- surface it too.
                    logger.warning(
                        f"[LIVE_BUFFER] {os.path.basename(data_path)}: line {line_index} "
                        f"is missing a usable tensor/label ({e})"
                    )
            file_index += 1

        # Keep only records matching the model's input shape (sequence x
        # features, e.g. 32x11). The tracked historical buffer once held
        # 6,312 flat 128-value random-noise vectors from the generator's old
        # fast mode, which crashed training at `seq_len, input_dim =
        # sample_x.shape`; mixed shapes would also break DataLoader batching.
        # The expected shape is the most common 2-D shape present.
        two_d = Counter(t.shape for _, t, _ in staged if t.ndim == 2)
        expected = two_d.most_common(1)[0][0] if two_d else None
        kept = [(k, t, y) for k, t, y in staged if t.shape == expected]
        wrong = len(staged) - len(kept)
        if wrong:
            other = Counter(t.shape for _, t, _ in staged if t.shape != expected)
            logger.warning(
                f"Replay buffers: skipped {wrong} record(s) not shaped {expected} "
                f"(found {dict(other)}) -- likely stale fast-mode noise or a model-shape change"
            )
        kept.sort(key=lambda item: item[0])
        for _, tensor_state, label in kept:
            self.samples.append((tensor_state, label))
            self.labels.append(label)

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
    # Keep `t` as native datetimes: WalkForwardValidator.create_windows()
    # filters the polars column against datetime window bounds. String
    # timestamps made that comparison raise "cannot compare date/datetime/time
    # to a string value", zeroing every validation window and vetoing all
    # promotions (observed 2026-09-27 in retrain_log3.txt).
    df["t"] = pd.to_datetime(df["t"])
    return pl.from_pandas(df)


async def run_real_validation(champion_model, challenger_model, scaler, device, input_dim,
                               symbol: str = "BTC-USD", days: int = 90):
    """Run both models through a real simulated backtest (via the full 5-brain
    committee) over the same real historical window, and return their results
    for direct comparison. This replaces fake/random validation with an actual
    measured outcome.

    NOTE: single-symbol, single-window -- see run_multi_symbol_walkforward_validation
    below for the broader check actually used to gate promotion as of 2026-09-20.
    Kept here as a cheap pre-check / for callers that want the old narrow behavior.
    """
    val_bars = fetch_validation_bars(symbol=symbol, days=days)
    logger.info(f"Fetched {len(val_bars)} real validation bars for {symbol} ({days}d)")

    set_ml_predictor_override(champion_model, scaler, device, input_dim)
    champ_result = await run_backtest(symbol=symbol, bars=val_bars, use_committee=True)
    reset_ml_predictor()

    set_ml_predictor_override(challenger_model, scaler, device, input_dim)
    challenger_result = await run_backtest(symbol=symbol, bars=val_bars, use_committee=True)
    reset_ml_predictor()

    return champ_result, challenger_result


# Symbols for the broader promotion-gate validation below. yfinance tickers
# (hyphenated), matching the convention already used by
# scripts/generate_replay_dataset.py's SYMBOLS list.
WALKFORWARD_VALIDATION_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]


async def run_multi_symbol_walkforward_validation(
    champion_model, challenger_model, scaler, device, input_dim,
    symbols: list[str] = WALKFORWARD_VALIDATION_SYMBOLS,
    days: int = 180,
    n_splits: int = 3,
) -> tuple[dict, dict]:
    """Broader model-promotion validation: real purged walk-forward
    (src/walkforward.py's WalkForwardValidator, not a single train/test split)
    across MULTIPLE symbols and MULTIPLE non-overlapping time windows per
    symbol, for both champion and challenger.

    Replaces the single-symbol/single-90-day-window check that
    run_real_validation() above did alone -- KNOWN_ISSUES.md flagged this as
    too narrow to fully trust ("Consider testing across multiple symbols and
    time windows before fully trusting promotions"). src/walkforward.py's
    WalkForwardValidator already existed and was well-built (purge/embargo,
    anchored windows) but had ZERO callers anywhere in the promotion path
    before this (ADVERSARIAL_AUDIT_2026-09-20.md §8/§9) -- it's the walk-forward
    engine actually used here as of 2026-09-20.

    Returns (champion_summary, challenger_summary) dicts with aggregated
    metrics across all symbol x window combinations, safe even for a single
    symbol/window pair that fails to produce enough windows (skipped, logged).
    """
    def _summarize(all_results: list) -> dict:
        if not all_results:
            return {"mean_return_pct": 0.0, "mean_sharpe": 0.0, "total_trades": 0,
                     "pct_positive_windows": 0.0, "n_windows": 0}
        returns = [r.total_return_pct for r in all_results]
        sharpes = [r.sharpe for r in all_results]
        return {
            "mean_return_pct": float(np.mean(returns)),
            "mean_sharpe": float(np.mean(sharpes)),
            "total_trades": int(sum(r.n_trades for r in all_results)),
            "pct_positive_windows": float(np.mean([r > 0 for r in returns]) * 100) if returns else 0.0,
            "n_windows": len(all_results),
        }

    champ_windows: list = []
    challenger_windows: list = []

    for symbol in symbols:
        try:
            val_bars = fetch_validation_bars(symbol=symbol, days=days)
        except Exception as e:
            logger.warning(f"Skipping {symbol} in walk-forward validation: could not fetch data ({e})")
            continue
        if len(val_bars) < 100:
            logger.warning(f"Skipping {symbol} in walk-forward validation: only {len(val_bars)} bars fetched")
            continue

        try:
            set_ml_predictor_override(champion_model, scaler, device, input_dim)
            champ_wf = await run_walkforward_validation(
                symbol=symbol, bars=val_bars, n_splits=n_splits, backtest_params={"use_committee": True},
            )
            champ_windows.extend(champ_wf.windows)
        except Exception as e:
            logger.warning(f"Champion walk-forward validation failed for {symbol} (non-fatal, skipping symbol): {e}")
        finally:
            reset_ml_predictor()

        try:
            set_ml_predictor_override(challenger_model, scaler, device, input_dim)
            challenger_wf = await run_walkforward_validation(
                symbol=symbol, bars=val_bars, n_splits=n_splits, backtest_params={"use_committee": True},
            )
            challenger_windows.extend(challenger_wf.windows)
        except Exception as e:
            logger.warning(f"Challenger walk-forward validation failed for {symbol} (non-fatal, skipping symbol): {e}")
        finally:
            reset_ml_predictor()

    return _summarize(champ_windows), _summarize(challenger_windows)


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
        
        # Loss of always predicting the training set's win rate. A model that
        # can't beat this on the holdout has learned nothing usable.
        train_labels = np.array([float(dataset[i][1]) for i in range(train_size)])
        val_labels = np.array([float(dataset[i][1]) for i in range(train_size, len(dataset))])
        base_rate = float(np.clip(train_labels.mean(), 1e-6, 1 - 1e-6))
        no_skill_loss = float(-np.mean(val_labels * np.log(base_rate) + (1 - val_labels) * np.log(1 - base_rate)))
        logger.info(f"No-skill baseline holdout loss (constant {base_rate:.3f}): {no_skill_loss:.4f}")

        # 1. Load and Evaluate Champion (the active model: persistent store
        # first, baked models/ otherwise -- see src/model_store.py)
        champion_loss = float("inf")
        champion_model = None
        model_path, scaler_path, config_path = model_store.transformer_paths()
        
        champ_layers = 4
        champ_embed = 128
        if os.path.exists(config_path):
            with open(config_path) as f:
                arch = json.load(f)
                champ_layers = arch.get("num_layers", 4)
                champ_embed = arch.get("embed_dim", 128)
                
        if os.path.exists(model_path):
            champ_q_heads = arch.get("num_q_heads", 8) if os.path.exists(config_path) else 8
            champ_kv_heads = arch.get("num_kv_heads", 2) if os.path.exists(config_path) else 2
            champion_model = GrokGQA_Transformer(input_dim=input_dim, num_layers=champ_layers, embed_dim=champ_embed, num_q_heads=champ_q_heads, num_kv_heads=champ_kv_heads, seq_len=seq_len).to(device)
            try:
                champion_model.load_state_dict(torch.load(model_path, map_location=device))
                champion_loss = evaluate_loss(champion_model, val_loader, nn.BCEWithLogitsLoss(), device)
                logger.info(f"Champion [{champ_layers}L, {champ_embed}D] Baseline Holdout Loss: {champion_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load Champion: {e}")
                champion_model = None
                
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
            # `verbose` was removed in PyTorch 2.x (deprecated in 1.13);
            # passing it crashes training instantly on the installed torch.
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            
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
        
        if champion_model is None:
            logger.warning("No loadable champion to validate against -- vetoing promotion as a precaution.")
        elif best_challenger_loss >= no_skill_loss:
            logger.info(f"Best challenger holdout loss {best_challenger_loss:.4f} does not beat the no-skill "
                        f"baseline {no_skill_loss:.4f} -- no learnable signal, keeping current Champion.")
        elif best_challenger_loss < champion_loss:
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

            val_scaler = joblib.load(scaler_path)

            try:
                # Broad gate: real purged walk-forward across multiple symbols
                # and multiple time windows (see run_multi_symbol_walkforward_validation
                # docstring -- this replaces the single-symbol/90-day-window
                # check as the actual promotion gate per KNOWN_ISSUES.md).
                champ_summary, challenger_summary = asyncio.run(
                    run_multi_symbol_walkforward_validation(champion_model, challenger_model, val_scaler, device, input_dim)
                )
                logger.info(f"Walk-Forward Validation - Champion:   mean_return={champ_summary['mean_return_pct']:.2f}% "
                            f"mean_sharpe={champ_summary['mean_sharpe']:.3f} windows={champ_summary['n_windows']} "
                            f"trades={champ_summary['total_trades']} pct_positive={champ_summary['pct_positive_windows']:.0f}%")
                logger.info(f"Walk-Forward Validation - Challenger: mean_return={challenger_summary['mean_return_pct']:.2f}% "
                            f"mean_sharpe={challenger_summary['mean_sharpe']:.3f} windows={challenger_summary['n_windows']} "
                            f"trades={challenger_summary['total_trades']} pct_positive={challenger_summary['pct_positive_windows']:.0f}%")

                if champ_summary["n_windows"] == 0 or challenger_summary["n_windows"] == 0:
                    logger.warning("Walk-forward validation produced zero usable windows for champion or "
                                    "challenger (data fetch issues?) -- vetoing promotion as a precaution.")
                    challenger_wins_validation = False
                else:
                    # Require the challenger to win on BOTH mean return AND mean
                    # Sharpe across all windows/symbols, not just one metric --
                    # a single-metric win is exactly the kind of thin margin
                    # that curve-fits to one lucky window (see
                    # ADVERSARIAL_AUDIT_2026-09-20.md §1 on edge verification).
                    # Beating the champion is not enough on its own: on
                    # 2026-09-29 a challenger that lost -1.49% per window was
                    # promoted because the champion lost -2.84%. It must also
                    # make money on its own.
                    challenger_wins_validation = (
                        challenger_summary["mean_return_pct"] > champ_summary["mean_return_pct"]
                        and challenger_summary["mean_sharpe"] > champ_summary["mean_sharpe"]
                        and challenger_summary["mean_return_pct"] > 0
                    )

                if challenger_wins_validation:
                    # Written to the persistent store (data volume), not the
                    # image's models/ dir, so the promotion survives a redeploy.
                    with model_store.staging("transformer") as staged:
                        staged_model = os.path.join(staged, "grok_gqa_v9_best.pth")
                        staged_config = os.path.join(staged, "transformer_config.json")
                        torch.save(best_challenger_state, staged_model)
                        with open(staged_config, "w") as f:
                            json.dump(best_challenger_arch, f)
                        store = model_store.promote("transformer", {
                            "grok_gqa_v9_best.pth": staged_model,
                            "transformer_config.json": staged_config,
                            "feature_scaler.pkl": scaler_path,
                        })
                    logger.info(f"Challenger PASSED validation (holdout loss AND multi-symbol walk-forward, "
                                f"beating champion on both mean return and mean Sharpe, with positive mean return). "
                                f"Promoted to {store}")
                else:
                    logger.info("Challenger beat Champion on holdout loss but FAILED walk-forward validation "
                                "(did not beat champion on both mean return and mean Sharpe across all "
                                "symbols/windows, or its mean return was not positive). Vetoing promotion, "
                                "keeping current Champion.")
            except Exception as e:
                logger.error("Walk-forward validation failed to run", error=str(e), traceback=traceback.format_exc())
                logger.info("Vetoing promotion as a precaution (cannot confirm the challenger is actually better).")
        else:
            logger.info("All Challengers FAILED to beat Champion on Loss. Discarding new weights.")
        return 0

    except Exception as e:
        logger.error("Transformer retraining failed", error=str(e), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(retrain_model())
