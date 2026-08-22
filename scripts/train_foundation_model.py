#!/usr/bin/env python3
"""
Stage 1: Foundation Transformer Pre-Training

This script downloads 730 days of 1-hour historical data across the 10-asset basket,
computes features, scales them globally, and trains the Grok GQA Transformer from scratch.
It saves the foundational weights and the feature scaler for live use.
"""

import os
import sys
import numpy as np
import polars as pl
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.committee.transformer_brain import GrokGQA_Transformer
from src.feature_engineering import add_features, get_active_features
from src.logging_config import get_logger

logger = get_logger("foundation_trainer")

# Config
SYMBOLS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "LINK-USD",
    "LTC-USD",
    "AVAX-USD",
    "BCH-USD"
]
DAYS = 730  # Bootstrap dataset length
SEQ_LEN = 32
HORIZON = 6
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 10
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "grok_gqa_v9_best.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")

# Basic Technical Feature Engineering
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def extract_sequences(df, features_cols):
    X, y = [], []
    feats = df[features_cols].values
    close = df["close"].values
    
    for i in range(len(df) - SEQ_LEN - HORIZON + 1):
        x_window = feats[i : i + SEQ_LEN]
        close_now = close[i + SEQ_LEN - 1]
        close_future = close[i + SEQ_LEN - 1 + HORIZON]
        
        label = 1.0 if close_future > close_now else 0.0
        X.append(x_window)
        y.append(label)
        
    return np.array(X), np.array(y)

def main() -> int:
    logger.info("Initializing Stage 1 Foundation Training...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Download and build dataset
    all_X = []
    all_y = []
    
    features_cols = get_active_features()
    
    for sym in SYMBOLS:
        logger.info(f"Downloading {sym}...")
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=f"{DAYS}d", interval="1h")
            if df.empty: continue
            
            df = df.reset_index()
            df = df.rename(columns={"Datetime": "t", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            raw_close = df["close"].copy()
            df = add_features(df)
            df["close"] = raw_close.values
            
            X, y = extract_sequences(df, features_cols)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            logger.error(f"Error processing {sym}: {e}")
            
    if not all_X:
        logger.error("No data extracted. Aborting.")
        return
        
    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    logger.info(f"Extracted {len(X_full)} total sequences.")
    
    # 2. Fit Global Scaler
    from sklearn.preprocessing import StandardScaler
    N, S, F = X_full.shape
    X_flat = X_full.reshape(-1, F)
    
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(N, S, F)
    
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved global feature scaler to {SCALER_PATH}")
    
    # 3. Chronological Split (80/20) & Class Balancing on Train
    split_idx = int(0.8 * len(X_scaled))
    X_train, y_train = X_scaled[:split_idx], y_full[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_full[split_idx:]
    
    idx_win = np.where(y_train == 1.0)[0]
    idx_loss = np.where(y_train == 0.0)[0]
    minority_count = min(len(idx_win), len(idx_loss))
    
    np.random.seed(42)
    idx_win_bal = np.random.choice(idx_win, minority_count, replace=False)
    idx_loss_bal = np.random.choice(idx_loss, minority_count, replace=False)
    bal_idx = np.concatenate([idx_win_bal, idx_loss_bal])
    np.random.shuffle(bal_idx)
    
    X_train_bal = X_train[bal_idx]
    y_train_bal = y_train[bal_idx]
    logger.info(f"Balanced training set: {len(X_train_bal)} samples. Val set: {len(X_val)} samples.")
    
    train_loader = DataLoader(SequenceDataset(X_train_bal, y_train_bal), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Initialize Grok GQA Transformer
    model = GrokGQA_Transformer(embed_dim=128, num_q_heads=4, num_layers=4, seq_len=SEQ_LEN, input_dim=F, dropout=0.2).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. Training Loop with Early Stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    logger.info("Starting Supervised Pre-Training...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(by)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                preds = (logits > 0.0).float()
                correct += (preds == by).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset) * 100.0
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            
            import json
            config_path = os.path.join(MODELS_DIR, "transformer_config.json")
            with open(config_path, "w") as f:
                json.dump({"num_layers": 4, "embed_dim": 128, "num_q_heads": 4, "num_kv_heads": 2}, f)
                
            logger.info(f"⭐ New Best Model! Saved to {MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
    logger.info(f"✅ Foundation Pre-Training Complete. Best Val Acc: {best_val_acc:.2f}%.")
    logger.info("The live bot and retraining pipeline are now ready to operate.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
