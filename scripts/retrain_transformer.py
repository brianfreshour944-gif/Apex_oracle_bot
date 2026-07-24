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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.committee.transformer_brain import GrokGQA_Transformer
from src.logging_config import get_logger

logger = get_logger("transformer_replay")

BUFFER_PATH = "data/transformer_replay_buffer.jsonl"
MODEL_PATH = "models/grok_gqa_v9_best.pth"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

class ReplayBufferDataset(Dataset):
    def __init__(self, data_path, max_size=10000):
        self.samples = []
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                lines = f.readlines()
                # Keep only the last `max_size` trades (Rolling buffer)
                for line in lines[-max_size:]:
                    try:
                        record = json.loads(line)
                        tensor_state = np.array(record["tensor"], dtype=np.float32)
                        label = float(record["label"])
                        self.samples.append((tensor_state, label))
                    except Exception as e:
                        pass
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)

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

def retrain_model():
    logger.info("Initializing Transformer Replay Retraining...")
    
    if not os.path.exists(BUFFER_PATH):
        logger.error(f"Replay buffer not found at {BUFFER_PATH}")
        return
        
    dataset = ReplayBufferDataset(BUFFER_PATH, max_size=10000)
    if len(dataset) < BATCH_SIZE:
        logger.warning(f"Not enough data in replay buffer ({len(dataset)} samples). Waiting for more trades.")
        return
        
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Loaded {len(dataset)} trade experiences. Split: {train_size} train, {val_size} holdout.")
    
    # Infer input shape from dataset
    sample_x, _ = dataset[0]
    seq_len, embed_dim = sample_x.shape
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrokGQA_Transformer(embed_dim=embed_dim, num_heads=4, num_layers=4, seq_len=seq_len).to(device)
    
    # Load existing weights if they exist
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            logger.info("Loaded existing weights for fine-tuning.")
        except Exception as e:
            logger.warning(f"Could not load existing weights, starting fresh: {e}")
            
    criterion = nn.BCEWithLogitsLoss()
    
    # 1. Evaluate Champion
    champion_loss = float("inf")
    if os.path.exists(MODEL_PATH):
        champion_loss = evaluate_loss(model, val_loader, criterion, device)
        logger.info(f"🏆 Champion Baseline Holdout Loss: {champion_loss:.4f}")
        
    # 2. Train Challenger
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
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
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_loss:.4f}")
        
    # 3. Evaluate Challenger
    challenger_loss = evaluate_loss(model, val_loader, criterion, device)
    logger.info(f"⚔️ Challenger Holdout Loss: {challenger_loss:.4f}")
    
    # 4. Champion vs Challenger
    if challenger_loss < champion_loss:
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"✅ Challenger WINS! Saved updated model to {MODEL_PATH}")
    else:
        logger.info("❌ Challenger FAILED to beat Champion. Discarding new weights.")

if __name__ == "__main__":
    retrain_model()
