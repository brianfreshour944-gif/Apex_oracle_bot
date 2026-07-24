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
        champion_model = GrokGQA_Transformer(input_dim=input_dim, num_layers=champ_layers, embed_dim=champ_embed, seq_len=seq_len).to(device)
        try:
            champion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            champion_loss = evaluate_loss(champion_model, val_loader, nn.BCEWithLogitsLoss(), device)
            logger.info(f"🏆 Champion [{champ_layers}L, {champ_embed}D] Baseline Holdout Loss: {champion_loss:.4f}")
        except Exception as e:
            logger.warning(f"Failed to load Champion: {e}")
            
    # 2. Define NAS Candidates
    candidates = [
        {"num_layers": 2, "embed_dim": 64},
        {"num_layers": 4, "embed_dim": 128},
        {"num_layers": 6, "embed_dim": 256}
    ]
    
    best_challenger_loss = float("inf")
    best_challenger_arch = None
    best_challenger_state = None
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Train all NAS candidates
    for arch in candidates:
        L = arch["num_layers"]
        E = arch["embed_dim"]
        logger.info(f"--- Training NAS Candidate [{L}L, {E}D] ---")
        
        model = GrokGQA_Transformer(input_dim=input_dim, num_layers=L, embed_dim=E, seq_len=seq_len).to(device)
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
                
        cand_loss = evaluate_loss(model, val_loader, criterion, device)
        logger.info(f"Candidate [{L}L, {E}D] Holdout Loss: {cand_loss:.4f}")
        
        if cand_loss < best_challenger_loss:
            best_challenger_loss = cand_loss
            best_challenger_arch = arch
            best_challenger_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # 4. Champion vs Best Challenger
    logger.info(f"⚔️ Best Challenger [{best_challenger_arch['num_layers']}L, {best_challenger_arch['embed_dim']}D] Loss: {best_challenger_loss:.4f}")
    
    if best_challenger_loss < champion_loss:
        os.makedirs("models", exist_ok=True)
        torch.save(best_challenger_state, MODEL_PATH)
        import json
        with open(config_path, "w") as f:
            json.dump(best_challenger_arch, f)
        logger.info(f"✅ Challenger WINS! Saved new NAS architecture to {config_path}")
    else:
        logger.info("❌ All Challengers FAILED to beat Champion. Discarding new weights.")

if __name__ == "__main__":
    retrain_model()
