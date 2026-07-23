"""Brain 1: Transformer Brain (PyTorch Grok GQA v9 Model Connector).

Loads PyTorch trained weights from grok_gqa_v9_best.pth and feature_scaler.pkl.
Performs live model inference if PyTorch & model weights exist, with graceful fallback.
"""

import os
import numpy as np
from typing import Optional
from .models import BrainVote

# Path to PyTorch model & feature scaler
MODEL_PATH = r"C:\Users\brian\OneDrive\Documents\Grok_alpaca_Apex_v8\grok_gqa_v9_best.pth"
SCALER_PATH = r"C:\Users\brian\OneDrive\Documents\Grok_alpaca_Apex_v8\feature_scaler.pkl"

_predictor_instance = None
_predictor_initialized = False

def get_ml_predictor():
    """Lazily loads the PyTorch SafeMLPredictor model."""
    global _predictor_instance, _predictor_initialized
    if _predictor_initialized:
        return _predictor_instance

    _predictor_initialized = True
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            import joblib
            import torch
            import torch.nn as nn

            import torch.nn.functional as F

            class GQA_TransformerBlock(nn.Module):
                def __init__(self, embed_dim=128, num_q_heads=8, num_kv_heads=2, dropout=0.1):
                    super().__init__()
                    self.num_q_heads = num_q_heads
                    self.num_kv_heads = num_kv_heads
                    self.head_dim = embed_dim // num_q_heads
                    self.q_proj = nn.Linear(embed_dim, num_q_heads * self.head_dim)
                    self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
                    self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
                    self.out_proj = nn.Linear(num_q_heads * self.head_dim, embed_dim)
                    self.norm1 = nn.LayerNorm(embed_dim)
                    self.norm2 = nn.LayerNorm(embed_dim)
                    self.ffn = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim * 4, embed_dim),
                    )
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    residual = x
                    norm_x = self.norm1(x)
                    batch, seq, _ = norm_x.shape
                    q = self.q_proj(norm_x).view(batch, seq, self.num_q_heads, self.head_dim).transpose(1, 2)
                    k = self.k_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v = self.v_proj(norm_x).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
                    v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
                    attn = F.scaled_dot_product_attention(q, k, v)
                    attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.num_q_heads * self.head_dim)
                    x = residual + self.dropout(self.out_proj(attn))
                    residual = x
                    norm_x = self.norm2(x)
                    x = residual + self.ffn(norm_x)
                    return x

            class GrokGQA_Transformer(nn.Module):
                def __init__(self, input_dim=11, seq_len=32, embed_dim=128, num_layers=4, num_q_heads=8, num_kv_heads=2, dropout=0.1):
                    super().__init__()
                    self.input_projection = nn.Linear(input_dim, embed_dim)
                    self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
                    self.dropout = nn.Dropout(dropout)
                    self.layers = nn.ModuleList([
                        GQA_TransformerBlock(embed_dim, num_q_heads, num_kv_heads, dropout)
                        for _ in range(num_layers)
                    ])
                    self.norm = nn.LayerNorm(embed_dim)
                    self.output_head = nn.Linear(embed_dim, 1)

                def forward(self, x):
                    x = self.input_projection(x)
                    x = x + self.pos_encoder 
                    x = self.dropout(x)
                    for layer in self.layers:
                        x = layer(x)
                    x = self.norm(x)
                    x = self.output_head(x[:, -1, :])
                    return x

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scaler = joblib.load(SCALER_PATH)

            # Determine feature dimension from scaler
            input_dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 11
            model = GrokGQA_Transformer(input_dim=input_dim).to(device)
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            _predictor_instance = {
                "model": model,
                "scaler": scaler,
                "device": device,
                "torch": torch,
                "input_dim": input_dim
            }
        except Exception as e:
            _predictor_instance = None
    return _predictor_instance

async def transformer_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates Grok GQA PyTorch Transformer model prediction with graceful fallback."""
    predictor = get_ml_predictor()
    raw_action = signal.get("action", "hold")
    regime = signal.get("regime", "unknown")
    reason = "Signal fallback logic (ML unavailable)"

    # Fallback prob derived from strategy action to eliminate silent sell bias
    if raw_action == "buy":
        prob = 0.65
    elif raw_action == "sell":
        prob = 0.35
    else:
        prob = 0.50

    if predictor is not None:
        try:
            torch = predictor["torch"]
            model = predictor["model"]
            scaler = predictor["scaler"]
            device = predictor["device"]

            # Construct input features if provided in signal data
            features = signal.get("features")
            if features is not None and len(features) > 0:
                feat_scaled = scaler.transform([features])
                # Repeat over 32 sequence length if 2D
                if len(feat_scaled.shape) == 2:
                    feat_seq = np.tile(feat_scaled, (1, 32, 1))
                else:
                    feat_seq = feat_scaled

                tensor_in = torch.tensor(feat_seq, dtype=torch.float32).to(device)
                with torch.no_grad():
                    logit = model(tensor_in).item()
                    prob = 1.0 / (1.0 + np.exp(-logit))  # Sigmoid output
                    reason = f"Grok PyTorch Inference prob={prob:.3f} (logit={logit:.2f})"
        except Exception as e:
            import logging
            logging.getLogger("bot").error(f"Transformer inference error: {e}")

    # Threshold probability decision logic
    if prob > 0.58:
        action = "buy"
    elif prob < 0.42:
        action = "sell"
    else:
        action = "hold"

    return BrainVote(
        name="transformer",
        action=action,
        confidence=prob,
        weight=0.35,
        regime=regime,
        reason=reason
    )
