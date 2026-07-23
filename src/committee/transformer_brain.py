"""Brain 1: Transformer Brain (PyTorch Grok GQA v9 Model Connector).

Loads PyTorch trained weights from grok_gqa_v9_best.pth and feature_scaler.pkl.
Performs live model inference if PyTorch & model weights exist, with graceful fallback.
"""

import os
import numpy as np
from typing import Optional
from .models import BrainVote
from src.config import settings

# Path to PyTorch model & feature scaler
MODEL_PATH = settings.TRANSFORMER_MODEL_PATH
SCALER_PATH = settings.TRANSFORMER_SCALER_PATH

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
    reason = "Signal fallback confidence"
    
    # Fallback prob derived from strategy action to eliminate silent sell bias
    if raw_action == "buy":
        prob = 0.65
    elif raw_action == "sell":
        prob = 0.35
    else:
        prob = 0.50

    if predictor is not None:
        try:
            import asyncio
            import os
                from src.feature_engineering import add_features
                # fetch_bars is actually located in data_fetcher or similar, wait I will just copy the import logic
                try:
                    from src.data_fetcher import fetch_bars
                except ImportError:
                    # fallback if fetch_bars was moved
                    from src.feature_engineering import fetch_bars
                    
                from alpaca.data.historical import CryptoHistoricalDataClient

                torch = predictor["torch"]
                model = predictor["model"]
                scaler = predictor["scaler"]
                device = predictor["device"]
                
                def _do_inference():
                    key = os.getenv("APCA_API_KEY_ID")
                    secret = os.getenv("APCA_API_SECRET_KEY")
                    if not key or not secret:
                        return None
                    
                    client = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
                    df_raw = fetch_bars(client, symbol.replace("/", ""), days=2)
                    if df_raw is None or len(df_raw) < 32:
                        return None
                        
                    # Inject On-Chain / Derivatives Data
                    try:
                        from src.onchain_data import fetch_derivatives_data
                        import asyncio
                        # We are inside to_thread but we can run an async function synchronously here
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        deriv_data = loop.run_until_complete(fetch_derivatives_data(symbol))
                        loop.close()
                        
                        df_raw["funding_rate"] = deriv_data.get("funding_rate", 0.0)
                        df_raw["open_interest"] = deriv_data.get("open_interest", 0.0)
                        df_raw["long_short_ratio"] = deriv_data.get("long_short_ratio", 1.0)
                    except Exception as e:
                        import logging
                        logging.getLogger("onchain").warning(f"Failed to append onchain data: {e}")
                        df_raw["funding_rate"] = 0.0
                        df_raw["open_interest"] = 0.0
                        df_raw["long_short_ratio"] = 1.0

                    df_feat = add_features(df_raw.copy())
                    
                    if hasattr(scaler, "feature_names_in_"):
                        cols = list(scaler.feature_names_in_)
                    else:
                        from src.feature_engineering import get_active_features
                        cols = get_active_features()
                    
                data = df_feat[cols].tail(32).values.astype(np.float32)
                if len(data) < 32:
                    return None
                    
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data = np.clip(data, -1e6, 1e6)
                data_scaled = scaler.transform(data).astype(np.float32)
                data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Let the shadow arena candidates process the raw dataframe using their own scaler
                try:
                    from src.shadow_arena import evaluate_candidates
                    evaluate_candidates(symbol, price, df_feat)
                except Exception as shadow_e:
                    import logging
                    logging.getLogger("shadow_arena").error(f"Shadow evaluation failed: {shadow_e}")
                
                x = torch.tensor(data_scaled).unsqueeze(0).to(device)
                
                # Causal Reasoning: Approximate SHAP via Gradients * Input
                x.requires_grad = True
                
                raw_logit = model(x).squeeze(1)
                raw_logit.backward()
                
                # Gradients w.r.t the last timestep's features
                grads = x.grad[0, -1, :].cpu().numpy()
                feats = data_scaled[-1, :]
                
                # Simple Feature Importance (Gradient * Input)
                contributions = grads * feats
                
                # Map contributions to feature names
                causal_reasoning = {}
                for i, col in enumerate(cols):
                    causal_reasoning[col] = float(contributions[i])
                
                with torch.no_grad():
                    out_prob = float(torch.sigmoid(torch.tensor(raw_logit.item())).item())
                    
                return out_prob, raw_logit.item(), causal_reasoning
                    
            res = await asyncio.to_thread(_do_inference)
            causal_reasoning_dict = None
            if res is not None:
                prob, logit, causal_reasoning_dict = res
                reason = f"Grok PyTorch Inference prob={prob:.3f} (logit={logit:.2f})"
        except Exception as e:
            import logging
            logging.getLogger("committee").error(f"Transformer inference error: {e}")

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
        reason=reason,
        causal_reasoning=causal_reasoning_dict
    )
