"""Brain 1: Transformer Brain (PyTorch Grok GQA v9 Model Connector).

Loads PyTorch trained weights from grok_gqa_v9_best.pth and feature_scaler.pkl.
Performs live model inference if PyTorch & model weights exist, with graceful fallback.

Supports three modes:
- Standard: Single model with MC-dropout uncertainty (default)
- Bayesian: Deep Ensemble + MC-dropout + Temperature Calibration (if USE_BAYESIAN_TRANSFORMER=True)
- Fast BatchEnsemble: Single forward pass with rank-1 perturbations (if USE_FAST_ENSEMBLE=True)
"""

import os
import numpy as np
import asyncio
import threading
from typing import Optional, List
from .models import BrainVote
from src.config import settings

# Path to PyTorch model & feature scaler
MODEL_PATH = settings.TRANSFORMER_MODEL_PATH
SCALER_PATH = settings.TRANSFORMER_SCALER_PATH

# Ensemble settings
USE_BAYESIAN_TRANSFORMER = getattr(settings, 'USE_BAYESIAN_TRANSFORMER', False)
USE_FAST_ENSEMBLE = getattr(settings, 'USE_FAST_ENSEMBLE', False)
ENSEMBLE_SIZE = getattr(settings, 'TRANSFORMER_ENSEMBLE_SIZE', 5)
MC_DROPOUT_PASSES = getattr(settings, 'TRANSFORMER_MC_DROPOUT_PASSES', 20)

# Thread lock to protect shared model state (train/eval mode) from concurrent access
# across symbols evaluated concurrently via asyncio.create_task()
_model_inference_lock = threading.Lock()

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

_predictor_instance = None
_predictor_initialized = False
_data_client_cache = None
_data_client_cache_key = None


def _get_data_client():
    """Return a cached CryptoHistoricalDataClient, creating one only
    when the API credentials change. Avoids creating a new client
    (and its underlying HTTP session) on every inference call."""
    global _data_client_cache, _data_client_cache_key
    key = (os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    if _data_client_cache is not None and _data_client_cache_key == key:
        return _data_client_cache
    from alpaca.data.historical import CryptoHistoricalDataClient
    _data_client_cache = CryptoHistoricalDataClient(
        api_key=key[0], secret_key=key[1]
    )
    _data_client_cache_key = key
    return _data_client_cache

def get_ml_predictor():
    """Lazily loads the PyTorch SafeMLPredictor model."""
    global _predictor_instance, _predictor_initialized
    if _predictor_initialized:
        return _predictor_instance

    _predictor_initialized = True
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            import joblib
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scaler = joblib.load(SCALER_PATH)

            # Determine feature dimension from scaler
            input_dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 11
            
            # Load dynamic architecture config if it exists
            config_path = os.path.join(os.path.dirname(MODEL_PATH), "transformer_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    arch = json.load(f)
                model = GrokGQA_Transformer(
                    input_dim=input_dim, 
                    num_layers=arch.get("num_layers", 4),
                    embed_dim=arch.get("embed_dim", 128),
                    num_q_heads=arch.get("num_q_heads", 8),
                    num_kv_heads=arch.get("num_kv_heads", 2)
                ).to(device)
            else:
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

def set_ml_predictor_override(model, scaler, device, input_dim):
    """Directly inject a specific model instance, bypassing file-based loading.
    Used by validation/comparison code (e.g. retrain_transformer.py) to run
    the champion and each challenger through real backtests without needing
    separate processes or touching disk.
    """
    global _predictor_instance, _predictor_initialized
    _predictor_instance = {
        "model": model,
        "scaler": scaler,
        "device": device,
        "torch": torch,
        "input_dim": input_dim,
    }
    _predictor_initialized = True


def reset_ml_predictor():
    """Clear the cached predictor override so the next call to
    get_ml_predictor() reloads normally from MODEL_PATH/SCALER_PATH."""
    global _predictor_instance, _predictor_initialized
    _predictor_instance = None
    _predictor_initialized = False


_fast_ensemble_instance = None
_fast_ensemble_initialized = False


def get_fast_ensemble_predictor():
    """Lazily loads the BatchEnsemble wrapped transformer model."""
    global _fast_ensemble_instance, _fast_ensemble_initialized
    if _fast_ensemble_initialized:
        return _fast_ensemble_instance

    _fast_ensemble_initialized = True
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            import joblib
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scaler = joblib.load(SCALER_PATH)

            # Determine feature dimension from scaler
            input_dim = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else 11
            
            # Load dynamic architecture config if it exists
            config_path = os.path.join(os.path.dirname(MODEL_PATH), "transformer_config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    arch = json.load(f)
                base_model = GrokGQA_Transformer(
                    input_dim=input_dim, 
                    num_layers=arch.get("num_layers", 4),
                    embed_dim=arch.get("embed_dim", 128),
                    num_q_heads=arch.get("num_q_heads", 8),
                    num_kv_heads=arch.get("num_kv_heads", 2)
                ).to(device)
            else:
                base_model = GrokGQA_Transformer(input_dim=input_dim).to(device)
            
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            base_model.load_state_dict(state_dict)
            base_model.eval()

            # Wrap with BatchEnsemble for fast ensemble inference
            from .batch_ensemble import convert_model_to_batchensemble
            fast_ensemble = convert_model_to_batchensemble(
                base_model, 
                ensemble_size=ENSEMBLE_SIZE,
                skip_output=True  # Keep output_head as single head for now
            )
            fast_ensemble.eval()

            _fast_ensemble_instance = {
                "model": fast_ensemble,
                "scaler": scaler,
                "device": device,
                "torch": torch,
                "input_dim": input_dim
            }
        except Exception as e:
            import logging
            logging.getLogger("committee").error(f"Fast ensemble loading failed: {e}")
            _fast_ensemble_instance = None
    return _fast_ensemble_instance


def reset_fast_ensemble():
    """Clear the cached fast ensemble predictor."""
    global _fast_ensemble_instance, _fast_ensemble_initialized
    _fast_ensemble_instance = None
    _fast_ensemble_initialized = False


async def _run_fast_ensemble_inference(
    symbol: str, 
    price: float, 
    signal: dict,
    predictor: dict
) -> tuple:
    """Run fast BatchEnsemble inference with uncertainty quantification."""
    import asyncio
    import os
    from src.feature_engineering import add_features
    from src.data_fetcher import fetch_bars
    from alpaca.data.historical import CryptoHistoricalDataClient

    torch = predictor["torch"]
    model = predictor["model"]
    scaler = predictor["scaler"]
    device = predictor["device"]
    
    def _do_inference():
        backtest_df = signal.get("backtest_df")
        if backtest_df is not None and len(backtest_df) >= 32:
            import pandas as pd
            import polars as pl
            if isinstance(backtest_df, pl.DataFrame):
                df_raw = backtest_df.to_pandas()
            else:
                df_raw = backtest_df.copy()
            if "t" in df_raw.columns:
                df_raw = df_raw.rename(columns={"t": "timestamp"})
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                df_raw.set_index("timestamp", inplace=True)
            if "vwap" not in df_raw.columns:
                df_raw["vwap"] = df_raw["close"]
            if "trade_count" not in df_raw.columns:
                df_raw["trade_count"] = 0.0
            is_live = False
        else:
            key = os.getenv("APCA_API_KEY_ID")
            secret = os.getenv("APCA_API_SECRET_KEY")
            if not key or not secret:
                return None

            client = _get_data_client()
            alpaca_symbol = symbol.replace("-", "/") if "-" in symbol else symbol
            df_raw = fetch_bars(client, alpaca_symbol, days=2)
            if df_raw is None or len(df_raw) < 32:
                return None
            is_live = True

        # Inject On-Chain / Derivatives Data
        if not is_live:
            df_raw["funding_rate"] = 0.0
            df_raw["open_interest"] = 0.0
            df_raw["long_short_ratio"] = 1.0
        try:
            if is_live:
                from src.onchain_data import fetch_derivatives_data_sync
                deriv_data = fetch_derivatives_data_sync(symbol)
                
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
        
        # Shadow arena
        try:
            from src.shadow_arena import evaluate_candidates
            evaluate_candidates(symbol, price, df_feat)
        except Exception:
            pass
        
        x = torch.tensor(data_scaled).unsqueeze(0).to(device)
        
        # Fast ensemble inference - single batched forward pass
        with _model_inference_lock:
            with torch.no_grad():
                # Get ensemble predictions: [batch, ensemble_size, output_dim]
                stats = model.get_ensemble_stats(x)
                mean_logit = stats['mean'].squeeze().item()
                epistemic_var = stats['variance'].squeeze().item()
                
                # Also run MC-dropout for aleatoric uncertainty if live
                aleatoric_var = 0.0
                if is_live:
                    model.train()  # enable dropout
                    mc_logits = []
                    for _ in range(MC_DROPOUT_PASSES):
                        with torch.no_grad():
                            mc_out = model(x)  # [B, E, D]
                            mc_logit = mc_out[:, :, -1].mean(dim=1).squeeze().item()
                            mc_logits.append(mc_logit)
                    model.eval()
                    aleatoric_var = np.var(mc_logits) if len(mc_logits) > 1 else 0.0
                
                out_prob = float(torch.sigmoid(torch.tensor(mean_logit)).item())
                
                # Modulate weight by total uncertainty
                total_uncertainty = epistemic_var + aleatoric_var
                transformer_weight = 0.35 * (1.0 / (1.0 + total_uncertainty))
                transformer_weight = max(0.05, min(0.50, transformer_weight))
                
                # Causal reasoning if decisive
                causal_reasoning = {}
                if out_prob > 0.58 or out_prob < 0.42:
                    x.requires_grad = True
                    raw_logit = model(x).mean(dim=1).squeeze()  # Average over ensemble
                    raw_logit.backward()
                    grads = x.grad[0, -1, :].cpu().numpy()
                    feats = data_scaled[-1, :]
                    contributions = grads * feats
                    for i, col in enumerate(cols):
                        causal_reasoning[col] = float(contributions[i])
                
                return out_prob, mean_logit, epistemic_var, aleatoric_var, causal_reasoning, data_scaled.tolist()
        
        return None
    
    return await asyncio.to_thread(_do_inference)


async def transformer_brain(symbol: str, price: float, signal: dict) -> BrainVote:
    """Evaluates Grok GQA PyTorch Transformer model prediction with graceful fallback.
    
    Supports three modes:
    - Standard: Single model with MC-dropout uncertainty (default)
    - Bayesian: Deep Ensemble + MC-dropout + Temperature Calibration (if USE_BAYESIAN_TRANSFORMER=True)
    - Fast BatchEnsemble: Single batched forward pass with rank-1 perturbations (if USE_FAST_ENSEMBLE=True)
    """
    # Use Fast BatchEnsemble if enabled (5x faster than Deep Ensemble)
    if USE_FAST_ENSEMBLE:
        predictor = get_fast_ensemble_predictor()
        if predictor is not None:
            res = await _run_fast_ensemble_inference(symbol, price, signal, predictor)
            if res is not None:
                prob, logit, epistemic_var, aleatoric_var, causal_reasoning_dict, tensor_state = res
                reason = f"Fast Ensemble prob={prob:.3f} (epi={epistemic_var:.4f}, alea={aleatoric_var:.4f})"
                # Weight already modulated by uncertainty in inference
                transformer_weight = 0.35 * (1.0 / (1.0 + epistemic_var + aleatoric_var))
                transformer_weight = max(0.05, min(0.50, transformer_weight))
                
                if prob > 0.55:
                    action = "buy"
                elif prob < 0.45:
                    action = "sell"
                else:
                    action = "hold"
                
                return BrainVote(
                    name="transformer",
                    action=action,
                    confidence=prob,
                    weight=transformer_weight,
                    regime=signal.get("regime", "unknown"),
                    reason=reason,
                    causal_reasoning=causal_reasoning_dict,
                    tensor_state=tensor_state
                )
        # Fall through to standard mode if fast ensemble fails
    
    # Use Bayesian ensemble if enabled
    if USE_BAYESIAN_TRANSFORMER:
        from .bayesian_transformer import bayesian_transformer_brain
        return await bayesian_transformer_brain(
            symbol, price, signal,
            ensemble_size=ENSEMBLE_SIZE,
            mc_passes=MC_DROPOUT_PASSES,
        )
    
    # Standard mode (original implementation)
    if not getattr(settings, 'ADAPTIVE_ML_ENABLED', True):
        # We are in backtest/analysis mode where ML is disabled to save time
        return BrainVote(
            name="transformer",
            action="stand_aside",
            confidence=0.0,
            weight=0.35,
            regime=signal.get("regime", "unknown"),
            reason="Analysis mode - ML disabled"
        )
        
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

    causal_reasoning_dict = None
    tensor_state = None
    transformer_weight = 0.35
    if predictor is not None:
        try:
            import asyncio
            import os
            from src.feature_engineering import add_features
            # fetch_bars is actually located in data_fetcher or similar, wait I will just copy the import logic
            from src.data_fetcher import fetch_bars
                
            from alpaca.data.historical import CryptoHistoricalDataClient

            torch = predictor["torch"]
            model = predictor["model"]
            scaler = predictor["scaler"]
            device = predictor["device"]
            
            def _do_inference():
                backtest_df = signal.get("backtest_df")
                if backtest_df is not None and len(backtest_df) >= 32:
                    import pandas as pd
                    import polars as pl
                    if isinstance(backtest_df, pl.DataFrame):
                        df_raw = backtest_df.to_pandas()
                    else:
                        df_raw = backtest_df.copy()
                    if "t" in df_raw.columns:
                        df_raw = df_raw.rename(columns={"t": "timestamp"})
                        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
                        df_raw.set_index("timestamp", inplace=True)
                    if "vwap" not in df_raw.columns:
                        df_raw["vwap"] = df_raw["close"]
                    if "trade_count" not in df_raw.columns:
                        df_raw["trade_count"] = 0.0
                    is_live = False
                else:
                    key = os.getenv("APCA_API_KEY_ID")
                    secret = os.getenv("APCA_API_SECRET_KEY")
                    if not key or not secret:
                        return None
    
                    client = _get_data_client()
                    alpaca_symbol = symbol.replace("-", "/") if "-" in symbol else symbol
                    df_raw = fetch_bars(client, alpaca_symbol, days=2)
                    if df_raw is None or len(df_raw) < 32:
                        return None
                    is_live = True
    
                # Inject On-Chain / Derivatives Data (live trading only - skip during backtest to avoid lookahead bias)
                if not is_live:
                    df_raw["funding_rate"] = 0.0
                    df_raw["open_interest"] = 0.0
                    df_raw["long_short_ratio"] = 1.0
                try:
                    if is_live:
                        from src.onchain_data import fetch_derivatives_data_sync
                        deriv_data = fetch_derivatives_data_sync(symbol)
                        
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
    
                # Protect all model state access (forward, train/eval mode, backward)
                # with a lock to prevent concurrent access from corrupting model state
                # across concurrently evaluated symbols.
                with _model_inference_lock:
                    # Cheap forward-only pass first (no autograd graph built) to
                    # get the probability. Measured: the full forward+backward
                    # pass used purely for gradient-based causal attribution
                    # costs ~110ms of extra event-loop scheduling delay (GIL
                    # contention from autograd bookkeeping) even though it runs
                    # inside asyncio.to_thread, on top of being wasted work for
                    # the majority of evaluations that end in "hold" and are
                    # never persisted to the decision-snapshot DB anyway.
                    with torch.no_grad():
                        raw_logit_fwd = model(x).squeeze(1)
                        out_prob = float(torch.sigmoid(raw_logit_fwd).item())
    
                    # MC-dropout for uncertainty estimation (10 stochastic forward passes)
                    # Only enable dropout during uncertainty estimation if in live mode (not backtest)
                    # to avoid slowing down backtest unnecessarily; but we keep it lightweight.
                    if not is_live:
                        # In backtest, we skip MC-dropout to save time; use fixed weight.
                        transformer_weight = 0.35  # base weight
                    else:
                        model.train()  # enable dropout
                        mc_probs = []
                        for _ in range(10):
                            with torch.no_grad():
                                mc_logit = model(x).squeeze(1)
                                mc_prob = float(torch.sigmoid(mc_logit).item())
                                mc_probs.append(mc_prob)
                        model.eval()  # back to eval mode
                        prob_var = np.var(mc_probs) if len(mc_probs) > 1 else 0.0
                        # Modulate weight: higher variance -> lower weight
                        transformer_weight = 0.35 * (1.0 / (1.0 + prob_var))  # simple inverse scaling
    
                    causal_reasoning = {}
                    logit_value = raw_logit_fwd.item()
    
                    # Only pay for the backward pass when this brain itself would
                    # actually vote buy/sell (thresholds mirror the decision logic
                    # below). This doesn't guarantee the committee as a whole will
                    # trade (other brains can still veto/override), but it's the
                    # best signal available at this point without restructuring
                    # the concurrent 5-brain gather in committee.py, and it skips
                    # the expensive path for the (typical) majority of "hold"
                    # evaluations.
                    if out_prob > 0.58 or out_prob < 0.42:
                        x.requires_grad = True
    
                        # Causal Reasoning: Approximate SHAP via Gradients * Input
                        raw_logit = model(x).squeeze(1)
                        raw_logit.backward()
    
                        # Gradients w.r.t the last timestep's features
                        grads = x.grad[0, -1, :].cpu().numpy()
                        feats = data_scaled[-1, :]
    
                        # Simple Feature Importance (Gradient * Input)
                        contributions = grads * feats
    
                        # Map contributions to feature names
                        for i, col in enumerate(cols):
                            causal_reasoning[col] = float(contributions[i])
    
                        logit_value = raw_logit.item()
    
                    return out_prob, logit_value, causal_reasoning, data_scaled.tolist()
                
            res = await asyncio.to_thread(_do_inference)
            
            if res is not None:
                prob, logit, causal_reasoning_dict, tensor_state = res
                reason = f"Grok PyTorch Inference prob={prob:.3f} (logit={logit:.2f})"
        except Exception as e:
            import logging
            logging.getLogger("committee").error(f"Transformer inference error: {e}")
    
    # Threshold probability decision logic
    # Use wider thresholds (0.55/0.45) so the model's directional calls
    # aren't suppressed by the tight 0.58/0.42 HOLD band. In sideways
    # markets the model rarely reaches 0.58, so it was effectively
    # disabled -- always voting HOLD and leaving the committee to
    # the quant brain alone.
    if prob > 0.55:
        action = "buy"
    elif prob < 0.45:
        action = "sell"
    else:
        action = "hold"
    
    return BrainVote(
        name="transformer",
        action=action,
        confidence=prob,
        weight=transformer_weight,
        regime=regime,
        reason=reason,
        causal_reasoning=causal_reasoning_dict,
        tensor_state=tensor_state
    )
