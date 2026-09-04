"""Bayesian / Deep Ensemble Transformer Brain for Calibrated Uncertainty.

Wraps the existing transformer model(s) to provide:
1. Deep Ensemble: Load multiple checkpoints, average predictions
2. MC-Dropout: Stochastic forward passes for epistemic uncertainty
3. Probability Calibration: Platt scaling / temperature scaling
4. Calibrated confidence intervals for committee weight modulation
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import settings
from src.logging_config import get_logger
from .models import BrainVote

from .models import BrainVote

logger = get_logger("bayesian_transformer")

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
ENSEMBLE_DIR = os.path.join(MODEL_DIR, 'ensemble')
CALIBRATION_PATH = os.path.join(MODEL_DIR, 'transformer_calibration.json')

# Default ensemble size
DEFAULT_ENSEMBLE_SIZE = 5
MC_DROPOUT_PASSES = 20


@dataclass
class EnsemblePrediction:
    """Prediction with uncertainty quantification."""
    mean_prob: float
    std_prob: float
    epistemic_uncertainty: float  # Model uncertainty (ensemble variance)
    aleatoric_uncertainty: float  # Data uncertainty (MC-dropout variance)
    calibrated_prob: float
    confidence_interval: tuple[float, float]  # 95% CI
    logit: float
    all_probs: list[float]


class TemperatureScaler(nn.Module):
    """Temperature scaling for probability calibration (Guo et al., 2017)."""
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)


class BayesianTransformerBrain:
    """Ensemble of transformers with MC-dropout and calibration."""
    
    def __init__(
        self,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        mc_passes: int = MC_DROPOUT_PASSES,
        calibration_path: str = CALIBRATION_PATH,
    ):
        self.ensemble_size = ensemble_size
        self.mc_passes = mc_passes
        self.calibration_path = calibration_path
        
        self.models: list[nn.Module] = []
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 11
        self.calibration_temp = 1.0
        self._lock = threading.Lock()
        self._initialized = False
    
    def load_ensemble(self, base_model_path: str, scaler_path: str) -> bool:
        """Load ensemble from checkpoint directory.
        
        Looks for:
        - {base_model_path}_ensemble_{i}.pth for i in 0..ensemble_size-1
        - Or just the base model if no ensemble found
        """
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
            self.input_dim = self.scaler.n_features_in_ if hasattr(self.scaler, "n_features_in_") else 11
            
            # Try to load ensemble members
            self.models = []
            for i in range(self.ensemble_size):
                ensemble_path = f"{base_model_path}_ensemble_{i}.pth"
                if not os.path.exists(ensemble_path):
                    # Fallback: use base model for missing members
                    if i == 0 and os.path.exists(base_model_path):
                        ensemble_path = base_model_path
                    else:
                        logger.warning(f"Ensemble member {i} not found at {ensemble_path}")
                        continue
                
                # Load architecture config
                config_path = os.path.join(os.path.dirname(base_model_path), "transformer_config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        arch = json.load(f)
                    model = self._create_model(arch)
                else:
                    model = self._create_model({})
                
                state_dict = torch.load(ensemble_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
            
            if not self.models:
                logger.error("No ensemble models loaded")
                return False
            
            # Load calibration temperature if available
            if os.path.exists(self.calibration_path):
                with open(self.calibration_path) as f:
                    cal = json.load(f)
                    self.calibration_temp = cal.get("temperature", 1.0)
                    logger.info(f"Loaded calibration temperature: {self.calibration_temp:.4f}")
            
            self._initialized = True
            logger.info(f"Loaded Bayesian ensemble with {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            self.models = []
            return False
    
    def _create_model(self, arch: dict) -> nn.Module:
        """Create transformer model from architecture config."""
        # Import the model class from transformer_brain
        from .transformer_brain import GrokGQA_Transformer
        return GrokGQA_Transformer(
            input_dim=self.input_dim,
            num_layers=arch.get("num_layers", 4),
            embed_dim=arch.get("embed_dim", 128),
            num_q_heads=arch.get("num_q_heads", 8),
            num_kv_heads=arch.get("num_kv_heads", 2),
            dropout=arch.get("dropout", 0.1),
        )
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        use_mc_dropout: bool = True,
    ) -> EnsemblePrediction:
        """Run ensemble inference with uncertainty quantification.
        
        Returns:
            EnsemblePrediction with mean, std, epistemic/aleatoric uncertainty,
            calibrated probability, and confidence interval.
        """
        if not self._initialized or not self.models:
            raise RuntimeError("Ensemble not initialized")
        
        all_probs = []
        all_logits = []
        
        with self._lock:
            for model in self.models:
                model.eval()
                
                # Standard forward pass
                with torch.no_grad():
                    logit = model(x).squeeze(1)
                    prob = torch.sigmoid(logit / self.calibration_temp).item()
                    all_logits.append(logit.item())
                    all_probs.append(prob)
                
                # MC-Dropout passes for aleatoric uncertainty
                if use_mc_dropout:
                    model.train()  # Enable dropout
                    mc_probs = []
                    for _ in range(self.mc_passes):
                        with torch.no_grad():
                            mc_logit = model(x).squeeze(1)
                            mc_prob = torch.sigmoid(mc_logit / self.calibration_temp).item()
                            mc_probs.append(mc_prob)
                    model.eval()
                    
                    if mc_probs:
                        _aleatoric_var = np.var(mc_probs)
                        all_probs.extend(mc_probs)
            
            # Compute statistics
            mean_prob = float(np.mean(all_probs))
            std_prob = float(np.std(all_probs))
            
            # Epistemic uncertainty = variance across ensemble members
            ensemble_probs = all_probs[:len(self.models)]
            epistemic_uncertainty = float(np.var(ensemble_probs)) if len(ensemble_probs) > 1 else 0.0
            
            # Aleatoric uncertainty = average MC-dropout variance per model
            aleatoric_uncertainty = 0.0
            if use_mc_dropout and len(all_probs) > len(self.models):
                mc_probs = all_probs[len(self.models):]
                # Average variance per model
                n_mc = self.mc_passes
                for i in range(len(self.models)):
                    model_mc = mc_probs[i * n_mc:(i + 1) * n_mc]
                    if len(model_mc) > 1:
                        aleatoric_uncertainty += np.var(model_mc)
                aleatoric_uncertainty /= max(1, len(self.models))
            
            # Calibrated probability (already temperature-scaled)
            calibrated_prob = mean_prob
            
            # 95% confidence interval
            ci_low = max(0.0, mean_prob - 1.96 * std_prob)
            ci_high = min(1.0, mean_prob + 1.96 * std_prob)
            
            return EnsemblePrediction(
                mean_prob=mean_prob,
                std_prob=std_prob,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                calibrated_prob=calibrated_prob,
                confidence_interval=(ci_low, ci_high),
                logit=float(np.mean(all_logits)),
                all_probs=all_probs,
            )
    
    def compute_weight_from_uncertainty(
        self,
        pred: EnsemblePrediction,
        base_weight: float = 0.35,
        uncertainty_penalty: float = 1.0,
    ) -> float:
        """Compute committee weight from uncertainty.
        
        Higher uncertainty -> lower weight.
        Uses both epistemic (model) and aleatoric (data) uncertainty.
        """
        total_uncertainty = pred.epistemic_uncertainty + pred.aleatoric_uncertainty
        # Penalize weight: weight = base_weight / (1 + penalty * uncertainty)
        weight = base_weight / (1.0 + uncertainty_penalty * total_uncertainty)
        return max(0.05, min(0.50, weight))  # Clamp to reasonable range
    
    def get_causal_reasoning(
        self,
        x: torch.Tensor,
        cols: list[str],
        threshold: float = 0.58,
    ) -> dict[str, float] | None:
        """Compute gradient-based feature importance (SHAP-like).
        
        Only runs if probability is above/below decision threshold.
        """
        if not self.models:
            return None
        
        model = self.models[0]  # Use first model for gradients
        model.eval()
        
        with self._lock:
            x_grad = x.clone().detach().requires_grad_(True)
            
            with torch.enable_grad():
                logit = model(x_grad).squeeze(1)
                logit.backward()
            
            grads = x_grad.grad[0, -1, :].cpu().numpy()
            feats = x[0, -1, :].cpu().numpy()
            
            contributions = grads * feats
            
            causal = {}
            for i, col in enumerate(cols):
                if i < len(contributions):
                    causal[col] = float(contributions[i])
            
            return causal


# Global instance
_bayesian_brain: BayesianTransformerBrain | None = None


def get_bayesian_transformer(
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    mc_passes: int = MC_DROPOUT_PASSES,
) -> BayesianTransformerBrain | None:
    """Get or create the global Bayesian transformer instance."""
    global _bayesian_brain
    if _bayesian_brain is not None:
        return _bayesian_brain
    
    _bayesian_brain = BayesianTransformerBrain(
        ensemble_size=ensemble_size,
        mc_passes=mc_passes,
    )
    
    # Try to load ensemble
    base_path = settings.TRANSFORMER_MODEL_PATH
    scaler_path = settings.TRANSFORMER_SCALER_PATH
    
    if not _bayesian_brain.load_ensemble(base_path, scaler_path):
        _bayesian_brain = None
        return None
    
    return _bayesian_brain


def reset_bayesian_transformer() -> None:
    """Reset global instance."""
    global _bayesian_brain
    _bayesian_brain = None


async def bayesian_transformer_brain(
    symbol: str,
    price: float,
    signal: dict,
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    mc_passes: int = MC_DROPOUT_PASSES,
) -> BrainVote:
    """Bayesian transformer brain with calibrated uncertainty.
    
    Replaces the standard transformer_brain with uncertainty-aware predictions.
    """

    if not getattr(settings, 'ADAPTIVE_ML_ENABLED', True):
        return BrainVote(
            name="transformer",
            action="stand_aside",
            confidence=0.0,
            weight=0.35,
            regime=signal.get("regime", "unknown"),
            reason="Analysis mode - ML disabled"
        )
    
    brain = get_bayesian_transformer(ensemble_size, mc_passes)
    raw_action = signal.get("action", "hold")
    regime = signal.get("regime", "unknown")
    reason = "Signal fallback confidence"
    
    # Fallback prob
    if raw_action == "buy":
        prob = 0.65
    elif raw_action == "sell":
        prob = 0.35
    else:
        prob = 0.50
    
    causal_reasoning_dict = None
    tensor_state = None
    transformer_weight = 0.35
    
    if brain is not None:
        try:
            import asyncio
            import os

            from src.data_fetcher import fetch_bars
            from src.feature_engineering import add_features
            
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
                    
                    from .transformer_brain import _get_data_client
                    client = _get_data_client()
                    alpaca_symbol = symbol.replace("-", "/") if "-" in symbol else symbol
                    df_raw = fetch_bars(client, alpaca_symbol, days=2)
                    if df_raw is None or len(df_raw) < 32:
                        return None
                    is_live = True
                
                # On-chain data
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
                
                if hasattr(brain.scaler, "feature_names_in_"):
                    cols = list(brain.scaler.feature_names_in_)
                else:
                    from src.feature_engineering import get_active_features
                    cols = get_active_features()
                
                data = df_feat[cols].tail(32).values.astype(np.float32)
                if len(data) < 32:
                    return None
                    
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data = np.clip(data, -1e6, 1e6)
                data_scaled = brain.scaler.transform(data).astype(np.float32)
                data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Shadow arena
                try:
                    from src.shadow_arena import evaluate_candidates
                    evaluate_candidates(symbol, price, df_feat)
                except Exception:
                    pass
                
                x = torch.tensor(data_scaled).unsqueeze(0).to(brain.device)
                
                # Bayesian prediction with uncertainty
                pred = brain.predict_with_uncertainty(x, use_mc_dropout=is_live)
                
                # Compute weight from uncertainty
                _transformer_weight = brain.compute_weight_from_uncertainty(pred)
                
                # Causal reasoning if decisive
                causal_reasoning = None
                if pred.mean_prob > 0.58 or pred.mean_prob < 0.42:
                    causal_reasoning = brain.get_causal_reasoning(x, cols)
                
                return pred.mean_prob, pred.logit, pred.epistemic_uncertainty, pred.aleatoric_uncertainty, causal_reasoning, data_scaled.tolist()
            
            res = await asyncio.to_thread(_do_inference)
            
            if res is not None:
                prob, logit, epistemic, aleatoric, causal_reasoning_dict, tensor_state = res
                reason = f"Bayesian Ensemble prob={prob:.3f} (epi={epistemic:.4f}, alea={aleatoric:.4f})"
        except Exception as e:
            import logging
            logging.getLogger("committee").error(f"Bayesian transformer inference error: {e}")
    
    # Threshold decision
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


# ── Calibration Training ────────────────────────────────────────────

def train_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.01,
    epochs: int = 100,
) -> float:
    """Train temperature scaling on validation set.
    
    Args:
        logits: Raw model logits (N,)
        labels: Binary labels 0/1 (N,)
    
    Returns:
        Optimal temperature parameter
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    
    scaler = TemperatureScaler(1.0).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=epochs)
    
    def eval_loss():
        optimizer.zero_grad()
        calibrated_logits = scaler(logits_t)
        loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels_t)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    optimal_temp = scaler.temperature.item()
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    
    return optimal_temp


def save_calibration(temperature: float, path: str = CALIBRATION_PATH) -> None:
    """Save calibration temperature."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).
    
    Lower is better. Perfect calibration = 0.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if not np.any(mask):
            continue
        bin_conf = np.mean(probs[mask])
        bin_acc = np.mean(labels[mask])
        ece += np.abs(bin_conf - bin_acc) * np.sum(mask) / len(probs)
    return ece


if __name__ == "__main__":
    # Quick test
    print("Bayesian Transformer module loaded successfully")
    print(f"Ensemble dir: {ENSEMBLE_DIR}")
    print(f"Calibration path: {CALIBRATION_PATH}")