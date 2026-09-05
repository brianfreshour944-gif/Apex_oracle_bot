"""Decision Transformer for Committee Decision Making.

Offline RL via sequence modeling: conditions on target return to generate
brain weights, position sizing, and action decisions. Trained on historical
decision snapshots (backtest + live trades).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.config import settings
from src.logging_config import get_logger

from .models import CommitteeResult

logger = get_logger("decision_transformer")

# Canonical brain roster (must match committee.py BRAINS)
BRAINS = ["transformer", "quant", "momentum", "sentinel", "llm"]
REGIMES = [
    "bull", "bear", "trending", "sideways",
    "high_volatility", "low_volatility", "neutral", "default"
]
ACTIONS = ["buy", "sell", "hold", "stand_aside", "skip"]

# Sequence / model hyperparameters
CONTEXT_LENGTH = 32          # max historical decisions in context
EMBED_DIM = 256
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.1
TARGET_RETURN_SCALE = 100.0  # normalize return_pct to ~[-1, 1]

# Quantile Regression settings (CVaR-aware Decision Transformer)
NUM_QUANTILES = 5
QUANTILES = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])  # tau values for CVaR
CVAR_QUANTILE_IDX = 0  # tau=0.05 for CVaR (worst-case)
MEDIAN_QUANTILE_IDX = 2  # tau=0.50 for size/threshold decisions

# Adaptive RTG settings
ADAPTIVE_RTG_ENABLED = getattr(settings, 'ADAPTIVE_RTG_ENABLED', True)
TARGET_ANNUAL_SHARPE = getattr(settings, 'TARGET_ANNUAL_SHARPE', 1.5)
KALMAN_GAIN = getattr(settings, 'KALMAN_GAIN', 0.1)

# Spectral Norm settings
USE_SPECTRAL_NORM = getattr(settings, 'USE_SPECTRAL_NORM', True)

# CQL settings
USE_CQL = getattr(settings, 'USE_CQL', True)
CQL_WEIGHT = getattr(settings, 'CQL_WEIGHT', 0.1)

# Temporal Action Smoothing (EMA)
TEMPORAL_EMA_ALPHA = getattr(settings, 'TEMPORAL_EMA_ALPHA', 0.3)

# Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DT_MODEL_PATH = os.path.join(MODEL_DIR, 'decision_transformer.pth')
DT_CONFIG_PATH = os.path.join(MODEL_DIR, 'decision_transformer_config.json')


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Adaptive RTG Scheduler ────────────────────────────────────────────

class AdaptiveRTGScheduler:
    """Dynamically scales Return-to-Go by realized volatility and target Sharpe.
    
    Instead of a fixed target return (e.g., +1.5%), computes RTG based on:
    - Online volatility estimate (exponential weighted)
    - Target annual Sharpe ratio
    - Time horizon
    
    This prevents reckless targets during high volatility and captures
    opportunity during low volatility.
    """
    
    def __init__(
        self,
        target_annual_sharpe: float = TARGET_ANNUAL_SHARPE,
        kalman_gain: float = KALMAN_GAIN,
        min_vol: float = 0.005,
        max_vol: float = 0.10,
    ):
        self.target_annual_sharpe = target_annual_sharpe
        self.kalman_gain = kalman_gain
        self.min_vol = min_vol
        self.max_vol = max_vol
        
        # Online volatility estimate (daily)
        self.vol_estimate = 0.02  # Starting 2% daily vol
        self._initialized = False
    
    def update(self, realized_return: float) -> None:
        """Update volatility estimate with new realized return.
        
        Uses exponential moving average (Kalman filter with fixed gain)
        for online adaptation to changing market conditions.
        """
        abs_return = abs(realized_return)
        if not self._initialized:
            self.vol_estimate = abs_return
            self._initialized = True
        else:
            # EMA update: vol = (1 - K) * vol + K * |return|
            self.vol_estimate = (1 - self.kalman_gain) * self.vol_estimate + self.kalman_gain * abs_return
        
        # Clip to reasonable bounds
        self.vol_estimate = float(np.clip(self.vol_estimate, self.min_vol, self.max_vol))
    
    def get_rtg(self, current_price: float, time_horizon_days: float = 1.0) -> float:
        """Compute adaptive target price based on volatility and target Sharpe.
        
        RTG = current_price * (1 + target_daily_return * sqrt(horizon))
        
        target_daily_return = (target_annual_sharpe * daily_vol) / sqrt(252)
                          = target_annual_sharpe * daily_vol / 15.87
        
        Args:
            current_price: Current asset price
            time_horizon_days: Investment horizon in days
            
        Returns:
            Target price for conditioning the Decision Transformer
        """
        # Convert annual Sharpe to daily return target
        # Annual Sharpe = daily_return / daily_vol * sqrt(252)
        # => daily_return = Sharpe * daily_vol / sqrt(252)
        daily_return_target = self.target_annual_sharpe * self.vol_estimate / 16.0  # ~sqrt(252)
        
        # Scale by sqrt of time horizon
        horizon_return = daily_return_target * np.sqrt(time_horizon_days)
        
        target_price = current_price * (1.0 + horizon_return)
        return float(target_price)
    
    def get_rtg_scaled(self, time_horizon_days: float = 1.0) -> float:
        """Get RTG normalized by TARGET_RETURN_SCALE for model input."""
        # This is used as the returns-to-go token in the sequence
        daily_return_target = self.target_annual_sharpe * self.vol_estimate / 16.0
        horizon_return = daily_return_target * np.sqrt(time_horizon_days)
        return horizon_return / (TARGET_RETURN_SCALE / 100.0)  # Convert to pct then scale
    
    def get_state(self) -> dict[str, float]:
        return {
            'vol_estimate': self.vol_estimate,
            'initialized': self._initialized,
            'target_annual_sharpe': self.target_annual_sharpe,
        }
    
    def load_state(self, state: dict[str, float]) -> None:
        self.vol_estimate = state.get('vol_estimate', 0.02)
        self._initialized = state.get('initialized', False)


# ── Spectral Normalization for Output Heads ──────────────────────────

def apply_spectral_norm_to_module(module: nn.Module, name: str = '') -> None:
    """Recursively apply spectral normalization to Linear layers in output heads.
    
    Spectral normalization constrains the Lipschitz constant of the network,
    preventing exploding outputs on OOD inputs.
    """
    for child_name, child in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        
        if isinstance(child, nn.Linear):
            # Check if this is an output head
            if any(key in full_name.lower() for key in ['head', 'output', 'weight', 'size', 'threshold', 'action']):
                # Apply spectral norm
                setattr(module, child_name, spectral_norm(child))
                logger.debug(f"Applied SpectralNorm to {full_name}")
        else:
            apply_spectral_norm_to_module(child, full_name)


def wrap_output_heads_with_spectral_norm(model: nn.Module) -> nn.Module:
    """Wrap all output heads with spectral normalization.
    
    Call this after model creation to add Lipschitz constraints.
    """
    # Find and wrap output heads
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Heuristic: output heads typically have small output dimensions
            # or specific names
            if module.out_features <= 10 or any(
                key in name.lower() for key in ['head', 'output', 'weight', 'size', 'threshold', 'action']
            ):
                spectral_norm(module)
                logger.debug(f"Applied SpectralNorm to {name}")
    return model


# ── Quantile Regression Loss (Pinball Loss for CVaR) ──────────────────

def quantile_huber_loss(
    pred_quantiles: torch.Tensor,  # [batch, num_quantiles]
    target: torch.Tensor,          # [batch, 1] - realized return
    quantiles: torch.Tensor,       # [num_quantiles]
    delta: float = 1.0,            # Huber threshold
) -> torch.Tensor:
    """Quantile Huber Loss (QR-DQN style) for distributional RL.
    
    Args:
        pred_quantiles: Predicted quantiles [batch, num_quantiles]
        target: Realized returns [batch, 1]
        quantiles: Quantile levels [num_quantiles] (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
        delta: Huber threshold for smooth L1
        
    Returns:
        Scalar loss
    """
    # target: [batch, 1] -> [batch, num_quantiles] for broadcasting
    target = target.expand_as(pred_quantiles)
    quantiles = quantiles.to(pred_quantiles.device).view(1, -1)
    
    error = target - pred_quantiles  # [batch, num_quantiles]
    
    # Quantile loss: max(tau * error, (tau - 1) * error)
    # Smoothed with Huber
    tau = quantiles
    abs_error = torch.abs(error)
    
    # Huber quantization: if |error| < delta: 0.5 * error^2 / delta else: |error| - 0.5 * delta
    huber = torch.where(
        abs_error < delta,
        0.5 * error * error / delta,
        abs_error - 0.5 * delta
    )
    
    # Quantile weighting
    loss = torch.where(error >= 0, tau * huber, (1 - tau) * huber)
    
    return loss.mean()


# ── Quantile Regression Decision Transformer ──────────────────────────

def compute_cql_loss(
    model: nn.Module,
    ood_states: torch.Tensor,
    ood_actions: torch.Tensor,
    weight: float = CQL_WEIGHT,
) -> torch.Tensor:
    """Compute Conservative Q-Learning regularization loss.
    
    CQL minimizes the likelihood of OOD (out-of-distribution) actions,
    pushing the policy towards uniform/in-distribution behavior.
    
    Args:
        model: The Decision Transformer model
        ood_states: OOD state samples [batch, seq_len, state_dim]
        ood_actions: OOD action samples [batch, seq_len, act_dim]
        weight: CQL loss weight
        
    Returns:
        CQL loss tensor
    """
    if not USE_CQL:
        return torch.tensor(0.0, device=ood_states.device)
    
    # Forward pass on OOD data
    with torch.no_grad():
        batch_size, seq_len = ood_states.shape[:2]
        # Create dummy RTG and timesteps for OOD
        rtg = torch.zeros(batch_size, seq_len, 1, device=ood_states.device)
        timesteps = torch.arange(seq_len, device=ood_states.device).unsqueeze(0).expand(batch_size, -1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=ood_states.device)
    
    # Get action logits from model
    outputs = model(ood_states, ood_actions, rtg, timesteps, mask)
    action_logits = outputs['action_logits']  # [batch, num_actions]
    
    # CQL loss: minimize log-sum-exp of action logits (pushes towards uniform)
    # logsumexp acts as soft maximum
    cql_loss = torch.logsumexp(action_logits, dim=-1).mean()
    
    return weight * cql_loss


# Global RTG scheduler instance
_rtg_scheduler: AdaptiveRTGScheduler | None = None


def get_rtg_scheduler() -> AdaptiveRTGScheduler:
    global _rtg_scheduler
    if _rtg_scheduler is None:
        _rtg_scheduler = AdaptiveRTGScheduler()
    return _rtg_scheduler


def reset_rtg_scheduler() -> None:
    global _rtg_scheduler
    _rtg_scheduler = None


# ── Model Architecture ────────────────────────────────────────────────

class DecisionTransformer(nn.Module):
    """GPT-style transformer for offline RL decision making.

    Input sequence: (regime, features, brain_votes, action, return_pct) per timestep
    Output: brain_weights, position_size_mult, confidence_threshold, action_logits
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_length: int = CONTEXT_LENGTH,
        embed_dim: int = EMBED_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.embed_dim = embed_dim

        # --- Embedding layers ---
        # State: regime_onehot + features + brain_votes
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Action: brain_weights (5) + size_mult (1) + conf_threshold (1) + action_logits (5)
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Return-to-go (target return) embedding
        self.rtg_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Timestep embedding
        self.timestep_encoder = nn.Embedding(max_length, embed_dim)

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.brain_weight_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, len(BRAINS)),
        )
        self.size_mult_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.conf_threshold_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, len(ACTIONS)),
        )

        # ── Quantile Regression Heads (CVaR-aware) ──
        # Quantile action head: [batch, num_quantiles * num_actions]
        self.quantile_action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, NUM_QUANTILES * len(ACTIONS)),
        )
        # Quantile size head: [batch, num_quantiles]
        self.quantile_size_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, NUM_QUANTILES),
        )
        # Quantile confidence threshold head: [batch, num_quantiles]
        self.quantile_conf_threshold_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, NUM_QUANTILES),
        )

        # Layer norm for output
        self.output_norm = nn.LayerNorm(embed_dim)

        # Apply spectral normalization to output heads for OOD robustness
        if USE_SPECTRAL_NORM:
            self._apply_spectral_norm()

    def _apply_spectral_norm(self) -> None:
        """Apply spectral normalization to output heads for OOD robustness."""
        # Standard heads
        self.brain_weight_head[2] = spectral_norm(self.brain_weight_head[2])
        self.size_mult_head[2] = spectral_norm(self.size_mult_head[2])
        self.conf_threshold_head[2] = spectral_norm(self.conf_threshold_head[2])
        self.action_head[2] = spectral_norm(self.action_head[2])
        # Quantile heads
        self.quantile_action_head[2] = spectral_norm(self.quantile_action_head[2])
        self.quantile_size_head[2] = spectral_norm(self.quantile_size_head[2])
        self.quantile_conf_threshold_head[2] = spectral_norm(self.quantile_conf_threshold_head[2])
        logger.info("Applied Spectral Normalization to all output heads (including quantile heads)")

    def forward(
        self,
        states: torch.Tensor,      # (batch, seq_len, state_dim)
        actions: torch.Tensor,     # (batch, seq_len, act_dim)
        returns_to_go: torch.Tensor,  # (batch, seq_len, 1)
        timesteps: torch.Tensor,   # (batch, seq_len)
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass. Returns dict of predictions for the last timestep."""
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        state_emb = self.state_encoder(states)        # (batch, seq_len, embed_dim)
        action_emb = self.action_encoder(actions)     # (batch, seq_len, embed_dim)
        rtg_emb = self.rtg_encoder(returns_to_go)     # (batch, seq_len, embed_dim)
        time_emb = self.timestep_encoder(timesteps)   # (batch, seq_len, embed_dim)

        # Interleave: (rtg, state, action) per timestep -> 3 * seq_len tokens
        # This follows the Decision Transformer paper design
        token_embeddings = torch.stack(
            [rtg_emb, state_emb, action_emb], dim=2
        ).reshape(batch_size, 3 * seq_len, self.embed_dim)  # (batch, 3*seq_len, embed_dim)

        # Expand timestep embeddings to match 3 tokens per timestep
        time_emb = time_emb.repeat_interleave(3, dim=1)  # (batch, 3*seq_len, embed_dim)

        # Add positional embedding
        token_embeddings = token_embeddings + time_emb

        # Attention mask: causal + padding
        if attention_mask is not None:
            # Expand mask to 3*seq_len (3 tokens per timestep)
            attention_mask = attention_mask.repeat_interleave(3, dim=1)

        # Transformer
        transformer_out = self.transformer(token_embeddings, src_key_padding_mask=attention_mask)

        # Extract the *state* token predictions (every 3rd token starting at index 1)
        # Indices: 0=rtg, 1=state, 2=action, 3=rtg, 4=state, 5=action, ...
        state_token_indices = torch.arange(1, 3 * seq_len, 3, device=transformer_out.device)
        state_predictions = transformer_out[:, state_token_indices, :]  # (batch, seq_len, embed_dim)

        # Use last timestep's state prediction for action prediction
        last_hidden = self.output_norm(state_predictions[:, -1, :])  # (batch, embed_dim)

        # Output heads
        brain_weights = self.brain_weight_head(last_hidden)      # (batch, 5)
        size_mult = self.size_mult_head(last_hidden)             # (batch, 1)
        conf_threshold = self.conf_threshold_head(last_hidden)   # (batch, 1)
        action_logits = self.action_head(last_hidden)            # (batch, 5)

        # Quantile predictions (for CVaR-aware inference)
        q_action_logits = self.quantile_action_head(last_hidden)      # (batch, 5*5)
        q_size_mult = self.quantile_size_head(last_hidden)            # (batch, 5)
        q_conf_threshold = self.quantile_conf_threshold_head(last_hidden)  # (batch, 5)

        # Reshape quantile outputs
        q_action_logits = q_action_logits.view(batch_size, NUM_QUANTILES, len(ACTIONS))  # [B, 5, 5]
        q_size_mult = q_size_mult.view(batch_size, NUM_QUANTILES)  # [B, 5]
        q_conf_threshold = q_conf_threshold.view(batch_size, NUM_QUANTILES)  # [B, 5]

        return {
            "brain_weights": brain_weights,
            "size_mult": size_mult,
            "conf_threshold": conf_threshold,
            "action_logits": action_logits,
            # Quantile predictions
            "q_action_logits": q_action_logits,       # [B, 5, 5]
            "q_size_mult": q_size_mult,               # [B, 5]
            "q_conf_threshold": q_conf_threshold,     # [B, 5]
        }

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference helper: returns predictions for the last timestep only."""
        return self.forward(states, actions, returns_to_go, timesteps, attention_mask)


# ── State / Action Encoding ──────────────────────────────────────────

@dataclass
class DTConfig:
    state_dim: int
    act_dim: int
    context_length: int
    embed_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    brains: list[str]
    regimes: list[str]
    actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "act_dim": self.act_dim,
            "context_length": self.context_length,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "brains": self.brains,
            "regimes": self.regimes,
            "actions": self.actions,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DTConfig:
        return cls(**d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> DTConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def encode_regime(regime: str, regimes: list[str]) -> np.ndarray:
    """One-hot encode regime."""
    vec = np.zeros(len(regimes), dtype=np.float32)
    if regime in regimes:
        vec[regimes.index(regime)] = 1.0
    return vec


def encode_brain_votes(brain_votes: dict[str, str], brains: list[str]) -> np.ndarray:
    """Encode brain votes as [buy=1, sell=-1, hold/other=0] per brain."""
    vec = np.zeros(len(brains), dtype=np.float32)
    for i, brain in enumerate(brains):
        action = brain_votes.get(brain, "hold")
        if action == "buy":
            vec[i] = 1.0
        elif action == "sell":
            vec[i] = -1.0
    return vec


def encode_features(features: dict[str, Any]) -> np.ndarray:
    """Encode market features (RSI, ATR, MACD, on-chain, sentiment)."""
    # Fixed feature order for consistency
    feature_keys = [
        "rsi", "atr", "macd",
        "funding_rate", "open_interest", "long_short_ratio", "bid_ask_imbalance",
        "sentiment_score", "sentiment_conf",
        "hurst", "adx", "bb_width"
    ]
    vec = np.zeros(len(feature_keys), dtype=np.float32)
    for i, key in enumerate(feature_keys):
        val = features.get(key, 0.0)
        if isinstance(val, (int, float)):
            vec[i] = float(val)
    # Normalize known ranges
    if vec[0] != 0:  # RSI: 0-100 -> -1 to 1
        vec[0] = (vec[0] - 50) / 50
    if vec[1] != 0:  # ATR: scale by typical price
        vec[1] = vec[1] / 1000
    if vec[2] != 0:  # MACD: clip
        vec[2] = np.clip(vec[2], -1, 1)
    return vec


def build_state_vector(
    regime: str,
    features: dict[str, Any],
    brain_votes: dict[str, str],
    regimes: list[str],
    brains: list[str],
) -> np.ndarray:
    """Build full state vector: regime_onehot + features + brain_votes."""
    regime_vec = encode_regime(regime, regimes)
    feat_vec = encode_features(features)
    vote_vec = encode_brain_votes(brain_votes, brains)
    return np.concatenate([regime_vec, feat_vec, vote_vec]).astype(np.float32)


def build_action_vector(
    brain_weights: dict[str, float],
    size_mult: float,
    conf_threshold: float,
    action: str,
    brains: list[str],
    actions: list[str],
) -> np.ndarray:
    """Build action vector: brain_weights + size_mult + conf_threshold + action_onehot."""
    weight_vec = np.array([brain_weights.get(b, 0.0) for b in brains], dtype=np.float32)
    size_vec = np.array([size_mult], dtype=np.float32)
    conf_vec = np.array([conf_threshold], dtype=np.float32)
    action_vec = np.zeros(len(actions), dtype=np.float32)
    if action in actions:
        action_vec[actions.index(action)] = 1.0
    return np.concatenate([weight_vec, size_vec, conf_vec, action_vec]).astype(np.float32)


def decode_action_vector(
    brain_weights: np.ndarray,
    size_mult: np.ndarray,
    conf_threshold: np.ndarray,
    action_logits: np.ndarray,
    brains: list[str],
    actions: list[str],
) -> tuple[dict[str, float], float, float, str]:
    """Decode model outputs to action components."""
    # Softmax brain weights
    brain_weights = F.softmax(torch.tensor(brain_weights), dim=-1).numpy()
    weights_dict = {brains[i]: float(brain_weights[i]) for i in range(len(brains))}

    # Size multiplier: sigmoid scaled to [0.5, 1.75]
    size_mult = float(torch.sigmoid(torch.tensor(size_mult)).item())
    size_mult = 0.5 + size_mult * 1.25

    # Confidence threshold: sigmoid scaled to [0.15, 0.35]
    conf_thresh = float(torch.sigmoid(torch.tensor(conf_threshold)).item())
    conf_thresh = 0.15 + conf_thresh * 0.20

    # Action: argmax over logits
    action_idx = int(np.argmax(action_logits))
    action = actions[action_idx] if action_idx < len(actions) else "stand_aside"

    return weights_dict, size_mult, conf_thresh, action


# ── Model Management ─────────────────────────────────────────────────

_DT_MODEL: DecisionTransformer | None = None
_DT_CONFIG: DTConfig | None = None


def get_decision_transformer() -> DecisionTransformer | None:
    """Load or return cached Decision Transformer model."""
    global _DT_MODEL, _DT_CONFIG
    if _DT_MODEL is not None:
        return _DT_MODEL

    if not os.path.exists(DT_MODEL_PATH) or not os.path.exists(DT_CONFIG_PATH):
        logger.info("Decision Transformer model not found. Run training script first.")
        return None

    try:
        _DT_CONFIG = DTConfig.load(DT_CONFIG_PATH)
        device = _get_device()

        _DT_MODEL = DecisionTransformer(
            state_dim=_DT_CONFIG.state_dim,
            act_dim=_DT_CONFIG.act_dim,
            max_length=_DT_CONFIG.context_length,
            embed_dim=_DT_CONFIG.embed_dim,
            num_layers=_DT_CONFIG.num_layers,
            num_heads=_DT_CONFIG.num_heads,
            dropout=_DT_CONFIG.dropout,
        ).to(device)

        state_dict = torch.load(DT_MODEL_PATH, map_location=device, weights_only=True)
        _DT_MODEL.load_state_dict(state_dict)
        _DT_MODEL.eval()

        logger.info("Decision Transformer loaded successfully.")
        return _DT_MODEL
    except Exception as e:
        logger.error(f"Failed to load Decision Transformer: {e}")
        _DT_MODEL = None
        return None


def reset_decision_transformer() -> None:
    """Clear cached model (for testing / after retraining)."""
    global _DT_MODEL, _DT_CONFIG
    _DT_MODEL = None
    _DT_CONFIG = None


# ── Inference: Committee Integration ──────────────────────────────────

def _get_recent_decisions(limit: int = CONTEXT_LENGTH) -> list[dict[str, Any]]:
    """Fetch recent closed decision snapshots for context."""
    from src.db import get_closed_decision_snapshots
    return get_closed_decision_snapshots(limit=limit)


def _build_context_sequence(
    decisions: list[dict[str, Any]],
    target_return: float,
    regimes: list[str],
    brains: list[str],
    actions: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build transformer input tensors from historical decisions.

    Returns: (states, actions, returns_to_go, timesteps, attention_mask)
    All tensors are (1, seq_len, ...) with batch=1.
    """
    seq_len = min(len(decisions), CONTEXT_LENGTH)
    if seq_len == 0:
        # Return empty tensors (will be padded)
        return _empty_tensors()

    # Take most recent `seq_len` decisions (already sorted by closed_at desc)
    decisions = decisions[:seq_len]

    # Build state/action sequences
    states = []
    actions_list = []
    returns_to_go = []
    timesteps = list(range(seq_len))

    # Target return-to-go for the *next* decision (what we want to achieve)
    # For historical decisions, use their realized return as the "target" they achieved
    for _i, dec in enumerate(decisions):
        regime = dec.get("regime", "default")
        features = dec.get("features", {})
        brain_votes = dec.get("brain_votes", {})

        state = build_state_vector(regime, features, brain_votes, regimes, brains)
        states.append(state)

        # Action vector from this decision
        brain_weights = {}
        for brain, _vote in brain_votes.items():
            brain_weights[brain] = 1.0 / len(brain_votes) if brain_votes else 0.0

        action_vec = build_action_vector(
            brain_weights=brain_weights,
            size_mult=1.0,
            conf_threshold=0.15,
            action=dec.get("final_action", "hold"),
            brains=brains,
            actions=actions,
        )
        actions_list.append(action_vec)

        # Return-to-go: realized return for this decision, scaled
        rtn = dec.get("realized_pnl", 0.0) / TARGET_RETURN_SCALE
        returns_to_go.append(rtn)

    # Pad to CONTEXT_LENGTH if needed
    pad_len = CONTEXT_LENGTH - seq_len
    if pad_len > 0:
        state_dim = len(states[0]) if states else _DT_CONFIG.state_dim
        act_dim = len(actions_list[0]) if actions_list else _DT_CONFIG.act_dim
        states = [np.zeros(state_dim, dtype=np.float32)] * pad_len + states
        actions_list = [np.zeros(act_dim, dtype=np.float32)] * pad_len + actions_list
        returns_to_go = [0.0] * pad_len + returns_to_go
        timesteps = list(range(pad_len)) + timesteps

    # Convert to tensors
    device = _get_device()
    states_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device).unsqueeze(0)
    actions_t = torch.tensor(np.stack(actions_list), dtype=torch.float32, device=device).unsqueeze(0)
    rtg_t = torch.tensor(returns_to_go, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    timesteps_t = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.zeros(1, CONTEXT_LENGTH, dtype=torch.bool, device=device)
    attention_mask[0, pad_len:] = True  # True = padding (masked out)

    return states_t, actions_t, rtg_t, timesteps_t, attention_mask


def _empty_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty padded tensors for cold start."""
    device = _get_device()
    state_dim = _DT_CONFIG.state_dim if _DT_CONFIG else 64 + 11 + 5  # regime + features + votes
    act_dim = 5 + 1 + 1 + 5  # weights + size + conf + action
    states = torch.zeros(1, CONTEXT_LENGTH, state_dim, device=device)
    actions = torch.zeros(1, CONTEXT_LENGTH, act_dim, device=device)
    rtg = torch.zeros(1, CONTEXT_LENGTH, 1, device=device)
    timesteps = torch.arange(CONTEXT_LENGTH, device=device).unsqueeze(0)
    mask = torch.ones(1, CONTEXT_LENGTH, dtype=torch.bool, device=device)
    return states, actions, rtg, timesteps, mask


async def run_decision_transformer(
    symbol: str,
    price: float,
    signal: dict[str, Any],
    target_return_pct: float = 2.0,  # target return we want to achieve
) -> CommitteeResult | None:
    """Run Decision Transformer inference for a trading decision.

    Returns CommitteeResult compatible with the committee system, or None if model unavailable.
    """
    model = get_decision_transformer()
    if model is None or _DT_CONFIG is None:
        return None

    try:
        # Use adaptive RTG scheduler if enabled
        if ADAPTIVE_RTG_ENABLED:
            rtg_scheduler = get_rtg_scheduler()
            # Update with recent realized returns (would come from signal or DB)
            # For now, use a default update
            recent_return = signal.get("recent_return", 0.0)
            if recent_return != 0.0:
                rtg_scheduler.update(recent_return)
            
            # Get adaptive target price
            adaptive_target_price = rtg_scheduler.get_rtg(price, time_horizon_days=1.0)
            # Convert to return percentage
            adaptive_target_return = (adaptive_target_price - price) / price * 100.0
            target_return_pct = adaptive_target_return
            logger.debug(f"Adaptive RTG: price={price:.2f}, target_price={adaptive_target_price:.2f}, "
                        f"target_return={target_return_pct:.2f}%, vol={rtg_scheduler.vol_estimate:.4f}")

        # Build context from recent decisions
        recent_decisions = _get_recent_decisions()
        states, actions, rtg, timesteps, mask = _build_context_sequence(
            recent_decisions,
            target_return_pct / TARGET_RETURN_SCALE,
            _DT_CONFIG.regimes,
            _DT_CONFIG.brains,
            _DT_CONFIG.actions,
        )

        # Override the *last* return-to-go with our target (what we want to achieve)
        rtg[0, -1, 0] = target_return_pct / TARGET_RETURN_SCALE

        # Build current state (what we're deciding on now)
        regime = signal.get("regime", "default")
        features = signal.get("features", {})
        brain_votes = {v.name: v.action for v in signal.get("votes", [])}
        current_state = build_state_vector(regime, features, brain_votes, _DT_CONFIG.regimes, _DT_CONFIG.brains)

        # Replace last state with current
        states[0, -1, :] = torch.tensor(current_state, dtype=torch.float32, device=states.device)

        # Inference
        with torch.no_grad():
            preds = model.get_action(states, actions, rtg, timesteps, mask)

        # ── CVaR-Aware Decoding (Quantile Regression) ──
        # Use tau=0.05 (worst-case) for action selection (CVaR optimization)
        # Use tau=0.50 (median) for size multiplier and confidence threshold
        
        q_action_logits = preds["q_action_logits"][0].cpu()       # [5, 5] - [quantiles, actions]
        q_size_mult = preds["q_size_mult"][0].cpu()               # [5]
        q_conf_threshold = preds["q_conf_threshold"][0].cpu()     # [5]
        
        # CVaR action: use tau=0.05 (worst-case) logits
        cvar_action_logits = q_action_logits[CVAR_QUANTILE_IDX]  # [5] - tau=0.05
        action_idx = int(torch.argmax(cvar_action_logits).item())
        action = _DT_CONFIG.actions[action_idx] if action_idx < len(_DT_CONFIG.actions) else "stand_aside"
        
        # Confidence from CVaR logits (softmax probability of chosen action)
        cvar_probs = F.softmax(cvar_action_logits, dim=-1)
        confidence = float(cvar_probs[action_idx].item())
        
        # Size multiplier: use median quantile (tau=0.50) for stability
        size_mult = float(torch.sigmoid(q_size_mult[MEDIAN_QUANTILE_IDX]).item())
        size_mult = 0.5 + size_mult * 1.25  # scale to [0.5, 1.75]
        
        # Confidence threshold: use median quantile
        conf_thresh = float(torch.sigmoid(q_conf_threshold[MEDIAN_QUANTILE_IDX]).item())
        conf_thresh = 0.15 + conf_thresh * 0.20  # scale to [0.15, 0.35]
        
        # ── Temporal Action EMA Smoothing ──
        # action_t = alpha * action_DT + (1-alpha) * action_{t-1}
        # Store previous action in signal for next iteration
        prev_action = signal.get("_prev_dt_action")
        prev_confidence = signal.get("_prev_dt_confidence")
        
        if prev_action is not None and prev_confidence is not None:
            # EMA on confidence (actions are discrete, smooth confidence instead)
            confidence = TEMPORAL_EMA_ALPHA * confidence + (1 - TEMPORAL_EMA_ALPHA) * prev_confidence
            # For action, we keep current DT action but could use prev_action if confidence drops
            # Simple approach: if confidence drops significantly, prefer previous action
            if confidence < 0.3 and prev_confidence > confidence:
                action = prev_action
                confidence = prev_confidence
        
        # Store current for next iteration
        signal["_prev_dt_action"] = action
        signal["_prev_dt_confidence"] = confidence

        # Standard brain_weights from standard head (for committee compatibility)
        brain_weights, _, _, _ = decode_action_vector(
            preds["brain_weights"][0].cpu().numpy(),
            preds["size_mult"][0].cpu().numpy(),
            preds["conf_threshold"][0].cpu().numpy(),
            preds["action_logits"][0].cpu().numpy(),
            _DT_CONFIG.brains,
            _DT_CONFIG.actions,
        )

        # Map action to committee format
        if action in ["hold", "skip"]:
            action = "stand_aside"

        # Build CommitteeResult
        from .adaptive_meta import BrainScore
        scores = []
        action_scores = {}
        active_weight = 0.0
        for brain, weight in brain_weights.items():
            # Get the brain's actual vote
            vote_action = brain_votes.get(brain, "hold")
            vote_conf = 0.5
            for v in signal.get("votes", []):
                if v.name == brain:
                    vote_conf = v.confidence
                    break
            scores.append(BrainScore(name=brain, action=vote_action, confidence=vote_conf, weight=weight))
            if vote_action in ["buy", "sell"]:
                action_scores[vote_action] = action_scores.get(vote_action, 0.0) + vote_conf * weight
                active_weight += weight

        if active_weight > 0:
            for act in action_scores:
                action_scores[act] /= active_weight

        final_action = action
        confidence = confidence  # Already CVaR-smoothed

        # Threshold check using median quantile threshold
        if confidence < conf_thresh:
            final_action = "stand_aside"
            confidence = 0.0
            size_mult = 0.0

        return CommitteeResult(
            action=final_action,
            score=confidence,
            size_multiplier=size_mult,
            entropy=0.0,  # DT doesn't output entropy
            votes=[],  # Will be filled by caller
            active_weights=brain_weights,
            decision_id=None,
            adaptive_used=True,
            adaptive_weights=brain_weights,
            explanation=f"QR-DT[{regime}] CVaR(action={action}, c={confidence:.3f}) | sz={size_mult:.2f}x | thresh={conf_thresh:.2f}",
        )

    except Exception as e:
        logger.error(f"Decision Transformer inference failed: {e}")
        return None


# ── Training Data Preparation ────────────────────────────────────────

def prepare_training_data(
    limit: int = 10000,
    min_seq_len: int = 5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Prepare training sequences from closed decision snapshots.

    Returns: (states_list, actions_list, returns_to_go_list)
    Each element is a sequence of length <= CONTEXT_LENGTH.
    """
    from src.db import get_closed_decision_snapshots

    decisions = get_closed_decision_snapshots(limit=limit)
    if len(decisions) < min_seq_len:
        logger.warning(f"Insufficient decisions for training: {len(decisions)}")
        return [], [], []

    # Group by symbol for symbol-level sequences (optional)
    # For now, treat all decisions as one long sequence per regime
    regime_groups = {}
    for dec in decisions:
        regime = dec.get("regime", "default")
        regime_groups.setdefault(regime, []).append(dec)

    all_states = []
    all_actions = []
    all_rtg = []

    for regime, decs in regime_groups.items():
        if len(decs) < min_seq_len:
            continue

        # Sort by time ascending (oldest first for sequence)
        decs = sorted(decs, key=lambda d: d.get("entry_time", ""))

        # Build sequence for this regime
        states = []
        actions_list = []
        rtg_list = []

        for dec in decs:
            regime = dec.get("regime", "default")
            features = dec.get("features", {})
            brain_votes = dec.get("brain_votes", {})

            state = build_state_vector(regime, features, brain_votes, REGIMES, BRAINS)
            states.append(state)

            # Action from this decision
            action_vec = build_action_vector(
                brain_weights={b: 1.0/len(brain_votes) for b in brain_votes} if brain_votes else {},
                size_mult=1.0,
                conf_threshold=0.15,
                action=dec.get("final_action", "hold"),
                brains=BRAINS,
                actions=ACTIONS,
            )
            actions_list.append(action_vec)

            # Realized return (scaled)
            rtn = dec.get("realized_pnl", 0.0) / TARGET_RETURN_SCALE
            rtg_list.append(rtn)

        # Convert to arrays
        if len(states) >= min_seq_len:
            all_states.append(np.stack(states))
            all_actions.append(np.stack(actions_list))
            all_rtg.append(np.array(rtg_list, dtype=np.float32))

    logger.info(f"Prepared {len(all_states)} training sequences from {len(decisions)} decisions")
    return all_states, all_actions, all_rtg


# ── Training Loop ────────────────────────────────────────────────────

def train_decision_transformer(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    target_return_scale: float = TARGET_RETURN_SCALE,
) -> dict[str, Any]:
    """Train Decision Transformer on historical decision snapshots."""
    device = _get_device()

    # Prepare data
    states_list, actions_list, rtg_list = prepare_training_data()
    if not states_list:
        return {"error": "No training data available"}

    # Concatenate all sequences
    # For simplicity, we'll sample random windows during training
    # In practice, use a proper Dataset/DataLoader
    max_seq_len = CONTEXT_LENGTH

    # Compute state_dim and act_dim from data
    state_dim = states_list[0].shape[1]
    act_dim = actions_list[0].shape[1]

    # Update global config
    global _DT_CONFIG
    _DT_CONFIG = DTConfig(
        state_dim=state_dim,
        act_dim=act_dim,
        context_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        brains=BRAINS,
        regimes=REGIMES,
        actions=ACTIONS,
    )

    # Create model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=CONTEXT_LENGTH,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_losses = []

        # Sample batch of sequences
        for _ in range(batch_size):
            # Randomly pick a regime sequence
            seq_idx = np.random.randint(len(states_list))
            states_seq = states_list[seq_idx]
            actions_seq = actions_list[seq_idx]
            rtg_seq = rtg_list[seq_idx]

            # Random window
            max_start = max(1, len(states_seq) - max_seq_len)
            start = np.random.randint(0, max_start)
            end = min(start + max_seq_len, len(states_seq))
            seq_len = end - start

            if seq_len < 3:
                continue

            # Extract window
            s = states_seq[start:end]
            a = actions_seq[start:end]
            r = rtg_seq[start:end]

            # Pad to max_seq_len
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                s = np.pad(s, ((pad_len, 0), (0, 0)), mode='constant')
                a = np.pad(a, ((pad_len, 0), (0, 0)), mode='constant')
                r = np.pad(r, (pad_len, 0), mode='constant')

            # Tensors
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            a_t = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
            r_t = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            t_t = torch.arange(max_seq_len, device=device).unsqueeze(0)
            mask = torch.zeros(1, max_seq_len, dtype=torch.bool, device=device)
            mask[0, pad_len:] = True

            # Forward
            preds = model(s_t, a_t, r_t, t_t, mask)

            # Targets: next action for each timestep
            # We predict the action at timestep t from state at t
            # So targets are the actions at each timestep
            target_brain_weights = a_t[:, :, :len(BRAINS)]
            target_size_mult = a_t[:, :, len(BRAINS):len(BRAINS)+1]
            target_conf_thresh = a_t[:, :, len(BRAINS)+1:len(BRAINS)+2]
            target_action = torch.argmax(a_t[:, :, -len(ACTIONS):], dim=-1)

            # Standard losses (MSE for continuous, CE for action)
            loss_bw = F.mse_loss(preds["brain_weights"], target_brain_weights[:, -1, :])
            loss_sm = F.mse_loss(preds["size_mult"], target_size_mult[:, -1, :])
            loss_ct = F.mse_loss(preds["conf_threshold"], target_conf_thresh[:, -1, :])
            loss_act = F.cross_entropy(preds["action_logits"], target_action[:, -1])

            # ── Quantile Regression Loss (CVaR-aware) ──
            # Target return for quantile regression (use the realized return from the last timestep)
            # r_t[:, -1, 0] contains the target return for the last timestep
            _ = r_t[:, -1, :]  # [batch, 1] - realized return
            
            # Quantile action loss: pinball loss for each quantile
            q_action_logits = preds["q_action_logits"]  # [B, 5, 5]
            # We predict action logits per quantile; target is the action taken
            # For quantile regression on actions, we use the realized return as target
            # and compute pinball loss on the logits for the taken action
            target_action_onehot = F.one_hot(target_action[:, -1], num_classes=len(ACTIONS)).float()  # [B, 5]
            # Get logits for the taken action per quantile
            _q_action_taken = torch.einsum('bqa,bq->bqa', preds["q_action_logits"], target_action_onehot).sum(dim=-1)  # [B, 5]
            # Target return per quantile (same target, different quantiles)
            target_return_expanded = r_t[:, -1, :].expand(-1, NUM_QUANTILES)  # [B, 5]
            loss_q_action = quantile_huber_loss(
                q_action_logits.mean(dim=-1),  # [B, 5] - average over actions as proxy
                target_return_expanded,
                QUANTILES.to(device)
            )
            
            # Quantile size loss
            _q_size = preds["q_size_mult"]  # [B, 5]
            target_size = target_size_mult[:, -1, :].expand(-1, NUM_QUANTILES)  # [B, 5]
            loss_q_size = quantile_huber_loss(
                preds["q_size_mult"],
                target_size,
                QUANTILES.to(device)
            )
            
            # Quantile conf threshold loss
            _q_conf = preds["q_conf_threshold"]  # [B, 5]
            target_conf = target_conf_thresh[:, -1, :].expand(-1, NUM_QUANTILES)  # [B, 5]
            loss_q_conf = quantile_huber_loss(
                preds["q_conf_threshold"],
                target_conf,
                QUANTILES.to(device)
            )

            loss = loss_bw + loss_sm + loss_ct + loss_act
            # Add quantile losses (weighted)
            loss = loss + 0.5 * (loss_q_action + loss_q_size + loss_q_conf)

            # CQL regularization for OOD robustness
            if USE_CQL:
                # Generate OOD samples by adding noise to states
                ood_states = s_t + torch.randn_like(s_t) * 0.1
                ood_actions = a_t + torch.randn_like(a_t) * 0.1
                cql_loss = compute_cql_loss(model, ood_states, ood_actions, weight=CQL_WEIGHT)
                loss = loss + cql_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        losses.append(avg_loss)
        scheduler.step()

        if epoch % 10 == 0:
            logger.info(f"DT Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), DT_MODEL_PATH)
    _DT_CONFIG.save(DT_CONFIG_PATH)

    # Update global cache
    global _DT_MODEL
    _DT_MODEL = model.eval()

    return {
        "epochs": epochs,
        "final_loss": losses[-1] if losses else 0.0,
        "loss_history": losses,
        "config": _DT_CONFIG.to_dict(),
    }


if __name__ == "__main__":
    # Quick test
    result = train_decision_transformer(epochs=5, batch_size=8)
    print(json.dumps(result, indent=2))