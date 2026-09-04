"""OOD Adversarial Discriminator — The 'Bullshit Detector'.

Detects when the current market state is structurally different from training data.
If triggered, overrides the Decision Transformer and forces safe mode: the
committee result is REPLACED with a full stand_aside veto (action=stand_aside,
size_multiplier=0.0) -- the trade is blocked outright, not merely resized.

Architecture: Lightweight MLP binary classifier.
Training: Historical states (Class 0) vs. Live states (Class 1).
Inference: Single forward pass (< 1ms). Triggers safe mode if P(Live) > threshold.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("ood_discriminator")

# Paths
OOD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ood')
OOD_MODEL_PATH = os.path.join(OOD_DIR, 'ood_discriminator.pth')
OOD_CONFIG_PATH = os.path.join(OOD_DIR, 'ood_discriminator_config.json')

# Default settings
OOD_STATE_DIM = getattr(settings, 'OOD_STATE_DIM', 64)
OOD_HIDDEN = getattr(settings, 'OOD_HIDDEN', 128)
OOD_THRESHOLD = getattr(settings, 'OOD_THRESHOLD', 0.75)
OOD_LR = getattr(settings, 'OOD_LR', 1e-3)
OOD_EPOCHS = getattr(settings, 'OOD_EPOCHS', 20)
OOD_RETRAIN_INTERVAL = getattr(settings, 'OOD_RETRAIN_INTERVAL', 3600)  # 1 hour


class OODDiscriminator(nn.Module):
    """Binary classifier that detects if current state is out-of-distribution.
    
    Outputs P(State is from Live Data | State).
    If > threshold, state is considered OOD (out-of-distribution).
    """
    
    def __init__(
        self,
        state_dim: int = OOD_STATE_DIM,
        hidden: int = OOD_HIDDEN,
        threshold: float = OOD_THRESHOLD,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.threshold = threshold
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )
        
        self._training_lock = threading.Lock()
        self._last_retrain = 0.0
        self._is_trained = False
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state: [batch_size, state_dim] or [state_dim]
        Returns:
            P(Live) in [0, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)
    
    def is_ood(self, state: torch.Tensor) -> tuple[bool, float]:
        """Check if state is out-of-distribution.
        
        Args:
            state: [state_dim] or [batch_size, state_dim]
        Returns:
            (is_ood: bool, ood_probability: float)
        """
        with torch.no_grad():
            prob = self.forward(state).item() if state.dim() == 1 else self.forward(state).mean().item()
        return prob > self.threshold, prob
    
    def train_on_data(
        self,
        historical_states: np.ndarray,
        live_states: np.ndarray,
        val_split: float = 0.2,
        epochs: int = OOD_EPOCHS,
        lr: float = OOD_LR,
    ) -> float:
        """Train discriminator on historical vs live states.
        
        Args:
            historical_states: [N, state_dim] from training data
            live_states: [M, state_dim] from recent live trading
            val_split: Validation split ratio
            epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Validation accuracy
        """
        with self._training_lock:
            # Prepare data
            X = np.vstack([historical_states, live_states]).astype(np.float32)
            y = np.hstack([
                np.zeros(len(historical_states), dtype=np.float32),
                np.ones(len(live_states), dtype=np.float32)
            ])
            
            # Shuffle
            idx = np.random.permutation(len(X))
            X, y = X[idx], y[idx]
            
            # Split
            split_idx = int(len(X) * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Convert to tensors
            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32)
            
            # Training
            self.train()
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
            _criterion = nn.BCELoss()
            
            best_acc = 0.0
            for epoch in range(epochs):
                # Training
                self.train()
                optimizer.zero_grad()
                preds = self.forward(X_train_t).squeeze()
                loss = nn.BCELoss()(preds, y_train_t)
                loss.backward()
                optimizer.step()
                
                # Validation
                self.eval()
                with torch.no_grad():
                    val_preds = self.forward(X_val_t).squeeze()
                    val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()
                    if val_acc > best_acc:
                        best_acc = val_acc
                
                if epoch % 5 == 0:
                    logger.debug(f"OOD Discriminator epoch {epoch}: loss={loss.item():.4f}, val_acc={val_acc:.3f}")
            
            self._is_trained = True
            self._last_retrain = time.time()
            logger.info(f"OOD Discriminator trained: best_val_acc={best_acc:.3f}")
            return best_acc
    
    def retrain_if_needed(
        self,
        historical_states: np.ndarray,
        live_states: np.ndarray,
        force: bool = False,
    ) -> float | None:
        """Retrain if enough time has passed or forced."""
        now = time.time()
        if force or (now - self._last_retrain) > OOD_RETRAIN_INTERVAL:
            if len(live_states) > 100:  # Need minimum live data
                return self.train_on_data(historical_states, live_states)
        return None
    
    def save(self, path: str = OOD_MODEL_PATH) -> None:
        """Save model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'threshold': self.threshold,
            'is_trained': self._is_trained,
            'last_retrain': self._last_retrain,
        }, path)
        logger.info(f"OOD Discriminator saved to {path}")
    
    def load(self, path: str = OOD_MODEL_PATH) -> bool:
        """Load model state."""
        if not os.path.exists(path):
            return False
        try:
            state = torch.load(path, map_location='cpu')
            self.load_state_dict(state['state_dict'])
            self.state_dim = state.get('state_dim', self.state_dim)
            self.threshold = state.get('threshold', self.threshold)
            self._is_trained = state.get('is_trained', False)
            self._last_retrain = state.get('last_retrain', 0.0)
            self.eval()
            logger.info(f"OOD Discriminator loaded from {path}")
            return True
        except Exception as e:
            logger.warning(f"OOD Discriminator load failed: {e}")
            return False


# Global instance
_ood_discriminator: OODDiscriminator | None = None


def get_ood_discriminator() -> OODDiscriminator:
    """Get or create global OOD discriminator instance."""
    global _ood_discriminator
    if _ood_discriminator is None:
        _ood_discriminator = OODDiscriminator()
        _ood_discriminator.load()
    return _ood_discriminator


def reset_ood_discriminator() -> None:
    global _ood_discriminator
    _ood_discriminator = None


# ── Temporal Action EMA Smoothing (for integration) ──────────────────

class TemporalEMASmoother:
    """Exponential Moving Average smoother for action signals."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.prev_action: str | None = None
        self.prev_confidence: float = 0.0
        self.prev_size_mult: float = 1.0
    
    def smooth(
        self,
        action: str,
        confidence: float,
        size_mult: float,
    ) -> tuple[str, float, float]:
        """Apply EMA smoothing."""
        if self.prev_confidence == 0.0:
            # First call
            self.prev_action = action
            self.prev_confidence = confidence
            self.prev_size_mult = size_mult
            return action, confidence, size_mult
        
        # EMA on confidence and size
        smoothed_conf = self.alpha * confidence + (1 - self.alpha) * self.prev_confidence
        smoothed_size = self.alpha * size_mult + (1 - self.alpha) * self.prev_size_mult
        
        # If confidence drops significantly, prefer previous action
        smoothed_action = action
        if smoothed_conf < 0.3 and self.prev_confidence > smoothed_conf:
            smoothed_action = self.prev_action
            smoothed_conf = self.prev_confidence
        
        self.prev_action = smoothed_action
        self.prev_confidence = smoothed_conf
        self.prev_size_mult = smoothed_size
        
        return smoothed_action, smoothed_conf, smoothed_size
    
    def reset(self) -> None:
        self.prev_action = None
        self.prev_confidence = 0.0
        self.prev_size_mult = 1.0


# ── Integration Helper ───────────────────────────────────────────────

def build_ood_state_vector(
    regime: str,
    features: dict[str, Any],
    brain_votes: dict[str, str],
    regimes_list: list[str],
    brains_list: list[str],
) -> np.ndarray:
    """Build state vector for OOD discriminator (matches DecisionTransformer input)."""
    from .decision_transformer import build_state_vector
    state = build_state_vector(regime, features, brain_votes, regimes_list, brains_list)
    return state


def check_ood_and_override(
    symbol: str,
    price: float,
    signal: dict[str, Any],
    committee_result: Any,
    ood_discriminator: OODDiscriminator,
) -> Any:
    """Check OOD and override committee result if OOD detected.
    
    Args:
        symbol: Trading symbol
        price: Current price
        signal: Signal dict with regime, features, votes
        committee_result: CommitteeResult from DT
        ood_discriminator: OOD discriminator instance
        
    Returns:
        Modified committee_result (or original if not OOD)
    """
    try:
        # Build state vector
        regime = signal.get("regime", "default")
        features = signal.get("features", {})
        brain_votes = {v.name: v.action for v in signal.get("votes", [])}
        
        from .decision_transformer import BRAINS, REGIMES
        state_vec = build_ood_state_vector(regime, features, brain_votes, REGIMES, BRAINS)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)
        
        is_ood, ood_prob = ood_discriminator.is_ood(state_tensor)
        
        if is_ood:
            logger.warning(
                f"OOD DETECTED for {symbol}: prob={ood_prob:.3f} > {ood_discriminator.threshold}. "
                f"Overriding to SAFE MODE (full stand_aside veto)."
            )
            
            # Create safe override: full veto (not a size reduction -- the
            # comment in the class docstring saying "90% size reduction" is
            # stale; this result blocks the trade outright).
            from .models import CommitteeResult
            safe_result = CommitteeResult(
                action="stand_aside",  # Force no trade
                score=0.0,
                size_multiplier=0.0,  # full veto: no trade
                entropy=1.0,
                votes=[],
                active_weights={},
                decision_id=None,
                adaptive_used=False,
                adaptive_weights={},
                explanation=f"OOD OVERRIDE: prob={ood_prob:.3f} > threshold. Safe mode activated.",
            )
            return safe_result
        
    except Exception as e:
        logger.error(f"OOD check failed for {symbol}: {e}")
    
    return committee_result


if __name__ == "__main__":
    # Quick test
    import torch
    
    disc = OODDiscriminator(state_dim=64)
    
    # Generate dummy data
    hist = np.random.randn(1000, 64).astype(np.float32)
    live = np.random.randn(200, 64).astype(np.float32) + 0.5  # Shifted = OOD
    
    acc = disc.train_on_data(hist, live)
    print(f"Training accuracy: {acc:.3f}")
    
    # Test OOD detection
    test_state = torch.randn(64) + 0.5  # OOD
    is_ood, prob = disc.is_ood(test_state)
    print(f"OOD test (shifted): is_ood={is_ood}, prob={prob:.3f}")
    
    normal_state = torch.randn(64)  # In-distribution
    is_ood, prob = disc.is_ood(normal_state)
    print(f"OOD test (normal): is_ood={is_ood}, prob={prob:.3f}")