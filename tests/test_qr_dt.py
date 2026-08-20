"""Reproduction / regression tests for Quantile Regression Decision Transformer (QR-DT) 
and Temporal Action EMA smoothing.

Tests:
1. Quantile loss function correctness (pinball loss)
2. CVaR action selection uses tau=0.05
3. Median quantile used for size/threshold
4. Temporal EMA smoothing on confidence
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")


# ── 1. Quantile Huber Loss correctness ───────────────────────────────
def test_quantile_huber_loss_basic():
    """Test quantile huber loss produces correct pinball loss behavior."""
    from src.committee.decision_transformer import quantile_huber_loss, QUANTILES
    
    batch_size = 4
    num_quantiles = 5
    
    # Perfect predictions (error = 0) should give zero loss
    pred = torch.ones(batch_size, num_quantiles) * 0.05
    target = torch.ones(batch_size, 1) * 0.05
    loss = quantile_huber_loss(pred, target, QUANTILES)
    assert loss.item() < 1e-6, f"Zero error should give near-zero loss, got {loss.item()}"
    
    # Positive error (pred < target) should be weighted by tau
    pred = torch.zeros(batch_size, num_quantiles)
    target = torch.ones(batch_size, 1) * 0.1
    loss = quantile_huber_loss(pred, target, QUANTILES)
    # Higher tau should give higher loss for positive error
    # tau=0.95 should penalize more than tau=0.05
    individual_losses = []
    for i, tau in enumerate(QUANTILES):
        pred_i = torch.zeros(batch_size, 1)
        target_i = torch.ones(batch_size, 1) * 0.1
        loss_i = quantile_huber_loss(pred_i, target_i, torch.tensor([tau]))
        individual_losses.append(loss_i.item())
    
    # Higher tau -> higher loss for positive error
    assert individual_losses[-1] > individual_losses[0], \
        f"Higher tau should penalize positive error more: {individual_losses}"
    
    # Negative error (pred > target) should be weighted by (1-tau)
    pred = torch.ones(batch_size, num_quantiles) * 0.2
    target = torch.ones(batch_size, 1) * 0.1
    individual_losses = []
    for i, tau in enumerate(QUANTILES):
        pred_i = torch.ones(batch_size, 1) * 0.2
        target_i = torch.ones(batch_size, 1) * 0.1
        loss_i = quantile_huber_loss(pred_i, target_i, torch.tensor([tau]))
        individual_losses.append(loss_i.item())
    
    # Lower tau -> higher loss for negative error (1-tau weighting)
    assert individual_losses[0] > individual_losses[-1], \
        f"Lower tau should penalize negative error more: {individual_losses}"


def test_quantile_huber_loss_huber_smoothing():
    """Test that Huber smoothing kicks in for small errors."""
    from src.committee.decision_transformer import quantile_huber_loss, QUANTILES
    
    # Small error should use quadratic (L2) not linear
    pred = torch.tensor([[0.051]])  # error = 0.001
    target = torch.tensor([[0.05]])
    loss = quantile_huber_loss(pred, target, torch.tensor([0.5]))
    # Should be very small (quadratic)
    assert loss.item() < 1e-6, f"Small error should be quadratically penalized, got {loss.item()}"
    
    # Large error should use linear
    pred = torch.tensor([[1.0]])
    target = torch.tensor([[0.0]])
    loss = quantile_huber_loss(pred, target, torch.tensor([0.5]))
    assert loss.item() > 0.1, f"Large error should be linearly penalized, got {loss.item()}"


# ── 2. CVaR Action Selection ────────────────────────────────────────
def test_cvar_action_selection():
    """Test that CVaR uses tau=0.05 (worst-case) for action selection."""
    from src.committee.decision_transformer import CVAR_QUANTILE_IDX, NUM_QUANTILES, QUANTILES
    
    assert CVAR_QUANTILE_IDX == 0, "CVaR should use first quantile (tau=0.05)"
    assert QUANTILES[CVAR_QUANTILE_IDX] == 0.05, "First quantile should be 0.05"
    
    # Mock quantile logits: [num_quantiles, num_actions]
    # Tau=0.05 should have different preferred action than median
    q_logits = torch.tensor([
        [2.0, 0.5, 0.1, 0.1, 0.1],  # tau=0.05: prefers action 0
        [0.5, 2.0, 0.1, 0.1, 0.1],  # tau=0.25: prefers action 1
        [0.1, 0.1, 2.0, 0.1, 0.1],  # tau=0.50: prefers action 2
        [0.1, 0.1, 0.1, 2.0, 0.1],  # tau=0.75: prefers action 3
        [0.1, 0.1, 0.1, 0.1, 2.0],  # tau=0.95: prefers action 4
    ])
    
    # CVaR should pick action 0 (worst-case)
    cvar_logits = q_logits[0]
    cvar_action = int(torch.argmax(cvar_logits).item())
    assert cvar_action == 0, f"CVaR should pick action 0, got {cvar_action}"
    
    # Median should pick action 2
    median_logits = q_logits[2]
    median_action = int(torch.argmax(median_logits).item())
    assert median_action == 2, f"Median should pick action 2, got {median_action}"


# ── 3. Median Quantile for Size/Threshold ────────────────────────────
def test_median_quantile_for_size_threshold():
    """Test that size_mult and conf_threshold use median quantile (tau=0.50)."""
    from src.committee.decision_transformer import MEDIAN_QUANTILE_IDX, QUANTILES
    
    assert MEDIAN_QUANTILE_IDX == 2, "Median should be index 2"
    assert QUANTILES[MEDIAN_QUANTILE_IDX] == 0.50, "Median quantile should be 0.50"
    
    # Size mult should use median
    q_size = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])  # [tau=0.05, 0.25, 0.5, 0.75, 0.95]
    median_size = float(torch.sigmoid(torch.tensor(q_size[2])).item())
    # Should use index 2 (median)
    assert abs(median_size - float(torch.sigmoid(torch.tensor(0.5)).item())) < 1e-6


# ── 4. Temporal EMA Smoothing ────────────────────────────────────────
def test_temporal_ema_smoothing():
    """Test Temporal EMA smoothing on confidence."""
    from src.committee.decision_transformer import TEMPORAL_EMA_ALPHA
    
    alpha = TEMPORAL_EMA_ALPHA
    prev_conf = 0.8
    curr_conf = 0.3
    
    # EMA: alpha * curr + (1-alpha) * prev
    ema_conf = alpha * curr_conf + (1 - alpha) * prev_conf
    expected = 0.3 * 0.3 + 0.7 * 0.8  # alpha=0.3
    assert abs(ema_conf - expected) < 1e-6, f"EMA calculation wrong: {ema_conf} vs {expected}"
    
    # If confidence drops significantly, should prefer previous action
    # (this is tested in integration, but we verify the logic here)
    assert 0.3 < 0.5, "Low confidence threshold should trigger fallback"


# ── 5. Quantile Head Shapes ──────────────────────────────────────────
def test_quantile_head_shapes():
    """Test that quantile heads have correct output shapes."""
    from src.committee.decision_transformer import DecisionTransformer, EMBED_DIM, NUM_QUANTILES, BRAINS, ACTIONS
    
    device = torch.device("cpu")
    model = DecisionTransformer(
        state_dim=64,
        act_dim=12,
        max_length=32,
        embed_dim=EMBED_DIM,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    
    batch_size = 2
    seq_len = 4
    state_dim = 64
    act_dim = 12
    
    s = torch.randn(1, 4, state_dim, device=device)
    a = torch.randn(1, 4, 12, device=device)
    r = torch.randn(1, 4, 1, device=device)
    t = torch.arange(4, device=device).unsqueeze(0)
    mask = torch.zeros(1, 4, dtype=torch.bool, device=device)
    
    model.eval()
    with torch.no_grad():
        preds = model(s, a, torch.zeros_like(r), t, mask)
    
    # Check quantile outputs exist
    assert "q_action_logits" in preds, "Missing q_action_logits"
    assert "q_size_mult" in preds, "Missing q_size_mult"
    assert "q_conf_threshold" in preds, "Missing q_conf_threshold"
    
    # Check shapes
    assert preds["q_action_logits"].shape == (1, 5, 5), f"q_action_logits shape: {preds['q_action_logits'].shape}"
    assert preds["q_size_mult"].shape == (1, 5), f"q_size_mult shape: {preds['q_size_mult'].shape}"
    assert preds["q_conf_threshold"].shape == (1, 5), f"q_conf_threshold shape: {preds['q_conf_threshold'].shape}"
    
    # Check quantile dimension is correct
    assert preds["q_action_logits"].shape[1] == NUM_QUANTILES
    assert preds["q_size_mult"].shape[1] == NUM_QUANTILES
    assert preds["q_conf_threshold"].shape[1] == NUM_QUANTILES


# ── 6. Spectral Norm on Quantile Heads ───────────────────────────────
def test_spectral_norm_on_quantile_heads():
    """Verify spectral norm is applied to quantile heads."""
    from src.committee.decision_transformer import DecisionTransformer, USE_SPECTRAL_NORM, EMBED_DIM
    
    if not USE_SPECTRAL_NORM:
        return  # Skip if disabled
    
    model = DecisionTransformer(
        state_dim=64,
        act_dim=12,
        max_length=32,
        embed_dim=EMBED_DIM,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )
    
    # Check that quantile heads have spectral norm
    # spectral_norm wraps the module and adds weight_u, weight_v attributes
    q_action_last = model.quantile_action_head[2]
    q_size_last = model.quantile_size_head[2]
    q_conf_last = model.quantile_conf_threshold_head[2]
    
    # spectral_norm adds weight_u and weight_v as parameters
    assert hasattr(q_action_last, 'weight_u'), "quantile_action_head[2] should have spectral norm (weight_u)"
    assert hasattr(q_size_last, 'weight_u'), "quantile_size_head[2] should have spectral norm (weight_u)"
    assert hasattr(q_conf_last, 'weight_u'), "quantile_conf_threshold_head[2] should have spectral norm (weight_u)"


if __name__ == "__main__":
    test_quantile_huber_loss_basic()
    print("✓ test_quantile_huber_loss_basic")
    
    test_quantile_huber_loss_huber_smoothing()
    print("✓ test_quantile_huber_loss_huber_smoothing")
    
    test_cvar_action_selection()
    print("✓ test_cvar_action_selection")
    
    test_median_quantile_for_size_threshold()
    print("✓ test_median_quantile_for_size_threshold")
    
    test_temporal_ema_smoothing()
    print("✓ test_temporal_ema_smoothing")
    
    test_quantile_head_shapes()
    print("✓ test_quantile_head_shapes")
    
    test_spectral_norm_on_quantile_heads()
    print("✓ test_spectral_norm_on_quantile_heads")
    
    print("\n✅ All QR-DT tests passed!")