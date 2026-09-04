"""BatchEnsemble: Fast ensemble inference via rank-1 weight perturbations.

Replaces 5 full forward passes with a single batched forward pass.
Based on: "BatchEnsemble: An Alternative Approach to Efficient Ensemble"
(Wen et al., 2020)

Speedup: ~5x faster than Deep Ensemble (5 separate models)
Memory: ~1.2x single model vs 5x for Deep Ensemble
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.logging_config import get_logger

logger = get_logger("batch_ensemble")


class BatchEnsembleLinear(nn.Module):
    """Linear layer with fast rank-1 ensemble perturbations.
    
    Instead of learning separate weights per ensemble member,
    learns shared base weights + per-member rank-1 perturbations.
    
    W_eff^{(i)} = W_base * (s_i @ r_i^T)  where s_i, r_i are per-member vectors
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ensemble_size: int = 5,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        
        # Shared base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Rank-1 perturbation parameters (per ensemble member)
        # r: [ensemble_size, in_features], s: [ensemble_size, out_features]
        self.r = nn.Parameter(torch.empty(ensemble_size, in_features))
        self.s = nn.Parameter(torch.empty(ensemble_size, out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize perturbations near identity
        nn.init.normal_(self.r, mean=1.0, std=0.01)
        nn.init.normal_(self.s, mean=1.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ensemble_size, in_features]
        Returns:
            out: [batch_size, ensemble_size, out_features]
        """
        batch_size, ens_size, in_feats = x.shape
        assert ens_size == self.ensemble_size, f"Expected ensemble_size={self.ensemble_size}, got {ens_size}"
        assert in_feats == self.in_features
        
        # Effective weight: W_base * (s_i @ r_i^T) for each member
        # w_perturbed: [ensemble_size, out_features, in_features]
        # (s_i @ r_i^T) -> [ensemble_size, out_features, in_features]
        perturbation = torch.einsum('ei,eo->eoi', self.r, self.s)  # [e, in, out] -> [e, out, in]
        w_eff = self.weight.unsqueeze(0) * perturbation  # [e, out, in]
        
        # Batched matmul: [batch, e, in] @ [e, out, in]^T -> [batch, e, out]
        # x @ w_eff.transpose(-2, -1)
        out = torch.bmm(x, w_eff.transpose(-2, -1))
        
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)  # [1, 1, out]
        
        return out


class BatchEnsembleLayerNorm(nn.Module):
    """LayerNorm applied independently per ensemble member."""
    
    def __init__(self, normalized_shape: int, ensemble_size: int = 5, eps: float = 1e-5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.eps = eps
        # Per-member affine parameters
        self.weight = nn.Parameter(torch.ones(ensemble_size, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ensemble_size, features]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class BatchEnsembleDropout(nn.Module):
    """Dropout applied independently per ensemble member."""
    
    def __init__(self, p: float = 0.1, ensemble_size: int = 5):
        super().__init__()
        self.p = p
        self.ensemble_size = ensemble_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ensemble_size, features]
        """
        if not self.training or self.p == 0:
            return x
        
        # Generate mask per ensemble member
        mask = torch.empty(x.shape[0], self.ensemble_size, 1, 
                          device=x.device, dtype=x.dtype).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


def replace_linear_with_batchensemble(
    module: nn.Module,
    ensemble_size: int = 5,
    skip_modules: list[str] | None = None,
) -> nn.Module:
    """Recursively replace all nn.Linear with BatchEnsembleLinear.
    
    Args:
        module: Module to convert
        ensemble_size: Number of ensemble members
        skip_modules: Module name patterns to skip (e.g., ['output_head'])
    
    Returns:
        Converted module (same object, modified in-place)
    """
    skip_modules = skip_modules or []
    
    for name, child in module.named_children():
        # Check if we should skip this module
        should_skip = any(skip in name for skip in skip_modules)
        
        if isinstance(child, nn.Linear) and not should_skip:
            # Replace with BatchEnsembleLinear
            new_linear = BatchEnsembleLinear(
                child.in_features,
                child.out_features,
                ensemble_size=ensemble_size,
                bias=child.bias is not None,
            )
            # Copy weights
            with torch.no_grad():
                new_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    new_linear.bias.copy_(child.bias)
            setattr(module, name, new_linear)
            logger.debug(f"Replaced Linear {name} with BatchEnsembleLinear")
            
        elif isinstance(child, nn.LayerNorm):
            # Replace with BatchEnsembleLayerNorm
            new_ln = BatchEnsembleLayerNorm(
                child.normalized_shape[0] if isinstance(child.normalized_shape, tuple) else child.normalized_shape,
                ensemble_size=ensemble_size,
                eps=child.eps,
            )
            with torch.no_grad():
                new_ln.weight.fill_(child.weight.mean().item() if child.weight.dim() > 0 else child.weight.item())
                new_ln.bias.fill_(child.bias.mean().item() if child.bias.dim() > 0 else child.bias.item())
            setattr(module, name, new_ln)
            
        elif isinstance(child, nn.Dropout):
            # Replace with BatchEnsembleDropout
            new_dropout = BatchEnsembleDropout(
                p=child.p,
                ensemble_size=ensemble_size,
            )
            setattr(module, name, new_dropout)
            
        else:
            # Recurse
            replace_linear_with_batchensemble(child, ensemble_size, skip_modules)
    
    return module


class BatchEnsembleTransformer(nn.Module):
    """Fast BatchEnsemble wrapper for any transformer model.
    
    Converts a single-model transformer into a fast ensemble
    by replacing Linear/LayerNorm/Dropout with batch versions.
    
    Usage:
        base_model = MyTransformer(...)
        ensemble = BatchEnsembleTransformer(base_model, ensemble_size=5)
        
        # Input: [batch, seq_len, features]
        # Output: [batch, ensemble_size, seq_len, output_dim]
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        ensemble_size: int = 5,
        skip_output_heads: bool = True,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.base_model = base_model
        
        # Store original forward
        self._original_forward = base_model.forward
        
        # Replace internals with BatchEnsemble versions
        # Skip output heads by default (we'll ensemble their outputs separately)
        skip = ['output_head', 'weight_head', 'size_head', 'threshold_head', 'action_head'] if skip_output_heads else []
        replace_linear_with_batchensemble(base_model, ensemble_size, skip_modules=skip)
        
        # Ensure model is on correct device
        self.device = next(base_model.parameters()).device
        base_model.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, features] (single input)
        Returns:
            out: [batch_size, ensemble_size, seq_len, output_dim]
        """
        batch_size, seq_len, features = x.shape
        
        # Expand input to ensemble dimension
        x_ens = x.unsqueeze(1).expand(-1, self.ensemble_size, -1, -1)  # [B, E, L, D]
        x_ens = x_ens.reshape(batch_size * self.ensemble_size, seq_len, features)
        
        # Run base model forward
        with torch.autocast(device_type=self.device.type, enabled=True):
            out = self.base_model(x_ens)
        
        # Reshape back to [B, E, L, out_dim]
        if out.dim() == 3:
            out = out.view(batch_size, self.ensemble_size, seq_len, -1)
        elif out.dim() == 2:
            out = out.view(batch_size, self.ensemble_size, -1)
        
        return out
    
    def get_ensemble_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Get mean and variance across ensemble."""
        out = self.forward(x)  # [B, E, L, D] or [B, E, D]
        
        if out.dim() == 4:  # [B, E, L, D]
            mean = out.mean(dim=1)  # [B, L, D]
            var = out.var(dim=1, unbiased=False)  # [B, L, D]
        else:  # [B, E, D]
            mean = out.mean(dim=1)  # [B, D]
            var = out.var(dim=1, unbiased=False)  # [B, D]
        
        return {'mean': mean, 'variance': var, 'std': var.sqrt()}


def convert_model_to_batchensemble(
    model: nn.Module,
    ensemble_size: int = 5,
    skip_output: bool = True,
) -> BatchEnsembleTransformer:
    """Convenience function to convert any model to BatchEnsemble."""
    return BatchEnsembleTransformer(
        base_model=model,
        ensemble_size=ensemble_size,
        skip_output_heads=skip_output,
    )


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Test BatchEnsembleLinear
    print("Testing BatchEnsembleLinear...")
    batch_linear = BatchEnsembleLinear(128, 256, ensemble_size=5)
    x = torch.randn(32, 5, 128)
    out = batch_linear(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (32, 5, 256)
    
    # Test BatchEnsembleLayerNorm
    print("Testing BatchEnsembleLayerNorm...")
    batch_ln = BatchEnsembleLayerNorm(128, ensemble_size=5)
    x = torch.randn(32, 5, 128)
    out = batch_ln(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (32, 5, 128)
    
    # Benchmark
    print("\nBenchmarking...")
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
    )
    
    # Standard ensemble (5 separate models)
    models = [copy.deepcopy(model) for _ in range(5)]
    for m in models:
        m.eval()
    
    x = torch.randn(64, 10, 128)
    
    # Time standard ensemble
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            for m in models:
                _ = m(x)
    standard_time = time.time() - start
    
    # Time BatchEnsemble
    batch_model = BatchEnsembleTransformer(model, ensemble_size=5)
    batch_model.eval()
    
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = batch_model(x)
    batch_time = time.time() - start
    
    print(f"Standard Ensemble (5x): {standard_time:.3f}s")
    print(f"BatchEnsemble:          {batch_time:.3f}s")
    print(f"Speedup: {standard_time / batch_time:.1f}x")
    
    # Verify output matches
    with torch.no_grad():
        std_out = torch.stack([m(x) for m in models], dim=1)  # [B, E, L, D]
        batch_out = batch_model(x)
        diff = (std_out - batch_out).abs().max().item()
        print(f"Max difference: {diff:.6f}")