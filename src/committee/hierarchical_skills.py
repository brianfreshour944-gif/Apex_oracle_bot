"""Hierarchical Skills (Options) Layer for Decision Transformer.

Implements the Options-Critic architecture for unsupervised skill discovery.
The QR-DT outputs skill embeddings (high-level intent), and a lightweight
LSTM executor handles low-level position adjustments over multiple steps.

Architecture:
- SkillEncoder: Maps state -> skill distribution (categorical over K skills)
- SkillLSTM: 2-layer LSTM that executes skill over H steps
- TerminationHead: Predicts when to switch skills (beta)
- Critic: Learns Q(s, skill) for intra-option policy improvement

Training: Options-Critic with entropy regularization for skill diversity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import settings
from src.logging_config import get_logger

logger = get_logger("hierarchical_skills")

# Configuration
NUM_SKILLS = getattr(settings, 'NUM_SKILLS', 4)  # e.g., trending, mean_revert, scalping, market_making
SKILL_EMBED_DIM = getattr(settings, 'SKILL_EMBED_DIM', 64)
EXECUTOR_HIDDEN = getattr(settings, 'EXECUTOR_HIDDEN', 128)
EXECUTOR_LAYERS = getattr(settings, 'EXECUTOR_LAYERS', 2)
SKILL_HORIZON = getattr(settings, 'SKILL_HORIZON', 10)  # steps per skill
ENTROPY_COEF = getattr(settings, 'SKILL_ENTROPY_COEF', 0.01)
TERMINATION_REG = getattr(settings, 'SKILL_TERMINATION_REG', 0.01)

# Paths
SKILLS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'skills')
SKILLS_MODEL_PATH = os.path.join(SKILLS_DIR, 'hierarchical_skills.pth')
SKILLS_CONFIG_PATH = os.path.join(SKILLS_DIR, 'hierarchical_skills_config.json')


@dataclass
class SkillConfig:
    num_skills: int = NUM_SKILLS
    skill_embed_dim: int = SKILL_EMBED_DIM
    executor_hidden: int = EXECUTOR_HIDDEN
    executor_layers: int = EXECUTOR_LAYERS
    skill_horizon: int = SKILL_HORIZON
    entropy_coef: float = ENTROPY_COEF
    termination_reg: float = TERMINATION_REG

    def to_dict(self) -> Dict[str, Any]:
        return {
            'num_skills': self.num_skills,
            'skill_embed_dim': self.skill_embed_dim,
            'executor_hidden': self.executor_hidden,
            'executor_layers': self.executor_layers,
            'skill_horizon': self.skill_horizon,
            'entropy_coef': self.entropy_coef,
            'termination_reg': self.termination_reg,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SkillConfig':
        return cls(**d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SkillConfig':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class SkillEncoder(nn.Module):
    """Encodes state into skill distribution (categorical over K skills).
    
    Instead of outputting raw actions, the DT now outputs a skill embedding
    that represents high-level intent (e.g., trending, mean_revert, scalping).
    """
    
    def __init__(
        self,
        state_dim: int,
        num_skills: int = NUM_SKILLS,
        skill_embed_dim: int = SKILL_EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_skills = num_skills
        self.skill_embed_dim = skill_embed_dim
        
        # Skill logits head
        self.skill_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_skills),
        )
        
        # Skill embeddings (learned per skill)
        self.skill_embeddings = nn.Parameter(
            torch.randn(num_skills, skill_embed_dim) * 0.02
        )
        
        # Skill-specific bias for action logits
        self.skill_action_bias = nn.Parameter(torch.zeros(num_skills))
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state: [batch, state_dim] or [batch, seq_len, state_dim]
        Returns:
            dict with skill_logits, skill_probs, skill_embeddings, entropy
        """
        if state.dim() == 3:
            batch, seq_len, _ = state.shape
            state_flat = state.reshape(-1, self.state_dim)
            skill_logits = self.skill_head(state_flat)  # [B*L, K]
            skill_logits = skill_logits.view(state.shape[0], state.shape[1], -1)
        else:
            skill_logits = self.skill_head(state)  # [B, K]
        
        skill_probs = F.softmax(skill_logits, dim=-1)
        skill_log_probs = F.log_softmax(skill_logits, dim=-1)
        
        # Entropy for regularization
        entropy = -(skill_probs * skill_log_probs).sum(dim=-1)  # [B] or [B, L]
        
        # Expected skill embedding (mixture)
        skill_embed = torch.einsum('...k,kd->...d', skill_probs, self.skill_embeddings)
        
        return {
            'skill_logits': skill_logits,
            'skill_probs': skill_probs,
            'skill_log_probs': skill_log_probs,
            'skill_embed': skill_embed,
            'entropy': entropy,
            'skill_action_bias': self.skill_action_bias,
        }
    
    def sample_skill(self, skill_logits: torch.Tensor) -> torch.Tensor:
        """Sample skill from categorical distribution."""
        dist = torch.distributions.Categorical(logits=skill_logits)
        return dist.sample()


class SkillLSTMExecutor(nn.Module):
    """Lightweight LSTM executor that executes a skill over multiple steps.
    
    Takes skill embedding + current state, outputs low-level action adjustments
    (position size delta, limit price offset, timing) for the next step.
    """
    
    def __init__(
        self,
        state_dim: int,
        skill_embed_dim: int = SKILL_EMBED_DIM,
        hidden_dim: int = EXECUTOR_HIDDEN,
        num_layers: int = EXECUTOR_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection: state + skill_embedding -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim + SKILL_EMBED_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # LSTM for temporal execution
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output heads for micro-adjustments
        self.size_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # position size multiplier delta
        )
        
        self.price_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # limit price offset in bps
        )
        
        self.timing_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # urgency score [0,1]
)
        
    def forward(
        self,
        state: torch.Tensor,
        skill_embed: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            state: [batch, seq_len, state_dim] or [batch, state_dim]
            skill_embed: [batch, seq_len, skill_embed_dim] or [batch, skill_embed_dim]
            hidden: Optional (h, c) tuple from previous step
        Returns:
            dict with size_delta, price_offset, urgency; new hidden state
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)
            skill_embed = skill_embed.unsqueeze(1)
            squeeze_seq = True
        else:
            squeeze_seq = False
        
        batch, seq_len, _ = state.shape
        
        # Concatenate state + skill embedding
        x = torch.cat([state, skill_embed], dim=-1)  # [B, L, state+skill]
        x = self.input_proj(x)  # [B, L, hidden]
        
        # LSTM forward
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=state.device)
            c0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=state.device)
            hidden = (h0, c0)
        
        lstm_out, new_hidden = self.lstm(x, hidden)  # [B, L, hidden], (h, c)
        
        # Output heads (use last step for single-step, all for sequence)
        if squeeze_seq:
            last_out = lstm_out[:, -1, :]  # [B, hidden]
        else:
            last_out = lstm_out  # [B, L, hidden]
        
        size_delta = self.size_delta_head(last_out)  # [B] or [B, L, 1]
        price_offset = self.price_offset_head(last_out)  # [B] or [B, L, 1]
        urgency = torch.sigmoid(self.timing_head(last_out))  # [B] or [B, L, 1]
        
        output = {
            'size_delta': size_delta.squeeze(-1),      # [B] or [B, L]
            'price_offset_bps': price_offset.squeeze(-1),  # [B] or [B, L]
            'urgency': urgency.squeeze(-1),            # [B] or [B, L]
        }
        
        if squeeze_seq:
            output = {k: v.squeeze(1) if v.dim() > 1 else v for k, v in output.items()}
        
        if squeeze_seq:
            output = {k: v.squeeze(1) if v.dim() > 1 else v for k, v in output.items()}
        
        # Update instance hidden state
        self._hidden_h, self._hidden_c = new_hidden
        
        return output, new_hidden
    
    def reset_hidden(self, batch_size: int, device: torch.device) -> None:
        """Reset hidden state for new episode."""
        self._hidden_h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        self._hidden_c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def get_hidden(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(self, '_hidden_h') and hasattr(self, '_hidden_c'):
            return (self._hidden_h, self._hidden_c)
        return None


class TerminationHead(nn.Module):
    """Predicts probability of skill termination (beta).
    
    Learns when to switch skills based on state + skill progress.
    """
    
    def __init__(
        self,
        state_dim: int,
        skill_embed_dim: int = SKILL_EMBED_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + skill_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, state: torch.Tensor, skill_embed: torch.Tensor) -> torch.Tensor:
        """Returns termination probability beta in [0, 1]."""
        x = torch.cat([state, skill_embed], dim=-1)
        return self.net(x)


class SkillCritic(nn.Module):
    """Critic for intra-option policy improvement: Q(s, skill).
    
    Learns the value of initiating each skill in a given state.
    Used for the Options-Critic gradient.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_skills: int = NUM_SKILLS,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_skills = num_skills
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_skills),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for each skill [batch, num_skills]."""
        return self.net(state)
    
    def get_q(self, state: torch.Tensor, skill_idx: torch.Tensor) -> torch.Tensor:
        """Get Q-value for specific skill indices."""
        q_values = self.forward(state)  # [B, K]
        return q_values.gather(1, skill_idx.unsqueeze(-1)).squeeze(-1)


class HierarchicalSkills(nn.Module):
    """Complete Hierarchical Skills module combining all components."""
    
    def __init__(
        self,
        state_dim: int,
        config: Optional[SkillConfig] = None,
    ):
        super().__init__()
        self.config = config or SkillConfig()
        self.state_dim = state_dim
        
        # Components
        self.skill_encoder = SkillEncoder(
            state_dim=state_dim,
            num_skills=self.config.num_skills,
            skill_embed_dim=self.config.skill_embed_dim,
        )
        
        self.executor = SkillLSTMExecutor(
            state_dim=state_dim,
            skill_embed_dim=self.config.skill_embed_dim,
            hidden_dim=self.config.executor_hidden,
            num_layers=self.config.executor_layers,
        )
        
        self.termination = TerminationHead(
            state_dim=state_dim,
            skill_embed_dim=self.config.skill_embed_dim,
        )
        
        self.critic = SkillCritic(
            state_dim=state_dim,
            num_skills=self.config.num_skills,
        )
        
        # Track execution state
        self._current_skill: Optional[torch.Tensor] = None
        self._skill_steps = 0
        self._executor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    
    def forward(
        self,
        state: torch.Tensor,
        skill_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for skill selection + execution.
        
        Args:
            state: [batch, state_dim] or [batch, seq_len, state_dim]
            skill_idx: Optional forced skill index [batch] or [batch, 1]
            
        Returns:
            dict with skill outputs, executor outputs, termination prob
        """
        # Skill encoding
        skill_out = self.skill_encoder(state)
        
        # Sample or use provided skill
        if skill_idx is None:
            skill_idx = self.skill_encoder.sample_skill(skill_out['skill_logits'])
        else:
            if skill_idx.dim() == 2:
                skill_idx = skill_idx.squeeze(-1)
        
        # Get skill embedding for selected skill
        batch_size = state.shape[0]
        skill_embed = self.skill_encoder.skill_embeddings[skill_idx]  # [B, skill_embed]
        
        # Termination probability
        if state.dim() == 3:
            # Sequence: use last step
            term_state = state[:, -1, :]
            term_embed = skill_embed.unsqueeze(1).expand(-1, state.shape[1], -1)[:, -1, :]
        else:
            term_state = state
            term_embed = skill_embed
        
        beta = self.termination(term_state, term_embed)  # [B, 1]
        
        # Execute skill
        exec_out, new_hidden = self.executor(state, skill_embed)
        
        return {
            'skill_idx': skill_idx,
            'skill_probs': skill_out['skill_probs'],
            'skill_embed': skill_embed,
            'skill_entropy': skill_out['entropy'],
            'skill_log_probs': skill_out['skill_log_probs'],
            'skill_action_bias': skill_out['skill_action_bias'],
            'size_delta': exec_out['size_delta'],
            'price_offset_bps': exec_out['price_offset_bps'],
            'urgency': exec_out['urgency'],
            'termination_prob': beta.squeeze(-1),
            'executor_hidden': new_hidden,
        }
    
    def reset_execution(self) -> None:
        """Reset executor state for new episode."""
        self.executor.reset_hidden(1, next(self.parameters()).device)
        self._current_skill = None
        self._skill_steps = 0
        self._executor_hidden = None
    
    def step(
        self,
        state: torch.Tensor,
        force_new_skill: bool = False,
    ) -> Dict[str, Any]:
        """Execute one step of hierarchical policy.
        
        Handles skill switching based on termination probability.
        """
        # Check if we should terminate current skill
        if self._current_skill is not None and not force_new_skill:
            with torch.no_grad():
                # Look up skill embedding for current skill index
                current_skill_embed = self.skill_encoder.skill_embeddings[self._current_skill]
                term_out = self.termination(state, current_skill_embed)
                should_terminate = term_out.squeeze(-1) > 0.5
                
                if should_terminate.any():
                    force_new_skill = True
                    logger.debug(f"Skill termination triggered: beta={term_out.squeeze().item():.3f}")
        
        if force_new_skill or self._current_skill is None:
            # Sample new skill
            with torch.no_grad():
                skill_out = self.skill_encoder(state)
                self._current_skill = self.skill_encoder.sample_skill(skill_out['skill_logits'])
                self._skill_steps = 0
                self.executor.reset_hidden(state.shape[0], state.device)
        
        # Execute current skill
        self._skill_steps += 1
        out = self.forward(state, self._current_skill)
        
        # Force termination if horizon reached
        if self._skill_steps >= self.config.skill_horizon:
            self._current_skill = None
            self._skill_steps = 0
        
        return out


# ── Options-Critic Training ──────────────────────────────────────────

class OptionsCriticTrainer:
    """Options-Critic training for hierarchical skills.
    
    Implements the intra-option policy gradient and termination gradient
    from Bacon, Harb, and Precup (2017).
    """
    
    def __init__(
        self,
        model: HierarchicalSkills,
        lr: float = 1e-4,
        entropy_coef: float = 0.01,
        termination_reg: float = 0.01,
        gamma: float = 0.99,
    ):
        self.model = model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.termination_reg = termination_reg
        
        # Optimizers
        self.actor_opt = torch.optim.Adam(
            list(model.skill_encoder.parameters()) + 
            list(model.executor.parameters()),
            lr=lr, weight_decay=1e-4
        )
        self.termination_opt = torch.optim.Adam(
            model.termination.parameters(),
            lr=lr, weight_decay=1e-4
        )
        self.critic_opt = torch.optim.Adam(
            model.critic.parameters(),
            lr=lr, weight_decay=1e-4
        )
    
    def compute_critic_loss(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gammas: torch.Tensor,
    ) -> torch.Tensor:
        """TD error for critic: Q(s, skill) -> r + gamma * max_a' Q(s', a')."""
        with torch.no_grad():
            next_q = self.model.critic(next_states)  # [B, K]
            next_q_max = next_q.max(dim=1)[0]  # [B]
            target = rewards + gammas * (1 - dones) * next_q_max
        
        q_values = self.model.critic(states)
        q_selected = q_values.gather(1, skills.unsqueeze(-1)).squeeze(-1)
        
        return F.mse_loss(q_values, target.unsqueeze(-1).expand_as(q_values))
    
    def compute_actor_loss(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """Policy gradient for skill selection + executor."""
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -self.entropy_coef * entropy.mean()
        return policy_loss + entropy_loss
    
    def compute_termination_loss(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        betas: torch.Tensor,  # Ground truth betas (for reference)
        q_values: torch.Tensor,
        v_values: torch.Tensor,
    ) -> torch.Tensor:
        """Termination gradient: beta should be high when Q < V.
        
        Uses the model's predicted termination probabilities for the gradient.
        """
        # Get predicted termination probabilities from the model
        pred_betas = self.model.termination(states, 
            self.model.skill_encoder.skill_embeddings[skills])
        
        # Termination advantage = Q(s, skill) - V(s)
        adv = q_values - v_values
        # Want beta high when adv < 0 (skill worse than random)
        # Gradient: beta * (Q - V) -> minimize this means increase beta when Q < V
        termination_loss = (pred_betas.squeeze(-1) * adv.detach()).mean()
        reg_loss = self.termination_reg * pred_betas.mean()  # encourage earlier termination
        return termination_loss + reg_loss
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single update step on a batch of transitions."""
        states = batch['states']
        skills = batch['skills']
        log_probs = batch['log_probs']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        gammas = batch['gammas']
        betas = batch['betas']
        entropies = batch['entropies']
        
        # Critic update
        self.critic_opt.zero_grad()
        critic_loss = self.compute_critic_loss(
            states, skills, rewards, next_states, dones, gammas
        )
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 1.0)
        self.critic_opt.step()
        
        # Actor update
        with torch.no_grad():
            q_vals = self.model.critic(states)
            q_selected = q_vals.gather(1, skills.unsqueeze(-1)).squeeze(-1)
            v_vals = q_vals.max(dim=1)[0]
            advantages = q_selected - v_vals
        
        # Store v_vals for termination loss
        v_vals_detached = v_vals.detach()
        
        self.actor_opt.zero_grad()
        actor_loss = self.compute_actor_loss(
            states, skills, log_probs, advantages, entropies
        )
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.skill_encoder.parameters()) + 
            list(self.model.executor.parameters()), 1.0
        )
        self.actor_opt.step()
        
        # Termination update
        self.termination_opt.zero_grad()
        term_loss = self.compute_termination_loss(
            states, skills, betas, q_selected.detach(), v_vals_detached
        )
        term_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.termination.parameters(), 1.0)
        self.termination_opt.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'term_loss': term_loss.item(),
        }


# ── Global Instance & Helpers ────────────────────────────────────────

_hierarchical_skills: Optional[HierarchicalSkills] = None
_skills_trainer: Optional[OptionsCriticTrainer] = None


def get_hierarchical_skills(state_dim: int) -> HierarchicalSkills:
    """Get or create global hierarchical skills instance."""
    global _hierarchical_skills
    if _hierarchical_skills is None:
        _hierarchical_skills = HierarchicalSkills(state_dim)
    return _hierarchical_skills


def get_skills_trainer(model: HierarchicalSkills) -> OptionsCriticTrainer:
    global _skills_trainer
    if _skills_trainer is None:
        _skills_trainer = OptionsCriticTrainer(model)
    return _skills_trainer


def reset_hierarchical_skills() -> None:
    global _hierarchical_skills, _skills_trainer
    _hierarchical_skills = None
    _skills_trainer = None


# ── Integration with Decision Transformer ────────────────────────────

async def run_hierarchical_skills(
    symbol: str,
    price: float,
    signal: Dict[str, Any],
    target_return_pct: float = 2.0,
) -> Optional[Any]:
    """Run hierarchical skills inference for a trading decision.
    
    Returns CommitteeResult compatible with committee system.
    """
    from src.committee.decision_transformer import get_decision_transformer, _DT_CONFIG
    from src.committee.adaptive_meta import BrainScore
    from src.committee.models import CommitteeResult
    
    model = get_hierarchical_skills(_DT_CONFIG.state_dim if _DT_CONFIG else 64)
    if model is None:
        return None
    
    try:
        # Build context (reuse DT context building)
        from src.committee.decision_transformer import _get_recent_decisions, _build_context_sequence
        
        recent_decisions = _get_recent_decisions()
        states, actions, rtg, timesteps, mask = _build_context_sequence(
            recent_decisions,
            target_return_pct / 100.0,
            _DT_CONFIG.regimes if _DT_CONFIG else ["default"],
            _DT_CONFIG.brains if _DT_CONFIG else ["transformer", "quant", "momentum", "sentinel", "llm"],
            _DT_CONFIG.actions if _DT_CONFIG else ["buy", "sell", "hold", "stand_aside", "skip"],
        )
        
        # Override RTG with target
        rtg[0, -1, 0] = target_return_pct / 100.0
        
        # Current state
        regime = signal.get("regime", "default")
        features = signal.get("features", {})
        brain_votes = {v.name: v.action for v in signal.get("votes", [])}
        from src.committee.decision_transformer import build_state_vector
        current_state = build_state_vector(regime, features, brain_votes, 
                                          _DT_CONFIG.regimes if _DT_CONFIG else ["default"],
                                          _DT_CONFIG.brains if _DT_CONFIG else [])
        
        states[0, -1, :] = torch.tensor(current_state, dtype=torch.float32, device=states.device)
        
        # Hierarchical inference
        model.eval()
        with torch.no_grad():
            out = model.step(states[:, -1:, :])  # Single step
        
        # Decode outputs
        action_idx = int(out['skill_idx'].item())
        action = ["buy", "sell", "hold", "stand_aside", "skip"][action_idx] if action_idx < 5 else "stand_aside"
        
        # CVaR confidence from skill probs
        skill_probs = out['skill_probs'][0].cpu().numpy()
        confidence = float(skill_probs[action_idx])
        
        # Size from executor
        size_mult = float(out['size_delta'].item())
        size_mult = 0.5 + size_mult * 1.25
        
        # Confidence threshold from median quantile
        conf_thresh = 0.15 + 0.20 * 0.5
        
        # Temporal EMA
        from src.committee.decision_transformer import TEMPORAL_EMA_ALPHA
        prev_conf = signal.get("_prev_dt_confidence", 0.0)
        if prev_conf > 0:
            confidence = 0.3 * confidence + 0.7 * prev_conf
        signal["_prev_dt_confidence"] = confidence
        signal["_prev_dt_action"] = action
        
        # Size multiplier from executor
        size_mult = 0.5 + float(torch.sigmoid(out['size_delta'])).item() * 1.25
        
        if action in ["hold", "skip"]:
            action = "stand_aside"
        
        if confidence < conf_thresh:
            action = "stand_aside"
            confidence = 0.0
            size_mult = 0.0
        
        # Build brain_weights for compatibility
        brain_weights = {b: skill_probs[i] if i < len(skill_probs) else 0.2 for i, b in enumerate(["transformer", "quant", "momentum", "sentinel", "llm"])}
        
        return CommitteeResult(
            action=action,
            score=confidence,
            size_multiplier=size_mult,
            entropy=0.0,
            votes=[],
            active_weights=brain_weights,
            decision_id=None,
            adaptive_used=True,
            adaptive_weights=brain_weights,
            explanation=f"HierSkills[{regime}] skill={action_idx} c={confidence:.3f} sz={size_mult:.2f}",
        )
        
    except Exception as e:
        logger.error(f"Hierarchical skills inference failed: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    print("Testing Hierarchical Skills...")
    
    device = torch.device("cpu")
    model = HierarchicalSkills(state_dim=64)
    model.eval()
    
    # Test forward
    state = torch.randn(2, 64)
    out = model(state)
    print(f"Skill probs shape: {out['skill_probs'].shape}")
    print(f"Skill idx: {out['skill_idx'].shape}")
    print(f"Executor out keys: {list(out.keys())}")
    
    # Test step
    model.reset_execution()
    for i in range(3):
        out = model.step(torch.randn(1, 64))
        print(f"Step {i}: skill={out['skill_idx'].item()}, term={out['termination_prob'].item():.3f}")
    
    print("Hierarchical Skills module loaded successfully!")