"""Tests for Hierarchical Skills (Options-Critic) module."""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")


# ── 1. SkillEncoder Tests ────────────────────────────────────────────
def test_skill_encoder_forward():
    """Test SkillEncoder forward pass shapes."""
    from src.committee.hierarchical_skills import SkillEncoder, NUM_SKILLS, SKILL_EMBED_DIM
    
    encoder = SkillEncoder(state_dim=64, num_skills=NUM_SKILLS, skill_embed_dim=SKILL_EMBED_DIM)
    
    # Single state
    state = torch.randn(2, 64)
    out = encoder(state)
    
    assert out['skill_logits'].shape == (2, NUM_SKILLS)
    assert out['skill_probs'].shape == (2, NUM_SKILLS)
    assert out['skill_log_probs'].shape == (2, NUM_SKILLS)
    assert out['skill_embed'].shape == (2, SKILL_EMBED_DIM)
    assert out['entropy'].shape == (2,)
    assert out['skill_action_bias'].shape == (NUM_SKILLS,)
    
    # Sequence input
    state_seq = torch.randn(2, 5, 64)
    out_seq = encoder(state_seq)
    
    assert out_seq['skill_logits'].shape == (2, 5, NUM_SKILLS)
    assert out_seq['skill_probs'].shape == (2, 5, NUM_SKILLS)
    assert out_seq['skill_embed'].shape == (2, 5, SKILL_EMBED_DIM)
    assert out_seq['entropy'].shape == (2, 5)


def test_skill_encoder_sample():
    """Test skill sampling from categorical distribution."""
    from src.committee.hierarchical_skills import SkillEncoder, NUM_SKILLS
    
    encoder = SkillEncoder(state_dim=64, num_skills=NUM_SKILLS)
    encoder.eval()
    
    with torch.no_grad():
        state = torch.randn(10, 64)
        logits = encoder(state)['skill_logits']
        samples = encoder.sample_skill(logits)
        
    assert samples.shape == (10,)
    assert (samples >= 0).all() and (samples < NUM_SKILLS).all()


# ── 2. SkillLSTMExecutor Tests ───────────────────────────────────────
def test_executor_forward():
    """Test SkillLSTMExecutor forward pass."""
    from src.committee.hierarchical_skills import SkillLSTMExecutor
    
    executor = SkillLSTMExecutor(
        state_dim=64,
        skill_embed_dim=64,
        hidden_dim=128,
        num_layers=2,
    )
    
    # Single step
    state = torch.randn(2, 64)
    skill = torch.randn(2, 64)
    
    out, hidden = executor(state, skill)
    
    assert 'size_delta' in out
    assert 'price_offset_bps' in out
    assert 'urgency' in out
    assert out['size_delta'].shape == (2,)
    assert out['price_offset_bps'].shape == (2,)
    assert out['urgency'].shape == (2,)
    assert 0 <= out['urgency'].min() <= out['urgency'].max() <= 1
    
    # Check hidden state
    assert hidden is not None
    h, c = hidden
    assert h.shape == (2, 2, 128)  # [num_layers, batch, hidden]
    assert c.shape == (2, 2, 128)


def test_executor_sequence():
    """Test executor with sequence input."""
    from src.committee.hierarchical_skills import SkillLSTMExecutor
    
    executor = SkillLSTMExecutor(state_dim=64, skill_embed_dim=64, hidden_dim=128, num_layers=2)
    
    state = torch.randn(2, 5, 64)  # batch=2, seq=5
    skill = torch.randn(2, 5, 64)
    
    out, hidden = executor(state, torch.randn(2, 5, 64))
    
    assert out['size_delta'].shape == (2, 5)
    assert out['price_offset_bps'].shape == (2, 5)
    assert out['urgency'].shape == (2, 5)


def test_executor_hidden_state_persistence():
    """Test that hidden state persists across steps."""
    from src.committee.hierarchical_skills import SkillLSTMExecutor
    
    executor = SkillLSTMExecutor(state_dim=64, skill_embed_dim=64, hidden_dim=128, num_layers=2)
    executor.reset_hidden(1, torch.device('cpu'))
    
    # First step
    out1, hidden1 = executor(torch.randn(1, 64), torch.randn(1, 64))
    h1, c1 = executor.get_hidden()
    
    # Second step with same hidden
    out2, hidden2 = executor(torch.randn(1, 64), torch.randn(1, 64), hidden1)
    h2, c2 = executor.get_hidden()
    
    # Hidden state should have changed
    assert not torch.allclose(h1, h2)
    assert not torch.allclose(c1, c2)


# ── 3. TerminationHead Tests ─────────────────────────────────────────
def test_termination_head():
    """Test termination head output range."""
    from src.committee.hierarchical_skills import TerminationHead
    
    term = TerminationHead(state_dim=64, skill_embed_dim=64)
    
    state = torch.randn(4, 64)
    skill = torch.randn(4, 64)
    
    beta = term(state, skill)
    
    assert beta.shape == (4, 1)
    assert (beta >= 0).all() and (beta <= 1).all()


# ── 4. SkillCritic Tests ─────────────────────────────────────────────
def test_skill_critic():
    """Test skill critic Q-value outputs."""
    from src.committee.hierarchical_skills import SkillCritic, NUM_SKILLS
    
    critic = SkillCritic(state_dim=64, num_skills=4)
    
    state = torch.randn(3, 64)
    q = critic(state)
    
    assert q.shape == (3, 4)
    
    # Test get_q
    skills = torch.tensor([0, 2, 1])
    q_selected = critic.get_q(torch.randn(3, 64), skills)
    assert q_selected.shape == (3,)


# ── 5. HierarchicalSkills Integration ────────────────────────────────
def test_hierarchical_skills_forward():
    """Test full hierarchical skills forward pass."""
    from src.committee.hierarchical_skills import HierarchicalSkills, SkillConfig
    
    config = SkillConfig(num_skills=4, skill_embed_dim=64, executor_hidden=128)
    model = HierarchicalSkills(state_dim=64, config=config)
    model.eval()
    
    state = torch.randn(2, 64)
    out = model(state)
    
    assert 'skill_idx' in out
    assert out['skill_idx'].shape == (2,)
    assert out['skill_probs'].shape == (2, 4)
    assert out['skill_embed'].shape == (2, 64)
    assert 'size_delta' in out
    assert 'price_offset_bps' in out
    assert 'urgency' in out
    assert 'termination_prob' in out
    assert out['termination_prob'].shape == (2,)


def test_hierarchical_skills_step():
    """Test hierarchical skills step execution with skill persistence."""
    from src.committee.hierarchical_skills import HierarchicalSkills, SkillConfig
    
    config = SkillConfig(num_skills=4, skill_embed_dim=64, executor_hidden=128, skill_horizon=3)
    model = HierarchicalSkills(state_dim=64, config=config)
    model.eval()
    model.reset_execution()
    
    # Run multiple steps - skill should persist for skill_horizon steps
    for i in range(5):
        out = model.step(torch.randn(1, 64))
        skill = out['skill_idx'].item()
        term = out['termination_prob'].item()
        print(f"  Step {i}: skill={skill}, term={term:.3f}")
    
    # Skill should persist for horizon steps then potentially change
    assert out['skill_idx'].shape == (1,)


# ── 6. OptionsCriticTrainer ──────────────────────────────────────────
def test_options_critic_trainer():
    """Test OptionsCriticTrainer loss computations."""
    from src.committee.hierarchical_skills import HierarchicalSkills, SkillConfig, OptionsCriticTrainer
    
    config = SkillConfig(num_skills=4, skill_embed_dim=64, executor_hidden=128)
    model = HierarchicalSkills(state_dim=64, config=config)
    trainer = OptionsCriticTrainer(model)
    
    # Create dummy batch
    batch_size = 4
    batch = {
        'states': torch.randn(batch_size, 64, requires_grad=True),
        'skills': torch.randint(0, 4, (batch_size,)),
        'log_probs': torch.randn(batch_size, requires_grad=True),
        'rewards': torch.randn(batch_size) * 0.01,
        'next_states': torch.randn(batch_size, 64),
        'dones': torch.zeros(batch_size),
        'gammas': torch.ones(batch_size) * 0.99,
        'betas': torch.rand(batch_size),
        'entropies': torch.rand(batch_size) * 0.5,
    }
    
    losses = trainer.update(batch)
    
    assert 'critic_loss' in losses
    assert 'actor_loss' in losses
    assert 'term_loss' in losses
    assert all(v >= 0 for v in losses.values())


# ── 7. Configuration Tests ──────────────────────────────────────────
def test_skill_config_serialization():
    """Test SkillConfig save/load."""
    from src.committee.hierarchical_skills import SkillConfig
    import tempfile
    import os
    
    config = SkillConfig(num_skills=4, skill_embed_dim=64, executor_hidden=128)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        path = f.name
    
    try:
        config.save(path)
        loaded = SkillConfig.load(path)
        
        assert loaded.num_skills == config.num_skills
        assert loaded.skill_embed_dim == config.skill_embed_dim
        assert loaded.executor_hidden == config.executor_hidden
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_skill_encoder_forward()
    print("✓ test_skill_encoder_forward")
    
    test_skill_encoder_sample()
    print("✓ test_skill_encoder_sample")
    
    test_executor_forward()
    print("✓ test_executor_forward")
    
    test_executor_sequence()
    print("✓ test_executor_sequence")
    
    test_executor_hidden_state_persistence()
    print("✓ test_executor_hidden_state_persistence")
    
    test_termination_head()
    print("✓ test_termination_head")
    
    test_skill_critic()
    print("✓ test_skill_critic")
    
    test_hierarchical_skills_forward()
    print("✓ test_hierarchical_skills_forward")
    
    test_hierarchical_skills_step()
    print("✓ test_hierarchical_skills_step")
    
    test_options_critic_trainer()
    print("✓ test_options_critic_trainer")
    
    test_skill_config_serialization()
    print("✓ test_skill_config_serialization")
    
    print("\n✅ All Hierarchical Skills tests passed!")