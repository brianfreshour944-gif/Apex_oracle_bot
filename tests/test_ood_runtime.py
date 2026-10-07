"""Runtime-behavior tests for the OOD discriminator (src/ood_discriminator.py).

The discriminator's default input width was 64 while the state vector it is
actually fed (build_state_vector) is 25, so the first is_ood() call raised a
shape error. That error was swallowed by check_ood_and_override's except, so
the OOD veto silently never armed. A stale 64-dim checkpoint on disk had the
same effect: load() hit a state_dict size mismatch and returned False.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="src.ood_discriminator imports torch at module level")

from src import ood_discriminator as ood  # noqa: E402


def _state(dim=25):
    return torch.zeros(dim, dtype=torch.float32)


def test_default_state_dim_matches_the_state_vector_the_bot_builds():
    """OOD_STATE_DIM must equal the width build_ood_state_vector produces."""
    from src.committee.decision_transformer import BRAINS, REGIMES, build_state_vector

    produced = build_state_vector("default", {}, {}, REGIMES, BRAINS)
    assert ood.OOD_STATE_DIM == produced.shape[0] == 25


def test_is_ood_runs_on_a_real_state_vector_without_shape_error():
    from src.committee.decision_transformer import BRAINS, REGIMES, build_state_vector

    disc = ood.OODDiscriminator()
    state = build_state_vector("default", {}, {}, REGIMES, BRAINS)
    is_ood, prob = disc.is_ood(torch.tensor(state))
    assert isinstance(is_ood, bool)
    assert 0.0 <= prob <= 1.0


def test_forward_rebuilds_net_on_width_mismatch():
    """A state whose width differs from the built net must not raise; the net
    is rebuilt to match instead of disabling the veto."""
    disc = ood.OODDiscriminator(state_dim=64)
    prob = disc.forward(_state(25))
    assert disc.state_dim == 25
    assert prob.shape[-1] == 1


def test_load_rebuilds_net_for_a_stale_width_checkpoint(tmp_path):
    """A checkpoint saved at the old 64-dim width loads successfully: the net
    is rebuilt to 64 before load_state_dict, instead of failing with a size
    mismatch and leaving an untrained net."""
    stale = ood.OODDiscriminator(state_dim=64)
    path = tmp_path / "ood_stale.pth"
    stale.save(str(path))

    fresh = ood.OODDiscriminator()  # 25-dim
    assert fresh.load(str(path)) is True
    assert fresh.state_dim == 64
    # and it can now run on a 64-dim input
    assert fresh.forward(_state(64)).shape[-1] == 1


def test_train_on_data_rebuilds_net_to_the_data_width():
    disc = ood.OODDiscriminator(state_dim=64)
    hist = np.zeros((40, 25), dtype=np.float32)
    live = np.ones((40, 25), dtype=np.float32)
    disc.train_on_data(hist, live, epochs=1)
    assert disc.state_dim == 25
    assert disc._is_trained is True


def test_bootstrap_min_history_is_exposed():
    """The bot job needs a bootstrap threshold it can train from without
    requiring _is_trained to already be True (the old deadlock)."""
    assert ood.OOD_BOOTSTRAP_MIN_HISTORY >= 1
