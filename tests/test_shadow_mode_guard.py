"""Regression tests for research/backtest -> live shadow-ledger contamination.

Original incident (2026-10-11, a Sunday): the scheduled 04:00 automatic
researcher ran ``run_backtest(symbol="BTC/USD", seed=99, regime="all")``. With
no real bars the backtest fell back to ``_generate_synthetic_bars`` (a series
that starts at $100), and because ``run_backtest`` defaults to
``use_committee=True`` the committee's ``transformer_brain`` called
``evaluate_candidates`` -- which wrote those synthetic prices straight into the
shared ``shadow_trades`` table. 687 live-opened BTC/USD positions were then
"closed" at ~$100, booking ~-$82.6k each (~-$56.97M).

These tests lock in the fail-closed fix:
  * only an explicit ``execution_mode == "live"`` may write shadow rows;
  * the real committee-enabled backtest path writes nothing;
  * live open/close/dedupe behaviour is preserved;
  * unsupported synthetic regimes raise instead of silently defaulting.
"""

import pytest

import src.db as db
from src.config import settings


@pytest.fixture
def fresh_db(tmp_path):
    """Isolated temporary database -- never the production DB."""
    original = settings.DATABASE_URL
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = "sqlite:///" + str(tmp_path / "bot.db").replace("\\", "/")
    db.init_db()
    yield
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
    db._tables_ensured = False
    settings.DATABASE_URL = original


def _shadow_rows():
    with db.get_db_session() as session:
        return session.execute(db.select(db.ShadowTrade)).scalars().all()


def _shadow_count():
    return len(_shadow_rows())


# ── tiny stand-ins: reach the shadow-arena call deterministically, no network ──

class _IdentityScaler:
    """Scaler stub: identity transform with an explicit feature list."""

    def __init__(self, feature_names):
        self.feature_names_in_ = list(feature_names)

    def transform(self, data):
        import numpy as np
        return np.asarray(data, dtype="float32")


def _active_cols():
    from src.feature_engineering import get_active_features
    return list(get_active_features())


def _make_buy_candidate(cols):
    """A candidate whose model always votes 'buy' (positive logit), so an
    unguarded arena write would definitely open a shadow row."""
    import torch

    class _Model(torch.nn.Module):
        def forward(self, x):
            # constant positive logit -> sigmoid > 0.5 -> action 'buy'
            return torch.full((x.shape[0], 1), 2.0)

    return {
        "model": _Model(),
        "device": "cpu",
        "config": {"threshold": 0.5, "stop_loss": 0.02, "profit_target": 0.04},
        "scaler": _IdentityScaler(cols),
        "cols": list(cols),
    }


def _patch_candidates(monkeypatch):
    import src.shadow_arena as sa
    monkeypatch.setattr(sa, "get_candidates",
                        lambda: {"Bot_A_Light": _make_buy_candidate(_active_cols())})


def _patch_transformer_predictor(monkeypatch):
    """Inject a real (tiny) transformer predictor so the REAL transformer_brain
    body runs through to the evaluate_candidates call, offline."""
    import torch

    from src.committee import transformer_brain as tb
    from src.feature_engineering import get_active_features

    class _TModel(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2)).unsqueeze(1)  # constant logit -> prob 0.5

    cols = list(get_active_features())
    tb.set_ml_predictor_override(_TModel(), _IdentityScaler(cols), torch.device("cpu"), len(cols))
    monkeypatch.setattr(settings, "ADAPTIVE_ML_ENABLED", True)
    monkeypatch.setattr(settings, "USE_FAST_ENSEMBLE", False)
    yield
    tb.reset_ml_predictor()


def _synthetic_bars(n=60):
    from src.backtest import _generate_synthetic_bars
    return {"BTC/USD": _generate_synthetic_bars("BTC/USD", n=n, seed=99, regime="volatile")}


# ── (A) the real committee-enabled synthetic backtest path ────────────────────

@pytest.mark.asyncio
async def test_committee_backtest_writes_no_shadow_rows(fresh_db, monkeypatch):
    """The actual reported failure path: run_portfolio_backtest(use_committee=True)
    over SYNTHETIC bars, real transformer_brain, no execution_mode -> 0 rows."""
    from src.backtest import run_portfolio_backtest

    _patch_candidates(monkeypatch)
    for _ in _patch_transformer_predictor(monkeypatch):
        pass
    await run_portfolio_backtest(_synthetic_bars(), start_equity=10_000.0, use_committee=True)
    assert _shadow_count() == 0, "backtest wrote to the live shadow ledger"


@pytest.mark.asyncio
async def test_backtest_actually_reaches_shadow_arena(fresh_db, monkeypatch):
    """Anti-vacuity guard for the test above.

    Proves the committee-enabled backtest genuinely invokes evaluate_candidates
    (so a zero-row assertion is meaningful), and that every such call carries
    is_live=False. Without this, the previous test could pass simply because the
    arena was never reached."""
    import src.shadow_arena as sa
    from src.backtest import run_portfolio_backtest

    _patch_candidates(monkeypatch)
    for _ in _patch_transformer_predictor(monkeypatch):
        pass

    seen = []
    real = sa.evaluate_candidates

    def spy(symbol, price, df_feat, is_live=False):
        seen.append(is_live)
        return real(symbol, price, df_feat, is_live=is_live)

    monkeypatch.setattr(sa, "evaluate_candidates", spy)
    await run_portfolio_backtest(_synthetic_bars(), start_equity=10_000.0, use_committee=True)

    assert seen, "backtest never reached the shadow arena (previous test would be vacuous)"
    assert all(v is False for v in seen), f"non-live calls reached the arena as live: {set(seen)}"


# ── (B) committee call with a non-live signal writes nothing ──────────────────

@pytest.mark.asyncio
async def test_committee_nonlive_signal_writes_nothing(fresh_db, monkeypatch):
    """run_committee with backtest_df set and NO execution_mode -> 0 rows.
    This exercises committee -> transformer_brain -> evaluate_candidates."""
    from src.backtest import _generate_synthetic_bars
    from src.committee.committee import run_committee

    _patch_candidates(monkeypatch)
    for _ in _patch_transformer_predictor(monkeypatch):
        pass
    bars = _generate_synthetic_bars("BTC/USD", n=60, seed=99, regime="volatile")
    signal = {"action": "buy", "regime": "neutral", "backtest_df": bars}
    await run_committee("BTC/USD", 100.0, signal)
    assert _shadow_count() == 0


# ── (C) live mode still writes (guard must not over-block) ────────────────────

@pytest.mark.asyncio
async def test_committee_live_signal_writes_rows(fresh_db, monkeypatch):
    """Positive control: execution_mode == 'live' must still reach the arena and
    open a shadow row (same call path as B)."""
    from src.backtest import _generate_synthetic_bars
    from src.committee.committee import run_committee

    _patch_candidates(monkeypatch)
    for _ in _patch_transformer_predictor(monkeypatch):
        pass
    bars = _generate_synthetic_bars("BTC/USD", n=60, seed=99, regime="volatile")
    signal = {"action": "buy", "regime": "neutral", "backtest_df": bars,
              "execution_mode": "live"}
    await run_committee("BTC/USD", 83_000.0, signal)
    rows = _shadow_rows()
    assert len(rows) == 1, "live signal did not write a shadow row"
    assert rows[0].status == "open" and rows[0].entry_price == 83_000.0


# ── (D) direct-API guards: fail closed + live preserved ───────────────────────

def test_evaluate_shadow_position_fails_closed(fresh_db):
    from src.shadow_arena import _evaluate_shadow_position
    _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83_000.0, 0.02, 0.04, new_action="buy")
    assert _shadow_count() == 0


def test_evaluate_candidates_fails_closed(fresh_db):
    from src.shadow_arena import evaluate_candidates
    evaluate_candidates("BTC/USD", 83_000.0, None)          # default is_live=False
    evaluate_candidates("BTC/USD", 83_000.0, None, is_live=False)
    assert _shadow_count() == 0


def test_live_mode_still_opens_and_closes(fresh_db):
    from src.shadow_arena import _evaluate_shadow_position
    _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83_000.0, 0.02, 0.04,
                              new_action="buy", is_live=True)
    assert _shadow_count() == 1
    _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 84_000.0, 0.02, 0.04,
                              new_action="close", is_live=True)
    rows = _shadow_rows()
    assert len(rows) == 1 and rows[0].status == "closed" and rows[0].exit_price == 84_000.0


def test_research_cannot_close_live_position(fresh_db):
    """A non-live call must not overwrite/close a live-opened position."""
    from src.shadow_arena import _evaluate_shadow_position
    _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83_000.0, 0.02, 0.04,
                              new_action="buy", is_live=True)
    _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 100.0, 0.02, 0.04,
                              new_action="close", is_live=False)
    rows = _shadow_rows()
    assert len(rows) == 1 and rows[0].status == "open"
    assert rows[0].exit_price is None and rows[0].realized_pnl == 0.0


def test_original_incident_not_recorded_as_loss(fresh_db):
    """The exact original mismatch: ~$83k live entry + synthetic $100 close."""
    from src.shadow_arena import _evaluate_shadow_position
    _evaluate_shadow_position("Bot_B_Standard", "BTC/USD", 82_736.05, 0.02, 0.04,
                              new_action="buy", is_live=True)
    _evaluate_shadow_position("Bot_B_Standard", "BTC/USD", 99.04, 0.02, 0.04,
                              new_action="close", is_live=False)   # research run
    rows = _shadow_rows()
    assert len(rows) == 1 and rows[0].status == "open"
    assert all(r.realized_pnl == 0.0 for r in rows), "synthetic close booked a live loss"


def test_live_dedupe_preserved(fresh_db):
    """Live behaviour unchanged: repeated live buys still open exactly one row."""
    from src.shadow_arena import _evaluate_shadow_position
    for _ in range(5):
        _evaluate_shadow_position("Bot_A_Light", "BTC/USD", 83_000.0, 0.02, 0.04,
                                  new_action="buy", is_live=True)
    assert _shadow_count() == 1


# ── (E) synthetic regime validation ───────────────────────────────────────────

def test_unsupported_regime_raises():
    from src.backtest import _generate_synthetic_bars
    with pytest.raises(ValueError):
        _generate_synthetic_bars("BTC/USD", n=10, seed=99, regime="all")


def test_supported_regimes_still_work():
    from src.backtest import _generate_synthetic_bars
    for regime in ("trending", "mean_reverting", "volatile"):
        df = _generate_synthetic_bars("BTC/USD", n=10, seed=99, regime=regime)
        assert len(df) == 10
