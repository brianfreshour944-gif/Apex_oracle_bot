"""Tests for the symbol-ban policy in scripts/weekly_analyzer.py.

The analyzer runs every 12h inside the bot process and writes
data/banned_symbols.json, which bot.py turns into a hard "Symbol Banned" veto.
Every run used to merge fresh bans over the persisted file with no removal path
whatsoever, so a single bad week disabled a symbol permanently -- one more way
the bot could go silently quiet with healthy infrastructure. Bans are now
lifted on recovery or after a TTL.
"""
from datetime import date, timedelta

import pytest

pytest.importorskip("pandas", reason="weekly_analyzer reads the trade DB with pandas")
pytest.importorskip("sqlalchemy", reason="weekly_analyzer builds a SQLAlchemy engine")

from scripts import weekly_analyzer

TODAY = date(2026, 9, 18)


def test_new_ban_is_timestamped():
    bans, lifted = weekly_analyzer.merge_bans(
        [], [{"symbol": "BTC/USD", "reason": "bad week"}], {}, today=TODAY
    )
    assert lifted == []
    assert bans == [{"symbol": "BTC/USD", "reason": "bad week", "banned_at": TODAY.isoformat()}]


def test_recovered_symbol_is_unbanned():
    existing = [{"symbol": "ETH/USD", "reason": "old", "banned_at": "2026-09-01"}]
    metrics = {"ETH/USD": {"total_trades": 8, "profit_factor": 1.4}}
    bans, lifted = weekly_analyzer.merge_bans(existing, [], metrics, today=TODAY)
    assert bans == []
    assert len(lifted) == 1
    assert lifted[0][0] == "ETH/USD"
    assert "recovered" in lifted[0][1]


def test_symbol_with_no_losses_is_unbanned():
    """profit_factor is inf when there is no gross loss -- that is recovery."""
    existing = [{"symbol": "SOL/USD", "reason": "old", "banned_at": "2026-09-10"}]
    metrics = {"SOL/USD": {"total_trades": 6, "profit_factor": float("inf")}}
    bans, lifted = weekly_analyzer.merge_bans(existing, [], metrics, today=TODAY)
    assert bans == []
    assert "recovered" in lifted[0][1]


def test_still_underperforming_stays_banned():
    existing = [{"symbol": "SOL/USD", "reason": "old", "banned_at": "2026-09-10"}]
    metrics = {"SOL/USD": {"total_trades": 9, "profit_factor": 0.5}}
    bans, lifted = weekly_analyzer.merge_bans(existing, [], metrics, today=TODAY)
    assert [b["symbol"] for b in bans] == ["SOL/USD"]
    assert bans[0]["banned_at"] == "2026-09-10"
    assert lifted == []


def test_ban_expires_after_ttl():
    stale = (TODAY - timedelta(days=weekly_analyzer.BAN_TTL_DAYS + 1)).isoformat()
    existing = [{"symbol": "XRP/USD", "reason": "old", "banned_at": stale}]
    bans, lifted = weekly_analyzer.merge_bans(existing, [], {}, today=TODAY)
    assert bans == []
    assert "expired" in lifted[0][1]


def test_legacy_ban_without_a_date_starts_its_ttl_now():
    """An un-timestamped entry must not make the whole file expire at once."""
    existing = [{"symbol": "SOL/USD", "reason": "legacy entry"}]
    bans, lifted = weekly_analyzer.merge_bans(existing, [], {}, today=TODAY)
    assert lifted == []
    assert bans[0]["banned_at"] == TODAY.isoformat()
    assert bans[0]["reason"] == "legacy entry"


def test_fresh_ban_wins_over_a_stale_duplicate():
    """A re-ban updates the reason but must NOT reset banned_at: the metrics
    are all-time (SELECT ... WHERE status='closed'), so a symbol that stays
    banned keeps producing the same fresh ban every run. Re-stamping the
    timestamp on each run reset the TTL clock and made the ban permanent."""
    existing = [{"symbol": "BTC/USD", "reason": "old", "banned_at": "2026-09-17"}]
    newly = [{"symbol": "BTC/USD", "reason": "fresh"}]
    bans, _ = weekly_analyzer.merge_bans(existing, newly, {}, today=TODAY)
    assert bans == [{"symbol": "BTC/USD", "reason": "fresh", "banned_at": "2026-09-17"}]


def test_requoting_the_same_ban_does_not_reset_the_ttl():
    """Regression: a symbol re-banned every run from unchanged all-time
    metrics must still age out. The first ban's date survives the re-ban, so
    once BAN_TTL_DAYS have passed the ban lifts even though the fresh metrics
    keep flagging the symbol."""
    banned_at = (TODAY - timedelta(days=weekly_analyzer.BAN_TTL_DAYS + 1)).isoformat()
    existing = [{"symbol": "BTC/USD", "reason": "old", "banned_at": banned_at}]
    newly = [{"symbol": "BTC/USD", "reason": "still bad (all-time PF)"}]
    bans, lifted = weekly_analyzer.merge_bans(existing, newly, {}, today=TODAY)
    assert bans == [], f"a TTL-expired ban must not be re-stamped back into existence: {bans}"
    assert "expired" in lifted[0][1]


def test_ban_entries_without_a_symbol_are_ignored():
    existing = [{"reason": "corrupt entry"}]
    bans, lifted = weekly_analyzer.merge_bans(existing, [], {}, today=TODAY)
    assert bans == []
    assert lifted == []


def test_learned_threshold_clamp_stays_below_the_veto_floor():
    """A persisted threshold must remain reachable by real committee scores."""
    assert 0 < weekly_analyzer.MAX_LEARNED_SCORE_THRESHOLD <= 0.55


def test_live_replay_buffer_exists_for_daily_learning():
    """The bot must learn from daily trades; live replay buffer is required."""
    import os
    assert os.path.exists("data/live_experiences.jsonl"), "live_experiences.jsonl missing — daily trade learning broken"


def test_ban_decision_uses_recent_trades_not_all_time():
    """A symbol whose only losses are OLD must not be re-banned from all-time
    metrics. Regression for the BTC deadlock: a ban blocks entries, so all-time
    PF freezes at the bad value and re-bans the symbol forever even after the
    TTL grants a fresh trial."""
    import pandas as pd

    now = pd.Timestamp("2026-10-12T00:00:00Z")
    df = pd.DataFrame(
        [
            # 6 losing trades, all closed ~200 days ago
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=200)},
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=201)},
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=202)},
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=203)},
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=204)},
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=205)},
            # 1 recent winning trade
            {"symbol": "BTC/USD", "realized_pnl": 5.0, "closed_at": now - pd.Timedelta(days=2)},
        ]
    )
    recent = weekly_analyzer._filter_recent_closed(df, 30, today=now)
    metrics = weekly_analyzer._symbol_metrics(recent)
    # Only the recent win survives -> no ban (below MIN_TRADES_TO_BAN).
    assert metrics["total_trades"] == 1
    assert not (
        metrics["total_trades"] >= weekly_analyzer.MIN_TRADES_TO_BAN
        and metrics["profit_factor"] < weekly_analyzer.MIN_PROFIT_FACTOR
    )
    # The all-time view still sees every trade (reporting/thresholds unchanged).
    assert weekly_analyzer._symbol_metrics(df)["total_trades"] == 7


def test_recent_losses_still_ban():
    """A symbol that keeps losing recently must stay banned on current evidence."""
    import pandas as pd

    now = pd.Timestamp("2026-10-12T00:00:00Z")
    df = pd.DataFrame(
        [
            {"symbol": "ETH/USD", "realized_pnl": -2.0, "closed_at": now - pd.Timedelta(days=d)}
            for d in range(1, 8)
        ]
    )
    recent = weekly_analyzer._filter_recent_closed(df, 30, today=now)
    metrics = weekly_analyzer._symbol_metrics(recent)
    assert metrics["total_trades"] >= weekly_analyzer.MIN_TRADES_TO_BAN
    assert metrics["profit_factor"] < weekly_analyzer.MIN_PROFIT_FACTOR


def test_filter_recent_closed_fails_closed_on_missing_timestamps():
    """Rows with missing/unparseable closed_at must be KEPT, not dropped --
    otherwise a broken timestamp column would silently un-ban the whole book."""
    import pandas as pd

    now = pd.Timestamp("2026-10-12T00:00:00Z")
    df = pd.DataFrame(
        [
            {"symbol": "SOL/USD", "realized_pnl": -1.0, "closed_at": None},
            {"symbol": "SOL/USD", "realized_pnl": -1.0, "closed_at": "not-a-date"},
            {"symbol": "SOL/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=500)},
        ]
    )
    recent = weekly_analyzer._filter_recent_closed(df, 30, today=now)
    assert len(recent) == 2  # the two undated rows are kept, the 500d-old one dropped


def test_filter_recent_closed_without_column_returns_all():
    """If closed_at is absent entirely, keep every row (fail-closed)."""
    import pandas as pd

    df = pd.DataFrame([{"symbol": "BTC/USD", "realized_pnl": -1.0}])
    assert len(weekly_analyzer._filter_recent_closed(df, 30)) == 1


def test_ttl_lifted_symbol_is_not_rebanned_from_stale_evidence():
    """End-to-end deadlock regression: after the TTL lifts BTC's ban, the SAME
    run's fresh-ban computation must use recent trades only, so months-old
    losses do not immediately re-ban it. Before the windowed fix the fresh ban
    was built from all-time metrics and re-banned BTC on every run forever."""
    import pandas as pd

    now = pd.Timestamp("2026-10-12T00:00:00Z")
    banned_at = (now - pd.Timedelta(days=weekly_analyzer.BAN_TTL_DAYS + 1)).date().isoformat()
    existing = [{"symbol": "BTC/USD", "reason": "PF 0.26", "banned_at": banned_at}]

    # All-time: 10 losing trades months ago (the stale evidence).
    df = pd.DataFrame(
        [
            {"symbol": "BTC/USD", "realized_pnl": -1.0, "closed_at": now - pd.Timedelta(days=200 + i)}
            for i in range(10)
        ]
    )
    recent = weekly_analyzer._filter_recent_closed(df, weekly_analyzer.BAN_METRIC_WINDOW_DAYS, today=now)
    ban_metrics = {s: weekly_analyzer._symbol_metrics(g) for s, g in recent.groupby("symbol")}

    newly_banned = [
        {"symbol": s, "reason": "stale"}
        for s, m in ban_metrics.items()
        if m["total_trades"] >= weekly_analyzer.MIN_TRADES_TO_BAN
        and m["profit_factor"] < weekly_analyzer.MIN_PROFIT_FACTOR
    ]
    assert newly_banned == [], "stale trades must not produce a fresh ban"

    bans, lifted = weekly_analyzer.merge_bans(existing, newly_banned, ban_metrics, today=now.date())
    assert bans == [], "TTL-expired BTC must stay lifted, not be re-banned"
    assert lifted and lifted[0][0] == "BTC/USD"