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
    existing = [{"symbol": "BTC/USD", "reason": "old", "banned_at": "2026-09-17"}]
    newly = [{"symbol": "BTC/USD", "reason": "fresh"}]
    bans, _ = weekly_analyzer.merge_bans(existing, newly, {}, today=TODAY)
    assert bans == [{"symbol": "BTC/USD", "reason": "fresh", "banned_at": TODAY.isoformat()}]


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