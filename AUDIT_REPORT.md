# Apex Oracle Bot - Comprehensive Audit Report

**Date:** 2025-08-22  
**Auditor:** Automated Code Review  
**Scope:** Full codebase audit per 5-point audit checklist

---

## Executive Summary

| Category | Status | Critical Findings |
|----------|--------|-------------------|
| Financial Correctness | ⚠️ **ISSUES FOUND** | 2 bugs in PnL/slippage calculations |
| Concurrency/Race Conditions | ⚠️ **ISSUES FOUND** | 9 race conditions, 5 check-then-act patterns |
| API Rate-Limit Compliance | ✅ **PASS** | Well under limits, but burst risk exists |
| Security Audit | ✅ **PASS** | No secrets in logs/git; 1 minor CVE in httpx |
| Memory/Soak Test | 📋 **PLAN CREATED** | Plan ready for 24-48h soak test |

---

## 1. Financial Correctness Audit

### ✅ Verified Correct
- **Position Sizing**: All math verified with concrete scenario (BTC $50k, trending, 1% risk → 0.05 BTC = $2500)
- **Profit Target**: 3% trigger works correctly
- **Stop Loss**: 4% trigger works correctly  
- **Trailing Stop**: Both long/short logic verified (4% activation, 3% distance)

### ❌ BUGS FOUND

#### Bug 1: Short Position PnL Sign Error (`src/bot.py` lines 81-86)
**Location:** `_record_committee_outcome()` function
```python
# Current (buggy):
if action == "buy":
    realized_pnl = (exit_price - entry_price) * qty
else:  # sell / short
    realized_pnl = (entry_price - exit_price) * qty  # BUG: qty can be negative!
```

**Impact:** Short positions that profit show negative PnL when qty comes from exchange (negative for shorts). Return % is correct but realized PnL sign is wrong.

**Fix:** Use `abs(qty)` or ensure qty is always positive in snapshot.

#### Bug 2: Close Path Slippage Calculation (`src/bot.py` lines 847-850)
```python
# Current (buggy):
expected_price = current_price  # Should be entry_price!
slippage_bps = abs(filled_price - expected_price) / expected_price * 10000
```

**Impact:** Slippage always calculated as 0 for closes since `filled_price == expected_price == current_price`.

---

## 2. Concurrency/Race-Condition Audit

### ✅ Well Protected
- Per-symbol asyncio locks (`_symbol_locks`) for cooldowns, position_adds, trailing_peaks
- `_exposure_lock` for exposure reservations
- `_equity_lock` for equity/drawdown updates
- `_peak_prices_lock` (threading.Lock) for trailing stop dict
- DB sessions: New session per operation, no sharing

### ⚠️ RACE CONDITIONS FOUND (9)

| # | Shared State | Risk | Location |
|---|-------------|------|----------|
| 1 | `_state.cooldowns` dict | Main loop cleanup iterates without lock | bot.py:1676-1678 |
| 2 | `_state.latest_scan_results` | Concurrent writers + heartbeat reader | bot.py:550,558,561 / 890 |
| 3 | `_state.trade_timestamps` list | Append vs list comprehension rebuild | bot.py:771 / 1631 |
| 4 | `_state.exchange_failure_count` int | Increment vs read without lock | bot.py:1671,1673 / 1636 |
| 5 | `RiskManager.peak_prices` | threading.Lock but async access pattern | risk.py:348-378 |
| 6 | `_state._background_tasks` set | Concurrent add/discard | Multiple locations |
| 7 | `Exchange._bars_cache` | Concurrent read/write with TTL | exchange.py:184-188 |
| 8 | `Exchange._order_cache` | Concurrent read/write with TTL | exchange.py:386-394 |
| 9 | `_reserved_exposure` release timing | Release after order but before fill confirmation | bot.py:761,768 |

### ⚠️ CHECK-THEN-ACT PATTERNS (5)

| Pattern | Check | Act | Gap |
|---------|-------|-----|-----|
| Exposure Reserve → Order | `check_and_reserve_exposure()` | `create_order()` | Lock released between |
| Position Limit → Order | `position_limit_exceeded` check | `create_order()` | No lock |
| Cooldown → Signal | Cooldown check | Signal generation | Protected by lock ✅ |
| Scale-in Gates | Max adds/time/score checks | Order placement | Protected by lock ✅ |
| Reservation Release | Order success | `release_reserved_exposure()` | If order fills after timeout |

### Critical Race Scenario: Exposure Overshoot
```
Task A (BTC): check_and_reserve_exposure($1000) → OK, lock released
Task B (ETH): check_and_reserve_exposure($1000) → OK (sees same headroom), lock released
Both place orders → Total exposure exceeds cap
```
**Mitigation:** `_reserved_exposure` list tracks pending, but window exists between order placement and `update_account_status()` reflecting actual fills.

---

## 3. API Rate-Limit Compliance

### ✅ Within Limits
| Scenario | Requests/Min | Limit | Margin |
|----------|-------------|-------|--------|
| Steady State | 4 | 200 | 98% |
| Active Trading (3 orders) | 67 | 200 | 66% |

### ⚠️ Burst Risk
- **Burst:** 63 requests in 10 seconds (378 req/min equivalent)
- **Risk:** Concurrent `asyncio.gather` for 3 symbols + 3 order confirmations
- **Mitigation:** Retry logic with exponential backoff + Retry-After header respect

### Recommendations
1. Add proactive rate limiter (token bucket) before bursts
2. Batch `get_bars` multi-symbol requests
3. Monitor rate limit headers proactively

---

## 4. Security Audit

### ✅ No Secrets in Logs/Code
- No API keys, secrets, tokens in logging calls
- No hardcoded secrets in config (uses Pydantic env vars)
- `.env` properly in `.gitignore`

### ✅ Git History Clean
- No `sk_live`, `ghp_`, `AKIA`, or `sk-` patterns in git history
- No `.env` file ever committed

### ⚠️ Dependency Vulnerabilities
| Package | Version | CVE | Status |
|---------|---------|-----|--------|
| httpx | 0.28.1 | CVE-2024-47881 (ReDoS) | **Fixed in 0.28.1** ✅ |

**Note:** pyproject.toml and requirements.txt have different dependencies (alpaca-trade-api vs alpaca-py). Consolidate to single source.

---

## 5. Memory/Soak Test Plan

### Plan Created: `soak_test_plan.py`

**Candidate Leak Sources (14 identified):**
1. `_state.latest_scan_results` - no cleanup
2. `_state.trade_timestamps` - list grows
3. `_state.cooldowns` - cleanup race
4. `_state.position_adds` - no cleanup on close
5. `_state._background_tasks` - task cleanup
6. `RiskManager._reserved_exposure` - release timing
7. `RiskManager._realized_tx_costs` - unbounded growth
8. `RiskManager.peak_prices` - trailing stop cleanup
9. `_state._symbol_locks` - never cleaned
10. `Exchange._bars_cache` - TTL 60s
11. `Exchange._order_cache` - TTL 300s
12. DB connection pool
13. Thread pool (asyncio.to_thread)
14. Background tasks set

**Soak Test Procedure:**
1. Deploy → capture baseline `docker stats`
2. Cron every 2h: `docker stats >> memory_log.txt`
3. Run 24-48h → analyze slope (target: ~0 MB/hour)
4. Red flags: >10MB/hour sustained growth, OOM kills, connection pool exhaustion

---

## Priority Fix List

| Priority | Issue | File:Line |
|----------|-------|-----------|
| 🔴 CRITICAL | Short PnL sign bug | bot.py:81-86 |
| 🔴 CRITICAL | Close slippage calc bug | bot.py:847-850 |
| 🟠 HIGH | Exposure check-then-act race | bot.py:686-763 |
| 🟠 HIGH | Position limit check-then-act | bot.py:598-612 |
| 🟠 HIGH | Race conditions (9 locations) | Multiple |
| 🟡 MEDIUM | httpx CVE-2024-47881 | requirements.txt |
| 🟡 MEDIUM | Dependency consolidation | pyproject.toml vs requirements.txt |
| 🟡 MEDIUM | Proactive rate limiting | exchange.py |
| 🟢 LOW | Memory leak monitoring | soak_test_plan.py |

---

## Files Generated During Audit

| File | Purpose |
|------|---------|
| `rate_limit_analysis.py` | API rate limit verification |
| `vuln_check.py` | Dependency vulnerability scan |
| `soak_test_plan.py` | Memory soak test procedure |
| `security_check.py` | Secrets scanning script |

---

## Next Steps

1. **Immediate:** Fix 2 critical financial bugs (PnL sign, slippage calc)
2. **High:** Add locks for 9 race conditions; fix 2 check-then-act patterns
3. **Medium:** Add proactive rate limiter; consolidate dependencies
4. **Low:** Execute 24h soak test per plan; monitor for leaks

---

*Report generated by automated code audit. Manual verification recommended for all critical findings.*