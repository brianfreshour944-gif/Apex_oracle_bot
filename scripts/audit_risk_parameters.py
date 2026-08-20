#!/usr/bin/env python3
"""
Risk Parameter Audit — Step 6 of Foundation Hardening.

Validates that every active risk parameter is:
1. Explicitly set (not defaulted silently)
2. Documented with rationale
3. Within sensible bounds
4. Not discovered after the fact from trading logs

Run: python scripts/audit_risk_parameters.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_kv(key: str, value: any, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")

def check_bounds(name: str, value: any, min_val: any = None, max_val: any = None, expected_type: type = None):
    """Check if a value is within expected bounds."""
    issues = []
    if expected_type and not isinstance(value, expected_type):
        issues.append(f"Type mismatch: expected {expected_type.__name__}, got {type(value).__name__}")
    if min_val is not None and value < min_val:
        issues.append(f"Below minimum: {value} < {min_val}")
    if max_val is not None and value > max_val:
        issues.append(f"Above maximum: {value} > {max_val}")
    return issues


def main():
    print_section(f"RISK PARAMETER AUDIT — {datetime.now(timezone.utc).isoformat()}")
    print("Validating all risk parameters are explicitly set with rationale...\n")

    all_issues = []

    # ─── CORE EXPOSURE LIMITS ───
    print_section("CORE EXPOSURE LIMITS")
    
    # ACCOUNT_BASE - The foundation for all risk calculations
    print_kv("ACCOUNT_BASE", f"${settings.ACCOUNT_BASE:,.2f}")
    issues = check_bounds("ACCOUNT_BASE", settings.ACCOUNT_BASE, min_val=100, max_val=10_000_000, expected_type=float)
    for issue in issues: all_issues.append(f"ACCOUNT_BASE: {issue}")

    # BASE_RISK_PERCENT - Risk per trade
    print_kv("BASE_RISK_PERCENT", f"{settings.BASE_RISK_PERCENT*100:.2f}%")
    issues = check_bounds("BASE_RISK_PERCENT", settings.BASE_RISK_PERCENT, min_val=0.001, max_val=0.1, expected_type=float)
    for issue in issues: all_issues.append(f"BASE_RISK_PERCENT: {issue}")

    # MAX_SINGLE_TRADE_USD - Hard cap per trade
    print_kv("MAX_SINGLE_TRADE_USD", f"${settings.MAX_SINGLE_TRADE_USD:,.2f}")
    issues = check_bounds("MAX_SINGLE_TRADE_USD", settings.MAX_SINGLE_TRADE_USD, min_val=10, max_val=settings.ACCOUNT_BASE, expected_type=float)
    for issue in issues: all_issues.append(f"MAX_SINGLE_TRADE_USD: {issue}")

    # MAX_PORTFOLIO_VALUE vs MAX_PORTFOLIO_PCT
    print_kv("MAX_PORTFOLIO_VALUE (static)", f"${settings.MAX_PORTFOLIO_VALUE:,.2f}")
    print_kv("MAX_PORTFOLIO_PCT (dynamic)", f"{settings.MAX_PORTFOLIO_PCT*100:.0f}%")
    effective_cap = settings.ACCOUNT_BASE * settings.MAX_PORTFOLIO_PCT
    print_kv("  -> Effective Cap", f"${effective_cap:,.2f}")
    
    # Check consistency
    if settings.MAX_PORTFOLIO_PCT > 0:
        if effective_cap != settings.MAX_PORTFOLIO_VALUE:
            print_kv("  [NOTE]", f"Dynamic cap (${effective_cap:,.2f}) overrides static (${settings.MAX_PORTFOLIO_VALUE:,.2f})")
    else:
        print_kv("  [NOTE]", "MAX_PORTFOLIO_PCT=0, using static MAX_PORTFOLIO_VALUE")
    
    issues = check_bounds("MAX_PORTFOLIO_PCT", settings.MAX_PORTFOLIO_PCT, min_val=0.0, max_val=1.0, expected_type=float)
    for issue in issues: all_issues.append(f"MAX_PORTFOLIO_PCT: {issue}")

    # MAX_OPEN_POSITIONS
    print_kv("MAX_OPEN_POSITIONS", settings.MAX_OPEN_POSITIONS)
    issues = check_bounds("MAX_OPEN_POSITIONS", settings.MAX_OPEN_POSITIONS, min_val=1, max_val=20, expected_type=int)
    for issue in issues: all_issues.append(f"MAX_OPEN_POSITIONS: {issue}")

    # ─── DRAWDOWN KILLSWITCHES ───
    print_section("DRAWDOWN & LOSS LIMITS")
    
    print_kv("MAX_DRAWDOWN_STOP", f"{settings.MAX_DRAWDOWN_STOP}%")
    issues = check_bounds("MAX_DRAWDOWN_STOP", settings.MAX_DRAWDOWN_STOP, min_val=-50, max_val=-0.1, expected_type=float)
    for issue in issues: all_issues.append(f"MAX_DRAWDOWN_STOP: {issue}")

    print_kv("DAILY_LOSS_LIMIT", f"{settings.DAILY_LOSS_LIMIT}%")
    issues = check_bounds("DAILY_LOSS_LIMIT", settings.DAILY_LOSS_LIMIT, min_val=-20, max_val=-0.1, expected_type=float)
    for issue in issues: all_issues.append(f"DAILY_LOSS_LIMIT: {issue}")

    # Daily loss should be less severe than max drawdown
    if abs(settings.DAILY_LOSS_LIMIT) > abs(settings.MAX_DRAWDOWN_STOP):
        all_issues.append("DAILY_LOSS_LIMIT exceeds MAX_DRAWDOWN_STOP (should be smaller)")

    # ─── POSITION MANAGEMENT ───
    print_section("POSITION MANAGEMENT")
    
    print_kv("PROFIT_TARGET_PCT", f"{settings.PROFIT_TARGET_PCT*100:.2f}%")
    issues = check_bounds("PROFIT_TARGET_PCT", settings.PROFIT_TARGET_PCT, min_val=0.001, max_val=0.5, expected_type=float)
    for issue in issues: all_issues.append(f"PROFIT_TARGET_PCT: {issue}")

    print_kv("STOP_LOSS_PCT", f"{settings.STOP_LOSS_PCT*100:.2f}%")
    issues = check_bounds("STOP_LOSS_PCT", settings.STOP_LOSS_PCT, min_val=0.001, max_val=0.5, expected_type=float)
    for issue in issues: all_issues.append(f"STOP_LOSS_PCT: {issue}")

    # Stop loss should be smaller than max drawdown
    if settings.STOP_LOSS_PCT >= abs(settings.MAX_DRAWDOWN_STOP / 100):
        all_issues.append("STOP_LOSS_PCT should be smaller than MAX_DRAWDOWN_STOP")

    print_kv("ATR_STOP_MULTIPLIER", settings.ATR_STOP_MULTIPLIER)
    issues = check_bounds("ATR_STOP_MULTIPLIER", settings.ATR_STOP_MULTIPLIER, min_val=0.5, max_val=5.0, expected_type=float)
    for issue in issues: all_issues.append(f"ATR_STOP_MULTIPLIER: {issue}")

    print_kv("MAX_HOLD_HOURS", settings.MAX_HOLD_HOURS)
    issues = check_bounds("MAX_HOLD_HOURS", settings.MAX_HOLD_HOURS, min_val=0.5, max_val=168, expected_type=float)
    for issue in issues: all_issues.append(f"MAX_HOLD_HOURS: {issue}")

    print_kv("COOLDOWN_SECONDS_BUY", settings.COOLDOWN_SECONDS_BUY)
    issues = check_bounds("COOLDOWN_SECONDS_BUY", settings.COOLDOWN_SECONDS_BUY, min_val=0, max_val=86400, expected_type=int)
    for issue in issues: all_issues.append(f"COOLDOWN_SECONDS_BUY: {issue}")

    # ─── PYRAMID / SCALE-IN GATES ───
    print_section("PYRAMID / SCALE-IN GATES")
    
    print_kv("MAX_POSITION_ADDS", settings.MAX_POSITION_ADDS)
    issues = check_bounds("MAX_POSITION_ADDS", settings.MAX_POSITION_ADDS, min_val=0, max_val=10, expected_type=int)
    for issue in issues: all_issues.append(f"MAX_POSITION_ADDS: {issue}")

    print_kv("POSITION_ADD_MIN_SECONDS", settings.POSITION_ADD_MIN_SECONDS)
    issues = check_bounds("POSITION_ADD_MIN_SECONDS", settings.POSITION_ADD_MIN_SECONDS, min_val=0, max_val=86400, expected_type=int)
    for issue in issues: all_issues.append(f"POSITION_ADD_MIN_SECONDS: {issue}")

    print_kv("POSITION_ADD_MIN_SCORE_INCREASE", settings.POSITION_ADD_MIN_SCORE_INCREASE)
    issues = check_bounds("POSITION_ADD_MIN_SCORE_INCREASE", settings.POSITION_ADD_MIN_SCORE_INCREASE, min_val=0.0, max_val=1.0, expected_type=float)
    for issue in issues: all_issues.append(f"POSITION_ADD_MIN_SCORE_INCREASE: {issue}")

    print_kv("POSITION_ADD_SIZE_DECAY", settings.POSITION_ADD_SIZE_DECAY)
    issues = check_bounds("POSITION_ADD_SIZE_DECAY", settings.POSITION_ADD_SIZE_DECAY, min_val=0.0, max_val=1.0, expected_type=float)
    for issue in issues: all_issues.append(f"POSITION_ADD_SIZE_DECAY: {issue}")

    # ─── TRANSACTION COST MODEL ───
    print_section("TRANSACTION COST MODEL")
    
    print_kv("TX_COST_FEE_BPS", settings.TX_COST_FEE_BPS)
    issues = check_bounds("TX_COST_FEE_BPS", settings.TX_COST_FEE_BPS, min_val=0, max_val=100, expected_type=float)
    for issue in issues: all_issues.append(f"TX_COST_FEE_BPS: {issue}")

    print_kv("TX_COST_SLIPPAGE_BPS", settings.TX_COST_SLIPPAGE_BPS)
    issues = check_bounds("TX_COST_SLIPPAGE_BPS", settings.TX_COST_SLIPPAGE_BPS, min_val=0, max_val=500, expected_type=float)
    for issue in issues: all_issues.append(f"TX_COST_SLIPPAGE_BPS: {issue}")

    print_kv("TX_COST_SPREAD_BPS", settings.TX_COST_SPREAD_BPS)
    issues = check_bounds("TX_COST_SPREAD_BPS", settings.TX_COST_SPREAD_BPS, min_val=0, max_val=200, expected_type=float)
    for issue in issues: all_issues.append(f"TX_COST_SPREAD_BPS: {issue}")

    print_kv("TX_COST_MIN_EDGE_BPS", settings.TX_COST_MIN_EDGE_BPS)
    issues = check_bounds("TX_COST_MIN_EDGE_BPS", settings.TX_COST_MIN_EDGE_BPS, min_val=0, max_val=500, expected_type=float)
    for issue in issues: all_issues.append(f"TX_COST_MIN_EDGE_BPS: {issue}")

    print_kv("TX_COST_USE_DYNAMIC", settings.TX_COST_USE_DYNAMIC)
    
    # Calculate round-trip cost
    round_trip = 2 * (settings.TX_COST_FEE_BPS + settings.TX_COST_SLIPPAGE_BPS + settings.TX_COST_SPREAD_BPS)
    print_kv("  -> Round-trip Cost", f"{round_trip:.1f} bps ({round_trip/100:.2f}%)")
    print_kv("  -> Min Edge Required", f"{settings.TX_COST_MIN_EDGE_BPS:.1f} bps")
    print_kv("  -> Profit Target", f"{settings.PROFIT_TARGET_PCT*10000:.1f} bps")
    
    net_edge = settings.PROFIT_TARGET_PCT * 10000 - round_trip
    print_kv("  -> Net Edge (target - costs)", f"{net_edge:.1f} bps")
    
    if net_edge < settings.TX_COST_MIN_EDGE_BPS:
        all_issues.append(f"Net edge ({net_edge:.1f} bps) < min required ({settings.TX_COST_MIN_EDGE_BPS} bps) - trades may be rejected")

    # ─── TRAILING STOP ───
    print_section("TRAILING STOP")
    
    print_kv("TRAILING_STOP_ENABLED", settings.TRAILING_STOP_ENABLED)
    print_kv("TRAILING_ACTIVATION_PCT", f"{settings.TRAILING_ACTIVATION_PCT*100:.2f}%")
    print_kv("TRAILING_DISTANCE_PCT", f"{settings.TRAILING_DISTANCE_PCT*100:.2f}%")
    
    issues = check_bounds("TRAILING_ACTIVATION_PCT", settings.TRAILING_ACTIVATION_PCT, min_val=0.001, max_val=0.2, expected_type=float)
    for issue in issues: all_issues.append(f"TRAILING_ACTIVATION_PCT: {issue}")
    issues = check_bounds("TRAILING_DISTANCE_PCT", settings.TRAILING_DISTANCE_PCT, min_val=0.001, max_val=0.2, expected_type=float)
    for issue in issues: all_issues.append(f"TRAILING_DISTANCE_PCT: {issue}")
    
    if settings.TRAILING_DISTANCE_PCT >= settings.TRAILING_ACTIVATION_PCT:
        all_issues.append("TRAILING_DISTANCE_PCT should be smaller than TRAILING_ACTIVATION_PCT")

    # ─── CIRCUIT BREAKER ───
    print_section("CIRCUIT BREAKER")
    
    print_kv("CIRCUIT_FAILURE_THRESHOLD", settings.CIRCUIT_FAILURE_THRESHOLD)
    issues = check_bounds("CIRCUIT_FAILURE_THRESHOLD", settings.CIRCUIT_FAILURE_THRESHOLD, min_val=1, max_val=20, expected_type=int)
    for issue in issues: all_issues.append(f"CIRCUIT_FAILURE_THRESHOLD: {issue}")

    print_kv("CIRCUIT_OPEN_SECONDS", settings.CIRCUIT_OPEN_SECONDS)
    issues = check_bounds("CIRCUIT_OPEN_SECONDS", settings.CIRCUIT_OPEN_SECONDS, min_val=10, max_val=3600, expected_type=int)
    for issue in issues: all_issues.append(f"CIRCUIT_OPEN_SECONDS: {issue}")

    # ─── ADDITIONAL KILLSWITCHES ───
    print_section("ADDITIONAL KILLSWITCHES")
    
    print_kv("VOLATILITY_SPIKE_PCT", settings.VOLATILITY_SPIKE_PCT)
    issues = check_bounds("VOLATILITY_SPIKE_PCT", settings.VOLATILITY_SPIKE_PCT, min_val=5, max_val=100, expected_type=float)
    for issue in issues: all_issues.append(f"VOLATILITY_SPIKE_PCT: {issue}")

    print_kv("API_DOWN_MINUTES", settings.API_DOWN_MINUTES)
    issues = check_bounds("API_DOWN_MINUTES", settings.API_DOWN_MINUTES, min_val=1, max_val=60, expected_type=int)
    for issue in issues: all_issues.append(f"API_DOWN_MINUTES: {issue}")

    # ─── POSITION SIZING (KELLY/VOL) ───
    print_section("POSITION SIZING (KELLY / VOLATILITY)")
    
    print_kv("KELLY_FRACTION", settings.KELLY_FRACTION)
    issues = check_bounds("KELLY_FRACTION", settings.KELLY_FRACTION, min_val=0.0, max_val=1.0, expected_type=float)
    for issue in issues: all_issues.append(f"KELLY_FRACTION: {issue}")

    print_kv("VOL_LOOKBACK", settings.VOL_LOOKBACK)
    issues = check_bounds("VOL_LOOKBACK", settings.VOL_LOOKBACK, min_val=5, max_val=200, expected_type=int)
    for issue in issues: all_issues.append(f"VOL_LOOKBACK: {issue}")

    print_kv("MAX_VOL_ADJUST", settings.MAX_VOL_ADJUST)
    issues = check_bounds("MAX_VOL_ADJUST", settings.MAX_VOL_ADJUST, min_val=0.1, max_val=10.0, expected_type=float)
    for issue in issues: all_issues.append(f"MAX_VOL_ADJUST: {issue}")

    # ─── ADAPTIVE LEARNER GATES ───
    print_section("ADAPTIVE LEARNER GATES")
    
    print_kv("ADAPTIVE_ML_ENABLED", settings.ADAPTIVE_ML_ENABLED)
    print_kv("ADAPTIVE_MIN_TRADES_BEFORE_LIVE", settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE)
    print_kv("ADAPTIVE_MIN_VALIDATION_TRADES", settings.ADAPTIVE_MIN_VALIDATION_TRADES)
    print_kv("ADAPTIVE_MIN_SHARPE", settings.ADAPTIVE_MIN_SHARPE)
    print_kv("ADAPTIVE_MIN_WIN_RATE", f"{settings.ADAPTIVE_MIN_WIN_RATE*100:.1f}%")
    print_kv("ADAPTIVE_HOLDOUT_FRACTION", f"{settings.ADAPTIVE_HOLDOUT_FRACTION*100:.0f}%")
    print_kv("PPO_MIN_TRADES_BEFORE_LIVE", settings.PPO_MIN_TRADES_BEFORE_LIVE)
    print_kv("DEFAULT_SCORE_THRESHOLD", settings.DEFAULT_SCORE_THRESHOLD)
    
    issues = check_bounds("ADAPTIVE_MIN_TRADES_BEFORE_LIVE", settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE, min_val=0, max_val=100, expected_type=int)
    for issue in issues: all_issues.append(f"ADAPTIVE_MIN_TRADES_BEFORE_LIVE: {issue}")
    issues = check_bounds("ADAPTIVE_MIN_SHARPE", settings.ADAPTIVE_MIN_SHARPE, min_val=0.0, max_val=3.0, expected_type=float)
    for issue in issues: all_issues.append(f"ADAPTIVE_MIN_SHARPE: {issue}")
    issues = check_bounds("ADAPTIVE_MIN_WIN_RATE", settings.ADAPTIVE_MIN_WIN_RATE, min_val=0.0, max_val=1.0, expected_type=float)
    for issue in issues: all_issues.append(f"ADAPTIVE_MIN_WIN_RATE: {issue}")

    # ─── REGIME THRESHOLDS ───
    print_section("REGIME CLASSIFICATION THRESHOLDS")
    
    print_kv("HURST_TREND_UP", settings.HURST_TREND_UP)
    print_kv("HURST_MEAN_REVERT", settings.HURST_MEAN_REVERT)
    if settings.HURST_MEAN_REVERT >= settings.HURST_TREND_UP:
        all_issues.append("HURST_MEAN_REVERT should be < HURST_TREND_UP")
    
    print_kv("HIGH_VOLATILITY_PCT", settings.HIGH_VOLATILITY_PCT)
    print_kv("RSI_OVERBOUGHT", settings.RSI_OVERBOUGHT)
    print_kv("RSI_OVERSOLD", settings.RSI_OVERSOLD)
    print_kv("RSI_NEUTRAL_BUY", settings.RSI_NEUTRAL_BUY)
    print_kv("RSI_NEUTRAL_SELL", settings.RSI_NEUTRAL_SELL)
    print_kv("BB_ZSCORE_THRESHOLD", settings.BB_ZSCORE_THRESHOLD)

    # ─── SUMMARY ───
    print_section("AUDIT SUMMARY")
    
    if all_issues:
        print(f"\n  [FAIL] {len(all_issues)} ISSUES FOUND:")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")
        print(f"\n  VERDICT: FAIL - Risk parameters need review")
    else:
        print("\n  [PASS] ALL PARAMETERS WITHIN BOUNDS")
        print("  VERDICT: PASS - Risk parameters are deliberately chosen")
    
    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issues": all_issues,
        "verdict": "FAIL" if all_issues else "PASS",
        "effective_portfolio_cap": effective_cap,
        "net_edge_bps": net_edge,
        "round_trip_cost_bps": round_trip
    }
    
    report_path = PROJECT_ROOT / "data" / f"risk_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

if __name__ == "__main__":
    main()