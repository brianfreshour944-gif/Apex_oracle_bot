#!/usr/bin/env python3
"""
Active Configuration Dumper — Step 8 of Foundation Hardening.

Reads the *actual* current config from settings, committee, risk, adaptive learner,
and external flag files, and prints a single authoritative "what's really on"
summary. No docstrings, no stale comments — just live truth.

Run: python scripts/dump_active_config.py
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.committee.committee import get_meta_learner, REGIME_WEIGHT_MATRIX, DEFAULT_SCORE_THRESHOLD
from src.risk import RiskManager
from src.bot import _state, read_regime_flag, get_banned_symbols


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_kv(key: str, value: any, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def main():
    print(f"\n[ACTIVE CONFIGURATION SNAPSHOT] — {datetime.now(timezone.utc).isoformat()}")
    print(f"Project root: {PROJECT_ROOT}")

    # ─── CORE SETTINGS ───
    print_section("CORE SETTINGS (src.config.settings)")
    print_kv("Bot Name", settings.BOT_NAME)
    print_kv("Exchange", "Alpaca Crypto" + (" Paper" if "paper-api" in settings.ALPACA_BASE_URL else " Live"))
    print_kv("Database URL", settings.DATABASE_URL)
    print_kv("Symbols", settings.SYMBOLS)
    print_kv("Quote Currency", settings.QUOTE_CURRENCY)
    print_kv("Account Base", f"${settings.ACCOUNT_BASE:,.2f}")

    # ─── RISK PARAMETERS ───
    print_section("RISK PARAMETERS (authoritative — risk.py reads these)")
    print_kv("Base Risk %", f"{settings.BASE_RISK_PERCENT*100:.2f}%")
    print_kv("Max Single Trade USD", f"${settings.MAX_SINGLE_TRADE_USD:,.2f}")
    print_kv("Max Portfolio Value (static cap)", f"${settings.MAX_PORTFOLIO_VALUE:,.2f}")
    print_kv("Max Portfolio % of Account Base", f"{settings.MAX_PORTFOLIO_PCT*100:.0f}%")
    effective_cap = settings.ACCOUNT_BASE * settings.MAX_PORTFOLIO_PCT
    print_kv("  -> Effective Cap (ACCOUNT_BASE * MAX_PORTFOLIO_PCT)", f"${effective_cap:,.2f}")
    print_kv("Max Open Positions", settings.MAX_OPEN_POSITIONS)
    print_kv("Max Drawdown Stop", f"{settings.MAX_DRAWDOWN_STOP}%")
    print_kv("Daily Loss Limit", f"{settings.DAILY_LOSS_LIMIT}%")
    print_kv("Profit Target", f"{settings.PROFIT_TARGET_PCT*100:.2f}%")
    print_kv("Stop Loss", f"{settings.STOP_LOSS_PCT*100:.2f}%")
    print_kv("ATR Stop Multiplier", settings.ATR_STOP_MULTIPLIER)
    print_kv("Max Hold Hours", settings.MAX_HOLD_HOURS)
    print_kv("Cooldown Seconds Buy", settings.COOLDOWN_SECONDS_BUY)
    print_kv("Dust Value USD", f"${settings.DUST_VALUE_USD:.2f}")

    # Position pyramid / scale-in gates
    print_kv("Max Position Adds", settings.MAX_POSITION_ADDS)
    print_kv("Position Add Min Seconds", settings.POSITION_ADD_MIN_SECONDS)
    print_kv("Position Add Min Score Increase", settings.POSITION_ADD_MIN_SCORE_INCREASE)
    print_kv("Position Add Size Decay", settings.POSITION_ADD_SIZE_DECAY)

    # Transaction costs
    print_kv("TX Cost Fee BPS", settings.TX_COST_FEE_BPS)
    print_kv("TX Cost Slippage BPS", settings.TX_COST_SLIPPAGE_BPS)
    print_kv("TX Cost Spread BPS", settings.TX_COST_SPREAD_BPS)
    print_kv("TX Cost Min Edge BPS", settings.TX_COST_MIN_EDGE_BPS)
    print_kv("TX Cost Use Dynamic", settings.TX_COST_USE_DYNAMIC)

    # Trailing stop
    print_kv("Trailing Stop Enabled", settings.TRAILING_STOP_ENABLED)
    print_kv("Trailing Activation %", f"{settings.TRAILING_ACTIVATION_PCT*100:.2f}%")
    print_kv("Trailing Distance %", f"{settings.TRAILING_DISTANCE_PCT*100:.2f}%")

    # Circuit breaker
    print_kv("Circuit Failure Threshold", settings.CIRCUIT_FAILURE_THRESHOLD)
    print_kv("Circuit Open Seconds", settings.CIRCUIT_OPEN_SECONDS)

    # Killswitches
    print_kv("Volatility Spike %", settings.VOLATILITY_SPIKE_PCT)
    print_kv("API Down Minutes", settings.API_DOWN_MINUTES)

    # Loop
    print_kv("Loop Interval Sec", settings.LOOP_INTERVAL_SEC)
    print_kv("Status Port", settings.STATUS_PORT)

    # ─── COMMITTEE / ADAPTIVE LAYER ───
    print_section("COMMITTEE & ADAPTIVE LAYER")

    # Regime weight matrix (what actually drives vote weighting)
    print_kv("Default Score Threshold", DEFAULT_SCORE_THRESHOLD)
    print("\n  Regime Weight Matrix (active in committee.run_committee):")
    for regime, weights in REGIME_WEIGHT_MATRIX.items():
        w_str = " ".join(f"{k}={v:.2f}" for k, v in weights.items())
        print(f"    {regime}: {w_str}")

    # Adaptive meta-learner state
    learner = get_meta_learner()
    if learner:
        print_kv("Adaptive ML Enabled (settings.ADAPTIVE_ML_ENABLED)", settings.ADAPTIVE_ML_ENABLED)
        print_kv("Adaptive Learner Loaded", True)
        print_kv("  State Path", learner.state_path)
        print_kv("  Learning Rate", learner.learning_rate)
        print_kv("  Min/Max Weight", f"{learner.min_weight:.2f} / {learner.max_weight:.2f}")
        print_kv("  Total Sample Count", learner.sample_count)
        print_kv("  Min Trades Before Live", settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE)
        print_kv("  Min Validation Trades", settings.ADAPTIVE_MIN_VALIDATION_TRADES)
        print_kv("  Min Sharpe for Validation", settings.ADAPTIVE_MIN_SHARPE)
        print_kv("  Min Win Rate for Validation", settings.ADAPTIVE_MIN_WIN_RATE)
        print_kv("  Holdout Fraction", settings.ADAPTIVE_HOLDOUT_FRACTION)

        print("\n  Per-Regime Status:")
        for regime in learner.regime_sample_count.keys():
            n = learner.regime_sample_count.get(regime, 0)
            validated = learner.regime_validated.get(regime, False)
            weights = learner.weights.get(regime, {})
            w_str = " ".join(f"{k}={v:.2f}" for k, v in weights.items())
            metrics = learner.get_regime_validation_metrics(regime)
            print(f"    {regime}: samples={n}, validated={validated}, weights=[{w_str}], sharpe={metrics['sharpe']:.3f}, win_rate={metrics['win_rate']:.3f}")
        
        # Regimes with zero samples
        all_regimes = set(REGIME_WEIGHT_MATRIX.keys()) | set(learner.regime_sample_count.keys())
        for regime in sorted(all_regimes):
            if regime not in learner.regime_sample_count:
                weights = learner.weights.get(regime, {})
                w_str = " ".join(f"{k}={v:.2f}" for k, v in weights.items()) if weights else "EQUAL (cold start)"
                print(f"    {regime}: samples=0, validated=False, weights=[{w_str}]")

        # PPO gate
        print_kv("PPO Min Trades Before Live", settings.PPO_MIN_TRADES_BEFORE_LIVE)
    else:
        print_kv("Adaptive ML Enabled", settings.ADAPTIVE_ML_ENABLED)
        print_kv("Adaptive Learner Loaded", False)

    # ─── EXTERNAL FLAG FILES ───
    print_section("EXTERNAL FLAG FILES (read at runtime)")

    # Regime flag
    regime_flag = read_regime_flag()
    print_kv("Regime Flag File", "data/regime_flag.txt")
    print_kv("  pause_grok", regime_flag.get("pause_grok", False))
    print_kv("  pause_oracle", regime_flag.get("pause_oracle", False))
    print_kv("  grok_multiplier", regime_flag.get("grok_multiplier", 1.0))
    print_kv("  oracle_multiplier", regime_flag.get("oracle_multiplier", 1.0))
    print_kv("  regime", regime_flag.get("regime", "normal"))

    # Banned symbols
    banned = get_banned_symbols()
    print_kv("Banned Symbols File", "data/banned_symbols.json")
    print_kv("  Banned Count", len(banned))
    if banned:
        print_kv("  Symbols", ", ".join(sorted(banned)))

    # Adaptive thresholds (weekly analyzer output)
    thresholds_path = PROJECT_ROOT / "data" / "adaptive_thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        print_kv("Adaptive Thresholds File", str(thresholds_path))
        print_kv("  Symbols with Custom Thresholds", len(thresholds))
        for sym, thresh in sorted(thresholds.items()):
            print(f"    {sym}: {thresh:.3f}")
    else:
        print_kv("Adaptive Thresholds File", "NOT FOUND (using DEFAULT_SCORE_THRESHOLD)")

    # ─── MODEL PATHS ───
    print_section("MODEL PATHS")
    print_kv("Transformer Model", settings.TRANSFORMER_MODEL_PATH)
    print_kv("Transformer Scaler", settings.TRANSFORMER_SCALER_PATH)
    print_kv("Adaptive State Path", settings.ADAPTIVE_STATE_PATH)

    model_path = PROJECT_ROOT / settings.TRANSFORMER_MODEL_PATH
    scaler_path = PROJECT_ROOT / settings.TRANSFORMER_SCALER_PATH
    adaptive_state_path = PROJECT_ROOT / settings.ADAPTIVE_STATE_PATH
    print_kv("  Transformer Model Exists", model_path.exists())
    print_kv("  Transformer Scaler Exists", scaler_path.exists())
    print_kv("  Adaptive State Exists", adaptive_state_path.exists())

    # ─── OOD DISCRIMINATOR ───
    print_section("OOD DISCRIMINATOR")
    try:
        from src.ood_discriminator import get_ood_discriminator
        ood = get_ood_discriminator()
        if ood is not None:
            print_kv("Loaded", True)
            print_kv("  Is Trained", ood._is_trained)
            print_kv("  Threshold", getattr(ood, "threshold", "N/A"))
        else:
            print_kv("Loaded", False)
    except Exception as e:
        print_kv("Error", str(e))

    # ─── TELEGRAM / DASHBOARD ───
    print_section("NOTIFICATIONS & DASHBOARDS")
    print_kv("Telegram Configured", bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID))
    print_kv("Base44 Dashboard Configured", bool(settings.DASHBOARD_API_KEY))
    print_kv("Prometheus Enabled", settings.ENABLE_PROMETHEUS)

    # ─── FEATURE ENGINEERING ───
    print_section("FEATURE ENGINEERING")
    print_kv("Timeframes", settings.TIMEFRAMES)
    print_kv("Base Timeframe", settings.FEATURE_BASE_TIMEFRAME)
    print_kv("Lookback Bars", settings.FEATURE_LOOKBACK_BARS)
    print_kv("Cross Asset List", settings.CROSS_ASSET_LIST)
    print_kv("Cross Asset Lookback", settings.CROSS_ASSET_LOOKBACK)
    print_kv("Cross Asset Max Lag", settings.CROSS_ASSET_MAX_LAG)
    print_kv("Cross Asset Method", settings.CROSS_ASSET_METHOD)

    # ─── REGIME THRESHOLDS ───
    print_section("REGIME CLASSIFICATION THRESHOLDS")
    print_kv("Hurst Trend Up", settings.HURST_TREND_UP)
    print_kv("Hurst Mean Revert", settings.HURST_MEAN_REVERT)
    print_kv("High Volatility %", settings.HIGH_VOLATILITY_PCT)
    print_kv("RSI Overbought", settings.RSI_OVERBOUGHT)
    print_kv("RSI Oversold", settings.RSI_OVERSOLD)
    print_kv("RSI Neutral Buy", settings.RSI_NEUTRAL_BUY)
    print_kv("RSI Neutral Sell", settings.RSI_NEUTRAL_SELL)
    print_kv("BB Z-Score Threshold", settings.BB_ZSCORE_THRESHOLD)

    # ─── POSITION SIZING (KELLY/VOL) ───
    print_section("POSITION SIZING (Kelly / Volatility)")
    print_kv("Kelly Fraction", settings.KELLY_FRACTION)
    print_kv("Vol Lookback", settings.VOL_LOOKBACK)
    print_kv("Max Vol Adjust", settings.MAX_VOL_ADJUST)

    # ─── ALERTS ───
    print_section("ALERTS")
    print_kv("Alert Email", settings.ALERT_EMAIL or "NOT CONFIGURED")
    print_kv("Alert Cooldown Sec", settings.ALERT_COOLDOWN_SEC)

    print(f"\n{'='*60}")
    print("  END OF ACTIVE CONFIG SNAPSHOT")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()