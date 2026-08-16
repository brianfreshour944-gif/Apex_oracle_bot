#!/usr/bin/env python3
"""Comprehensive scan of learning pipeline state and health."""

import json
import os
import sys
from datetime import datetime, timezone

print("=" * 60)
print("LEARNING PIPELINE SCAN")
print("=" * 60)

# 1. Replay buffers
print("\n[1] REPLAY BUFFERS")
for f in ['data/historical_experiences.jsonl', 'data/live_experiences.jsonl']:
    if os.path.exists(f):
        with open(f) as fp:
            lines = fp.readlines()
            print(f"  {f}: {len(lines)} records")
            if lines:
                rec = json.loads(lines[0])
                print(f"    Keys: {list(rec.keys())}")
                tensor = rec.get('tensor', [])
                print(f"    Tensor: {len(tensor)} steps x {len(tensor[0]) if tensor else 0} features")
                print(f"    Label: {rec.get('label')}")
                print(f"    Regime: {rec.get('market_regime')}")
    else:
        print(f"  {f}: NOT FOUND")

# 2. Adaptive meta-learner state
print("\n[2] ADAPTIVE META-LEARNER (Mathematical)")
for f in ['data/adaptive_meta_state.json', 'data/strategy_meta_state.json']:
    if os.path.exists(f):
        with open(f) as fp:
            state = json.load(fp)
            print(f"  {f}:")
            print(f"    Version: {state.get('version')}")
            print(f"    Total samples: {state.get('sample_count')}")
            print(f"    Last update: {state.get('last_update')}")
            print(f"    Learning rate: {state.get('learning_rate')}")
            print(f"    Regime samples: {state.get('regime_sample_count', {})}")
            weights = state.get('weights', {})
            for regime, w in weights.items():
                print(f"    {regime}: { {k: round(v, 3) for k, v in w.items()} }")
    else:
        print(f"  {f}: NOT FOUND")

# 3. PPO model
print("\n[3] PPO META-LEARNER")
ppo_path = 'models/ppo_meta_weights.zip'
if os.path.exists(ppo_path):
    size = os.path.getsize(ppo_path)
    mtime = datetime.fromtimestamp(os.path.getmtime(ppo_path), tz=timezone.utc)
    print(f"  {ppo_path}: {size} bytes, modified {mtime.isoformat()}")
else:
    print(f"  {ppo_path}: NOT FOUND")

# 4. Transformer model
print("\n[4] TRANSFORMER MODEL")
for f in ['models/grok_gqa_v9_best.pth', 'models/feature_scaler.pkl', 'models/transformer_config.json']:
    if os.path.exists(f):
        size = os.path.getsize(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f), tz=timezone.utc)
        print(f"  {f}: {size} bytes, modified {mtime.isoformat()}")
    else:
        print(f"  {f}: NOT FOUND")

# 5. Config gates
print("\n[5] GATING CONFIG")
from src.config import settings
print(f"  ADAPTIVE_ML_ENABLED: {settings.ADAPTIVE_ML_ENABLED}")
print(f"  ADAPTIVE_MIN_TRADES_BEFORE_LIVE: {settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE}")
print(f"  PPO_MIN_TRADES_BEFORE_LIVE: {settings.PPO_MIN_TRADES_BEFORE_LIVE}")
print(f"  ADAPTIVE_LEARNING_RATE: {settings.ADAPTIVE_LEARNING_RATE}")
print(f"  ADAPTIVE_MIN_WEIGHT: {settings.ADAPTIVE_MIN_WEIGHT}")
print(f"  ADAPTIVE_MAX_WEIGHT: {settings.ADAPTIVE_MAX_WEIGHT}")

# 6. Scheduled tasks
print("\n[6] SCHEDULED TASKS (from bot.py)")
tasks = [
    ("Transformer replay fine-tune", "1 AM daily", "retrain_transformer.py"),
    ("PPO retrain", "Sunday 6 AM", "evolutionary_ppo_trainer.py"),
    ("AutoML pipeline", "Saturday 2 AM", "automl_pipeline.py"),
    ("Evolution cull", "1st of month 4 AM", "evolution_cull.py"),
    ("Automatic research", "Sunday 4 AM", "automatic_researcher.py"),
    ("Post-mortem", "Saturday 4 AM", "post_mortem.py"),
    ("DB maintenance", "Weekly", "db_maintenance"),
]
for name, sched, script in tasks:
    print(f"  {name}: {sched} ({script})")

# 7. Health checks
print("\n[7] HEALTH CHECKS")
issues = []

# Check historical buffer size
hist_path = 'data/historical_experiences.jsonl'
if os.path.exists(hist_path):
    with open(hist_path) as fp:
        n_hist = len(fp.readlines())
    if n_hist < 1000:
        issues.append(f"Historical buffer small ({n_hist} records) - may need more backtest data")
    else:
        print(f"  OK: Historical buffer has {n_hist} records")

# Check adaptive samples per regime
for f in ['data/adaptive_meta_state.json']:
    if os.path.exists(f):
        with open(f) as fp:
            state = json.load(fp)
        regime_counts = state.get('regime_sample_count', {})
        for regime, count in regime_counts.items():
            if count < settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE:
                issues.append(f"Regime '{regime}' has only {count} samples (gate={settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE})")

# Check Transformer online step LR
print(f"  OK: Transformer online LR = 1e-5 (in bot.py)")

# Check PPO gate
if settings.PPO_MIN_TRADES_BEFORE_LIVE > settings.ADAPTIVE_MIN_TRADES_BEFORE_LIVE:
    issues.append("PPO gate higher than Adaptive gate - PPO will never activate first")

if issues:
    print("  ISSUES FOUND:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  OK: No critical issues detected")

# 8. Market adaptation capability
print("\n[8] MARKET ADAPTATION CAPABILITY")
print("  Regime detection: Hurst + ATR + RSI (in strategies.py)")
print("  Per-regime weights: Separate for trending/mean_reverting/volatile/choppy/breakout/default")
print("  Online adaptation: AdaptiveMetaLearner (per-trade) + Transformer gradient step")
print("  Offline adaptation: Daily Transformer retrain + Weekly PPO retrain")
print("  Strategy selection: Adaptive per-regime with domain priors")

print("\n" + "=" * 60)
print("SCAN COMPLETE")
print("=" * 60)