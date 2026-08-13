#!/usr/bin/env python3
"""
Promote backtest-learned brain weights into the live trading state.

WHY THIS EXISTS
----------------
run_adaptive_backtest.py deliberately writes to a separate sandbox state
file (data/sandbox_meta_state.json), NOT the live trading state
(data/adaptive_meta_state.json) that run_committee() actually reads. This
is intentional: a single backtest run should never silently overwrite
weeks of live-learned weights.

But that means backtest-derived learning previously had no path into live
trading at all - the sandbox was a dead end. This script is that path: an
explicit, reviewable, opt-in promotion step, never automatic.

WHAT IT DOES
------------
For each regime present in the backtest (sandbox) state, blends it into
the live state using a sample-count-weighted average:

    new_weight = (live_weight * live_samples + backtest_weight * backtest_samples)
                 / (live_samples + backtest_samples)

This means a small/thin backtest run can't wipe out weeks of live-learned
weights, and a very large, decisive backtest run CAN meaningfully move
weights - proportional to how much evidence it actually represents. Sample
counts are summed (not replaced), so future live updates keep building on
the correct total. Regimes not present in the backtest state are left
untouched.

USAGE
-----
    # Dry run (default): shows the diff, writes nothing.
    python scripts/promote_backtest_learning.py

    # Actually write the merged result to live state.
    python scripts/promote_backtest_learning.py --confirm

    # Use different state files.
    python scripts/promote_backtest_learning.py --live data/adaptive_meta_state.json \
        --backtest data/sandbox_meta_state.json --confirm

A timestamped backup of the live state is written before any overwrite.
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.committee.adaptive_meta import AdaptiveMetaLearner, BRAINS
from src.logging_config import get_logger

logger = get_logger("promote_backtest_learning")


def merge_weights(live_learner: AdaptiveMetaLearner, backtest_learner: AdaptiveMetaLearner) -> dict:
    """Return {regime: {brain: new_weight}} for every regime in the backtest state,
    sample-count-weighted against the live state. Does not mutate either learner."""
    merged = {}
    for regime, bt_weights in backtest_learner.weights.items():
        bt_samples = backtest_learner.sample_count_for_regime(regime)
        live_weights = live_learner.weights.get(regime, {b: 1.0 / len(BRAINS) for b in BRAINS})
        live_samples = live_learner.sample_count_for_regime(regime)

        total_samples = live_samples + bt_samples
        if total_samples == 0:
            merged[regime] = dict(bt_weights)
            continue

        blended = {}
        for brain in BRAINS:
            lw = live_weights.get(brain, 1.0 / len(BRAINS))
            bw = bt_weights.get(brain, 1.0 / len(BRAINS))
            blended[brain] = (lw * live_samples + bw * bt_samples) / total_samples

        total = sum(blended.values())
        if total > 0:
            blended = {b: w / total for b, w in blended.items()}
        merged[regime] = blended

    return merged


def print_diff(live_learner: AdaptiveMetaLearner, merged: dict) -> None:
    print("\n=== Proposed changes (live state -> merged) ===")
    for regime, new_weights in merged.items():
        old_weights = live_learner.weights.get(regime, {})
        old_n = live_learner.sample_count_for_regime(regime)
        print(f"\nRegime: {regime}  (live samples: {old_n})")
        for brain in BRAINS:
            old_w = old_weights.get(brain, 1.0 / len(BRAINS))
            new_w = new_weights.get(brain, 0.0)
            delta = new_w - old_w
            arrow = "->" if abs(delta) > 0.005 else "=="
            print(f"  {brain:12s} {old_w:.3f} {arrow} {new_w:.3f}  ({delta:+.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", default="data/adaptive_meta_state.json",
                         help="Path to the live trading state (default: data/adaptive_meta_state.json)")
    parser.add_argument("--backtest", default="data/sandbox_meta_state.json",
                         help="Path to the backtest sandbox state (default: data/sandbox_meta_state.json)")
    parser.add_argument("--confirm", action="store_true",
                         help="Actually write the merged result. Without this flag, only prints the diff.")
    args = parser.parse_args()

    if not os.path.exists(args.backtest):
        logger.error(f"Backtest state file not found: {args.backtest}. "
                      f"Run run_adaptive_backtest.py first to generate it.")
        sys.exit(1)

    live_learner = AdaptiveMetaLearner(state_path=args.live)
    backtest_learner = AdaptiveMetaLearner(state_path=args.backtest)

    if not backtest_learner.weights:
        logger.error(f"Backtest state at {args.backtest} has no learned weights. Nothing to promote.")
        sys.exit(1)

    merged = merge_weights(live_learner, backtest_learner)
    print_diff(live_learner, merged)

    if not args.confirm:
        print("\nDry run only - no changes written. Re-run with --confirm to apply.")
        return

    if os.path.exists(args.live):
        backup_path = f"{args.live}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(args.live, backup_path)
        logger.info(f"Backed up live state to {backup_path}")

    for regime, new_weights in merged.items():
        live_learner.weights[regime] = new_weights
        bt_samples = backtest_learner.sample_count_for_regime(regime)
        live_learner.regime_sample_count[regime] = (
            live_learner.regime_sample_count.get(regime, 0) + bt_samples
        )
        live_learner.sample_count += bt_samples

    live_learner.save(args.live)
    logger.info(f"Promoted backtest learning into {args.live}")
    print(f"\n✅ Live state updated: {args.live}")


if __name__ == "__main__":
    main()