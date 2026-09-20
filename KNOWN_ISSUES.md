# Known Issues / Follow-Up Work

Last updated: 2026-07-26

## Stale replay dataset (data/historical_experiences.jsonl) -- UPDATE 2026-09-20

The original corruption (raw-OHLCV scaler, commit 2b91681 fix) is resolved in
scripts/train_foundation_model.py, which now uses get_active_features().

However: the dataset WAS regenerated (commit f30eb2f, 2026-08-16), but that
same commit introduced scripts/generate_replay_dataset.py's REPLAY_FAST_MODE,
which defaulted ON at the time. Fast mode fills every record's tensor_state
with seeded `np.random.RandomState(seed).randn(128)` -- random noise, not
real feature vectors -- via a monkey-patched transformer_brain. The commit
message doesn't mention setting REPLAY_FAST_MODE=0, so the currently-committed
data/historical_experiences.jsonl was very likely generated with noise tensors,
not the corrected real features.

REPLAY_FAST_MODE's default was flipped to 0 (real model) on 2026-09-20 so this
can't recur silently. A second scaler-fit-before-split leakage bug was also
found and fixed the same day in scripts/train_foundation_model.py (StandardScaler
was fit on the full train+val concatenation before splitting; now fit on train
only, split per-symbol before concatenation).

Action needed (unchanged, now more urgent): regenerate
data/historical_experiences.jsonl by re-running
scripts/generate_replay_dataset.py (now defaults to the real model) before
using it for any future fine-tuning via scripts/retrain_transformer.py. This
requires real market data access and was NOT run as part of this fix pass.
Verify with scan_learning.py that tensor_state values no longer look like
N(0,1) noise before trusting the regenerated file.

## Real walk-forward validation is narrow in scope

commit 4fe36ed added real (non-fake) walk-forward validation to the model
promotion gate in scripts/retrain_transformer.py, replacing a previous
version that used random.uniform() to fake results. Current validation only
tests a single symbol (BTC-USD) over a single 90-day window. Consider testing
across multiple symbols and time windows before fully trusting promotions.

## PPO meta-learner has no real training data

models/ppo_meta_weights.zip loads successfully, but as of 2026-07-26 there
are zero real closed trades in the database. The PPO model has not been
trained on any real trade outcomes. It is bypassed during backtesting
(commit 4324e3a) but NOT during live trading - it will begin influencing
decisions once ADAPTIVE_MIN_TRADES_BEFORE_LIVE real trades accumulate.

## Fee/slippage fix not yet validated against live trading

commit 4fe36ed added real fee and slippage modeling to src/backtest.py's
run_backtest() (previously fee_pct/slippage_pct were accepted but never
applied to any P&L calculation). Validated in the backtest context but not
cross-checked against actual realized fees/slippage from live paper fills.

## Portfolio exposure cap fix - deployed but not stress-tested

commit 1052393 fixed a race condition and added automatic corrective action
for sustained cap breaches. Deployed but not yet observed handling an actual
multi-symbol simultaneous-signal scenario in production. Monitor logs for
"Exposure reservation denied" and "Closed {symbol} to reduce exposure".

## Repo root debris (partially cleaned)

commit bc26d2b removed 18 stale/debug files from the repo root. Worth a
periodic re-check (git ls-files at repo root) since ad-hoc debugging on this
project has repeatedly left behind similar artifacts.
