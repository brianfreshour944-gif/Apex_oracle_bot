# Known Issues / Follow-Up Work

Last updated: 2026-07-26

## Stale replay dataset (data/historical_experiences.jsonl)

The 2,782-record replay dataset was generated (commit 4fe36ed) using a
foundation model whose feature_scaler.pkl was fit on the WRONG feature set
(raw OHLCV prices instead of the 11 institutional features actually used at
inference time - see commit 2b91681 for the fix). Every tensor_state in that
dataset was computed with this broken scaler, so the values are corrupted.

Action needed: regenerate data/historical_experiences.jsonl by re-running
scripts/generate_replay_dataset.py before using it for any future fine-tuning
via scripts/retrain_transformer.py.

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
