# PREP — CAPTURE ENABLEMENT (Option 3: validate, then decide) — READ-ONLY

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push, or
deploy. `CAPTURE_SUBMISSION_QUOTE` remains **`False`**.

Authorized by "Option 3: authorize preparation for capture enablement, then
validate before changing the default." This prepares and validates. It does not
change the default.

Supersedes the "enable" section of `CAPTURE_DEFAULT_DECISION_PREP_2026-10-25.md`,
which predated the read-path commit `89e83a4`.

---

## 1. Headline: enablement needs NO code change

`settings` reads `CAPTURE_SUBMISSION_QUOTE` from the environment with
`env_prefix=""` and `case_sensitive=True` (`src/config.py:31-36`). The field
default is `False`, but the **environment overrides it**.

Verified empirically (dummy creds, no network):

```
env CAPTURE_SUBMISSION_QUOTE=true  -> True
env (unset)                        -> False   (field default)
env CAPTURE_SUBMISSION_QUOTE=false -> False
```

**Consequence for Option 3:** the default can stay `False` in source *and*
capture can still be turned on per-deployment via the environment. There is no
need to change committed code to collect quotes. That is the lowest-risk form of
enablement — it is reversible by unsetting a variable, and it leaves the
committed default conservative.

## 2. The two enablement paths

| | A. Env var (recommended) | B. Flip the committed default |
| --- | --- | --- |
| Change | add `CAPTURE_SUBMISSION_QUOTE=true` to the deployment env | `src/config.py:428` `default=False` → `True` |
| Code diff | **none** | 1 line + test update |
| Affects | that one deployment | **every** deployment/checkout |
| Revert | unset the var | another commit |
| Fits Option 3 | **yes** — default stays conservative | no — that IS changing the default |

Option 3 explicitly says "validate before changing the default". Path A satisfies
it exactly: it enables measurement without changing the default.

## 3. Blast radius if enabled (unchanged by isolation)

Per order submission, `create_order` (`src/exchange.py:888`) performs one extra
`get_latest_quote` round-trip. Verified properties at HEAD `89e83a4`:

| Property | Status | Evidence |
| --- | --- | --- |
| Gated | yes | `if getattr(settings, "CAPTURE_SUBMISSION_QUOTE", False)` |
| Failure-isolated | yes | `except Exception` → `None`; never raises |
| Time-bounded | yes | `asyncio.wait_for(..., _QUOTE_CAPTURE_TIMEOUT_SEC=2.0)` |
| Worker-isolated | yes | dedicated 2-worker `quote-capture` pool |
| Read by a decision | **no** | measurement only; only `_persist_order_record` consumes it |
| Read by tooling | **yes, now** | analyzer slippage section (commit `89e83a4`) |

Residual risk that isolation does **not** remove:

- **Shared API-key rate-limit quota** — the data request uses the same Alpaca
  credentials; `_rate_limit_remaining` is inert (assigned `exchange.py:231`,
  never read).
- Shared loop / process / host (inherent).
- `create_order`'s own unbounded socket call (out of scope).

At the observed turnover (24 closed trades over ~22 days) this is negligible in
throughput terms, but non-zero and invisible to the bot.

## 4. Validation performed (read-only)

**4.1 Env-override mechanism — PROVEN.** Three subprocess runs above. The first
attempt also surfaced the required-credentials guard, which incidentally proved
the env var is read (`input_value={'CAPTURE_SUBMISSION_QUOTE': 'true'}`).

**4.2 Capture path is covered end-to-end at capture=True.** `tests/test_submission_quote_capture.py`
has 18 tests, including:

- `test_capture_enabled_attaches_quote` — runs `create_order` with the flag on
  and asserts the quote is attached (line 279).
- `test_capture_disabled_skips_quote` (246), `test_quote_failure_does_not_block_order` (308),
  `test_hung_quote_does_not_starve_submit_order` (357),
  `test_wait_for_releases_caller_on_slow_quote` (399),
  `test_close_shuts_down_quote_executor` (431),
  `test_capture_runs_off_default_executor` (448).

So the enabled path is exercised without touching production credentials.

**4.3 Read path is covered.** 38 analyzer tests (commit `89e83a4`), including the
sign convention, the spread decomposition, every exclusion bucket, and the
legacy-schema path.

**4.4 Persistence/migration covered.** `test_migration_adds_quote_columns_to_legacy_orders`,
`test_missing_quote_persists_null_not_zero`, `test_partial_quote_persists_only_present_fields`.

**4.5 Deployment env files present?** No `.env` in this checkout (only
`.env.example`, which does **not** mention the flag). No `docker-compose.yml` in
this checkout. So the deployment's actual env is **not** visible from here — the
operator must confirm where it sets variables.

## 5. What is NOT validated (honest limits)

- **No real quote data has ever been captured** — the flag has been `False`, and
  the local `data/bot.db` has 0 orders. So the slippage *output* is unvalidated
  against real rows. Only the logic is tested.
- **The production `.env` / compose is not in this repo.** Enabling requires the
  operator to edit the deployment's env, which is outside this repository and
  outside my access.
- **A single symbol's data endpoint** (`CryptoLatestQuoteRequest`) is assumed
  available; no live call was made (and must not be).
- **24 trades is a tiny sample.** Even with clean slippage data, conclusions about
  execution cost will be low-confidence until more orders accumulate.

## 6. Recommended sequence (Option 3, concretely)

1. (this) prep + validation — done, read-only.
2. Operator enables capture **in the deployment env** (Path A) — no code change,
   no commit, reversible.
3. Let the bot run until a meaningful number of orders carry quotes.
4. Pull a fresh snapshot and run the analyzer; inspect the SLIPPAGE section,
   including the exclusion counts.
5. **Only then** decide whether to flip the committed default (Path B) — and that
   decision would be based on real data rather than a design argument.

## 7. Enablement checklist (for a later authorization)

- [ ] Confirm the deployment's env mechanism (compose `environment:` / `.env` /
      platform vars) — **not visible in this repo**.
- [ ] Set `CAPTURE_SUBMISSION_QUOTE=true` there.
- [ ] Optionally document the flag in `.env.example` (a separate, additive change).
- [ ] Restart/apply per the deployment's normal procedure — **not authorized here**.
- [ ] After orders accumulate, pull a snapshot and run:
      `python scripts/analyze_deployed_trades.py <snapshot> --json out.json`
- [ ] Inspect the SLIPPAGE section + exclusion counts before drawing conclusions.

## 8. Rollback

Path A rollback is unsetting the env var (or setting it `false`) and applying.
No code, no commit, no migration. The quote columns are additive and nullable, so
existing rows are unaffected and the analyzer tolerates their absence.

## Boundary

This prep authorizes **nothing**. `CAPTURE_SUBMISSION_QUOTE` stays **False**. No
code changed, no default changed, no push, no deploy, no live call.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `89e83a4` (7 ahead), `main` `7a8ca12`
unchanged; 0 staged, 0 unstaged.
