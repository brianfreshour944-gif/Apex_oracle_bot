# DEPLOYMENT VERIFICATION — READ-ONLY (Step 1)

Date: 2026-10-25. **Read-only.** No edits, staging, commits, DB changes, push,
deploy, or live call. The committed default `CAPTURE_SUBMISSION_QUOTE=False` is
unchanged.

Purpose: establish *where* the paper bot runs and *how* its environment is
configured, before any operational change. This is Step 1 of the agreed sequence.

---

## 1. Correction to an earlier report (important)

`reports/CAPTURE_ENABLEMENT_PREP_2026-10-25.md` §4.5 said *"No docker-compose.yml
in this checkout."* **That was wrong.** Both deployment files exist, are tracked,
and are on `main`:

| File | Status |
| --- | --- |
| `docker-compose.yaml` | tracked, present on `main` (`7a8ca12`) |
| `Dockerfile` | tracked, present on `main` |
| `.dockerignore` | tracked, present on `main` |

Cause of the earlier error: my search used `docker-compose*.yml`, which does not
match `.yaml`. The files were always there. Correcting the record rather than
letting the stale claim stand.

## 2. What the deployment IS (evidence from the repo)

**Orchestrator: Coolify** (stated explicitly in both files' comments). The deploy
builds `Dockerfile` via `docker-compose.yaml`.

**Container:** `python:3.12-slim`, deps via `uv`, non-root user `botuser`
(uid 10001), entrypoint `python -m src.cli run`, service listens on `:8000`
(`expose`, not published — Coolify's Traefik routes the FQDN).

**Persistence:** a named volume `bot_data` mounted at `/app/data`. The comment is
explicit about why: a relative bind mount would resolve against Coolify's
ephemeral per-deploy `/artifacts` dir and **wipe the trade DB on every redeploy**.
This is where `data/bot.db` lives in production.

**Deployed env baked into the image** (`Dockerfile` `ENV`):

```
ADAPTIVE_ML_ENABLED=true
HIGH_VOLATILITY_PCT=5.0
```

The Dockerfile comment records that these were *only* in a local `.env` until
2026-09-29, so the VM ran with code defaults (`ADAPTIVE_ML_ENABLED=False`), which
switched every ML component off. **The Dockerfile also states: "Environment
variables set in Coolify still override these."** That is the documented override
mechanism.

**`CAPTURE_SUBMISSION_QUOTE` is absent from both `docker-compose.yaml` and
`Dockerfile`.** So the container currently runs with the code default `False`.

## 3. Paper vs live endpoint

- Code default: `ALPACA_BASE_URL = "https://paper-api.alpaca.markets"`
  (`src/config.py:42-43`) — **paper by default**.
- The bot logs its mode at startup: `Paper={ALPACA_BASE_URL.endswith('paper-api.alpaca.markets')}`
  (`src/bot.py:3663`, `src/config.py:914`).
- `ALPACA_BASE_URL` is settable via env (`.env.example` documents it).
- **Not determinable from the repo:** which value the Coolify deployment actually
  sets. The compose file does not set it, so it is either the code default
  (paper) or a Coolify env override (unknown).

## 4. How to enable capture for this deployment (operational, NOT done)

Because `settings` reads the env with `env_prefix=""`/`case_sensitive=True`
(proven in the enablement prep), and Coolify env overrides the image:

1. In the Coolify app for this service, add the env var
   `CAPTURE_SUBMISSION_QUOTE=true`.
2. Redeploy/restart per Coolify's normal procedure.
3. The **committed default stays `False`** — no code change, no commit.
4. Verify after startup that quotes are being written (see §6).

Alternative (also valid, but a code change): add the var to `docker-compose.yaml`
under the `bot` service `environment:`. That is a committed change affecting
whatever else uses this compose file, so the Coolify-env route is preferable.

## 5. What I cannot verify from here (needs the operator)

- **Which Coolify app/host runs the bot**, and its FQDN.
- **The deployment's actual env values** — especially `ALPACA_BASE_URL`
  (paper vs live) and whether any override is already set. The repo shows
  defaults, not the running values.
- **Whether the volume `bot_data` is intact** and how many orders it currently
  holds. The local `data/bot.db` is a *different*, empty file (0 orders).
- **The bot's current running state / version** — is the VM on `main` `7a8ca12`,
  or another branch? The repo has other remote branches
  (`audit-fixes`, `paper-guard-7a8ca12`, `research/net-expectancy`).
- **Whether a fresh snapshot can be pulled** and how (Coolify shell, SSH, or a
  DB export). I have no access to the host and must not attempt to reach it.

## 6. Verification steps once capture is enabled (later, read-only)

1. Confirm startup log says `Paper=True` (if paper is the intent).
2. After orders accumulate, snapshot the DB and run:
   `python scripts/analyze_deployed_trades.py <snapshot> --json out.json`
3. Inspect the SLIPPAGE section:
   - `rows_considered` > 0 (capture is working),
   - the exclusion buckets (`no_quote`, `no_quote_ts`, `bad_spread`,
     `stale_quote`, `age_unverified`) — high `age_unverified` would mean the
     fill/submit timestamps are not being written as expected,
   - `excess_slippage_bps` vs `half_spread_bps` — the W1 payoff.
4. Check fee provenance on the orders (`orders.commission` only; no flag).

## 7. Honest limits

- Static file reading does not prove runtime behavior; the deployment's live
  config may differ from the repo.
- No live call was made and none should be.
- 24 trades is a small sample; even clean slippage data will be low-confidence
  until more orders accumulate.

## 8. Security observation (unrelated to this task, but found here)

The local git remote URL embeds a GitHub token in plaintext:

```
origin  https://<REDACTED>@github.com/brianfreshour944-gif/Apex_oracle_bot.git
```

- It lives in `.git/config`, which is **not tracked** (`git ls-files .git/config`
  is empty), so it is **not** committed and will not be pushed. That is the good
  part.
- It is still a live credential sitting in plaintext on this filesystem, and it
  appears in the shell history of any command that prints the remote.
- **Recommendation:** rotate that token and switch the remote to SSH or a
  credential helper so the value is not stored inline:
  `git remote set-url origin git@github.com:brianfreshour944-gif/Apex_oracle_bot.git`
- I have not changed the remote and will not; that is an operator action.
- The value is deliberately not reproduced in this report.

## Boundary

This report authorizes **nothing**. No code changed, no default changed, no
compose/Dockerfile edit, no push, no deploy, no live call.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `89e83a4` (7 ahead), `main` `7a8ca12`
unchanged; 0 staged, 0 unstaged. `CAPTURE_SUBMISSION_QUOTE` default `False`.
