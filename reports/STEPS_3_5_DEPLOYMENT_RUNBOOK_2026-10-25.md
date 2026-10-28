# STEPS 3–5 RUNBOOK — DEPLOYMENT VERIFICATION, BASELINE, CAPTURE ENABLEMENT

Date: 2026-10-25. **Read-only from this environment.** No edits to the
production database, no code changes, no commits, no push, no deploy, no live
call. `CAPTURE_SUBMISSION_QUOTE` stays `False`.

---

## 0. Access reality — stated up front

Steps 3 and 5 require the Coolify dashboard and the production host. **I do not
have access to either, and I will not attempt to obtain it.** Verified:

| Check | Result |
| --- | --- |
| Coolify token / URL in this environment | **none** (0 vars matching `coolify`) |
| SSH keys | **none** (`~/.ssh` does not exist) |
| Host / FQDN for the deployment | **not present anywhere** in repo or memory |
| Network egress | works (github.com → 200) — but there is nothing to reach |
| Coolify app credentials | not provided, and not something I should seek |

So the dashboard-driven parts of Step 3 are **operator actions**. What I *can*
do — and have done below — is give an exact, minimal checklist and a safe
read-only snapshot procedure, and verify what is verifiable from here.

## 1. Step 3 — deployment verification (OPERATOR, in the Coolify dashboard)

I cannot read the dashboard. Here is the checklist, in the order that matters.

**3.1 Which application and host**
- Coolify → the project containing the `Apex_oracle_bot` service. Record the
  application name, the server it is deployed to, and the FQDN Traefik routes
  to it (the compose file uses `expose: 8000`, so the FQDN is set in Coolify).

**3.2 Which branch and commit are deployed**
- Coolify shows the configured branch (expected `main`) and the deployed commit.
- **Expected commit: `7a8ca12`.** If it differs, that is a finding — the repo has
  other branches (`audit-fixes`, `paper-guard-7a8ca12`, `research/net-expectancy`)
  and a non-`main` deploy would change what is running.
- Confirm the commit the *container* actually built, not just the configured ref.

**3.3 Effective Alpaca endpoint (paper vs live)**
- Coolify → the app's Environment Variables. Look for `ALPACA_BASE_URL`.
- **Paper** = `https://paper-api.alpaca.markets`. **Live** = `https://api.alpaca.markets`.
- If unset, the code default is paper (`src/config.py:42-43`).
- **Independent confirmation:** the startup log line
  `Exchange: Alpaca Crypto (Paper=True)` (`src/bot.py:3663`). A live endpoint
  would print `Paper=False`. **This is the authoritative runtime check** — the
  env var shows intent, the log shows what the process actually used.

**3.4 Whether `bot_data` is intact and holds orders**
- Coolify → the app's Persistent Storage / Volumes. Confirm `bot_data` is
  mounted at `/app/data`.
- The volume is the only thing preventing the trade DB being wiped on redeploy
  (see the compose comment). If it is missing, **stop** — the data situation is
  not what the earlier snapshot reflects.
- Then count orders per §2.2.

## 2. Step 4 — establish a safe baseline (OPERATOR runs on the host)

**2.1 Locate the database**
- Default path is `/app/data/bot.db` (`DATABASE_URL=sqlite:///data/bot.db`).
- Confirm from the app's env whether `DATABASE_URL` is overridden (the config
  also supports PostgreSQL; if it is Postgres, the SQLite procedure below does
  not apply and a different dump method is needed).

**2.2 Take a CONSISTENT snapshot — do not `cp` a live SQLite file**
A plain `cp` of a SQLite database that is being written can capture a torn state.
Use the SQLite backup API, which takes a consistent copy:

```bash
# On the host, into a scratch path (NOT over the live file):
python - <<'PY'
import sqlite3
src = sqlite3.connect("file:/app/data/bot.db?mode=ro", uri=True)
dst = sqlite3.connect("/tmp/bot_baseline.db")
with dst:
    src.backup(dst)          # consistent, atomic snapshot
dst.close(); src.close()
print("snapshot written")
PY
```

Alternatively, if the `sqlite3` CLI is available:
```bash
sqlite3 /app/data/bot.db ".backup '/tmp/bot_baseline.db'"
```

Both are **read-only with respect to the live database**. Do not run `VACUUM`,
`DELETE`, or any write against `/app/data/bot.db`.

**2.3 Record the baseline** (read-only, on the copy):
```bash
python - <<'PY'
import sqlite3
c = sqlite3.connect("file:/tmp/bot_baseline.db?mode=ro&immutable=1", uri=True)
print("integrity:", c.execute("PRAGMA integrity_check").fetchone()[0])
print("orders:", c.execute("SELECT COUNT(*) FROM orders").fetchone()[0])
print("filled:", c.execute("SELECT COUNT(*) FROM orders WHERE lower(status)='filled'").fetchone()[0])
print("snapshots:", c.execute("SELECT COUNT(*) FROM decision_snapshots").fetchone()[0])
print("closed:", c.execute("SELECT COUNT(*) FROM decision_snapshots WHERE lower(status)='closed'").fetchone()[0])
print("net:", c.execute("SELECT ROUND(SUM(realized_pnl),4) FROM decision_snapshots WHERE lower(status)='closed'").fetchone()[0])
print("cols:", [r[1] for r in c.execute("PRAGMA table_info(orders)")])
PY
```

**2.4 Then analyze the copy** (never the live file):
```bash
python scripts/analyze_deployed_trades.py /tmp/bot_baseline.db --json /tmp/bot_baseline.json
```
Note the analyzer opens with `immutable=1`, which asserts the file will not
change. That is correct for a static copy and **wrong for a live database** —
another reason to analyze the copy, not `/app/data/bot.db`.

**2.5 Compare against the known baseline.** The 2026-10-25 snapshot had:
24 closed, 84 orders (all filled), `SUM(realized_pnl) = −2.1053`. If the current
counts match exactly, **no new orders have been recorded** and there is nothing
new to measure yet. If they have grown, that is new data.

## 3. What I verified from here (read-only)

- The local `/workspace/project/bot_snapshot.db` **is the same 24-trade file**
  from the prior session: sha256 `52e39180d5d9e9e10eb0…3cfca1b`, 2,105,344 bytes,
  mtime 2026-10-25 00:36. It is **NOT a fresh snapshot** and cannot evidence the
  current deployment state.
- Its integrity check passes: 24 closed, 84 filled orders, net −2.1053.
- The analyzer runs against it without error and reports the SLIPPAGE section as
  "not captured on this snapshot (missing: mid_at_submit, spread_at_submit,
  quote_ts, fee_source)" — consistent with capture never having been enabled.
- `sqlite3.Connection.backup` is available in this environment's stdlib, so the
  §2.2 procedure is sound.

## 4. Step 5 — enable capture (DECISION, only after §1 and §2 pass)

**Gate: do not enable unless all of these are true.**
- [ ] The deployed commit is the intended one (`7a8ca12` or a deliberately chosen
      later commit).
- [ ] The effective endpoint is **paper** — confirmed by the startup log
      `Paper=True`, not merely by the env var.
- [ ] `bot_data` is mounted and the baseline snapshot was taken successfully.
- [ ] The baseline order count is recorded.

If all pass, the change is **one environment variable in Coolify**:

```
CAPTURE_SUBMISSION_QUOTE=true
```

then redeploy. **The committed default stays `False`; no source change.** The
env var overrides it (proven empirically in
`reports/CAPTURE_ENABLEMENT_PREP_2026-10-25.md`).

**After enabling, verify** (read-only):
1. Startup log still says `Paper=True`.
2. After orders accumulate, snapshot again (§2.2) and analyze.
3. The SLIPPAGE section should show `rows_considered > 0`. Watch the exclusion
   buckets — a high `age_unverified` count would mean the fill/submit timestamps
   are not being written as expected.

**Rollback:** unset the variable (or set it `false`) and redeploy. No code, no
migration. The quote columns are additive and nullable.

## 5. Honest limits

- **I cannot execute Steps 3 or 5.** They are dashboard/host actions, and I have
  no credentials or access. This document is a runbook, not a verification.
- **No runtime fact is proven by this report.** Everything above about the live
  deployment is inference from repository files; the repository cannot prove what
  the running container does.
- The only snapshot available here is **stale** (2026-10-25) and predates any
  capture.
- If `DATABASE_URL` is Postgres in production, the SQLite snapshot procedure does
  not apply and needs a different, equally read-only dump method.

## Boundary

This report authorizes **nothing** and changed **nothing** in the repo or the
deployment. `CAPTURE_SUBMISSION_QUOTE` stays `False`. No live call was made.

## Safety state

Branch `w1-instrumentation-accounting`, HEAD `89e83a4` (7 ahead), `main` `7a8ca12`
= `origin/main`, unchanged; 0 staged, 0 unstaged.
