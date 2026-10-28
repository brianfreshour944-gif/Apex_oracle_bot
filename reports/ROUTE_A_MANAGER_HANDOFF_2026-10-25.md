# ROUTE A — HANDOFF TO THE SERVER MANAGER (run the baseline)

Prepared 2026-10-25. **Read the BLOCKER first** — without it the manager will hit
"file not found" and stall.

No credentials, no server address, and no access are needed by anyone but the
manager. Nothing is merged, pushed, or redeployed.

---

## BLOCKER: the script is NOT on GitHub

The script exists **only** in two places, neither of which the manager can reach
through the repository:

| Location | Reachable by manager? |
| --- | --- |
| `origin/main` | **No** — `main` is `7a8ca12` and does not contain it |
| Any of the 6 `origin/*` branches | **No** — scanned all of them; none contain it |
| Local branch `w1-instrumentation-accounting` (`96979b6`) | **Not pushed** — local only |
| Sandbox copy `/workspace/project/baseline_review/prod_baseline.py` | Only inside the OpenHands workspace |
| The user's local clone | Only on the user's machine |

So **the manager cannot `git clone` and find the file.** It must be handed over
(see "Getting the file to the manager" below), *unless* the deployed container
already has it (see step 1).

---

## Step 1 — does the running container already have the script?

This depends on which branch Coolify builds from:

- **If Coolify deploys `w1-instrumentation-accounting` at a commit ≥ `96979b6`** →
  the file is already in the image at `/app/scripts/prod_baseline.py`. Skip the
  handover and go straight to step 3 (command 3a).
- **If Coolify deploys `main` (`7a8ca12`)** → the file is **not** in the image. Do
  the handover (step 2), then use command 3b.

Find the container name first — **Coolify renames services per deploy and
`docker-compose.yaml` has no `container_name`, so `bot` may not resolve:**

```bash
docker ps --format '{{.Names}}' | grep -i bot
```

## Step 2 — getting the file to the manager (only if step 1 says "not in image")

The file is **244 lines / 9,028 bytes**, stdlib-only, single file. Any of these
works:

- **Preferred:** the user copies it from the OpenHands workspace (workspace root →
  `baseline_review/prod_baseline.py`, *not* inside the `Apex_oracle_bot` folder)
  and pastes it into the team channel / attaches it. Verify first:

  ```
  sha256 = 591cab03a144eaf1aea71cf0170c5a123b7334c72cbbdbf1f4e5c7452713f8ec
  ```

- Or the user, from their local clone on branch `w1-instrumentation-accounting`:
  ```bash
  git show 96979b6:scripts/prod_baseline.py > /tmp/prod_baseline.py
  ```
  and hand over `/tmp/prod_baseline.py`.

**Reject any copy whose SHA-256 does not match the value above.**

## Step 3 — run it

The tool opens the database **read-only**, copies it, and analyses the copy. It
never writes to the source file.

**3a. Script already in the container:**
```bash
docker exec <container> python /app/scripts/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db
```

**3b. Script copied in:**
```bash
docker cp /path/to/prod_baseline.py <container>:/tmp/prod_baseline.py
docker exec <container> python /tmp/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db
```

Rules the tool enforces (it exits 2 and explains if violated):
- `--out` must not be the source database.
- `--out` must not be an existing file you need, unless you pass `--force`.
- `/tmp` inside the container is a safe destination.
- The source **directory** (`/app/data`) must allow SQLite to create `-wal`/`-shm`.
  It is a writable volume, so this is satisfied.
- **Do not** add `immutable=1` — on a live database it reads a stale view.

## Step 4 — paste back the full output

Expected shape (from the local copy of the snapshot):

```
--- counts (on the copy) ---
  integrity: ok
  orders: 84
  filled: 84
  snapshots: 24
  closed: 24
  net: -2.1053
--- quote instrumentation ---
  present: NONE
  missing: ['mid_at_submit', 'spread_at_submit', 'quote_ts', 'fee_source']
  NOTE: no quote columns is consistent with capture never having
        been enabled. It is NOT evidence of a broken analyzer.
--- comparison vs 2026-10-25 baseline ---
  closed: current=24 baseline=24 MATCH
  orders: current=84 baseline=84 MATCH
  filled: current=84 baseline=84 MATCH
  net: current=-2.1053 baseline=-2.1053 MATCH
  => Counts match exactly: NO new trading evidence since 2026-10-25.
```

If production differs from this, **the differences are the finding.**

---

## Also please report these three (read-only)

1. **Branch + commit Coolify is running** (Coolify dashboard → Source panel).
   Note: `git -C /app rev-parse HEAD` will **not** work — `.git` is excluded by
   `.dockerignore`.
2. **Paper mode:** `docker logs <container> 2>&1 | grep "Paper="` → must show
   `Exchange: Alpaca Crypto (Paper=True)`. If `False`, **stop.**
3. **Volume:** `bot_data` mounted at `/app/data`, and `/app/data/bot.db` exists and
   is non-zero. If missing or 0 bytes, **stop.**

---

## NOT to do

- No merge into `main`, no cherry-pick, no push.
- No redeploy, no restart, no config change.
- No change to `CAPTURE_SUBMISSION_QUOTE` (stays `False`).
- No credentials, tokens, or SSH keys shared.
- Do not post any of this as a GitHub issue — **the repo is public.**
