# REQUEST TO SERVER MANAGER — 3 read-only facts about the deployment

Please forward this to whoever manages the Coolify server. **No credentials, no
access, and no changes are needed** — these are three read-only observations.

Context: we want to compare the live bot database against a known snapshot. We do
not need anyone to modify the bot, redeploy it, or change its configuration.

---

## 1. Which branch and commit is deployed?

**From the Coolify dashboard (preferred):** open the application, and report the
**Source** / **Repository** panel — the repository, the configured **branch**, and
the **commit SHA** of the current deployment.

**Note:** `git -C /app rev-parse HEAD` will **not** work — `.git` is excluded from
the image by `.dockerignore`. Use the dashboard, or the host-side commands below.

**From the host, if you have shell access:**
```bash
# the image reference of the running container
docker ps --format '{{.Names}}\t{{.Image}}' | grep -i bot
# or, from the deploy directory:
docker compose config --images
```

What we need: **the branch name** and **the commit SHA** the running container was
built from.

## 2. Is paper trading confirmed?

Report the startup log line that contains `Paper=`:

```bash
# find the actual container name first (Coolify renames services per deploy,
# so 'bot' may not resolve):
docker ps --format '{{.Names}}' | grep -i bot
# then, using that name:
docker logs <container> 2>&1 | grep "Paper="
```

We are looking for exactly:
```
Exchange: Alpaca Crypto (Paper=True)
```
`Paper=True` is required. If it says `Paper=False`, stop — that means live trading
and we need to know before anything else.

## 3. Is the database volume persistent and mounted?

```bash
docker volume ls | grep bot_data
docker inspect <container> --format '{{range .Mounts}}{{.Name}} -> {{.Destination}}{{"\n"}}{{end}}'
docker exec <container> ls -la /app/data/
```

What we need: confirmation that the named volume **`bot_data`** is mounted at
**`/app/data`**, and that **`/app/data/bot.db` exists and is non-zero**.

If `/app/data/bot.db` is missing or 0 bytes, stop — the database is not persisting.

---

## Optional (only if convenient) — the baseline itself

If you are comfortable running one read-only command, this produces the number we
actually need. It requires **no merge and no redeploy** — the script is
stdlib-only and is copied into the running container:

```bash
# run from a checkout of the repo (branch w1-instrumentation-accounting, commit 96979b6)
docker cp scripts/prod_baseline.py <container>:/tmp/prod_baseline.py
docker exec <container> python /tmp/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db
```

Then paste the full stdout. Please note:

- `--out` must not be the source and must not be an existing file you need
  (the tool refuses both, and says so).
- The source **directory** (`/app/data`) must allow SQLite to create `-wal`/`-shm`
  sidecars; as a writable volume it does.
- The tool opens the database **read-only** and never modifies it. It copies to
  `/tmp` and analyses the copy.
- Do **not** add `immutable=1` — on a live database it reads a stale view.

The expected output looks like this (from our local copy of the snapshot):

```
--- counts (on the copy) ---
  integrity: ok
  orders: 84
  filled: 84
  snapshots: 24
  closed: 24
  net: -2.1053
--- comparison vs 2026-10-25 baseline ---
  closed: current=24 baseline=24 MATCH
  orders: current=84 baseline=84 MATCH
  filled: current=84 baseline=84 MATCH
  net: current=-2.1053 baseline=-2.1053 MATCH
  => Counts match exactly: NO new trading evidence since 2026-10-25.
```

If production differs from this, the differences are the finding.

---

## What we are NOT asking for

- No credentials, tokens, or SSH keys.
- No redeploy, no restart, no configuration change.
- No merge into `main`.
- No change to `CAPTURE_SUBMISSION_QUOTE` (it stays `False`).

The immediate goal is only to learn **what is running and what the database
contains**, before any behavior change is considered.
