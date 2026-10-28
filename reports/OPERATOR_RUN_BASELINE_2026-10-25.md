# OPERATOR RUNBOOK — run the baseline on the production host

Date: 2026-10-25. Prepared by agent. **The agent has no host access** (verified: no
`~/.ssh`, no `/app`, no Coolify token/URL, no hostname in repo or env). Every
command below must be run by the operator on the Coolify host.

## BLOCKER TO READ FIRST

`scripts/prod_baseline.py` is committed on branch
`w1-instrumentation-accounting` (HEAD `96979b6`), but it is **NOT on `main`**:

```
$ git merge-base --is-ancestor 96979b6 main   -> NO
$ git cat-file -e main:scripts/prod_baseline.py -> ABSENT
```

The deployment builds from the deployed branch. If that is `main`, the script is
**not in the running container** and step 2 below will fail with "No such file".
Confirm which branch Coolify builds from before proceeding.

The good news: the script is **stdlib-only** (no `src` imports), so it can be run
without being in the image — see step 2b. That is the reliable path.

## Step 1 — identify what is actually running

**1a. Deployed commit.** `.git` is in `.dockerignore`, so the image carries no git
metadata; `git -C /app rev-parse HEAD` will **not** work. Use one of:

```bash
# from the Coolify dashboard: the deploy's commit SHA / "Source" panel
# from the host, if you can see the compose project:
docker compose config --images
docker inspect --format '{{.Config.Image}}' $(docker ps -qf name=bot)
```

**1b. Paper mode — the definitive startup log line.** The bot logs exactly:

```
Exchange: Alpaca Crypto (Paper=True)
```

emitted at `src/bot.py:3663` (also `src/config.py:914`). Get it with:

```bash
docker compose logs bot | grep "Paper="
# or: docker logs <container> 2>&1 | grep "Paper="
```

Require `Paper=True`. `False` means live trading is enabled — **stop and report**.

**1c. Persistent volume.** `docker-compose.yaml` mounts the named volume
`bot_data` at `/app/data`. Confirm it exists and is not empty:

```bash
docker volume ls | grep bot_data
docker volume inspect <project>_bot_data
# and confirm the DB is inside it:
docker compose exec bot ls -la /app/data/bot.db
```

If `/app/data/bot.db` is absent or 0 bytes, stop — the DB is not persisting.

## Step 2 — run the baseline

**2a. If the script IS in the container** (deployed branch contains `96979b6`):

```bash
docker compose exec bot python /app/scripts/prod_baseline.py /app/data/bot.db \
    --out /tmp/bot_baseline.db
```

**2b. If the script is NOT in the container (likely — see BLOCKER).** Copy the
committed file in and run it. It needs only the stdlib:

```bash
# from the host, where the repo checkout is:
docker compose cp scripts/prod_baseline.py bot:/tmp/prod_baseline.py
docker compose exec bot python /tmp/prod_baseline.py /app/data/bot.db \
    --out /tmp/bot_baseline.db
```

**Output-path rules (enforced by the tool; exit 2 with a clear message otherwise):**
- `--out` must not be the source. `--out /app/data/bot.db` is refused.
- `--out` must not be an existing file you need, unless you pass `--force`.
- `/tmp` inside the container is a safe destination (it is not the volume).
- The source **directory** must allow SQLite to create `-wal`/`-shm`. `/app/data`
  is a writable volume, so this is satisfied. Do **not** use `immutable=1` on the
  live DB (it reads a stale view and can miss the schema).

## Step 3 — bring back the output

Copy the report out and return it verbatim:

```bash
docker compose cp bot:/tmp/bot_baseline.db ./bot_baseline_prod.db   # optional
# the important part is the stdout of step 2 — paste it in full
```

### How to read the verdict

| Result | Meaning | Next |
| --- | --- | --- |
| `closed/orders/filled/net` all `MATCH` | No new trading evidence since 2026-10-25 | Investigate **why no new orders** (is the bot trading at all?) before touching anything |
| Any `DIFFERS` | New data exists | Run the analyzer on the copy, then inspect |
| `quote instrumentation: present: [...]` | Capture is on and populated | Slippage analysis is meaningful |
| `present: NONE` | Capture never enabled | **Expected.** Not a defect — do not read an empty slippage report as breakage |
| `INCONCLUSIVE` (exit 2) | DB could not be read as expected | A schema/read problem, not new trades |

## What NOT to do

- Do not push or deploy merely because the tool is committed.
- Do not enable `CAPTURE_SUBMISSION_QUOTE` (default stays `False`).
- Do not change the committed default.
- Do not interpret an empty slippage report as a broken analyzer when no quote
  data was ever captured.

## Expected stdout shape (from the local snapshot, for reference)

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
