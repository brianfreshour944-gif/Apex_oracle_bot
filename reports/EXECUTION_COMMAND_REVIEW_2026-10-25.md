# EXECUTION COMMAND REVIEW — prod_baseline.py (2026-10-25)

Status: **script transferred and hash-verified on the host. NOT executed.**
Awaiting explicit authorization to run.

## Established facts

| Item | Value | Evidence |
| --- | --- | --- |
| Script on host | `/tmp/prod_baseline.py` | operator terminal |
| Host hash | `591cab03a144eaf1aea71cf0170c5a123b7334c72cbbdbf1f4e5c7452713f8ec` | operator `sha256sum` — **MATCH** |
| Transfer method | `curl -fsS -o` from the temp HTTPS server | operator terminal |
| Temp server | **STOPPED** (process gone, port closed, URL 502) | agent |
| Output path | `/tmp/bot_baseline.db` | operator `ls -l` → **does not exist** (clear) |
| Baseline executed | **nowhere** | — |

## The command

```bash
docker exec <container> python /tmp/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db
```

Wait — `/tmp/prod_baseline.py` is on the **host**, not inside the container. It must
be copied in first:

```bash
docker cp /tmp/prod_baseline.py <container>:/tmp/prod_baseline.py
docker exec <container> python /tmp/prod_baseline.py /app/data/bot.db --out /tmp/bot_baseline.db
```

Discover `<container>` first (Coolify renames services; `bot` may not resolve):

```bash
docker ps --format '{{.Names}}' | grep -i bot
```

## What the command does — and does not — touch

**Reads:** `/app/data/bot.db` — opened `mode=ro` via SQLite URI. Never opened
read-write. No VACUUM, no DELETE, no write ever touches it.

**Writes:**
- `/tmp/bot_baseline.db` **inside the container** — a fresh consistent copy. This is
  a container-local path; it is NOT in the `bot_data` volume and disappears when the
  container is replaced.
- Possibly `/app/data/bot.db-wal` and `-shm` **beside the source** — SQLite creates
  these the moment a WAL-mode database is opened, even read-only. **Expected.**
  Do not delete them; they belong to the live database.

**Does not touch:** the source database file, any order, any trade, any config,
any environment variable, the running bot process. It does not restart or signal
the bot.

**Verified locally:** running the tool against the snapshot left the source
database byte-identical (sha256 unchanged), and the `-shm` sidecar's mtime
updated — confirming sidecar creation with an unmodified source. This is the
documented PB-5 behaviour.

## Expected output

From the local 2026-10-25 snapshot:

```
source (read-only): /app/data/bot.db
consistent copy:    /tmp/bot_baseline.db

--- counts (on the copy) ---
  integrity: ok
  orders: 84
  filled: 84
  snapshots: 24
  closed: 24
  net: -2.1053
  order_cols: [...]

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

## How to read the result

| Result | Meaning | Next |
| --- | --- | --- |
| All four `MATCH` | No new trading evidence since 2026-10-25 | Ask why no new orders are being taken |
| Any `DIFFERS` | New data exists | Analyze the copy (see below) |
| `quote instrumentation: present: [...]` | Capture is on and populated | Slippage analysis becomes meaningful |
| `present: NONE` | Capture never enabled | **Expected** — not a defect |
| `INCONCLUSIVE` (exit 2) | DB could not be read as expected | A read/schema problem, **not** new trades |

**Exit codes:** `0` = ran and compared. `2` = refused or inconclusive (source
missing, `--out` collision, unreadable schema). Either way, read the printed
message.

## The open question this answers

Production `/app/data/bot.db` was reported at ~2.2 MB versus the local snapshot's
2,105,344 bytes (2.11 MB decimal) — roughly 95 KB larger. SQLite size is **not** a
trade count (freelist pages, page fragmentation, and an uncheckpointed WAL all
move it), so this is a hypothesis, not a finding. The `closed`/`orders`/`filled`/`net`
comparison is the test that settles it.

## Notes before running

1. **`--out` is clear** (`/tmp/bot_baseline.db` does not exist). If it appears
   between now and the run, either remove it or add `--force`. Without `--force`
   the tool refuses rather than overwriting — that guard is intentional.
2. **Do not add `immutable=1`** to the source open. On a live WAL database it reads
   a stale view and can miss committed, uncheckpointed data — and even the schema.
3. **`/app/data` must allow sidecar creation.** It is a writable volume, so this
   holds. A read-only directory would abort the run.
4. **The container runs as `botuser`** (non-root). `docker cp` preserves ownership;
   `/tmp` inside the container is world-writable, so both the copied script and the
   output are fine.
5. If the second step errors with `No such file or directory` for the script, the
   `docker cp` did not land — re-run it and verify with
   `docker exec <container> ls -l /tmp/prod_baseline.py`.

## If DIFFERS: the follow-up

```bash
docker cp <container>:/tmp/bot_baseline.db /tmp/bot_baseline.db
docker cp /tmp/prod_baseline.py <container>:/tmp/   # analyzer needs the same env
```
Then run the analyzer against the copy:
`python scripts/analyze_deployed_trades.py /tmp/bot_baseline.db --json out.json`
(that script is on the feature branch, not `main`; copy it in the same way if
needed).

## Not authorized

No merge, no cherry-pick, no push, no redeploy, no restart, no config change, no
change to `CAPTURE_SUBMISSION_QUOTE` (stays `False`), no trading-behavior change.
Running the command above is the only action under consideration.
