# ADDENDUM — TIMEOUT HARDENING IS NOT SUFFICIENT (thread-pool coupling)

Date: 2026-10-25. **Read-only. No code changed.** Sharpens point 5 of the
decision sequence: do not assume `asyncio.wait_for` alone fixes the order-path
blocking risk.

---

## Your point, validated empirically

You said: *"don't assume asyncio.wait_for alone cancels the underlying thread's
network request."* Confirmed — and the consequence is worse than "the thread
keeps running harmlessly."

### Test 1 — `wait_for` stops waiting; the thread does not stop

```
wait_for raised TimeoutError after 0.50s
thread finished shortly after timeout?  False
thread finished eventually?          True
```

`asyncio.to_thread` runs the blocking call on the loop's **default executor**.
`wait_for` cancels the *await*, not the thread. The socket read continues until
the OS/HTTP layer gives up.

### Test 2 — the default executor is SHARED and SMALL

- `asyncio.to_thread` → `loop.run_in_executor(None, ...)` → the loop's single
  default `ThreadPoolExecutor`.
- On this host: `max_workers = min(32, cpu+4) = 8`.
- **`exchange.py` sends both `trading_client` and `data_client` calls through
  that same pool** — 9 `asyncio.to_thread` call sites, including `submit_order`.

### Test 3 — hung quote threads starve order submission

Occupying all 8 workers with hung "quote" calls:

```
submit_order got a worker after 2.00s (should be ~0 if pool free)
```

**The order submission queued behind the hung quotes.** This is the real risk:
a hung quote capture does not merely delay *its own* order — repeated hangs can
occupy every worker and delay *any* order.

## Why this matters for the decision

`asyncio.wait_for(..., timeout=N)` would make the *caller* stop waiting at N
seconds, so the order proceeds. But:

- The thread still holds a pool worker until the socket times out.
- Under repeated hangs, workers accumulate and the pool saturates.
- So `wait_for` bounds the *latency* but not the *resource leak*.

**A sufficient fix needs a real socket timeout** — pass `timeout=` to the
underlying HTTP request (or configure the session), so the thread releases its
worker. `wait_for` alone is necessary but not sufficient.

## Consequence for CAPTURE_SUBMISSION_QUOTE

Keeping the default `True` **explicitly accepts** the added order-path risk:

| Risk | Accepted by `True`? |
| --- | --- |
| One extra API round-trip per order | yes (immaterial at low turnover) |
| A hang delays that order (bounded by `wait_for`, if added) | yes |
| A hang consumes a shared pool worker until socket timeout | **yes — this is the material one** |
| Repeated hangs can delay *unrelated* orders via pool saturation | **yes** |

Stated plainly: `True` is not risk-free merely because the bot already has
unbounded calls elsewhere. It adds a new unbounded call to the same small pool
the order path depends on.

## Revised recommendation

1. **Retain `True`** as the W1 default — but record the accepted risk above in
   the commit message / config docstring, so it is a decision, not an inheritance.
2. **Do not treat `wait_for` as the fix.** The scoped hardening change should add
   a **real socket/HTTP timeout** to the data-client request path; `wait_for` can
   accompany it but is not sufficient.
3. **Prioritise the timeout hardening before enabling quote capture on a deployed
   bot** — per your point 5. This ordering is now evidence-backed, not merely
   cautious.

## Scope

Changes nothing. The hardening is a separate change needing its own scope, tests,
and approval — as you specified.

## Safety state

No branch, 0 commits, 0 staged; tracker `44136fa3…`, drift `82a78c0d…`,
snapshot `52e39180…` unchanged.
