# SCOPE NOTE — WHAT EXECUTOR ISOLATION DOES AND DOES NOT PROTECT

Date: 2026-10-25. **Read-only. No code changed.** Substantiates the
qualification: *executor isolation protects order submission from worker
starvation caused by quote capture; it does not guarantee the entire order path
is immune to every delay or failure.*

That qualification is correct. This note maps exactly what is covered, what is
not, and why — so the isolation change is scoped honestly.

---

## What executor isolation DOES fix

A dedicated pool for quote capture removes the **worker-starvation** coupling.
Verified previously: with a separate pool, 8 hung captures left `submit_order`
waiting **0.00s**; on the shared pool it waited **1.95s**.

**Additional confirmations (this note):**

- `trading_client` and `data_client` are **separate objects** with **separate
  `requests.Session` instances** (`RESTClient.__init__` creates its own
  `self._session = Session()`), so there is **no shared HTTP connection pool**.
- `get_latest_quote` does **not** use the `circuit_breaker`, so it shares **no
  lock** with the trading calls.
- `get_latest_quote` carries **no `@retry` decorator**, so it does not enter the
  retry/backoff path that `create_order` uses.

So the isolation is genuinely structural for the **worker-pool** resource.

## What executor isolation does NOT fix

Your qualification names these; here they are concretely.

**1. Shared API key and its rate-limit budget.**
Both clients authenticate with the **same** `self.api_key`
(`exchange.py:252–253`). Alpaca applies rate limits per API key, not per client.
So a burst of quote requests consumes the same quota the order path needs. If
quote capture drives the key into 429 territory, `create_order` hits its
`@retry(..., wait=wait_exponential(min=2, max=30))` backoff — **a delay caused by
measurement, despite isolation.**

Isolation does not help here: the shared resource is the *credential's quota*,
not the thread pool.

**2. Shared event loop.**
Both paths run on the same asyncio loop. A blocking call on a dedicated pool
still occupies a thread; if the loop itself is saturated by many such threads'
completion callbacks, scheduling latency can rise. Isolation narrows but does not
eliminate loop-level contention.

**3. Shared process and host.**
Both run in one process on one host. CPU, memory, file descriptors, and network
stack are shared. Isolation does not partition those.

**4. `create_order`'s own unbounded network call.**
`create_order` already has **no socket timeout** (same root cause). Isolation of
*capture* does nothing about the order path's *own* exposure.

**5. Rate-limit accounting is inert.**
`_rate_limit_remaining` is initialized to 200 and **never read** — grep shows the
only occurrence is the assignment. So nothing tracks remaining quota or
proactively throttles. Quote capture's consumption is invisible to the bot.

## Correct scoping of the change

| Claim | True? |
| --- | --- |
| Isolation prevents capture from starving the worker pool | **yes — structural** |
| Isolation prevents capture from consuming the shared rate-limit quota | **no** |
| Isolation makes the whole order path delay-free | **no** |
| Isolation plus a caller `wait_for` fully de-risks capture | **no — see rate limit** |

So the honest statement is: isolation removes the *worker-starvation* risk, which
was the demonstrated one. It leaves the *shared-quota* risk, which is smaller at
low turnover but real.

## Implication for the design

Isolation is necessary but should not be presented as complete. To be thorough,
the scoped change could also:

- **Budget the quote calls**: skip capture if recent call volume is high, or
  space captures so they cannot meaningfully eat the key's quota.
- **Never let capture block the order**: since `get_latest_quote` has no retry,
  a 429 on capture simply returns `None` — which is already the correct,
  fail-open behaviour. This is a point in the current design's favour: capture
  cannot *retry-loop* itself into a quota hole.
- **Consider capturing asynchronously**: fire the capture and attach it to the
  order row on a best-effort basis, so it is off the critical path entirely.

## Consequence for the deployment decision

Keep capture **disabled in deployment** until isolation is implemented and
verified — as agreed. Note that isolation addresses the demonstrated risk
(worker starvation), **not** every possible delay (shared quota). The default
`True` should be reconsidered on evidence once isolation lands, with the quota
question explicitly acknowledged rather than assumed away.

## Safety state

No branch, 0 commits, 0 staged; tracker `44136fa3…`, drift `82a78c0d…`,
snapshot `52e39180…` unchanged.
