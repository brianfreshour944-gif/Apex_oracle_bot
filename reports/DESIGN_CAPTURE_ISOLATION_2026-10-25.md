# DESIGN NOTE — CAPTURE ISOLATION vs TIMEOUT HARDENING

Date: 2026-10-25. **Read-only. No code changed.** Responds to the refinement:
*documenting acceptance of a risk does not make it operationally safe; because
quote capture is measurement-only, it should not be allowed to compromise the
order path.*

That reframing is correct, and it changes the recommendation from "accept and
document" to "isolate so it cannot interfere." This note tests the isolation.

---

## The problem, restated

Quote capture currently runs on the **same default executor** as `submit_order`.
A hang therefore couples measurement to trading. Two candidate fixes:

- **A. Timeout hardening** — bound the socket so the worker is released.
- **B. Executor isolation** — run capture on a *separate* pool so it can never
  occupy a worker the order path needs.

## Finding: timeout hardening is not cleanly available

`alpaca.common.rest.RESTClient._request(...)` accepts
`(method, path, data, base_url, api_version)` — **no `timeout` parameter**. The
request is issued as `self._session.request(method, url, **opts)` with no
timeout. So a real socket timeout cannot be passed through the public API; it
would require reaching into the client's private `_session` and setting a default
timeout on the session (monkeypatching a third-party object) — fragile and
version-dependent.

So **A is not a simple change.** It touches a private attribute of a dependency.

## Finding: executor isolation works, and is structurally guaranteed

Measured, 8 hung "quote" calls competing with a "submit_order" call:

```
SHARED default pool (current design): submit_order waited 1.95s
DEDICATED quote pool (isolated)     : submit_order waited 0.00s
```

With a dedicated pool for quote capture, hung captures **cannot** delay order
submission — the order path keeps its own workers regardless of what capture
does. This is a structural guarantee, not a bounded-latency best-effort.

## Why isolation is the better fit for a measurement feature

| Property | Timeout hardening (A) | Executor isolation (B) |
| --- | --- | --- |
| Fixes worker starvation | partly (until socket timeout) | **yes, structurally** |
| Relies on dependency internals | **yes** (private `_session`) | no |
| Affects order path code | yes | no |
| Failure mode of the feature | capture still couples | capture fails alone |
| Fits "measurement must not compromise orders" | no | **yes** |

Isolation directly satisfies the principle you stated: a measurement-only
feature should be incapable of compromising the order path — not merely
unlikely to.

## Recommended design (for the separate, scoped change)

1. Give quote capture its **own small `ThreadPoolExecutor`** (e.g. 1–2 workers),
   created once, used only by `get_latest_quote`.
2. Add an `asyncio.wait_for` bound **as well**, so the *caller* stops waiting and
   the order proceeds promptly even if the thread lingers.
3. Optionally also set a socket timeout if it can be done without private
   attributes — but do **not** rely on it; isolation is the guarantee.
4. Test: a stalled quote request must (a) release the caller, and (b) leave
   `submit_order` unaffected. The second assertion is the one that matters and is
   currently missing.

## Consequence for the capture default

With isolation in place, `CAPTURE_SUBMISSION_QUOTE=True` no longer carries the
order-path risk — the risk is designed out rather than accepted. **That is the
cleaner basis for the default decision than a documented acceptance.**

Until then, the conservative choice is to keep capture **off** in any deployment:
`False` is safe, and the isolation change is what makes `True` safe.

## Scope

Changes nothing. Both A and B are separate changes with their own scope, tests,
and approval. This note only establishes which one is sound.

## Safety state

No branch, 0 commits, 0 staged; tracker `44136fa3…`, drift `82a78c0d…`,
snapshot `52e39180…` unchanged.
