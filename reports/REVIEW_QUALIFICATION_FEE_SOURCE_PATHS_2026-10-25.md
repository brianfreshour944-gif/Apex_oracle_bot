# REVIEW QUALIFICATION — fee-provenance source paths (read-only)

Date: 2026-10-25. **Read-only inspection. No code changed, nothing staged, not
deployed.** Item 1 stays on hold exactly as implemented.

Purpose: answer the reviewer's qualification — *"could a positive commission ever
be supplied by a fallback or another estimate without the `commission_estimated`
flag set?"* — before `fee_source="exchange"` is treated as reliable.

---

## 1. The rule under review

`src/bot.py:_persist_order_record` infers provenance:

```python
if order_result.get("commission_estimated"):
    _fee_source = "estimated"
elif _commission > 0.0:
    _fee_source = "exchange"   # <-- the assumption under scrutiny
else:
    _fee_source = None
```

`"exchange"` is claimed whenever a commission is positive **and** the flag is
absent. That is safe only if "positive + no flag" is genuinely exchange-observed.
It is an *inferred* label, not a declared one.

## 2. Every producer of a positive commission in `src/`

Exhaustive grep for a commission value being set (not read) across `src/`:

| Location | Sets commission? | Sets the flag? |
| --- | --- | --- |
| `exchange.py:951` `_apply_estimated_commission` | **yes, the only positive writer** | **yes, always (952)** |
| `exchange.py:593` `get_orders` | `0.0` | n/a |
| `exchange.py:866/879/890` `create_order` fill paths | `0.0` | n/a |
| `exchange.py` `get_order`, `_find_existing_order_by_client_id`, `_protective_stop_info`, `create_order`'s `order_info` literal | **no `commission` key at all** | n/a |
| `risk.py` | none | n/a |
| `bot.py` | passes dicts through; no positive writer | n/a |

**Conclusion:** in the entire production source tree there is exactly **one**
place a positive commission is created — `exchange.py:951` — and it **always**
sets `commission_estimated = True` on the next line. The only other positive
commissions anywhere are inside **test doubles** (§4).

## 3. The "observed commission" early-return is unreachable in production

`_apply_estimated_commission` returns early without the flag when
`commission > 0` (line 942) — the branch that would represent an
exchange-observed fee. For that branch to be taken, a caller must hand in a dict
that already has a positive `commission`.

Traced every caller:

- `get_order` / `get_orders` / `_find_existing_order_by_client_id` /
  `_protective_stop_info` / `create_order`'s `order_info` literal — **none set
  `commission`**, so `order_info.get("commission", 0.0)` is `0.0`.
- The Alpaca SDK `Order` has no `.commission` attribute (asserted by
  `tests/test_alpaca_sdk_contract.py`: `"commission" not in Order.model_fields`),
  so the adapter cannot read one.
- The one caller that passes an already-processed dict — `bot.py:1181`
  (`fresh = AlpacaExchange._apply_estimated_commission(fresh)`) — re-runs the
  estimator, which is idempotent and leaves the flag set.

Empirical branch sweep (real code, all branches):

```
rate= 25.0 filled, comm=0, rate>0      -> commission=0.2500 flag=True
rate=  0.0 filled, comm=0, rate>0      -> commission=0.0000 flag=False
rate= 25.0 filled, comm>0 (observed)   -> commission=2.5000 flag=False   <-- only way to get positive+no-flag
rate= 25.0 filled, price=0             -> commission=0.0000 flag=False
rate= 25.0 filled, qty=0               -> commission=0.0000 flag=False
```

**So:** a positive commission with no flag is reachable **only** by a caller
injecting one directly. No production caller does. Therefore
`fee_source="exchange"` is currently **unreachable on the live path** — the
branch is effectively dead code today.

## 4. Where the assumption *is* breakable: test doubles

Several test fixtures inject a positive commission with **no flag**:

| File | Value |
| --- | --- |
| `tests/test_integration_trading_loop.py` | `"commission": 5.0` (×5 sites) |
| `tests/test_committee_min_hold.py` | `"commission": 1.0` |
| `tests/test_fee_recording.py` | `commission: 2.5` (observed-fee case) |

If such a dict reached `_persist_order_record`, it would be labelled
`"exchange"`. That is exactly the reviewer's scenario: a positive fee from a
**non-exchange source** (a mock, or a future fallback) promoted to
"exchange-observed" purely because the flag was absent.

This does not mislabel production data today, but it demonstrates the label is
**not self-enforcing** — it rests on an invariant ("nothing but the estimator
ever writes a positive commission") that no code enforces and a future fallback
could silently break.

## 5. Direct answer to the qualification

| Question | Answer |
| --- | --- |
| Can a positive commission be produced by a fallback/estimate *without* the flag, in production? | **No** — every production positive commission comes from `exchange.py:951`, which always sets the flag |
| Is `fee_source="exchange"` reachable on the live path today? | **No** — no production caller injects a positive commission |
| Is the "positive + no flag ⇒ exchange" assumption *enforced*? | **No** — it is an unenforced invariant |
| Could a mock/fallback positive fee be mislabelled "exchange"? | **Yes** — demonstrated by the test doubles in §4 |
| Is any production row currently mislabelled? | **No** |

**Net:** the reviewer's qualification is correct and the label is safe *today*
only because of an invariant that nothing guarantees. The risk is future, not
present.

## 6. Recommended future hardening (NOT authorized, NOT implemented)

Make provenance **declared at the source** instead of inferred downstream. Two
options, both requiring separate authorization:

- **Preferred:** have the producer stamp provenance on the dict when it
  sets/observes the fee (e.g. the adapter sets `commission_source` alongside
  `commission`), and have `_persist_order_record` persist that field verbatim —
  never infer "exchange" from "positive + no flag". Then a mock/fallback value
  can never be silently promoted.
- **Minimal:** require an explicit positive signal (e.g. only the
  observed-commission early-return sets `"exchange"`), so absence of the
  estimate flag is not, by itself, treated as evidence of exchange origin.

Neither is in scope now. The current `"exchange"` value is retained per the
approved design, understood as **currently unreachable / future-supported**.

## 7. Audit-trail record (both checksums preserved)

| Artifact | Value |
| --- | --- |
| Accounting patch sha256 **before** item-1 edits | `d1131e48b4c58e4f21db4f30aeca196c116b2a7cff2e37ac5cd8907cdfbfd4a9` |
| Accounting patch sha256 **after** item-1 edits | `8103cc41fb5c69921167c9d646123a556a2ab9ef9b20e088f3ac5a4932b74ff7` |
| Isolated-diff proof | HEAD `7a8ca12` + `all-tracked-changes.patch`, diffed vs working tree ⇒ **only `fee_source` lines differ**; the patch's own hunks are byte-identical |

The checksum moved solely because the edits live in the same two files
(`bot.py`, `db.py`) the patch touches. Both values are kept; the original
`d1131e48…` record is **not** overwritten.

## 8. Status

- Item 1: implemented and tested; **on hold**, not deployed.
- Items 2–5: untouched, unauthorized.
- Nothing changed by this inspection: 0 staged, no commit/deploy/restart/order.
- Momentum72 artifacts and the accounting patch: preserved unchanged.
