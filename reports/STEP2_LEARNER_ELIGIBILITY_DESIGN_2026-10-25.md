# STEP 2 — Learner-eligibility design proposal (READ-ONLY)

Date: 2026-10-25. **READ-ONLY DESIGN REPORT. No source/test/config change, no DB
write, no historical repair, no learner/risk modification, nothing staged.**
Baseline frozen (`reports/BASELINE_FREEZE_2026-10-25.md`, HEAD `7a8ca12`).

**No implementation is authorized.** This report proposes rules for review only.

Governing principle, applied throughout: **`reconciled` does not mean `audited`,
and neither one automatically means `learner-eligible`.**

---

## 1. The three concepts (kept strictly separate)

| Concept | Question | Evidence source | Current value on the deployed snapshot |
| --- | --- | --- | --- |
| **Linkage confidence** | Does this order belong to this decision? | timestamps, qty ratio, `client_order_id` | HIGH — 11/11 exits uniquely linked (Item 3) |
| **Accounting confidence** | Can qty/fees/P&L be reconstructed reliably? | `reconstruct_position_outcome` verdict | 11/24 `reconciled`; **0 audited** |
| **Learner eligibility** | Is the outcome suitable for training? | a policy over the above | **undefined today — the subject of this report** |

**They are not nested.** Item 3 proved the 11 exits link with high confidence, yet
**none** became certifiable, because their fees were never recorded. Item 4
explained *why* the links were lost (code defects, since fixed) without changing
accounting confidence. So: **linkage ≠ accounting ≠ eligibility.** A design that
collapses them will mis-train the learner — the exact failure mode this report
exists to prevent.

---

## 2. Requirement 1 — how eligibility should treat verified / estimated / unknown fees

### Current reality (from the classifier + item-1 qualification)
- The exchange adapter **cannot observe** Alpaca's fee; it **estimates** at
  `ESTIMATED_TAKER_FEE_BPS=25` and sets `commission_estimated=True`
  (`exchange.py:951-952`).
- The reconstruction certifies a ledger round trip only when **both legs have a
  nonzero fee** (`db.py:655`, `unverified_fee_provenance`).
- Item 1 adds a `fee_source` column: `"estimated"` / `"exchange"` / `NULL`.
  The item-1 review proved `"exchange"` is **currently unreachable** in
  production — nothing but the estimator writes a positive fee.

### The three fee classes and their correct eligibility treatment

| Fee class | How identified | Certifiable? | Learner-eligible? |
| --- | --- | --- | --- |
| **Verified** (exchange-observed) | `fee_source="exchange"` | yes | yes — **but none exist today** |
| **Estimated** (adapter estimate) | `fee_source="estimated"` | yes (certifies *with an estimate*) | **yes, with a recorded caveat** |
| **Unknown** (0.0 / NULL) | fee `0.0` or `fee_source IS NULL` | **no** | **no** |

### The design rule
> **Eligibility requires a nonzero fee on every leg, with provenance recorded.**
> An estimated fee is sufficient for *eligibility* (it is the best available
> evidence and is what the live path produces); an unknown fee is not.

**Why estimated is allowed:** refusing estimated fees would exclude **100%** of
the live path, because the adapter *only* ever estimates. That would make the
learner permanently un-trainable on live data — a worse failure than training on
a documented estimate.

**Why unknown is refused:** a recorded `0.0` means "no fee recorded", not
"fee-free". Certifying it would manufacture certainty (Item 3 §4c: the 11
orphaned exits are exactly this shape).

### The critical non-claim
Eligibility under an estimated fee is **"reconciled with estimated costs"**, never
**"audited"**. Any downstream metric that reads learner eligibility must not
describe it as verified P&L. `fee_source` is the machine-readable way to carry
that distinction — **which is why item 1 must land before any filter is enabled.**

---

## 3. Requirement 2 — legacy rows with NULL / missing `outcome_status`

### Facts
- The deployed snapshot **has no `outcome_status` column** (predates it).
- A naive `WHERE outcome_status='reconciled'` therefore excludes **all 24 rows** —
  the learner would get **zero** history and never validate.

### Options (evidence-based)

| Option | Behaviour on legacy NULL | Deployed-snapshot result | Verdict |
| --- | --- | --- | --- |
| **A — exclude only explicit `'unreconciled'`; keep `NULL`** | legacy retained | 24/24 kept | **Recommended** |
| B — allow-list `'reconciled'`; legacy kept only if \|return_pct\| ≤ sane band | legacy mostly retained | ~21/24 (drops the −805% / −70218% outliers) | Viable alternative |
| C — strict `WHERE ='reconciled'` | legacy excluded entirely | **0/24** | **Rejected** — zeroes history |

### The rule
> **Treat `NULL` as "unknown", never as "reconciled". Under option A, legacy rows
> remain eligible but are marked `legacy_unknown`; they are not promoted to
> certified.** Their fees are still unknown, so they carry the same caveat as §2's
> "estimated" class only if a fee is actually present; where the fee is `0.0`,
> §2's rule excludes them regardless.

**Note the interaction:** option A keeps 24/24 rows *for linkage purposes*, but
§2's fee rule still excludes every fee-free row. So A is permissive about
*provenance labelling* while §2 stays strict about *fee presence*. The two rules
compose; neither overrides the other.

**Recommendation:** adopt **A + §2** as the default, with **B** as the fallback
for the specific case of corrupt `return_pct` outliers if the learner is found to
consume `return_pct` directly (it does — see §4, L3).

---

## 4. Requirement 3 — minimum sample size and the risk of excluding whole categories

### Measured impact on the deployed snapshot

| Scenario | Kept | `sideways` | `low_volatility` | `neutral` | `bull` | BTC | ETH | SOL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **S0 all rows** | 24 | 11 | 10 | 2 | 1 | 10 | 6 | 8 |
| **S1 reconciled-only** | **11** | **9** | **0** | 2 | 0 | **0** | 4 | 7 |
| **S2 exclude explicit unreconciled, keep NULL** | 24 | 11 | 10 | 2 | 1 | 10 | 6 | 8 |

### The threshold collapse (decisive)
Per-regime gate `VALIDATION_MIN_TRADES = 10` (`adaptive_meta.py:40`):

| Regime | All | Reconciled | Effect |
| --- | ---: | ---: | --- |
| `sideways` | 11 | **9** | **falls below 10 — stops validating** |
| `low_volatility` | 10 | **0** | **erased entirely** |
| `neutral` | 2 | 2 | already < 10 |
| `bull` | 1 | 0 | already < 10 |

**A naive reconciled-only filter removes every BTC row and every `low_volatility`
row, and pushes `sideways` below the gate it currently passes.** That is a
**non-random, symbol- and regime-biased** deletion driven by a *linkage bug*
(Item 4), not by strategy quality. The learner would silently stop validating
regimes it validates today.

### Minimum-sample rules (proposed)
1. **Never filter below a per-regime minimum of `VALIDATION_MIN_TRADES` (10)**
   without an explicit warning — if a filter would take a regime from ≥10 to <10,
   that regime's learned weights must be **frozen at their current value**, not
   silently reset.
2. **Require a minimum eligible count before enabling any filter** (proposal:
   `n_eligible ≥ max(VALIDATION_MIN_TRADES, 2 × n_regimes)`), so a young system
   cannot filter itself into an untrainable state.
3. **Prefer widening history over tightening eligibility** when the two conflict
   (i.e. wait for post-fix data rather than drop pre-fix data).

### The sequencing recommendation (unchanged from Item 2/3)
Because the exclusions are **linkage-caused**, the higher-value order is:
**let post-2026-10-13 data accumulate (already correctly keyed + fee-estimated) →
then apply the filter**, rather than filtering the uncertifiable historical window.
This *adds* eligible rows instead of deleting them.

---

## 5. Requirement 4 — which consumers need separate eligibility controls

Filtering must happen **at each learner call site**, never in the shared reader.

### The shared reader must NOT change
`get_closed_decision_snapshots()` has **four** call sites, two of which are
learners and two of which are **not** risk controls but share the same reader:

| Call site | Consumer | Filter? |
| --- | --- | --- |
| `bot.py:3606` | adaptive-learner bootstrap | **yes** (learner) |
| `decision_transformer.py:731` | DT inference context | **yes** (learner) |
| `decision_transformer.py:1002` | DT training data | **yes** (learner) |
| `evolutionary_ppo_trainer.py:368` | PPO training | **yes** (learner) |

**None of these is a risk control** — but changing the reader would also change
any *future* caller. So the filter belongs at the four sites above, individually.

### Consumers that need **separate** controls (NOT learner filtering)
These read the same outcomes but **change trading behavior**; they are out of
scope and must not be touched by a learner filter:

| Consumer | Path | Why separate |
| --- | --- | --- |
| Rolling soft loss gate | `bot.py:2172` → `db.py:1670` `get_recent_realized_pnl` | sums **every** closed row by design (fail-safe); filtering changes entry blocking |
| Kelly size cap | `risk.py:957` → `performance_tracker.py:251` | win-rate/payoff → position size |
| Sharpe/win-rate decay alerts | `performance_tracker.py:206-227` | can tighten size |
| Brain-C validation gate | `adaptive_meta.py:453-474` | gates whether weights go live |

### Live-path labels (L1/L4/L5 from Item 2)
`learner.update`, the live-buffer append (`bot.py:295`), and the online step
(`bot.py:356`) compute labels from the **just-closed** trade's `outcome_status`,
which is available in-line. These need their **own** guard at the call site.

**Per-consumer eligibility matrix (proposed):**

| Consumer | Apply fee rule (§2)? | Apply legacy rule (§3)? | Notes |
| --- | --- | --- | --- |
| L1 adaptive `update` | yes | yes | live `outcome_status` available |
| L2 bootstrap `rebuild_from_history` | yes | yes | historical; main exposure |
| L3 strategy learner + perf tracker | yes | yes | **but** perf tracker is also a §5 risk input — see below |
| L4 live buffer | yes | n/a | live |
| L5 online step | yes | n/a | live |
| L6/L7 decision transformer | yes | yes | uses `realized_pnl/100` |
| L8 PPO | yes | yes | reward ∝ \|pnl\| |
| L9 OOD | **no** | **no** | features only, not a label consumer |

**The L3 conflict (flagged, unresolved):** `strategy_selector` feeds **both** the
strategy learner (a learner) **and** `performance_tracker` (a §5 sizing input).
Applying a learner filter there would leak into sizing. **Recommendation: split
the two** — filter only the learner branch; leave the tracker on all rows. This
needs its own analysis before implementation.

---

## 6. Requirement 5 — distinguishing the three confidences (operational definition)

Proposed machine-readable separation, so no consumer can conflate them:

```
linkage_confidence:  high | low        # can we attribute the legs? (Item 3/4 evidence)
accounting_confidence: verified | estimated | unknown   # from fee_source + reconstruction
learner_eligible:    bool              # policy output, requires:
                                         #   accounting_confidence != unknown
                                         #   AND linkage_confidence == high
                                         #   AND row is not explicit 'unreconciled'
```

- **`reconciled` maps to `accounting_confidence ∈ {verified, estimated}`** — never
  automatically `verified`.
- **`learner_eligible` is computed, not read** — it is a policy, so it can change
  without rewriting the stored row.
- **No stored field should be named `audited`** until an independent
  exchange-statement reconciliation exists (it does not).

---

## 7. Unresolved questions

1. Should **live** `unreconciled` closes be skipped, or only historical ones?
2. Option **A vs B** for legacy `NULL` rows — keep-all, or sane-band fallback for
   the two corrupt `return_pct` outliers (`f110556591` −805%, `35e92c4ad1`
   −70218%)? These two are a **sign-flip** risk for L1/L3.
3. How to **split L3** so the strategy learner is filtered but the performance
   tracker is not.
4. Should a regime whose count would drop below `VALIDATION_MIN_TRADES` be
   **frozen** (recommended) or reset?
5. Is there value in a **separate `audited` tier** ever, given no exchange
   statement reconciliation exists?

## 8. Non-goals (explicit)

- No implementation, no source/test change, no DB write.
- No change to shared readers, risk controls, learner behavior, or the validation
  gate.
- No historical relink/backfill/repair.
- No staging, commit, deploy, restart, or orders.

## 9. Recommendations (summary)

1. **Fee rule:** eligible iff every leg has a **nonzero** fee; estimated is
   eligible-with-caveat; unknown is not.
2. **Legacy rule:** option **A** (exclude only explicit `'unreconciled'`, keep
   `NULL` as `legacy_unknown`); option B as fallback for corrupt outliers.
3. **Sample rule:** never let a filter push a regime below
   `VALIDATION_MIN_TRADES`; freeze rather than reset; require a minimum eligible
   count before enabling any filter.
4. **Scope:** filter **only** at the four shared-reader learner call sites + the
   three live-label sites; **split L3**; leave the rolling loss gate, Kelly, decay
   alerts, and the validation gate untouched.
5. **Sequencing:** prefer accumulating post-fix data over filtering the
   uncertifiable historical window.
6. **Naming:** never label anything `audited`; `reconciled` = ledger-balanced with
   (at best, estimated) fees — **not** verified.
