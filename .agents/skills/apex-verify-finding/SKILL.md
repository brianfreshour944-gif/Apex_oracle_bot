---
name: apex-verify-finding
description: Independently reproduce and adjudicate Apex Oracle Bot audit findings as CONFIRMED, PLAUSIBLE, or REJECTED before any fix is allowed.
---

# Apex Finding Verifier

Independently verify one or more audit findings. Follow the shared finding and
verification contracts in `AGENTS.md`.

## Procedure

For each finding:

1. Parse the finding ID, claim, evidence, and proposed reproduction.
2. Locate every cited file and current line yourself. Report stale citations explicitly.
3. Reconstruct the claimed execution path and identify whether it is production,
   test-only, or unknown.
4. Run the cheapest safe reproduction that actually exercises the claim. Prefer existing
   `vuln_check.py`, `audit_repro.py`, `verify_*.py`, or targeted pytest commands.
5. For DB/state claims, use an available read-only sqlite/postgres MCP query. Never infer
   real data solely from code.
6. Compare actual output with the claim. Do not accept an auditor's narrative or confidence.
7. Assign exactly one verdict:
   - `CONFIRMED`: fresh output reproduces the claim.
   - `PLAUSIBLE`: reasoning holds but a direct safe reproduction is impossible or missing.
   - `REJECTED`: fresh output contradicts the claim.

## Read-only boundary

Do not edit source, tests, configuration, git state, or live data. Never trade, cancel an
order, contact a live endpoint, commit, or push.

## Output

Return the original finding fields plus the shared verification contract, then:

```text
VERIFICATION SUMMARY
Total: <count>
Confirmed: <IDs>
Plausible: <IDs>
Rejected: <IDs>
Fix authorization: <IDs approved for the fixer, or NONE>
```
