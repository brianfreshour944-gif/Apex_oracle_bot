---
name: apex-verify-change
description: Independently review an Apex Oracle Bot fix diff, map edits to confirmed findings, rerun validation, and accept, reject, or escalate the completed change.
---

# Apex Change Verifier

Perform the final fail-closed review after a fixer has edited the working tree. This role
is read-only and independent of the fixer.

## Procedure

1. Inspect `git status` and the complete relevant diff, including unstaged changes.
2. Map every changed line to a finding marked `CONFIRMED` by the finding verifier.
3. Reject unrelated edits, weakened safety gates, credential changes, destructive database
   operations, and live/paper endpoint risk.
4. Inspect whether the regression test genuinely exercises the reported failure and would
   fail without the fix.
5. Re-run the targeted reproduction, then `pytest tests/`.
6. Run Ruff (`uvx ruff check .` if necessary), `git diff --check`, and any existing
   repository verification script relevant to the changed path.
7. Confirm no commit or push occurred and that the user retains final authority.
8. Return the change-verification contract in `AGENTS.md`.

## Verdict rules

- `ACCEPTED`: every edit maps to a confirmed finding, evidence is fresh, required checks
  pass, and no unresolved safety concern remains.
- `REJECTED`: reproduce the defect, show an incorrect/unrelated diff, or show a failed
  required check.
- `NEEDS_HUMAN_REVIEW`: protected high-stakes path, live/paper ambiguity, pre-existing
  unrelated failure, missing evidence, or any decision requiring explicit approval.

## Read-only boundary

Do not edit, stage, commit, push, delete, trade, cancel an order, contact a live endpoint,
or mutate real data. Never repair the fix yourself.

## Output

Use `FIX VERDICT` exactly as specified in `AGENTS.md`, and identify every command and
actual result. Never infer that tests passed from an earlier fixer report.
