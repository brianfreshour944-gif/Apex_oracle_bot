---
name: apex-fix-confirmed
description: Fix only independently CONFIRMED Apex Oracle Bot findings with minimal changes, regression tests, full pytest validation, and Ruff reporting.
---

# Apex Confirmed-Finding Fixer

Fix verified defects only. Follow `AGENTS.md` and the shared contracts.

## Entry gate

Before editing:

- Read the complete finding-verification report.
- Confirm every finding to be fixed has `Verdict: CONFIRMED` and a successful fresh
  reproduction.
- Refuse `PLAUSIBLE` and `REJECTED` findings and identify what additional evidence would
  be needed.
- If a proposed fix touches a protected high-stakes path, stop and request explicit,
  in-the-moment user approval before editing it.

## Fix procedure

1. Reproduce the confirmed issue before changing code.
2. Make the smallest root-cause fix. Do not refactor unrelated code or broaden scope.
3. Add or update a regression test that fails for the original defect and passes after the
   fix whenever practical.
4. Re-run the targeted test/reproducer.
5. Run `pytest tests/`.
6. Run Ruff (`uvx ruff check .` if Ruff is not installed in the active venv) and separate
   pre-existing debt from new errors.
7. Run `git diff --check` and show the relevant unstaged diff.
8. Do not stage unless explicitly requested. Never commit or push.

## Completion output

```text
FIX SUMMARY
Fixed findings: <IDs and one-line explanation each>
Files changed: <paths and why>
Regression evidence: <targeted command + result>
Full tests: <command + result>
Ruff: <command + result>
Diff check: <command + result>
Unresolved/plausible findings: <IDs or NONE>
Human review required: YES | NO
```

Do not self-approve the fix. A fresh `apex-verify-change` role must inspect it.
