---
name: apex-audit
description: Perform a read-only adversarial audit of Apex Oracle Bot and return structured, reproducible findings using the shared finding contract.
---

# Apex Auditor

Perform an adversarial, read-only code audit. Follow the `AGENTS.md` safety contract and
the finding format there.

## Method

1. Establish the requested scope and inspect real callers, not only isolated functions.
2. Ask whether safety logic is actually reachable from the live/paper execution path.
3. Prefer evidence that exercises behavior: existing checks, targeted tests, temporary
   reproducers outside the repository, or read-only database queries.
4. Distinguish production defects from defects in tests or verification harnesses.
5. Check documentation/code drift and concurrent or partial-failure behavior.
6. Assign severity and confidence independently. Do not inflate findings for effect.

## Scope and context limits

- Use only the explicit source/test paths in the task prompt; never default to a repository-wide audit.
- Exclude `data/`, `models/`, `.venv/`, `.git/`, `.opencode/node_modules/`, notebooks, `*.jsonl`, `*.db`, `*.pyc`, reports, and generated artifacts unless explicitly required.
- Keep the scope to at most 20 source/test files and use bounded line reads for large files.
- If the scope is too broad or the context budget is at risk, stop with `PIPELINE STATE: BLOCKED` and request a narrower scope.

## Read-only boundary

- Do not edit, create, stage, delete, commit, or push repository files.
- Do not run commands that place orders, cancel orders, contact live Alpaca, or write
  database state.
- Temporary executable reproducers may be created only outside the repository and must be
  cleaned up. Record the exact commands and observations in the report.

## Output

Return one or more findings in the shared contract, followed by:

```text
AUDIT SUMMARY
Scope inspected: <paths/behavior>
Production paths checked: <paths and caller evidence>
Commands executed: <read-only commands>
Not covered: <limitations>
Overall risk: <summary>
```

Do not claim a finding is proven unless its reproduction was actually executed. A finding
requiring live trading or unavailable state must remain `PLAUSIBLE` for independent
verification.
