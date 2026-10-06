---
name: code-fixer
description: This skill should be used when the user asks to "fix this bug", "make the tests pass", "fix the failing test", "repair this error", or mentions lint errors, type errors, tracebacks, or broken imports in a Python project.
triggers:
- fix this
- fix the bug
- make the tests pass
- failing test
- fix the error
- repair
- lint errors
---

# Code Fixer

Repair a specific, already-understood defect with the smallest correct change,
then prove it with a test. This skill is for fixing a named problem, not for
open-ended auditing. If the problem is not yet diagnosed, analyze it first.

This skill is portable across repositories. Discover the tooling before
editing.

## Step 0: discover the tooling

```bash
# test runner
cat pyproject.toml pytest.ini setup.cfg tox.ini 2>/dev/null | grep -iE "pytest|testpaths|addopts"
ls tests/ 2>/dev/null | head
# linter and formatter
grep -iE "ruff|black|flake8|mypy" pyproject.toml requirements.txt 2>/dev/null
# how tests are run
grep -A3 "\[tool.pytest" pyproject.toml 2>/dev/null
```

Record the exact test command and lint command. Use what the project already
has; do not introduce a new runner or linter.

## Procedure

1. Reproduce the defect first. Capture the exact command and the failing
   output. Do not start editing until you can see it fail.
2. Identify the root cause. Trace from the error into the code. Do not patch
   a symptom.
3. Make the smallest change that fixes the root cause. Do not refactor
   unrelated code, rename things, or reformat files.
4. Add or update a regression test that fails without the fix and passes with
   it. If a test is impractical, say why and show the manual reproduction.
5. Run the targeted test, then the full suite.
6. Run the project linter and separate pre-existing issues from new ones.
7. Show `git diff --check` and the unstaged diff.
8. Do not stage, commit, or push. Summarize what changed and why.

## Protected paths

If the repository declares high-stakes paths (for example risk, exchange, or
strategy modules, or a `.env`), stop and get explicit approval before editing
them. Check `AGENTS.md` and `.clinerules` for the list.

## Output

```text
FIX SUMMARY
Problem: <one-line description>
Root cause: <what was actually wrong>
Files changed: <paths and why>
Regression evidence: <command + result>
Full tests: <command + result>
Lint: <command + result, with baseline vs new>
Diff check: <command + result>
Not fixed / out of scope: <items or NONE>
Human review required: YES | NO
```

## Rules

- Never weaken a test, skip it, or delete an assertion to make a suite pass.
- Never silence a linter by widening an ignore list; fix the cause or add a
  targeted, justified suppression.
- Never touch credentials, `.env` values, or secrets.
- Never commit or push.
- One defect per fix. If you find others, list them separately.
