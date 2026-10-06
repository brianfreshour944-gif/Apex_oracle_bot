# Command Reference

Quick reference for running the Apex pipeline, the OpenCode tooling, and the
repository skills.

## Run the pipeline (JetBrains / OpenCode)

The pipeline is a fail-closed 5-stage workflow. One command starts it:

```text
/apex <bounded task>
```

Example:

```text
/apex Audit the order-idempotency path. Scope: src/exchange.py and its tests.
```

From the terminal:

```bash
opencode --agent apex-manager
```

```bash
opencode run --command apex "<bounded task>"
```

Running `/apex` with no task makes the manager stop and ask for a bounded
scope. Keep every task to one subsystem or at most 20 source/test files, or
the manager returns `PIPELINE STATE: BLOCKED` and asks you to narrow it.

## What the pipeline runs

You do not invoke these stages yourself. The manager dispatches them in order:

```text
apex-manager (Space Bunny)
  -> apex-auditor-nexagi (GLM 5.3 Flash)      read-only audit
  -> apex-finding-verifier (Solar Pro 4)      CONFIRMED / PLAUSIBLE / REJECTED
  -> apex-confirmed-fixer (Laguna S 2.1)      fixes CONFIRMED findings only
  -> apex-change-verifier (Space Bunny)       diff review, final verdict
  -> human                                    commit and push
```

A subagent shows no output while it runs. The manager announces each dispatch.
Silence is normal; a stage usually takes 3-6 minutes.

## Inspect the OpenCode setup

```bash
opencode agent list          # list configured agents
opencode models              # list available models
opencode debug config        # resolved configuration
opencode debug agent apex-manager
opencode debug skill         # verify skill discovery
```

## Validation commands

The fixer and change verifier run these and show the output:

```bash
pytest tests/
uvx ruff check .             # if Ruff is not in the active venv
git diff --check
git diff
```

## Git

The agents never stage, commit, or push. That is always the human's step:

```bash
git status
git diff                     # review what the fixer changed
git add -A
git commit -m "..."
git push
```

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` and load on demand. This is
the Agent Skills standard, so the same files also work in OpenHands.

| Skill | Purpose |
| - | - |
| `apex-manage` | Orchestrate the audit and fix pipeline |
| `apex-audit` | Read-only adversarial audit |
| `apex-verify-finding` | Independently reproduce and adjudicate findings |
| `apex-fix-confirmed` | Fix only CONFIRMED findings |
| `apex-verify-change` | Independently review the fix diff |
| `financial-analysis` | Review sizing, risk limits, PnL, execution cost, parity |
| `code-fixer` | Smallest correct fix for a named defect, with a regression test |
| `strategy-edge-analysis` | Check whether measured performance is a real edge |

Trigger them in a normal conversation with phrases like "analyze the
financials", "fix this bug", or "is this overfit".

## OpenHands

There is no `/apex` in OpenHands, and it does not read `.opencode/` or
`opencode.json`. Selecting this repository in OpenHands gives a single agent
conversation that can load the skills above, without the multi-stage pipeline
or its permission gates.

## Related documents

- `AGENTS.md` - safety rules and the shared finding/verification contracts
- `OPENCODE_AGENT_WORKFLOW.md` - the full pipeline procedure
- `AIR_AGENT_WORKFLOW.md` - JetBrains Air variant
- `.clinerules` - hard rules shared across the trading-bot repositories
