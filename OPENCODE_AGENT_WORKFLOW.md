# OpenCode Multi-Agent Workflow

This repository uses OpenCode's native primary-agent, subagent, task-permission, project
instruction, and skill systems with a fail-closed, CrewAI-like sequence:

```text
apex-manager (Space Bunny: thinker/final voice)
  -> apex-auditor-nexagi (GLM 5.3 Flash)
  -> apex-finding-verifier (Solar Pro 4)
  -> apex-confirmed-fixer (Laguna S 2.1)
  -> apex-change-verifier (Space Bunny)
  -> apex-manager final report
  -> Human review
```

## What is configured

- `AGENTS.md` is loaded as the project-wide safety and handoff contract.
- `opencode.json` loads this guide and blocks dangerous Git, volume-delete, and direct
  network commands.
- `.opencode/agents/apex-manager.md` defines the Space Bunny primary orchestrator/final voice.
- `.opencode/agents/apex-auditor-nexagi.md` defines the read-only GLM 5.3 Flash auditor.
- `.opencode/agents/apex-finding-verifier.md` defines Solar Pro 4 reproduction/verdicts.
- `.opencode/agents/apex-confirmed-fixer.md` defines Laguna S 2.1 minimal confirmed-finding fixes.
- `.opencode/agents/apex-change-verifier.md` defines Space Bunny's final read-only diff review.
- `.agents/skills/*/SKILL.md` contains reusable role instructions loaded on demand.

All critical roles now run on paid models, because the previous free-tier assignments
repeatedly failed with upstream rate limits and provider overloads.

| Role | Model |
| --- | --- |
| Manager, thinker, final voice | `openrouter/stealth/space-bunny-alpha` |
| Auditor | `openrouter/z-ai/glm-5.3-flash` |
| Finding verifier | `openrouter/upstage/solar-pro4` |
| Confirmed fixer | `openrouter/poolside/laguna-s-2.1` |
| Change verifier | `openrouter/stealth/space-bunny-alpha` |

The manager and the change verifier deliberately share Space Bunny for reliability. The
auditor, verifier, and fixer use different models, so the audit and verification gates
remain model-independent. Every model in this table passed a live smoke test.

The obsolete `apex-auditor-glm` file is not in the manager's allowed task list and is not part
of the active workflow.

## Context and scope limits

The repository contains very large local data and model artifacts, including historical data
and binary model files. The workflow therefore fails closed rather than attempting a full
repository dump:

- Give each audit a specific subsystem or at most 20 source/test files.
- Exclude `data/`, `models/`, `.venv/`, `.git/`, `.opencode/node_modules/`, notebooks, JSONL,
  databases, compiled files, reports, and generated artifacts unless a finding explicitly
  requires one.
- Use targeted searches and bounded line reads.
- If no scope is supplied, the manager asks you to choose a subsystem; it does not audit
  everything by default.
- A context-limit failure is `BLOCKED`, not a clean audit and not permission to continue.

## Start the workflow

List the configured agents:

```text
opencode agent list
```

Start the manager directly:

```text
opencode --agent apex-manager
```

Or select `apex-manager` from OpenCode's agent switcher. Alternatively, use the project
command `/apex` in an interactive OpenCode/Air session, or invoke the same command
non-interactively:

```text
opencode run --command apex "<bounded request>"
```

For example:

```text
Audit and repair the order-idempotency path. Scope: src/exchange.py and its tests.
Do not change protected paths without asking me. Preserve every plausible finding.
```

The manager is responsible for calling subagents in sequence. You do not need to open five
sessions manually.

## Permission model

### Manager

- Cannot edit files.
- Can invoke only the four named Apex workflow subagents.
- Cannot use web search/fetch.
- Passes complete prior responses as context and stops on incomplete evidence.

### Auditors and verifiers

- Cannot edit files or invoke further agents.
- Cannot use web search/fetch.
- Can inspect code and run approved, safe reproductions through OpenCode's bash permission.
- Must output the structured contracts in `AGENTS.md`.

### Fixer

- Can edit only after receiving a complete verifier report containing `CONFIRMED` findings.
- Is denied access to `.env` and `.env.*` and cannot invoke more agents.
- Protected source paths require both an explicit user response to OpenCode's approval
  request and the repository's higher-level instruction to wait for in-the-moment approval.
- Is denied `git commit`, `git push`, destructive Git cleanup, `docker compose down -v`,
  Docker volume deletion, `curl`, and `wget`.
- Cannot stage because no staging permission or command is granted by the agent prompt.

## Safety gates

- The GLM audit is independent of the Solar Pro 4 verifier; neither can edit or authorize a fix.
- The finding verifier must reproduce each claim before assigning a verdict.
- A fixer may act only on explicit `CONFIRMED` findings with fresh evidence.
- `PLAUSIBLE` findings remain human decisions and are never silently fixed.
- The final change verifier reviews the actual diff and independently reruns validation.
- Audit and verification roles must not contact live Alpaca endpoints or mutate real data.
- A human reviews the final diff and performs commit/push.

## Model selection

Every active role has an explicit provider-qualified model as listed above. To change a
role's model, edit its `model:` line in `.opencode/agents/*.md` using an ID from:

```text
opencode models
```

Use provider-qualified IDs such as `openrouter/...`; never place API keys in agent files.

## Troubleshooting

Inspect the resolved configuration:

```text
opencode debug config
opencode debug agent apex-manager
opencode debug agent apex-confirmed-fixer
```

Verify skill discovery:

```text
opencode debug skill
```

If a subagent is unavailable, check the manager's `task` permission and the exact subagent
name. If a permission appears in the wrong order, remember that OpenCode permission rules
use last-match-wins ordering.

## Validation expectations

The fixer and final verifier must independently run and show:

```text
pytest tests/
uvx ruff check .                 # if Ruff is absent from the active venv
git diff --check
git diff
```

If the full suite or Ruff has pre-existing failures, compare with the pre-change baseline,
identify new failures, and do not claim completion until the new failures are resolved or
the user explicitly accepts the limitation.
