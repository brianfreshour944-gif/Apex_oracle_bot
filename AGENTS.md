# Apex Oracle Bot — OpenCode Agent Contract

This file is the shared contract for OpenCode and other compatible agents in this
repository. It defines a fail-closed CrewAI-like workflow using OpenCode primary agents,
subagents, permissions, and skills.

## Non-negotiable safety rules

These rules have the same force as `.clinerules`:

- Never run `git commit` or `git push`. Stage only when explicitly requested, show
  `git diff --staged`, and let the human commit and push.
- Never modify `.env`, `.env.example` values, API keys, or credential fields in
  `docker-compose.yml`. Report required changes instead.
- Never place a live trade, cancel a live order, or contact a real Alpaca endpoint.
  If paper/backtest mode is uncertain, stop and ask.
- Never run `docker compose down -v` or delete a volume.
- Never delete files outside this repository.
- Never claim a file or verification exists without showing proving command output.
- Never expose secrets in output, logs, diffs, or reports.

Before changing these high-stakes paths, explain the exact plan and wait for explicit,
in-the-moment user approval:

- `src/exchange.py` or `exchange.py`
- `src/risk.py` or `risk.py`
- `strategies.py`
- `src/committee/committee.py` or root `committee.py`
- any `brains/` or `src/committee/` path when changing trading behavior
- `database.py`, `db.py`, or `src/db.py` (schema changes must be additive)
- migrations or destructive database operations

## Sequential agent workflow

Treat this as a fail-closed pipeline. Each role must hand off output before the next starts:

1. **Manager / orchestrator** — defines scope, invokes the audit, routes findings to
   independent verification, and permits fixes only for confirmed findings.
2. **Auditor** — read-only. Produces evidence-backed, reproducible findings.
3. **Finding verifier** — read-only and independent. Assigns exactly `CONFIRMED`,
   `PLAUSIBLE`, or `REJECTED` by fresh reproduction.
4. **Fixer** — edits only `CONFIRMED` findings and makes the smallest safe change.
5. **Change verifier** — read-only. Reviews the diff and independently reruns validation.

The manager is not an expert substitute. The OpenCode primary `apex-manager` invokes
isolated, role-specific subagents through OpenCode's `task` tool and passes each complete
response forward as the next role's context. Each role loads its matching skill from
`.agents/skills/<role>/SKILL.md`. Do not collapse independent roles into one reasoning
context, and do not claim a subagent ran when it did not.

## Shared finding contract

Every finding must contain:

```text
Finding ID: APEX-<short-stable-id>
Claim: <one falsifiable sentence>
Severity: CRITICAL | HIGH | MEDIUM | LOW
Category: <risk | concurrency | correctness | data-integrity | exchange | tests | docs | other>
Evidence: <file:line citations and observed output>
Confidence: <0.0-1.0>
Does it matter: <impact and failure path>
Reproduction: <exact safe command, script, test, or read-only query>
Expected observation: <required result>
Actual observation: <observed result, or NOT_REPRODUCED>
Status: PROPOSED
```

- Check current file and line citations; a citation alone is not proof.
- Reasoning-only claims are at most `PLAUSIBLE`.
- Static analysis does not prove runtime behavior; state limitations.
- Prefer existing repro scripts and targeted tests.
- Query real data through a read-only DB/MCP tool when evidence depends on it.
- Never include credentials, tokens, environment dumps, or personal data.


## Shared verification contract

The finding verifier may add or update only verification fields:

```text
Verification ID: VER-<finding-id>
Finding ID: <finding-id>
Verdict: CONFIRMED | PLAUSIBLE | REJECTED
Reproduction command: <exact command>
Observed result: <actual output summary>
Missing evidence: <what prevents confirmation, or NONE>
Verified live-path impact: YES | NO | UNKNOWN
```

- `CONFIRMED` requires fresh output that reproduces the claim.
- `PLAUSIBLE` means direct safe reproduction was impossible or unavailable.
- `REJECTED` requires output that contradicts the claim.
- Never trust an auditor's narrative, confidence, or claimed output.
- Report stale citations rather than silently correcting them.

## Fix authorization and completion

The fixer must confirm that every finding it intends to change is `CONFIRMED`. Refuse
`PLAUSIBLE` and `REJECTED` findings. If a fix touches a protected path, stop for explicit
user approval before editing.

After editing:

1. Run the smallest relevant targeted test/reproduction.
2. Run the full suite: `pytest tests/`.
3. Run Ruff, preferably `uvx ruff check .` if Ruff is absent from the venv.
4. Show `git diff --check` and the relevant unstaged diff.
5. Do not stage unless explicitly requested. Never commit or push.
6. Summarize changes, reasons, failures, and limitations in plain language.

A pre-existing Ruff debt is not permission to hide new lint errors. Report baseline and
post-change results. If a required suite fails, diagnose it; stop for a decision if the
failure is unrelated.

## Change-verifier contract

The final verifier independently inspects the actual diff, maps edits to confirmed
findings, checks security/regression risk, reruns targeted and full validation, and
returns:

```text
FIX VERDICT: ACCEPTED | REJECTED | NEEDS_HUMAN_REVIEW
Accepted findings: <IDs or NONE>
Rejected/partial findings: <IDs or NONE>
Tests: <command + result>
Lint: <command + result>
Diff safety: <summary>
Residual risks: <summary or NONE>
```

Use `NEEDS_HUMAN_REVIEW` for high-stakes files, live/paper uncertainty, missing evidence,
or any condition requiring user approval. A fixer may not self-approve.

## OpenCode operation

Use the project-defined `apex-manager` as the primary agent. In OpenCode, select it with
the agent switcher or start it explicitly:

```text
opencode --agent apex-manager
```

Then provide a bounded scope and acceptance criteria. The manager must delegate in this
order:

1. `apex-auditor-nexagi` produces the bounded, evidence-backed audit.
2. `apex-finding-verifier` independently reproduces the complete audit output.
3. `apex-confirmed-fixer` fixes only findings explicitly marked `CONFIRMED`.
4. `apex-change-verifier` reviews the verification output and complete working-tree diff.
5. `apex-manager` presents the final pipeline status; the manager is the final voice, but
   only the independent verifier supplies the fix verdict.

The manager itself cannot edit files. Auditors and verifiers cannot edit files. The fixer
cannot invoke more agents, stage, commit, or push. Protected paths require explicit user
approval even when the configured edit permission asks for confirmation. See
`OPENCODE_AGENT_WORKFLOW.md` for the complete procedure and permission model. When running
OpenCode through JetBrains Air, also follow `AIR_AGENT_WORKFLOW.md`.
