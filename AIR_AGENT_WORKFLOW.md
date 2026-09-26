# JetBrains Air — Apex OpenCode Workflow

JetBrains Air is an ACP frontend for this repository's OpenCode agent workflow. The
trading code and permissions remain project-local and editor-agnostic; switching to
VS Code, Codium, a terminal, or another OpenCode host uses the same workflow.

## Air connection

JetBrains Air launches OpenCode through Agent Client Protocol (ACP). The current Air
registration is:

```text
OpenCode
  command: <absolute path to opencode.exe>
  args:    ["acp"]
```

This registration is stored in JetBrains Air's user-level configuration. It is not part of
the repository and should never contain API keys or environment secrets. Do not edit it
unless OpenCode moves.

## Starting the workflow in Air

1. Open the repository in JetBrains Air.
2. Open AI Chat and choose **OpenCode** in the agent selector.
3. Start a new thread.
4. Use the `/apex` command to start the workflow manager, or use the **Session Mode**
   dropdown in the OpenCode chat settings and select **`apex-manager`**.

Because `default_agent` is set to `apex-manager` in this repository, new OpenCode sessions
in Air start with `apex-manager` already selected as the Session Mode.

Example:

```text
/apex Audit the order-idempotency path. Scope: src/exchange.py and tests/test_exchange.py.
Do not change protected paths without asking me. Preserve every plausible finding.
```

`/apex` explicitly selects the `apex-manager` OpenCode agent. If `apex-manager` does not
appear as a separate entry in Air's agent selector, that is expected: Air shows the
external ACP agent (`OpenCode`), while OpenCode's internal primary agents are selected with
its own controls or through a custom command. Use `/apex` for reliable switching.

The manager automatically invokes these subagents in sequence:

```text
apex-auditor-nexagi
  -> apex-finding-verifier
  -> apex-confirmed-fixer
  -> apex-change-verifier
  -> apex-manager final report
  -> Human review
```

Do not ask Air to combine all five roles into one prompt. OpenCode performs the actual
delegation, and each subagent has its own permissions, model, and skill.

## What OpenCode loads in Air

When Air starts OpenCode in this repository, OpenCode loads:

- `AGENTS.md` — safety contract, structured findings, verification/fix/verdict formats.
- `OPENCODE_AGENT_WORKFLOW.md` — full OpenCode procedure and permission model.
- `opencode.json` — project permissions and instruction loading.
- `.opencode/agents/*.md` — manager, auditors, verifiers, and fixer definitions.
- `.agents/skills/*/SKILL.md` — reusable role procedures.

## Air-specific notes

- Use OpenCode for this workflow. Hermes and OpenHands are separate ACP agents and do not
  inherit this project's manager/subagent chain.
- If Air's agent selector does not show `apex-manager`, start a new OpenCode thread and
  verify discovery with `opencode agent list` in the repository root.
- Air can display OpenCode's permission prompts. Approve only actions you understand.
- Protected trading paths require explicit, in-the-moment approval even if OpenCode also
  asks for confirmation.
- Air's ACP registration is user-local and cannot be staged in this repository.

## Verify the Air integration

From the repository root:

```text
opencode --version
opencode agent list
opencode debug skill
```

Expected agent names include `apex-manager` as a primary agent and the following
subagents:

```text
apex-auditor-nexagi
apex-finding-verifier
apex-confirmed-fixer
apex-change-verifier
```

Do not run `opencode debug config` casually: it can print provider credentials in its
resolved configuration output. If credentials are exposed in a log or chat, rotate the
affected provider key immediately.

## Compatibility

This workflow is project-local. It does not create, remove, or modify VS Code/Codium
configuration. The only editor-specific integration is JetBrains Air's user-level ACP
registration, which simply launches the same `opencode acp` command.
