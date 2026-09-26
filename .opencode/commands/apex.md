---
description: Start the Apex workflow manager
agent: apex-manager
subtask: false
---

Act as `apex-manager` according to `AGENTS.md`, `OPENCODE_AGENT_WORKFLOW.md`, and
`AIR_AGENT_WORKFLOW.md`.

User request:

$ARGUMENTS

If the user request above is empty or too broad, stop and ask for one bounded subsystem or
at most 20 source/test files. Do not audit the entire repository by default. If the request
is bounded, proceed through the fail-closed pipeline:

1. `apex-auditor-nexagi`
2. `apex-finding-verifier`
3. `apex-confirmed-fixer`
4. `apex-change-verifier`
5. Return the final manager report.

Do not edit files yourself. Do not claim a subagent ran unless it actually ran. Stop for
human approval before any protected-path change, and never commit or push.
