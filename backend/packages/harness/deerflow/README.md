# DeerFlow Agent Kernel

`deerflow` is the Agent kernel package.

Responsibilities:

- Agent planning and runtime graph execution.
- LLM model gateway and role policy selection.
- Tool registry, permission checks, and tool call management.
- Sandbox client integration.
- Runtime event streaming and durable Agent store integration.
- Feedback analysis and verification support.
- Memory hooks used during Agent task execution.

The product platform imports this package from `backend/app`; `deerflow` should
avoid owning HTTP routes, auth flows, or product API schemas.
