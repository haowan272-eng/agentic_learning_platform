# Skills

This layer stores prompt-level Skills. A Skill is reusable task guidance that
the Agent Runtime can retrieve and inject into prompts; it is not a Tool and
does not execute external actions.

Suggested layout:

```text
skills/
|-- builtin/         # Built-in Skills shipped with the product
|-- generated/       # Online-evolved Skills created from durable feedback
|-- registry.json    # Optional catalog metadata
`-- README.md
```

Runtime behavior:

1. The planner retrieves a small set of relevant Skills from built-in,
   generated, and user-level Skill directories.
2. Only high-confidence matches are expanded into full `SKILL.md` instructions
   and appended to the Agent prompt.
3. When online evolution is enabled, completed tasks can create or merge
   generated Skills with provenance logs and version snapshots.

Generated Skill evolution artifacts are written under
`.agentic_learning_rag/skill-evolution/`.

Cross-language Skill payload contracts belong in `contracts/`.
