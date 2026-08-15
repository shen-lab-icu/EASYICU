# Runtime restart and workspace hardening

- Human plan review now persists a digest-bound checkpoint and resumes on a fresh `ResearchAgentPipeline` without rerunning Planner.
- Web review recovery preserves the paused Provider hard-stop ledger and rejects study, model, configuration, artifact, or checkpoint drift.
- Pi workspace static checks bind `checked_sha256`; preview requires the same bytes. JavaScript is checked from a 0600 immutable copy.
- One Pi turn shares an eight-mutation write/edit ceiling.
- Web startup takes an OS file lease and rejects a second worker until lifecycle ownership is externalized.
- Focused evidence: 104 review/lifecycle tests, 116 Web/provider adjacency tests, 33 new direct regressions, workspace 24/24; Ruff, Node parse, diff check, and architecture ratchet passed. Full CI intentionally deferred under E1 development policy.
