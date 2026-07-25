# Research-agent provider runtime grouping (2026-07-19)

## Decision

The user explicitly retired compatibility with pre-v1 diagnostic runs and
authorized fresh E3/H2/E2 and Figure 2 runs after the architecture cleanup.
The old flat provider import paths are therefore removed rather than retained
as facade files.

## Change

- Moved the production LLM client to `providers/llm.py`.
- Moved deterministic offline clients to `providers/mocks.py`; importing the
  production provider no longer initializes the mock layer.
- Moved cost metering and structured retry to `providers/`.
- Moved the versioned prompt loader and all four prompt text files to
  `providers/prompts/` and declared the text files as package data.
- Rewired production code to import provider-neutral message/protocol types
  from `providers/protocol.py` and mock clients from `providers/mocks.py`.
- Retired the five old top-level modules and the old `prompts/` namespace.
- Hardened the release test to require the canonical modules and prompt text
  in both sdist and wheel, then import them from the extracted wheel.

The top-level research-agent Python-file count falls from 87 to 82 in this
batch (160 before the breaking facade-retirement program).

## Figure 2 authority consequence

Two evaluator leakage guards now inspect the canonical prompt location. This
changes the scorer-tree digest. Because no final Figure 2 result will reuse
the old diagnostic authority, the current paper rubric is explicitly
re-authorized as `easyicu.figure2_paper_rubric/20260719-v2`; the old
2026-07-18 authority is not represented as unchanged. This is a pre-run
authority supersession: no published or canonical scorecard exists under the
retired reference, and no completed experiment result was rewritten.

## Verification

- Provider, retry, cost, replication, prompt, and retired-path focused suite:
  337 tests exercised; one stale monkeypatch was redirected to the canonical
  mock module, then the affected 165-test slice passed.
- Figure 2 rubric/scoring authority suite: 170 passed.
- Ruff, compileall, module-graph diff, and `git diff --check`: passed.
- Release sdist/wheel package-data and extracted-wheel import smoke: 1 passed.
- Cache and audit-cache identity suite: 30 passed.
- Characterization golden, meta-generalization, and capability registry: 31
  passed.
- The module-graph supported-surface gate now requires all seven canonical
  provider modules; the architecture baseline was regenerated under tool
  version 1.3.0.
