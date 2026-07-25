# Research-agent core package grouping (2026-07-19)

## Decision

The pre-v1 flat Agent implementation paths are retired. The next E3/H2/E2
and Figure 2 runs will be fresh, so no old diagnostic capsule or import path is
allowed to keep the open-source package flat.

## Change

- Moved the twelve planner/coder/analyzer/writer role implementations from
  `agents.py` to `agents/core.py`.
- Moved the optional CLI-backed coder from `agentic_coder.py` to
  `agents/agentic_coder.py`.
- Added a lazy `agents/__init__.py` that exposes only public Agent classes and
  the optional coder adapter; private parser and prompt helpers stay in
  `agents.core`.
- Rewired production consumers to import the implementation module directly,
  while the root public API preserves Agent-class object identity.
- Retired the old flat files rather than leaving facade files.
- Added source-tree, lazy-import, root-identity, module-graph, and built-wheel
  boundary checks.

The top-level research-agent Python-file count falls from 82 to 80 in this
batch (160 before the breaking facade-retirement program).

## Authority consequence

Agent class `__module__` coordinates and the engine source-tree digest change.
This deliberately invalidates pre-v1 caches and capsules. No completed
canonical E3/H2/E2 or Figure 2 result is reused; all paper experiments will be
run fresh after the architecture freeze.

## Verification

- Agent package, agentic coder, parser, analysis-family, capability, and repair
  boundary slice: 73 passed, 1 skipped after correcting five stale private
  helper imports.
- Direct Agent/prompt/control-plane slice: 683 passed after correcting one
  stale root-module import and two old prompt filesystem paths.
- Retired-path and Agent package boundary: 168 passed.
- Figure 2 scorer and release archive/wheel smoke: 68 passed.
- Characterization golden, meta-generalization, and capability registry: 31
  passed.
- Module graph remains acyclic; architecture measurement gate has no
  lower-is-better regression.
