# 2026-08-14 — research-agent duplicate-helper audit and first consolidation

## Scope

Audit whether post-decomposition files are necessary and whether the package
reinvents shared infrastructure. Then implement the first low-risk
consolidation and a CI ratchet that prevents regression.

## Measured baseline

- 517 Python modules / 5,232 top-level definitions.
- 129 names defined in more than one module after this batch (130 before).
- Module graph remains zero SCC.
- Zero-in-degree modules remain governed by
  `docs/research_agent_capability_inventory.md`; no bulk deletion authorized.

Full classification and the library-vs-domain judgment live in
`docs/research_agent_duplication_audit.md`.

## Consolidation: `_sha256_file` (16 → 0 local definitions)

Four implementations across 16 modules computed the same SHA-256 file digest:

1. one whole-file `Path.read_bytes()` implementation;
2. nine identical 1 MiB streamed implementations;
3. five streamed implementations differing only by `Path(path)` coercion;
4. one configurable-chunk streamed implementation.

Owner is now `canonical_json.sha256_file(path: Path | str,
chunk_size=1 MiB)`. It streams, accepts both historical path forms, preserves
module imports it as `_sha256_file`, preserving private import identity and
existing callers (`cohort.materializer._sha256_file` remains importable).

## CI governance

`tools/arch_baselines/research_agent_duplicate_helpers.json` is an upper-bound
allowlist consumed by `tools/audit_repository_hygiene.py`:

- `_sha256_file`: no local definitions allowed;
- `_finite`: the current 15 file/count pairs are grandfathered; shrinkage is
  allowed, additions or a second definition in one file fail CI.

The check is AST-based (top-level sync/async functions) and has a negative
regression test in `tests/test_repository_hygiene.py`.

## Verification

- canonical owner + hygiene + legacy materialization: 12 passed;
- all 16 consumer families (artifact/declared/typed contracts, idea mining,
  longitudinal/source status, materialized metadata, cohort materializer,
  development sample, execution input, runner, robustness, family primary):
  522 passed;
- `test_validators.py`: 217 passed after the endpoint semantic-key fix;
- resource-context baseline unchanged; repository hygiene OK; module graph
  zero SCC; ruff clean.
- architecture baseline re-emitted with reason: only accepted growth is the
  reviewed `validators.py +9` semantic-key fix, not the consolidation.

## Follow-up order

1. E1 fresh Web acceptance before more structural work.
2. `_finite` migration by semantic family into `scalar_utils.py`, shrinking
   the allowlist after each tested batch.
3. `_method_head`, typed figure-product builders, and atomic-write helpers in
   separate reviewed batches; no name-only mass replacement.
