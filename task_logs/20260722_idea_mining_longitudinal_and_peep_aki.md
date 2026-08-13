# FIG5-DISC-017/018 — longitudinal design discovery and PEEP–AKI re-triage

## Scope

This batch answers two user-raised Idea Mining questions without extracting any
database data:

1. Can EasyICU discover a cross-database SOFA-2 trajectory-transportability
   question rather than only static predictor/outcome pairs?
2. Does the historical Agent-mined PEEP–AKI candidate remain actionable under
   the current data and prior-art gates?

Prepared data authority remained:

`/Volumes/外置硬盘/easyicu_data/full6_20260717`

No six-database extraction was run.

## Root cause of the missed SOFA-2 design

The existing data-first generator enumerated `predictor × outcome` pairs. The
actionability selector also required a pair-level joint-completeness signal.
Thus a design whose scientific identity is `repeated trajectory ×
cross-database transportability` had no representable entry point and could be
discarded before novelty review. File presence alone was correctly
insufficient, but there was no host-owned repeated-measure readiness route.

## General fix

Added:

- `src/easyicu/research_agent/discovery/idea_mining_longitudinal.py`
- `tools/run_idea_mining_longitudinal_discovery.py`
- `tests/research_agent/test_idea_mining_longitudinal.py`

The new leaf:

- accepts host-declared unit, time, and value coordinates;
- binds the full parquet bytes by SHA-256 and the full row count;
- profiles a bounded data sample for distinct times, repeat support, and value
  coverage;
- admits a database only when longitudinal semantics are demonstrated rather
  than inferred from a filename;
- emits human-review trajectory-transportability candidates, never novelty,
  result, or paper authorization.

The command uses the existing EasyICU concept catalog and
`TIME_SERIES_COMPATIBLE_MODULES`; it is not specific to SOFA-2.

## FIG5-DISC-017 — SOFA-2 result

Artifact:

`research_output/experiments/FIG5-DISC-017/longitudinal_sofa2/longitudinal_discovery_manifest.json`

The existing six prepared `sofa2_score.parquet` files contain 80,100,887 rows
in total. All six expose an explicit ICU unit identifier, `charttime`, and
`sofa2`. The provider-free Idea Mining route emitted exactly one ready SOFA-2
candidate:

- ready databases: 6/6 (`aumc`, `eicu`, `hirid`, `miiv`, `mimic`, `sic`);
- sampled repeat-unit fraction: minimum 0.971 across the six databases;
- sampled SOFA-2 non-null coverage: 1.000 in every database;
- full artifact row counts and SHA-256 values are recorded in the manifest;
- candidate question: whether prespecified longitudinal trajectory
  features/classes are reproducible and transportable across databases.

This proves *data/design answerability*, not novelty or a scientific finding.
Protocol choice (time zero, grid, window, feature representation, cluster or
trajectory method, stability criterion, and external comparison) and prior-art
review remain required.

## FIG5-DISC-018 — current PEEP–AKI re-triage

Artifact:

`research_output/experiments/FIG5-DISC-018/peep_aki_current/`

The current provider-free pair route was rerun on the existing 94,458-stay
prepared MIIV cohort, using `peep` and `aki`; no old receipt or result was
promoted.

Current evidence:

- concepts are dictionary-resolvable in 6/6 harmonized databases;
- joint complete rows: 39,403/94,458 (41.7%);
- current PubMed screen: 51 broad hits, 40 exact hits;
- direct same-topic PMIDs include `40114273`, the 2025 article that generated
  the historical Agent candidate;
- generic PEEP→AKI association: `hold`, `crowded_but_differentiable`, with no
  current specific differentiator;
- current shortlist instead proposes a cross-database PEEP
  measurement/source-status audit, because the 41.7% completeness must not be
  interpreted as clinical absence.

Therefore PEEP–AKI was a genuine Agent-mined idea, but the generic association
is not a defensible new discovery under the improved current search. A narrower
time-varying exposure protocol or source-status question could still be
reviewed, but must be differentiated and human-confirmed.

Historical handoff checked (read-only):

`/Volumes/外置硬盘/EasyICU_归档/research_output/20260710/agent_run_peep_aki_20260625_primitives2/discovery_handoff.json`

SHA-256: `31d21276d64034ab718dcf5eaf59dc1205150e682ce10305a307317f61dcbef5`

## Verification

- `pytest -q tests/research_agent/test_idea_mining*.py` → 190 passed.
- focused longitudinal tests → 7 passed, including empty/static/duplicate
  authority fail-closed cases.
- Ruff + Black: clean.
- `arch_measure --diff`: zero regression.
- `research_agent_module_graph --diff`: zero new cycle/regression.

