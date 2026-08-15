# SOFA-2 evidence and conformance contract closure

- Task: `SOFA2-PRODUCTION-CONFORMANCE-V4`
- Branch: `fix/pi-workspace-review-20260809`
- Implementation commit: `b13b582` (`fix(clinical): separate SOFA-2 evidence contracts`)
- Scope: data foundation + Clinical Trust Gate

## Outcome

The frozen review design is implemented in the shipped dictionary, production
callbacks, direct scorers, clinical-contract registry and resolver tests:

1. `delirium_tx_proxy` is the medication-exposure concept.  The deprecated
   `delirium_tx` name is only a compatibility alias and never confirms that a
   drug was used for delirium.
2. `delirium_tx_evidence` has four states: `confirmed`, `proxy_only`,
   `not_detected` and `unavailable`.  Current medication-only database mappings
   can emit `proxy_only` or `unavailable`; they cannot manufacture
   `not_detected` without verified source/time coverage.  `not_detected` means
   no qualifying evidence was found in an assessable source, not that delirium
   was absent.
3. The main CNS output is a conservative database implementation of canonical
   SOFA-2.  Only `confirmed` evidence raises a GCS-15 row to one point.
   `sofa2_cns_proxy_sensitivity` is the explicitly named alternative that also
   counts `proxy_only`; `sofa2_cns_ascertainment` exposes the evidence boundary.
4. Aggregate completeness now separates
   `sofa2_n_observed_components` (real observations in the active scoring
   window before LOCF) from `sofa2_n_available_components` (evidence-backed
   values available after legal LOCF).  Deprecated `sofa2_n_components` aliases
   the latter.  Missing-as-normal contributes zero to the score but to neither
   count.
5. Clinical contracts now separate `spec_golden_vectors` from
   `runtime_golden_vectors`.  Only runtime fixture inputs must be owned by the
   shipped resolver graph.
6. Parameterized resolver coverage executes all six SOFA-2 components plus the
   aggregate.  Formal spec fixtures independently retain transient respiratory
   observations, treatment ceilings, sedation, RRT episode and non-renal RRT
   footnote cases.

## Frozen CNS acceptance matrix

| GCS | Evidence | Main | Proxy sensitivity | Ascertainment |
| --- | --- | ---: | ---: | --- |
| 15 | `confirmed` | 1 | 1 | `complete` |
| 15 | `proxy_only` | 0 | 1 | `proxy_only` |
| 15 | `not_detected` | 0 | 0 | `complete_for_proxy_source` |
| 15 | `unavailable` | 0 | 0 | `unavailable` |
| 13-14 | any | 1 | 1 | GCS result is already decisive |
| 9-12 | any | 2 | 2 | GCS result is already decisive |
| 6-8 | any | 3 | 3 | GCS result is already decisive |
| 3-5 | any | 4 | 4 | GCS result is already decisive |

## Verification

Canonical local environment: `.venv`, Python 3.11.15.

- Focused score, contract, resolver, catalog and public-API suite: `135 passed`.
- Clinical-conformance marker suite: `54 passed, 12768 deselected`.
- Broader callback/dictionary/catalog contract suite run during implementation:
  `233 passed`.
- Generated clinical matrix and static catalog equality: passed.
- Ruff, JSON parsing and `git diff --check`: passed.
- Progress-layer lint: recorded after the progress update in the evidence commit.

No Provider call and no patient-data read was used.  The SOFA-2 dictionary SHA-256
for this implementation is
`702da76208ab8d0189fbef73371f6ec0a68afd01b4333da95ad53d0743e551f8`.
The publication extraction lock remains deliberately unrefreshed: a future
six-database extraction must create new evidence rather than silently relabel an
older export.

## Remaining declared limitations

- All six database mappings remain `mapping_only`; this change does not claim
  independent clinical validation or algorithm-golden database results.
- Current medication mappings do not supply attributable indication or verified
  negative source/time coverage, so production resolver output cannot yet emit
  `confirmed` or `not_detected` from those mappings.
- RRT termination/non-renal indication, continuous `other_vaso >=1h`, and
  respiratory/cardiovascular treatment-ceiling ascertainment still require
  database-specific evidence work.
- Aggregate ascertainment is intentionally later work; it was not invented as
  part of this patch.
