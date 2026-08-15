# SOFA-2 production conformance v2 + artifact renderer closure

- Date: 2026-08-09
- Branch: `fix/pi-workspace-review-20260809`
- Review base: `c50e6d769c50c98d4dad6d1fd6e6edbcd0be5210`
- Task ID: `SOFA2-PRODUCTION-CONFORMANCE-V2`

## Scope and authority

This patch closes the nine findings in the latest external review at the code,
test and governance layers. The SOFA-2 scoring rules are bound to the 2025 JAMA
SOFA-2 publication. Automated conformance remains distinct from independent
clinician review and database-level ascertainment validation.

The production databases do not currently expose every newly required signal
as a validated concept. In particular, treatment unavailability/ceiling,
one-hour oxygenation persistence, active RRT episode termination and non-renal
RRT indication remain database-specific ascertainment work. The implementation
therefore fails closed and the registry status is `source_bound_golden`; it does
not invent these signals or claim clinical validation.

## Closed findings

### Web artifact renderer

- Escaped artifact-table titles, empty-state text and object-derived column
  headings at the final HTML sink.
- Added hostile title, empty-state and object-key vectors to the direct Node
  regression (five cases total).
- Added an explicit CI step that executes the Node regression rather than only
  inspecting the test source from Python.

### SOFA-2 canonical and compatibility behavior

- Every ratio-based respiratory score now requires a sustained one-hour
  oxygenation signal; score 3/4 additionally requires respiratory support or a
  documented unavailable/ceiling exception. Unknown evidence cannot silently
  turn a transient change into a score. ECMO retains its independent 4-point rule.
- Cardiovascular adjunct escalation includes dopamine, dobutamine and other
  vasoactive adjuncts. Dopamine-only scoring is restricted to a genuine sole
  agent path.
- CNS scoring accepts a pre-sedation GCS contract; delirium treatment remains an
  independent one-point floor, including when GCS is 15.
- Renal scoring accepts explicit active-RRT-episode and non-renal-only signals.
- Historical public callbacks now delegate to the canonical SOFA-2 owner rather
  than maintaining a second active scientific rule set.

### Production adapters and missingness

- The concept callback preserves string ECMO indication values instead of
  coercing `VA`/`VV` to numeric missingness.
- The production aggregate records the observed component count.
- PaO2/FiO2 matching no longer silently assumes FiO2 21%. Room-air imputation is
  explicit opt-in and records observed/imputed/reason provenance; without the
  opt-in the result remains missing.
- Duplicate exact timestamp ratio rows from bidirectional as-of matching are
  removed.

### Clinical contracts

- Added source-bound contracts and independent fixtures for all six SOFA-2
  components.
- Every component contract names its production executor. Golden vectors run
  through both the direct scorer and the actual production component callback.
- The aggregate contract depends on all six component contracts, cannot outrank
  its weakest dependency and is exercised through the actual production
  aggregate callback.
- The generated conformance matrix and dictionary status now state the remaining
  ascertainment limitations instead of claiming `validated_definition`.
- A wider neighbor regression exposed a pre-existing Sepsis-3 catalog error:
  its prose mentions a SOFA “score”, so a description heuristic mislabeled the
  binary phenotype as non-binary. The dictionary now declares an explicit
  `outcome_type: binary`, and the catalog prefers that typed contract.

### Exact-head Research Agent contract alignment

- The previous `c50e6d7` full Research Agent CI had three stale-contract reds.
  The Sepsis-3 catalog red is fixed by the typed outcome contract above.
- The ordinal association test and normalized characterisation bundle still
  expected `reportable`, although the validated registry intentionally permits
  only survival and adjusted association to default to reportable. They now
  assert and freeze the fail-closed `analysis_only` receipt with
  `scientific_validator_unavailable`; no production authority was relaxed.

## Verification

- Direct renderer regression: `5` hostile-input cases passed under Node.
- Focused clinical/concept/Web suite: `112 passed`.
- Additional clinical/static neighbor suite: `24 passed`.
- Wider callback, dictionary, Research Agent catalog and static-route suite:
  `406 passed`; this run exposed and then verified the explicit Sepsis-3 binary
  outcome contract described above.
- Capability registry, assessment, normalized golden bundle and catalog closure:
  `28 passed`.
- Ruff on all changed Python owners/tests: passed.
- Python compilation, JSON parsing, clinical-contract validation and
  `git diff --check`: passed.
- The local service returned HTTP 200, but the in-app browser security policy
  rejected localhost tab control. No automated visual desktop QA is claimed;
  this renderer patch changes escaping and cache version only, not layout.
- No patient data, provider call, paid model call or manuscript result was used.

## Remaining explicit boundary

The code-level review is closed. Independent clinician review and measured
database-by-database mappings for the source-bound signals remain release
evidence work. Those rows stay `mapping_only`; this patch deliberately does not
promote them to clinically validated.
