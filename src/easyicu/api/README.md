# Native derivation context

`extraction.py` retains the complete arguments actually consumed by the existing
SOFA-1 Sepsis kernel before publishing cropped module views. Streamed batches
also retain their cohort and any existing SI short-circuit reason. One-shot
calls retain their merged arguments without changing values or selecting a new
SI window. No clinical algorithm is duplicated in `derivation_context.py`.

`_manifest.json.derivation_contexts.sep3_sofa1` references content-addressed,
host-private files in `.derivation-context/`. They bind the consumed inputs,
batch/cohort coverage, executable and dictionary hashes, public file hashes,
runtime provenance and actual LOS-based publication bounds. They are excluded
from `files`, column metadata and research-variable discovery. These files
contain patient data: they are local replay evidence, never external-LLM input.
Raw/prepared source provenance still requires its independently verified source
snapshot; this contract does not manufacture historical producer authority.

```python
from easyicu.api import validate_native_derivation_context

receipt = validate_native_derivation_context(
    candidate_directory,
    expected_manifest_sha256=retained_manifest_hash,
    expected_patient_ids=authorized_local_cohort,  # optional additional binding
)
```

The validator snapshots and hashes the exact bytes it reads, calls the current
`scores.sepsis.sep3` defaults, then applies the same native label time filter.
The publisher runs this same validator before exposing the root manifest when
producer context exists. It reports aggregate membership/onset consistency only. Missing, damaged,
incomplete, cohort-drifted or executable-mismatched contexts refuse complete
verification; old exports remain usable as public views but cannot acquire
this stronger evidence retrospectively. Each replay holds one recorded shard
at a time. Even one-shot calls are saved as disjoint complete-stay shards:
at most 128 stays and 65,536 combined input rows; each dependency must fit
32 MiB decoded. An oversized single stay is refused without truncation or SI
reselection. Cohort/label sets and LOS maps use memory proportional to stay
count; observations never accumulate globally. Collector copies bytes without
loading DataFrames; snapshots use bounded copy buffers and temporary disk.
Temporary snapshots need space for the largest input or public dependency
file; this extra IO and retained private data count toward extraction budgets.

Cropping dependencies first can change firstSI, remove the cumulative-minimum
baseline, or promote a second onset after the first is cropped. Regression
tests cover all three counterexamples, exact boundaries and negative/missing
inputs. Retaining actual dependencies fixes reproducibility without changing
firstSI, the infection window, SOFA threshold, or publication policy. A passing
receipt is engineering evidence, not clinical closure or research approval.

## Sparse source exports

The web runner publishes native-v2 manifests even when source tables have only
headers or selected concepts have no events. Each physical file records
`concept_status`: `available` has observed values; `produced_all_null` has no
values after evaluating the available source; `structurally_unavailable_placeholder`
has a typed null column because a mapping, required table or score output is
unavailable. Missing primary columns without a producer receipt still fail
publication. Existing catalog coverage and review readers exclude both null
statuses from observed coverage.

For suspected infection and both Sepsis-3 outputs, complete required source
tables with no antibiotic events yield `produced_all_null`,
`evaluation_status=evaluated`, and `reason_code=no_antibiotics_observed`.
The empty positive-event set represents zero positive stays. Missing any
required antibiotic source table yields `structurally_unavailable_placeholder`,
`evaluation_status=not_evaluable`, `reason_code=required_source_table_missing`,
and the missing table names. This includes a partial multi-table antibiotic
source: its empty observed component cannot establish a negative. An absent
component or invalid time metadata uses `required_component_missing`,
`antibiotic_time_index_missing` or `component_time_index_missing` instead.
Dedicated score owners that return no physical output are unavailable with
`producer_returned_no_physical_output`; their missing score is never zero.
Populated source values and the clinical scoring definitions are preserved.
