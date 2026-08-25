# Meta-generalization benchmark (anti-overfit stress test)

A small held-out benchmark whose purpose is **not** to add more questions the
framework already answers well, but to check that EasyICU generalises *off* the
canonical 9 — and, just as importantly, that it **fails closed with a surfaced
reason** when a question is out of scope, instead of fabricating a result.

The canonical 9 (E1–E3, M1–M3, H1–H3) are almost all MIMIC-IV, mortality/LOS
outcomes, ICU-admission time origin. A framework tuned only against them can
quietly overfit: hard-coded column names, an implicit single time origin, a
database-specific missingness assumption. This benchmark deliberately varies six
axes away from that comfort zone.

## The six axes

| Axis | Canonical-9 comfort zone | What this benchmark adds |
| --- | --- | --- |
| **Exposure** | vasopressor, KDIGO stage, ventilation | RBC transfusion, sedation tier, beta-blocker, driving pressure, an *unmapped* drug |
| **Outcome** | in-hospital mortality, LOS | new-onset AF, prolonged ventilation, extubation-or-death, post-discharge readmission |
| **Time origin** | ICU admission | intubation, ventilation start, septic-shock onset, hospital discharge, symptom onset (pre-hospital) |
| **Database** | mostly MIMIC-IV | eICU, HiRID, AmsterdamUMCdb, SICdb |
| **Missingness** | complete-case-ish | structural absence, informative measurement, truncated windows, unresolvable columns |
| **Analysis family** | association / survival / causal | all six families incl. LLM-coded prediction & phenotyping, plus an unsupported family |

## Two kinds of item

Each item declares an `expected_behavior`:

- **`bound_result`** (MG01–MG07) — the pipeline should produce a bound primary
  estimand via the expected runner (or the LLM-coded path for prediction /
  phenotyping / descriptive), with a valid figure. This tests *positive*
  generalisation to unseen coordinates.
- **`fail_closed`** (MG08–MG12) — the pipeline should **block or degrade to
  `diagnostic_only` with a specific surfaced reason**, never fabricate. This
  tests the fail-closed / gap-report behavior documented in
  `capability_registry.FAIL_CLOSED_LADDER`. These probes are the point: a system
  that generalises must also *know its own edges*.

The `fail_closed` probes map directly onto the ladder:

| Item | Gap | Expected surfaced reason | Level |
| --- | --- | --- | --- |
| MG08 | follow-up beyond discharge not observable | "follow-up beyond discharge is not derivable" | runner block or gate |
| MG09 | exposure absent from the DB mapping | `Missing required causal columns` | runner block |
| MG10 | pre-hospital time origin | "time origin … not reconstructable" | gate → diagnostic_only |
| MG11 | dose-response asked of a binary exposure | "graded ordinal exposure (>=3 levels)" | runner block |
| MG12 | competing-risks CIF (unsupported family) | "no deterministic competing-risks runner" | gate → diagnostic_only |

**MG12 is intentionally a real capability gap**: competing-risks CIF is not yet a
supported deterministic estimand. It is included so the benchmark keeps an honest
inventory of what the framework *cannot* do — and flags it for the registry.

## Feasibility tiers

The spec is runnable incrementally; each item declares a `feasibility`:

- **`runnable_now`** — a probe expressible on the existing MIMIC-IV export; the
  fail-closed ones (MG08, MG10, MG11) block by construction regardless of export
  details, so they can be exercised first.
- **`needs_universe`** — needs a universe built from an existing prepared export
  (e.g. MIMIC-IV MG04/MG06; eICU MG01/MG07/MG09 once the eICU export exists).
- **`needs_database`** — needs the raw database mounted + converted first
  (HiRID MG02, AmsterdamUMCdb MG03, SICdb MG05).

Build a universe the same way as the canonical 9 (an `ehrflowbench.jsonl`
universe under `research_output/`), then run:

```bash
python tools/run_research_agent_bench.py \
  --bench-kind analysis --arms aware --provider openai --model <model> \
  --ehrflowbench-jsonl research_output/universe_<id>/universe_<id>_ehrflowbench.jsonl \
  --out-root research_output/meta_benchmark/bench_<id>
```

## What "pass" means

- `bound_result` item → `execution_complete` and a bound primary estimand from
  the `expected_runner` (or a validated LLM-coded estimand for LLM-coded
  families), with the manuscript headline == the registered primary estimand.
- `fail_closed` item → the run does **not** reach `manuscript_ready`; the
  scorecard tristate is `diagnostic_only`; and the surfaced reason contains the
  `expected_gap_reason` substring (for `runner_block`) or the run floors to
  `diagnostic_only` (for `gate_diagnostic_only`). A `fail_closed` item that
  produces a confident bound result is a **failure** — it means the framework
  fabricated past a gap.

## Files

- `meta_benchmark.jsonl` — one JSON object per item (the machine-readable spec).
- `qualification12_literature_design_pack_20260825.json` — two reviewed
  comparator/design-analogue sources per item, with seven aggregate design
  dimensions, exact full-text/supplement receipts, and the explicit rule that
  published effect estimates are **not** benchmark answers.
- `tools/build_qualification12_literature_design_pack.py` — reproducibly rebuilds
  the tracked seed pack from the external full-text review manifests.
- `tools/audit_qualification12_literature_design_pack.py` — zero-Provider audit
  of question identity, typed authority, seven-dimension coverage, and optional
  external source-file digests.
- `tests/research_agent/test_meta_benchmark_spec.py` — a coverage lint: asserts
  the spec spans all six axes, includes enough fail-closed probes, and does not
  simply re-test canonical-9 coordinates. It validates the *spec*, not a run.

The design pack is an input to planning, not a Qualification12 result. Loading
it does not authorize a manuscript claim, and a fail-closed benchmark item must
remain fail-closed even when an analogue paper demonstrates that the method is
possible with richer data.
