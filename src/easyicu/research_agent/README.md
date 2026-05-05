# `easyicu.research_agent`

A traceable, ICU-aware analysis-agent layer that extends EasyICU from
"data extraction and visualisation" to "data extraction → analysis →
manuscript scaffold" — without giving up provenance.

## Related work this layer builds on

This module deliberately fuses three complementary lines of recent
work and keeps each as a citable inspiration rather than a
dependency:

| Source | What we borrowed | Where it lives |
|---|---|---|
| **OpenLens-AI** [^1] | Five-agent pipeline shape (planner / coder / analyzer / writer / supervisor); LaTeX export; visual QA | `agents.py`, `latex.py`, `visual_qa.py`, `literature.py` |
| **M4** [^2] | MCP-style tool exposure; reusable clinical-skill recipes that short-circuit free-form planning | `mcp_server.py`, `skills.py` |
| **HealthFlow** [^3] | Self-evolving meta-planning by feeding past lessons back into the planner | `memory.py` |
| **nature-skills** [^4] | Claim-first publication figure contract; editable SVG/PDF/PNG/TIFF export; panel-level evidence logic | `publication_figures.py` |

EasyICU's distinct contribution is the **ICU-aware research context**
(`schema.py`, `icu_rules.py`, `context.py`, `case_contexts.py`) and
the **deterministic hashed evidence store** (`evidence.py`,
`validators.py`) that every agent output must pass through before it
can affect the manuscript.

[^1]: OpenLens-AI: Fully Autonomous Research Agent for Health Informatics. <https://github.com/jarrycyx/openlens-ai>
[^2]: M4: Infrastructure for AI-Assisted Clinical Research (MCP + clinical-skills tooling).
[^3]: HealthFlow: A Self-Evolving AI Agent with Meta-Planning for Autonomous Healthcare Research.
[^4]: Yuan1z0825/nature-skills: Nature-style scientific figure-making skill. <https://github.com/Yuan1z0825/nature-skills>

## Why this exists

Generic data-analysis agents (OpenLens-AI, AutoAnalyst, AI Scientist,
DataMind …) are improving fast at the *engineering* of a science
pipeline — planning, code generation, sandboxed execution, paper
writing — but they are uniformly weak on the *medical* part: they
treat ordinal SOFA components as continuous, average GCS values,
silently impute missing PaO₂ to 0.21, fall for the SOFA==0 high-
mortality artefact, and confuse ICU mortality with hospital
mortality.

`easyicu.research_agent` does not try to outdo those projects on
agent architecture. It does one thing they don't: it injects an
ICU-aware **research context** — the EasyICU concept dictionary plus
explicit aggregation rules, time windows, missingness semantics and
known pitfalls — into the agent loop, and it routes every produced
artefact through a SHA-256-hashed evidence store that the manuscript
scaffold is allowed to cite from. Sentences without an evidence id
are blocked.

## Architecture

```
question + cohort  ────────────►  optional: ClinicalSkill (M4)
        │                              │ deterministic plan
        ▼                              ▼
   build_research_context        → research_context.json
        │                          (variable types, units, ranges,
        │                           ordinal levels, allowed aggregations,
        │                           missingness profile, ICU pitfalls)
        ▼
   RunMemory.digest_for_prompt   → memory_digest.md           (HealthFlow)
        │   past runs' notable findings fed to the planner
        ▼
   PlannerAgent  (skipped when a ClinicalSkill is selected)
        │   → analysis_plan.json
        │     (Table 1 / outcome / missingness / association /
        │      SOFA-stratum audit / cross-database)
        ▼
   ┌────── per step ──────┐
   │ CoderAgent → script  │
   │ ConceptUsageAuditor  │  ← static checks (no mean-of-ordinal etc.)
   │ CodeRunner           │  ← subprocess sandbox, captures everything
   │ StatisticalValidator │  ← cross-checks reported numbers vs cohort
   │ AnalyzerAgent        │  ← short interpretation, evidence-bound
   └──────────────────────┘
        ▼
   LiteratureAgent               → literature_bundle.json     (OpenLens)
   VisualQAAuditor               → findings on figures        (OpenLens)
   PublicationFigureContract     → claim-first SVG/PDF/TIFF   (nature-skills)
        ▼
   WriterAgent                   → manuscript_scaffold.md
        │   (Methods + Results sentences with {evidence:<id>}
        │    placeholders; Discussion left to human)
        ▼
   EvidenceStore.bind_manuscript → manuscript_scaffold_bound.md
                                   (placeholders → file links + sha256)
        ▼
   scaffold_to_latex             → manuscript_scaffold.tex    (OpenLens)
        ▼
   RunMemory.record              → .memory/runs/<run_id>.json (HealthFlow)
   manifest.json + results_report.md
```

The deterministic gates (auditor, validator, evidence store) sit
between every LLM step and the next, so the LLM has wide creative
latitude but cannot push unverified numbers into the manuscript.

## Quick start

### Free-form research question

```python
from easyicu.research_agent import ResearchAgentPipeline

pipeline = ResearchAgentPipeline(workdir="./research_output")
result = pipeline.run(
    question="Is admission SOFA-2 score associated with ICU mortality?",
    cohort="path/to/easyicu_cohort.parquet",
    cohort_name="MIMIC-IV first ICU admissions",
    database="miiv",
    target_outcome="death",
    cross_database_validation=["eicu", "hirid"],
)
print(result.report_path)
print(result.manuscript_path)
```

### Pre-canned ClinicalSkill (M4-style)

```python
from easyicu.research_agent import ResearchAgentPipeline, list_skills

print([s.key for s in list_skills()])
# → ['sofa_mortality', 'aki_kdigo_mortality',
#    'vaso_exposure_mortality', 'lactate_trajectory_mortality']

pipeline = ResearchAgentPipeline(workdir="./research_output")
result = pipeline.run(
    skill="sofa_mortality",                 # short-circuits the planner
    cohort="path/to/cohort.parquet",
    database="miiv",
    cross_database_validation=["eicu", "hirid"],
    manuscript_authors=["A. Researcher", "B. Clinician"],
)
```

### MCP server (for Claude Desktop / Continue / etc.)

```bash
python -m easyicu.research_agent.mcp_server   # stdin/stdout JSON-RPC stub
```

```python
from easyicu.research_agent import mcp_dispatch
mcp_dispatch("research_agent.list_skills")
mcp_dispatch("research_agent.run", {
    "question": "Is admission SOFA-2 associated with ICU mortality?",
    "cohort_path": "cohort.parquet",
    "database": "miiv",
    "target_outcome": "death",
})
```

The pipeline runs offline by default with the deterministic
:class:`MockLLMClient` (useful for tests, CI and a baseline for paper
ablations). Pass a real LLM client to enable richer planning and
prose:

```python
from easyicu.research_agent import OpenAIClient
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    llm=OpenAIClient(model="gpt-4o-mini"),
)
```

There is also a CLI:

```bash
easyicu-research-agent \
    --question "Is admission SOFA-2 score associated with ICU mortality?" \
    --cohort path/to/cohort.parquet \
    --database miiv \
    --target-outcome death \
    --workdir ./research_output
```

For the lactate-MAP-vasopressor case, the deterministic replication
runner consumes EasyICU concept-export packages directly and writes
per-database cohorts, source manifests and summary tables. Missing
exports can be marked as planned targets so the manuscript does not
overclaim replication:

```bash
easyicu-research-replication \
    --target miiv=/path/to/miiv_easyicu_export \
    --build-target eicu=/path/to/prepared_eicu \
    --max-patients 1000 \
    --minimal-export \
    --pending hirid \
    --output ./lactate_map_vaso_replication
```

The same case also has a formal EasyICU context contract. It records
source concept files, first-24h windows, missingness semantics,
forbidden transformations and cross-database caveats:

```python
from easyicu.research_agent import (
    build_lactate_map_vaso_research_context,
    write_research_context,
)

ctx = build_lactate_map_vaso_research_context(
    cohort="miiv_lactate_map_vaso_24h.parquet",
    source_manifest="miiv_lactate_map_vaso_24h_source_manifest.json",
    database="miiv",
)
write_research_context(ctx, "research_context.json")
```

A self-contained demo that reproduces the SOFA2==0 missingness
artefact lives at `examples/research_agent_mortality_sofa.py`.

### Publication figure contract

Agent-generated plots are not treated as decorative side effects.
Before drawing a manuscript figure, the figure can be specified as a
claim-first contract: one core claim, one role per panel, explicit
evidence ids, review risks, and required exports. The helper then
applies a Nature-style matplotlib setup and saves editable SVG first,
with PDF/PNG/TIFF as secondary formats.

```python
import matplotlib.pyplot as plt
from easyicu.research_agent import (
    make_figure_contract,
    apply_publication_style,
    save_publication_figure,
    audit_publication_exports,
)

contract = make_figure_contract(
    figure_id="Figure2",
    core_claim="SOFA2=0 is an audit target, not a simple low-risk group.",
    panels=[
        {
            "panel_id": "a",
            "title": "Mortality by SOFA-2 stratum",
            "role": "overview",
            "claim": "Mortality rises with SOFA-2 but score zero is non-monotonic.",
            "evidence_ids": ["outcome_rate", "sofa_strata"],
        },
        {
            "panel_id": "b",
            "title": "Component missingness",
            "role": "audit",
            "claim": "The zero stratum concentrates missing SOFA-2 components.",
            "evidence_ids": ["missingness", "stratum_missingness_comparison"],
        },
    ],
    export_formats=["svg", "pdf", "png", "tiff"],
)

palette = apply_publication_style()
fig, ax = plt.subplots(figsize=(3.5, 2.4))
# ... draw with `palette` ...
paths = save_publication_figure(fig, "figures/Figure2", contract=contract)
findings = audit_publication_exports(paths)
```

## What goes in `research_context.json`

For each column in the cohort we emit a `ConceptDescriptor`
containing role, dtype, unit, valid range, ordinal levels (when
applicable), allowed aggregations, source databases (from the
EasyICU concept dictionary), source export files, known pitfalls,
missingness semantics, forbidden transformations, cross-database
caveats and a missingness profile. The agents never see raw row-level
data through the prompt — only this structured context — so they
cannot invent variables or invalid aggregations.

A snippet:

```json
{
  "name": "sofa2",
  "role": "composite_score",
  "dtype": "int64",
  "is_ordinal": true,
  "valid_range": [0, 24],
  "allowed_aggregations": ["max_or_last", "first_value", "median_only"],
  "aggregation_default": "max_or_last",
  "source_concept": "sofa2",
  "pitfalls": [
    "SOFA-2 follows the same 0-4 component structure as SOFA; treat as ordinal.",
    "EasyICU has empirically observed elevated mortality in the sofa2==0 stratum on at least one source; this often reflects component-level missingness, not low illness severity. Verify component availability before reporting."
  ],
  "missingness": {"fraction_missing": 0.0, "n_missing": 0, "n_total": 1500, "missingness_kind": "MCAR_likely"}
}
```

## Validators

| Validator | When | Severity model |
|---|---|---|
| `CohortAuditor` | After cohort materialisation | `error` blocks the run |
| `ConceptUsageAuditor` | On every generated script before it executes | `error` blocks the step; `warning` recorded |
| `StatisticalValidator` | After every step that produces artefacts | `error` blocks manuscript writeback for that fact; `warning` recorded |

Hard rules currently encoded:

- Mean / std of ordinal score columns (SOFA components, totals, GCS,
  KDIGO stages) is an error.
- `fillna(0)` on any numeric column triggers a warning unless the
  script also documents the imputation.
- Lab columns summarised by mean with no `median(...)` reference in
  the same script trigger a warning.
- A SOFA-stratum table with `score==0` outcome rate exceeding
  `score==1` triggers a warning citing the missingness-vs-severity
  ambiguity.
- Reported `outcome_rate` that disagrees with a cohort recompute by
  more than 0.001 is an error.

## How this fits into a paper

The intended publication framing is:

> EasyICU bridges generic medical research agents and ICU databases
> by injecting structured concept metadata, time-window constraints,
> and aggregation rules into the agent's reasoning context, and by
> routing every produced artefact through a SHA-256-hashed evidence
> store that the manuscript scaffolder is allowed to cite from. We
> demonstrate that the same off-the-shelf agent loop, run with vs.
> without the EasyICU context layer, behaves substantially
> differently on canonical ICU pitfalls (SOFA==0 missingness
> ambiguity, ordinal-score averaging, mortality-definition
> conflation) — providing a reproducible, traceable analysis pipeline
> for high-stakes medical research.

The `examples/research_agent_mortality_sofa.py` demo is the seed for
the **hero ablation**: run the same generated cohort with a generic
agent (no context) and with EasyICU's context layer, and contrast
their handling of the SOFA2==0 stratum.

## What's intentionally out of scope (v1)

- **No Discussion / clinical claims by the writer agent.** The
  scaffold ends with a one-line stub deferring Discussion to the
  human author. This is policy, enforced by the prompt, and is what
  lets the layer be safely published.
- **No Docker / OpenHands sandbox.** The runner is a plain
  subprocess with a wall-clock timeout. Replacing it is one class.
- **No unbounded automatic cross-database execution.** v1 can build
  deterministic replication packages from supplied EasyICU exports or
  local raw database paths, but long-running targets such as HiRID are
  allowed to remain explicitly pending rather than being silently
  omitted or overclaimed.

These are deliberate v1 limits. The structure is in place to lift
each of them without changing the public API.
