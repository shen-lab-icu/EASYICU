# `easyicu.research_agent`

A traceable, ICU-aware analysis-agent layer that extends EasyICU from
"data extraction and visualisation" to "data extraction → analysis →
manuscript scaffold" — without giving up provenance.

Current architecture and retirement status is tracked in
[`docs/research_agent_module_inventory_20260719.md`](../../../docs/research_agent_module_inventory_20260719.md),
with the live component scoreboard in
`task_logs/20260717_remaining_debt_register.md`. The Figure 2 task-generalization
protocol is [`docs/figure2_taskbank_9x3_protocol.md`](../../../docs/figure2_taskbank_9x3_protocol.md).

## What this layer adds

EasyICU's distinct contribution is the **ICU-aware research context**
(`schema.py`, `icu_rules.py`, `context.py`, `case_contexts.py`) and
the **deterministic hashed evidence store** (`authority/evidence_store.py`,
`validators.py`) that every agent output must pass through before it
can affect the manuscript.

In one line: the LLM has wide creative latitude, but it **cannot push an
unverified number into a reportable manuscript**. Concretely, this layer
adds:

- **Evidence binding** — every artefact (script, log, table, statistic,
  figure) is hashed into a SHA-256 `EvidenceStore`, and every reported
  number is registered as a numeric claim and re-checked against the
  value it was produced with.
- **Fail-closed, three-state output** — four readiness gates
  (`execution-complete` / `evidence-complete` / `numeric-verified` /
  `analysis-validated`) are computed mechanically and sort the run into
  **gate-reportable**, **analysis-only**, or **diagnostic-only**. An
  unverifiable claim is intercepted at the manuscript boundary and routed
  for repair / code re-run / human review, then must pass the same gates
  again — it is closed-loop correction, not a one-shot kill switch.
- **Cross-database replication for reliability** — cross-database work
  defaults to a replication protocol (re-run the same question on other
  supported databases) so a conclusion's robustness can be inspected,
  rather than a claim that one database is "better".
- **Deterministic auditing, not LLM-as-judge** — the hard checks are
  rule-based and reproducible, so the system does not rely on one LLM
  grading another.

## Why this exists

Many general-purpose analysis pipelines are strong at orchestration
but weak on ICU semantics: they treat ordinal SOFA components as
continuous, average GCS values, silently impute missing PaO₂ to 0.21,
skip component-completeness checks on composite scores, and confuse ICU
mortality with hospital mortality.

`easyicu.research_agent` addresses that gap by injecting an ICU-aware
**research context** — the EasyICU concept dictionary plus
explicit aggregation rules, time windows, missingness semantics and
known pitfalls — into the agent loop, and it routes every produced
artefact through a SHA-256-hashed evidence store that the manuscript
scaffold is allowed to cite from. Evidence-carrying claims without an
evidence id are blocked or surfaced for human review.

## Architecture

The runtime now makes a four-layer design explicit:

Deterministic statistical kernels are grouped under `methods/` (survival,
missing-data, fairness, multiple-testing, sensitivity, RMST, DeLong and
conformal helpers). New method implementations belong in that package rather
than as additional top-level `research_agent/*.py` modules; stable convenience
symbols continue to be exposed from `easyicu.research_agent`.

- **Layer 1 — ICU Data Foundation**: unified concept abstraction, cohort provenance, deterministic temporal semantics, and ICU episode resolution.
- **Layer 2 — Safe Analytical Runtime**: evidence store, audit log, validators, execution replay, and workflow graph artifacts.
- **Layer 3 — Agent Orchestration**: planner / replanner / coder / analyzer / writer coordinated through a runtime supervisor pattern.
- **Layer 4 — Candidate Hypothesis Ranking**: a pre-plan hypothesis blueprint that distills literature, feasibility, self-critique and ICU domain gates before the planner executes, ranking candidate research questions for human curation. It is **not** an autonomous scientific-discovery system; it is a ranking module whose outputs are filtered by humans and constrained by Layers 1–2.

### Control-plane responsibility packages

The execute loop is being decomposed along authority boundaries rather than by
mechanically moving every top-level file:

| Package | Owns | Must not own |
| --- | --- | --- |
| `gates/` | read-only contract, visual and concept findings/decisions | provider calls, repair budget, checkpoint/evidence mutation, scientific design |
| `execution/` | provider-backed concept audit, explicit host-service boundaries, filesystem execution helpers, and preparation/rendering of already-authorized figure products | cohort, exposure, outcome, method or estimand selection |
| `execution/runners/` | auxiliary deterministic calculations over an Agent-locked specification | primary estimand selection or benchmark-case routing |
| `authority/` | replanner-candidate invariants, typed input/evidence binding, typed success-promotion boundaries and digest-bound direct-parent resolution around the existing EvidenceStore/checkpoint authorities | provider calls, scientific plan selection, a second `current` selector, sibling scan, or independent evidence ledger |
| `repairs/` | typed repair reasons, exact patch/full-rewrite transport, deterministic source repair and summary salvage | scientific redesign, silent gate relaxation or case-specific repair rules |
| `discovery/` | candidate-hypothesis ranking, feasibility and discovery handoff | manuscript authority or autonomous scientific claims |
| `reporting/` | manuscript binding, article/display contracts, review artifacts and rendering helpers | unverified numeric claims or a second evidence authority |
| `planning/` | case-neutral analysis families, pure cohort/robustness contracts, study-design playbooks, capability/method registries and article figure strategy | case-specific benchmark instructions or execution/evidence mutation |
| `providers/` | provider-neutral LLM protocol plus lazy concrete-client construction | planning, scientific policy, evidence authority or eager import of the concrete client |
| `research_context/` | typed context authority, cohort-aware context construction and step-scoped Coder projection | scientific method selection, provider calls or run authority mutation |
| `cohort/` | typed cohort schema, deterministic materialization, artifact facts and case-neutral dataframe primitives | outcome choice, censoring rules or case-specific cohort criteria |
| `acquisition/` | export catalog, coverage checks and materialization handoff for caller-declared concepts | hidden outcome/static-variable defaults or estimand construction |
| `contracts/` | dependency-neutral runtime handoffs and deterministic declared-product, ordered-group and robustness execution contracts | provider calls, scientific design selection or evidence mutation |
| `evaluation/` | optional cross-model concordance and Tier-2 jury/rubric adapters over completed artifacts | primary scientific adjudication or hidden manuscript benchmark authority |
| `review/` | deterministic post-analysis causal-claim and method-appropriateness review | exposure, outcome, cohort, method or estimand selection |

`pipeline_execute.py` remains the orchestration caller. PlanAuthority's pure
candidate projection and TypedBindingResolver are now canonical under
`authority/`; provider calls, revision/evidence registration, cohort mutation,
runner rebuilding and repair/replan budgets deliberately remain in the sole
orchestrator. The remaining ExecutionState and StepExecutor/RunCoordinator
boundaries are now closed as domain-specific state objects and deliberately
narrow coordination components: they do not absorb plan authority, gates,
repair, evidence mutation, or scientific decisions. The remaining execute-loop
body is sequence-sensitive orchestration and maintenance debt, not a reason to
create a universal state bag or inflate the executor. This is a pre-v1
breaking-cleanup line: retired top-level module paths are absent rather than
retained as façades. Fresh runs import responsibility packages directly, and
the archive tag preserves the former diagnostic state.

The production module-top-level static import graph is now acyclic. `pipeline_execute` and
`execution.publication_figure` receive host-owned compatibility helpers through
a fresh immutable `execution.host_services.ExecutePhaseServices` snapshot; they
do not statically reverse-import `pipeline`. Sealed-renderer digest selection
uses a controlled registry-mediated dynamic import of current implementation
modules. This dependency inversion changes neither
the Agent's scientific ownership nor the sealed renderer template, and an
explicit static-graph regression requires zero cyclic modules.

Current scope note: Layer 4 is bounded. It produces an auditable
`hypothesis_blueprint.json` before planning, ranking candidates by
coverage / literature-saturation / gate-pass weights, but it should not
be described as "Scientific Discovery" in paper-facing text. In manuscripts use
"candidate hypothesis ranking, human-curated".

### Engine ownership — what is swappable vs. house-owned forever

The LLM/agent "brain" is a **commodity, swappable engine**, not part of the
moat. General coding agents (Codex, Claude Code, and whatever comes next) will
keep getting better at the *generative* roles (planning, writing analysis code,
drafting prose), and we deliberately let them — they plug in behind the
`LLMClient` protocol (`complete(messages) -> str`). What does **not**
commoditise, and what this layer owns permanently, is the ICU concept
foundation (Layer 1) and the value-level evidence runtime (Layer 2). The table
below is the contract; treat it as a guardrail when integrating any external
agent.

| Role / component | Ownership | Rule |
| --- | --- | --- |
| planner / replanner | engine-swappable | route to any `LLMClient` (mock / API / CLI agent) |
| coder | engine-swappable | generates code *text*; execution stays house-owned |
| writer / analyzer | engine-swappable | output still passes the evidence gate below |
| concept resolution / cross-DB mapping (Layer 1) | **house-owned forever** | domain knowledge a general agent does not have |
| evidence store / `NumericClaim` binding / `bind_numeric_values` | **house-owned forever, non-LLM** | the provenance moat |
| validators / replay / restricted-AST formula evaluator | **house-owned forever, non-LLM** | deterministic, auditable, never delegated |

Two invariants enforce this split:

1. **The engine is optional, never required — capability degradation ladder.**
   `providers.llm.build_llm_client(prefer=...)` walks
   `CLI agent (codex/claude) → OpenAI → OpenRouter → MockLLMClient`, returning
   an `LLMClientSelection` that records what actually ran and why it fell back.
   A user without any coding-agent CLI or API key still gets a working,
   end-to-end pipeline on the offline `MockLLMClient` floor (design rule #1 in
   `providers/llm.py`). Adding `CLIAgentLLMClient` did **not** introduce a hard
   dependency.

2. **No engine bypasses `NumericClaim` binding — engine-agnostic provenance.**
   `manuscript_post.bind_numeric_values` takes only the manuscript *string* plus
   the `EvidenceStore`; in STRICT mode every printed number must trace to a
   registered claim or it is rejected, regardless of which brain wrote it
   (mock, API, or a local Codex/Claude CLI). Stronger engines hallucinate more
   confidently, so this gate *gains* value as the brain improves. Do not add an
   engine-specific "trusted" path that lets some provider's numbers skip
   binding; `tests/research_agent/test_numeric_provenance_engine_agnostic.py`
   fails if you do.

**Opt-in agentic coder (altitude-2a).** Setting
`EASYICU_AGENTIC_CODER_BACKEND=codex` (or `claude`) makes the execute phase
delegate *script authoring + self-repair* to a local coding-agent CLI
(`agentic_coder.AgenticCoderAgent`): the CLI runs its own write→run→fix loop in
a sandbox against the cohort parquet (`COHORT_PARQUET`) and returns only the
final script. That script is then executed and evidence-bound by the
instrumented runtime exactly as the LLM coder's output is — so the *numbers*
still come from our pipeline, never from the CLI's own run. It is **off by
default**, degrades to the LLM `CoderAgent` when the CLI is absent, and applies
the same method-compatibility check. The real CLI-driven loop is validated
under `needs_real_data`; the offline unit tests
(`tests/research_agent/test_agentic_coder.py`) lock the delegation/fallback
contract.

S1 idea-triage note: `concept_availability.real_data_concept_feasibility`
adds an outcome-blind data-layer feasibility probe for future
review/editorial-derived idea triage. It reports denominator counts,
per-concept non-missing counts, and joint completeness across requested
concepts from a single exported wide cohort table. The input is **not** a raw
database directory or sharded concept store. The probe records requested
time-window and aggregation metadata for downstream triage, but S1 does not
apply temporal filtering or aggregation. It does **not** compute outcome rates,
subgroup outcomes, p-values, or effect estimates, and it does not trigger an
analysis run.

S2 preregistration note: `idea_registry.IdeaCandidateRegistry` provides the
append-only human gate for future idea-mining candidates. `source_snapshot_id`
is required but intentionally opaque at this stage; the later literature
snapshot layer owns freeze/hash semantics. Only candidates whose latest ledger
entry is `accepted` pass `assert_executable()`, so ranking or extraction stages
cannot silently launch analyses from proposed or rejected ideas.

S3 ranking note: `hypothesis_generator.generate_hypotheses` remains a pure
deterministic ranking helper. It does not read parquet files, call the S1 probe,
or trigger pipeline execution. Callers may pass precomputed pair-level
`feasibility_by_pair` signals, in which case ranking uses
`joint_fraction_complete` as the coverage term and records a `feasibility_note`
when joint completeness is low. The `literature_saturation_signal` field is a
ranking signal only, not a novelty claim; higher values mean the supplied
citations mention the predictor-outcome pair more densely.

S4 extraction note: `idea_mining.py` adds the first upstream literature
idea-extraction layer. In v1 it only supports `metadata_only` and
`user_supplied_excerpt` extraction paths; `open_access_fulltext` and licensed
full-text ingestion remain behind DG1. Source snapshot manifests store citation
metadata, locators, hashes and character counts, but not source bodies. Extracted
ideas carry source quotes and can be mapped to executable candidates only after
concept names are reconciled and outcome determinability is known. Binary
outcomes stored as event-positive present/NA values must be normalized to
explicit known 0/1 before feasibility probing, or the candidate remains
non-executable for joint-feasibility ranking. S4 does not register, accept, or
run analyses.

S5 dry-run note: `idea_mining.run_idea_mining_dry_run()` wires the upstream
idea extraction into the downstream feasibility/ranking/registry gates without
launching an analysis. It writes a source snapshot manifest, candidate triage
report, and S2 registry ledger, then stops with every candidate still in
`proposed` status. S5 uses each `ExecutableHypothesisCandidate.feasibility_pair_key`
as the canonical anchor, runs pair-level feasibility one predictor/outcome pair
at a time, and only passes those precomputed joint-completeness signals into
`generate_hypotheses()`. If no data path or feasibility probe is supplied, S5
withholds ranking rather than silently falling back to single-variable
missingness. The dry-run report surfaces extraction/mapping yield, unresolved
terms, outcome-determinability gates, preregistered family size, and triage-only
causal-audit risk markers. The preregistration report deliberately separates the
all-considered candidate denominator from the executable-candidate denominator;
the former is an anti-fishing audit trail, while the latter is the scope that a
later p-value adjustment layer would normally consume. The causal-audit marker is
a static dry-run reminder, not a per-candidate causal analysis.

S6 discovery-output note: prior-art assessment is a triage layer, not a
tool-authored novelty claim. S6 builds broad and exact PubMed-style queries for
each literature-derived candidate, freezes query strings, counts, PMIDs, top
hits, direct same-topic decisions and rationales, and labels candidates as
`already_done`, `crowded_but_differentiable`, `sparse`, or `apparently_gap`.
The query construction deliberately uses the literature wording and
differentiators (`time_window_hint`, trajectory/clearance wording, analysis
family, clinical scenario), not canonical EasyICU concept keys. Feasibility may
use `lact`, but novelty search must preserve phrases such as "lactate clearance
trajectory" so the idea is not drowned in broad lactate literature. S6 can pass
the frozen prior-art saturation signal into `generate_hypotheses()` as a
caller-supplied pure-function input; ranking still does not connect to PubMed.
The human-readable discovery report is rendered from structured records and
must include source, gap quote, candidate topic, prior-art triage, database
feasibility, Go/No-go, risks, and `clinical_plausibility_requires_human` for
every candidate.

Literature scope (discovery lever 1): `idea_scope.py` turns a declarative
`LiteratureScopeSpec` (journal set/preset, publication types, date window,
user-supplied topic terms) into a deterministic PubMed query via
`build_pubmed_query_from_scope(...)`. The spec is case-neutral — it hardcodes
only publication metadata (journal presets, `[pt]` clauses), never a
disease/exposure/score/database; topic terms are always caller-supplied, and
`last_n_years` resolves against an explicit `reference_year` to stay
reproducible. Letting each user scope their own corpus is the cheapest lever
against cross-user candidate duplication. `run_idea_mining_dry_run(...)` accepts
an optional `scope` plus an injected `source_search_client`: when no explicit
`materials` are passed it builds the query, retrieves `metadata_only` source
materials (titles/venues/ids only — no abstract or full-text body), and freezes
the resolved query to `scope_query.json`. Network I/O happens only through the
explicitly injected client; supplying a `scope` without a client (and without
materials) fails closed.

Current evaluation-protocol note: the historical name **"ICUAgentBench"**
is retained for code and on-disk compatibility, but in the manuscript
this module is described as the **EasyICU evaluation protocol** (an
internal evaluation scaffold), not as a benchmark. The protocol is
layered: Tier 1 deterministic checks, Tier 2 frontier-LLM jury for
process / writing quality, Tier 3 clinician spot-check for clinical
plausibility. **Only Tier 1 is executed in the current submission**;
Tier 2 and Tier 3 are outlined in the Supplementary Methods
(`02_npj_Digital_Medicine/tier_evaluation_protocol_20260527.md`) and
will be added in the revision response if reviewers request them. No
"gold answers", "benchmark scores", or "scientific discovery" claims
should be surfaced through this module in paper-facing text.

```
question + cohort  ────────────►  optional: ClinicalSkill
        │                              │ deterministic plan
        ▼                              ▼
   build_research_context        → research_context.json
        │                          (variable types, units, ranges,
        │                           ordinal levels, allowed aggregations,
        │                           missingness profile, ICU pitfalls)
        ▼
   RunMemory.digest_for_prompt   → memory_digest.md
        │   past lessons + StrategyCards + meta-planner skill ranking
        ▼
   optional concept retrieval    → research_context_agent_prompt.json
        │   top-K variables to agents; full context retained for validators
        ▼
   LiteratureAgent + HypothesisBlueprintAgent
        │   → preplan_literature_bundle.json + hypothesis_blueprint.json
        │     literature-grounded hypothesis, step skeleton, self-critique,
        │     cross-DB concept feasibility, and ICU domain gates
        ▼
   PlannerAgent  (skipped when a ClinicalSkill is selected)
        │   deterministic fallback if hosted model returns invalid JSON
        │   → analysis_plan.json
        │     (task-family-specific modules chosen from cohort summary,
        │      outcome incidence, missingness audit, primary association,
        │      advanced-analysis protocol, score-specific QC, or
        │      cross-database protocol)
        ▼
   ┌────── per step ──────┐
   │ CoderAgent → script  │
   │ ConceptUsageAuditor  │  ← static checks; repair before execution
   │ CodeRunner           │  ← subprocess sandbox, captures everything
   │ deterministic code fallback for bad/no-output hosted scripts
   │ StatisticalValidator │  ← cross-checks reported numbers vs cohort
   │ AnalyzerAgent        │  ← short interpretation, evidence-bound
   └──────────────────────┘
        ▼
   LiteratureAgent               → literature_bundle.json     (curated + optional PubMed/Tavily)
   VisualQAAuditor               → findings on figures        (deterministic + optional VLM)
   PublicationFigureContract     → claim-first SVG/PDF/TIFF
        ▼
   WriterAgent                   → manuscript_scaffold.md
        │   (Methods + Results sentences with {evidence:<id>}
        │    placeholders; Discussion left to human)
        ▼
   EvidenceStore.bind_manuscript → manuscript_scaffold_bound.md
                                   (placeholders → file links + sha256)
        ▼
   scaffold_to_latex             → manuscript_scaffold.tex
        ▼
   audit_log.jsonl               → runtime supervision trace
   workflow_graph.json/.md       → execution graph + Mermaid view
   execution_replay.json         → deterministic replay bundle
   RunMemory.record              → .memory/runs/<run_id>.json
   RunMemory StrategyCards        → .memory/strategies/<strategy_id>.json
   manifest.json + results_report.md
```

The deterministic gates (auditor, validator, evidence store) sit
between every LLM step and the next, so the LLM has wide creative
latitude but cannot push unverified numbers into the manuscript.

The plan is intentionally dynamic. Table 1, outcome incidence,
missingness, score-completeness checks and cross-database steps are
included only when the question, analysis type and available context
justify them. Composite-score completeness is treated as pre-analysis,
outcome-blind quality control: cohorts may expose a generic
`<score>_n_components` column, and the deterministic probe reports
whether low score strata are under-measured without looking at the
outcome. Cross-database work defaults to a replication protocol unless
an external cohort is actually available.

## Quick start

The pipeline now requires an explicit `llm=` client — there is no silent
mock fallback. The examples below use `OpenAIClient`; for tests / CI and
ablation baselines, swap in `MockLLMClient` (see
[Testing / CI](#testing--ci) below).

### Free-form research question

```python
from easyicu.research_agent import OpenAIClient, ResearchAgentPipeline

pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    llm=OpenAIClient(model="gpt-4o-mini"),
)
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

### Deterministic Analysis-Family Skills

```python
from easyicu.research_agent import OpenAIClient, ResearchAgentPipeline, list_skills

print([s.key for s in list_skills()])
# → ['association_analysis', 'prediction_model', 'data_quality_audit']

pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    llm=OpenAIClient(model="gpt-4o-mini"),
)
result = pipeline.run(
    skill="association_analysis",           # binds variables from this context
    cohort="path/to/cohort.parquet",
    database="miiv",
    question="Is the selected ICU exposure associated with the selected outcome?",
    target_outcome="death",
    cross_database_validation=["eicu", "hirid"],
    manuscript_authors=["A. Researcher", "B. Clinician"],
)
```

### Config-driven experiment spec

```python
from easyicu.research_agent import (
    CohortInputSpec,
    ExperimentSpec,
    OpenAIClient,
    ResearchAgentPipeline,
    RuntimeSpec,
)

spec = ExperimentSpec(
    question="Is admission SOFA-2 score associated with ICU mortality?",
    cohort=CohortInputSpec(
        cohort="path/to/easyicu_cohort.parquet",
        cohort_name="MIMIC-IV first ICU admissions",
        database="miiv",
        target_outcome="death",
        user_preferences={"inferred_analysis_family": "association_study"},
    ),
    runtime=RuntimeSpec(workdir="./research_output", stop_after_analysis=True),
)

pipeline = ResearchAgentPipeline(
    workdir=spec.runtime.workdir,
    llm=OpenAIClient(model="gpt-4o-mini"),
)
result = pipeline.run_from_spec(spec)
print(result.manifest_path)
```

### MCP server (for Claude Desktop / Continue / Cursor / etc.)

```bash
python -m easyicu.research_agent.mcp_server --transport stdio
```

```python
from easyicu.research_agent import mcp_dispatch
mcp_dispatch("research_agent.list_skills")
mcp_dispatch("research_agent.list_concepts", {"cohort_path": "cohort.parquet"})
mcp_dispatch("research_agent.audit_cohort", {"cohort_path": "cohort.parquet"})
mcp_dispatch("research_agent.run", {
    "question": "Is admission SOFA-2 associated with ICU mortality?",
    "cohort_path": "cohort.parquet",
    "database": "miiv",
    "target_outcome": "death",
})
```

The server answers MCP JSON-RPC methods `initialize`, `tools/list` and
`tools/call` over stdio. In addition to the end-to-end
`research_agent.run`, it exposes atomic tools for external agents:
`build_context`, `list_concepts`, `describe_concept`, `load_concepts`,
`audit_cohort`, `run_validator`, `cross_database_concept_availability`,
and `bind_evidence`. These are standardized extraction and evidence
tools, not raw SQL tools: external agents can call EasyICU's existing
`load_concepts` API for any supported concept set (vitals, labs,
therapies, outcomes, scores, sepsis definitions, SOFA/SOFA-2 components,
etc.), check cross-database derivability before extraction, and register
the resulting table/figure/log outputs into the SHA-256 EvidenceStore for
downstream manuscript binding. A minimal legacy SSE bridge is also
available:

```bash
python -m easyicu.research_agent.mcp_server --transport sse --port 8765
```

### Testing / CI

`ResearchAgentPipeline.run()` requires an explicit `llm=` client and no
longer silently falls back to a mock — see `pipeline.py` for the runtime
check. The deterministic :class:`MockLLMClient` is intended for three
specific situations:

- unit / integration tests that should not hit a network,
- CI smoke runs where reproducibility matters more than prose quality,
- the **naive arm** of the paper ablation, which intentionally exercises
  the same orchestrator without the EasyICU context layer.

```python
from easyicu.research_agent import MockLLMClient, ResearchAgentPipeline
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    llm=MockLLMClient(),
)
```

Tests that hit a real LLM provider should be marked
`@pytest.mark.needs_real_llm` (declared in `pytest.ini`); they are
skipped by default and only run with `pytest --run-real-llm` plus a
matching API key in the environment (`OPENAI_API_KEY`,
`OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY`). This keeps real-provider
costs opt-in and makes "passes only under mock" testing visible in CI
reports rather than hidden as a green test.

### Optional VLM figure review

Deterministic figure checks are always available. A vision-language
review pass is enabled automatically when `vlm_client` or
`visual_qa_adapter` is configured, or explicitly through
`enable_vlm_visual_qa=True`:

```python
from easyicu.research_agent import OpenAIClient, ResearchAgentPipeline

vision = OpenAIClient(model="gpt-4o-mini")
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    llm=OpenAIClient(model="gpt-4o-mini"),
    vlm_client=vision,
)
```

### Optional Tavily literature search

The literature bundle is curated and offline by default. PubMed and
Tavily are opt-in enrichment layers:

```python
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    enable_pubmed=True,
    pubmed_email="you@example.org",
    enable_tavily=True,  # reads TAVILY_API_KEY unless tavily_api_key=... is passed
)
```

Tavily is intended for material PubMed can miss, such as preprints,
guideline pages, trial registries and PDFs.

### Chinese manuscript scaffold

```python
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    manuscript_language="zh",
)
```

Evidence placeholders stay ASCII (`{evidence:table_one}`), so the
same binder and SHA-256 provenance checks apply in English and Chinese.

### Prompt-sized context retrieval

```python
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    context_top_k=40,
)
```

Agents see only the retrieved concept slice plus required id/time/outcome
variables. Validators, manifests and cohort audits still use the full
research context.

### LaTeX venue templates and editable PPTX figures

```python
pipeline = ResearchAgentPipeline(
    workdir="./research_output",
    latex_venue_template="npj",  # article, nature, npj, lancet
)
```

Publication figures can also be exported as PowerPoint decks:

```python
from easyicu.research_agent import save_publication_figure

paths = save_publication_figure(contract, output_dir, export_formats=["svg", "pptx"])
```

### OpenHands / Docker runner demo

```bash
python examples/research_agent_openhands.py --pull
```

The demo uses `runner_kind="docker"` with an OpenHands-compatible
runtime image; the rest of the research-agent contract is unchanged.

There is also a CLI:

```bash
easyicu-research-agent \
    --question "Is admission SOFA-2 score associated with ICU mortality?" \
    --cohort path/to/cohort.parquet \
    --database miiv \
    --target-outcome death \
    --manuscript-language zh \
    --enable-tavily \
    --workdir ./research_output
```

For paper reproduction, the replication CLI now supports a second,
paper-aware mode. It first parses the paper into a typed replication
spec, then runs the standard analysis pipeline, then emits both a
replication report and, only when the gates pass, a showcase
replication manuscript:

```bash
easyicu-research-replication \
    --paper ./papers/critical_care_example.md \
    --cohort ./cohorts/miiv_analysis_cohort.parquet \
    --database miiv \
    --mode manuscript \
    --llm openai \
    --openai-model qwen3-coder-30b \
    --output ./research_output/paper_replication
```

This mode writes:

- `paper_profile.json`
- `replication_spec.json`
- `paper_claim_ledger.csv`
- `replication_comparison.csv`
- `replication_report.md`
- `deviation_report.md`
- `manuscript_ready.md` only when both the standard manuscript gates and
  the paper-aware publication-claim audit pass

The paper-aware readiness bundle extends the existing fail-closed gates
with:

- `design_reproduced`
- `paper_claims_parsed`
- `result_alignment_audited`
- `replication_report_ready`
- `showcase_manuscript_ready`

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

A self-contained demo that exercises composite-score component-completeness
QC lives at `examples/research_agent_mortality_sofa.py`.

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
    core_claim="Composite-score component completeness is checked before analysis.",
    panels=[
        {
            "panel_id": "a",
            "title": "Pre-analysis completeness",
            "role": "overview",
            "claim": "Composite-score rows with low component completeness are flagged before outcome modeling.",
            "evidence_ids": ["score_completeness", "missingness"],
        },
        {
            "panel_id": "b",
            "title": "Component missingness",
            "role": "qc",
            "claim": "Component missingness is summarized before outcome modeling.",
            "evidence_ids": ["missingness", "score_completeness"],
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
    "SOFA-2 totals may reflect component-level missingness; cross-check component completeness before drawing clinical conclusions."
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
- Composite/ordinal score completeness is checked before analysis when
  the cohort exposes component completeness such as `<score>_n_components`;
  this quality-control signal is outcome-blind and does not require a
  score-specific audit step.
- Reported `outcome_rate` that disagrees with a cohort recompute by
  more than 0.001 is an error.

Coverage note: these validators are deterministic, rule-based checks
for a curated set of ICU-specific failure modes. They are not a formal
verifier for arbitrary Python semantics, dynamic aliasing, causal
identification, or clinical interpretation.

## How this fits into a paper

The intended publication framing is:

> EasyICU bridges generic medical research agents and ICU databases
> by injecting structured concept metadata, time-window constraints,
> and aggregation rules into the agent's reasoning context, and by
> routing every produced artefact through a SHA-256-hashed evidence
> store that the manuscript scaffolder is allowed to cite from. In
> controlled demonstrations, the same off-the-shelf agent loop, run
> with vs. without the EasyICU context layer, can be audited on
> canonical ICU pitfalls (composite-score component-completeness,
> ordinal-score averaging, mortality-definition conflation) —
> providing a reproducible, traceable workflow for ICU analysis.

The `examples/research_agent_mortality_sofa.py` demo can be run with a
generic context (no ICU-aware metadata) and with EasyICU's context layer
to contrast their handling of composite-score component completeness —
the seed of the context-layer ablation. The paper-facing evaluation of
context on/off is produced through the EasyICU evaluation protocol
(Tier 1 deterministic scorecard, `evaluation_scorecard.py`), not a
standalone ablation script.

## What's intentionally out of scope (v1)

- **No Discussion / clinical claims by the writer agent.** The
  scaffold ends with a one-line stub deferring Discussion to the
  human author. This is policy, enforced by the prompt, and is what
  lets the layer be safely published.
- **No default Docker requirement.** The default runner remains a plain
  subprocess with a wall-clock timeout so local demos and CI stay
  simple. It captures provenance and enforces timeouts, but it is not a
  strong security sandbox. Docker/OpenHands is opt-in via
  `runner_kind="docker"` or a user-supplied `runner_factory`.
- **No unbounded automatic cross-database execution.** v1 can build
  deterministic replication packages from supplied EasyICU exports or
  local raw database paths, but long-running targets such as HiRID are
  allowed to remain explicitly pending rather than being silently
  omitted or overclaimed.

These are deliberate v1 limits. The structure is in place to lift
each of them without changing the public API.
