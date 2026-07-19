# EasyICU Research Agent Architecture Glossary

Drafted 2026-05-27 for the npj Digital Medicine engineering cleanup.

This glossary fixes the language used in code review, Methods text, and
architecture figures. The research-agent package is not presented as a
free-form autonomous scientist. It is an ICU-aware research workflow in
which LLM components operate inside deterministic clinical, statistical,
and provenance boundaries.

## Layer 1: ICU Data Foundation

Definition: clinician- and engineer-curated semantics that define what a
cohort, variable, time window, and outcome mean before an LLM sees the
task.

Includes: EasyICU concept dictionary, cohort descriptors, temporal
semantics, concept availability, database adapters, and source-table
provenance.

Reviewer-facing claim: disease cohorts and variables are resolved through
EasyICU concepts rather than free SQL, ICD-code shortcuts, or ad hoc
column-name guessing.

Code examples: `schema.py`, `research_context/builder.py`,
`research_context/typed.py`, `concept_availability.py`,
`temporal_semantics.py`, `case_contexts.py`.

## Layer 2: LLM Orchestration

Definition: LLM roles that choose the analysis shape, write code, repair
failed steps, interpret outputs, and draft manuscript text.

Includes: planner, replanner, coder, analyzer, writer, literature helper,
and their prompts.

Reviewer-facing claim: LLMs make bounded research-workflow decisions, but
they do not directly create numeric results or rewrite ICU rules.

Code examples: `agents.py`, `prompts.py`, `prompts/v1/`, `plan_utils.py`.

## Layer 3: Safe Analytical Runtime

Definition: deterministic execution, repair, validation, and audit logic
that turns an LLM-authored analysis step into traceable artifacts or a
blocked finding.

Includes: code runner, code hygiene, deterministic repair, concept-use
auditor, clinical-constraint validator, statistical validator, visual QA,
and failure findings.

Reviewer-facing claim: generated analysis code is executed and checked by
software gates. A failed or unsafe analysis path is demoted or blocked
rather than trusted because an LLM said it was valid.

Code examples: `pipeline_execute.py`, `repairs/source.py`,
`repairs/coordination.py`, `gates/preflight.py`, `code_hygiene.py`,
`audits/validators.py`, `gates/visual.py`, `causal_audit.py`.

## Layer 4: Evidence And Provenance

Definition: content-hashed artifacts, value-level numeric claims, derived
formula claims, evidence-bound writing, and strict manuscript binding.

Includes: EvidenceStore records, NumericClaim registry, derived-claim
formulas, writer digest v1/v2, manuscript numeric binder, evidence
footnotes, and reproducibility envelope.

Reviewer-facing claim: every reported result is linked to a registered
artifact or formula-derived numeric claim. In submission mode, untraced
numbers fail the run.

Code examples: `evidence.py`, `pipeline_writer_aux.py`,
`manuscript_post.py`, `replication/envelope.py`.

## Layer 5: Evaluation And Submission Scaffold

Definition: benchmark harnesses, locked comparison baselines, canonical
submission profiles, run manifests, and packaging logic.

Includes: paper-facing `aware` arm, baseline lock files, strict evidence
profile, reproducibility envelope, report generation, figure bundle, and
LaTeX/PDF packaging.

Reviewer-facing claim: the paper reports a versioned evaluation protocol,
not a mutable leaderboard or a generic agent score.

Code examples: `tools/run_research_agent_bench.py`,
`baselines/REGISTRY.md`, `baselines/LOCK.json`, future
`pipeline_profiles.py`.

## Naming Rules

- Use "LLM agent" only for planner, replanner, coder, analyzer, writer,
  and related prompt-driven roles.
- Use "deterministic gate" or "runtime check" for validators, binders,
  repair rules, and safety checks.
- Use "evidence/provenance layer" for artifact hashing, numeric binding,
  derived formulas, and reproducibility outputs.
- Use "evaluation scaffold" for benchmark runners, submission profiles,
  baseline locks, and packaging.
- Do not describe clinician-curated ICU rules, concept dictionaries, or
  validators as agent memory.

## Methods Sentence Seed

EasyICU's research-agent workflow separates LLM orchestration from
deterministic clinical and provenance gates: LLM components plan analyses,
write executable code, and draft evidence-aware prose, while ICU concept
resolution, sandboxed execution, validators, content-hashed artifacts,
value-level numeric binding, and versioned submission profiles define the
auditable boundary of each run.

For cohort specification, EasyICU records cohort definitions as
time-anchored concept predicates (concept, window, aggregation, operator,
and value) and locks the definition by SHA before execution. In the current
MVP, time-window aggregation is supplied by upstream concept loaders and is
not re-verified at the dataframe filter step; the deterministic robustness
adapter fits pre-specified panels with statsmodels logistic/linear models,
Wald confidence intervals, and complete-case or mean/median imputation.
Cox/Poisson estimators, bootstrap intervals, and MICE imputation are
deferred.
