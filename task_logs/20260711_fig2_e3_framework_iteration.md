# Figure 2 E3 framework iteration — 2026-07-11

## Scope

- Case: `E3_kdigo_gradient`
- Arm/model: full EasyICU workflow, `aware` only, `gpt-5.6-luna`
- Run: `research_output/_diagnostic_e3_fresh_20260711/bench_e3_gpt-56-luna/E3_kdigo_gradient/aware/run_20260711T063414_7e96d3`
- Method: generate one fresh plan, then resume only the failing/current step. Existing code was reused only once for the already agent-generated Step 02 output whose checkpoint write had been interrupted.
- Current milestone: 9/12 planned steps are green through `05_stage_stratified_outcomes_and_trend_figure`. Steps `06_secondary_adjusted_association` and `07_cohort_definition_sensitivity_comparison` executed under the older gates but fail the new scientific contracts and are diagnostic-only; the final Step 07 figure has not run. E3 is not frozen and Figure 2 remains 6/9.

## Framework failures found and repaired

1. The LLM cohort-flow step read the already locked `cohort_analysis.parquet` and mislabeled its 74,829 rows as the universe. A new deterministic cohort-flow runner now replays the locked CTAS predicates from raw `cohort.parquet`, verifies the final set against the materialized cohort and provenance, and fails closed on mismatch.
2. The ordinal owner over-claimed exposure derivation/QC, descriptive trend, and supportive adjusted steps as the primary ordinal model. Routing now requires a real model/effect contract and excludes non-primary roles while retaining true primary adjusted ordinal association.
3. The missingness runner did not honor plan aggregate choice or analytic-denominator outputs. It now reads current step inputs/outputs, prefers the declared aggregate (`aki_stage_max` here), emits `analytic_denominators.csv`, traces value/flag discordance, and blocks on absent declared inputs.
4. The first repaired Step 03 artifact classified `rrt_first=0` with `rrt_measured=0` as 73,746 missing measurements. Source inspection showed a sparse positive-event encoding: `rrt_first == rrt_measured == (rrt_n > 0)`, with 1,083 positives. The generic repair recognizes event status only when the complete binary value, non-missing flag, and independent event-count signal agree exactly; otherwise the conflict remains visible.
5. The initial missingness figure scanned every paired flag in the wide table, so unrelated SOFA variables dominated an E3 plot. With declared step inputs, the audit now scopes strictly to those input families and includes explicitly requested direct variables such as age, sex, admission type, LoS, and outcome. Full-table discovery remains only for legacy plans without an input contract.
6. The clinical validator treated the phrase “dose-response gradient” for ordered organ dysfunction as treatment language and emitted an immortal-time warning. Bare `dose` no longer triggers treatment semantics; actual treatment/therapy/drug/assignment signals still do.
7. The figure renderer now propagates event-status semantics, labels the RRT row as analytic event status, and calls the mixed panel `Analytic availability`. Its contract states that absence-as-negative is analytic status availability, not literal measurement capture.
8. Component-QC contracts previously allowed one aggregate count check to conceal an omitted authoritative measured family. The contract now derives every declared `*_measured → *_n` pair and requires an exact per-input checked/unavailable entry. Step 04 reports four checked families, including the authoritative exposure, with 299,316 comparisons and zero discordance.
9. The concept auditor previously saw only the first 12k characters of a long generated script and could warn about code it had not actually inspected. Its code window is now 64k and its denominator/count semantics distinguish audit-only counts from exposures/covariates.
10. Resume provenance could label repaired code as pure reuse and content-addressed evidence could hide the latest execution lineage. Generation mode now records LLM/concept/runner repair separately, and non-LLM code evidence IDs include the execution mode suffix.
11. The first Step 04 figure looked visually clean but paired locked-cohort percentages with the valid-observed denominator and accepted a 99.84% sum as “100%” under a loose tolerance. A generic source-data gate now checks `percentage = 100*count/denominator` on every row and routes figure contract/source failures through the early repair loop.
12. Repeated LLM figure rewrites took roughly 33 minutes per attempt and still failed the deterministic denominator gate. EasyICU now supports a closed step-level `figure_data_family` contract and a case-neutral ordered-category renderer. It routes explicit artifact contract → exact controlled-method compatibility adapter → existing family/name fallback, with explicit unknown/ambiguous contracts failing closed.
13. Informational audit records were passed to Critic as actionable strings, making a clean deterministic step `needs_revision`. Critic inputs now include only warning/error findings; the skipped optional LLM audit remains registered as `info` without weakening any warning/error gate.
14. Adversarial review found a figure-source trace false pass: a structural join could choose an arbitrary identifier-like category and return success when renamed value columns meant zero values were compared. Every named/index/structural join path now requires at least one genuinely verified value; row-aligned cross-name numeric verification is supported, and unrelated declared source tables cannot launder a match.
15. The clustering router used bare substring signals, so `cluster-robust` or hospital-level clustering in a mixed-effects association step could be hijacked by the trajectory runner. Clustering, effect, prediction, cohort-change and publication ownership now require an exact normalized method owner plus a closed structured product. Production primary science remains agent-owned; deterministic runners are auxiliary calculations/renderers only.
16. Canonical-plan augmentation no longer rewrites the agent's step id/method/intent or forces benchmark-shaped mega-steps. Without method-head plus structured-product ownership, the plan is preserved and only a critic/replanner warning is emitted. The primary deterministic owner registry is empty for survival, causal, ordinal and clustering methods.
17. A direct-parent compatibility gate prevents an ordered-category supporting renderer from consuming outcome/trend/effect-bearing tables merely because their shape or figure name looks compatible. The real Step 05 result table is therefore rejected by the supporting distribution renderer instead of producing the wrong scientific figure.
18. The Step 05 figure draft exposed three general output-integrity gaps. The shared coder contract now requires count-derived rates to reconcile to positive denominators, forbids replacing missing confidence limits with point estimates, requires disclosure of excluded invalid rows, and forbids numeric summaries on `n == 0` rows. No case-specific variable or benchmark id was added to the shared prompt.
19. A concept-repair provider failure previously discarded the generated-but-unexecuted draft, forcing another full coder call. The new isolated quarantine checkpoint keeps the latest rejected candidate and its blocking audit errors outside ordinary evidence/notebook generation. Explicit resume must materially repair it and re-run the full concept gate; exact/comment/whitespace/`pass`/standalone-constant no-ops remain blocked. A concept-approved repair retires the stale quarantine before runner entry; a failed deterministic fallback retains the agent draft, while a successful fallback retires it with distinct fallback lineage.
20. An independent adversarial pass then hardened the recovery boundary itself: all quarantine path components reject symlinks; cleanup errors fail closed; ordinary code reuse is confined to `evidence/`, requires an exact SHA-256 match, and accepts only agent-rooted `llm/repaired/runner_repaired` lineage. Unrooted or fallback-derived `resumed_code_reuse` records cannot be laundered into agent code. A partially repaired script that still fails audit replaces the older checkpoint so the next retry continues from the newest candidate.
21. Every post-audit code mutation now returns through one digest-gated concept audit. The exact approved digest is checked against the script that actually ran before any output or code evidence is accepted; a self-modifying script is cleared and marked `blocked_script_integrity`.
22. The primary-model contract now activates from a closed method/product or an emitted contract key, not a step-name substring. It uses exact concept identities plus controlled aggregate suffixes, distinguishes binary and continuous outcomes, requires finite point estimates and typed continuous effect scales, detects sparse zero-event cells including numeric categorical encodings, and requires controlled penalized interval/convergence provenance.
23. Locked robustness IDs alone are no longer sufficient. For every cohort-axis spec the gate replays the plan-locked CTAS predicates on `EASYICU_UNIVERSE_PARQUET` and verifies axis, universe N, retained N, overlap, entries and exits against the agent's structured rows.
24. Only outer step records with `status=ok` may contribute a primary effect, robustness row, adapter payload, covariate set or declared spec ID. An explicit estimator adapter must declare both estimator family and missing-data policy; the framework never defaults those scientific choices to logistic/complete-case.
25. The remaining automatic cohort-sensitivity science runners were removed from preflight ownership because they selected definitions, exposures, covariates and models. Clustering, ordinal, cohort, replan, terminal-rendering and figure-rescue routes now use an exact method head plus closed products/direct-parent artifact contracts; prose, step IDs and `<head>_with_<rider>` riders cannot hijack ownership.
26. O22 multiple-testing correction is now family-aware rather than run-wide. It admits exact raw-p fields, excludes untyped coefficient dumps/nuisance/sensitivity rows, treats an explicit family ID as authoritative, conservatively deduplicates current CSV/JSON evidence, ignores superseded resume checkpoints, and versions the registered report so citations cannot retain a stale SHA.
27. Long generated scripts are sampled from head, middle and tail within the bounded LLM audit window rather than truncating everything after the first 64k characters.
28. The targeted benchmark launcher now performs a minimal real completion preflight by default. A healthy `/models` response can coexist with exhausted upstream quota; the launcher therefore fails in seconds on a 502 instead of entering a long coder/repair pipeline. `COMPLETION_PROBE=0` is the explicit opt-out.
29. Artifact authority is now step-local and append-only: only the latest outer record for a step is authoritative, and only its latest `status=ok` evidence may enter figure repair, primary-effect extraction, robustness panels, summaries, or packaging. Stale successful outputs cannot outvote a newer blocked attempt; modern figure repair also requires active same-step evidence with an exact SHA-256 match.
30. Automatic repair is now centrally authorized at every mutation boundary. Cohort, exposure, outcome, model, estimand, missing-data, and method changes default to deny; the method-substitution allowlist is empty. Resume summaries, concept repair, contract repair, runner fallback, case-plugin repair, and pre/post-summary rewrites all pass through the same authorization and digest checks, including unknown or undefined helper-generated rewrites.
31. Robustness infrastructure can no longer synthesize primary science. Generic implicit cohort refits are disabled in production, adapter-derived primary rows cannot enter the formal panel or digest, rank-deficient designs fail instead of dropping predictors, implicit ridge stabilization is removed, and low-cardinality numeric variables are not silently retyped as categorical. Deterministic robustness remains an auxiliary calculation over agent-planned typed contracts.
32. Hidden ordinal/formula primary refits and the zero-impute-to-complete-case concept rewrite were removed rather than retained as fallback paths; their dead helper implementations were deleted. The method-suite registry now points only to the safe `robustness_sensitivity` auxiliary owner, and generated capability/method documentation was refreshed without relaxing meta-generalization or capability-drift probes.

## Real-run evidence

### Step 01 — primary cohort flow

- Deterministic replay: universe 94,458; adult filter excludes 0; ICU LoS ≥1 day excludes 19,629; final analytic cohort 74,829.
- Final stay IDs and counts exactly match `cohort_analysis.parquet`, `cohort_locked.json`, and provenance.
- Status `ok`, 0 repairs, no contract findings.
- Figure step produced PNG/SVG/PDF/TIFF plus source CSV and contract; contract/source/visual findings are empty. Manual visual inspection found no clipping.

### Step 02 — baseline table

- The original Luna-generated analysis and `table_one.csv` had completed before an interrupted post-step checkpoint.
- `REUSE_STEP_CODE=1` was used only for the same run/model/step to rerun deterministic audits and register the checkpoint; no new model call and no cross-model provenance reuse.
- Final step record status is `ok`.

### Step 03 — data quality and missingness

- Status `ok`, deterministic generation, 0 code/concept repairs, no contract/stat/guard/clinical findings.
- Declared inputs resolved: 19/19; missing declared inputs: none.
- Joint complete denominator: 67,911 / 74,829.
- `aki_stage` value column: `aki_stage_max`; missing 121 (0.161702%).
- RRT: `raw_indicator_one_n=1,083`, `indicator_semantics=binary_event_presence`, `event_count_column=rrt_n`, analytic status missing 0. The 73,746 negative rows are not presented as measured patients or as true source-capture validation.
- Audit scope contains exactly 12 declared variable families: AKI and components, creatinine, urine output, RRT, age, sex, admission type, ICU LoS, and death. Unrequested SOFA variables are absent.

### Step 03 figure

- Rendering-only deterministic bundle with 12 rows and one analytic event-status row.
- Produced PNG/SVG/PDF/TIFF, source data, and figure contract.
- Step status `ok`; contract/source/stat/clinical/guard findings all empty; 0 repairs.
- Panel B title is `Analytic availability`; the RRT display label is `rrt — analytic event status`; the contract includes the locked absence-as-negative caveat.
- Manual visual inspection confirmed readable labels and no clipping.

### Step 04 — ordered exposure derivation/QC

- Status `ok`; locked input 74,829 with no added row filter.
- Valid observed ordered levels: 74,708. Counts for levels 0/1/2/3 are 37,433 / 14,061 / 19,593 / 3,621; no valid source: 121.
- All four measured/count families are checked: authoritative summary plus creatinine, urine-output, and replacement-therapy components. Total count/flag comparisons: 299,316; discordant: 0.
- Component reconstruction covers 74,708 valid rows with zero mismatch to the authoritative ordered exposure.

### Step 04 figure

- Case-neutral deterministic two-panel bundle produced PNG/SVG/PDF/TIFF, editable SVG, source CSV, and figure contract in about 5 seconds.
- Panel A is the ordered-level distribution conditional on 74,708 valid-observed rows and sums to 100%; Panel B is the complete source-status partition against the 74,829 locked cohort and sums to 100%.
- Every plotted row satisfies `percentage = 100 * count / denominator`; maximum observed floating-point difference is `2.78e-17`.
- Final manifest record: `status=ok`, `generation_mode=fallback`, `deterministic_code_fallback=publication_figure_parent_outputs_preflight`.
- Critic `pass`; statistical, clinical, guard, figure-contract, and figure-source findings all empty; visual QA error count 0. The only concept-audit record is informational: deterministic audits ran and the optional LLM audit was skipped.

### Step 05 — ordered stage-stratified outcomes and trend

- Status `ok`, `generation_mode=repaired`, return code 0; this is the eighth completed planned step.
- The locked 74,829-row cohort is retained without new eligibility filtering; 74,708 rows have a valid ordered exposure and 121 remain explicitly source-missing/invalid for this analysis set.
- The agent produced stage-stratified mortality counts/risks/Wilson intervals, ICU length-of-stay median/IQR, Cochran-Armitage and Jonckheere-Terpstra trend rows, and the prespecified Holm two-test family. Contract, statistical guard and execution findings are empty.
- The stored clinical warning about treatment-style time zero is advisory and is now addressed by the general semantic validator change for severity gradients; it did not alter the result or turn the step into a treatment-effect analysis.

### Step 05 figure — repaired and structurally accepted

- The earlier draft was correctly blocked for an unreconciled death percentage, fabricated zero-width confidence limits, and silent invalid/zero-denominator row handling. A later targeted repair produced the formal bundle without rerunning upstream science.
- Final record is `status=ok`, `generation_mode=resumed_code_reuse`; PNG/SVG/PDF/TIFF, `absolute_risk_by_stage_source_data.csv`, and the figure contract are registered.
- Deterministic figure-contract, source-trace and visual gates are clear. Critic retained one semantic concern about the first-24-hour label, but that definition is explicitly established in the locked exposure step/plan rather than invented by the figure.

### Step 06 — operational output rejected by the new model contract

- The older run record says `status=ok`, but that launcher status is not scientific acceptance. Replaying the current deterministic validator blocks the artifact on six issue classes: invalid/missing fitted-term intervals, continuous quantile effects mislabeled as log-odds, unverified penalized interval provenance, unverified penalized convergence, unreported zero-cell separation, and an unpenalized fit despite a zero-event categorical cell.
- In the real cohort, admission category `EYE` has two rows and zero deaths. The old complete-case MLE produced boundary-scale coefficients/odds ratios and missing inferential quantities while claiming convergence and no separation. The artifact remains diagnostic-only until agent repair runs under the new contract.

### Step 07 — operational output rejected by locked-spec and membership replay

- The older script replaced all seven locked specs with four invented IDs and read the 74,829-row locked cohort as its universe, so its relaxed variants repeated identical memberships and effects. It also reports penalized intervals and convergence without controlled provenance.
- The new gate reports all seven locked IDs missing, four extras, and 15 missing membership fields. Independent oracle-only replay of the locked predicates gives: `alt_all_adult_stays` 94,444 (19,615 enter, 0 leave); `alt_adult_los_12h` 90,467 (15,638 enter, 0 leave); `alt_age_21_plus` 74,318 (0 enter, 511 leave). These numbers are QA references only; paper-facing findings must be regenerated by the research-agent pipeline.
- The Step 07 figure remains unrun. The invalid old sensitivity figure/output bundle is not reportable evidence.

## Verification

- Exact user-requested figure-source, clustering-routing and meta-generalization command: 63/63 passed; neither meta spec nor capability-drift probes were relaxed.
- Core primary-model/source/anti-pipeline/execute integration: 294/294 passed. The overlapping contract/robustness/estimator/post-repair-SHA extension independently passed 137/137.
- Routing, structural figure rescue and anti-pipeline focused batches: 137/137; deterministic robustness/adapter focused: 13/13.
- O22 family-aware extraction: 20/20 non-pipeline tests. The default-enabled slow pipeline initially exposed a missing zero-test audit finding; after adding the truthful empty-family finding (without admitting an untyped coefficient dump), that test passed in 107.05 seconds. The disabled path had already passed, so both slow paths are green.
- Full resume/replan/capability integration ran 127 tests: 126 passed and the sole failure exposed two revoked sensitivity runners still listed in the capability registry. Their automatic-capability declarations were removed and the generated matrix refreshed; the exact live-dispatch drift test then passed, and the capability/method-suite/package-boundary/meta batch passed 53/53. Thus all 127 cases are accounted green without restoring the science-selecting runners.
- Final adversarial governance batch (`figure source trace`, `trajectory clustering routing`, `meta benchmark`, capability registry, method suite, anti-pipeline robustness, and repair registry) passed 145/145. A subsequent focused follow-up covering repair authorization, model ownership, estimator/robustness behavior, and generated-doc drift passed 64/64.
- Final Ruff over every modified/untracked Python file, production `py_compile`, launcher `bash -n`, `git diff --check`, and `python tools/lint_progress.py` all passed; progress lint reported 0 warnings.

## Local commit checkpoints

- `2719ce4 refactor(agent): enforce scientific ownership and evidence authority` — production architecture, capability/method documentation, and case-neutral meta benchmark probes.
- `00ad15b test(agent): lock anti-pipeline and provenance governance` — regression coverage for routing, repair authorization, model ownership, artifact authority, provenance, robustness, and figure-source integrity.
- These commits are local only. The unrelated concurrent Web/Copilot working-tree changes were deliberately excluded, and no push was performed.

## Next exact action

Wait for the upstream quota to reset, then resume only `06_secondary_adjusted_association` with same-run code reuse so the old script first encounters the new model contract and receives a targeted repair. Accept it only after separation, scale, interval and convergence checks are clean. Then resume `07_cohort_definition_sensitivity_comparison`, requiring all locked specs and replayed membership fields before running its separate figure step. Do not restart E3 and do not increment the Figure 2 score until the full run is reportable and frozen.
