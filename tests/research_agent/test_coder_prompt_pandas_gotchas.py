"""Prompt guardrails for pandas idioms seen in real LLM pilot runs."""

from __future__ import annotations

import inspect


def test_coder_prompt_names_pandas_categorical_codes_gotcha() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "PANDAS IDIOM GOTCHAS" in coder_prompt
    assert "pd.Categorical(x).cat.codes" in coder_prompt
    assert "pd.Categorical(x).codes" in coder_prompt
    assert 'pd.Series(x).astype("category").cat.codes' in coder_prompt


def test_coder_prompt_closes_categorical_distribution_denominators() -> None:
    from easyicu.research_agent.agents import CoderAgent
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.lower().split())

    assert "closed partition" in normalized
    assert "counts must sum exactly to `n_nonmissing`" in normalized
    assert "silently omitted from the category rows" in normalized

    repair_normalized = " ".join(
        inspect.getsource(CoderAgent.repair).lower().split()
    )
    assert "map each non-missing value to" in repair_normalized
    assert "category counts sum to" in repair_normalized
    assert "silently omitting it from all categories" in repair_normalized


def test_coder_prompt_forbids_bitwise_not_on_scalar_dtype_predicates() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Negate such scalar predicates with `not`, never with `~`" in coder_prompt
    assert "reserve `~` for elementwise boolean Series/array masks" in coder_prompt


def test_coder_prompt_treats_input_cohort_as_already_locked() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "already-materialised, locked analysis cohort" in coder_prompt
    assert "Do not re-derive eligibility" in coder_prompt
    assert "length-of-stay" in coder_prompt
    assert "cohort-definition sensitivity" in coder_prompt
    assert "equal to `len(df)`" in coder_prompt


def test_coder_prompt_keeps_fraction_units_and_zero_status_categories() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "keys containing" in coder_prompt
    assert "`fraction` are proportions in [0, 1]" in coder_prompt
    assert "Never copy a percentage" in coder_prompt
    assert "complete four" in coder_prompt
    assert "zero-frequency" in coder_prompt
    assert "not 0%" in coder_prompt
    assert "source-status rows partition the locked cohort" in coder_prompt
    assert "valid-observed distribution denominator" in coder_prompt


def test_coder_prompt_bounds_probability_intervals_without_constraining_effects() -> (
    None
):
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "confidence bound for one of those quantities" in coder_prompt
    assert "methods.ordered_trends.wilson_interval" in coder_prompt
    assert "floating-point boundary artefacts" in coder_prompt
    assert (
        "Never" in coder_prompt and "clip a genuinely invalid estimate" in coder_prompt
    )
    assert "does not apply to effect scales" in coder_prompt
    assert "risk ratios, odds ratios, hazard" in coder_prompt


def test_coder_prompt_controls_ordered_stratified_tools_and_reporting() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "exactly `ordinal_stratified_descriptive_analysis`" in coder_prompt
    assert "agent remains" in coder_prompt
    assert "responsible for resolving the declared input columns" in coder_prompt
    assert "easyicu.research_agent.methods.ordered_trends" in coder_prompt
    assert "wilson_interval" in coder_prompt
    assert "cochran_armitage_trend" in coder_prompt
    assert "jonckheere_terpstra_trend" in coder_prompt
    assert "event_counts=event_counts" in coder_prompt
    assert "totals=totals" in coder_prompt
    assert "values=individual_outcomes" in coder_prompt
    assert "groups=aligned_group_labels" in coder_prompt
    assert "`ci.ci_low` and `ci.ci_high`" in coder_prompt
    assert "`result.statistic_type`" in coder_prompt
    assert "`result.chi_square is None`" in coder_prompt
    assert "`result.score_scheme is None`" in coder_prompt
    assert "explicit `scores` vector" in coder_prompt
    assert "equal-spacing consecutive-rank assumption" in coder_prompt
    assert "individual-level outcome values" in coder_prompt
    assert '"JT-equivalent"' in coder_prompt
    assert "serialize or report a p value as `0` or `0.0`" in coder_prompt
    assert "one prespecified family of exactly two" in coder_prompt
    assert '`family_id="ordered_trend_outcomes"`' in coder_prompt
    assert '`multiplicity_policy="holm_familywise"`' in coder_prompt
    assert '`step_summary["ordered_stratified_contract"]`' in coder_prompt
    assert "field name is exactly `ordered_levels`" in coder_prompt
    assert "`explicit_ordered_levels`" in coder_prompt
    assert "lists of numbers" in coder_prompt
    assert "stringify these values" in coder_prompt
    for field in (
        "schema_version",
        "ordered_exposure_column",
        "ordered_levels",
        "cochran_armitage_scores",
        "score_scheme",
        "binary_outcome_column",
        "continuous_outcome_column",
        "locked_cohort_n",
        "valid_ordered_exposure_n",
        "ci_method",
        "ci_alpha",
        "continuous_summary",
        "quantile_method",
        "stratified_table",
        "trend_table",
        "tests",
        "multiplicity_policy",
        "multiplicity_family_size",
    ):
        assert f"`{field}" in coder_prompt
    assert "Do not embed trend-result payloads" in coder_prompt
    assert "The stratified CSV must contain one row per declared level" in coder_prompt
    assert "The trend CSV must contain exactly the two planned outcome rows" in (
        coder_prompt
    )


def test_coder_prompt_keeps_sparse_event_negatives_in_exposure_denominator() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert (
        "Do not define the analytic cohort for a binary event/exposure" in coder_prompt
    )
    assert "<concept>_measured == 1" in coder_prompt
    assert "event-negative" in coder_prompt
    assert "untriggered" in coder_prompt
    assert 'indicator_semantics="binary_event_presence"' in coder_prompt
    assert "retain reconciled count-zero/flag-zero rows as the" in coder_prompt
    assert "Never restrict the analytic denominator to `measured == 1`" in (
        coder_prompt
    )
    assert "if the triad is incomplete or discordant, fail closed" in coder_prompt
    assert "structurally missing on reconciled negative rows" in coder_prompt
    assert "do not require an explicit" in coder_prompt
    assert "Every reconciled positive row must carry" in coder_prompt
    assert "methods.source_status.reconcile_binary_event_presence" in coder_prompt
    assert "never selects the concept, exposure, cohort" in coder_prompt
    assert "`BinaryEventPresenceResult` dataclass, not a mapping" in coder_prompt
    assert "`result.values`" in coder_prompt
    assert "Do not require a dict" in coder_prompt
    assert "The three column arguments are keyword-only" in coder_prompt
    assert "count_column=count_col" in coder_prompt
    assert "do not reconstruct overlapping source-status masks" in coder_prompt


def test_coder_prompt_applies_source_status_to_every_measurement_summary() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "several per-stay summaries" in coder_prompt
    assert "apply the same source-status" in coder_prompt
    assert "consistency audit to EVERY summary" in coder_prompt
    assert "numeric/domain validity mask separate" in coder_prompt
    assert "must not exclude individual values" in coder_prompt
    assert "change descriptive/model denominators" in coder_prompt
    assert "fail the completed step instead of filtering" in coder_prompt


def test_coder_prompt_uses_one_host_replayed_count_flag_contract() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "count_flag_comparison_n" not in coder_prompt
    assert "count_flag_discordant_n" not in coder_prompt
    assert "count_consistency_by_measured_input" not in coder_prompt
    assert "need not duplicate this host-replayed audit" in coder_prompt


def test_coder_prompt_requires_host_replayed_measurement_provenance_audit() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    assert "requirement is not limited to a component-QC step" in normalized
    assert '`step_summary["measurement_provenance_audit"]["source"]`' in normalized
    assert "one record under `checks`" in normalized
    assert '`"COHORT_PARQUET"`' in normalized
    assert "`invalid_pair_n`" in normalized
    assert '`role="audit_only"`' in normalized
    assert "fail-closed provenance error" in normalized
    assert "reject omitted measured families, convenient" in normalized
    assert "omitted from the planner's" in normalized
    assert "provenance only" in normalized
    assert "adjustment covariate, or new cohort rule" in normalized
    assert "Figure-only steps and steps without" in normalized
    assert "need not duplicate this host-replayed audit" in normalized
    assert "Preserve any trailing time-window suffix" in normalized
    assert "even when the companion count is absent" in normalized
    assert "boolean, datetime, or timedelta count" in normalized


def test_coder_prompt_separates_continuous_and_ordinal_source_status_rules() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "summary semantics explicit" in coder_prompt
    assert "Never apply an ordinal/integer-like check" in coder_prompt
    assert "continuous laboratory value" in coder_prompt
    assert "plausibility, display, or audit range as a range flag" in coder_prompt
    assert "locked protocol says to retain and flag" in coder_prompt


def test_coder_prompt_keeps_long_provenance_notes_out_of_plot_canvas() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Do not place long audit/provenance/reporting notes" in coder_prompt
    assert "inside result plots" in coder_prompt
    assert "write the full notes as a separate table" in coder_prompt


def test_coder_prompt_blocks_mixed_effect_scales_on_one_forest_axis() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Never mix incompatible effect scales on a single forest-plot axis" in (
        coder_prompt
    )
    assert "risk differences" in coder_prompt
    assert "split" in coder_prompt
    assert "the plot by `effect_scale`" in coder_prompt
    assert "Use reader-facing labels in figures" in coder_prompt


def test_coder_prompt_prevents_resume_evidence_polluting_figure_rendering() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "For rendering-only figure steps" in coder_prompt
    assert "explicitly named upstream" in coder_prompt
    assert "previous figure source-data CSVs" in coder_prompt
    assert "robustness panels" in coder_prompt
    assert "render from" in coder_prompt
    assert "that table alone" in coder_prompt
    assert "exact current upstream step outputs directory first" in coder_prompt
    assert "Every typed result input remains part" in coder_prompt
    assert "do not ignore a bound `statistic:` input" in coder_prompt
    assert "Never infer or" in coder_prompt
    assert "recompute a missing statistic" in coder_prompt
    assert "most recent completed record" in coder_prompt
    assert "never rank all historical resume records together" in coder_prompt
    assert "<figure_stem>_source_data.csv" in coder_prompt
    assert "matching PNG, SVG, PDF, and TIFF" in coder_prompt
    assert "one local CSV basename" in coder_prompt
    assert "Never put a dict" in coder_prompt
    assert "absolute path, or path metadata" in coder_prompt
    assert "EASYICU_RUN_DIR" in coder_prompt
    assert "EASYICU_MANIFEST_PARTIAL" in coder_prompt
    assert "100 * count / denominator" in coder_prompt
    assert "conditional on valid-observed records" in coder_prompt
    assert "Never" in coder_prompt and '"sums to 100"' in coder_prompt


def test_figure_contract_repair_guidance_requires_flat_local_source_names() -> None:
    from easyicu.research_agent.plan_utils import _step_contract_repair_guidance
    from easyicu.research_agent.schema import AnalysisStep

    guidance = _step_contract_repair_guidance(
        step=AnalysisStep(
            step_id="02_render",
            intent="Render the planned source-backed result.",
            inputs=["table:result"],
            expected_outputs=["figure:result"],
            method="visualization",
        ),
        step_summary={"figure_files": ["result.png"]},
        code="contract = {'source_data': [{'path': 'result_source_data.csv'}]}",
    )

    assert "one local CSV basename string" in guidance
    assert "flat list of local CSV basename strings" in guidance
    assert "Never write a dict" in guidance


def test_coder_prompt_uses_canonical_positional_source_trace_columns() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "For a positional source-row trace" in coder_prompt
    assert "`source_row_index`" in coder_prompt
    assert "zero-based original row position" in coder_prompt
    assert "`source_table`" in coder_prompt
    assert "Do not" in coder_prompt and "prefix or rename" in coder_prompt
    assert "`key_column` agree with the CSV column name" in coder_prompt


def test_coder_prompt_keeps_figure_source_data_minimal_and_traceable() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Keep this source-data export minimal" in coder_prompt
    assert "unplotted derived" in coder_prompt
    assert "integer-validity" in coder_prompt
    assert "One traced plotted value does not authenticate another" in coder_prompt


def test_coder_prompt_fail_closes_unverifiable_plotted_values() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "before plotting any count-derived rate" in coder_prompt
    assert "denominator to be greater than zero" in coder_prompt
    assert "verify the displayed value from the source event/count numerator" in (
        coder_prompt
    )
    assert "Missing confidence limits are unavailable evidence" in coder_prompt
    assert "Never replace a missing" in coder_prompt
    assert "zero-width interval" in coder_prompt
    assert "disclose every excluded row" in coder_prompt
    assert "source-data audit fields" in coder_prompt
    assert "A row with `n == 0` cannot carry or display" in coder_prompt


def test_coder_prompt_never_renders_partial_structural_accounting() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    assert "Structural accounting figures are stricter" in coder_prompt
    assert "never filter an invalid required source row" in normalized
    assert "leave `figure_files` empty" in normalized
    assert "emit no partial accounting figure" in normalized


def test_coder_prompt_forbids_arbitrary_figure_column_fallbacks() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    normalized = " ".join(load_prompt_pack()["coder"].split())

    assert "Resolve source columns only from explicit" in normalized
    assert "Never fall back to the first numeric column" in normalized
    assert "first non-numeric column" in normalized
    assert "arbitrary frame-order choice" in normalized
    assert "fail the rendering step closed" in normalized


def test_coder_prompt_resolves_prior_tables_by_step_and_kind() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "filter manifest evidence by the exact" in coder_prompt
    assert "`produced_by_step`" in coder_prompt
    assert "required artefact `kind`" in coder_prompt
    assert "never to a code, log, critique" in coder_prompt
    assert "shared a step label" in coder_prompt
    assert "registered only one table" in coder_prompt
    assert "Semantic filename matching is only a tie-breaker" in coder_prompt
    assert "`group_type` plus `group_value`" in coder_prompt
    assert "`level_0` and numeric group value `0`" in coder_prompt
    assert "`*_level` request can map to parent `exposure_level`" in coder_prompt
    assert "exact selected" in coder_prompt
    assert "every range flag declared" in coder_prompt
    assert '`estimate_type == "outcome_risk"`' in coder_prompt
    assert "do not take prevalence `estimate`" in coder_prompt
    assert "observed` and `valid observed`" in coder_prompt


def test_prompt_pack_preserves_the_singular_primary_exposure() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    prompts = load_prompt_pack()

    assert "authoritative primary estimand" in prompts["system"]
    assert "its model is primary" in prompts["coder"]
    assert "singular `primary_exposure`" in prompts["replanner"]
    assert "primary estimand" in prompts["replanner"]
    assert "never mutually adjust" in prompts["coder"]
    assert '`step_summary["model_contracts"]`' in prompts["coder"]
    assert "`separation_detected`" in prompts["coder"]
    assert "term-level coefficient CSV" in prompts["coder"]


def test_coder_prompt_audits_categorical_model_cells_and_reference_choices() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    assert "audit every categorical predictor" in normalized
    assert "its intended reference level" in normalized
    assert "event and non-event counts" in normalized
    assert "sparse, zero-event, and zero-non-event cells" in normalized
    assert "`drop_first=True`" in normalized
    assert "outcome-blind frequency rule" in normalized
    assert "Never change pooling or the reference" in normalized
    assert "finite interior maximum-likelihood" in normalized
    assert "invalid Hessian/Fisher diagnostics" in normalized
    assert "failed ordinary MLE even if" in normalized
    assert "Do not clip a log coefficient" in normalized
    assert "largest finite float" in normalized
    assert "manufacture a huge finite effect" in normalized


def test_coder_prompt_contracts_mixed_outcomes_and_model_effect_scales() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    for field in (
        "outcome",
        "outcome_type",
        "model_family",
        "effect_scale",
        "outcome_unit",
        "interval_method",
    ):
        assert f"`{field}`" in normalized
    assert "A step with mixed outcomes" in normalized
    assert "binary-event count for a binary outcome" in normalized
    assert "is null for a continuous outcome" in normalized
    assert "scale and unit implied by its own model family" in normalized
    assert "conditional-quantile difference in the outcome" in normalized
    assert "never a log-odds or odds-ratio effect" in normalized
    assert "both models share one CSV writer" in normalized


def test_coder_prompt_fail_closes_unvetted_penalized_inference() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    assert "By default a penalized fallback is" in normalized
    assert "point-only" in normalized
    assert '`interval_method="unavailable"`' in normalized
    assert "pseudoinverse or penalized Hessian" in normalized
    assert "`converged=true` merely because" in normalized
    assert "vetted Firth, bootstrap, or shared" in normalized
    assert "penalty curvature matches the fitted objective" in normalized
    assert "optimizer success and objective/KKT diagnostics" in normalized
    assert "Audit-only coefficient p-values are not automatically" in normalized
    assert "Do not pool intercepts, adjustment terms, separate outcomes" in normalized
    assert "prespecified estimands within an outcome" in normalized
    assert "post hoc global family" in normalized


def test_coder_prompt_replays_relaxed_cohort_sensitivity_from_universe() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]
    normalized = " ".join(coder_prompt.split())

    assert "planned cohort sensitivity that can relax primary eligibility" in (
        normalized
    )
    assert "`EASYICU_UNIVERSE_PARQUET`" in normalized
    assert "`COHORT_PARQUET` is already locked and filtered" in normalized
    assert "cannot recover excluded rows" in normalized
    assert "every locked robustness `spec_id`" in normalized
    assert "exactly once, plus one explicit primary row" in normalized
    assert "do not rename, invent, or" in normalized
    assert "substitute variants" in normalized
    assert "universe, primary, and variant counts" in normalized
    assert "intersection, moved-in, and moved-out counts" in normalized
    assert "retain every required row with null estimates" in normalized


def test_coder_prompt_keeps_footnotes_and_raw_ids_out_of_result_figures() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "remove suffixes like `_modeled_or`" in coder_prompt
    assert "replace underscores with spaces" in coder_prompt
    assert "Do not draw figure captions, long footnotes" in coder_prompt
    assert "inside the" in coder_prompt
    assert "saved plotting canvas" in coder_prompt
    assert "so it cannot be clipped" in coder_prompt
