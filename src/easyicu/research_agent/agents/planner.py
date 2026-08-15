"""PlannerAgent and the planner prompt/contract machinery."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..planning.analysis_types import (
    canonical_analysis_family,
    infer_analysis_type,
    locked_analysis_type_guide,
    CATALOG_DETAIL_LADDER,
    planner_analysis_family_authority_guide, validate_host_authorized_analysis_family,
)
from ..planning import scientific_action_catalog as _scientific_actions
from ..planning.primary_result_contract import (
    primary_result_contract_guide,
    validate_required_primary_result as _validate_required_primary_result,
)
from ..trajectory.plan_contract import (
    trajectory_planner_contract_guide,
)
from ..resources import ContextBudgetExceeded, bounded_request_metrics
from ..cohort.schema import (
    ALLOWED_CTAS_AGGREGATIONS,
    _resolve_predicate_column,
    known_concept_ids,
    validate_plan_typed_bindings_against_context,
)
from ..icu_rules import (
    GENERAL_ICU_ANALYSIS_PRINCIPLES,
)
from ..providers.protocol import LLMClient, LLMMessage
from ..providers.capabilities import llm_supports_strict_json_schema
from ..providers.llm import llm_is_mockish
from ..providers.prompt_budget import (
    CONSERVATIVE_BYTES_PER_TOKEN,
    DEFAULT_MAX_PROMPT_TOKENS,
)
from ..authority.secret_redaction import (
    redact_text_secrets,
)
from ..contracts.primary_cohort import primary_analysis_cohort_plan_findings
from ..planning.robustness_contract import (
    COMPLETE_CASE_STRATEGY as _COMPLETE_CASE_STRATEGY,
    COMPLETE_CASE_VARIABLES_KEY as _COMPLETE_CASE_VARIABLES_KEY,
    RobustnessPlanError,
    validate_planner_robustness_specs,
)
from ..planning.planner_output_contract import (
    validate_fresh_planner_typed_product_specs,
)
from ..planning.adjustment_authority import (
    validate_plan_against_adjustment_authority,
)
from ..authority.declared_levels import bind_step_declared_levels
from ..authority.table_one_binding import bind_table_one_execution_spec
from ..research_context.prompt_scope import (
    planner_variable_catalog,
    scoped_planner_context,
)
from ..research_context.prompt_variables import opaque_level_tokens
from ..schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES,
    COHORT_DEFINITION_COHORT_OUTPUT,
    COHORT_DEFINITION_FLOW_OUTPUT,
    AnalysisPlan,
    ResearchContext,
)
from ..planning.sensitivity_authority import EXECUTABLE_METHODS_BY_STRATEGY
from . import plan_payload as _payload
from .plan_payload import (
    _normalise_plan_payload,
)

from ._support import LLM_PARSE_DEBUG_CHARS, PLANNER_MAX_RETRIES, _SYSTEM_GUIDE, _closed_cohort_product_sentence, _dump_raw, _first_json_block, _format_context, _host_executed_cohort_step_sentence, _robustness_axis_vocabulary, _strip_code_fence

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


_COHORT_PREDICATE_AGGREGATIONS = (
    "max",
    "min",
    "first",
    "last",
    "mean",
    "median",
    "any",
    "sum",
    "count",
)


def _bindable_concept_ids(columns: Sequence[str]) -> list[str]:
    """The concept ids that resolve against THIS run's sealed columns.

    A cohort predicate is checked by ``cohort/schema.py`` against the sealed
    input, so the dictionary is the wrong set to publish: it is what EasyICU
    can define, not what this export contains.

    MEASURED on canary12's E3 cohort (104 columns): the prompt published 264
    ids as "the ONLY values acceptable" and 15 of them bound -- 94.3% of the
    menu was unusable. The Planner picked ``kdigo_aki``, the scientifically
    correct concept for an AKI-stage cohort, from the list the host handed it,
    and the host then refused it for having no bound column; the next attempt
    improvised ``aki_stage``, which is not a concept at all. Two of five
    planning attempts spent on a menu that was wrong to begin with.
    """

    resolve = _resolve_predicate_column
    names = [str(column) for column in columns if str(column).strip()]
    if not names:
        return []
    return sorted(
        concept_id
        for concept_id in known_concept_ids()
        if any(
            resolve(names, concept_id, aggregation, column_bindings={}) is not None
            for aggregation in _COHORT_PREDICATE_AGGREGATIONS
        )
    )


def _format_concept_id_allowlist(columns: Sequence[str] = ()) -> str:
    """Render legal EasyICU concept ids for CTAS planner prompts."""

    concept_ids = _bindable_concept_ids(columns)
    scope = (
        "that bind against this run's sealed export"
        if concept_ids
        # No sealed columns to check against -- publish the dictionary rather
        # than an empty menu, which would make the cohort unwritable. The
        # binder still refuses an unbound predicate downstream, so this is
        # permissive in the prompt only.
        else "in the concept dictionary (this run's sealed columns were not "
        "available to narrow them)"
    )
    if not concept_ids:
        concept_ids = sorted(known_concept_ids())
    if not concept_ids:
        return (
            "ALLOWED concept_ids — no concept dictionary entries were loaded. "
            "Do not invent concept_id values; ask for a configured concept "
            "dictionary before emitting a CohortDefinition."
        )
    lines = [
        f"ALLOWED concept_ids — the ONLY values acceptable in any "
        f"CohortDefinition or RobustnessSpec.cohort_override.concept_id field. "
        f"This is the set {scope}; a concept EasyICU can define but this "
        "export does not carry is not on it, however apt it sounds. "
        'Synthesizing new names (e.g. "score_at_admission", '
        '"concept_peak_window", "condition_onset_window") is forbidden — these '
        "are operationalizations, not concepts. To operationalize a concept "
        "over a time window, use the 5-tuple form: "
        'concept_id="<one from the list below>" + time_window + aggregation.',
        "",
    ]
    lines.extend(f"- `{concept_id}`" for concept_id in concept_ids)
    return "\n".join(lines)


def _format_ctas_schema_constraints() -> str:
    """Render CTAS enum and cross-field constraints for planner prompts."""

    enum_values = ", ".join(f'"{value}"' for value in ALLOWED_CTAS_AGGREGATIONS)
    return (
        "CTAS SCHEMA CONSTRAINTS — you MUST satisfy ALL of these in every "
        "ConceptPredicate, including those inside "
        "robustness_specs[*].cohort_override:\n\n"
        "1. aggregation MUST be exactly one of these 10 values (case-sensitive, "
        f"no synonyms): {enum_values}. Do NOT use ICU rule labels or synonyms "
        'such as "first_value", "max_or_last", "mean_or_median", '
        '"mean_median", "median_only", "latest", "most_recent", '
        '"earliest", "average", or "total" as aggregation values; these '
        "are not CTAS enum values and will be rejected.\n\n"
        "2. time_window.end_offset_hours MUST be strictly greater than "
        "time_window.start_offset_hours. Zero-width windows (end == start) "
        "are invalid. If you want a single instant, use a small window like "
        "[0h, 1h] instead.\n\n"
        '3. aggregation "any" / "all" can only be paired with op in '
        '{"==", "!=", "missing", "not_missing"}; they yield '
        "booleans, not numeric thresholds.\n\n"
        "4. If the ResearchContext shows an ICU rule label in an "
        "`agg_default=... (icu_rule=...)` annotation, translate it before "
        'writing CTAS JSON: first_value -> "first"; max_or_last -> '
        '"max" or "last" (prefer "max" for acuity scores); '
        'mean_or_median / mean_median -> "mean" or "median" (prefer '
        '"median" for robustness checks); median_only -> "median".'
    )


def _build_planner_user_prompt(
    context: ResearchContext,
    *,
    know_how_context: str = "",
    planning_contract_context: str = "",
    catalog_detail: str = "full",
    strict_transport_schema: bool = False,
) -> str:
    """Build the planner user prompt with runtime concept-id grounding.

    ``catalog_detail`` is chosen by :func:`_planner_prompt_within_budget` from
    the byte budget alone.  Do not vary it on anything else.
    """

    planner_context = scoped_planner_context(context)
    inferred_analysis_type = infer_analysis_type(context)

    sensitivity_specs = (
        context.user_preferences.sensitivity_specs
        if context.user_preferences is not None
        else []
    )
    sensitivity_guide = ""
    if sensitivity_specs:
        rows = []
        for spec in sensitivity_specs:
            methods = ", ".join(
                repr(value)
                for value in sorted(EXECUTABLE_METHODS_BY_STRATEGY[spec.strategy])
            )
            rows.append(
                f"- {spec.spec_id}: axis={spec.axis}, strategy={spec.strategy}; "
                f"bind this exact id in sensitivity_spec_ids and use one of [{methods}]."
            )
        sensitivity_guide = (
            "\n\nThe typed `user_preferences.sensitivity_specs` are exact "
            "user-reviewed authority, not suggestions. Every listed spec must "
            "be implemented by an executable primary/secondary/sensitivity step "
            "that emits a table/statistic/model/dataset/artifact and copies its "
            "exact id into `sensitivity_spec_ids`. Copy its execution variables "
            "and numeric/eligibility settings exactly; do not replace them with "
            "prose or merge away a scientific axis. Legal bindings for this "
            "context are:\n" + "\n".join(rows) + "\n\n"
        )

    opaque_binary_levels = list(opaque_level_tokens(2))
    opaque_binary_json = json.dumps(
        opaque_binary_levels, ensure_ascii=True, separators=(",", ":")
    )
    opaque_level_1_json = json.dumps(opaque_binary_levels[0], ensure_ascii=True)
    opaque_level_2_json = json.dumps(opaque_binary_levels[1], ensure_ascii=True)

    prompt = (
        "Produce an ICU-AWARE RESEARCH PLAN as JSON matching the "
        "AnalysisPlan schema. First infer the EHR analysis type, "
        "then choose only the steps justified by that family and "
        "the available context. The plan must not assume that "
        "every task needs Table 1, outcome incidence, missingness, "
        "or a primary association model."
        + sensitivity_guide
        + "Table 1 is standard for observational/association and prediction "
        "families (STROBE item 14 / TRIPOD): include a `table:table_one` step "
        "describing the analytic cohort before the primary analysis. Omit it "
        "only for a family that genuinely does not call for one, such as a pure "
        "feasibility/protocol task or clustering already described per cluster. "
        "A step that declares the exact output `table:table_one` MUST also "
        "declare `table_one_spec`: group_by, at least two closed group_levels, "
        "and a variables roster whose name/kind/summary/test/closed levels "
        "encode the scientific comparison. THE COLUMN YOU GROUP ON IS NOT ALSO "
        "A ROW -- it would report each group as 100% itself. Name it in "
        "`group_by` or in `variables`, never in both. "
        "That same step's `inputs` must explicitly list its `group_by` and "
        "every `variables[*].name`, in addition to the typed cohort artifact; "
        "a column named inside the spec is not implicitly an input. "
        "Levels follow the variable kind: a "
        "'categorical' row summarised 'count_percent' requires at least two "
        "closed levels; an 'ordinal' row summarised numerically may declare "
        "its closed levels (a 0-4 organ score, a 0-3 stage) and the host then "
        "stops the step on any value outside them, or may omit levels "
        "entirely; a 'continuous' row must leave levels empty. Table 1 means "
        "Overall plus grouped "
        "columns. Preserve observed scalar types exactly: numeric 0/1 levels "
        "must be JSON numbers, never the strings '0'/'1'. "
        "When the variable catalog withholds categorical literals and supplies "
        "`opaque_levels`, copy those exact opaque tokens into group_levels or a "
        "categorical variable's levels. The host will bind them locally to the "
        "digest-verified observed values; never guess a hidden label. For a "
        f"two-level field the exact token array is `{opaque_binary_json}`; use "
        "those same tokens for scalar selectors such as an event value or a "
        "reference/comparison level. "
        "Report per-group missing n (%), one variable-appropriate P value, "
        "and the test name. Declare `missing_group_policy`: 'fail_closed' "
        "stops the step if any row's group_by value is missing, and "
        "'exclude_and_report' removes those rows from the whole table, Overall "
        "included, and reports how many were removed. Check the variable "
        "catalog's missingness for the column you group on: a grouping "
        "variable derived from measurements is rarely observed on every stay, "
        "and 'fail_closed' on such a column ends the step with no result. "
        "A step with `table_one_spec` may declare only "
        "`table:table_one` plus the optional host-audit outputs "
        "`table:cohort_flow` and `log:source_row_count_reconciliation`; put "
        "every other result or figure in a separate step. If only an "
        "ungrouped cohort description is wanted, "
        "emit `table:cohort_summary` instead and omit table_one_spec. "
        "For counts, events, prevalence, absolute risk, or outcome by group BY "
        "EXPOSURE LEVEL, declare `table:exposure_outcome_distribution` and its "
        "spec. Another table name sends the same science to generated code: a "
        "different shape every run that no host figure can consume it. Name the "
        "step whatever your reader should see; it is the declared OUTPUT that "
        "decides who computes it.\n\n"
        "That product covers ONE outcome. A summary of a CONTINUOUS variable by "
        "the same exposure levels cannot be added to it. Wanting both is not a "
        "reason to give up the product: put the continuous summary in a separate "
        "step and keep the event/rate table typed.\n\n"
        "A host-drawn figure consumes EXACTLY the typed product it renders. "
        "Adding adjusted estimates, robustness matrices, audit tables, or other "
        "inputs asks for a composite figure; no host renderer can draw it. Put "
        "that context in text or put it in its own figure step.\n\n"
        "A step that declares the exact output "
        "`table:exposure_outcome_distribution` MUST also declare "
        "`exposure_outcome_distribution_spec`, and it must consume exactly one "
        "typed cohort input. Every field is required: the exposure column and "
        "its closed exposure_levels; the outcome column and its closed "
        "outcome_levels; the exact outcome_positive_value, which must be one "
        "of outcome_levels; a level_match_policy of 'exact_typed' or "
        "'numeric_string_equivalent'; a denominator_policy of "
        "'all_declared_rows' or 'observed_outcome_rows'; a "
        "missing_outcome_policy of 'fail_closed', 'exclude_from_denominator' "
        "(which requires 'observed_outcome_rows') or "
        "'structural_absence_is_non_event' (which requires "
        "'all_declared_rows'); a missing_exposure_policy of 'fail_closed' "
        "(the default: any stay with no exposure value stops the step) or "
        "'exclude_from_denominator' (complete-case on the exposure, and the "
        "count that left travels in the table) -- check the catalog's "
        "missingness for the exposure column, because 'fail_closed' on a "
        "column derived from measurements ends the step with no result; and a "
        "confidence_level. " + _payload.counts_only_distribution_guide() + " Close outcome_levels "
        "over every value the source can actually hold: any other observed "
        "value stops the step, because an undeclared value would otherwise be "
        "counted as a non-event and silently deflate every rate. Which "
        "denominator a prevalence or rate is taken over, what an unobserved "
        "outcome means, and at what coverage an interval is built are parts of "
        "the study design, not rendering details, so state them. The host will "
        "not infer the exposure, the outcome, the event value, or any policy "
        "from column names or from input order. Preserve observed scalar types "
        "exactly, as for table_one_spec: a boolean column is never matched by "
        "a numeric level. When the research question asks for an absolute-risk "
        "DIFFERENCE, the spec must additionally declare exactly one "
        "risk_difference_contrast with typed reference_exposure_level and "
        "comparison_exposure_level (reported as comparison minus reference), "
        "effect_measure='risk_difference', and "
        "interval_method='linear_probability_wald'. Do not sort or infer those "
        "levels. Do not declare `dependence`: when the StudyContext carries a "
        "typed repeated-unit analysis_design and verified patient grouping, the "
        "host binds that exact covariance authority after parsing; otherwise an "
        "invented grouping must fail closed. If a post-baseline exposure remains "
        "descriptive because exposure opportunity is unresolved, a PRIMARY "
        "distribution step must carry descriptive_claim with "
        "claim_ceiling='descriptive_only' and the typed limitation "
        "'post_baseline_exposure_opportunity_unresolved'; that contract does not "
        "authorize association or causal language. "
        + _payload.descriptive_claim_shape_guide()
        + " "
        "A primary cohort construction/eligibility + attrition step is also a "
        "strict execution boundary: it must declare exactly one materialised "
        "closed cohort product (" + _closed_cohort_product_sentence() + ") "
        "plus only canonical attrition or "
        "denominator tables. Do not place a baseline/cohort summary, Table 1, "
        "model, statistic, figure, or other side output in that raw-universe "
        "step. Put each such output in a downstream step that consumes the "
        "declared closed cohort product. "
        "Table 1 enum values are exact: `variable_kind` is one of "
        "`continuous`, `categorical`, or `ordinal` (a binary variable is "
        "`categorical`); `summary` is one of `mean_sd`, `median_iqr`, `both`, "
        "or `count_percent`; `test` is one of `welch_t_or_anova`, "
        "`mann_whitney_or_kruskal`, or "
        "`chi_square_with_fisher_exact_for_sparse_2x2`. Do not emit shorthand "
        "aliases such as `binary`, `n_percent`, `mann_whitney_u`, or "
        "`chi_square`. "
        "The host executes the declared Table 1 design; it does not choose the "
        "grouping variable or tests for you. The main Table 1 grouping must "
        "represent the study's primary scientific comparison (for example, an "
        "exposure group or outcome group). Auxiliary measurement/source-status "
        "flags belong in a separately named data-quality table and must not be "
        "used as the main `table:table_one` grouping unless the user explicitly "
        "requests that comparison. "
        "If cross-database "
        "replication is requested, include a cross-database step, "
        "but mark it as a feasibility / protocol step unless the "
        "ResearchContext explicitly provides external cohort files. "
        "Use score-specific QC steps only when a relevant score is "
        "actually central to the question. Do not put invented "
        "prefixed variables such as eicu:age in `inputs`. Honor "
        "explicit user preferences and requested outputs when they "
        "are compatible with the cohort and analysis family. Similar-study "
        "eligibility criteria are design candidates, not automatic authority: "
        "apply one only when it matches the current target population, estimand, "
        "index time, and available typed fields. Put an unverifiable literature "
        "criterion in the rationale as unresolved rather than inventing a field "
        "or silently excluding rows. Never claim first admission, one stay per "
        "patient, or patient-level deduplication unless a patient identifier and "
        "the required admission chronology are actually available.\n"
        "Choose step boundaries that make the analysis reviewable. A figure is "
        "its own step: declaring a `figure:` product in the same step as the "
        "table or statistic that feeds it asks one executor for two different "
        "kinds of work, and the host owns those separately -- so a bundled step "
        "loses the deterministic owner the result table would otherwise have "
        "had. Declare the result table in one step and the figure that displays "
        "it in a downstream visualization step that consumes it. "
        "Within one plan a bare product name is one answer: never declare the "
        "same name under two kinds, such as `table:x` together with "
        "`statistic:x`. A table and a scalar are two different answers and need "
        "two different names; one name declared twice can only be satisfied by "
        "handing the same artifact over for both, which tells a reader who "
        "asked for a number to read a table instead. "
        "A later step "
        "may consume an earlier standardized artifact only when that dependency "
        "is explicit in `inputs` and the producer declares it in "
        "`expected_outputs`; never rely on hidden in-memory state. Do not force "
        "prediction, clustering, or any other family into a hard-coded mega-step "
        "or split it solely to fit a shared pipeline template.\n\n"
        + primary_result_contract_guide()
        + "\n\n"
        "The typed `model_requirements` roster currently covers only a complex "
        "binary/continuous adjusted-association step whose method is exactly "
        "`adjusted_association_models` and whose expected outputs include "
        "`table:adjusted_association_estimates`. For that supported contract, "
        "record the step's pre-specified estimand in the roster instead of "
        "leaving the scientific decision only in prose. "
        "ONE MODEL PER STEP: the roster is that step's own model, so declare "
        "exactly one entry. A second estimand -- a different outcome, or the "
        "same one under a different analysis set -- is its own step with its "
        "own roster entry, and a step declaring two loses the deterministic "
        "owner its result table would otherwise have had, exactly as a bundled "
        "figure does. Each "
        "entry has `requirement_id`, `outcome`, `outcome_type` (binary or "
        "continuous), `method_family`, `exposure_source`, `analysis_role` "
        "(primary, secondary, or sensitivity), `analysis_set` (source_aware or "
        "complete_case), `required_for_step_success`, and `covariates`. You "
        "decide this roster; "
        "the execution layer only verifies it. `method_family` is matched "
        "against an exact set, so here is the set. outcome_type 'binary': "
        + ", ".join(sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES))
        + ". outcome_type 'continuous': "
        + ", ".join(sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES))
        + ". No other label passes, and neither does one from the other list. "
        + _payload.planner_adjusted_association_owner_guidance()
        +
        "Primary and secondary entries must be required for step "
        "success; only a sensitivity entry may be optional. "
        "AN ORDINAL OR CATEGORICAL EXPOSURE is one model with several "
        "contrasts, not one number. When the exposure has discrete levels -- a "
        "severity stage, a grade, a pre-specified quantile band -- declare "
        "`exposure_levels` (the closed level set, in order), "
        "`exposure_reference_level` (what every contrast is taken against) and "
        "`primary_contrast_level` (the ONE contrast the manuscript reports). "
        "All three together or none: with more than two levels the host will "
        "not choose which contrast is the headline, because the top level "
        "against the reference and a per-level trend are different scientific "
        "claims. Leave all three unset for a binary or continuous exposure, "
        "which is one contrast and one row. Declaring them makes the host fit "
        "and report every contrast from a single model; omitting them on a "
        "multi-level exposure fits it as one linear term, which answers a "
        "different question.\n\n"
        "`covariates` is the exact adjustment set for that model: list the "
        "exact analysis column names you intend to condition on, use `[]` to "
        "declare a deliberately unadjusted model, and omit it only when you "
        "genuinely have not fixed the adjustment set. The execution layer will "
        "not reconstruct an adjustment set from the step inputs -- listing "
        "columns under `inputs` states what the step may read, not what the "
        "model conditions on, and the difference between those two is a "
        "scientific decision that is yours. It must not contain the outcome or "
        "the exposure. Leave the array empty "
        "for survival, prediction, "
        "mixed-effects, clustering, and every other analysis family; those use "
        "their own family-specific planning and validation contracts.\n\n"
        "For a counting-only measurement/missingness audit step, set "
        "`measurement_audit_spec.products`: one entry per declared table "
        "product, each giving the product's exact `product_id` (the declared "
        "name without its `table:` prefix) and which `audit` it is. Name your "
        "products whatever your reader should see; the `audit` field is what "
        "the execution layer reads, so a descriptive name no longer costs the "
        "step its deterministic executor. Legal audits are "
        "`measurement_missingness` (was each concept measured at all, and how "
        "much of its value column is missing), `missingness_profile` (the same "
        "counts as a plain missing-n/percent profile), `measurement_source` "
        "(where each value could have come from, including flag/value "
        "disagreements), `measurement_process` (how often and when each "
        "concept was observed), `event_timing` (per event-timed concept: "
        "present, absent, before the declared origin, time missing), "
        "`component_completeness` (per component of a composite score), "
        "`analytic_denominators` (rows surviving each analytic filter) and "
        "`cohort_flow`. Two products may not name the same audit -- that would "
        "promise a reader two tables and deliver one twice. Declare only "
        "audits from this list; if the step needs something else, leave "
        "`measurement_audit_spec` unset rather than mislabelling it.\n\n"
        + _payload.planner_descriptive_method_guidance(inferred_analysis_type.key)
        + "For a step that re-estimates the ALREADY-LOCKED robustness "
        "specification grid without changing the estimand, set "
        "`robustness_replay_spec.products`: one entry per declared product, "
        "each giving the bare `product_id` and which `output` it is. Legal "
        "outputs are `robustness_matrix` (per locked specification: estimate, "
        "interval, n), `robustness_summary`, `specification_grid` (the locked "
        "grid itself), `membership_change` (cohort overlap and attrition "
        "against the primary set), `outcome_label_executability`, "
        "`missingness_strategy_notes`, `primary_effect` and `complete_case_n`. "
        "Two products may not name the same output. Declaring this spec is a "
        "SCIENTIFIC CLAIM that the step is exactly that replay -- if the step "
        "introduces new science (a different estimator, a causal-emulation or "
        "weighting variant, an E-value, a negative control), it is a different "
        "step and must NOT carry this spec, even though doing so would give it "
        "a host runner. Name the step and its products whatever your reader "
        "should see; the `output` field is what the execution layer reads.\n\n"
        "Use `input_consumption_contracts` when a step consumes typed result "
        "tables and cardinality matters. `all_rows` preserves the complete "
        "table; `single_row` is valid only for a true singleton; "
        "`one_per_role` requires an exact `role_column` and complete "
        "`expected_roles` roster. Never select the first row or assume a table "
        "has one result merely because the downstream step renders one figure. "
        "Every item uses the exact keys `input_key` and `mode`, for example "
        '`{"input_key":"table:exact_product","mode":"all_rows"}`; never '
        "rename them to `input` and `cardinality`. "
        "Leave this array empty when no typed-table cardinality rule is needed.\n\n"
        + _payload.figure_panel_shape_guide() + " Leave `figure_panels` empty on non-visualization steps.\n\n"
        "When the ResearchContext carries `materialized_inputs`, every raw "
        "dataframe field in `steps[*].inputs`, `table_one_spec`, "
        "`model_requirements` outcome/exposure fields, and robustness "
        "missingness/outcome fields MUST copy an exact name from the sealed "
        "cohort column roster. Concept ids belong in typed cohort predicates; "
        "they are not aliases for raw step inputs. An upstream logical product "
        "must use its explicit `kind:name` value from a producer's "
        "`expected_outputs`. Never substitute a human label or a plausible "
        "synonym for a sealed column. Cohort `id_columns` and `time_columns` "
        "are HOST NAVIGATION COORDINATES, not executable analysis fields: do "
        "not list them in step inputs, Table 1, model requirements, or "
        "robustness fields. Cohort accounting should consume the explicit "
        "analysis-cohort product and report its denominator; the host owns row "
        "identity and navigation.\n\n"
        "When a plan requests a manuscript-facing figure, declare a top-level "
        "`display_labels` object for every variable, contrast, endpoint, or "
        "robustness spec id whose human-facing wording matters. Labels are "
        "presentation metadata only: they must describe the already-selected "
        "scientific object and must not change its exposure, outcome, method, "
        "cohort, or estimand. The renderer will not infer that a token such as "
        "`death` means ICU, hospital, or fixed-day mortality.\n\n"
        "Keep eligibility separate from exposure: primary-cohort predicates "
        "must preserve every closed level compared by a downstream Table 1 or "
        "required primary estimand, including prevalence denominators.\n\n"
        + _format_concept_id_allowlist(
            [variable.name for variable in context.variables]
        )
        + "\n\n"
        + _format_ctas_schema_constraints()
        + "\n\n"
        "Every cohort/exposure/outcome concept used to define the "
        "analysis population must be represented as a typed cohort "
        "definition: concept_id, time_window, aggregation, operator, "
        'and value. You may write `cohort: {"from_named": "..."}` '
        "only when the caller has explicitly registered that named "
        "pattern for this case; otherwise supply the full five-tuple "
        "predicate. Free-text cohort strings are invalid.\n"
        "If your plan includes any cohort-definition, eligibility, or "
        "attrition step, choose exactly one population mode: either set "
        "`cohort.selection_mode='predicate_filtered'` and provide at least one "
        "structured `inclusion`/`exclusion` predicate, or explicitly set "
        "`cohort.selection_mode='all_input_rows'` with both predicate arrays "
        "empty. A default empty cohort is rejected because it cannot distinguish "
        "an intentional full-input population from omitted free-text 纳排.\n\n"
        "If a cohort-definition or eligibility step also reports attrition, "
        "its `expected_outputs` MUST declare exactly one materialised closed "
        "primary-cohort product: " + _closed_cohort_product_sentence() + ". "
        "A definition, "
        "protocol, or status output such as `artifact:cohort_defined` is not a "
        "cohort dataset and cannot replace that product. The other outputs in "
        "that step may only be canonical attrition/flow tables. Every "
        "downstream step consuming that cohort declares the SAME key in its "
        "`inputs`; a step whose row authority is the closed cohort product but "
        "which spells it any other way is executed by nobody.\n\n"
        # The wide vocabulary above is legal but only one pair is EXECUTABLE by
        # the host, and a Planner given five equally-endorsed spellings picked
        # a non-executable one in 64 of 282 recorded plans.
        #
        # This paragraph was ~2.8 KB when first written, carrying its measured
        # motivation ("127 of 127", "59% of every cascade") at full length. On
        # a synthetic minimal context that left 29 KB of headroom and looked
        # free; on the real typed contexts it is not. h1 died at 80,071 bytes
        # against an 80,000 limit -- 71 over, from an addition of 2,819. The
        # Planner needs the INSTRUCTION; the evidence for it belongs in the
        # commit message and the tests, which is where it now lives. Measure
        # any further addition against a REAL large context, never a minimal
        # one.
        "Of those spellings, exactly one lets the HOST perform this step "
        "rather than the code generator: `expected_outputs` is exactly "
        + _host_executed_cohort_step_sentence()
        + " and nothing else. The cohort is already materialised and "
        "digest-bound, so under that declaration the host reads the bound "
        "rows, writes the attrition table and emits the identity receipt "
        "itself. Any other spelling of the same two products, or a third "
        "output beside them, goes to the code generator -- the most expensive "
        "place in the system to lose the host.\n\n"
        "When that step has no exclusion left to apply -- the bound cohort IS "
        "the analysis set -- also declare `cohort_definition_spec`: "
        "`identity_column` (the row key the receipt hashes; the host will not "
        "guess it) and one `eligibility_criteria` entry per attrition row, "
        "each a `criterion_id` plus a `description`. Those entries DOCUMENT "
        "eligibility the bound cohort already satisfies, so a criterion that "
        "still has rows to remove does not belong there; a step that must "
        "genuinely exclude rows omits the spec.\n\n"
        "When robustness is required, pre-specify one or more executable, "
        "task-supported `robustness_specs`; never invent an unsupported axis, "
        "endpoint, or variable. "
        # "Never invent an unsupported axis" was already here, and the closed
        # set it refers to was not. A real Planner guessed 'model' on one run
        # and 'functional_form' on the next, each costing an attempt. Read the
        # vocabulary off the contract so publishing it cannot drift from
        # enforcing it. A design that needs a different axis belongs in its own
        # analysis step, not in a widened robustness axis.
        "`axis` is closed: use exactly one of "
        + ", ".join(f"'{value}'" for value in _robustness_axis_vocabulary())
        + ". A sensitivity that does not fit one of these (for example an "
        "alternative model form or link function) is a separate analysis "
        "step, not a new axis value. "
        # Published because it was enforced and never stated: the worked
        # example below used to show `{"strategy": "complete_case"}` with no
        # variable list, and half of all recorded complete-case specs copied
        # it and were refused at execution. Which variables are held complete
        # is a scientific choice, so the host asks rather than infers.
        f"A `missing_override` whose `strategy` is '{_COMPLETE_CASE_STRATEGY}' "
        f"MUST also carry `{_COMPLETE_CASE_VARIABLES_KEY}`: the exact list of "
        "column names whose completeness defines the analysed set -- normally "
        "the exposure, the outcome and every covariate of the primary model. "
        "The host will not infer them from the model, because restricting on a "
        "narrower or wider set than the model uses is a different analysis, and "
        "a spec without them is refused. "
        "Add an auxiliary post-primary step with "
        "`method='robustness_sensitivity'` producing "
        "`table:robustness_matrix` and `statistic:robustness_summary`. "
        "Variants must not change the primary analysis. "
        # The step this sentence asks for IS the step that re-estimates the
        # locked grid, and saying so here is the whole point: a real run
        # declared the grid, created this step, left robustness_replay_spec
        # unset because that field is described ninety lines earlier behind a
        # warning about new science, and ended with "locked robustness
        # specifications that no step estimated" -- blank panel rows and a step
        # that blocked itself with n_converged_variants=0 after producing every
        # other output. Generic refitting is deliberately disabled, so the
        # declaration is the ONLY route to an estimate.
        "THAT STEP MUST CARRY `robustness_replay_spec`: declaring "
        "`robustness_specs` obliges exactly one step to re-estimate them, and "
        "with no such declaration the locked grid is reported as estimated by "
        "nobody and the step fails. Its `products[*].product_id` are this "
        "step's OWN declared output names with the `kind:` prefix removed "
        "(declare `table:robustness_matrix`, write `product_id` "
        "'robustness_matrix'), which is a different field from `output` -- "
        "naming an `output` value the step does not itself declare is "
        "refused."
        + _payload.planner_descriptive_robustness_guidance(inferred_analysis_type.key)
        + locked_analysis_type_guide(inferred_analysis_type)
        + "\n\n"
        + _scientific_actions.planner_scientific_action_guide(inferred_analysis_type.key, detail=catalog_detail) + "\n\n"
        + planner_analysis_family_authority_guide(
            context, inferred_analysis_type, detail=catalog_detail
        )
        + "\n\n"
        + trajectory_planner_contract_guide(
            context=context,
            analysis_type=inferred_analysis_type.key,
        )
        + "\n\n"
        "OUTPUT FORMAT — VERY IMPORTANT:\n"
        "Return *only* a single JSON object matching the "
        "AnalysisPlan schema. No prose, no markdown headings, no "
        "trailing commentary. A ```json … ``` fence is acceptable; "
        "anything outside that fence will be discarded.\n\n"
        "Required JSON shape (truncated example):\n"
        "The example values are illustrative only; do not prefer SOFA or "
        "any example concept unless the ResearchContext supports it. The "
        "distribution example is secondary so it cannot accidentally become a "
        "second headline; change it to primary only when it is the plan's sole "
        "headline estimand. Across the complete plan, at most one step is "
        "primary.\n"
        "{\n"
        '  "research_question": "<copy from context>",\n'
        '  "display_labels": {"<exact variable or spec id>": "<human-facing label>"},\n'
        '  "cohort": {\n'
        '    "name": "primary",\n'
        '    "inclusion": [\n'
        "      {\n"
        '        "concept_id": "<easyicu concept id>",\n'
        '        "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 24},\n'
        '        "aggregation": "max",\n'
        '        "op": ">=",\n'
        '        "value": 1\n'
        "      }\n"
        "    ],\n"
        '    "exclusion": []\n'
        "  },\n"
        # ResearchContext is the unique endpoint authority. The projection keeps
        # old plans readable; Planner-authored contents would be a second truth.
        '  "endpoint": null,\n'
        '  "steps": [\n'
        # The cohort-definition step is shown first because it IS first in
        # every recorded plan, and because prose alone did not convey the
        # nested criterion objects -- the same lesson the distribution spec
        # below already paid for. The two product names are literal, not
        # placeholders: this exact pair is what the host executes.
        # Only the fields this step's shape actually turns on: the two literal
        # product names and the nested criterion object. The other keys are
        # already demonstrated by the two steps below, and every byte here is
        # a byte taken from the typed context.
        "    {\n"
        '      "step_id": "01_define_analysis_cohort",\n'
        '      "planned_analysis_role": "auxiliary",\n'
        '      "intent": "<one sentence>",\n'
        '      "inputs": [],\n'
        '      "expected_outputs": ["' + COHORT_DEFINITION_COHORT_OUTPUT + '", '
        '"' + COHORT_DEFINITION_FLOW_OUTPUT + '"],\n'
        '      "method": "cohort_definition_and_attrition",\n'
        '      "cohort_definition_spec": {\n'
        '        "identity_column": "<exact id column from the cohort roster>",\n'
        '        "eligibility_criteria": [\n'
        '          {"criterion_id": "<flow-table row key>", '
        '"description": "<what it required>"}\n'
        "        ]\n"
        "      }\n"
        "    },\n"
        "    {\n"
        '      "step_id": "02_table_one",\n'
        '      "planned_analysis_role": "auxiliary",\n'
        '      "intent": "<one sentence>",\n'
        '      "inputs": ["<variable names from context>"],\n'
        '      "expected_outputs": ["table:table_one"],\n'
        '      "method": "descriptive",\n'
        '      "icu_rule_refs": ["aggregation_rule_for"],\n'
        '      "model_requirements": [],\n'
        '      "input_consumption_contracts": [],\n'
        '      "table_one_spec": {\n'
        '        "group_by": "<declared grouping variable>",\n'
        '        "group_levels": ' + opaque_binary_json + ",\n"
        '        "variables": [\n'
        "          {\n"
        '            "name": "<declared row variable>",\n'
        '            "variable_kind": "continuous",\n'
        '            "summary": "median_iqr",\n'
        '            "test": "mann_whitney_or_kruskal",\n'
        '            "levels": []\n'
        "          }\n"
        "        ]\n"
        "      },\n"
        '      "trajectory_stability_spec": null,\n'
        '      "exposure_outcome_distribution_spec": null,\n'
        '      "cohort_definition_spec": null\n'
        "    },\n"
        # A second example step exists for one reason: the distribution spec
        # was described in prose only, and a real Planner then guessed its
        # shape four different ways in five attempts -- `exposure_column`
        # instead of `exposure`, and twice an `{"column": ..., "levels": ...}`
        # object where a plain column name belongs. Prose is enough to convey
        # which choices are the Planner's; it is not enough to convey a key
        # name or whether a field nests. Show the shape, as table_one_spec
        # already does.
        "    {\n"
        '      "step_id": "03_exposure_outcome_distribution",\n'
        '      "planned_analysis_role": "secondary",\n'
        '      "intent": "<one sentence>",\n'
        # Exactly one typed input (the cohort artifact, which carries the
        # digest and product contract), plus the exposure and outcome column
        # names as bare inputs -- the schema requires both spec columns to be
        # explicit step inputs, and a bare name is not a second typed input.
        '      "inputs": ["artifact:analysis_cohort", '
        '"<declared exposure column name>", '
        '"<declared outcome column name>"],\n'
        '      "expected_outputs": ["table:exposure_outcome_distribution"],\n'
        '      "method": "descriptive",\n'
        '      "icu_rule_refs": [],\n'
        '      "model_requirements": [],\n'
        '      "input_consumption_contracts": [],\n'
        '      "table_one_spec": null,\n'
        '      "trajectory_stability_spec": null,\n'
        '      "exposure_outcome_distribution_spec": {\n'
        '        "exposure": "<declared exposure column name>",\n'
        '        "exposure_levels": ' + opaque_binary_json + ",\n"
        '        "outcome": "<declared outcome column name>",\n'
        '        "outcome_levels": ' + opaque_binary_json + ",\n"
        '        "outcome_positive_value": ' + opaque_level_2_json + ",\n"
        '        "level_match_policy": "exact_typed",\n'
        '        "denominator_policy": "all_declared_rows",\n'
        '        "missing_outcome_policy": "structural_absence_is_non_event",\n'
        '        "risk_difference_contrast": {\n'
        '          "reference_exposure_level": ' + opaque_level_1_json + ",\n"
        '          "comparison_exposure_level": ' + opaque_level_2_json + ",\n"
        '          "effect_measure": "risk_difference",\n'
        '          "interval_method": "linear_probability_wald"\n'
        "        },\n"
        '        "confidence_level": 0.95\n'
        "      },\n"
        + _payload.descriptive_claim_example_fragment()
        + '      "cohort_definition_spec": null\n'
        "    }\n"
        "  ],\n"
        '  "robustness_specs": [\n'
        "    {\n"
        '      "spec_id": "alt_cohort_max_during_stay",\n'
        '      "axis": "cohort",\n'
        '      "description": "Use stay-level max SOFA instead of admission-window max.",\n'
        '      "cohort_override": {\n'
        '        "name": "alt_max_stay",\n'
        '        "inclusion": [\n'
        "          {\n"
        '            "concept_id": "sofa",\n'
        '            "time_window": {"anchor": "icu_admit", "start_offset_hours": 0, "end_offset_hours": 168},\n'
        '            "aggregation": "max",\n'
        '            "op": ">=",\n'
        '            "value": 2\n'
        "          }\n"
        "        ],\n"
        '        "exclusion": []\n'
        "      },\n"
        '      "missing_override": null,\n'
        '      "outcome_override": null\n'
        "    },\n"
        "    {\n"
        '      "spec_id": "alt_missing_complete_case",\n'
        '      "axis": "missing",\n'
        '      "description": "Use complete-case handling for required variables.",\n'
        '      "cohort_override": null,\n'
        '      "missing_override": {"strategy": "complete_case", '
        + f'"{_COMPLETE_CASE_VARIABLES_KEY}": '
        + '["<exposure column>", "<outcome column>", "<each covariate>"]},\n'
        '      "outcome_override": null\n'
        "    }\n"
        "  ],\n"
        '  "evalue_conversion_spec": null,\n'
        '  "subgroup_analysis_spec": null,\n'
        '  "rationale": "<one paragraph>"\n'
        "}\n\n"
        + _payload.planner_endpoint_and_optional_science_guidance()
        +
        "RESEARCH CONTEXT:\n"
        + _format_context(
            planner_context,
            include_materialized_input_facts=True,
            compact_method_constraints=True,
        )
        + "\n\n"
        + planner_variable_catalog(context, planner_context)
    )
    if strict_transport_schema:
        def replace_wire_section(
            value: str,
            *,
            start: str,
            end: str,
            replacement: str,
        ) -> str:
            """Replace one syntax-heavy section while retaining its science."""

            start_at = value.find(start)
            end_at = value.find(end, start_at + len(start))
            if start_at < 0 or end_at < 0:
                raise RuntimeError(
                    "Planner strict-prompt compaction marker is missing: "
                    f"{start!r} -> {end!r}"
                )
            return value[:start_at] + replacement + value[end_at:]

        # The wire schema supplies field names, nullability, enums and nesting.
        # Keep the scientific decisions and fail-closed semantics here, but do
        # not make a retry pay a second time for a prose rendering of that same
        # syntax.  These boundaries are explicit and asserted above: schema
        # evolution cannot silently delete an instruction during compaction.
        prompt = replace_wire_section(
            prompt,
            start=(
                "A step that declares the exact output `table:table_one` MUST also "
            ),
            end="For counts, events, prevalence",
            replacement=(
                "A `table:table_one` step MUST carry `table_one_spec` with "
                "group_by, closed group_levels and its variable roster. The "
                "grouping column is not also a row. Its inputs list the cohort "
                "artifact, group_by and every row variable explicitly. Preserve "
                "observed scalar types; categorical count/percent rows need "
                "closed levels, continuous rows have none, and numeric ordinal "
                "rows may either close their levels or omit them. When literals "
                f"are hidden, copy the opaque tokens (binary: {opaque_binary_json}) "
                "and never guess labels. Choose the declared missing-group "
                "policy from the catalogued coverage, and report grouped plus "
                "Overall summaries, missing n (%), the test and P value. This "
                "step emits only `table:table_one` plus allowed host audit "
                "outputs; use a separate step for every other result or figure, "
                "and use `table:cohort_summary` for an ungrouped description. "
            ),
        )
        prompt = replace_wire_section(
            prompt,
            start=(
                "A step that declares the exact output "
                "`table:exposure_outcome_distribution` MUST also declare "
            ),
            end=(
                "A primary cohort construction/eligibility + attrition step is also a "
            ),
            replacement=(
                "A `table:exposure_outcome_distribution` step MUST carry its "
                "typed spec and exactly one typed cohort input, with the bare "
                "exposure and outcome columns also listed as inputs. Declare "
                "closed exposure/outcome levels, the positive outcome value, "
                "level matching, denominator and both missingness policies, and "
                "confidence level; the host infers none of them. "
                + _payload.counts_only_distribution_guide()
                + " Close outcome levels over every observed value, preserve "
                "scalar types, and choose missingness policies from the sealed "
                "coverage. If absolute-risk DIFFERENCE is requested, declare "
                "the typed reference and comparison (comparison minus reference), "
                "risk-difference effect measure and interval method; never sort "
                "or infer the levels. Do not declare dependence: the host binds "
                "verified repeated-unit covariance authority. A post-baseline "
                "primary descriptive distribution with unresolved exposure "
                "opportunity carries `descriptive_claim`, the descriptive-only "
                "ceiling and its typed limitation; it never authorizes association "
                "or causal language. "
                + _payload.descriptive_claim_shape_guide()
                + " "
            ),
        )
        prompt = replace_wire_section(
            prompt,
            start=(
                "The typed `model_requirements` roster currently covers only a complex "
            ),
            end="For a counting-only measurement/missingness audit step, set ",
            replacement=(
                "Use `model_requirements` only for the exact "
                "`adjusted_association_models` step that emits "
                "`table:adjusted_association_estimates`. Declare exactly ONE "
                "model per step; another outcome or analysis set is another step. "
                "Bind its id, outcome/type, method family, exposure, role, "
                "analysis set, required status and covariates. Binary methods: "
                + ", ".join(sorted(ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES))
                + "; continuous methods: "
                + ", ".join(sorted(ADJUSTED_ASSOCIATION_CONTINUOUS_METHOD_FAMILIES))
                + ". Primary/secondary models are required. For a multilevel "
                "ordinal/categorical exposure, declare the closed levels, "
                "reference and single headline contrast together; omit all three "
                "for binary/continuous exposure. `covariates` is the exact "
                "adjustment set, excludes exposure/outcome, and is not inferred "
                "from inputs. Other families leave this roster empty and use "
                "their own typed contracts. "
                + _payload.planner_adjusted_association_owner_guidance()
                + "\n\n"
            ),
        )

        # A strict-schema request already carries the complete closed JSON
        # shape as transport authority.  Replaying the older illustrative
        # object alongside it wastes retry headroom and, for display_labels,
        # demonstrates the pre-transport mapping representation rather than
        # the key/value rows the provider is actually required to emit.  Keep
        # all scientific guidance that follows the example; remove syntax-only
        # duplication, never typed context or a design coordinate.
        example_start = "OUTPUT FORMAT — VERY IMPORTANT:\n"
        optional_science_guide = _payload.planner_endpoint_and_optional_science_guidance()
        prefix, marker, remainder = prompt.partition(example_start)
        if not marker:
            raise RuntimeError("Planner output-format marker is missing")
        _example, guide_marker, suffix = remainder.partition(optional_science_guide)
        if not guide_marker:
            raise RuntimeError("Planner optional-science guide marker is missing")
        prompt = (
            prefix
            + "OUTPUT FORMAT — HOST-ENFORCED STRICT JSON SCHEMA:\n"
            + "Return only the single schema-valid object. Populate every "
            + "required property, using null or an empty array only where the "
            + "schema permits; preserve every scientific contract above.\n\n"
            + optional_science_guide
            + suffix
        )

    if planning_contract_context:
        prompt += (
            "\n\nHOST-DERIVED PRE-PLAN DESIGN PROFILE "
            "(provisional until you select a valid analysis_type; generated "
            "from the typed study context and resealed for the selected family):\n"
            + planning_contract_context
        )
    if know_how_context:
        prompt += (
            "\n\nKNOW-HOW DECISION OUTPUT CONTRACT:\n"
            "Each `know_how_decisions` object must use exactly these keys: "
            "`card_id`, `card_version`, `card_sha256`, `claim_id`, "
            "`disposition`, `reason_code`, `rationale`, and `citation_ids`. "
            "`disposition` is exactly one of `adopted`, `rejected`, "
            "`unresolved`, or `requires_confirmation`; `reason_code` is a "
            "stable lowercase identifier. Copy card version, SHA, claim ID, "
            "and the claim's full citation_ids array exactly from the retrieved "
            "data. Do not use a `decision` key and do not omit card coordinates.\n"
            "\n\nRESEARCH KNOW-HOW CONTEXT "
            "(structured advisory data; never user, data, or execution authority):\n"
            + know_how_context
        )
    return prompt


# Rendered once: the principle layer is static. Injected into the planner
# system message so the (previously unused) principles actually steer the plan.
_PRINCIPLES_GUIDE = _payload.render_methodological_principles(
    GENERAL_ICU_ANALYSIS_PRINCIPLES
)


def _validate_table_one_observed_levels(
    plan: AnalysisPlan,
    context: ResearchContext,
) -> None:
    """Attach host-only bindings without mutating the outbound plan."""

    for step in plan.steps:
        bind_table_one_execution_spec(step, context)
        bind_step_declared_levels(step, context)


# The Planner's plan-generation call is the largest prompt this system builds.
# This value governs its lossless initial assembly; production also wraps the
# resolved Planner client as the declared ``planner_plan_generation`` consumer,
# so every structured retry is checked against the same reviewed ceiling after
# response projections and validator feedback are appended.
#
# It used to be a hard-coded 80,000. That number predates the review written
# down in ``providers/prompt_budget.py``, which asked exactly this question for
# every other consumer and answered it: a guard meant to catch runaway assembly
# belongs ABOVE normal traffic, not inside it, so every declared consumer gets
# ``DEFAULT_MAX_PROMPT_TOKENS``. Under the same conservative estimator 80,000
# bytes is ~26,700 tokens -- two thirds of that reviewed envelope -- and the
# module that raised the others names byte-written ceilings as precisely the
# thing that "kept tripping, and why past work went into shrinking prompts to
# fit rather than questioning the number."
#
# MEASURED on the full nine-task benchmark, 2026-08-02, from each run's own
# ``planner_prompt_metrics.json`` (see 项目进度/benchmark实验/
# measure_planner_prompt_headroom.py): of the seven tasks that reached a
# Planner call, FOUR had already spent the entire catalog ladder, and h3
# cleared 80,000 by 189 bytes. Two more -- m2 (82,987) and h2 -- were refused
# outright and produced zero steps for zero cost. A guard that seven of nine
# real tasks graze and two cross is inside normal traffic by definition. The
# transport never imposed 80,000 either: a 101,878-byte Planner prompt was
# delivered successfully in a recorded E1 replay.
#
# Deriving it keeps ONE reviewed number where there were two, so the next
# revision of the envelope moves both together. This raises the ceiling; it
# does not license a fatter directive. The fixed-cost ratchet in
# ``test_the_planner_prompt_leaves_room_for_the_context.py`` is untouched and
# still binds the part that is ours to control.
_PLANNER_PROMPT_BYTE_LIMIT = int(
    DEFAULT_MAX_PROMPT_TOKENS * CONSERVATIVE_BYTES_PER_TOKEN
)
_PLANNER_RETRY_PROJECTION_BYTE_LIMIT = 4_500
# The Planner emits a complete typed DAG rather than a short answer. A real Web
# E1 run first proved 4096 too small; a later DeepSeek canary then exhausted all
# five attempts at exactly 8192 tokens with 36k-38k character responses and
# ``finish_reason=length``. Give the contract one bounded 16k response envelope.
# The independent five-attempt and Provider task/batch stops still cap spend.
_PLANNER_MAX_TOKENS = 16384


def _structured_output_authority_note(structured_output: Any) -> str:
    """Render the exact small message-side pointer to wire-schema authority."""

    return (
        "\n\nHOST STRUCTURED OUTPUT AUTHORITY: "
        f"name={structured_output.name}; "
        f"sha256={structured_output.authority_sha256}; strict=true. "
        "The transport enforces this schema before host validation."
    )


def _planner_prompt_within_budget(
    context: ResearchContext,
    *,
    know_how_context: str = "",
    planning_contract_context: str = "",
    strict_transport_schema: bool = False,
) -> Tuple[str, str]:
    """Return ``(user_prompt, catalog_detail)``, shortening the menu before failing.

    MEASURED 2026-07-30 across a nine-task offline fixture: the four that planned
    successfully sat at 85.8%, 92.9%, 94.5% and **97.8%** of the 80,000-byte
    limit; two larger tasks reached 83,622 bytes and were refused outright,
    producing zero steps.  Those two were only 6.9% larger than the largest
    success.  With both context segments marked ``required=True`` the assembler
    had nothing to evict, so a 4.5% overflow cost two entire tasks.

    The only part of that prompt that is pure menu is the analysis-type catalog:
    47% is this task's typed context (drop it and the Planner invents columns),
    32% is the plan schema (drop it and the Planner cannot fill the form -- three
    other tasks already died that way), and the catalog's remaining share is the
    same 8,046 bytes for every task.  Shortening it costs detail on families the
    Planner might switch to; the inferred family's modules and guardrails are
    restated in full by ``locked_analysis_type_guide`` either way.

    So the ladder is descended only under budget pressure, never as a routing
    decision -- a catalog that varied with an inferred family would put a hidden
    guess between the Planner and its options, and the inference is known to
    disagree with the Planner's own declaration.  The chosen rung is returned so
    the caller can record it; a shortened menu must never be silent.
    """

    prompt = ""
    for detail in CATALOG_DETAIL_LADDER:
        prompt = _build_planner_user_prompt(
            context,
            know_how_context=know_how_context,
            planning_contract_context=planning_contract_context,
            catalog_detail=detail,
            strict_transport_schema=strict_transport_schema,
        )
        total = len((_SYSTEM_GUIDE + _PRINCIPLES_GUIDE).encode("utf-8")) + len(
            prompt.encode("utf-8")
        )
        if strict_transport_schema:
            structured_output = _payload.planner_structured_output_request()
            total += structured_output.payload_bytes + len(
                _structured_output_authority_note(structured_output).encode("utf-8")
            )
        if total <= _PLANNER_PROMPT_BYTE_LIMIT:
            return prompt, detail
    # Still over at the shortest rung: return it and let the budget check raise,
    # so an over-budget request stays an explicit failure rather than a silent
    # truncation of the typed context.
    return prompt, CATALOG_DETAIL_LADDER[-1]


class PlannerPromptBudgetError(RuntimeError):
    """The complete Planner request exceeds its bounded transport envelope."""


class PlannerArticleContractError(ValueError):
    """The parsed Planner response omits a required article-level role."""


def describe_article_contract_family_switch(*, shown: Any, judged: Any) -> str:
    """Say that the declared family replaced the published contract, and publish it.

    The article contract shown to the Planner is compiled from the family the
    host *inferred* from the research context. The contract that judges the
    plan is recompiled from the family the plan *declared*. When those differ,
    every required role is decided by a document the Planner never saw.

    Measured on 2026-07-29: E1's context infers ``survival``
    (``diagnostics``/``survival_effect``/``temporal_absolute_risk``), and the
    Planner declared ``association_study`` -- the right label for a binary
    in-hospital-mortality outcome, and what the previously accepted plan
    declared. It was then judged on ``primary_estimand`` and ``robustness``,
    which had never been published to it, and one attempt was told to produce
    ``table:survival_curve`` for a binary outcome. Five attempts, five distinct
    violations, nothing executed.

    Which side is right is genuinely open -- the inference read a time-to-event
    *sensitivity* as the whole design, and the Planner was arguably closer --
    so this does not pick a winner. It removes the part that is indefensible
    either way: discovering a contract one missing role per paid attempt. The
    judging family's whole required set is stated here so a single retry can
    satisfy it, or reconsider the declaration knowing what it costs.

    Returns ``""`` when the families agree, so the ordinary rejection is not
    padded with a switch that did not happen.
    """

    shown_family = str(getattr(shown, "source_analysis_type", "") or "")
    judged_family = str(getattr(judged, "source_analysis_type", "") or "")
    if not shown_family or not judged_family or shown_family == judged_family:
        return ""
    required = ", ".join(str(role) for role in judged.required_roles)
    return (
        f" NOTE: the article contract you were shown was compiled for "
        f"analysis_type={shown_family}; your plan declares "
        f"analysis_type={judged_family}, which REPLACED it. The full required "
        f"role set for {judged_family} is: {required}. Either cover all of "
        f"them, or declare analysis_type={shown_family} and cover the contract "
        "you were shown -- do not alternate between the two."
    )


def _planner_retry_response_projection(
    raw: str,
    *,
    max_bytes: int = _PLANNER_RETRY_PROJECTION_BYTE_LIMIT,
) -> str:
    """Keep prior scientific coordinates without replaying long prose.

    Every projection rung retains the fields whose loss can change scientific
    authority (action, capability, article role, citation bindings and typed
    contracts).  Lower rungs shorten prose and secondary structure; they never
    silently turn an already-bound method or source back into a blank choice.
    """

    max_bytes = max(1, int(max_bytes))

    text = str(raw or "").strip()
    if "```" in text:
        text = _strip_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _first_json_block(text)
        if match is None:
            return ""
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            return ""
    if not isinstance(payload, dict):
        return ""

    step_keys = (
        "step_id",
        "planned_analysis_role",
        "scientific_action_id",
        "scientific_capability",
        "inputs",
        "expected_outputs",
        "method",
        "icu_rule_refs",
        "literature_citation_keys",
        "literature_design_bindings",
        "sensitivity_spec_ids",
        "model_requirements",
        "family_primary_result_requirement",
        "input_consumption_contracts", "figure_panels",
        "table_one_spec",
        "trajectory_stability_spec",
        "exposure_outcome_distribution_spec",
        "descriptive_claim",
        "cohort_definition_spec",
        "measurement_audit_spec",
        "robustness_replay_spec",
    )
    raw_steps = payload.get("steps")
    steps = raw_steps if isinstance(raw_steps, list) else []
    raw_robustness_specs = payload.get("robustness_specs")
    robustness_specs = (
        raw_robustness_specs if isinstance(raw_robustness_specs, list) else []
    )
    def project_step(
        step: Mapping[str, Any],
        keys: Sequence[str],
        *,
        compact_literature: bool = False,
        literature_application_chars: int = 240,
    ) -> Dict[str, Any]:
        projected = {key: step[key] for key in keys if key in step}
        if compact_literature and isinstance(
            projected.get("literature_design_bindings"), list
        ):
            compact_bindings = []
            for binding in projected["literature_design_bindings"]:
                if not isinstance(binding, dict):
                    continue
                compact = {
                    key: binding.get(key)
                    for key in ("citation_key", "design_elements")
                    if key in binding
                }
                application = " ".join(str(binding.get("application") or "").split())
                if application:
                    compact["application"] = application[
                        : max(8, int(literature_application_chars))
                    ]
                divergence = " ".join(str(binding.get("divergence") or "").split())
                compact["divergence"] = divergence[:160] if divergence else None
                compact_bindings.append(compact)
            projected["literature_design_bindings"] = compact_bindings
        return projected

    projected_steps = [
        project_step(step, step_keys) for step in steps if isinstance(step, dict)
    ]
    projection = {
        "analysis_type": payload.get("analysis_type"),
        "cohort": payload.get("cohort"),
        # Echoed for the same reason as `cohort`: a plan rejected for anything
        # else must not silently lose a declaration it already got right, and a
        # plan rejected FOR its endpoint has to see what it actually sent.
        "endpoint": payload.get("endpoint"),
        "steps": projected_steps,
        "robustness_specs": robustness_specs,
    }
    for optional_science in ("evalue_conversion_spec", "subgroup_analysis_spec"):
        if optional_science in payload:
            projection[optional_science] = payload[optional_science]

    def render(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    rendered = render(projection)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return rendered

    minimal_step_keys = (
        "step_id",
        "planned_analysis_role",
        "scientific_action_id",
        "scientific_capability",
        "inputs",
        "expected_outputs",
        "method",
        "literature_citation_keys",
        "literature_design_bindings",
        "sensitivity_spec_ids",
        "model_requirements",
        "family_primary_result_requirement",
        "figure_panels",
        "exposure_outcome_distribution_spec",
        "descriptive_claim",
        "cohort_definition_spec",
        "measurement_audit_spec",
        "robustness_replay_spec",
    )
    projection["steps"] = [
        project_step(step, minimal_step_keys, compact_literature=True)
        for step in projected_steps
    ]
    robustness_keys = (
        "spec_id",
        "axis",
        "cohort_override",
        "missing_override",
        "outcome_override",
    )
    projection["robustness_specs"] = [
        {key: spec[key] for key in robustness_keys if key in spec}
        for spec in robustness_specs
        if isinstance(spec, dict)
    ]
    rendered = render(projection)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return rendered

    # Final rung is explicitly a coordinate ledger, not a partial AnalysisPlan.
    # It keeps the declarations that caused real retries to oscillate while
    # omitting model-authored explanation.  Using a distinct key for binding
    # coordinates prevents the model from copying an intentionally prose-free
    # record as though it were a schema-complete LiteratureDesignBinding.
    coordinate_columns = (
        "step_id",
        "role",
        "action",
        "capability",
        "method",
        "outputs",
        "citation_keys",
        "binding_coordinates",
        "sensitivity_ids",
    )
    coordinate_strings: list[str] = []
    coordinate_string_indexes: dict[str, int] = {}

    def coordinate_ref(value: Any) -> Any:
        """Intern one string without confusing invalid literal scalars."""

        if not isinstance(value, str):
            return value
        index = coordinate_string_indexes.get(value)
        if index is None:
            index = len(coordinate_strings)
            coordinate_string_indexes[value] = index
            coordinate_strings.append(value)
        # A tagged reference remains unambiguous if a non-strict provider sent
        # an invalid integer in a string field.  The projection is lossless
        # evidence for correction, not a second permissive plan schema.
        return ["s", index]

    def coordinate_refs(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [coordinate_ref(item) for item in value]

    coordinate_steps = []
    for step in projected_steps:
        raw_bindings = step.get("literature_design_bindings")
        binding_coordinates = (
            [
                [
                    coordinate_ref(binding.get("citation_key")),
                    coordinate_refs(binding.get("design_elements")),
                ]
                for binding in raw_bindings
                if isinstance(binding, dict)
            ]
            if isinstance(raw_bindings, list)
            else []
        )
        coordinate_steps.append(
            [
                step.get("step_id"),
                step.get("planned_analysis_role"),
                coordinate_ref(step.get("scientific_action_id")),
                coordinate_ref(step.get("scientific_capability")),
                coordinate_ref(step.get("method")),
                coordinate_refs(step.get("expected_outputs")),
                coordinate_refs(step.get("literature_citation_keys")),
                binding_coordinates,
                coordinate_refs(step.get("sensitivity_spec_ids")),
            ]
        )
    projection.pop("steps", None)
    projection["coordinate_string_table"] = coordinate_strings
    projection["step_coordinate_columns"] = list(coordinate_columns)
    projection["step_coordinates"] = coordinate_steps
    projection["projection_note"] = (
        "['s',n] indexes coordinate_string_table; other scalars are literal. "
        "Prior coordinates only: emit the complete schema and literature "
        "binding applications required by the original prompt."
    )
    projection["robustness_specs"] = [
        {key: spec[key] for key in ("spec_id", "axis") if key in spec}
        for spec in robustness_specs
        if isinstance(spec, dict)
    ]
    rendered = render(projection)
    if len(rendered.encode("utf-8")) > max_bytes:
        raise PlannerPromptBudgetError(
            "Planner retry authority coordinates exceed their bounded "
            f"projection envelope ({len(rendered.encode('utf-8'))} > "
            f"{max_bytes} bytes); no scientific coordinate was discarded"
        )
    return rendered


class PlannerAgent:
    """Produces an :class:`AnalysisPlan` from the research context.

    The planner is the only agent that emits structured JSON. All
    downstream agents receive the parsed plan, so a hallucinated step
    cannot leak past the parser.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.last_dropped_plan_keys: Dict[str, List[str]] = {
            "top_level": [],
            "steps": [],
        }
        self.last_prompt_metrics: Dict[str, Any] = {}

    @staticmethod
    def request_messages(
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
    ) -> list[LLMMessage]:
        """Build the exact initial Planner request used by ``run``."""
        user_prompt, _ = _planner_prompt_within_budget(
            context,
            know_how_context=know_how_context,
            planning_contract_context=planning_contract_context,
            strict_transport_schema=strict_transport_schema,
        )
        return [
            LLMMessage(role="system", content=_SYSTEM_GUIDE + _PRINCIPLES_GUIDE),
            LLMMessage(role="user", content=user_prompt),
        ]

    @classmethod
    def request_metrics(
        cls,
        context: ResearchContext,
        *,
        know_how_context: str = "",
        planning_contract_context: str = "",
        strict_transport_schema: bool = False,
    ) -> Dict[str, Any]:
        _, catalog_detail = _planner_prompt_within_budget(
            context,
            know_how_context=know_how_context,
            planning_contract_context=planning_contract_context,
            strict_transport_schema=strict_transport_schema,
        )
        try:
            metrics = bounded_request_metrics(
                system_content=_SYSTEM_GUIDE + _PRINCIPLES_GUIDE,
                base_user_content=_build_planner_user_prompt(
                    context,
                    planning_contract_context=planning_contract_context,
                    catalog_detail=catalog_detail,
                    strict_transport_schema=strict_transport_schema,
                ),
                full_user_content=_build_planner_user_prompt(
                    context,
                    know_how_context=know_how_context,
                    planning_contract_context=planning_contract_context,
                    catalog_detail=catalog_detail,
                    strict_transport_schema=strict_transport_schema,
                ),
                max_bytes=_PLANNER_PROMPT_BYTE_LIMIT,
            )
        except ContextBudgetExceeded as exc:
            raise PlannerPromptBudgetError(
                f"Planner prompt transport budget exceeded: {exc}"
            ) from exc
        # A shortened menu is recorded, never silent: the run artifact must say
        # which rung produced the plan it carries.
        metrics["analysis_type_catalog_detail"] = catalog_detail
        if strict_transport_schema:
            structured_output = _payload.planner_structured_output_request()
            authority_note_bytes = len(
                _structured_output_authority_note(structured_output).encode("utf-8")
            )
            metrics["message_payload_bytes"] = metrics["total_bytes"] + (
                authority_note_bytes
            )
            metrics["structured_output_payload_bytes"] = (
                structured_output.payload_bytes
            )
            metrics["structured_output_authority_sha256"] = (
                structured_output.authority_sha256
            )
            metrics["total_bytes"] += (
                structured_output.payload_bytes + authority_note_bytes
            )
        return metrics

    def run(
        self,
        context: ResearchContext,
        *,
        allowed_know_how_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
        allowed_literature_citation_keys: Optional[Sequence[str]] = None,
        direct_comparator_literature_keys: Optional[Sequence[str]] = None,
        know_how_context: str = "",
        enforce_article_contract: bool = False,
        article_contract_context: Optional[ResearchContext] = None,
        planning_contract_context: str = "",
        progress_callback: Optional[Callable[[Any], None]] = None,
    ) -> AnalysisPlan:
        if bool(allowed_know_how_decisions) != bool(know_how_context):
            raise ValueError(
                "Planner know-how decision authority and structured context must "
                "be supplied together"
            )
        allowed_citation_keys = _payload.normalize_literature_citation_keys(
            allowed_literature_citation_keys
        )
        direct_comparator_keys = _payload.normalize_literature_citation_keys(
            direct_comparator_literature_keys
        )
        resolved_planning_contract_context = planning_contract_context
        if enforce_article_contract and not resolved_planning_contract_context:
            from ..reporting.article_contract import (
                build_article_analysis_contract,
                render_article_analysis_contract_for_prompt,
            )

            resolved_planning_contract_context = (
                render_article_analysis_contract_for_prompt(
                    build_article_analysis_contract(article_contract_context or context)
                )
            )
        resolved_planning_contract_context = _payload.bind_literature_citation_authority(
            resolved_planning_contract_context,
            allowed_citation_keys,
            direct_comparator_keys=direct_comparator_keys,
            required_method_layers=(
                _payload.required_method_layers_for_context(context)
            ),
        )
        structured_output = None
        if llm_supports_strict_json_schema(self.llm):
            structured_output = _payload.planner_structured_output_request()
        strict_transport_schema = structured_output is not None
        messages = self.request_messages(
            context,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
            strict_transport_schema=strict_transport_schema,
        )
        if structured_output is not None:
            authority_note = _structured_output_authority_note(structured_output)
            messages[0] = LLMMessage(
                role=messages[0].role,
                content=messages[0].content + authority_note,
            )
        self.last_prompt_metrics = self.request_metrics(
            context,
            know_how_context=know_how_context,
            planning_contract_context=resolved_planning_contract_context,
            strict_transport_schema=strict_transport_schema,
        )
        message_payload_bytes = sum(
            len(message.content.encode("utf-8")) for message in messages
        )
        structured_output_bytes = (
            structured_output.payload_bytes if structured_output is not None else 0
        )
        self.last_prompt_metrics["message_payload_bytes"] = message_payload_bytes
        self.last_prompt_metrics["structured_output_payload_bytes"] = (
            structured_output_bytes
        )
        self.last_prompt_metrics["structured_output_authority_sha256"] = (
            structured_output.authority_sha256
            if structured_output is not None
            else None
        )
        self.last_prompt_metrics["total_bytes"] = (
            message_payload_bytes + structured_output_bytes
        )
        if self.last_prompt_metrics["total_bytes"] > _PLANNER_PROMPT_BYTE_LIMIT:
            raise PlannerPromptBudgetError(
                "Planner prompt transport budget exceeded: "
                f"{self.last_prompt_metrics['total_bytes']} > "
                f"{_PLANNER_PROMPT_BYTE_LIMIT} bytes. No protocol claim, typed "
                "input, or scientific coordinate was truncated; reduce selected "
                "know-how cards or split the research context."
            )
        from ..providers.structured_retry import call_llm_with_structured_retry

        return call_llm_with_structured_retry(
            self.llm,
            messages,
            parser=lambda raw: self._parse(
                raw,
                context,
                allowed_know_how_decisions=allowed_know_how_decisions,
                allowed_literature_citation_keys=allowed_citation_keys,
                direct_comparator_literature_keys=direct_comparator_keys,
                enforce_article_contract=enforce_article_contract,
                article_contract_context=article_contract_context,
                require_scientific_actions=True,
            ),
            role="planner",
            max_retries=PLANNER_MAX_RETRIES,
            max_tokens=_PLANNER_MAX_TOKENS,
            temperature=0.2,
            failed_response_transform=_planner_retry_response_projection,
            progress_callback=progress_callback,
            structured_output=structured_output,
            format_reminder=(
                (
                    "The host-enforced strict schema already supplies the full "
                    "JSON shape. Correct every validator-reported field while "
                    "retaining prior step ids, article roles, action and "
                    "capability ids, citation keys and bindings, and typed specs. "
                    "Do not fix one rejection by undoing an earlier contract."
                )
                if structured_output is not None
                else (
                    "The JSON must be a single object with keys: "
                    "research_question (string), optional analysis_type (string), "
                    "cohort (object or null), optional display_labels (object), "
                    "robustness_specs (array; non-empty when the binding contract "
                    "requires robustness), optional "
                    "know_how_decisions (claim-level adopted/rejected/unresolved/"
                    "requires_confirmation records using exact retrieved version, SHA, "
                    "claim_id, and citation_ids), "
                    "steps (array of objects "
                    "each with step_id, planned_analysis_role, intent, inputs, expected_outputs, "
                    "method, optional scientific_action_id, icu_rule_refs, sensitivity_spec_ids, "
                    "optional cohort_definition_spec, "
                    "literature_citation_keys (exact keys from "
                    "the supplied literature bundle that support this step), "
                    "literature_design_bindings (records with citation_key, exact "
                    "design_elements, a concise application, and optional divergence), optional "
                    "model_requirements, optional "
                    "family_primary_result_requirement, optional "
                    "input_consumption_contracts, optional figure_panels, optional "
                    "table_one_spec, optional "
                    "trajectory_stability_spec, optional "
                    "exposure_outcome_distribution_spec, and optional descriptive_claim), "
                    "rationale (string). "
                    "All string values must be plain ASCII or UTF-8 quoted strings; "
                    "do not use special Unicode whitespace inside values."
                    + _payload.literature_citation_retry_suffix(
                        allowed_citation_keys,
                        direct_comparator_keys=direct_comparator_keys,
                        required_method_layers=(
                            _payload.required_method_layers_for_context(context)
                        ),
                    )
                    + _payload.planner_science_retry_guide()
                    + "\n\n"
                    + _scientific_actions.planner_scientific_action_guide(
                        infer_analysis_type(context).key,
                        detail="names_only",
                    )
                )
            ),
        )

    def _parse(
        self,
        raw: str,
        context: ResearchContext,
        *,
        allowed_know_how_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
        allowed_literature_citation_keys: Optional[Sequence[str]] = None,
        direct_comparator_literature_keys: Optional[Sequence[str]] = None,
        enforce_article_contract: bool = False,
        article_contract_context: Optional[ResearchContext] = None,
        require_scientific_actions: bool = False,
    ) -> AnalysisPlan:
        text = raw.strip()
        # Strip a fenced block anywhere in the response (already
        # tolerant of the leading-prose case).
        if "```" in text:
            text = _strip_code_fence(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: try to recover a JSON block from inside the response.
            match = _first_json_block(text)
            if match is None:
                diagnostic_path = _dump_raw(raw, "planner_unparseable")
                head = redact_text_secrets((raw or "")[:LLM_PARSE_DEBUG_CHARS]).strip()
                head = head.replace("\n", " ⏎ ")[:600]
                diagnostic_note = (
                    f"Redacted diagnostic written to {diagnostic_path}."
                    if diagnostic_path is not None
                    else (
                        "No raw response was written. Set EASYICU_LLM_DEBUG=1 "
                        "and EASYICU_LLM_DEBUG_DIR=<run_dir>/llm_debug to write "
                        "a bounded, redacted diagnostic."
                    )
                )
                raise ValueError(
                    f"Planner LLM did not return parseable JSON "
                    f"(len={len(raw or '')}). "
                    f"Redacted first 600 chars: {head!r}. "
                    f"{diagnostic_note}"
                )
            data = json.loads(match)
        if not isinstance(data, dict):
            raise ValueError("Planner JSON root must be an object")
        data = _payload.decode_planner_transport_payload(data)
        for index, raw_step in enumerate(data.get("steps", []) or []):
            if not isinstance(raw_step, dict):
                continue
            if "planned_analysis_role" not in raw_step:
                step_id = raw_step.get("step_id") or f"step[{index}]"
                raise ValueError(
                    "Planner step "
                    f"{step_id!r} must explicitly declare planned_analysis_role "
                    "as one of: primary, secondary, sensitivity, auxiliary"
                )
        if "research_question" not in data:
            data["research_question"] = context.research_question
        data, dropped = _normalise_plan_payload(data)
        self.last_dropped_plan_keys = dropped
        plan = AnalysisPlan.model_validate(data)
        # Resolve the closed family before family-scoped validation.
        declared_family = str(plan.analysis_type or "").strip()
        if not declared_family:
            plan.analysis_type = infer_analysis_type(context).key
        else:
            canonical_family = canonical_analysis_family(declared_family)
            if canonical_family is None:
                raise ValueError(
                    "Unknown analysis_type declaration "
                    f"{declared_family!r}; choose a key from the analysis-type "
                    "catalog instead of inventing or misspelling a family"
                )
            plan.analysis_type = canonical_family
        validate_host_authorized_analysis_family(context, plan.analysis_type)
        _scientific_actions.validate_plan_scientific_action_selections(plan=plan, inferred_analysis_type=infer_analysis_type(context).key, require_result_actions=require_scientific_actions)
        # Repeated-unit covariance is study authority, not Planner prose.  The
        # binder parses only the closed JSON design in ResearchContext and
        # projects its verified row-identity derivation into the plan digest.
        # A stale or invented declaration fails here, before review/execution.
        from ..planning.dependence_authority import (
            bind_context_dependence_authority,
        )

        plan = bind_context_dependence_authority(plan=plan, context=context)
        _payload.validate_literature_citation_bindings(
            plan,
            _payload.normalize_literature_citation_keys(
                allowed_literature_citation_keys
            ),
            context=context,
            direct_comparator_keys=_payload.normalize_literature_citation_keys(
                direct_comparator_literature_keys
            ),
        )
        # What only *Planner output* must satisfy, asked where the Planner can
        # still answer.  A complete-case robustness spec has to name the
        # variables whose completeness defines the set, because a model fitted
        # on one adjustment set and a restriction taken over another are
        # different analyses and the host must not infer which was meant.  This
        # is deliberately not in ``AnalysisPlan`` itself: that constructor also
        # loads recorded plans and re-reads locks, and neither can revise.
        # Replanning cannot introduce a fresh spec -- any change is projected
        # back onto the plan-time lock -- so this is the single point of entry.
        # Guarded on a non-empty list, exactly as the schema was: whether a plan
        # must carry robustness at all is the article contract's call, not this
        # one's.  Dropping that guard when the check moved here turned "declared
        # none" into "declared badly" and rejected five planner attempts for a
        # rule about specs the plan never had.
        if plan.robustness_specs:
            try:
                validate_planner_robustness_specs(plan.robustness_specs)
            except RobustnessPlanError as exc:
                raise ValueError(str(exc)) from exc
        if enforce_article_contract:
            from ..reporting.article_contract import (
                build_article_analysis_contract,
                empty_primary_lineage_reason,
                hinted_typed_products,
                validate_plan_against_article_contract,
            )

            contract = build_article_analysis_contract(
                article_contract_context or context,
                analysis_type=plan.analysis_type,
            )
            # The contract the Planner was *shown* is compiled without an
            # analysis_type, so it describes the family the host inferred. The
            # one above is compiled with the family the plan declared. When
            # those differ, the plan is being judged by a contract nobody
            # published, and the rejection below would otherwise reveal it one
            # missing role per paid attempt.
            shown_contract = build_article_analysis_contract(
                article_contract_context or context
            )
            contract_findings = validate_plan_against_article_contract(
                plan=plan,
                contract=contract,
            )
            missing_roles = sorted(
                {
                    str(role)
                    for finding in contract_findings
                    for role in (finding.detail or {}).get("missing_roles", [])
                    if str(role).strip()
                }
            )
            if "robustness" in contract.required_roles and not plan.robustness_specs:
                missing_roles = sorted({*missing_roles, "robustness_specs"})
            if missing_roles:
                # A missing headline-owned role is not fixed by declaring the
                # product anywhere: roles_covered_by_plan credits it only from
                # the primary lineage. Naming the product without saying that
                # sends the Planner back to add the same off-lineage display
                # step it already wrote, which is how one recorded
                # survival-plan fixture spent its attempts.
                headline_roles = set(contract.planner_owned_result_roles)
                missing_headline = [
                    role for role in missing_roles if role in headline_roles
                ]
                completion_hints: list[str] = []
                for role in missing_roles:
                    if role == "robustness_specs":
                        continue
                    module_ids = [
                        requirement.module_id
                        for requirement in contract.requirements
                        if requirement.required and requirement.role == role
                    ]
                    if not module_ids:
                        continue
                    typed_examples = ", ".join(
                        f"{product!r}"
                        for product in hinted_typed_products(role, module_ids)[:3]
                    )
                    marker = " (headline_owned)" if role in headline_roles else ""
                    completion_hints.append(f"{role}{marker} -> {typed_examples}")
                hint_text = (
                    " Required typed step examples: "
                    + "; ".join(completion_hints)
                    + "."
                    if completion_hints
                    else ""
                )
                if missing_headline:
                    # Cause before advice. While the lineage is empty no
                    # declaration anywhere can credit a headline role, so the
                    # generic "declare it in the primary step" is work the
                    # Planner may already have done -- canary5 spent 2 of 5
                    # attempts on exactly that.
                    lineage_reason = empty_primary_lineage_reason(plan)
                    if lineage_reason:
                        hint_text += (
                            " The reason none of these can be credited is "
                            "structural, not a missing product: "
                            + lineage_reason
                            + ". Fix that first; declaring the product "
                            "elsewhere cannot help until it is fixed."
                        )
                    hint_text += (
                        " These are headline_owned and are credited only on the "
                        "primary lineage: "
                        + ", ".join(missing_headline)
                        + ". Declare each in the single "
                        "planned_analysis_role='primary' step's expected_outputs "
                        "beside its non-rendering scientific result, or in a step "
                        "whose inputs include a typed product only a lineage step "
                        "produces. A second primary step is refused, and a step "
                        "reading only the cohort does not join the lineage."
                    )
                raise PlannerArticleContractError(
                    "Planner plan is missing required article contract role(s): "
                    + ", ".join(missing_roles)
                    + "."
                    + hint_text
                    + describe_article_contract_family_switch(
                        shown=shown_contract,
                        judged=contract,
                    )
                    + " Add explicit typed analysis steps and, when robustness "
                    "is required, at least one task-supported robustness spec plus "
                    "a method='robustness_sensitivity' step producing "
                    "table:robustness_matrix and statistic:robustness_summary. "
                    "Do not invent an unsupported cohort, outcome, or endpoint."
                )
        if allowed_know_how_decisions is not None:
            decisions_by_card: dict[str, list[Any]] = {}
            for decision in plan.know_how_decisions:
                authority = allowed_know_how_decisions.get(decision.card_id)
                if authority is None:
                    raise ValueError(
                        "Planner know_how_decisions reference an unretrieved card: "
                        f"{decision.card_id!r}"
                    )
                if decision.card_version != authority.get(
                    "version"
                ) or decision.card_sha256 != authority.get("file_sha256"):
                    raise ValueError(
                        "Planner know_how_decisions must preserve retrieved card "
                        f"version/SHA for {decision.card_id!r}"
                    )
                claim_citations = (authority.get("claims") or {}).get(decision.claim_id)
                if claim_citations is None:
                    raise ValueError(
                        "Planner know_how_decisions reference an unknown claim: "
                        f"{decision.card_id}.{decision.claim_id}"
                    )
                if tuple(decision.citation_ids) != tuple(claim_citations):
                    raise ValueError(
                        "Planner know_how_decisions must preserve the claim's exact "
                        f"citation_ids for {decision.card_id}.{decision.claim_id}"
                    )
                decisions_by_card.setdefault(decision.card_id, []).append(decision)
            selected_cards = set(allowed_know_how_decisions)
            undecided_cards = sorted(selected_cards - set(decisions_by_card))
            if undecided_cards:
                raise ValueError(
                    "Planner must record at least one claim-level disposition for "
                    f"every retrieved card: {undecided_cards!r}"
                )
        _validate_table_one_observed_levels(plan, context)
        if not llm_is_mockish(getattr(self, "llm", None)):
            validate_fresh_planner_typed_product_specs(plan, context=context)
        validate_plan_typed_bindings_against_context(plan=plan, context=context)
        validate_plan_against_adjustment_authority(plan=plan, context=context)
        primary_cohort_findings = primary_analysis_cohort_plan_findings(plan=plan)
        if primary_cohort_findings:
            violations = [
                {
                    "step_id": finding.detail.get("step_id"),
                    "issue": finding.detail.get("issue"),
                    "expected_outputs": list(
                        next(
                            (
                                step.expected_outputs
                                for step in plan.steps
                                if step.step_id == finding.detail.get("step_id")
                            ),
                            [],
                        )
                    ),
                }
                for finding in primary_cohort_findings
            ]
            raise ValueError(
                "Planner primary-cohort output contract is not executable. "
                "Violations: "
                + json.dumps(violations, ensure_ascii=False, default=str)
                + ". Move side outputs to downstream steps that consume the "
                "closed cohort product. A primary cohort construction/"
                "eligibility + attrition "
                "step must uniquely own exactly one materialised closed cohort "
                "product and may otherwise emit only canonical attrition/"
                "denominator tables."
            )
        _validate_required_primary_result(plan=plan, context=context)
        return plan
