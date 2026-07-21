"""Step-scoped prompt and metadata projection for the analysis coder.

The planner owns the scientific contract.  This module only reduces transport:
it selects guidance from exact method/output structure and projects the already
registered variable metadata needed by the current step.  It never chooses an
exposure, outcome, cohort, method, or estimand.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from ..planning.analysis_types import canonical_analysis_family
from ..contracts.ordered_stratified import is_ordered_stratified_analysis_step
from ..plan_utils import (
    clustering_contract_applies,
    cohort_change_contract_applies,
    effect_output_authorized,
    prediction_contract_applies,
)
from ..schema import (
    PLANNED_MODEL_REQUIREMENTS_STEP_METHOD,
    AnalysisStep,
    ResearchContext,
)
from .typed import project_research_context_variables
from ..trajectory.plan_contract import trajectory_step_roles

_COMPANION_SUFFIXES = (
    "_measured",
    "_first_time",
    "_last_time",
    "_first",
    "_max",
    "_min",
    "_mean",
    "_n",
)

_FIGURE_METHODS = frozenset(
    {
        "figure",
        "publication_figure",
        "visualization",
        "descriptive_visualization",
        "forest_plot",
        "kaplan_meier_plot",
    }
)
_TABLE_METHODS = frozenset(
    {
        "binary_outcome_incidence_and_absolute_risk",
        "cohort_description",
        "data_quality",
        "descriptive",
        "descriptive_statistics",
        "descriptive_summary",
        "incidence",
        "missingness",
        "missingness_audit",
        "table_one",
    }
)
_DESCRIPTIVE_TABLE_PRODUCTS = frozenset(
    {
        "cohort_summary",
        "missingness",
        "missingness_audit",
        "outcome_incidence",
        "source_status",
        "table_one",
    }
)
_ROBUSTNESS_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "prespecified_robustness_analysis",
        "robustness_analysis",
        "sensitivity_analysis",
    }
)
_QUALITY_CONTROL_METHODS = frozenset(
    {
        "data_quality",
        "exposure_distribution_and_missingness_audit",
        "exposure_quality_control",
        "longitudinal_missingness_and_score_quality_audit",
        "missingness_and_measurement_audit",
        "ordinal_exposure_quality_control",
        "ordinal_exposure_qc_and_missingness_audit",
        "ordered_exposure_derivation_and_qc",
        "ordered_category_exposure_qc",
        "ordinal_exposure_derivation_and_quality_control",
        "ordered_exposure_quality_control",
        "quality_control",
    }
)
_ORDERED_QUALITY_CONTROL_METHODS = frozenset(
    {
        "ordinal_exposure_quality_control",
        "ordinal_exposure_qc_and_missingness_audit",
        "ordered_exposure_derivation_and_qc",
        "ordered_category_exposure_qc",
        "ordered_exposure_quality_control",
        "ordinal_exposure_derivation_and_quality_control",
    }
)
_ADJUSTED_ASSOCIATION_METHODS = frozenset(
    {
        "adjusted_association_models",
        "adjusted_regression",
        "association_analysis",
        "mixed_effects_regression",
        "ordinal_dose_response",
        "regression_analysis",
    }
)
_TIMING_METHODS = frozenset(
    {
        "cox_proportional_hazards",
        "landmark_analysis",
        "survival_analysis",
        "time_to_event",
        "time_to_event_analysis",
    }
)

_COMPACT_MECHANICAL_GUIDANCE = """MECHANICAL PYTHON CONTRACT:
- Use valid Python collection literals/constructors; never write set(value) with multiple positional values.
- With NumPy 2, create nullable string labels via an object/string pandas Series plus `.loc`; do not mix strings and NaN in `np.where` or `np.select`.
- Import only packages listed in AVAILABLE ANALYTICAL LIBRARIES. Mechanical validation still runs before execution."""

_COMPACT_SERIALIZATION_GUIDANCE = """OUTPUT SERIALIZATION CONTRACT:
- JSON values must be Python primitives. Convert NumPy scalars to int/float/bool, arrays to lists, and pandas/NumPy missing or non-finite values to None; always pass a `default=` converter to `json.dump`.
- Every CSV cell must be one scalar. Emit separate columns for median, quartiles, counts, and percentages; emit one row per categorical level.
- Never assign the result of an inplace pandas operation, and prefer stable `.agg`/`.transform` over mixed-shape `groupby.apply`."""

_TABLE_ONE_SDK_GUIDANCE = """GROUPED TABLE 1 CONTRACT:
- `table_one_spec` is the sole authority for grouping, closed levels, summaries, and tests.
- Call `easyicu.research_agent.methods.table_one.build_grouped_table_one(frame, table_one_spec)` and save its returned long-form source table unchanged as `table_one.csv`.
- Do not hand-roll another test, coerce values, or replace the grouped table with an overall-only description."""

_COMPACT_ADJUSTED_CLINICAL_GUIDANCE = """ADJUSTED-MODEL CLINICAL INPUT CONTRACT:
- Treat declared measured/count/status companions as audit-only provenance; never use them to redefine the authoritative value, denominator, cohort, exposure, or outcome.
- For every declared measured/count pair, call the host `measurement_provenance_receipt` with the locked cohort and exact keyword column names, let any validation error propagate, and publish the unchanged receipts as `measurement_provenance_audit={"source":"COHORT_PARQUET","checks":[...]}`.
- Preserve missing physiological values. Never recode unmeasured laboratory, vital-sign, exposure, or outcome values to zero; implement only the Planner-declared complete-case or missing-indicator strategy and report the resulting model denominator.
- Preserve ordinal variables as declared ordered categories or an explicitly justified rank-preserving representation; do not average an ordinal score merely to make a model fit.
- Numeric coercion must count newly invalid values and fail closed on any lossy or non-finite non-missing input before model fitting or scientific output."""


def normalised_method_head(method: object) -> str:
    """Return the exact scientific method head before an optional rider."""

    normalised = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalised.split("_with_", 1)[0]


def _typed_products(values: Iterable[object]) -> tuple[tuple[str, str], ...]:
    products = []
    for raw in values:
        kind, separator, name = str(raw or "").strip().lower().partition(":")
        if separator and kind and name:
            products.append((kind, name))
    return tuple(products)


def _descriptive_table_contract_applies(step: AnalysisStep) -> bool:
    """Return whether exact method/product structure owns a descriptive table."""

    method = normalised_method_head(step.method)
    method_tokens = frozenset(method.split("_"))
    output_names = {
        name
        for kind, name in _typed_products(step.expected_outputs or [])
        if kind == "table"
    }
    semantic_summary = bool(
        "descriptive" in method_tokens
        or (
            "summary" in method_tokens
            and not method_tokens
            & {
                "model",
                "prediction",
                "provenance",
                "reporting",
                "robustness",
                "sensitivity",
            }
        )
    )
    return bool(
        method in _TABLE_METHODS
        or output_names & _DESCRIPTIVE_TABLE_PRODUCTS
        or semantic_summary
    )


def _quality_control_contract_applies(step: AnalysisStep) -> bool:
    """Recognise a case-neutral QC/audit method from its structural label."""

    method = normalised_method_head(step.method)
    tokens = frozenset(method.split("_"))
    return bool(
        method in _QUALITY_CONTROL_METHODS
        or "quality_control" in method
        or (
            "audit" in tokens
            and tokens & {"data", "exposure", "measurement", "missingness"}
        )
    )


def _robustness_contract_applies(step: AnalysisStep) -> bool:
    """Recognise a robustness/sensitivity step without intent-text routing."""

    method = normalised_method_head(step.method)
    tokens = frozenset(method.split("_"))
    return bool(method in _ROBUSTNESS_METHODS or tokens & {"robustness", "sensitivity"})


def _reporting_contract_applies(step: AnalysisStep) -> bool:
    """Return whether this is a typed manuscript/report assembly step."""

    method_tokens = frozenset(normalised_method_head(step.method).split("_"))
    output_kinds = {kind for kind, _ in _typed_products(step.expected_outputs or [])}
    return bool(
        output_kinds
        and output_kinds <= {"artifact", "report"}
        and method_tokens & {"manuscript", "provenance", "render", "reporting"}
    )


def _binary_event_presence_contract_applies(step: AnalysisStep) -> bool:
    """Select the sparse-event exception only for an explicit paired design."""

    method_tokens = frozenset(normalised_method_head(step.method).split("_"))
    output_tokens = {
        token
        for _, name in _typed_products(step.expected_outputs or [])
        for token in name.split("_")
    }
    semantic_tokens = method_tokens | output_tokens
    if not semantic_tokens & {"binary", "event", "incidence", "presence"}:
        return False
    names = {
        str(value or "").strip().lower()
        for value in (step.inputs or [])
        if ":" not in str(value or "")
    }

    def _companion_stem(name: str, role: str) -> str | None:
        match = re.fullmatch(rf"(.+)_{role}(?:_\d+(?:h|d))?", name)
        return match.group(1) if match else None

    count_stems = {
        stem for name in names if (stem := _companion_stem(name, "n")) is not None
    }
    measured_stems = {
        stem
        for name in names
        if (stem := _companion_stem(name, "measured")) is not None
    }
    return bool(count_stems & measured_stems)


def _figure_contract_applies(step: AnalysisStep) -> bool:
    """Return whether this is rendering-only, not a model with a figure output."""

    method = normalised_method_head(step.method)
    output_kinds = {kind for kind, _ in _typed_products(step.expected_outputs or [])}
    figure_kinds = {"figure", "plot", "chart", "heatmap"}
    return bool(
        method in _FIGURE_METHODS or (output_kinds and output_kinds <= figure_kinds)
    )


def _guide_segments(full_guide: str) -> dict[str, str]:
    """Split the versioned guide at stable semantic headings.

    Failing closed here is deliberate: a prompt-pack edit must preserve these
    headings or update this selector and its tests instead of silently sending
    the entire guide again.
    """

    anchors = {
        "adjusted": (
            "- For a regression step that explicitly requests separate "
            "source-aware and"
        ),
        "model_safety": "  Before fitting, audit every categorical predictor",
        "runtime": (
            "- Treat `COHORT_PARQUET` as the already-materialised, locked "
            "analysis cohort."
        ),
        "upstream": ("- If a step depends on an artefact produced by a previous step,"),
        "helper_guard": (
            "- Run every host-owned input-validation or provenance helper"
        ),
        "figure": "- For rendering-only figure steps,",
        "trajectory": "- OPTIONAL trajectory:",
        "visual": '- Use matplotlib\'s "Agg" backend;',
        "source": "- When reporting a source-status count map,",
        "ordered": "- CONTROLLED ORDERED-STRATIFIED METHOD:",
        "derived": "- DERIVED NUMBERS (optional):",
        "serialization": "PANDAS IDIOM GOTCHAS — common LLM mistakes to avoid:",
        "table": "TABLE-ONE / DESCRIPTIVE SUMMARIES:",
        "clinical": "CLINICAL SCORE AND MISSINGNESS SEMANTICS:",
        "binary_event": "- BINARY EVENT-PRESENCE EXCEPTION:",
        "clinical_tail": "- A shared source-status helper must make",
        "complete_case": "- Before any complete-case model,",
        "timing_guard": (
            "- If an exposure can be an intervention or treatment marker,"
        ),
        "timing": "- Exposure/event TIMING is available in the wide cohort:",
        "statistics": "STATISTICS APIs:",
        "model_failure": "- For a model-fitting failure only,",
        "hygiene": "PYTHON HYGIENE:",
        "prediction": "PREDICTION / CLUSTERING APIs:",
        "robustness": "ROBUSTNESS:",
    }
    positions = {name: full_guide.find(anchor) for name, anchor in anchors.items()}
    missing = [name for name, position in positions.items() if position < 0]
    if missing:
        raise ValueError(f"Coder prompt is missing scoped section anchors: {missing}")
    order = [
        "adjusted",
        "model_safety",
        "runtime",
        "upstream",
        "helper_guard",
        "figure",
        "trajectory",
        "visual",
        "source",
        "ordered",
        "derived",
        "serialization",
        "table",
        "clinical",
        "binary_event",
        "clinical_tail",
        "complete_case",
        "timing_guard",
        "timing",
        "statistics",
        "model_failure",
        "hygiene",
        "prediction",
        "robustness",
    ]
    if [positions[name] for name in order] != sorted(positions.values()):
        raise ValueError("Coder prompt scoped section anchors are out of order")
    segments = {"core": full_guide[: positions[order[0]]].strip()}
    for index, name in enumerate(order):
        end = positions[order[index + 1]] if index + 1 < len(order) else len(full_guide)
        segments[name] = full_guide[positions[name] : end].strip()
    return segments


def coder_guide_for_step(
    full_guide: str,
    step: AnalysisStep,
    *,
    _exclude_sections: frozenset[str] = frozenset(),
) -> str:
    """Select prompt sections from exact method and typed-product evidence."""

    sections = _guide_segments(full_guide)
    method = normalised_method_head(step.method)
    inputs = _typed_products(step.inputs or [])
    outputs = _typed_products(step.expected_outputs or [])
    output_kinds = {kind for kind, _ in outputs}
    input_names = {name for _, name in inputs}
    output_names = {name for _, name in outputs}
    is_data_quality_audit = canonical_analysis_family(method) == "data_quality_audit"
    is_quality_control = bool(
        _quality_control_contract_applies(step) or is_data_quality_audit
    )
    is_ordered = is_ordered_stratified_analysis_step(step)
    is_ordered_semantics = bool(
        is_ordered
        or method in _ORDERED_QUALITY_CONTROL_METHODS
        or {"ordinal", "ordered"} & set(method.split("_"))
        or any(
            token in name for name in output_names for token in ("ordinal", "ordered")
        )
    )
    is_cohort_change = cohort_change_contract_applies(step)
    is_robustness = _robustness_contract_applies(step)
    is_reporting = _reporting_contract_applies(step)
    is_trajectory = bool(
        trajectory_step_roles(step)
        or clustering_contract_applies(step)
        or step.trajectory_stability_spec is not None
    )

    selected = {"core", "runtime", "helper_guard"}
    if any(
        kind not in {"cohort", "dataset"} and name != "analysis_cohort"
        for kind, name in inputs
    ):
        # The host maps a locked cohort/dataset input to COHORT_PARQUET.  The
        # longer evidence-directory lookup tutorial is only useful when this
        # step must resolve some other typed upstream product itself.
        selected.add("upstream")
    is_figure = _figure_contract_applies(step)
    has_figure_output = bool(output_kinds & {"figure", "plot", "chart", "heatmap"})
    is_descriptive_table = _descriptive_table_contract_applies(step)
    if is_figure and "table" not in output_kinds:
        # A render-only product can legitimately retain the producer's
        # descriptive method label.  That label must not pull the full
        # table-one/clinical-statistics tutorial into a figure-only prompt.
        is_descriptive_table = False
    if is_figure:
        selected.add("figure")
    if has_figure_output:
        selected.add("visual")
    if is_trajectory:
        selected.update(("trajectory", "prediction"))
    if is_descriptive_table or is_data_quality_audit:
        selected.update(("source", "table"))
    elif is_quality_control:
        selected.add("source")
        if "table" in output_kinds:
            selected.add("table")
    if is_ordered:
        selected.update(("source", "ordered"))
    is_table = is_descriptive_table
    is_prediction = prediction_contract_applies(step)
    is_timing = method in _TIMING_METHODS or bool(
        output_names
        & {
            "hazard_ratio",
            "kaplan_meier",
            "survival_curve",
            "time_to_event",
        }
    )
    known_non_adjusted = bool(
        is_figure
        or is_table
        or is_trajectory
        or is_prediction
        or is_timing
        or is_quality_control
        or is_ordered
        or is_cohort_change
        or is_robustness
        or is_reporting
    )
    is_effect = effect_output_authorized(step)
    is_adjusted = bool(
        step.model_requirements
        or is_effect
        or method in _ADJUSTED_ASSOCIATION_METHODS
        or (not known_non_adjusted and not (is_figure or is_table or is_trajectory))
    )
    if is_adjusted:
        selected.add("model_safety")
        if step.model_requirements or method == PLANNED_MODEL_REQUIREMENTS_STEP_METHOD:
            selected.add("adjusted")
    has_provenance_inputs = any(
        re.search(
            r"(?:_n|_measured|_status|_first_time|_last_time)" r"(?:_\d+(?:h|d))?$",
            name,
        )
        for raw in (step.inputs or [])
        for name in [str(raw or "").strip().lower()]
        if ":" not in name
    )
    compact_adjusted_clinical = False
    if (
        is_quality_control
        or is_ordered
        or is_table
        or not (is_figure or is_trajectory or is_reporting)
    ):
        if is_adjusted and not (is_quality_control or is_ordered or is_table):
            compact_adjusted_clinical = True
        else:
            selected.add("clinical")
            if is_quality_control or has_provenance_inputs:
                selected.add("clinical_tail")
        if not is_ordered_semantics and _binary_event_presence_contract_applies(step):
            selected.add("binary_event")
    # The long generic tutorial is transport-heavy. Keep its few residual
    # safety rules in the compact contract below; deterministic preflight and
    # repair continue to own syntax, imports, and host-helper call shapes.
    selected.discard("hygiene")
    needs_statistics = bool(
        is_adjusted
        or is_prediction
        or is_timing
        or is_ordered
        or output_names & {"absolute_risk", "outcome_incidence", "outcome_rate"}
    )
    if needs_statistics:
        selected.update(("statistics", "model_failure"))
    elif is_cohort_change:
        selected.add("model_failure")
    if step.model_requirements or is_adjusted:
        selected.add("complete_case")
    if is_timing:
        selected.update(("timing_guard", "timing", "derived"))
    elif is_adjusted:
        selected.add("timing_guard")
    if is_prediction:
        selected.update(("prediction", "derived"))
    if is_robustness:
        selected.update(("derived", "robustness"))
    if any(
        name.endswith(("source_status", "missingness_audit")) for name in input_names
    ):
        selected.add("source")

    canonical_order = list(sections)
    parts = [
        sections[name]
        for name in canonical_order
        if name in selected and name not in _exclude_sections and sections[name]
    ]
    if "hygiene" not in _exclude_sections:
        parts.append(_COMPACT_MECHANICAL_GUIDANCE)
    if "serialization" not in _exclude_sections:
        parts.append(_COMPACT_SERIALIZATION_GUIDANCE)
    if step.table_one_spec is not None:
        parts.append(_TABLE_ONE_SDK_GUIDANCE)
    if compact_adjusted_clinical:
        parts.append(_COMPACT_ADJUSTED_CLINICAL_GUIDANCE)
    return "\n\n".join(parts).strip()


def coder_rewrite_guide_for_step(full_guide: str, step: AnalysisStep) -> str:
    """Select method/product guidance without duplicated transport tutorials.

    Full rewrite already carries the complete previous script, the compact
    typed-input/output contract, mechanical guardrails, and complete scoped
    scientific authority. Repeating generic runtime, pandas-serialization,
    and hygiene tutorials consumes transport without adding method-family
    evidence. Initial generation keeps those sections unchanged.
    """

    return coder_guide_for_step(
        full_guide,
        step,
        _exclude_sections=frozenset(
            {"runtime", "upstream", "helper_guard", "serialization", "hygiene"}
        ),
    )


def coder_context_requires_method_constraints(step: AnalysisStep) -> bool:
    """Return whether the scoped context needs model-compatibility prose."""

    method = normalised_method_head(step.method)
    if step.model_requirements:
        return True
    if effect_output_authorized(step):
        return True
    method_tokens = frozenset(normalised_method_head(step.method).split("_"))
    output_products = _typed_products(step.expected_outputs or [])
    output_kinds = {kind for kind, _ in output_products}
    output_tokens = {
        token for _, name in output_products for token in str(name or "").split("_")
    }
    if (
        output_kinds
        and output_kinds <= {"artifact", "statistic", "table"}
        and (method_tokens | output_tokens)
        & {"balance", "diagnostic", "diagnostics", "positivity"}
    ):
        # These steps diagnose a prespecified model design but do not own the
        # final effect. The compact model-safety guide plus deterministic
        # compatibility gate carry the same host rules without repeating the
        # full variable-by-variable compatibility prose in every prompt.
        return False
    if canonical_analysis_family(
        method
    ) == "data_quality_audit" or _quality_control_contract_applies(step):
        return False
    if is_ordered_stratified_analysis_step(step):
        return False
    if cohort_change_contract_applies(step):
        return False
    if _descriptive_table_contract_applies(step):
        return False
    if _figure_contract_applies(step):
        return False
    if _reporting_contract_applies(step):
        return False
    return method not in (_FIGURE_METHODS | _TABLE_METHODS | _QUALITY_CONTROL_METHODS)


def _variable_family(name: object) -> str:
    lowered = str(name or "").strip().lower()
    for suffix in _COMPANION_SUFFIXES:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _is_required_source_companion(variable: object) -> bool:
    """Keep provenance coordinates, not every unrequested sibling summary.

    ``source_concept`` groups physical export columns.  A selected value needs
    its count/measurement/time/status coordinates for provenance auditing, but
    unrelated ``first/min/max/mean`` representations are different scientific
    variables and should not be injected unless the Planner declared them.
    """

    name = str(getattr(variable, "name", "") or "").strip().lower()
    return bool(
        re.search(
            r"(?:_n|_measured|_status|_valid|_first_time|_last_time)"
            r"(?:_\d+(?:h|d))?$",
            name,
        )
    )


def _is_automatically_required_source_companion(variable: object) -> bool:
    """Return provenance fields that follow a selected value automatically.

    Count/measurement/status coordinates are needed to distinguish observed,
    missing, and structurally unavailable values. Timing columns are distinct
    scientific inputs: include them when the Planner declares them, but do not
    pull every unrequested sibling time column into a wide step merely because
    another representation from the same source concept was selected.
    """

    name = str(getattr(variable, "name", "") or "").strip().lower()
    return bool(
        re.search(
            r"(?:_n|_measured|_status|_valid)(?:_\d+(?:h|d))?$",
            name,
        )
    )


def scoped_coder_context(
    context: ResearchContext,
    step: AnalysisStep,
    *,
    code: str = "",
    max_variables: int = 36,
) -> ResearchContext:
    """Project context variables to the current step without changing science."""

    declared = {
        str(value or "").strip().lower()
        for value in (step.inputs or [])
        if ":" not in str(value or "") and str(value or "").strip()
    }
    declared.update(
        str(value or "").strip().lower()
        for requirement in (step.model_requirements or [])
        for value in (requirement.outcome, requirement.exposure_source)
        if str(value or "").strip()
    )
    direct = {
        str(value).strip().lower()
        for value in (context.target_outcome, context.primary_exposure)
        if value
    }
    seed_names = declared | direct
    code_referenced = set()
    if code:
        code_referenced.update(
            variable.name.lower()
            for variable in context.variables
            if re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
                code,
            )
        )
        seed_names.update(code_referenced)
    families = {_variable_family(value) for value in seed_names}
    source_concepts = {
        str(variable.source_concept).strip().lower()
        for variable in context.variables
        if variable.name.lower() in seed_names and variable.source_concept
    }
    priority = []
    referenced = []
    for variable in context.variables:
        name = variable.name.lower()
        source_concept = str(variable.source_concept or "").strip().lower()
        if (
            name in declared
            or name in direct
            or name in code_referenced
            or (
                _is_automatically_required_source_companion(variable)
                and (
                    _variable_family(name) in families
                    or (source_concept and source_concept in source_concepts)
                )
            )
        ):
            priority.append(variable)
        elif code and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(variable.name)}(?![A-Za-z0-9_])",
            code,
        ):
            referenced.append(variable)
    # ``max_variables`` is a transport target, not permission to cut an
    # authoritative input set in half. Keep every declared/direct/code variable
    # and its required provenance coordinates. Do not pad spare capacity with
    # unrelated cohort columns; that was the main source of 36-column prompts
    # whose useful metadata was still incomplete.
    cap = max(1, int(max_variables))
    selected = list(priority)
    if len(selected) < cap:
        selected.extend(referenced[: cap - len(selected)])
    return project_research_context_variables(
        context,
        selected,
        additional_concept_ids=tuple(sorted(declared)),
        # The selected variables already include the value and its required
        # provenance companions. Retaining every other physical column with the
        # same source_concept would recreate the wide-context leak inside the
        # host-owned materialized attachment.
        include_source_concept_siblings=False,
    )


__all__ = [
    "coder_context_requires_method_constraints",
    "coder_guide_for_step",
    "normalised_method_head",
    "scoped_coder_context",
]
