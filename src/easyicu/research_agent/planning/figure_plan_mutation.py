"""Typed figure-plan mutation and render-source authorization."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from ..contracts.declared_product import (
    effect_adjustment_family,
    effect_bearing_product,
    effect_estimand_tier,
    effect_measure_family,
    effect_role_family,
    typed_product,
)
from ..contracts.product_identity import normalised_method_head as _normalised_method_head
from ..contracts.step_families import (
    _EFFECT_CONTRACT_METHODS,
    _ROBUSTNESS_EFFECT_CONTRACT_METHODS,
    _step_is_figure_only,
    _typed_effect_result_identities,
    effect_output_authorized,
)
from .figure_step_contract import _output_declares_figure, _step_produces_figure
from ..schema import (
    ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES,
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
    ResearchContext,
    ValidationFinding,
)

_RENDER_SOURCE_OUTPUT_KINDS = frozenset({"statistic", "table"})


def _typed_render_source_outputs(outputs: Sequence[str]) -> List[str]:
    """Return exact finalized parent result products a render child may consume.

    Raw artifacts, datasets, and models stay on the scientific parent.  A
    rendering-only child receives only finalized table/statistic products so it
    cannot silently reopen cohort, exposure, outcome, or model decisions.
    """

    render_inputs: List[str] = []
    for raw in outputs or []:
        value = str(raw or "").strip()
        parsed = typed_product(value)
        if parsed is not None and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS:
            render_inputs.append(value)
    return render_inputs


def _typed_render_source_identities(outputs: Sequence[str]) -> set[Tuple[str, str]]:
    """Return canonical typed identities eligible as scientific render inputs."""

    return {
        parsed
        for raw in (outputs or [])
        if (parsed := typed_product(raw)) is not None
        and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS
    }


def _effect_figure_semantics_supported_by_inputs(
    *,
    figure_outputs: Sequence[str],
    effect_input_products: set[Tuple[str, str]],
) -> bool:
    """Return whether each effect figure is supported by one bound table.

    Scientific authority is conjunctive and input-local.  A renderer may not
    borrow an OR scale from one sibling product, an ``adjusted`` qualifier from
    another, and a subgroup role from a third.  Every planned effect figure
    must have at least one *actually bound* effect table whose explicit scale,
    role, and adjustment qualifiers support that figure.  A generic figure may
    use a generic effect table, but may not relabel a subgroup/interaction-only
    table as an overall effect.
    """

    effect_figures = [
        raw
        for raw in figure_outputs
        if (parsed := typed_product(raw)) is not None
        and parsed[0] == "figure"
        and effect_bearing_product(raw)
    ]
    if not effect_figures:
        return True

    table_inputs = {
        product for product in effect_input_products if product[0] == "table"
    }
    if not table_inputs:
        return False

    declarations = [f"{kind}:{name}" for kind, name in table_inputs]
    for figure_output in effect_figures:
        output_measure = effect_measure_family(figure_output)
        output_role = effect_role_family(figure_output)
        output_tier = effect_estimand_tier(figure_output)
        output_adjustment = effect_adjustment_family(figure_output)
        supported = False
        for declaration in declarations:
            input_measure = effect_measure_family(declaration)
            input_role = effect_role_family(declaration)
            input_tier = effect_estimand_tier(declaration)
            input_adjustment = effect_adjustment_family(declaration)
            if output_measure is not None and input_measure != output_measure:
                continue
            if output_role is not None:
                if input_role != output_role:
                    continue
            elif input_role is not None:
                # A specialized-only source cannot silently become an overall
                # or otherwise generic effect display.
                continue
            if output_tier is not None:
                if input_tier != output_tier:
                    continue
            elif input_tier in {"secondary", "sensitivity", "corroborative"}:
                # Primary is the default estimand tier for an otherwise generic
                # effect figure. Supporting-only estimates may not silently be
                # promoted into that default role.
                continue
            if output_adjustment is not None and input_adjustment != output_adjustment:
                continue
            supported = True
            break
        if not supported:
            return False
    return True


def _effect_figure_semantics_supported_by_model_roster(
    *,
    step: AnalysisStep,
    figure_outputs: Sequence[str],
    effect_input_products: set[Tuple[str, str]],
) -> bool:
    """Authorize a primary adjusted-effect render from a typed model roster.

    The legacy adjusted-association product name is intentionally generic, but
    a non-empty ``model_requirements`` roster is Planner-owned and fixes the
    single primary model.  It can therefore support only a generic/primary
    adjusted-effect figure (or an explicit OR for a binary logistic primary),
    never a subgroup, interaction, secondary, sensitivity, HR, RR, or RD claim.
    """

    if ("table", "adjusted_association_estimates") not in effect_input_products:
        return False
    primary_requirements = [
        requirement
        for requirement in step.model_requirements or []
        if requirement.analysis_role == "primary"
    ]
    if len(primary_requirements) != 1:
        return False
    primary = primary_requirements[0]
    primary_method = re.sub(
        r"[^a-z0-9]+", "_", str(primary.method_family or "").lower()
    ).strip("_")
    for output in figure_outputs:
        if not effect_bearing_product(output):
            continue
        if effect_role_family(output) is not None:
            return False
        if effect_estimand_tier(output) not in {None, "primary"}:
            return False
        if effect_adjustment_family(output) not in {None, "adjusted"}:
            return False
        measure = effect_measure_family(output)
        if measure is None:
            continue
        if not (
            measure == "odds_ratio"
            and primary.outcome_type == "binary"
            and primary_method in ADJUSTED_ASSOCIATION_BINARY_METHOD_FAMILIES
        ):
            return False
    return True


def _render_only_figure_step_intent(
    *,
    source_step_id: str,
    figure_outputs: Sequence[str],
) -> str:
    """Return the exact framework-authored intent for a split render child."""

    return (
        f"Render the publication figure(s) declared by step "
        f"'{source_step_id}' ({', '.join(figure_outputs)}). Treat this as "
        "a rendering-only step: first read the table/statistic outputs "
        f"produced by '{source_step_id}' from the registered evidence files "
        "or from that step's outputs directory. Do not redefine the "
        "cohort, exposure, outcome, missing-data policy, or model inside "
        "this figure step; if the upstream table cannot support the "
        "requested figure, write a step_summary.json explaining the "
        "missing source-data contract instead of re-analysing "
        "``os.environ['COHORT_PARQUET']``. Save PNG and SVG copies of "
        "every figure with matching stems into "
        "``os.environ['STEP_OUT_DIR']``. Always write a valid "
        "step_summary.json into ``STEP_OUT_DIR`` listing each produced "
        "file under ``figure_files`` even if rendering fails — use a "
        "try/except so the step never aborts before writing the summary."
    )


def _effect_figure_source_authorized(
    *,
    step: AnalysisStep,
    completed_step_records: Optional[Sequence[Dict[str, Any]]],
    resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> bool:
    """Authorize only a figure name rendered from a successful effect parent.

    A render child never becomes an effect-method owner. This narrow host-side
    authority permits its planned/registered *figure* name only when the child
    is structurally linked to the latest successful direct parent through an
    exact typed effect-result input. Numeric summaries and non-figure effect
    products remain governed by the ordinary effect-method authorization.

    Rendering is recognised by what the step declares, not by what it is
    called.  The conditions below already say it exactly: every declared output
    is ``figure:``/``log:``, and every input is a typed ``statistic:``/``table:``
    bound by evidence id and sha256 to a successful effect parent -- a step
    built that way has no cohort to re-analyse.  Requiring the *name* to match a
    list of figure spellings on top of that refused legitimate render children:
    ``forest_plot``, the standard name for this figure and one the Planner is
    told it may use, was absent from the list.  What survives is the property
    that veto was reaching for -- a render child may not claim to be the
    analysis -- read from ``_EFFECT_CONTRACT_METHODS``, the same vocabulary
    :func:`effect_output_authorized` uses to decide who owns an effect result,
    rather than from a second private list that can drift away from it.
    """

    step_id = str(step.step_id or "")
    output_products = [typed_product(raw) for raw in (step.expected_outputs or [])]
    if (
        _normalised_method_head(str(step.method or ""))
        in (_EFFECT_CONTRACT_METHODS | _ROBUSTNESS_EFFECT_CONTRACT_METHODS)
        or not output_products
        or any(product is None for product in output_products)
        or not any(product[0] == "figure" for product in output_products if product)
        or any(
            product[0] not in {"figure", "log"}
            for product in output_products
            if product
        )
        or not completed_step_records
        or not resolved_input_bindings
    ):
        return False

    child_inputs: List[Tuple[Tuple[str, str], str]] = []
    producer_by_product: Dict[Tuple[str, str], str] = {}
    effect_parent_steps: Dict[str, AnalysisStep] = {}
    for raw in step.inputs or []:
        raw_input = str(raw or "")
        parsed = typed_product(raw_input)
        if parsed is None or parsed[0] not in _RENDER_SOURCE_OUTPUT_KINDS:
            return False
        binding = resolved_input_bindings.get(raw_input)
        if not isinstance(binding, Mapping):
            return False
        binding_product = typed_product(
            f"{binding.get('declared_kind', '')}:{binding.get('product', '')}"
        )
        if (
            binding_product != parsed
            or not str(binding.get("evidence_id") or "").strip()
            or re.fullmatch(
                r"[0-9a-fA-F]{64}", str(binding.get("sha256") or "").strip()
            )
            is None
        ):
            return False
        producer_id = str(binding.get("produced_by_step") or "").strip()
        if not producer_id:
            return False
        prior_producer = producer_by_product.get(parsed)
        if prior_producer is not None and prior_producer != producer_id:
            return False
        producer_by_product[parsed] = producer_id
        child_inputs.append((parsed, producer_id))

    if not child_inputs:
        return False
    latest_records: Dict[str, Mapping[str, Any]] = {}
    for record in completed_step_records:
        if isinstance(record, Mapping):
            record_step_id = str(record.get("step_id") or "").strip()
            if record_step_id:
                latest_records[record_step_id] = record

    effect_input_products: set[Tuple[str, str]] = set()
    for child_product, parent_step_id in child_inputs:
        if step_id == parent_step_id:
            return False
        parent_record = latest_records.get(parent_step_id)
        if (
            parent_record is None
            or str(parent_record.get("status") or "").strip().lower() != "ok"
        ):
            return False
        analysis_request = parent_record.get("analysis_request")
        raw_parent_step = (
            analysis_request.get("step")
            if isinstance(analysis_request, Mapping)
            else None
        )
        if not isinstance(raw_parent_step, Mapping):
            return False
        try:
            parent_step = AnalysisStep.model_validate(raw_parent_step)
        except (TypeError, ValueError, ValidationError):
            return False
        parent_render_products = _typed_render_source_identities(
            parent_step.expected_outputs or []
        )
        if (
            str(parent_step.step_id) != parent_step_id
            or child_product not in parent_render_products
        ):
            return False
        parent_effect_products = _typed_effect_result_identities(
            parent_step.expected_outputs or []
        )
        if child_product in parent_effect_products:
            if not effect_output_authorized(parent_step):
                return False
            effect_input_products.add(child_product)
            effect_parent_steps[parent_step_id] = parent_step

    return bool(
        any(kind == "table" for kind, _product in effect_input_products)
        and (
            _effect_figure_semantics_supported_by_inputs(
                figure_outputs=step.expected_outputs or [],
                effect_input_products=effect_input_products,
            )
            or (
                len(effect_parent_steps) == 1
                and _effect_figure_semantics_supported_by_model_roster(
                    step=next(iter(effect_parent_steps.values())),
                    figure_outputs=step.expected_outputs or [],
                    effect_input_products=effect_input_products,
                )
            )
        )
    )


_PUBLICATION_FIGURE_TRIGGER_TOKENS = (
    "publication-ready figure",
    "publication ready figure",
    "publication figure",
    "produce a heatmap",
    "produce a figure",
    "publication-ready",
    "and a figure",
    "and a heatmap",
    "and a publication",
    "publication-quality figure",
)


def _split_table_and_figure_outputs_in_plan(
    plan: AnalysisPlan,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Split steps that declare both table and figure outputs into two steps.

    A step like ``expected_outputs=['table:table_one', 'figure:table_one_visual']``
    asks the coder agent to produce *both* a CSV table and a publication
    figure inside a single executable script. Naive arms frequently
    deliver only the table and ignore the figure, then exhaust the
    LLM-repair budget without recovering. Splitting the step into a
    table-only step plus a downstream figure-only step gives the agent
    a focused target for each artefact while keeping the analytic
    intent intact.

    The split is conservative: it only fires when a single step
    declares at least one ``table:`` (or ``statistic:``) output *and*
    at least one ``figure:`` output. Non-figure outputs stay on the
    original step; figure outputs migrate to a new appended step
    inserted directly after the original. Other steps in the plan are
    left untouched.
    """
    if not plan.steps:
        return plan, []

    new_steps: List[AnalysisStep] = []
    findings: List[ValidationFinding] = []
    existing_step_ids = {str(step.step_id) for step in plan.steps}
    outputs_by_step = {
        str(step.step_id): list(step.expected_outputs or []) for step in plan.steps
    }
    rehomed_figure_dependencies: Dict[str, Dict[str, str]] = {}
    dedicated_dedup_records: List[Tuple[str, str, str]] = []

    # Prefer an explicit rendering-only owner when a Planner also attached the
    # exact same figure to a mixed scientific step. Keeping both would create
    # two evidence owners; splitting the mixed step would make that duplication
    # even less visible. The exact product identity makes this a structural
    # de-duplication only: no chart, source table, or scientific role is chosen
    # by the host.
    dedicated_figure_owners: Dict[Tuple[str, str], List[str]] = {}
    for candidate in plan.steps:
        if not _step_is_figure_only(candidate):
            continue
        for output in candidate.expected_outputs or []:
            product = typed_product(output)
            if product is not None and product[0] == "figure":
                dedicated_figure_owners.setdefault(product, []).append(
                    str(candidate.step_id)
                )
    for step in plan.steps:
        if _step_is_figure_only(step):
            continue
        step_id = str(step.step_id)
        for output in list(outputs_by_step[step_id]):
            product = typed_product(output)
            owners = dedicated_figure_owners.get(product or ("", ""), [])
            if product is None or product[0] != "figure" or len(owners) != 1:
                continue
            outputs_by_step[step_id].remove(output)
            dedicated_dedup_records.append((step_id, str(output), owners[0]))

    # A planner may attach a figure to the wrong mixed-output step even though
    # another step declares the figure's exact typed table/statistic product.
    # Repair only that structural, case-neutral identity: ``figure:x`` can be
    # rehomed to the sole ``table:x``/``statistic:x`` producer.  Figures whose
    # names intentionally differ from their source (for example ``love_plot``)
    # remain Planner-owned and are not guessed from keywords.
    render_sources_by_name: Dict[str, List[Tuple[str, str]]] = {}
    for candidate in plan.steps:
        candidate_id = str(candidate.step_id)
        for output in candidate.expected_outputs or []:
            parsed = typed_product(output)
            if parsed is not None and parsed[0] in _RENDER_SOURCE_OUTPUT_KINDS:
                render_sources_by_name.setdefault(parsed[1], []).append(
                    (candidate_id, str(output))
                )

    for step in plan.steps:
        step_id = str(step.step_id)
        step_outputs = outputs_by_step[step_id]
        # A dedicated rendering step already owns the figure and its typed
        # dependency. Rehoming applies only when a Planner mixed a figure into
        # a scientific/result step. Moving the sole output out of an explicit
        # renderer would erase that step's contract and leave an empty action
        # in the plan presented for human approval.
        step_has_non_figure_output = any(
            not _output_declares_figure(candidate)
            for candidate in step_outputs
        )
        if not step_has_non_figure_output:
            continue
        for output in list(outputs_by_step[step_id]):
            parsed = typed_product(output)
            if parsed is None or parsed[0] != "figure":
                continue
            exact_sources = render_sources_by_name.get(parsed[1], [])
            if len(exact_sources) != 1:
                continue
            source_step_id, source_output = exact_sources[0]
            if source_step_id == step_id:
                continue
            source_step = next(
                item for item in plan.steps if str(item.step_id) == source_step_id
            )
            source_already_owns_figure = any(
                (candidate_product := typed_product(candidate_output)) is not None
                and candidate_product[0] == "figure"
                for candidate_output in outputs_by_step[source_step_id]
            )
            if (
                f"{source_step_id}_figure" in existing_step_ids
                or source_already_owns_figure
                or _normalised_method_head(str(source_step.method or ""))
                in {"association_robustness", "bias_audit_association", "clustering"}
                or typed_product(source_output)[0] != "table"
            ):
                continue
            outputs_by_step[step_id].remove(output)
            outputs_by_step[source_step_id].append(output)
            rehomed_figure_dependencies.setdefault(source_step_id, {})[
                str(output)
            ] = source_output
            findings.append(
                ValidationFinding(
                    validator="plan_contract",
                    severity="warning",
                    message=(
                        f"Rehomed '{output}' from step '{step_id}' to the sole "
                        f"exact typed source producer '{source_step_id}'."
                    ),
                    detail={
                        "reason": "figure_exact_typed_source_rehome",
                        "figure_output": str(output),
                        "original_step_id": step_id,
                        "source_step_id": source_step_id,
                        "source_output": source_output,
                    },
                )
            )

    for source_step_id, figure_output, dedicated_owner_id in dedicated_dedup_records:
        findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    f"Removed duplicate '{figure_output}' from mixed step "
                    f"'{source_step_id}'; dedicated rendering step "
                    f"'{dedicated_owner_id}' remains its sole owner."
                ),
                detail={
                    "reason": "figure_duplicate_owned_by_dedicated_renderer",
                    "figure_output": figure_output,
                    "mixed_step_id": source_step_id,
                    "dedicated_renderer_step_id": dedicated_owner_id,
                },
            )
        )

    typed_product_producers: Dict[Tuple[str, str], Set[str]] = {}
    for candidate in plan.steps:
        for output in outputs_by_step[str(candidate.step_id)]:
            parsed = typed_product(output)
            if parsed is not None:
                typed_product_producers.setdefault(parsed, set()).add(
                    str(candidate.step_id)
                )

    for step in plan.steps:
        outputs = outputs_by_step[str(step.step_id)]
        working_step = (
            step
            if outputs == list(step.expected_outputs or [])
            else step.model_copy(update={"expected_outputs": outputs})
        )
        method = _normalised_method_head(str(working_step.method or ""))
        typed_table_inputs = [
            str(raw_input)
            for raw_input in working_step.inputs
            if (parsed_input := typed_product(raw_input)) is not None
            and parsed_input[0] == "table"
        ]
        if (
            method == "visualization"
            and typed_table_inputs
            and not working_step.input_consumption_contracts
        ):
            working_step = working_step.model_copy(
                update={
                    "input_consumption_contracts": [
                        ArtifactConsumptionContract(
                            input_key=input_key,
                            mode="all_rows",
                        )
                        for input_key in typed_table_inputs
                    ]
                }
            )
            findings.append(
                ValidationFinding(
                    validator="plan_contract",
                    severity="warning",
                    message=(
                        f"Bound visualization step '{working_step.step_id}' to "
                        "preserve all rows from each exact typed table input; "
                        "role-specific row selection requires an explicit Planner "
                        "consumption contract."
                    ),
                    detail={
                        "reason": "visualization_all_rows_consumption_default",
                        "step_id": working_step.step_id,
                        "inputs": typed_table_inputs,
                    },
                )
            )
        if method in {
            "association_robustness",
            "bias_audit_association",
            "clustering",
        }:
            # ``prediction_model`` is intentionally NOT in this skip-list:
            # the canonical ``01_model_training`` step bundles both a
            # ``table:model_performance`` analytic payload and a
            # ``figure:discrimination_calibration`` figure, and the agent
            # frequently forgets to render the figure when both are demanded
            # in a single script. Splitting yields a sibling
            # ``01_model_training_figure`` whose contract is purely visual,
            # which is what
            # ``test_mock_planner_emits_prediction_analysis_and_publication_for_prediction_question``
            # pins.
            new_steps.append(working_step)
            continue
        figure_outputs = [out for out in outputs if _output_declares_figure(out)]
        non_figure_outputs = [out for out in outputs if out not in figure_outputs]
        # Split only when the figure has a typed parent data/model product to
        # consume. A log is a sidecar, not render source data; splitting a
        # ``figure + log`` step would create an empty-input child that can only
        # guess or scan unrelated evidence.
        render_source_outputs = _typed_render_source_outputs(non_figure_outputs)
        explicit_rehomed_dependencies = rehomed_figure_dependencies.get(
            str(step.step_id), {}
        )
        if explicit_rehomed_dependencies:
            # Rehoming is authorized by the exact typed product role that caused
            # the move. Do not widen that closed dependency to every table owned
            # by the producer. Figures requiring a multi-product renderer
            # contract remain Planner-owned and are not inferred here.
            render_source_outputs = list(
                dict.fromkeys(
                    explicit_rehomed_dependencies[figure_output]
                    for figure_output in figure_outputs
                    if figure_output in explicit_rehomed_dependencies
                )
            )
        render_source_identities = {
            parsed
            for output in render_source_outputs
            if (parsed := typed_product(output)) is not None
        }
        sources_have_unique_parent = all(
            typed_product_producers.get(identity) == {str(step.step_id)}
            for identity in render_source_identities
        )
        has_render_source_table = any(
            (parsed := typed_product(output)) is not None and parsed[0] == "table"
            for output in render_source_outputs
        )
        effect_figure_requested = any(
            effect_bearing_product(output) for output in figure_outputs
        )
        effect_source_products = _typed_effect_result_identities(render_source_outputs)
        effect_figure_supported = _effect_figure_semantics_supported_by_inputs(
            figure_outputs=figure_outputs,
            effect_input_products=effect_source_products,
        ) or _effect_figure_semantics_supported_by_model_roster(
            step=step,
            figure_outputs=figure_outputs,
            effect_input_products=effect_source_products,
        )
        if (
            not figure_outputs
            or not has_render_source_table
            or not sources_have_unique_parent
            or (effect_figure_requested and not effect_figure_supported)
        ):
            new_steps.append(working_step)
            continue
        # Keep the original step with the non-figure outputs.
        non_figure_step = working_step.model_copy(
            update={"expected_outputs": non_figure_outputs}
        )
        new_steps.append(non_figure_step)
        # Synthesise a follow-up figure-only step.
        figure_step_id = f"{step.step_id}_figure"
        if figure_step_id in existing_step_ids:
            new_steps[-1] = step
            continue
        figure_intent = _render_only_figure_step_intent(
            source_step_id=str(step.step_id),
            figure_outputs=figure_outputs,
        )
        figure_step = AnalysisStep(
            step_id=figure_step_id,
            planned_analysis_role="auxiliary",
            intent=figure_intent,
            inputs=render_source_outputs,
            expected_outputs=figure_outputs,
            method="visualization",
            icu_rule_refs=list(working_step.icu_rule_refs or [])
            + ["visualization_rule"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key=str(input_key),
                    mode="all_rows",
                )
                for input_key in render_source_outputs
                if (parsed_input := typed_product(input_key)) is not None
                and parsed_input[0] == "table"
            ],
        )
        new_steps.append(figure_step)
        findings.append(
            ValidationFinding(
                validator="plan_contract",
                severity="warning",
                message=(
                    f"Split step '{step.step_id}' into a table/statistic "
                    f"step and a follow-up figure step "
                    f"'{figure_step_id}' so the coder can target each "
                    "artefact independently."
                ),
                detail={
                    "source_step_id": step.step_id,
                    "non_figure_outputs": non_figure_outputs,
                    "figure_outputs": figure_outputs,
                    "appended_step_id": figure_step_id,
                },
            )
        )

    if not findings:
        return plan, []
    return plan.model_copy(update={"steps": new_steps}), findings


def _research_question_implies_figure(question: str) -> bool:
    """Heuristic: does the research question call for a figure deliverable?"""
    text = (question or "").lower()
    if not text:
        return False
    if any(token in text for token in _PUBLICATION_FIGURE_TRIGGER_TOKENS):
        return True
    return re.search(r"\bfigure\s+or\b", text) is not None


def _ensure_publication_figure_step_in_plan(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
    force: bool = False,
) -> Tuple[AnalysisPlan, List[ValidationFinding]]:
    """Append a fallback figure step when the planner forgot one.

    Naive arms (no ICU narrative context) sometimes emit a single-step
    plan that omits the publication figure even when the research
    question explicitly asks for one. Task contracts in the EasyICU
    experiment runner still require a ``figure`` artefact in those
    cases. Detect the gap and append a generic figure step so the
    coder agent has a concrete target. The step's ``intent`` is broad
    enough that the coder can still tailor the chart shape (bar, box,
    forest, heatmap…) based on the upstream analytics.

    ``force=True`` bypasses the research-question heuristic: the caller
    already knows a figure will be produced (e.g. the publication-figure
    skill is enabled), so the plan should *declare* the figure even when
    the question text never says "figure". Used by the execute phase,
    where the plan that actually runs is the replanner's — which the
    plan-phase, question-gated guard never sees.
    """
    if any(_step_produces_figure(step) for step in plan.steps or []):
        return plan, []
    if not force and not _research_question_implies_figure(
        context.research_question or ""
    ):
        return plan, []

    # A host guard may request a missing display deliverable, but it must not
    # choose a scientific result by scanning arbitrary run files.  Bind the
    # renderer only to planner-declared table products with a unique producer.
    producer_ids: Dict[Tuple[str, str], Set[str]] = {}
    ordered_table_outputs: List[Tuple[Tuple[str, str], str]] = []
    for candidate in plan.steps or []:
        for raw_output in candidate.expected_outputs or []:
            parsed = typed_product(raw_output)
            if parsed is None or parsed[0] != "table":
                continue
            producer_ids.setdefault(parsed, set()).add(str(candidate.step_id))
            ordered_table_outputs.append((parsed, str(raw_output)))
    render_inputs: List[str] = []
    seen_inputs: Set[Tuple[str, str]] = set()
    for identity, raw_output in ordered_table_outputs:
        if identity in seen_inputs or len(producer_ids.get(identity, set())) != 1:
            continue
        seen_inputs.add(identity)
        render_inputs.append(raw_output)
    if not render_inputs:
        return plan, [
            ValidationFinding(
                validator="plan_contract",
                severity="error",
                message=(
                    "The plan requires a publication figure but declares no "
                    "uniquely produced typed table that a rendering-only step "
                    "can consume. The planner must declare the intended figure "
                    "and its exact typed source instead of asking the runtime "
                    "to scan prior outputs and choose a scientific result."
                ),
                detail={"reason": "missing_typed_figure_source"},
            )
        ]

    next_index = len(plan.steps or []) + 1
    fallback_step = AnalysisStep(
        step_id=f"{next_index:02d}_publication_figure_fallback",
        planned_analysis_role="auxiliary",
        intent=(
            "Render a publication-ready overview using only the exact typed "
            "table inputs bound by the host. Do not scan the run directory, "
            "choose a different result, fit a model, or calculate a new "
            "estimand. Copy every plotted value into a matching source-data "
            "CSV and declare that CSV in the figure contract, then save "
            "the figure as both PNG and SVG with the same stem into "
            "``os.environ['STEP_OUT_DIR']`` (set by the runner). Record "
            "every produced path in step_summary.json under "
            "``figure_files``."
        ),
        method="visualization",
        inputs=render_inputs,
        expected_outputs=["figure:overview"],
        icu_rule_refs=["visualization_rule"],
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key=input_key, mode="all_rows")
            for input_key in render_inputs
        ],
    )
    new_steps = list(plan.steps or []) + [fallback_step]
    preserved = plan.model_copy(update={"steps": new_steps})
    findings = [
        ValidationFinding(
            validator="plan_contract",
            severity="warning",
            message=(
                "Plan did not declare a figure step even though the "
                "research question asked for a publication-ready "
                "figure; appended a fallback figure step "
                f"'{fallback_step.step_id}' to preserve the task contract."
            ),
            detail={"appended_step_id": fallback_step.step_id},
        )
    ]
    return preserved, findings

