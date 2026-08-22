"""Select trusted executors for complete Planner-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...authority.plausibility import FlagOnlyPlausibilityScope
from ...authority.current_case_scientific_runtime import (
    AssociationModelGridRuntimeAuthority,
    LandmarkSplineRuntimeAuthority,
    SourceFeasibilityRuntimeAuthority,
    load_current_case_scientific_runtime_authority,
)
from ...contracts.ownership_verdict import OwnershipVerdict
from .deterministic_robustness import (
    ROBUSTNESS_REPLAY_ANALYSIS_KIND,
    robustness_replay_declaration_verdict,
)
from ...schema import AnalysisPlan, AnalysisStep
from .adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    adjusted_association_executor_code,
    adjusted_association_executor_verdict,
)
from .association_model_grid_executor import (
    ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
    association_model_grid_executor_code,
    association_model_grid_executor_owns_step,
)
from .exposure_outcome_distribution_render import (
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
    exposure_outcome_distribution_figure_declaration_verdict,
    exposure_outcome_distribution_figure_code,
    exposure_outcome_distribution_figure_owns_step,
)
from .exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_declaration_verdict,
    exposure_outcome_distribution_executor_code,
    exposure_outcome_distribution_executor_owns_step,
)
from .cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
)
from .ordered_stratified_executor import (
    ORDERED_STRATIFIED_ANALYSIS_KIND,
    ordered_stratified_consumed_input_keys,
    ordered_stratified_executor_code,
    ordered_stratified_executor_owns_step,
)
from .deterministic_missingness import (
    is_compact_missingness_measurement_contract,
    is_measurement_bias_audit_contract,
    is_missingness_complete_case_contract,
    missingness_audit_cohort_input_key,
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
    source_availability_audit_executor_owns_step,
)
from .missingness_measurement_figure_executor import (
    missingness_measurement_figure_declaration_verdict,
    measurement_missingness_figure_executor_code,
    measurement_missingness_figure_executor_owns_step,
    missingness_measurement_figure_executor_code,
    missingness_measurement_figure_executor_owns_step,
)
from .robustness_figure_executor import (
    robustness_figure_consumed_input_keys,
    robustness_figure_executor_code,
    robustness_figure_executor_owns_step,
)
from .adjusted_association_figure_executor import (
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
    adjusted_association_figure_executor_code,
    adjusted_association_figure_executor_owns_step,
)
from .audit_panel_executor import (
    audit_panel_executor_code,
    audit_panel_executor_owns_step,
)
from .cohort_flow_figure_executor import (
    COHORT_FLOW_INPUT,
    cohort_flow_figure_executor_code,
    cohort_flow_figure_executor_owns_step,
)
from .composite_descriptive_figure_executor import (
    composite_descriptive_figure_consumed_input_keys,
    composite_descriptive_figure_executor_code,
    composite_descriptive_figure_executor_owns_step,
)
from .landmark_association_figure_executor import (
    landmark_association_figure_executor_code,
    landmark_association_figure_executor_owns_step,
)
from .descriptive_result_figure_executor import (
    descriptive_result_figure_executor_code,
    descriptive_result_figure_executor_owns_step,
)
from .descriptive_distribution_executor import (
    DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
    descriptive_distribution_executor_code,
    descriptive_distribution_executor_owns_step,
)
from .descriptive_association_executor import (
    DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
    descriptive_association_executor_code,
    descriptive_association_executor_owns_step,
)
from .prevalence_outcome_figure_executor import (
    PREVALENCE_OUTCOME_FIGURE_INPUT,
    prevalence_outcome_figure_executor_code,
    prevalence_outcome_figure_executor_owns_step,
)
from .prevalence_mortality_figure_executor import (
    PREVALENCE_MORTALITY_FIGURE_INPUTS,
    prevalence_mortality_figure_executor_code,
    prevalence_mortality_figure_executor_owns_step,
)
from .table_one_executor import table_one_executor_code, table_one_executor_owns_step
from .survival_primary_executor import (
    SURVIVAL_PRIMARY_ANALYSIS_KIND,
    survival_primary_executor_code,
    survival_primary_executor_verdict,
)
from .trajectory_stability_executor import (
    STABILITY_EXECUTOR_INPUTS,
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)
from .trajectory_scientific_candidate_executor import (
    SCIENTIFIC_CANDIDATE_INPUTS,
    trajectory_scientific_candidate_executor_code,
    trajectory_scientific_candidate_executor_owns_step,
)
from .trajectory_scientific_representation_executor import (
    trajectory_scientific_representation_executor_code,
    trajectory_scientific_representation_executor_owns_step,
)
from .typed_input_binding import sole_typed_cohort_input
from .landmark_spline_executor import (
    LANDMARK_SPLINE_ANALYSIS_KIND,
    landmark_spline_executor_code,
    landmark_spline_executor_owns_step,
)
from .landmark_spline_robustness_executor import (
    LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
    landmark_spline_robustness_executor_code,
    landmark_spline_robustness_executor_owns_step,
)
from .source_feasibility_executor import (
    SOURCE_FEASIBILITY_ANALYSIS_KIND,
    source_feasibility_executor_code,
    source_feasibility_executor_owns_step,
)
from .feasibility_protocol_executor import (
    FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
    feasibility_protocol_consumed_input_keys,
    feasibility_protocol_executor_code,
    feasibility_protocol_executor_owns_step,
)
from .prediction_model_executor import (
    PREDICTION_MODEL_ANALYSIS_KIND,
    prediction_model_consumed_input_keys,
    prediction_model_executor_code,
    prediction_model_executor_owns_step,
)
from .prediction_figure_executor import (
    PREDICTION_COMPOSITE_FIGURE_INPUTS,
    PREDICTION_FIGURE_ANALYSIS_KIND,
    prediction_figure_executor_code,
    prediction_figure_executor_owns_step,
)


def _consumed_typed_cohort_inputs(step: AnalysisStep) -> tuple[str, ...]:
    """The typed cohort input this step's owner will actually read.

    Read from the published vocabulary rather than matched by prefix here.
    ``sole_typed_cohort_input`` is where "which keys name the closed cohort
    product" is decided, and every owner's ``owns_step`` already routes through
    it -- so a branch that spells the rule out again is asserting the owner
    reads something the owner does not.

    That is not hypothetical.  Four branches below each carried the same
    hand-written match on ``cohort:`` or exactly ``artifact:analysis_cohort``,
    blind to ``table:analysis_cohort`` and ``dataset:analysis_cohort``.  The key
    it missed became no entry in ``consumed_input_keys``, so the host stamped no
    input-binding receipt, so ``step_summary_integrity`` reported the step had
    not accounted for an input the host itself had resolved -- and the host then
    dispatched a contract repair against its OWN rendered code.  On 2026-08-01
    that repair inserted a field the host's own spec model forbids, and the E1
    distribution step died on a defect no model authored.

    Measured over 3,170 recorded plan steps: the hand-written match disagreed
    with the published reader on 145, missing ``dataset:analysis_cohort`` 60
    times and ``table:analysis_cohort`` 18 times.
    """

    typed_cohort_input = sole_typed_cohort_input(step)
    return (typed_cohort_input,) if typed_cohort_input else ()


__all__ = [
    "StandardExecutorCandidate",
    "StandardExecutorSelection",
    "select_standard_executor",
]


@dataclass(frozen=True, slots=True)
class StandardExecutorSelection:
    """One deterministic implementation of already-fixed Planner science."""

    analysis_kind: str
    selection_reason: str
    progress_message: str
    code: str
    consumed_input_keys: tuple[str, ...]
    # Rendering-only host adapters already embody the reviewed scientific
    # result and figure contract.  Visual QA may reject their output, but must
    # never hand their source to a model for an unauthorised rewrite.  This
    # grants the host-code digest/no-rewrite policy only; repair-registry
    # renderers separately carry the legacy repair-id, parent-snapshot and
    # product-slot receipt bundle.
    host_sealed_renderer: bool = False


@dataclass(frozen=True, slots=True)
class StandardExecutorCandidate:
    """What one deterministic owner answered, recorded by the decider itself.

    A diagnostic that re-derives ownership from the same predicates is a second
    registry: it cannot see the extra gates this function applies after a
    contract matches, so it eventually reports an owner the selector declined.
    The trace is therefore emitted here, by the code that actually decides.
    """

    analysis_kind: str
    contract_matches: bool
    outcome: str  # "selected" | "declined_receipt_required" | "contract_declined"
    #: Non-empty only when the owner declined *solely* because the Planner left
    #: a field it owns undeclared -- i.e. this step is one declaration away from
    #: a deterministic result.  A plain ``contract_declined`` cannot be read that
    #: way, which is why 54 of 59 real primary-model steps went to the coder with
    #: nobody able to say they need not have.
    missing_declarations: tuple[str, ...] = ()
    decline_reason: str = ""


def select_standard_executor(
    step: AnalysisStep,
    *,
    plan: AnalysisPlan,
    plausibility_scope: FlagOnlyPlausibilityScope | None = None,
    resolved_bindings: Mapping[str, Any] | None = None,
    trajectory_scientific_runtime_authority: Mapping[str, Any] | None = None,
    current_case_scientific_runtime_authority: Mapping[str, Any] | None = None,
    scientific_runtime_projection_sha256: str | None = None,
    trace: list[StandardExecutorCandidate] | None = None,
) -> StandardExecutorSelection | None:
    """Select by exact typed contract, never by prose or benchmark identity.

    ``resolved_bindings`` is the host's own typed-input binding map for this
    step.  An executor whose readable schema is fixed by the *producing* step
    rather than by the Planner's product name uses it to confirm the bound
    product contract before claiming the step; without it such an executor
    declines and the ordinary coder path runs.

    ``trace``, when supplied, receives one :class:`StandardExecutorCandidate`
    per owner consulted, in consultation order, recording what this function
    actually concluded.  Reporting reads that; it must not re-run the
    predicates itself.
    """

    if plausibility_scope is not None:
        plausibility_scope.require_step(step.step_id)
    receipt_required = bool(
        plausibility_scope is not None and plausibility_scope.expected_columns
    )

    def _note(
        analysis_kind: str,
        contract_matches: bool,
        outcome: str,
        *,
        missing_declarations: tuple[str, ...] = (),
        decline_reason: str = "",
    ) -> None:
        if trace is not None:
            trace.append(
                StandardExecutorCandidate(
                    analysis_kind=analysis_kind,
                    contract_matches=contract_matches,
                    outcome=outcome,
                    missing_declarations=missing_declarations,
                    decline_reason=decline_reason,
                )
            )

    def _missed(analysis_kind: str) -> None:
        _note(analysis_kind, False, "contract_declined")

    def _declined(verdict: OwnershipVerdict) -> None:
        """Record a typed decline, keeping the two kinds apart.

        ``contract_matches`` stays False either way -- no owner ran -- but an
        incomplete declaration carries the fields whose absence is the only
        reason, so the pre-registration gate can refuse the plan instead of
        the host silently substituting the coder for an owner it already has.
        """

        _note(
            verdict.analysis_kind,
            False,
            "contract_declined",
            missing_declarations=verdict.missing_declarations,
            decline_reason=verdict.reason,
        )

    def _receipt_declined(analysis_kind: str) -> None:
        _note(analysis_kind, True, "declined_receipt_required")

    def _selected(
        selection: StandardExecutorSelection,
        owner_key: str | None = None,
    ) -> StandardExecutorSelection:
        # ``owner_key`` keeps one stable name per owner in the trace.  The
        # missingness owner resolves to one of four ``analysis_kind`` variants
        # once it claims, and a trace that named it differently depending on
        # whether it claimed could not be read as a list of consulted owners.
        _note(owner_key or selection.analysis_kind, True, "selected")
        return selection

    if current_case_scientific_runtime_authority is not None:
        sealed_current = load_current_case_scientific_runtime_authority(
            current_case_scientific_runtime_authority
        )
        projection_digest = str(scientific_runtime_projection_sha256 or "")
        if isinstance(sealed_current, AssociationModelGridRuntimeAuthority):
            if association_model_grid_executor_owns_step(
                step,
                plan=plan,
                authority=sealed_current,
            ):
                return _selected(
                    StandardExecutorSelection(
                        analysis_kind=ASSOCIATION_MODEL_GRID_ANALYSIS_KIND,
                        selection_reason=(
                            "signed_association_model_grid_contract_preflight"
                        ),
                        progress_message=(
                            "Using verified adjusted-association model-grid adapter"
                        ),
                        code=association_model_grid_executor_code(
                            step,
                            plan=plan,
                            authority=sealed_current,
                            runtime_projection_sha256=projection_digest,
                            plausibility_scope=plausibility_scope,
                        ),
                        consumed_input_keys=(
                            sealed_current.cohort_product,
                            sealed_current.parent_product,
                        ),
                    )
                )
            _missed(ASSOCIATION_MODEL_GRID_ANALYSIS_KIND)
        elif isinstance(sealed_current, LandmarkSplineRuntimeAuthority):
            if landmark_spline_executor_owns_step(
                step,
                plan=plan,
                authority=sealed_current,
            ):
                return _selected(
                    StandardExecutorSelection(
                        analysis_kind=LANDMARK_SPLINE_ANALYSIS_KIND,
                        selection_reason=("signed_landmark_spline_contract_preflight"),
                        progress_message=("Using signed landmark spline executor"),
                        code=landmark_spline_executor_code(
                            step,
                            authority=sealed_current,
                            runtime_projection_sha256=projection_digest,
                            plausibility_scope=plausibility_scope,
                        ),
                        consumed_input_keys=_consumed_typed_cohort_inputs(step),
                    )
                )
            _missed(LANDMARK_SPLINE_ANALYSIS_KIND)
            if landmark_spline_robustness_executor_owns_step(
                step,
                plan=plan,
                authority=sealed_current,
            ):
                return _selected(
                    StandardExecutorSelection(
                        analysis_kind=LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND,
                        selection_reason=(
                            "signed_landmark_spline_robustness_projection_preflight"
                        ),
                        progress_message=(
                            "Using signed landmark spline robustness projection"
                        ),
                        code=landmark_spline_robustness_executor_code(
                            step,
                            authority=sealed_current,
                            runtime_projection_sha256=projection_digest,
                        ),
                        consumed_input_keys=(
                            *(value for value in step.inputs if ":" in value),
                        ),
                    )
                )
            _missed(LANDMARK_SPLINE_ROBUSTNESS_ANALYSIS_KIND)
        elif isinstance(sealed_current, SourceFeasibilityRuntimeAuthority):
            if source_feasibility_executor_owns_step(
                step,
                plan=plan,
                authority=sealed_current,
            ):
                return _selected(
                    StandardExecutorSelection(
                        analysis_kind=SOURCE_FEASIBILITY_ANALYSIS_KIND,
                        selection_reason=(
                            "signed_source_feasibility_contract_preflight"
                        ),
                        progress_message=("Using signed source-feasibility executor"),
                        code=source_feasibility_executor_code(
                            authority=sealed_current,
                            runtime_projection_sha256=projection_digest,
                        ),
                        consumed_input_keys=(),
                    )
                )
            _missed(SOURCE_FEASIBILITY_ANALYSIS_KIND)

    if feasibility_protocol_executor_owns_step(step):
        if receipt_required:
            _receipt_declined(FEASIBILITY_PROTOCOL_ANALYSIS_KIND)
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind=FEASIBILITY_PROTOCOL_ANALYSIS_KIND,
                selection_reason="planner_declared_feasibility_protocol",
                progress_message=(
                    "Recording the Planner-declared non-executable protocol"
                ),
                code=feasibility_protocol_executor_code(step),
                consumed_input_keys=feasibility_protocol_consumed_input_keys(step),
            )
        )
    _missed(FEASIBILITY_PROTOCOL_ANALYSIS_KIND)

    if prediction_model_executor_owns_step(step):
        return _selected(
            StandardExecutorSelection(
                analysis_kind=PREDICTION_MODEL_ANALYSIS_KIND,
                selection_reason="typed_static_prediction_contract_preflight",
                progress_message="Using deterministic static prediction adapter",
                code=prediction_model_executor_code(step),
                consumed_input_keys=prediction_model_consumed_input_keys(step),
            )
        )
    _missed(PREDICTION_MODEL_ANALYSIS_KIND)

    if cohort_summary_executor_owns_step(step):
        # This executor emits the flag-only receipt itself, so a receipt
        # obligation no longer sends a step the host can compute exactly to
        # the stochastic Coder.
        typed_cohort_inputs = _consumed_typed_cohort_inputs(step)
        return _selected(
            StandardExecutorSelection(
                analysis_kind="descriptive_cohort_summary",
                selection_reason="cohort_summary_contract_preflight",
                progress_message="Using planner-scoped cohort summary executor",
                code=cohort_summary_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
    _missed("descriptive_cohort_summary")
    if ordered_stratified_executor_owns_step(step, plan=plan):
        return _selected(
            StandardExecutorSelection(
                analysis_kind=ORDERED_STRATIFIED_ANALYSIS_KIND,
                selection_reason="typed_ordered_stratified_contract_preflight",
                progress_message="Using deterministic ordered-trend adapter",
                code=ordered_stratified_executor_code(step, plan=plan),
                consumed_input_keys=ordered_stratified_consumed_input_keys(step),
            )
        )
    _missed(ORDERED_STRATIFIED_ANALYSIS_KIND)
    if exposure_outcome_distribution_executor_owns_step(step):
        typed_cohort_inputs = _consumed_typed_cohort_inputs(step)
        return _selected(
            StandardExecutorSelection(
                analysis_kind="exposure_outcome_distribution",
                selection_reason="exposure_outcome_distribution_contract_preflight",
                progress_message=(
                    "Using planner-declared exposure/outcome distribution executor"
                ),
                code=exposure_outcome_distribution_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
    # Not a bare miss. Measured over every recorded run, 28 steps promise this
    # owner's science under the Planner's own product label, declare its spec 0
    # times, and were never asked -- while 29 of the 33 steps that DO declare it
    # are claimed and pass. Declining silently is what let an 82 %-passing step
    # emit a table with a different shape every run, killing every figure over
    # it (14 recorded, 0 ok). The verdict reports the gap where the Planner can
    # still close it; it stays quiet on any step this owner could not compute
    # however it were declared.
    distribution_declaration_verdict = (
        exposure_outcome_distribution_declaration_verdict(step)
    )
    if distribution_declaration_verdict.missing_declarations:
        _declined(distribution_declaration_verdict)
    else:
        _missed("exposure_outcome_distribution")
    if exposure_outcome_distribution_figure_owns_step(step):
        if receipt_required:
            _receipt_declined("exposure_outcome_distribution_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="exposure_outcome_distribution_figure",
                selection_reason=(
                    "exposure_outcome_distribution_figure_contract_preflight"
                ),
                progress_message=(
                    "Using planner-scoped exposure/outcome distribution renderer"
                ),
                code=exposure_outcome_distribution_figure_code(
                    step,
                    display_labels=plan.display_labels,
                ),
                consumed_input_keys=(EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,),
                host_sealed_renderer=True,
            )
        )
    distribution_figure_verdict = (
        exposure_outcome_distribution_figure_declaration_verdict(step)
    )
    if distribution_figure_verdict.missing_declarations:
        _declined(distribution_figure_verdict)
    else:
        _missed("exposure_outcome_distribution_figure")
    if prevalence_outcome_figure_executor_owns_step(step):
        if receipt_required:
            _receipt_declined("prevalence_outcome_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="prevalence_outcome_figure",
                selection_reason="prevalence_outcome_figure_contract_preflight",
                progress_message=(
                    "Using planner-scoped prevalence/outcome figure executor"
                ),
                code=prevalence_outcome_figure_executor_code(step),
                consumed_input_keys=(PREVALENCE_OUTCOME_FIGURE_INPUT,),
                host_sealed_renderer=True,
            )
        )
    _missed("prevalence_outcome_figure")
    if robustness_figure_executor_owns_step(step, resolved_bindings=resolved_bindings):
        if receipt_required:
            _receipt_declined("robustness_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="robustness_figure",
                selection_reason="robustness_figure_contract_preflight",
                progress_message=("Using planner-scoped robustness figure executor"),
                code=robustness_figure_executor_code(step),
                # Every optional parent this renderer READS, not only the
                # matrix it plots.  The host stamps an input-binding receipt
                # per consumed key, so declaring one while reading four left
                # three inputs unstamped and ``step_summary_integrity``
                # refused the step for incomplete coverage -- naming exactly
                # the three the renderer had just drawn from.
                consumed_input_keys=robustness_figure_consumed_input_keys(
                    resolved_bindings
                ),
                host_sealed_renderer=True,
            )
        )
    _missed("robustness_figure")
    if adjusted_association_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined("adjusted_association_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="adjusted_association_figure",
                selection_reason="adjusted_association_figure_contract_preflight",
                progress_message=(
                    "Using planner-scoped adjusted association figure executor"
                ),
                code=adjusted_association_figure_executor_code(step),
                consumed_input_keys=(ADJUSTED_ASSOCIATION_FIGURE_INPUT,),
                host_sealed_renderer=True,
            )
        )
    _missed("adjusted_association_figure")
    if cohort_flow_figure_executor_owns_step(step, resolved_bindings=resolved_bindings):
        if receipt_required:
            _receipt_declined("cohort_flow_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="cohort_flow_figure",
                selection_reason="cohort_flow_figure_contract_preflight",
                progress_message="Using digest-bound cohort-flow renderer",
                code=cohort_flow_figure_executor_code(step),
                consumed_input_keys=(COHORT_FLOW_INPUT,),
                host_sealed_renderer=True,
            )
        )
    _missed("cohort_flow_figure")
    if landmark_association_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined("landmark_association_composite_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="landmark_association_composite_figure",
                selection_reason=(
                    "landmark_association_composite_figure_contract_preflight"
                ),
                progress_message=(
                    "Using digest-bound landmark association composite renderer"
                ),
                code=landmark_association_figure_executor_code(step),
                consumed_input_keys=tuple(str(value) for value in step.inputs),
                host_sealed_renderer=True,
            )
        )
    _missed("landmark_association_composite_figure")
    if prediction_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined(PREDICTION_FIGURE_ANALYSIS_KIND)
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind=PREDICTION_FIGURE_ANALYSIS_KIND,
                selection_reason="static_prediction_figure_contract_preflight",
                progress_message="Using source-bound static prediction renderer",
                code=prediction_figure_executor_code(step),
                consumed_input_keys=PREDICTION_COMPOSITE_FIGURE_INPUTS,
                host_sealed_renderer=True,
            )
        )
    _missed(PREDICTION_FIGURE_ANALYSIS_KIND)
    if composite_descriptive_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined("composite_descriptive_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="composite_descriptive_figure",
                selection_reason="composite_descriptive_figure_contract_preflight",
                progress_message=("Using digest-bound composite descriptive renderer"),
                code=composite_descriptive_figure_executor_code(
                    step,
                    display_labels=plan.display_labels,
                ),
                consumed_input_keys=(
                    composite_descriptive_figure_consumed_input_keys(step)
                ),
                host_sealed_renderer=True,
            )
        )
    _missed("composite_descriptive_figure")
    if descriptive_result_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined("descriptive_result_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="descriptive_result_figure",
                selection_reason="descriptive_result_figure_contract_preflight",
                progress_message=("Using digest-bound descriptive result renderer"),
                code=descriptive_result_figure_executor_code(step),
                consumed_input_keys=(step.inputs[0],),
                host_sealed_renderer=True,
            )
        )
    _missed("descriptive_result_figure")
    if prevalence_mortality_figure_executor_owns_step(step):
        if receipt_required:
            _receipt_declined("prevalence_mortality_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="prevalence_mortality_figure",
                selection_reason="prevalence_mortality_figure_contract_preflight",
                progress_message=(
                    "Using planner-scoped prevalence/mortality figure executor"
                ),
                code=prevalence_mortality_figure_executor_code(
                    step,
                    display_labels=plan.display_labels,
                ),
                consumed_input_keys=PREVALENCE_MORTALITY_FIGURE_INPUTS,
                host_sealed_renderer=True,
            )
        )
    _missed("prevalence_mortality_figure")
    if measurement_missingness_figure_executor_owns_step(
        step, plan=plan, resolved_bindings=resolved_bindings
    ):
        if receipt_required:
            _receipt_declined("measurement_missingness_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="measurement_missingness_figure",
                selection_reason="measurement_missingness_figure_contract_preflight",
                progress_message=(
                    "Using digest-bound measurement-missingness figure renderer"
                ),
                code=measurement_missingness_figure_executor_code(step, plan=plan),
                consumed_input_keys=(str(step.inputs[0]),),
                host_sealed_renderer=True,
            )
        )
    _missed("measurement_missingness_figure")
    if missingness_measurement_figure_executor_owns_step(
        step,
        plan=plan,
        resolved_bindings=resolved_bindings,
    ):
        if receipt_required:
            _receipt_declined("missingness_measurement_figure")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="missingness_measurement_figure",
                selection_reason=("missingness_measurement_figure_contract_preflight"),
                progress_message=(
                    "Using planner-scoped missingness/measurement figure executor"
                ),
                code=missingness_measurement_figure_executor_code(step, plan=plan),
                consumed_input_keys=tuple(str(value) for value in step.inputs),
                host_sealed_renderer=True,
            )
        )
    # Not a bare miss when one string is the only thing between the plan and a
    # deterministic figure. Measured over every recorded run, 9 figure steps
    # name one of the two audit tables while their own parent produces both;
    # the renderer sits idle and the Coder writes a source-data table whose
    # columns cannot be traced to the parent they came from, which is exactly
    # how m1's 09_missingness_audit_figure died. The verdict stays quiet on the
    # 31 steps whose sibling table no step produces: closing those means asking
    # a parent for a different analysis, which is not this owner's call.
    missingness_declaration_verdict = (
        missingness_measurement_figure_declaration_verdict(step, plan=plan)
    )
    if missingness_declaration_verdict.missing_declarations:
        _declined(missingness_declaration_verdict)
    else:
        _missed("missingness_measurement_figure")
    if audit_panel_executor_owns_step(step):
        if receipt_required:
            _receipt_declined("audit_panel")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="audit_panel",
                selection_reason="framework_audit_panel_contract_preflight",
                progress_message="Using deterministic audit-panel renderer",
                code=audit_panel_executor_code(step),
                consumed_input_keys=(),
                host_sealed_renderer=True,
            )
        )
    _missed("audit_panel")
    if descriptive_distribution_executor_owns_step(step):
        return _selected(
            StandardExecutorSelection(
                analysis_kind=DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
                selection_reason="grouped_descriptive_distribution_contract_preflight",
                progress_message=(
                    "Using planner-scoped grouped descriptive distribution executor"
                ),
                code=descriptive_distribution_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=_consumed_typed_cohort_inputs(step),
            )
        )
    _missed(DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND)
    if descriptive_association_executor_owns_step(step):
        return _selected(
            StandardExecutorSelection(
                analysis_kind=DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
                selection_reason="descriptive_association_contract_preflight",
                progress_message=(
                    "Using planner-scoped descriptive association executor"
                ),
                code=descriptive_association_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=_consumed_typed_cohort_inputs(step),
            )
        )
    _missed(DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND)
    if table_one_executor_owns_step(step):
        typed_cohort_inputs = _consumed_typed_cohort_inputs(step)
        return _selected(
            StandardExecutorSelection(
                analysis_kind="grouped_table_one",
                selection_reason="table_one_spec_preflight",
                progress_message="Using planner-specified grouped Table 1 executor",
                code=table_one_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
    _missed("grouped_table_one")
    if missingness_audit_executor_owns_step(step):
        source_availability = source_availability_audit_executor_owns_step(step)
        compact_measurement = is_compact_missingness_measurement_contract(
            step.method,
            step.expected_outputs,
        )
        measurement_bias = is_measurement_bias_audit_contract(
            step.method,
            step.expected_outputs,
        )
        complete_case = is_missingness_complete_case_contract(
            step.method,
            step.expected_outputs,
        )
        typed_cohort_input = missingness_audit_cohort_input_key(step)
        # Each named contract keeps its own kind; anything claimed only by the
        # capability rule is named for what it is.  The complete-case label used
        # to be the trailing ``else``, so once ownership widened it would have
        # been stamped on measurement-process audits that are not complete-case
        # analyses at all -- a claim the record could not be read back from.
        if source_availability:
            analysis_kind = "missingness_source_availability_audit"
            selection_reason = "missingness_source_availability_contract_preflight"
        elif measurement_bias:
            analysis_kind = "measurement_bias_audit"
            selection_reason = "measurement_bias_contract_preflight"
        elif compact_measurement:
            analysis_kind = "missingness_measurement_audit"
            selection_reason = "missingness_measurement_contract_preflight"
        elif complete_case:
            analysis_kind = "missingness_complete_case_audit"
            selection_reason = "missingness_complete_case_contract_preflight"
        else:
            analysis_kind = "declared_missingness_audit_products"
            selection_reason = "missingness_audit_product_capability_preflight"
        selection = StandardExecutorSelection(
            analysis_kind=analysis_kind,
            selection_reason=selection_reason,
            progress_message="Using planner-specified missingness audit executor",
            code=missingness_measurement_audit_code(
                step,
                plausibility_scope=plausibility_scope,
            ),
            consumed_input_keys=(
                (typed_cohort_input,) if typed_cohort_input is not None else ()
            ),
        )
        return _selected(selection, "missingness_audit")
    _missed("missingness_audit")
    if trajectory_scientific_representation_executor_owns_step(
        step,
        plan=plan,
        authority=trajectory_scientific_runtime_authority,
    ):
        return _selected(
            StandardExecutorSelection(
                analysis_kind="trajectory_signed_representation",
                selection_reason="signed_trajectory_representation_authority",
                progress_message="Using signed trajectory representation executor",
                code=trajectory_scientific_representation_executor_code(
                    authority=trajectory_scientific_runtime_authority,
                    runtime_projection_sha256=str(
                        scientific_runtime_projection_sha256 or ""
                    ),
                ),
                consumed_input_keys=(),
            )
        )
    _missed("trajectory_signed_representation")
    if trajectory_scientific_candidate_executor_owns_step(
        step,
        plan=plan,
        authority=trajectory_scientific_runtime_authority,
    ):
        return _selected(
            StandardExecutorSelection(
                analysis_kind="trajectory_signed_candidate_selection",
                selection_reason="signed_trajectory_candidate_authority",
                progress_message="Using signed trajectory candidate selector",
                code=trajectory_scientific_candidate_executor_code(
                    authority=trajectory_scientific_runtime_authority,
                    runtime_projection_sha256=str(
                        scientific_runtime_projection_sha256 or ""
                    ),
                ),
                consumed_input_keys=tuple(sorted(SCIENTIFIC_CANDIDATE_INPUTS)),
            )
        )
    _missed("trajectory_signed_candidate_selection")
    if trajectory_stability_executor_owns_step(step, plan=plan):
        if receipt_required:
            _receipt_declined("trajectory_cluster_stability")
            return None
        return _selected(
            StandardExecutorSelection(
                analysis_kind="trajectory_cluster_stability",
                selection_reason="trajectory_stability_spec_preflight",
                progress_message=(
                    "Using planner-specified trajectory stability executor"
                ),
                code=trajectory_stability_executor_code(
                    step,
                    plan=plan,
                    scientific_runtime_authority=(
                        trajectory_scientific_runtime_authority
                    ),
                    runtime_projection_sha256=(scientific_runtime_projection_sha256),
                ),
                consumed_input_keys=tuple(sorted(STABILITY_EXECUTOR_INPUTS)),
            )
        )
    _missed("trajectory_cluster_stability")
    survival_verdict = survival_primary_executor_verdict(step)
    if survival_verdict.claimed:
        survival_requirement = step.family_primary_result_requirement
        assert survival_requirement is not None
        typed_cohort_inputs = (str(survival_requirement.input_product),)
        return _selected(
            StandardExecutorSelection(
                analysis_kind=SURVIVAL_PRIMARY_ANALYSIS_KIND,
                selection_reason="survival_primary_contract_preflight",
                progress_message="Using planner-declared primary Cox executor",
                code=survival_primary_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
    _declined(survival_verdict)
    adjusted_association_verdict = adjusted_association_executor_verdict(step)
    if adjusted_association_verdict.claimed:
        # This owner renders the flag-only receipt itself, like the cohort
        # summary and Table 1 owners, so a receipt obligation does not send the
        # study's primary estimate to the stochastic coder.
        typed_cohort_inputs = _consumed_typed_cohort_inputs(step)
        return _selected(
            StandardExecutorSelection(
                analysis_kind=ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
                selection_reason="adjusted_association_model_contract_preflight",
                progress_message=(
                    "Using planner-declared adjusted-association executor"
                ),
                code=adjusted_association_executor_code(
                    step,
                    plausibility_scope=plausibility_scope,
                ),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
    _declined(adjusted_association_verdict)
    # Consulted for its declaration gap only, and deliberately never claimed
    # here.  The robustness replay is already reachable as a preflight
    # substitute *before* the Coder is asked, and no recorded step carries an
    # emittable spec -- so a claim path in this function could not be exercised
    # by any real plan, which is the opposite of what deterministic ownership is
    # for.  What the plan-time gate needs is the gap: measured 2026-07-30, 20
    # recorded steps promise a product this replay is the registered emitter of
    # and declare no spec at all, so the Coder invents the specification grid.
    # Moving the routing into this function is a separate, characterised change
    # for when a real run first produces an emittable spec.
    robustness_declaration_verdict = robustness_replay_declaration_verdict(step)
    if robustness_declaration_verdict.missing_declarations:
        _declined(robustness_declaration_verdict)
    else:
        _missed(ROBUSTNESS_REPLAY_ANALYSIS_KIND)
    return None
