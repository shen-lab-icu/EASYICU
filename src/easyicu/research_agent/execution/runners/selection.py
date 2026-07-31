"""Select trusted executors for complete Planner-owned contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...authority.plausibility import FlagOnlyPlausibilityScope
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
from .exposure_outcome_distribution_render import (
    EXPOSURE_OUTCOME_DISTRIBUTION_FIGURE_INPUT,
    exposure_outcome_distribution_figure_code,
    exposure_outcome_distribution_figure_owns_step,
)
from .exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_code,
    exposure_outcome_distribution_executor_owns_step,
)
from .cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
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
    MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
    missingness_measurement_figure_executor_code,
    missingness_measurement_figure_executor_owns_step,
)
from .robustness_figure_executor import (
    ROBUSTNESS_FIGURE_INPUT,
    robustness_figure_executor_code,
    robustness_figure_executor_owns_step,
)
from .adjusted_association_figure_executor import (
    ADJUSTED_ASSOCIATION_FIGURE_INPUT,
    adjusted_association_figure_executor_code,
    adjusted_association_figure_executor_owns_step,
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
from .trajectory_stability_executor import (
    STABILITY_EXECUTOR_INPUTS,
    trajectory_stability_executor_code,
    trajectory_stability_executor_owns_step,
)

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

    if cohort_summary_executor_owns_step(step):
        # This executor emits the flag-only receipt itself, so a receipt
        # obligation no longer sends a step the host can compute exactly to
        # the stochastic Coder.
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
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
    if exposure_outcome_distribution_executor_owns_step(step):
        if receipt_required:
            # This owner emits no flag-only receipt, so it declines rather than
            # claiming a step whose obligation it cannot discharge.
            _receipt_declined("exposure_outcome_distribution")
            return None
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
        return _selected(
            StandardExecutorSelection(
                analysis_kind="exposure_outcome_distribution",
                selection_reason="exposure_outcome_distribution_contract_preflight",
                progress_message=(
                    "Using planner-declared exposure/outcome distribution executor"
                ),
                code=exposure_outcome_distribution_executor_code(step),
                consumed_input_keys=typed_cohort_inputs,
            )
        )
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
            )
        )
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
                consumed_input_keys=(ROBUSTNESS_FIGURE_INPUT,),
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
            )
        )
    _missed("adjusted_association_figure")
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
            )
        )
    _missed("prevalence_mortality_figure")
    if missingness_measurement_figure_executor_owns_step(
        step, resolved_bindings=resolved_bindings
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
                code=missingness_measurement_figure_executor_code(step),
                consumed_input_keys=MISSINGNESS_MEASUREMENT_FIGURE_INPUTS,
            )
        )
    _missed("missingness_measurement_figure")
    if table_one_executor_owns_step(step):
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
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
                code=trajectory_stability_executor_code(step, plan=plan),
                consumed_input_keys=tuple(sorted(STABILITY_EXECUTOR_INPUTS)),
            )
        )
    _missed("trajectory_cluster_stability")
    adjusted_association_verdict = adjusted_association_executor_verdict(step)
    if adjusted_association_verdict.claimed:
        # This owner renders the flag-only receipt itself, like the cohort
        # summary and Table 1 owners, so a receipt obligation does not send the
        # study's primary estimate to the stochastic coder.
        typed_cohort_inputs = tuple(
            str(value or "").strip()
            for value in step.inputs
            if str(value or "").strip().startswith("cohort:")
            or str(value or "").strip() == "artifact:analysis_cohort"
        )
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
