"""Deterministic findings for one executed analysis step."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..contracts.declared_product import declared_product_contract_findings, is_failed_step_status
from ..contracts.ordered_stratified import ordered_stratified_structure_findings
from ..contracts.step_families import (
    _clustering_contract_applies,
    _cohort_change_contract_applies,
    _effect_contract_applies,
    _output_declares_auxiliary_log,
    _prediction_contract_applies,
    effect_output_authorized,
)
from ..contracts.table_one import table_one_output_findings
from ..planning.figure_plan_mutation import _effect_figure_source_authorized
from ..planning.figure_step_contract import _output_declares_figure
from ..scalar_utils import _first_numeric_scalar_with_key_fragment, _first_present_scalar, _flatten_scalar_dict
from ..schema import AnalysisStep, ResearchContext, ValidationFinding
from .step_result_evidence import (
    _cluster_count_from_summary,
    _cluster_selection_evidence_key,
    _clustering_evidence_from_completed_records,
    _prediction_auroc_from_completed_records,
    _prediction_calibration_from_completed_records,
    _primary_effect_from_summary,
    _problematic_metric_keys,
)

def _step_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    context: Optional[ResearchContext] = None,
    completed_step_records: Optional[Sequence[Dict[str, Any]]] = None,
    resolved_input_bindings: Optional[Mapping[str, Mapping[str, Any]]] = None,
    step_record: Optional[Mapping[str, Any]] = None,
    effect_output_is_authorized: Optional[bool] = None,
    out_dir: Optional[Path] = None,
    trajectory_role_contract_applies: bool = True,
) -> List[ValidationFinding]:
    if not isinstance(step_summary, dict) or not step_summary:
        return [
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} did not produce a readable step_summary.json, "
                    "so required outputs cannot be verified."
                ),
                detail={"step_id": step.step_id},
            )
        ]

    findings: List[ValidationFinding] = []
    reported_status = str(step_summary.get("status") or "").strip().lower()
    if is_failed_step_status(reported_status):
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} reported status={reported_status!r} "
                    "inside step_summary.json and cannot be recorded as a "
                    "successful completed step."
                ),
                detail={
                    "step_id": step.step_id,
                    "reported_status": reported_status,
                    "blocking_reason": step_summary.get("blocking_reason"),
                    "error": step_summary.get("error"),
                },
            )
        )
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    intent = (step.intent or "").lower()

    # Closed, method-specific declaration for an agent-authored ordered-group
    # descriptive step.  The contract records the agent's variable/order/
    # denominator decisions; a separate cohort replay verifies the numbers.
    findings.extend(
        ordered_stratified_structure_findings(
            step=step,
            step_summary=step_summary,
        )
    )
    findings.extend(table_one_output_findings(step=step, out_dir=out_dir))
    # Figure-only follow-up steps (created by ``_split_table_and_figure_outputs_in_plan``)
    # inherit the parent's step_id with a ``_figure`` suffix, e.g.
    # ``04_primary_association_figure`` / ``01_model_training_figure``. Their
    # expected_outputs contain *only* figure items — the analytic payload
    # (table/statistic/etc.) lives in the sibling parent step. Without this guard
    # the substring matches ``primary_association``/``model_training``/``cluster``
    # below would falsely demand effect/prediction/clustering metrics from a
    # render-only step that legitimately has no such fields in its summary.
    figure_only_step = (
        bool(step.expected_outputs)
        and any(_output_declares_figure(out) for out in step.expected_outputs)
        and all(
            _output_declares_figure(out) or _output_declares_auxiliary_log(out)
            for out in step.expected_outputs
        )
    )
    findings.extend(
        declared_product_contract_findings(
            step=step,
            step_summary=step_summary,
            effect_method_authorized=effect_output_authorized(
                step, step_record=step_record
            ) if effect_output_is_authorized is None else effect_output_is_authorized,
            effect_figure_source_authorized=_effect_figure_source_authorized(
                step=step,
                completed_step_records=completed_step_records,
                resolved_input_bindings=resolved_input_bindings,
            ),
            out_dir=out_dir,
            trajectory_role_contract_applies=trajectory_role_contract_applies,
        )
    )
    from ..figures.distribution_availability import (
        distribution_availability_parent_contract_issue,
    )

    distribution_parent_issue = distribution_availability_parent_contract_issue(
        planned_method=step.method,
        parent_out=out_dir,
        parent_summary=step_summary,
        expected_outputs=step.expected_outputs or [],
        planned_inputs=step.inputs or [],
        host_context=context,
    )
    if distribution_parent_issue is not None:
        findings.append(
            ValidationFinding(
                validator="distribution_availability_parent_contract",
                severity="error",
                message=(
                    "The controlled distribution/availability audit did not "
                    "produce the closed parent schema required by its declared "
                    "renderer. Preserve the Planner-selected exposure and write "
                    "the two declared table roles plus their matching summary "
                    "contracts before this step can be successful."
                ),
                detail={
                    "step_id": step.step_id,
                    **distribution_parent_issue,
                },
            )
        )

    # The input parquet is already the locked analysis cohort. A generated
    # downstream QC/model/descriptive script must not relabel itself as a cohort
    # definition/sensitivity step and silently re-run eligibility. Check the
    # plan's own method/id/intent/output contract rather than trusting the
    # generated summary's family (the latter is exactly what can drift).
    cohort_change_authorized = _cohort_change_contract_applies(step)
    summary_family = str(step_summary.get("analysis_family") or "").lower()
    summary_cohort = step_summary.get("cohort_definition")
    summary_claims_cohort_change = summary_family in {
        "cohort_definition",
        "cohort_definition_sensitivity",
        "cohort_sensitivity",
        "definition_sensitivity",
    } or bool(
        isinstance(summary_cohort, dict)
        and summary_cohort.get("current_step_is_cohort_definition_sensitivity")
    )
    if (
        not figure_only_step
        and summary_claims_cohort_change
        and not cohort_change_authorized
    ):
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=(
                    f"Step {step.step_id} is not a cohort-definition or "
                    "alternative-cohort step, but its summary relabels it as "
                    "cohort-definition sensitivity. Treat COHORT_PARQUET as the "
                    "already locked analysis cohort; remove age, length-of-stay, "
                    "identifier, outcome-availability, and other eligibility "
                    "filters from this step."
                ),
                detail={
                    "kind": "unauthorized_cohort_redefinition",
                    "step_id": step.step_id,
                    "planned_method": step.method,
                    "reported_analysis_family": summary_family or None,
                    "reported_current_step_is_cohort_definition_sensitivity": (
                        summary_cohort.get(
                            "current_step_is_cohort_definition_sensitivity"
                        )
                        if isinstance(summary_cohort, dict)
                        else None
                    ),
                },
            )
        )

    def _append_missing(message: str, keys: Sequence[str]) -> None:
        findings.append(
            ValidationFinding(
                validator="step_contract",
                severity="error",
                message=message,
                detail={
                    "step_id": step.step_id,
                    "expected_outputs": list(step.expected_outputs or []),
                    "summary_keys": sorted(step_summary.keys()),
                    "skipped": step_summary.get("skipped"),
                    "error": step_summary.get("error"),
                    "required_keys": list(keys),
                },
            )
        )

    effect_required = not figure_only_step and _effect_contract_applies(step)
    if effect_required:
        effect_value = _primary_effect_from_summary(step_summary)
        if effect_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a primary association "
                    "estimate, but no numeric effect size was recorded."
                ),
                ("estimate", "primary_or", "odds_ratio", "adjusted_or"),
            )

    prediction_required = not figure_only_step and _prediction_contract_applies(step)
    if prediction_required:
        auroc_value = _first_present_scalar(
            step_summary,
            (
                "auroc",
                "statistic:auroc",
                "auc",
                "statistic:auc",
                "held_out_auroc",
                "statistic:held_out_auroc",
                "cv_auroc",
                "statistic:cv_auroc",
                "cv_auroc_mean",
                "statistic:cv_auroc_mean",
                "mean_auroc",
                "auroc_mean",
                "auroc_median",
            ),
        )
        if auroc_value is None:
            auroc_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("auroc", "auc"),
            )
        if auroc_value is None:
            # The discrimination estimate may have been produced and bound by an
            # upstream training step that this (figure/rendering) step renders;
            # mirror the primary-association cross-step fallback so a key-naming
            # mismatch between two steps does not fail the run when the metric is
            # genuinely auditable elsewhere.
            auroc_fallback = _prediction_auroc_from_completed_records(
                completed_step_records,
                current_step_id=str(step.step_id or ""),
            )
            if auroc_fallback is not None:
                source_step_id, _source_auroc = auroc_fallback
                auroc_value = _source_auroc
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own AUROC-style "
                            f"discrimination metric, but the requirement was satisfied "
                            f"by successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if auroc_value is None:
            problematic = _problematic_metric_keys(step_summary, ("auroc", "auc"))
            if problematic:
                keys = ", ".join(str(item["key"]) for item in problematic)
                message = (
                    f"Step {step.step_id} was expected to report AUROC-style "
                    "discrimination. AUROC-like metric keys were present but "
                    f"null/non-finite ({keys}), so the validation model did not "
                    "produce an auditable discrimination estimate."
                )
            else:
                message = (
                    f"Step {step.step_id} was expected to report AUROC-style "
                    "discrimination, but no AUROC metric was recorded."
                )
            _append_missing(
                message,
                ("auroc", "cv_auroc", "mean_auroc", "auroc_median"),
            )
        calibration_value = _first_present_scalar(
            step_summary,
            (
                "brier_score",
                "statistic:brier_score",
                "cv_brier_mean",
                "statistic:cv_brier_mean",
                "brier_mean",
                "held_out_brier",
                "statistic:held_out_brier",
                "brier_median",
                "calibration_slope",
                "statistic:calibration_slope",
                "calibration_slope_median",
                "calibration_intercept",
                "statistic:calibration_intercept",
                "calibration_intercept_median",
            ),
        )
        if calibration_value is None:
            calibration_value = _first_numeric_scalar_with_key_fragment(
                step_summary,
                ("brier", "calibration_slope", "calibration_intercept"),
            )
        if calibration_value is None:
            calibration_fallback = _prediction_calibration_from_completed_records(
                completed_step_records,
                current_step_id=str(step.step_id or ""),
            )
            if calibration_fallback is not None:
                source_step_id, _source_cal = calibration_fallback
                calibration_value = _source_cal
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own calibration/"
                            f"Brier-style metric, but the requirement was satisfied by "
                            f"successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if calibration_value is None:
            problematic = _problematic_metric_keys(
                step_summary,
                ("brier", "calibration_slope", "calibration_intercept"),
            )
            if problematic:
                keys = ", ".join(str(item["key"]) for item in problematic)
                message = (
                    f"Step {step.step_id} was expected to report calibration or "
                    "Brier-style evaluation metrics. Calibration/Brier-like keys "
                    f"were present but null/non-finite ({keys}), so the validation "
                    "model did not produce an auditable calibration estimate."
                )
            else:
                message = (
                    f"Step {step.step_id} was expected to report calibration or "
                    "Brier-style evaluation metrics, but none were recorded."
                )
            _append_missing(
                message,
                (
                    "brier_score",
                    "cv_brier_mean",
                    "held_out_brier",
                    "calibration_slope",
                    "calibration_intercept",
                ),
            )

    # Apply the cluster metric contract only to a method-owned clustering step
    # with declared standard products.  Existing class membership, hospital-level
    # clustering and cluster-robust standard errors are association details, not
    # phenotype-discovery ownership.
    clustering_required = (not figure_only_step) and _clustering_contract_applies(
        method=str(step.method or ""),
        step_id=str(step.step_id or ""),
        intent=str(step.intent or ""),
        expected_outputs=step.expected_outputs or [],
    )
    if clustering_required:
        cluster_count = _cluster_count_from_summary(step_summary)
        selection_key, explicit_manifest_invalid = _cluster_selection_evidence_key(
            step_summary,
            cluster_count=cluster_count,
        )
        if not explicit_manifest_invalid and (
            cluster_count is None or selection_key is None
        ):
            # The clustering estimate may have been produced and bound by a
            # dedicated sibling clustering step that this (figure/rendering or
            # feature-prep) step does not re-register under a recognised key;
            # require both selected cluster count and the agent's native
            # selection/stability evidence from the same successful owner.
            cluster_fallback, sibling_manifest_invalid = (
                _clustering_evidence_from_completed_records(
                    completed_step_records,
                    current_step_id=str(step.step_id or ""),
                )
            )
            if sibling_manifest_invalid:
                explicit_manifest_invalid = True
            elif cluster_fallback is not None:
                source_step_id, source_count, source_selection_key = cluster_fallback
                cluster_count = source_count
                selection_key = source_selection_key
                findings.append(
                    ValidationFinding(
                        validator="step_contract",
                        severity="warning",
                        message=(
                            f"Step {step.step_id} did not record its own "
                            f"cluster count and native selection/stability evidence, "
                            f"but the requirement "
                            f"was satisfied by successful step {source_step_id}."
                        ),
                        detail={
                            "step_id": step.step_id,
                            "fallback_step_id": source_step_id,
                            "expected_outputs": list(step.expected_outputs or []),
                            "summary_keys": sorted(step_summary.keys()),
                        },
                    )
                )
        if cluster_count is None or selection_key is None:
            missing = []
            if cluster_count is None:
                missing.extend(("n_clusters", "cluster_count"))
            if selection_key is None:
                missing.extend(
                    (
                        "cluster_selection",
                        "cluster_stability",
                    )
                )
            _append_missing(
                (
                    f"Step {step.step_id} was expected to report a clustering summary, "
                    "but it did not record both the selected cluster count and an "
                    "agent-declared native selection/stability criterion."
                ),
                tuple(missing),
            )

    # Enforce figure_required when:
    # (a) the intent *explicitly* demands a publication-ready figure, OR
    # (b) the step is figure-only (its expected_outputs are exclusively
    #     figure tokens — usually the child produced by
    #     ``_split_table_and_figure_outputs_in_plan``).
    # For unsplit mixed steps (figure declared alongside table/statistic
    # outputs without an explicit "publication-ready figure" intent), the
    # splitter handles decomposition in production, so we treat the figure
    # output as an optional companion here. This mirrors how downstream
    # contracts evaluate the parent and the figure-only child separately.
    figure_required = ("publication-ready figure" in intent) or (
        figure_only_step and "figure:" in expected
    )
    if figure_required:
        # When the step itself declares it skipped because the underlying data
        # are unavailable, do not fail the figure contract. The
        # skipped reason is the documented absence; the manuscript binder
        # already treats `skipped` as a first-class signal. Otherwise figure-
        # only steps would block the entire run whenever a sensitivity branch
        # has no eligible cohort.
        _skipped = (
            step_summary.get("skipped") if isinstance(step_summary, dict) else None
        )
        if _skipped:
            return findings
        figure_value = None
        for _key, value in _flatten_scalar_dict(step_summary).items():
            lowered_value = str(value).lower()
            if (
                lowered_value.endswith((".png", ".svg", ".pdf", ".tiff", ".tif"))
                or ".png" in lowered_value
                or ".svg" in lowered_value
                or ".pdf" in lowered_value
                or ".tiff" in lowered_value
                or ".tif" in lowered_value
            ):
                figure_value = value
                break
        if figure_value is None:
            # The flattened scan above reaches list elements, but only the
            # ones that coerce to a scalar. The coder prompt recommends
            # recording multiple figure paths in list-valued keys such as
            # ``figure_files`` / ``figure_file`` / ``figure_paths``; read those
            # by name so a nested or non-scalar shape still counts when it
            # contains at least one figure-shaped path.
            for list_key in (
                "figure_files",
                "figure_file",
                "figure_paths",
                "plot_files",
            ):
                candidate = (step_summary or {}).get(list_key)
                if isinstance(candidate, (list, tuple)):
                    candidate_values = []
                    for item in candidate:
                        if isinstance(item, dict):
                            candidate_values.extend(
                                str(value) for value in item.values()
                            )
                        else:
                            candidate_values.append(str(item))
                    if any(
                        value.lower().endswith(
                            (".png", ".svg", ".pdf", ".tiff", ".tif")
                        )
                        for value in candidate_values
                    ):
                        figure_value = candidate
                        break
        if figure_value is None:
            _append_missing(
                (
                    f"Step {step.step_id} was expected to produce a figure artifact, "
                    "but the step summary did not record any figure path or figure output."
                ),
                (
                    "figure_path",
                    "figure_files",
                    "figure_file",
                    "plot_path",
                    "png",
                    "svg",
                ),
            )

    return findings
