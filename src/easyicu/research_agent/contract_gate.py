"""Contract gates — deterministic-contract / figure-contract findings, as a module.

Real cross-file extraction of the deterministic contract-gate family out of the
``pipeline_execute`` orchestration loop (Codex-ordered, after the visual
``gate_evaluator`` move). Every function here only READS step state (via injected
validators + the filesystem) and RETURNS findings; control flow, repair,
demotions, step-status decisions, and evidence authority all stay in the
execution layer (``pipeline_execute``).

Two public gate entrypoints, both re-exported by ``pipeline_execute`` for
back-compat (so existing imports and monkeypatch call sites that target the
execution loop keep resolving):

* ``_step_deterministic_contract_findings`` — the shared pre-registration
  deterministic contract-validator sequence evaluated identically by the early
  in-loop repair gate and the final ``_evaluate_final_deterministic_gates``
  authority gate. It composes the ``_step_contract_findings`` /
  ``_primary_exposure_*`` / cohort-definition-sensitivity / cross-step validator
  sequence. NOTE: it looks its collaborators up in THIS module's namespace, so a
  test that stubs one of them must ``monkeypatch.setattr`` on ``contract_gate``
  (not ``pipeline_execute``) — same rule the ``gate_evaluator`` move established.
* ``_post_canonicalization_figure_findings`` — the figure-contract / figure-source
  / ordered-stratified findings evaluated AFTER the early figure-contract
  canonicalization repair (kept separate on purpose; see its docstring).

It also hosts the figure-contract SHAPING / canonicalization helpers that pair
with the figure gate (``_ensure_step_figure_contract``, the
``_figure_contract_source_data_canonicalization_*`` candidate/install pair,
``_step_has_figure_only_output_contract``, panel-role / reader-label / summary-path
helpers, ``_family_has_deterministic_figure_renderer``). These write the figure
contract file / return a canonicalization code candidate; the decision to install
and the demotion that consumes ``_family_has_deterministic_figure_renderer`` stay
in the execution layer. All are re-exported by ``pipeline_execute`` for back-compat.

Imports only leaf modules (schema / contracts / audits / plan_utils /
declared_product_contract / robustness_* / runtime_artifacts /
deterministic_robustness / ordered_stratified_contract / publication_figures) so
there is no import cycle with ``pipeline_execute``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .audits.step_summary_integrity import StepSummaryIntegrityValidator
from .audits.validators import (
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    PrimaryModelContractValidator,
    StepSummaryFractionValidator,
)
from .contracts import ValidationFinding
from .declared_product_contract import primary_analysis_cohort_integrity_findings
from .deterministic_robustness import replay_locked_memberships
from .ordered_stratified_contract import ordered_stratified_numeric_findings
from .publication_figures import make_figure_contract
from .plan_utils import (
    _normalised_expected_output_names,
    _primary_exposure_contract_findings,
    _primary_exposure_measurement_filter_findings,
    _primary_exposure_overadjustment_findings,
    _primary_model_leakage_findings,
    _step_contract_findings,
    _step_expects_figure,
)
from .robustness_execution_contract import (
    ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES,
    _executed_robustness_result_issues,
)
from .robustness_panel import RobustnessSpec
from .runtime_artifacts import current_successful_step_records
from .schema import AnalysisPlan, AnalysisStep, ResearchContext


def _read_locked_robustness_spec_dicts(run_dir: Path) -> List[Dict[str, Any]]:
    payload = json.loads(
        (Path(run_dir) / "robustness_specs_locked.json").read_text(encoding="utf-8")
    )
    raw_specs = payload.get("specs") if isinstance(payload, dict) else None
    if not isinstance(raw_specs, list):
        raise ValueError("robustness_specs_locked.json has no specs list")
    return [dict(spec) for spec in raw_specs if isinstance(spec, dict)]


_AGENT_OWNED_ROBUSTNESS_RESULT_METHODS = frozenset(
    {
        "cohort_definition_sensitivity",
        "prespecified_robustness_analysis",
    }
)


_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS = frozenset(
    {
        "cohort_definition_overlap_attrition",
        "cohort_overlap_and_attrition",
        "complete_case_n",
        "missingness_strategy_notes",
        "primary_or",
        "robustness_grid",
        "robustness_matrix",
        "robustness_summary",
        "sensitivity_comparison",
        "sensitivity_specification_matrix",
    }
)


def _is_cohort_definition_sensitivity_result_step(step: AnalysisStep) -> bool:
    """Return true for an agent-owned, plan-locked robustness result step.

    This predicate attaches specifications and validation; it does *not*
    dispatch a deterministic runner.  Ownership requires an exact controlled
    method head and a closed structured-product set so prose, step ids, or one
    stray robustness keyword cannot opt an unrelated analysis into the gate.
    """

    if _step_expects_figure(step):
        return False
    if _method_head(str(step.method or "")) not in (
        _AGENT_OWNED_ROBUSTNESS_RESULT_METHODS
    ):
        return False
    products = _closed_auxiliary_output_products(
        step.expected_outputs or [],
        supported_products=_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS,
    )
    return products is not None and bool(
        products
        & {
            "robustness_grid",
            "robustness_matrix",
            "robustness_summary",
            "sensitivity_comparison",
            "sensitivity_specification_matrix",
        }
    )


def _authoritative_primary_robustness_contract(
    *,
    completed_step_records: Sequence[Mapping[str, Any]],
    context: Optional[ResearchContext],
) -> Optional[Dict[str, Any]]:
    """Return one fitted primary model contract for robustness re-estimation.

    The robustness step is auxiliary: it may execute Planner-locked variants,
    but it may not select a different estimator, outcome, or exposure.  Bind it
    to the latest successful agent-produced primary contract that exactly
    matches the research context. Ambiguity remains fail-closed.
    """

    expected_exposure = str(
        (context.primary_exposure if context is not None else None) or ""
    ).strip()
    expected_outcome = str(
        (context.target_outcome if context is not None else None) or ""
    ).strip()
    for record in reversed(
        list(current_successful_step_records(completed_step_records))
    ):
        summary = record.get("step_summary")
        if not isinstance(summary, Mapping):
            continue
        candidates: List[Dict[str, Any]] = []
        for raw_contract in summary.get("model_contracts") or []:
            if not isinstance(raw_contract, Mapping):
                continue
            contract = dict(raw_contract)
            if str(contract.get("analysis_role") or "").strip().lower() != "primary":
                continue
            if str(contract.get("exposure_role") or "primary").strip().lower() != (
                "primary"
            ):
                continue
            if str(contract.get("fit_status") or "").strip().lower() != "fitted":
                continue
            if contract.get("converged") is not True:
                continue
            if expected_exposure and str(contract.get("exposure_source") or "") != (
                expected_exposure
            ):
                continue
            if expected_outcome and str(contract.get("outcome") or "") != (
                expected_outcome
            ):
                continue
            candidates.append(contract)
        if len(candidates) != 1:
            continue
        contract = candidates[0]
        contract["source_step_id"] = str(record.get("step_id") or "")
        final_terms = summary.get("final_design_terms")
        if isinstance(final_terms, list):
            contract["final_design_terms"] = list(final_terms)
        return contract
    return None


def _nonnegative_integral_value(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _declared_sensitivity_csv_paths(
    *,
    step_summary: Dict[str, Any],
    out_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Return declared spec-table paths and denominator-table paths."""

    spec_roles = {
        "table:robustness_grid",
        "table:sensitivity_specification_matrix",
        "table:robustness_summary",
        "table:robustness_matrix",
        "robustness_grid",
        "sensitivity_specification_matrix",
        "robustness_summary",
        "robustness_matrix",
    }
    denominator_roles = spec_roles | {
        "table:cohort_definition_overlap_attrition",
        "table:cohort_overlap_and_attrition",
        "cohort_definition_overlap_attrition",
        "cohort_overlap_and_attrition",
    }
    spec_names = {
        "robustness_grid.csv",
        "sensitivity_specification_matrix.csv",
        "robustness_summary.csv",
        "robustness_matrix.csv",
    }
    denominator_names = spec_names | {
        "cohort_definition_overlap_attrition.csv",
        "cohort_overlap_and_attrition.csv",
    }
    root = Path(out_dir).resolve()

    def _local_csv(value: Any) -> Optional[Path]:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".csv":
            return None
        return path if path.is_file() else None

    spec_paths: List[Path] = []
    denominator_paths: List[Path] = []
    output_files = step_summary.get("output_files")
    if isinstance(output_files, dict):
        for role, value in output_files.items():
            normalised_role = str(role or "").strip().lower()
            path = _local_csv(value)
            if path is None:
                continue
            if normalised_role in spec_roles:
                spec_paths.append(path)
            if normalised_role in denominator_roles:
                denominator_paths.append(path)
    for values in (
        output_files if isinstance(output_files, list) else [],
        (
            step_summary.get("outputs")
            if isinstance(step_summary.get("outputs"), list)
            else []
        ),
    ):
        for value in values:
            path = _local_csv(value)
            if path is None:
                continue
            if path.name in spec_names:
                spec_paths.append(path)
            if path.name in denominator_names:
                denominator_paths.append(path)
    for name in denominator_names:
        path = root / name
        if not path.is_file():
            continue
        if name in spec_names:
            spec_paths.append(path)
        denominator_paths.append(path)
    return list(dict.fromkeys(spec_paths)), list(dict.fromkeys(denominator_paths))


def _sensitivity_csv_rows(paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw_row in csv.DictReader(handle):
                    row = dict(raw_row)
                    # ``definition_id`` is the natural identifier in cohort
                    # overlap/attrition tables, while the locked robustness
                    # contract calls the same key ``spec_id``.  Normalize the
                    # typed table role here; values are still checked against
                    # the digest-bound lock and deterministic membership
                    # replay below, so this cannot authorize an invented id.
                    if (
                        not str(row.get("spec_id") or "").strip()
                        and str(row.get("definition_id") or "").strip()
                    ):
                        row["spec_id"] = row["definition_id"]
                    rows.append(row)
        except (OSError, csv.Error):
            continue
    return rows


def _cohort_definition_sensitivity_contract_findings(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    out_dir: Path,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Optional[Path] = None,
    context: Optional[ResearchContext] = None,
    completed_step_records: Sequence[Mapping[str, Any]] = (),
) -> List[ValidationFinding]:
    """Verify that an agent executed, rather than replaced, its locked specs."""

    if not _is_cohort_definition_sensitivity_result_step(step):
        return []
    # Findings emitted here belong to one exact planner step.  Keep that
    # ownership machine-readable so a successful retry/resume can supersede an
    # older failure without parsing prose or treating the whole robustness
    # subsystem as one global gate.
    step_detail = {"step_id": str(step.step_id)}
    try:
        locked_specs = _read_locked_robustness_spec_dicts(run_dir)
    except Exception as exc:
        return [
            ValidationFinding(
                validator="robustness_spec_lock",
                severity="error",
                message=f"Locked robustness definitions are unavailable: {exc}",
                detail={
                    **step_detail,
                    "lock_path": str(Path(run_dir) / "robustness_specs_locked.json"),
                },
            )
        ]

    locked_by_id = {
        str(spec.get("spec_id") or "").strip(): spec
        for spec in locked_specs
        if str(spec.get("spec_id") or "").strip()
    }
    executed_result_issues = _executed_robustness_result_issues(
        locked_by_id=locked_by_id,
        step_summary=step_summary,
        out_dir=out_dir,
        context=context,
        primary_model_contract=_authoritative_primary_robustness_contract(
            completed_step_records=completed_step_records,
            context=context,
        ),
    )
    reported_rows: List[Dict[str, Any]] = []
    raw_rows = step_summary.get("robustness_rows")
    if raw_rows is None and isinstance(step_summary.get("robustness_panel"), dict):
        raw_rows = step_summary["robustness_panel"].get("rows")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            reported_rows.append(dict(row))

    spec_paths, denominator_paths = _declared_sensitivity_csv_paths(
        step_summary=step_summary,
        out_dir=out_dir,
    )
    reported_rows.extend(
        _sensitivity_csv_rows(list(dict.fromkeys([*spec_paths, *denominator_paths])))
    )

    rows_by_id: Dict[str, List[Dict[str, Any]]] = {}
    for row in reported_rows:
        spec_id = str(row.get("spec_id") or "").strip()
        if spec_id:
            rows_by_id.setdefault(spec_id, []).append(row)
    reported_ids = set(rows_by_id)

    locked_ids = set(locked_by_id)
    missing_ids = sorted(locked_ids - reported_ids)
    extra_ids = sorted(reported_ids - locked_ids - {"primary"})
    findings: List[ValidationFinding] = []
    if executed_result_issues:
        findings.append(
            ValidationFinding(
                validator="robustness_executed_result",
                severity="error",
                message=(
                    "Each locked robustness specification must have exactly one "
                    "typed executed-result row bound to its fitted model and "
                    "coefficient evidence. Declaration and membership tables "
                    "cannot substitute for execution. Issues="
                    f"{executed_result_issues}."
                ),
                detail={
                    **step_detail,
                    "required_spec_ids": sorted(locked_by_id),
                    "issues": executed_result_issues,
                    "point_only_policy": (
                        "Penalized point-only fits may be executed but must set "
                        "reportable=false and interval_method=unavailable."
                    ),
                },
            )
        )
    missing_axis_ids: List[str] = []
    axis_mismatches: List[Dict[str, str]] = []
    for spec_id, spec in locked_by_id.items():
        expected_axis = str(spec.get("axis") or "").strip().lower()
        reported_axes = {
            str(row.get("axis") or "").strip().lower()
            for row in rows_by_id.get(spec_id, [])
            if str(row.get("axis") or "").strip()
        }
        if spec_id in reported_ids and not reported_axes:
            missing_axis_ids.append(spec_id)
        for reported_axis in sorted(reported_axes - {expected_axis}):
            axis_mismatches.append(
                {
                    "spec_id": spec_id,
                    "expected_axis": expected_axis,
                    "reported_axis": reported_axis,
                }
            )
    if missing_ids or extra_ids or missing_axis_ids or axis_mismatches:
        missing_definitions = [locked_by_id[spec_id] for spec_id in missing_ids]
        findings.append(
            ValidationFinding(
                validator="robustness_spec_lock",
                severity="error",
                message=(
                    "Cohort-definition sensitivity outputs must cover every "
                    "plan-time locked spec_id and axis without substitutes. "
                    f"Missing={missing_ids}; extra={extra_ids}; "
                    f"missing_axis={missing_axis_ids}; axis_mismatches="
                    f"{axis_mismatches}; missing locked "
                    "definitions="
                    + json.dumps(
                        missing_definitions,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
                detail={
                    **step_detail,
                    "locked_spec_ids": sorted(locked_ids),
                    "reported_spec_ids": sorted(reported_ids),
                    "missing_spec_ids": missing_ids,
                    "extra_spec_ids": extra_ids,
                    "missing_axis_spec_ids": missing_axis_ids,
                    "axis_mismatches": axis_mismatches,
                    "missing_spec_definitions": missing_definitions,
                    "specification_tables": [str(path) for path in spec_paths],
                },
            )
        )

    cohort_specs = [
        RobustnessSpec.from_dict(spec)
        for spec in locked_specs
        if str(spec.get("axis") or "").strip().lower() == "cohort"
    ]
    if cohort_specs:
        membership_issues: List[Dict[str, Any]] = []
        try:
            import pandas as pd  # type: ignore

            if cohort_path is None:
                raise ValueError("locked analysis cohort path is unavailable")
            universe = pd.read_parquet(universe_path)
            cohort = pd.read_parquet(cohort_path)
            replay_rows = replay_locked_memberships(
                specs=cohort_specs,
                cohort=cohort,
                universe=universe,
                context=context,
                exposure=str((context.primary_exposure if context else None) or ""),
            )
        except Exception as exc:
            findings.append(
                ValidationFinding(
                    validator="robustness_cohort_membership",
                    severity="error",
                    message=f"Could not replay locked cohort memberships: {exc}",
                    detail={
                        **step_detail,
                        "universe_path": str(universe_path),
                        "cohort_path": str(cohort_path) if cohort_path else None,
                    },
                )
            )
            return findings

        replay_by_id = {
            str(row.get("spec_id") or ""): row
            for row in replay_rows
            if str(row.get("spec_id") or "")
        }
        aliases = ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES
        for spec in cohort_specs:
            spec_id = spec.spec_id
            expected = replay_by_id.get(spec_id) or {}
            expected_inflow = _nonnegative_integral_value(expected.get("inflow_n"))
            expected_outflow = _nonnegative_integral_value(expected.get("outflow_n"))
            expected_primary = _nonnegative_integral_value(
                expected.get("primary_membership_n")
            )
            expected_variant = _nonnegative_integral_value(
                expected.get("variant_membership_n")
            )
            expected_values = {
                "universe_n": _nonnegative_integral_value(expected.get("universe_n")),
                "variant_membership_n": expected_variant,
                "inflow_n": expected_inflow,
                "outflow_n": expected_outflow,
                "overlap_n": (
                    expected_primary - expected_outflow
                    if expected_primary is not None and expected_outflow is not None
                    else None
                ),
            }
            if not expected.get("membership_executable") or any(
                value is None for value in expected_values.values()
            ):
                membership_issues.append(
                    {
                        "spec_id": spec_id,
                        "issue": "locked_membership_replay_not_executable",
                        "replay": expected,
                    }
                )
                continue

            reported_for_spec = rows_by_id.get(spec_id, [])
            for field, field_aliases in aliases.items():
                claims = {
                    value
                    for row in reported_for_spec
                    for alias in field_aliases
                    if (value := _nonnegative_integral_value(row.get(alias)))
                    is not None
                }
                if not claims:
                    membership_issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "missing_membership_field",
                            "field": field,
                            "accepted_aliases": list(field_aliases),
                        }
                    )
                elif claims != {expected_values[field]}:
                    membership_issues.append(
                        {
                            "spec_id": spec_id,
                            "issue": "membership_value_mismatch",
                            "field": field,
                            "expected": expected_values[field],
                            "reported": sorted(claims),
                        }
                    )

        if membership_issues:
            findings.append(
                ValidationFinding(
                    validator="robustness_cohort_membership",
                    severity="error",
                    message=(
                        "Cohort-axis robustness rows must match deterministic "
                        "replay of their plan-locked predicates on "
                        "EASYICU_UNIVERSE_PARQUET, including retained N, overlap, "
                        "entries, and exits. "
                        f"Issues={membership_issues}."
                    ),
                    detail={
                        **step_detail,
                        "universe_path": str(universe_path),
                        "cohort_path": str(cohort_path),
                        "cohort_spec_ids": sorted(
                            spec.spec_id for spec in cohort_specs
                        ),
                        "issues": membership_issues,
                    },
                )
            )
    return findings


_AUXILIARY_OUTPUT_KINDS = frozenset({"table", "statistic", "log"})


def _closed_auxiliary_output_products(
    expected_outputs: Sequence[str],
    *,
    supported_products: set[str] | frozenset[str],
) -> Optional[set[str]]:
    """Return all declared products only when one auxiliary owns all of them.

    Every non-empty output participates in the closed-contract decision,
    including bare filenames.  Unsupported artifact kinds and even one foreign
    product return ``None`` so a compact runner cannot silently ignore the rest
    of a mixed agent step.
    """

    products: set[str] = set()
    for raw in expected_outputs or []:
        value = str(raw or "").strip().lower()
        if not value:
            continue
        kind, separator, _product = value.partition(":")
        if separator and kind not in _AUXILIARY_OUTPUT_KINDS:
            return None
        normalized = _normalised_expected_output_names([value])
        if len(normalized) != 1:
            return None
        products.update(normalized)
    if not products or not products.issubset(set(supported_products)):
        return None
    return products


def _method_head(method: str) -> str:
    """Return the scientific owner of a ``<head>_with_<rider>`` method."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(method or "").strip().lower()).strip(
        "_"
    )
    return normalized.split("_with_", 1)[0]


def _step_deterministic_contract_findings(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    context: ResearchContext,
    step_summary: Dict[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Path,
    execution_cohort_path: Path,
    cross_step_cohort_lock_validator: CrossStepCohortLockValidator,
    cross_step_registered_output_validator: CrossStepRegisteredOutputValidator,
    cross_step_reconciliation_trace_validator: CrossStepReconciliationTraceValidator,
    step_summary_integrity_validator: StepSummaryIntegrityValidator,
    step_summary_fraction_validator: StepSummaryFractionValidator,
    cross_step_source_status_validator: CrossStepSourceStatusValidator,
    primary_model_contract_validator: PrimaryModelContractValidator,
) -> List[ValidationFinding]:
    """The shared pre-registration deterministic contract-validator sequence.

    Both the early pre-registration gate inside ``_execute_one_step`` and the
    final deterministic gate ``_evaluate_final_deterministic_gates`` evaluate this
    IDENTICAL 14-validator sequence in the SAME order — the early gate runs it
    before evidence registration so contract errors enter the in-run repair loop
    instead of becoming a terminal record. This is that single reusable sequence.

    It is pure with respect to run state: it returns findings and reads the
    filesystem via the validators, but does NOT mutate ``step_record``, publish
    evidence, apply primary-runner/figure demotions, or decide the step status —
    those, plus the figure-contract/figure-source validators and any
    canonicalization repair, stay at each call site because they differ between
    the early and final gates. ``execution_cohort_path`` is the universe-or-cohort
    path (``universe_path`` when the primary-analysis cohort producer uses the raw
    universe, else ``cohort_path``); each caller passes its already-resolved value
    (``step_execution_cohort_path`` in the early gate / ``execution_cohort_path``
    in the final gate — equal by the same ``primary_analysis_cohort_producer_uses_universe``
    predicate).
    """

    findings: List[ValidationFinding] = _step_contract_findings(
        step=step,
        step_summary=step_summary,
        context=context,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
        out_dir=out_dir,
    )
    findings += _cohort_definition_sensitivity_contract_findings(
        step=step,
        step_summary=step_summary,
        out_dir=out_dir,
        run_dir=run_dir,
        universe_path=universe_path,
        cohort_path=cohort_path,
        context=context,
        completed_step_records=completed_step_records,
    )
    findings += primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=step_summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=cohort_path,
    )
    findings += cross_step_cohort_lock_validator.audit(
        step=step,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
    )
    findings += cross_step_registered_output_validator.audit(
        step=step,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
    )
    findings += cross_step_reconciliation_trace_validator.audit(
        step=step,
        step_summary=step_summary,
        out_dir=out_dir,
    )
    findings += step_summary_integrity_validator.audit(
        step=step,
        step_summary=step_summary,
        resolved_input_bindings=resolved_input_bindings,
        cohort_path=execution_cohort_path,
    )
    findings += step_summary_fraction_validator.audit(
        step=step,
        step_summary=step_summary,
    )
    findings += cross_step_source_status_validator.audit(
        step=step,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
    )
    findings += primary_model_contract_validator.audit(
        step=step,
        step_summary=step_summary,
        context=context,
        completed_step_records=completed_step_records,
        out_dir=out_dir,
        cohort_path=execution_cohort_path,
    )
    findings += _primary_exposure_contract_findings(
        step=step,
        step_summary=step_summary,
        context=context,
    )
    findings += _primary_exposure_measurement_filter_findings(
        step=step,
        step_summary=step_summary,
        context=context,
    )
    findings += _primary_exposure_overadjustment_findings(
        step=step,
        context=context,
        out_dir=out_dir,
    )
    findings += _primary_model_leakage_findings(
        step=step,
        context=context,
        out_dir=out_dir,
    )
    return findings


def _post_canonicalization_figure_findings(
    *,
    step: AnalysisStep,
    out_dir: Path,
    run_dir: Path,
    step_summary: Dict[str, Any],
    completed_step_records: Sequence[Mapping[str, Any]],
    resolved_input_bindings: Mapping[str, Mapping[str, Any]],
    execution_cohort_path: Path,
    figure_contract_validator: FigureContractQualityValidator,
    figure_source_validator: FigureSourceDataValidator,
) -> List[ValidationFinding]:
    """Figure-contract / figure-source / ordered-stratified findings evaluated
    AFTER the early figure-contract canonicalization repair.

    These are kept OUT of ``_step_deterministic_contract_findings`` on purpose:
    the early pre-registration gate must interleave the figure-contract
    canonicalization repair BETWEEN the shared contract sequence and these figure
    audits (the audits must see the already-canonicalized contracts). So the
    early gate calls the shared contract sequence, then runs the canonicalization
    repair inline, then calls this — preserving that hard ordering while still
    lifting the figure-audit block out of the execution loop.
    """

    findings: List[ValidationFinding] = figure_contract_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
    )
    findings += figure_source_validator.audit(
        step=step,
        out_dir=out_dir,
        run_dir=run_dir,
        step_summary=step_summary,
        completed_step_records=completed_step_records,
        resolved_input_bindings=resolved_input_bindings,
    )
    # For the controlled ordered-stratified method, replay the agent-authored
    # tables from the locked cohort before evidence registration. Numeric/method
    # errors therefore return to the existing coder repair loop instead of
    # becoming a late warning.
    findings += ordered_stratified_numeric_findings(
        cohort_path=execution_cohort_path,
        step=step,
        out_dir=out_dir,
        step_summary=step_summary,
    )
    return findings


def _step_has_figure_only_output_contract(step: AnalysisStep) -> bool:
    """Whether replacing ``outputs/`` can only replace presentation artifacts.

    Deterministic renderers install a complete staged bundle.  They are safe as
    a preflight or whole-directory repair only for an explicitly figure-only
    step; a mixed table/model + figure contract must stay with the coder so a
    renderer cannot erase or silently stand in for scientific products.
    """

    outputs = [
        str(output or "").strip()
        for output in (step.expected_outputs or [])
        if str(output or "").strip()
    ]

    def _is_typed_figure_product(output: str) -> bool:
        token = str(output or "").strip().lower()
        kind, separator, _product = token.partition(":")
        if separator:
            # The artifact kind is authoritative. A scientific table/model
            # whose product name happens to contain ``figure`` or ``plot`` is
            # still a mixed contract and must remain coder-owned.
            return kind.strip() in {"figure", "plot", "chart", "fig", "heatmap"}
        # Legacy bare declarations are figure-only only when they name an
        # actual image/vector export, never from a keyword in the stem.
        return token.endswith((".png", ".svg", ".pdf", ".tif", ".tiff"))

    return bool(outputs) and all(_is_typed_figure_product(output) for output in outputs)


def _reader_label_from_stem(stem: str) -> str:
    words = [
        token for token in stem.replace("-", "_").replace(".", "_").split("_") if token
    ]
    if not words:
        return "Manuscript figure"
    return " ".join(
        word.capitalize() if len(word) > 3 else word.upper() for word in words
    )


def _infer_step_figure_panel_role(step: AnalysisStep, stem: str) -> str:
    text = " ".join(
        [
            step.step_id,
            step.intent or "",
            step.method or "",
            stem,
            " ".join(step.expected_outputs or []),
        ]
    ).lower()
    if any(token in text for token in ("robustness", "sensitivity", "specification")):
        return "robustness"
    if any(
        token in text
        for token in (
            "missingness",
            "measurement",
            "quality",
            "baseline",
            "table one",
            "attrition",
            "cohort",
            "audit",
        )
    ):
        return "audit"
    if any(
        token in text
        for token in ("association", "effect", "forest", "estimate", "outcome")
    ):
        return "relationship"
    return "overview"


def _step_summary_paths(
    value: Any,
    *,
    out_dir: Path,
    allowed_suffixes: Optional[set[str]] = None,
) -> List[Path]:
    raw_values: List[Any] = []
    if isinstance(value, (str, Path)):
        raw_values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    paths: List[Path] = []
    for raw in raw_values:
        path = Path(str(raw))
        if not path.is_absolute():
            path = out_dir / path
        if not path.exists() or not path.is_file():
            continue
        if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
            continue
        paths.append(path)
    return sorted(dict.fromkeys(paths))


def _ensure_step_figure_contract(
    *,
    step: AnalysisStep,
    out_dir: Path,
    step_summary: Mapping[str, Any],
    evidence_ids: Sequence[str],
) -> Optional[Path]:
    """Create a minimal manuscript-facing contract for valid figure exports.

    Coder prompts already ask for ``*.figure_contract.json``. This runner-level
    fallback covers the common successful-plot / missing-boilerplate case without
    weakening result-bearing figure gates: association and robustness figures
    still keep their result-like roles, so the contract validator can require
    multi-panel evidence when appropriate.
    """

    if sorted(out_dir.glob("*.figure_contract.json")):
        return None
    figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
    figure_paths = _step_summary_paths(
        step_summary.get("figure_files") or step_summary.get("figure_path"),
        out_dir=out_dir,
        allowed_suffixes=figure_suffixes,
    )
    if not figure_paths:
        figure_paths = sorted(
            path
            for path in out_dir.iterdir()
            if path.is_file() and path.suffix.lower() in figure_suffixes
        )
    if not figure_paths:
        return None
    source_paths = _step_summary_paths(
        step_summary.get("source_data_files")
        or step_summary.get("source_data")
        or step_summary.get("source_table"),
        out_dir=out_dir,
    )
    primary_stem = figure_paths[0].stem
    label = _reader_label_from_stem(primary_stem)
    role = _infer_step_figure_panel_role(step, primary_stem)
    contract = make_figure_contract(
        figure_id=primary_stem,
        core_claim=(
            f"{label} summarizes the planned manuscript figure from registered "
            "source data."
        ),
        panels=[
            {
                "panel_id": "A",
                "title": label,
                "role": role,
                "claim": (
                    "This panel displays the step result using registered "
                    "source data and preserved code provenance."
                ),
                "evidence_ids": list(evidence_ids),
                "review_risk": (
                    "Review the source data and upstream step contract before "
                    "using the panel in manuscript text."
                ),
            }
        ],
        export_formats=[
            suffix.lstrip(".")
            for suffix in (".svg", ".pdf", ".png", ".tiff")
            if any(path.suffix.lower() == suffix for path in figure_paths)
        ]
        or ["svg", "png"],
        source_data=[path.name for path in source_paths],
        statistics_note="Auto-generated by the runner from step summary metadata.",
        image_integrity_note="No values were invented or visually altered by this contract synthesis.",
    )
    contract_path = out_dir / f"{primary_stem}.figure_contract.json"
    contract_path.write_text(
        json.dumps(contract.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return contract_path


def _figure_contract_source_data_canonicalization_candidate(
    *,
    contract_path: Path,
    out_dir: Path,
) -> Optional[Tuple[str, str, List[str]]]:
    """Return an exact legacy-descriptor -> flat-basename JSON rewrite.

    ``make_figure_contract`` accepts small path mappings as an in-memory input
    compatibility layer but persists canonical ``List[str]`` source data.
    Some legacy agent scripts wrote those mappings directly to JSON.  This
    representation-only migration is deliberately strict: every populated path
    alias must agree, every source must be an existing ordinary local CSV in
    the exact step output directory, and non-empty evidence references are not
    discarded.  Anything else is left untouched for the validator to block.
    """

    output_root = Path(out_dir).resolve()
    candidate_path = Path(contract_path)
    try:
        if (
            candidate_path.parent.resolve() != output_root
            or candidate_path.resolve(strict=True).parent != output_root
            or not candidate_path.is_file()
            or candidate_path.is_symlink()
            or candidate_path.stat().st_nlink != 1
        ):
            return None
        before = candidate_path.read_text(encoding="utf-8")
        payload = json.loads(before)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    raw_sources = payload.get("source_data")
    if isinstance(raw_sources, Mapping):
        source_items: List[Any] = [raw_sources]
    elif isinstance(raw_sources, list):
        source_items = list(raw_sources)
    else:
        return None
    if not source_items or not any(isinstance(item, Mapping) for item in source_items):
        return None

    path_keys = ("file", "filename", "path", "relative_path")
    canonical_names: List[str] = []
    for item in source_items:
        if isinstance(item, str):
            source_name = item.strip()
        elif isinstance(item, Mapping):
            if item.get("evidence_id") not in (None, "") or item.get(
                "evidence_ids"
            ) not in (None, "", []):
                return None
            populated: List[str] = []
            for key in path_keys:
                value = item.get(key)
                if value in (None, ""):
                    continue
                if not isinstance(value, str) or not value.strip():
                    return None
                populated.append(value.strip())
            if len(set(populated)) != 1:
                return None
            source_name = populated[0]
        else:
            return None
        if (
            not source_name
            or Path(source_name).name != source_name
            or "/" in source_name
            or "\\" in source_name
            or Path(source_name).suffix.lower() != ".csv"
        ):
            return None
        source_path = output_root / source_name
        try:
            if (
                source_path.resolve(strict=True).parent != output_root
                or not source_path.is_file()
                or source_path.is_symlink()
                or source_path.stat().st_nlink != 1
            ):
                return None
        except OSError:
            return None
        canonical_names.append(source_name)

    canonical_payload = dict(payload)
    canonical_payload["source_data"] = canonical_names
    after = json.dumps(canonical_payload, indent=2, ensure_ascii=False) + "\n"
    if before == after:
        return None
    return before, after, canonical_names


def _install_figure_contract_source_data_canonicalization(
    *,
    contract_path: Path,
    expected_before: str,
    canonical_text: str,
) -> None:
    """Atomically install one pre-authorized contract-schema rewrite.

    The generated step controls its output directory, so a predictable temp
    path is unsafe: it could be pre-created as a symlink before the host writes.
    ``mkstemp`` gives us an exclusive random regular file.  The destination is
    also reopened without following symlinks and must still match the exact
    content reviewed by the authorization boundary.
    """

    contract_path = Path(contract_path)
    parent = contract_path.parent
    read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    contract_fd = os.open(contract_path, read_flags)
    try:
        opened_stat = os.fstat(contract_fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise ValueError("figure contract must remain one ordinary file")
        with os.fdopen(contract_fd, "r", encoding="utf-8") as handle:
            contract_fd = -1
            observed_before = handle.read()
        if observed_before != expected_before:
            raise ValueError("figure contract changed after canonicalization review")

        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{contract_path.name}.",
            suffix=".schema.tmp",
            dir=parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_text)
                handle.flush()
                os.fsync(handle.fileno())
            current_stat = os.stat(contract_path, follow_symlinks=False)
            if (
                not stat.S_ISREG(current_stat.st_mode)
                or current_stat.st_nlink != 1
                or current_stat.st_dev != opened_stat.st_dev
                or current_stat.st_ino != opened_stat.st_ino
            ):
                raise ValueError("figure contract identity changed before replace")
            os.replace(temporary_path, contract_path)
            try:
                directory_fd = os.open(
                    parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            temporary_path.unlink(missing_ok=True)
    finally:
        if contract_fd >= 0:
            os.close(contract_fd)


def _family_has_deterministic_figure_renderer(context: Any) -> bool:
    """True when this study-design family builds its PRIMARY publication figure
    deterministically in the write phase (``render_family_figure``).

    Lazy import keeps ``pipeline_execute`` free of a ``figures`` /
    ``study_design`` import-order dependency and fail-safes to False (strict) if
    the family cannot be inferred.
    """
    try:
        from .figures import FAMILY_RENDERERS
        from .study_design import infer_study_design_family

        return str(infer_study_design_family(context)) in FAMILY_RENDERERS
    except Exception:
        return False


__all__ = [
    "_step_deterministic_contract_findings",
    "_post_canonicalization_figure_findings",
    "_read_locked_robustness_spec_dicts",
    "_is_cohort_definition_sensitivity_result_step",
    "_authoritative_primary_robustness_contract",
    "_cohort_definition_sensitivity_contract_findings",
    "_closed_auxiliary_output_products",
    "_method_head",
    "_nonnegative_integral_value",
    "_declared_sensitivity_csv_paths",
    "_sensitivity_csv_rows",
    "_AGENT_OWNED_ROBUSTNESS_RESULT_METHODS",
    "_AGENT_OWNED_ROBUSTNESS_RESULT_PRODUCTS",
    "_AUXILIARY_OUTPUT_KINDS",
    "_step_has_figure_only_output_contract",
    "_reader_label_from_stem",
    "_infer_step_figure_panel_role",
    "_step_summary_paths",
    "_ensure_step_figure_contract",
    "_figure_contract_source_data_canonicalization_candidate",
    "_install_figure_contract_source_data_canonicalization",
    "_family_has_deterministic_figure_renderer",
]
