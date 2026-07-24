"""Contract gates — deterministic-contract / figure-contract findings, as a module.

Real cross-file extraction of the deterministic contract-gate family out of the
``execution.phase`` orchestration loop (Codex-ordered, after the visual
``gate_evaluator`` move). Every function here only READS step state (via injected
validators + the filesystem) and RETURNS findings; control flow, repair,
demotions, step-status decisions, and evidence authority all stay in the
execution layer (``execution.phase``).

Two public gate entrypoints, both re-exported by ``execution.phase`` for
back-compat (so existing imports and monkeypatch call sites that target the
execution loop keep resolving):

* ``_step_deterministic_contract_findings`` — the shared pre-registration
  deterministic contract-validator sequence evaluated identically by the early
  in-loop repair gate and the final ``_evaluate_final_deterministic_gates``
  authority gate. It composes the ``_step_contract_findings`` /
  ``_primary_exposure_*`` / cohort-definition-sensitivity / cross-step validator
  sequence. NOTE: it looks its collaborators up in THIS module's namespace, so a
  test that stubs one of them must ``monkeypatch.setattr`` on ``contract_gate``
  (not ``execution.phase``) — same rule the ``gate_evaluator`` move established.
* ``_post_canonicalization_figure_findings`` — the figure-contract / figure-source
  / ordered-stratified findings evaluated AFTER the early figure-contract
  canonicalization repair (kept separate on purpose; see its docstring).

This module holds ONLY read-only findings gates. The figure-contract PREPARATION
helpers that WRITE the figure contract file / build the canonicalization repair
candidate live in the sibling ``figure_contract_preparation`` module — the
read-only-gate vs writes-files boundary is deliberate (Codex-ordered split).

Imports only leaf modules (schema / contracts / audits / plan_utils /
declared_product_contract / robustness_* / runtime_artifacts /
deterministic_robustness / ordered_stratified_contract) so there is no import
cycle with ``execution.phase``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..audits.step_summary_integrity import StepSummaryIntegrityValidator
from ..audits.envelope_consumers import StepSummaryFractionEnvelopeDualReader
from ..audits.validators import (
    CrossStepCohortLockValidator,
    CrossStepReconciliationTraceValidator,
    CrossStepRegisteredOutputValidator,
    CrossStepSourceStatusValidator,
    FigureContractQualityValidator,
    FigureSourceDataValidator,
    PrimaryModelContractValidator,
    StepSummaryFractionValidator,
)
from ..contracts.result_envelope import StepResultEnvelope
from ..contracts.runtime import ValidationFinding
from ..cohort.schema import ANALYSIS_COHORT_FILENAME
from ..contracts.declared_product import (
    primary_analysis_cohort_integrity_findings,
    primary_analysis_cohort_producer_uses_universe,
)
from ..robustness.membership import replay_locked_memberships
from ..contracts.ordered_stratified import ordered_stratified_numeric_findings
from ..plan_utils import (
    _normalised_expected_output_names,
    _primary_exposure_contract_findings,
    _primary_exposure_measurement_filter_findings,
    _primary_exposure_overadjustment_findings,
    _primary_model_leakage_findings,
    _step_contract_findings,
    _step_expects_figure,
)
from ..contracts.robustness_execution import (
    ROBUSTNESS_COHORT_MEMBERSHIP_ALIASES,
    _executed_robustness_result_issues,
)
from ..robustness.panel import RobustnessSpec
from ..authority.runtime_artifacts import current_successful_step_records
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext


def _primary_cohort_integrity_authority_paths(
    *,
    step: AnalysisStep,
    plan: AnalysisPlan,
    run_dir: Path,
    universe_path: Path,
    cohort_path: Path,
    execution_cohort_path: Path,
) -> tuple[Path, Path]:
    """Keep a cohort producer's full authority outside dev sampling."""

    if not primary_analysis_cohort_producer_uses_universe(step=step, plan=plan):
        return Path(execution_cohort_path), Path(cohort_path)
    return Path(universe_path), Path(run_dir) / ANALYSIS_COHORT_FILENAME


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
        "prespecified_sensitivity_analysis_grid",
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
    final_fraction_envelope_validator: (
        StepSummaryFractionEnvelopeDualReader | None
    ) = None,
    final_fraction_envelope: StepResultEnvelope | None = None,
    final_fraction_current_status: str | None = None,
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
    integrity_universe_path, integrity_cohort_path = (
        _primary_cohort_integrity_authority_paths(
            step=step,
            plan=plan,
            run_dir=run_dir,
            universe_path=universe_path,
            cohort_path=cohort_path,
            execution_cohort_path=execution_cohort_path,
        )
    )
    findings += primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        context=context,
        step_summary=step_summary,
        out_dir=out_dir,
        universe_path=integrity_universe_path,
        authoritative_cohort_path=integrity_cohort_path,
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
        cohort_path=integrity_universe_path,
    )
    legacy_fraction_findings = step_summary_fraction_validator.audit(
        step=step,
        step_summary=step_summary,
    )
    if final_fraction_envelope_validator is not None:
        findings += final_fraction_envelope_validator.audit(
            step=step,
            step_summary=step_summary,
            envelope=final_fraction_envelope,
            current_status=final_fraction_current_status,
            legacy_findings=legacy_fraction_findings,
        )
    else:
        findings += legacy_fraction_findings
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
        cohort_path=integrity_universe_path,
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
]
