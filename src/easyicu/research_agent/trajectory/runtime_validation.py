"""Cross-step validation for the signed deterministic trajectory runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

_REPRESENTATION = "signed_fixed_window_trajectory_representation"
_CANDIDATES = "observed_data_diagonal_gaussian_mixture_candidate_selection"
_STABILITY = "trajectory_cluster_stability_characterization"
_FIGURE = "signed_trajectory_selection_diagnostic_figure"
_KINDS = {
    _REPRESENTATION: "trajectory_signed_representation",
    _CANDIDATES: "trajectory_signed_candidate_selection",
    _STABILITY: "trajectory_cluster_stability",
    _FIGURE: "trajectory_selection_diagnostic_figure",
}
_CONTRACT_REF = re.compile(r"^scientific_runtime_contract:([0-9a-f]{64})$")


def signed_trajectory_plan_claimed(plan: object) -> bool:
    methods = [str(getattr(step, "method", "") or "") for step in getattr(plan, "steps", ()) or ()]
    return _REPRESENTATION in methods and _CANDIDATES in methods


def signed_trajectory_plan_contract_errors(plan: object) -> list[str]:
    """Require one closed representation -> selection -> stability -> figure DAG."""

    steps = tuple(getattr(plan, "steps", ()) or ())
    methods = tuple(str(getattr(step, "method", "") or "") for step in steps)
    expected = (_REPRESENTATION, _CANDIDATES, _STABILITY, _FIGURE)
    if methods != expected:
        return ["signed trajectory plan does not contain the four ordered owners"]
    errors: list[str] = []
    roles = tuple(getattr(step, "planned_analysis_role", None) for step in steps)
    if roles != ("auxiliary", "primary", "auxiliary", "auxiliary"):
        errors.append("signed trajectory plan has invalid scientific roles")
    refs = [tuple(getattr(step, "icu_rule_refs", ()) or ()) for step in steps]
    matches = [
        _CONTRACT_REF.fullmatch(str(values[0]))
        if len(values) == 1
        else None
        for values in refs
    ]
    if any(match is None for match in matches) or len(
        {match.group(1) for match in matches if match is not None}
    ) != 1:
        errors.append("signed trajectory owners do not share one runtime contract")
    candidate_inputs = tuple(getattr(steps[1], "inputs", ()) or ())
    if candidate_inputs != (
        "artifact:trajectory_representation",
        "manifest:trajectory_representation_schema",
    ):
        errors.append("signed trajectory candidate owner has invalid inputs")
    stability_inputs = tuple(getattr(steps[2], "inputs", ()) or ())
    if stability_inputs != (
        "artifact:trajectory_representation",
        "artifact:candidate_cluster_assignments",
        "manifest:cluster_selection",
        "manifest:trajectory_representation_schema",
        "manifest:candidate_cluster_solution_schema",
    ):
        errors.append("signed trajectory stability owner has invalid inputs")
    if getattr(steps[2], "trajectory_stability_spec", None) is None:
        errors.append("signed trajectory stability design is absent")
    figure_inputs = tuple(getattr(steps[3], "inputs", ()) or ())
    if figure_inputs != (
        "table:trajectory_candidate_selection",
        "table:feature_availability",
    ):
        errors.append("signed trajectory diagnostic figure has invalid inputs")
    return errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_output_path(
    *, run_dir: Path, step_id: str, summary: Mapping[str, Any], product: str
) -> Path | None:
    files = summary.get("output_files")
    filename = files.get(product) if isinstance(files, Mapping) else None
    if not isinstance(filename, str) or Path(filename).name != filename:
        return None
    return Path(run_dir) / "steps" / step_id / "outputs" / filename


def _binding(summary: Mapping[str, Any], input_key: str) -> Mapping[str, Any] | None:
    for value in summary.get("input_bindings") or []:
        if isinstance(value, Mapping) and value.get("input_key") == input_key:
            return value
    return None


def _authority(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = summary.get("scientific_runtime_authority")
    return value if isinstance(value, Mapping) else None


def _candidate_selection_errors(selection: Any) -> tuple[list[str], int | None]:
    if not isinstance(selection, Mapping):
        return ["signed trajectory selection receipt is absent"], None
    rows = selection.get("candidates")
    if not isinstance(rows, list) or len(rows) < 2:
        return ["signed trajectory candidate grid is incomplete"], None
    try:
        candidates = [
            (int(row["n_clusters"]), float(row["criterion_value"]))
            for row in rows
            if isinstance(row, Mapping)
        ]
        selected = int(selection["selected_n_clusters"])
    except (KeyError, TypeError, ValueError):
        return ["signed trajectory candidate grid is malformed"], None
    if len(candidates) != len(rows) or any(not math.isfinite(v) for _, v in candidates):
        return ["signed trajectory BIC values are incomplete or non-finite"], None
    ks = [k for k, _ in candidates]
    expected = min(candidates, key=lambda item: (item[1], item[0]))[0]
    errors: list[str] = []
    if ks != sorted(set(ks)) or ks[0] < 2:
        errors.append("signed trajectory candidate k grid is not closed and ordered")
    if (
        selection.get("criterion") != "bic"
        or selection.get("direction") != "minimize"
        or selection.get("selection_rule") != "minimum"
        or selected != expected
    ):
        errors.append("signed trajectory candidate selection does not replay")
    return errors, selected


def signed_trajectory_runtime_bundle_errors(
    *, plan: object, records: Sequence[Mapping[str, Any]], run_dir: Path
) -> list[str]:
    """Validate the signed cross-step decision, including an honest non-solution."""

    errors = signed_trajectory_plan_contract_errors(plan)
    if errors:
        return errors
    steps = tuple(getattr(plan, "steps", ()) or ())
    by_kind: dict[str, list[Mapping[str, Any]]] = {kind: [] for kind in _KINDS.values()}
    for record in records:
        kind = str(record.get("deterministic_standard_analysis") or "")
        if kind in by_kind and isinstance(record.get("step_summary"), Mapping):
            by_kind[kind].append(record)
    if any(len(values) != 1 for values in by_kind.values()):
        return ["signed trajectory validator requires one current receipt per owner"]
    ordered_records = [by_kind[_KINDS[method]][0] for method in (_REPRESENTATION, _CANDIDATES, _STABILITY, _FIGURE)]
    if any(
        record.get("step_id") != step.step_id or record.get("status") != "ok"
        for step, record in zip(steps, ordered_records, strict=True)
    ):
        errors.append("signed trajectory receipts do not match the current plan owners")
    rep, candidate, stability, figure = [record["step_summary"] for record in ordered_records]
    if any(summary.get("status") != "ok" for summary in (rep, candidate, stability, figure)):
        errors.append("signed trajectory owner did not complete successfully")

    contract_match = _CONTRACT_REF.fullmatch(str(steps[0].icu_rule_refs[0]))
    assert contract_match is not None
    contract_sha = contract_match.group(1)
    rep_authority = _authority(rep)
    candidate_authority = _authority(candidate)
    stability_authority = _authority(stability)
    try:
        protocol_values = {
            str(value["protocol_content_sha256"])
            for value in (rep_authority, candidate_authority, stability_authority)
            if value is not None
        }
        runtime_values = {
            str(rep["runtime_projection_sha256"]),
            str(candidate_authority["runtime_projection_sha256"]),
            str(stability_authority["runtime_projection_sha256"]),
        }
        contract_values = {
            str(value["execution_contract_sha256"])
            for value in (rep_authority, candidate_authority, stability_authority)
            if value is not None
        }
        binding_ok = (
            all(value is not None for value in (rep_authority, candidate_authority, stability_authority))
            and protocol_values and all(len(value) == 64 for value in protocol_values)
            and len(runtime_values) == 1 and all(len(value) == 64 for value in runtime_values)
            and contract_values == {contract_sha}
        )
    except (KeyError, TypeError):
        binding_ok = False
    if not binding_ok:
        errors.append("signed trajectory runtime authority bindings disagree")

    families = rep.get("observation_family")
    columns = rep.get("representation_columns")
    if not (
        isinstance(families, list)
        and len(families) >= 2
        and len(families) == len(set(families))
        and isinstance(columns, list)
        and len(columns) >= len(families) * 2
        and len(columns) == len(set(columns))
        and int(rep.get("eligible_n") or 0) > 0
    ):
        errors.append("signed trajectory representation receipt is incomplete")

    selection_errors, selected_k = _candidate_selection_errors(candidate.get("cluster_selection"))
    errors.extend(selection_errors)
    if selected_k is not None and candidate.get("n_clusters") != selected_k:
        errors.append("signed trajectory summary disagrees with selected k")

    rep_schema_path = _safe_output_path(
        run_dir=run_dir,
        step_id=steps[0].step_id,
        summary=rep,
        product="manifest:trajectory_representation_schema",
    )
    candidate_schema_path = _safe_output_path(
        run_dir=run_dir,
        step_id=steps[1].step_id,
        summary=candidate,
        product="manifest:candidate_cluster_solution_schema",
    )
    selection_path = _safe_output_path(
        run_dir=run_dir,
        step_id=steps[1].step_id,
        summary=candidate,
        product="manifest:cluster_selection",
    )
    try:
        candidate_schema = json.loads(candidate_schema_path.read_text("utf-8"))
        selection_payload = json.loads(selection_path.read_text("utf-8"))
        schema_binding = _binding(candidate, "manifest:trajectory_representation_schema")
        candidate_schema_binding = _binding(
            stability, "manifest:candidate_cluster_solution_schema"
        )
        selection_binding = _binding(stability, "manifest:cluster_selection")
        file_bindings_ok = (
            rep_schema_path is not None
            and rep_schema_path.is_file()
            and schema_binding is not None
            and _sha256(rep_schema_path) == schema_binding.get("sha256")
            and candidate_schema_path is not None
            and candidate_schema_path.is_file()
            and _sha256(candidate_schema_path)
            == candidate.get("candidate_solution_schema_sha256")
            == (candidate_schema_binding or {}).get("sha256")
            and selection_path is not None
            and selection_path.is_file()
            and _sha256(selection_path) == (selection_binding or {}).get("sha256")
            and selection_payload == candidate.get("cluster_selection")
        )
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
        candidate_schema = {}
        file_bindings_ok = False
    if not file_bindings_ok:
        errors.append("signed trajectory artifact digests do not close across owners")

    rejected = candidate.get("scientific_status") == "failed_closed"
    if rejected:
        reason = str(candidate.get("reason_code") or "")
        selected_grid = candidate.get("cluster_selection", {}).get("candidates", [])
        max_k = max(int(row["n_clusters"]) for row in selected_grid) if selected_grid else None
        boundary_rejection = candidate.get("reportable_result") == (
            "no_interior_solution_in_prespecified_candidate_range"
        )
        if not (
            reason
            and candidate.get("stability_authorized") is False
            and candidate_schema.get("stability_authorized") is False
            and candidate_schema.get("scientific_selection_reason_code") == reason
            and (not boundary_rejection or selected_k == max_k)
            and stability.get("scientific_status") == "failed_closed"
            and stability.get("reason_code") == reason
            and stability.get("freeze_status") == "not_frozen_candidate_selection_failed_closed"
            and stability.get("stability_refits_executed") == 0
            and stability.get("reportable_result") == "no_stable_phenotype_solution"
            and stability.get("outcome_binding_received_by_executor") is False
            and not stability.get("outcome_bindings_received")
            and figure.get("scientific_status") == "failed_closed"
            and figure.get("reason_code") == reason
        ):
            errors.append("signed trajectory failed-closed decision is incoherent")
    else:
        if not (
            candidate.get("scientific_status") == "selected"
            and candidate.get("stability_authorized") is True
            and stability.get("selected_n_clusters") == selected_k
            and int(stability.get("n_successful_resamples") or 0) > 0
            and stability.get("stability_threshold_passed") is not False
            and stability.get("outcome_binding_received_by_executor") is False
            and not stability.get("outcome_bindings_received")
        ):
            errors.append("signed trajectory stable-solution decision is incoherent")
    return errors


__all__ = [
    "signed_trajectory_plan_claimed",
    "signed_trajectory_plan_contract_errors",
    "signed_trajectory_runtime_bundle_errors",
]
