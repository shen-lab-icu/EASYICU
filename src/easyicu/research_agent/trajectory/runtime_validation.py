"""Cross-step validation for the signed deterministic trajectory runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..authority.prespecified_rule_outcomes import (
    RULE_OUTCOMES_KEY,
    validate_rule_outcome,
)
from ..contracts.phenotype_comparison import (
    COMPARISON_ACTION,
    COMPARISON_PRODUCT,
    TRAJECTORY_ASSIGNMENTS_PRODUCT,
    TRAJECTORY_NO_SOLUTION_REASON,
    comparison_cohort_input,
    comparison_label_source,
)
from ..contracts.primary_cohort import (
    HOST_BOUND_COHORT_METHOD,
    is_host_bound_cohort_publisher,
)

_REPRESENTATION = "signed_fixed_window_trajectory_representation"
_CANDIDATES = "observed_data_diagonal_gaussian_mixture_candidate_selection"
#: The signed candidate owner for declared ordinal coordinates; it fills the
#: same role as the Gaussian candidate owner.
_MIXED_MODE_CANDIDATES = "observed_data_mixed_mode_latent_class_candidate_selection"
_STABILITY = "trajectory_cluster_stability_characterization"
_FIGURE = "signed_trajectory_selection_diagnostic_figure"
_KINDS = {
    _REPRESENTATION: "trajectory_signed_representation",
    _CANDIDATES: "trajectory_signed_candidate_selection",
    _STABILITY: "trajectory_cluster_stability",
    _FIGURE: "trajectory_selection_diagnostic_figure",
}
_CONTRACT_REF = re.compile(r"^scientific_runtime_contract:([0-9a-f]{64})$")
#: The stability owner's two rejections of a selected solution, each with the
#: freeze status it writes and the disposition of the rule outcome it states.
_STABILITY_REJECTIONS = {
    "TRAJECTORY_STABILITY_BELOW_THRESHOLD": (
        "not_frozen_stability_threshold_failed",
        "stability_below_threshold",
    ),
    "TRAJECTORY_STABILITY_REFITS_BELOW_MINIMUM": (
        "not_frozen_stability_refits_below_minimum",
        "too_few_successful_refits",
    ),
}

#: The four signed owners, in execution order.
SIGNED_TRAJECTORY_OWNER_METHODS = (_REPRESENTATION, _CANDIDATES, _STABILITY, _FIGURE)
#: Exact public inputs of the owners after the representation.  The host
#: projection is built from these, so the plan it emits and the contract it is
#: validated against cannot drift apart.
SIGNED_TRAJECTORY_CANDIDATE_INPUTS = (
    "artifact:trajectory_representation",
    "manifest:trajectory_representation_schema",
)
SIGNED_TRAJECTORY_STABILITY_INPUTS = (
    "artifact:trajectory_representation",
    "artifact:candidate_cluster_assignments",
    "manifest:cluster_selection",
    "manifest:trajectory_representation_schema",
    "manifest:candidate_cluster_solution_schema",
)
SIGNED_TRAJECTORY_FIGURE_INPUTS = (
    "table:trajectory_candidate_selection",
    "table:feature_availability",
    "table:trajectory_profiles",
    "table:cluster_sizes",
    "table:cluster_stability",
)
#: The long trajectory's stay identity.  The representation keys its rows by
#: it, so the stability owner's frozen labels carry it too.
SIGNED_TRAJECTORY_IDENTITY_COLUMN = "stay_id"


def signed_trajectory_plan_claimed(plan: object) -> bool:
    methods = [str(getattr(step, "method", "") or "") for step in getattr(plan, "steps", ()) or ()]
    return _REPRESENTATION in methods and bool(
        {_CANDIDATES, _MIXED_MODE_CANDIDATES} & set(methods)
    )


def _owner_steps(plan: object) -> tuple[Any, ...] | None:
    """The four owner steps in plan order, or None unless each occurs once, in order."""

    steps = tuple(getattr(plan, "steps", ()) or ())
    owners = tuple(
        step
        for step in steps
        if str(getattr(step, "method", "") or "")
        in {*SIGNED_TRAJECTORY_OWNER_METHODS, _MIXED_MODE_CANDIDATES}
    )
    methods = tuple(
        _CANDIDATES if method == _MIXED_MODE_CANDIDATES else method
        for method in (str(getattr(step, "method", "") or "") for step in owners)
    )
    return owners if methods == SIGNED_TRAJECTORY_OWNER_METHODS else None


def _frozen_class_description_errors(
    descriptions: Sequence[Any], roots: Sequence[Any]
) -> list[str]:
    """Admit one description of the frozen classes on the host-bound run cohort.

    The host wires it (``bind_plan``): it reads the cohort the run selected,
    republished byte for byte by the interpretation-free root, and the
    stability owner's labels and freeze record, and it feeds no owner.
    """

    if len(descriptions) != 1 or len(roots) != 1:
        return [
            "signed trajectory plan may describe the frozen classes once, "
            "on one host-bound run cohort"
        ]
    description, root = descriptions[0], roots[0]
    spec = getattr(description, "phenotype_comparison_spec", None)
    try:
        cohort_key = comparison_cohort_input(description)
        label_source = comparison_label_source(description)
    except ValueError:
        cohort_key = label_source = None
    if not (
        is_host_bound_cohort_publisher(root)
        and getattr(root, "planned_analysis_role", None) == "auxiliary"
        and label_source == TRAJECTORY_ASSIGNMENTS_PRODUCT
        and cohort_key == root.expected_outputs[0]
        and getattr(description, "planned_analysis_role", None) == "secondary"
        and list(getattr(description, "expected_outputs", ()) or ()) == [COMPARISON_PRODUCT]
        and getattr(spec, "identity_column", None) == SIGNED_TRAJECTORY_IDENTITY_COLUMN
    ):
        return [
            "signed trajectory description step "
            f"{getattr(description, 'step_id', '')!r} is not wired to the host-bound "
            "run cohort and the frozen trajectory labels"
        ]
    return []


def _companion_errors(plan: object, owners: Sequence[Any]) -> list[str]:
    """A companion may only render the owners' tables or describe their classes.

    The host appends deterministic renderers after binding, such as the cohort
    flow figure over the representation's ``table:cohort_flow``.  Such a step
    reads no row-level artifact, publishes no product an owner publishes, and
    claims no scientific role, so the signed decision stays closed.  The one
    description of the frozen classes is checked by its own wiring.
    """

    owner_ids = {id(step) for step in owners}
    owner_outputs = {
        str(value) for step in owners for value in getattr(step, "expected_outputs", ()) or ()
    }
    owner_tables = {value for value in owner_outputs if value.startswith("table:")}
    companions = [
        step for step in tuple(getattr(plan, "steps", ()) or ()) if id(step) not in owner_ids
    ]
    descriptions = [
        step
        for step in companions
        if getattr(step, "scientific_action_id", None) == COMPARISON_ACTION
    ]
    roots = [
        step
        for step in companions
        if str(getattr(step, "method", "") or "") == HOST_BOUND_COHORT_METHOD
    ]
    errors: list[str] = (
        _frozen_class_description_errors(descriptions, roots)
        if descriptions or roots
        else []
    )
    described = {id(step) for step in (*descriptions, *roots)}
    for step in companions:
        if id(step) in described:
            continue
        inputs = {str(value) for value in getattr(step, "inputs", ()) or ()}
        outputs = [str(value) for value in getattr(step, "expected_outputs", ()) or ()]
        if (
            getattr(step, "planned_analysis_role", None) != "auxiliary"
            or getattr(step, "trajectory_stability_spec", None) is not None
            or not inputs
            or not inputs <= owner_tables
            or not outputs
            or any(not value.startswith("figure:") for value in outputs)
            or any(value in owner_outputs for value in outputs)
        ):
            errors.append(
                "signed trajectory companion step "
                f"{getattr(step, 'step_id', '')!r} is not a render-only view of owner tables"
            )
    return errors


def signed_trajectory_plan_contract_errors(plan: object) -> list[str]:
    """Require one closed representation -> selection -> stability -> figure DAG.

    Host companions that only render the owners' tables may sit beside it.
    """

    owners = _owner_steps(plan)
    if owners is None:
        return ["signed trajectory plan does not contain the four ordered owners"]
    steps = owners
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
    if candidate_inputs != SIGNED_TRAJECTORY_CANDIDATE_INPUTS:
        errors.append("signed trajectory candidate owner has invalid inputs")
    stability_inputs = tuple(getattr(steps[2], "inputs", ()) or ())
    if stability_inputs != SIGNED_TRAJECTORY_STABILITY_INPUTS:
        errors.append("signed trajectory stability owner has invalid inputs")
    if getattr(steps[2], "trajectory_stability_spec", None) is None:
        errors.append("signed trajectory stability design is absent")
    figure_inputs = tuple(getattr(steps[3], "inputs", ()) or ())
    if figure_inputs != SIGNED_TRAJECTORY_FIGURE_INPUTS:
        errors.append("signed trajectory diagnostic figure has invalid inputs")
    errors.extend(_companion_errors(plan, owners))
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
    steps = _owner_steps(plan)
    assert steps is not None
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
    elif stability.get("scientific_status") == "failed_closed":
        # The selected solution failed its prespecified stability rule: the
        # owner completed, froze nothing, and states the rule's outcome.
        if not _stability_rejection_is_coherent(
            candidate=candidate, stability=stability, figure=figure, selected_k=selected_k
        ):
            errors.append("signed trajectory unstable-solution decision is incoherent")
    else:
        if not (
            candidate.get("scientific_status") == "selected"
            and candidate.get("stability_authorized") is True
            and stability.get("selected_n_clusters") == selected_k
            and int(stability.get("n_successful_resamples") or 0) > 0
            and stability.get("stability_threshold_passed") is not False
            and stability.get("outcome_binding_received_by_executor") is False
            and not stability.get("outcome_bindings_received")
            and figure.get("reportable_phenotype_solution") is not False
        ):
            errors.append("signed trajectory stable-solution decision is incoherent")
    return errors


def _stability_rejection_is_coherent(
    *,
    candidate: Mapping[str, Any],
    stability: Mapping[str, Any],
    figure: Mapping[str, Any],
    selected_k: int | None,
) -> bool:
    expected = _STABILITY_REJECTIONS.get(str(stability.get("reason_code") or ""))
    if expected is None:
        return False
    freeze_status, disposition = expected
    try:
        (outcome,) = [
            validate_rule_outcome(item)
            for item in stability.get(RULE_OUTCOMES_KEY) or ()
        ]
    except (TypeError, ValueError):
        return False
    return (
        candidate.get("scientific_status") == "selected"
        and candidate.get("stability_authorized") is True
        and stability.get("selected_n_clusters") == selected_k
        and stability.get("freeze_status") == freeze_status
        and stability.get("reportable_result") == "no_stable_phenotype_solution"
        and outcome.rule == "class_solution_stability"
        and outcome.disposition == disposition
        and outcome.selected_class_count == selected_k
        and outcome.successful_resamples == stability.get("n_successful_resamples")
        and stability.get("outcome_binding_received_by_executor") is False
        and not stability.get("outcome_bindings_received")
        and figure.get("reportable_phenotype_solution") is False
        and figure.get("scientific_status") == "failed_closed"
        and figure.get("reason_code") == TRAJECTORY_NO_SOLUTION_REASON
    )


__all__ = [
    "SIGNED_TRAJECTORY_CANDIDATE_INPUTS",
    "SIGNED_TRAJECTORY_FIGURE_INPUTS",
    "SIGNED_TRAJECTORY_IDENTITY_COLUMN",
    "SIGNED_TRAJECTORY_OWNER_METHODS",
    "SIGNED_TRAJECTORY_STABILITY_INPUTS",
    "signed_trajectory_plan_claimed",
    "signed_trajectory_plan_contract_errors",
    "signed_trajectory_runtime_bundle_errors",
]
