"""Run-level authority and replay for split trajectory-phenotyping bundles.

The agent may produce representation, selection, stability, assignments, and
profiles in one step or across several steps.  This module does not choose that
shape or any scientific method.  It resolves the agent-declared canonical
products through the current execution/evidence authority, stages only
digest-verified files, and delegates value-level checks to the existing
trajectory artifact validator.

The integration hook is intentionally run-level: call
``trajectory_bundle_findings`` after step execution, when the final
``per_step_records`` ledger and EvidenceStore are available.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..authority.evidence_store import EvidenceStore, sha256_of_file
from ..authority.runtime_artifacts import (
    current_evidence_records,
    current_successful_step_records,
    verified_run_evidence_path,
)
from ..schema import AnalysisPlan, AnalysisStep, ResearchContext, ValidationFinding
from .contract import (
    trajectory_phenotyping_artifact_findings,
    trajectory_phenotyping_contract_applies,
)
from .plan_contract import evaluate_trajectory_plan_dag

_CORE_CANONICAL_FILES: tuple[str, ...] = (
    "trajectory_missingness_policy.json",
    "trajectory_membership.csv",
    "cluster_assignments.csv",
    "trajectory_profiles.csv",
    "cohort_flow.csv",
    "cluster_sizes.csv",
    "cluster_stability.csv",
    "cluster_stability_assignments.csv",
    "cluster_selection.json",
)
_OUTCOME_CANONICAL_FILE = "outcome_by_cluster.csv"
_OUTCOME_PRODUCT_ALIASES = frozenset(
    {
        "outcome_by_cluster",
        "cluster_outcomes",
        "cluster_outcome_summary",
        "cluster_mortality",
    }
)
_CANONICAL_FILES = frozenset((*_CORE_CANONICAL_FILES, _OUTCOME_CANONICAL_FILE))
_EXPECTED_OUTPUT_KIND = {
    "trajectory_missingness_policy.json": "manifest",
    "trajectory_membership.csv": "table",
    "cluster_assignments.csv": "table",
    "trajectory_profiles.csv": "table",
    "cohort_flow.csv": "table",
    "cluster_sizes.csv": "table",
    "cluster_stability.csv": "table",
    "cluster_stability_assignments.csv": "table",
    "cluster_selection.json": "manifest",
    "outcome_by_cluster.csv": "table",
}
_EVIDENCE_KIND = {
    filename: "log" if filename.endswith(".json") else "table"
    for filename in _CANONICAL_FILES
}
_SUMMARY_GROUPS: dict[str, tuple[str, ...]] = {
    "cluster_selection": ("cluster_selection",),
    "n_clusters": ("n_clusters", "cluster_count"),
    "clustering_method": ("clustering_method", "algorithm"),
    "min_observed_windows": ("min_observed_windows",),
}
_ROLE_CANONICAL_FILES: dict[str, tuple[str, ...]] = {
    "representation": ("trajectory_membership.csv",),
    "candidate_selection": ("cluster_selection.json",),
    "stability_freeze": (
        # The policy binds representation choices to the finally selected
        # method/k, so a split DAG can only finalize it after selection.
        "trajectory_missingness_policy.json",
        "cluster_assignments.csv",
        "cluster_stability.csv",
        "cluster_stability_assignments.csv",
    ),
    "characterization": (
        "trajectory_profiles.csv",
        "cluster_sizes.csv",
    ),
}


@dataclass(frozen=True)
class TrajectoryBundlePlanAuthority:
    """Canonical bundle owners resolved from the agent-declared plan DAG."""

    applies: bool
    owners: Mapping[str, str]
    contributor_step_ids: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _declared_canonical_file(output: Any) -> str | None:
    kind, separator, raw_product = str(output or "").strip().lower().partition(":")
    if not separator:
        return None
    product = Path(raw_product).name
    if product != raw_product or "/" in raw_product or "\\" in raw_product:
        return None
    stem = Path(product).stem
    if stem in _OUTCOME_PRODUCT_ALIASES and kind in {
        "table",
        "dataset",
        "artifact",
        "manifest",
    }:
        return _OUTCOME_CANONICAL_FILE
    for filename in _CANONICAL_FILES:
        if stem != Path(filename).stem:
            continue
        if kind == _EXPECTED_OUTPUT_KIND[filename]:
            return filename
    return None


def _registered_basename(record: Any) -> str | None:
    relative = Path(str(_record_field(record, "relative_path") or ""))
    evidence_id = str(_record_field(record, "evidence_id") or "")
    prefix = f"{evidence_id}__"
    if not evidence_id or not relative.name.startswith(prefix):
        return None
    basename = relative.name[len(prefix) :]
    return basename if basename in _CANONICAL_FILES else None


def _finding(
    kind: str,
    message: str,
    *,
    contributor_step_ids: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    **detail: Any,
) -> ValidationFinding:
    return ValidationFinding(
        validator="trajectory_bundle_authority",
        severity="error",
        message=message,
        evidence_ids=list(dict.fromkeys(str(value) for value in evidence_ids if value)),
        detail={
            "kind": kind,
            "contributor_step_ids": sorted(
                {str(value) for value in contributor_step_ids if str(value).strip()}
            ),
            **detail,
        },
    )


def resolve_trajectory_bundle_plan_authority(
    *,
    plan: AnalysisPlan,
    context: ResearchContext,
) -> TrajectoryBundlePlanAuthority:
    """Bind canonical files to explicit declarations on their DAG-role owners.

    Role ownership comes from the agent-declared plan DAG.  It does not make a
    file authoritative by itself: every canonical product must also be
    explicitly declared with its required type on the corresponding owner.
    ``cohort_flow.csv`` is the exception to role mapping because cohort
    attrition may be emitted before or within trajectory representation; it
    still requires exactly one explicit typed declaration.
    """

    evaluation = evaluate_trajectory_plan_dag(plan=plan, context=context)
    if not evaluation.applies:
        finding = _finding(
            "trajectory_plan_dag_not_applicable",
            "The plan is not stamped as a fixed-window trajectory analysis.",
        )
        return TrajectoryBundlePlanAuthority(False, {}, (), (finding,))
    bundle_owner_findings = {
        "trajectory_role_product_owner_mismatch",
        "trajectory_typed_product_producer_ambiguous",
    }
    blocking_plan_findings = [
        item
        for item in evaluation.findings
        if str((item.detail or {}).get("kind") or "") not in bundle_owner_findings
    ]
    if blocking_plan_findings:
        contributor_step_ids = tuple(
            step.step_id
            for step in plan.steps or []
            if step.step_id in set(evaluation.role_owners.values())
        )
        finding = _finding(
            "trajectory_plan_dag_invalid",
            "Canonical trajectory authority cannot be resolved from an invalid plan DAG.",
            contributor_step_ids=contributor_step_ids,
            plan_finding_kinds=sorted(
                {
                    str((item.detail or {}).get("kind") or "")
                    for item in blocking_plan_findings
                    if str((item.detail or {}).get("kind") or "")
                }
            ),
        )
        return TrajectoryBundlePlanAuthority(
            True,
            {},
            contributor_step_ids,
            (finding,),
        )

    declared: dict[str, list[str]] = {filename: [] for filename in _CANONICAL_FILES}
    for step in plan.steps or []:
        for output in step.expected_outputs or []:
            filename = _declared_canonical_file(output)
            if filename is not None and step.step_id not in declared[filename]:
                declared[filename].append(step.step_id)

    owners: dict[str, str] = {}
    findings: list[ValidationFinding] = []
    role_by_file = {
        filename: role
        for role, filenames in _ROLE_CANONICAL_FILES.items()
        for filename in filenames
    }
    if declared[_OUTCOME_CANONICAL_FILE]:
        role_by_file[_OUTCOME_CANONICAL_FILE] = "characterization"

    for filename, role in role_by_file.items():
        candidates = declared[filename]
        expected_owner = evaluation.role_owners[role]
        if not candidates:
            findings.append(
                _finding(
                    "missing_canonical_declaration",
                    f"Trajectory bundle product {filename!r} is not explicitly declared on its plan role owner.",
                    contributor_step_ids=[expected_owner],
                    canonical_file=filename,
                    expected_role=role,
                    expected_owner_step_id=expected_owner,
                    required_output=(
                        f"{_EXPECTED_OUTPUT_KIND[filename]}:{Path(filename).stem}"
                    ),
                )
            )
        elif len(candidates) > 1:
            findings.append(
                _finding(
                    "ambiguous_plan_owner",
                    f"Trajectory bundle product {filename!r} has multiple plan owners.",
                    contributor_step_ids=candidates,
                    canonical_file=filename,
                    owner_step_ids=sorted(candidates),
                )
            )
        elif candidates[0] != expected_owner:
            findings.append(
                _finding(
                    "canonical_declaration_owner_mismatch",
                    f"Trajectory bundle product {filename!r} is declared outside its plan role owner.",
                    contributor_step_ids=[expected_owner, candidates[0]],
                    canonical_file=filename,
                    expected_role=role,
                    expected_owner_step_id=expected_owner,
                    declared_owner_step_id=candidates[0],
                )
            )
        else:
            owners[filename] = expected_owner

    flow_candidates = declared["cohort_flow.csv"]
    if not flow_candidates:
        findings.append(
            _finding(
                "missing_canonical_declaration",
                "Trajectory bundle product 'cohort_flow.csv' has no explicit typed declaration.",
                canonical_file="cohort_flow.csv",
                required_output="table:cohort_flow",
            )
        )
    elif len(flow_candidates) > 1:
        findings.append(
            _finding(
                "ambiguous_plan_owner",
                "Trajectory bundle product 'cohort_flow.csv' has multiple plan owners.",
                contributor_step_ids=flow_candidates,
                canonical_file="cohort_flow.csv",
                owner_step_ids=sorted(flow_candidates),
            )
        )
    else:
        owners["cohort_flow.csv"] = flow_candidates[0]

    owner_ids = set(owners.values())
    contributor_step_ids = tuple(
        step.step_id for step in plan.steps or [] if step.step_id in owner_ids
    )
    return TrajectoryBundlePlanAuthority(
        True,
        owners,
        contributor_step_ids,
        tuple(findings),
    )


def _summary_view(
    records_by_step: Mapping[str, Mapping[str, Any]],
    contributor_step_ids: Sequence[str],
) -> tuple[dict[str, Any], list[ValidationFinding]]:
    def fingerprint(field: str, value: Any) -> str:
        if field in {"n_clusters", "min_observed_windows"}:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(numeric):
                    return f"numeric:{numeric:g}"
        if field == "clustering_method":
            normalized = re.sub(
                r"[^a-z0-9]+", "_", str(value or "").strip().lower()
            ).strip("_")
            return f"method:{normalized}"
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)

    summary: dict[str, Any] = {"status": "ok"}
    findings: list[ValidationFinding] = []
    for canonical_key, aliases in _SUMMARY_GROUPS.items():
        reported: list[tuple[str, str, Any]] = []
        for step_id in contributor_step_ids:
            step_summary = records_by_step.get(step_id, {}).get("step_summary")
            if not isinstance(step_summary, Mapping):
                continue
            for alias in aliases:
                if alias in step_summary and step_summary[alias] is not None:
                    reported.append((step_id, alias, step_summary[alias]))
        if not reported:
            continue
        encodings = {fingerprint(canonical_key, value) for _, _, value in reported}
        if len(encodings) > 1:
            findings.append(
                _finding(
                    "conflicting_step_summaries",
                    f"Trajectory contributors disagree on {canonical_key!r}.",
                    contributor_step_ids=[
                        step_id for step_id, _alias, _value in reported
                    ],
                    summary_field=canonical_key,
                    reported=[
                        {"step_id": step_id, "key": alias, "value": value}
                        for step_id, alias, value in reported
                    ],
                )
            )
            continue
        summary[canonical_key] = reported[0][2]
    return summary, findings


def _run_level_replay_finding(
    finding: ValidationFinding,
    *,
    contributor_step_ids: Sequence[str],
    evidence_ids: Sequence[str],
) -> ValidationFinding:
    detail = dict(finding.detail or {})
    detail.pop("step_id", None)
    detail["contributor_step_ids"] = list(contributor_step_ids)
    merged_evidence_ids = list(
        dict.fromkeys([*(finding.evidence_ids or []), *evidence_ids])
    )
    return finding.model_copy(
        update={"detail": detail, "evidence_ids": merged_evidence_ids}
    )


def trajectory_bundle_findings(
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    per_step_records: Sequence[Mapping[str, Any]],
    evidence: EvidenceStore,
    run_dir: Path,
    cohort_path: Path,
) -> list[ValidationFinding]:
    """Validate one current, registered trajectory bundle across plan steps.

    Missing, duplicate, failed/stale, tampered, undeclared, or wrong-owner
    artifacts fail before value replay.  The returned findings are run-level:
    no single ``detail.step_id`` is asserted for a cross-step bundle.
    """

    plan_authority = resolve_trajectory_bundle_plan_authority(
        plan=plan,
        context=context,
    )
    if plan_authority.findings:
        return list(plan_authority.findings)
    owners = dict(plan_authority.owners)
    findings: list[ValidationFinding] = []

    current_successful = current_successful_step_records(per_step_records)
    records_by_step = {
        str(record.get("step_id") or ""): record
        for record in current_successful
        if str(record.get("step_id") or "").strip()
    }
    required_files = list(owners)
    current_records = current_evidence_records(evidence.records(), per_step_records)
    candidates: dict[str, list[Any]] = {filename: [] for filename in required_files}
    unexpected_canonical: list[tuple[str, Any]] = []
    for record in current_records:
        filename = _registered_basename(record)
        if filename is None:
            continue
        if filename in candidates:
            candidates[filename].append(record)
        else:
            unexpected_canonical.append((filename, record))

    for filename, record in unexpected_canonical:
        findings.append(
            _finding(
                "undeclared_canonical_artifact",
                f"Current evidence contains undeclared trajectory product {filename!r}.",
                contributor_step_ids=[
                    str(_record_field(record, "produced_by_step") or "")
                ],
                evidence_ids=[str(_record_field(record, "evidence_id") or "")],
                canonical_file=filename,
            )
        )

    verified: dict[str, tuple[Any, Path]] = {}
    for filename in required_files:
        records = candidates[filename]
        if not records:
            findings.append(
                _finding(
                    "missing_current_evidence",
                    f"Trajectory bundle product {filename!r} has no current evidence.",
                    contributor_step_ids=[owners[filename]],
                    canonical_file=filename,
                    owner_step_id=owners[filename],
                )
            )
            continue
        if len(records) > 1:
            findings.append(
                _finding(
                    "duplicate_current_evidence",
                    f"Trajectory bundle product {filename!r} has duplicate current evidence.",
                    contributor_step_ids=[
                        str(_record_field(record, "produced_by_step") or "")
                        for record in records
                    ],
                    evidence_ids=[
                        str(_record_field(record, "evidence_id") or "")
                        for record in records
                    ],
                    canonical_file=filename,
                )
            )
            continue
        record = records[0]
        evidence_id = str(_record_field(record, "evidence_id") or "")
        producer = str(_record_field(record, "produced_by_step") or "")
        if producer != owners[filename] or producer not in records_by_step:
            findings.append(
                _finding(
                    "evidence_owner_mismatch",
                    f"Trajectory bundle product {filename!r} is not bound to its current plan owner.",
                    contributor_step_ids=[owners[filename], producer],
                    evidence_ids=[evidence_id],
                    canonical_file=filename,
                    expected_owner_step_id=owners[filename],
                    produced_by_step=producer or None,
                )
            )
            continue
        expected_kind = _EVIDENCE_KIND[filename]
        reported_kind = str(_record_field(record, "kind") or "")
        if reported_kind != expected_kind:
            findings.append(
                _finding(
                    "evidence_kind_mismatch",
                    f"Trajectory bundle product {filename!r} has the wrong evidence kind.",
                    contributor_step_ids=[producer],
                    evidence_ids=[evidence_id],
                    canonical_file=filename,
                    expected_kind=expected_kind,
                    reported_kind=reported_kind,
                )
            )
            continue
        path = verified_run_evidence_path(run_dir, record)
        if path is None:
            findings.append(
                _finding(
                    "evidence_digest_or_path_invalid",
                    f"Trajectory bundle product {filename!r} failed path/digest verification.",
                    contributor_step_ids=[producer],
                    evidence_ids=[evidence_id],
                    canonical_file=filename,
                )
            )
            continue
        verified[filename] = (record, path)

    if findings:
        return findings

    plan_by_step = {step.step_id: step for step in plan.steps or []}
    contributor_step_ids = list(plan_authority.contributor_step_ids)
    raw_inputs = list(
        dict.fromkeys(
            value
            for step_id in contributor_step_ids
            for value in (plan_by_step[step_id].inputs or [])
            if (
                (variable := context.variable(str(value))) is not None
                and variable.fixed_window_trajectory is not None
            )
        )
    )
    expected_outputs = list(
        dict.fromkeys(
            value
            for step_id in contributor_step_ids
            for value in (plan_by_step[step_id].expected_outputs or [])
        )
    )
    synthetic_step = AnalysisStep(
        step_id="trajectory_bundle_validation",
        intent="Replay the agent-declared cross-step trajectory bundle.",
        inputs=raw_inputs,
        expected_outputs=expected_outputs,
        method="trajectory_clustering",
    )
    if not trajectory_phenotyping_contract_applies(
        context=context,
        step=synthetic_step,
    ):
        return [
            _finding(
                "missing_trajectory_source_inputs",
                "Trajectory bundle contributors do not declare at least two ordered fixed-window source inputs from one family for replay.",
                contributor_step_ids=contributor_step_ids,
                declared_trajectory_inputs=raw_inputs,
            )
        ]

    summary, summary_findings = _summary_view(
        records_by_step,
        contributor_step_ids,
    )
    if summary_findings:
        return summary_findings

    bundle_evidence_ids = [
        str(_record_field(verified[filename][0], "evidence_id") or "")
        for filename in required_files
    ]
    with tempfile.TemporaryDirectory(prefix="easyicu-trajectory-bundle-") as tmp:
        out_dir = Path(tmp) / "outputs"
        out_dir.mkdir()
        for filename, (record, source) in verified.items():
            target = out_dir / filename
            shutil.copyfile(source, target)
            if sha256_of_file(target) != str(_record_field(record, "sha256") or ""):
                return [
                    _finding(
                        "staged_digest_mismatch",
                        f"Trajectory bundle product {filename!r} changed while being staged.",
                        contributor_step_ids=contributor_step_ids,
                        evidence_ids=[str(_record_field(record, "evidence_id") or "")],
                        canonical_file=filename,
                    )
                ]
        replay_findings = trajectory_phenotyping_artifact_findings(
            context=context,
            cohort_path=cohort_path,
            step=synthetic_step,
            out_dir=out_dir,
            step_summary=summary,
        )

    return [
        _run_level_replay_finding(
            finding,
            contributor_step_ids=contributor_step_ids,
            evidence_ids=bundle_evidence_ids,
        )
        for finding in replay_findings
    ]


__all__ = [
    "TrajectoryBundlePlanAuthority",
    "resolve_trajectory_bundle_plan_authority",
    "trajectory_bundle_findings",
]
