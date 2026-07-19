"""Authority and cross-step replay for canonical trajectory bundles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import easyicu.research_agent.trajectory_bundle as trajectory_bundle
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    ValidationFinding,
    VariableRole,
)
from easyicu.research_agent.trajectory_contract import (
    infer_fixed_window_trajectory_metadata,
)


_FILE_OWNER = {
    "trajectory_missingness_policy.json": "stability",
    "trajectory_membership.csv": "representation",
    "cohort_flow.csv": "representation",
    "cluster_selection.json": "selection",
    "cluster_assignments.csv": "stability",
    "cluster_stability.csv": "stability",
    "cluster_stability_assignments.csv": "stability",
    "trajectory_profiles.csv": "profiles",
    "cluster_sizes.csv": "profiles",
}
_OUTPUT = {
    "trajectory_missingness_policy.json": "manifest:trajectory_missingness_policy",
    "trajectory_membership.csv": "table:trajectory_membership",
    "cluster_assignments.csv": "table:cluster_assignments",
    "trajectory_profiles.csv": "table:trajectory_profiles",
    "cohort_flow.csv": "table:cohort_flow",
    "cluster_sizes.csv": "table:cluster_sizes",
    "cluster_stability.csv": "table:cluster_stability",
    "cluster_stability_assignments.csv": "table:cluster_stability_assignments",
    "cluster_selection.json": "manifest:cluster_selection",
}
_SELECTION = {
    "criterion": "agent criterion",
    "selection_rule": "minimum",
    "direction": "minimize",
    "selected_n_clusters": 2,
    "candidates": [
        {"n_clusters": 2, "criterion_value": 10.0},
        {"n_clusters": 3, "criterion_value": 12.0},
    ],
    "rationale": "Selected the finite minimum.",
}


def _context() -> ResearchContext:
    variables = []
    for start, end in ((0, 6), (6, 12)):
        name = f"unseen_burden_h{start}_{end}"
        variables.append(
            ConceptDescriptor(
                name=name,
                role=VariableRole.VITAL,
                dtype="float64",
                fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
                    column_name=name,
                    values=pd.Series([0.0, 0.5, 1.0]),
                    source_scale="continuous",
                ),
            )
        )
    return ResearchContext(
        research_question="Discover fixed-window phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="trajectory-bundle-test",
            database="synthetic",
            n_patients=3,
            n_stays=3,
            id_columns=["stay_id"],
        ),
        variables=variables,
    )


def _plan(*, split: bool) -> AnalysisPlan:
    raw_inputs = ["unseen_burden_h0_6", "unseen_burden_h6_12"]
    if not split:
        return AnalysisPlan(
            research_question="Validate one agent-owned trajectory bundle.",
            analysis_type="trajectory_clustering",
            steps=[
                AnalysisStep(
                    step_id="monolithic",
                    intent="Produce the full trajectory bundle.",
                    inputs=raw_inputs,
                    expected_outputs=[
                        "artifact:trajectory_features",
                        *list(_OUTPUT.values()),
                    ],
                    method="gaussian_mixture_model",
                )
            ],
        )
    return AnalysisPlan(
        research_question="Validate a split agent-owned trajectory bundle.",
        analysis_type="trajectory_clustering",
        steps=[
            AnalysisStep(
                step_id="representation",
                intent="Declare the representation and membership.",
                inputs=raw_inputs,
                expected_outputs=[
                    "artifact:trajectory_features",
                    *[
                        _OUTPUT[filename]
                        for filename, owner in _FILE_OWNER.items()
                        if owner == "representation"
                    ],
                ],
                method="fixed_anchor_missingness_aware_feature_representation",
            ),
            AnalysisStep(
                step_id="selection",
                intent="Select the agent-owned cluster solution.",
                inputs=["artifact:trajectory_features"],
                expected_outputs=[
                    "manifest:cluster_selection",
                    "artifact:candidate_cluster_fits",
                ],
                method="latent_class_trajectory_clustering",
            ),
            AnalysisStep(
                step_id="stability",
                intent="Freeze assignments and assess stability.",
                inputs=[
                    "artifact:trajectory_features",
                    "artifact:candidate_cluster_fits",
                ],
                expected_outputs=[
                    "artifact:stable_cluster_assignments",
                    *[
                        _OUTPUT[filename]
                        for filename, owner in _FILE_OWNER.items()
                        if owner == "stability"
                    ],
                ],
                method="bootstrap_cluster_stability",
            ),
            AnalysisStep(
                step_id="profiles",
                intent="Characterize the frozen solution.",
                inputs=["artifact:stable_cluster_assignments"],
                expected_outputs=[
                    _OUTPUT[filename]
                    for filename, owner in _FILE_OWNER.items()
                    if owner == "profiles"
                ],
                method="descriptive_cluster_characterization",
            ),
        ],
    )


def _summary(step_id: str) -> dict:
    if step_id in {"representation", "monolithic"}:
        payload = {"min_observed_windows": 1}
    else:
        payload = {}
    if step_id in {"selection", "monolithic"}:
        payload.update(
            {
                "cluster_selection": _SELECTION,
                "n_clusters": 2,
                "clustering_method": "agent_selected_model",
            }
        )
    return payload


def _register_bundle(
    tmp_path: Path,
    *,
    split: bool,
    omit: str | None = None,
) -> tuple[ResearchContext, AnalysisPlan, EvidenceStore, list[dict]]:
    context = _context()
    plan = _plan(split=split)
    evidence = EvidenceStore(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    evidence_ids_by_step = {step.step_id: [] for step in plan.steps}
    for filename, split_owner in _FILE_OWNER.items():
        source = source_dir / filename
        source.write_text("{}" if filename.endswith(".json") else "x\n1\n")
        if filename == omit:
            continue
        owner = split_owner if split else "monolithic"
        record = evidence.register_file(
            kind="log" if filename.endswith(".json") else "table",
            description=f"Canonical trajectory artifact {filename}.",
            source_path=source,
            produced_by_step=owner,
            evidence_id=f"evidence_{Path(filename).stem}",
        )
        evidence_ids_by_step[owner].append(record.evidence_id)
    records = [
        {
            "step_id": step.step_id,
            "status": "ok",
            "evidence_ids": evidence_ids_by_step[step.step_id],
            "step_summary": _summary(step.step_id),
        }
        for step in plan.steps
    ]
    return context, plan, evidence, records


def _call(
    tmp_path: Path,
    fixture: tuple[ResearchContext, AnalysisPlan, EvidenceStore, list[dict]],
) -> list[ValidationFinding]:
    context, plan, evidence, records = fixture
    return trajectory_bundle.trajectory_bundle_findings(
        context=context,
        plan=plan,
        per_step_records=records,
        evidence=evidence,
        run_dir=tmp_path,
        cohort_path=tmp_path / "cohort.parquet",
    )


@pytest.mark.parametrize("split", [False, True])
def test_plan_authority_binds_canonical_files_to_dag_role_owners(
    split: bool,
) -> None:
    context = _context()
    plan = _plan(split=split)

    authority = trajectory_bundle.resolve_trajectory_bundle_plan_authority(
        plan=plan,
        context=context,
    )

    assert authority.findings == ()
    expected_owners = {
        filename: owner if split else "monolithic"
        for filename, owner in _FILE_OWNER.items()
    }
    assert authority.owners == expected_owners


def test_role_owner_must_explicitly_declare_exact_typed_canonical_product() -> None:
    context = _context()
    plan = _plan(split=True)
    selection = next(step for step in plan.steps if step.step_id == "selection")
    selection.expected_outputs = [
        "table:cluster_selection" if output == "manifest:cluster_selection" else output
        for output in selection.expected_outputs
    ]

    authority = trajectory_bundle.resolve_trajectory_bundle_plan_authority(
        plan=plan,
        context=context,
    )

    assert any(
        finding.detail["kind"] == "missing_canonical_declaration"
        and finding.detail["canonical_file"] == "cluster_selection.json"
        and finding.detail["expected_owner_step_id"] == "selection"
        for finding in authority.findings
    )


def test_canonical_declaration_outside_its_dag_role_owner_fails_closed() -> None:
    context = _context()
    plan = _plan(split=True)
    stability = next(step for step in plan.steps if step.step_id == "stability")
    profiles = next(step for step in plan.steps if step.step_id == "profiles")
    profiles.expected_outputs.remove("table:cluster_sizes")
    stability.expected_outputs.append("table:cluster_sizes")

    authority = trajectory_bundle.resolve_trajectory_bundle_plan_authority(
        plan=plan,
        context=context,
    )

    assert any(
        finding.detail["kind"] == "canonical_declaration_owner_mismatch"
        and finding.detail["canonical_file"] == "cluster_sizes.csv"
        and finding.detail["expected_owner_step_id"] == "profiles"
        and finding.detail["declared_owner_step_id"] == "stability"
        for finding in authority.findings
    )


def test_cohort_flow_requires_one_unique_explicit_owner() -> None:
    context = _context()
    plan = _plan(split=True)
    selection = next(step for step in plan.steps if step.step_id == "selection")
    selection.expected_outputs.append("table:cohort_flow")

    authority = trajectory_bundle.resolve_trajectory_bundle_plan_authority(
        plan=plan,
        context=context,
    )

    assert any(
        finding.detail["kind"] == "ambiguous_plan_owner"
        and finding.detail["canonical_file"] == "cohort_flow.csv"
        for finding in authority.findings
    )


@pytest.mark.parametrize("split", [False, True])
def test_monolithic_and_split_bundles_stage_verified_current_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: bool,
) -> None:
    fixture = _register_bundle(tmp_path, split=split)
    seen: dict = {}

    def fake_replay(**kwargs):
        out_dir = kwargs["out_dir"]
        seen["files"] = sorted(path.name for path in out_dir.iterdir())
        seen["inputs"] = kwargs["step"].inputs
        seen["summary"] = kwargs["step_summary"]
        return []

    monkeypatch.setattr(
        trajectory_bundle,
        "trajectory_phenotyping_artifact_findings",
        fake_replay,
    )

    assert _call(tmp_path, fixture) == []
    assert seen["files"] == sorted(_FILE_OWNER)
    assert seen["inputs"] == ["unseen_burden_h0_6", "unseen_burden_h6_12"]
    assert seen["summary"] == {
        "status": "ok",
        "cluster_selection": _SELECTION,
        "n_clusters": 2,
        "clustering_method": "agent_selected_model",
        "min_observed_windows": 1,
    }


def test_replay_findings_are_run_level_with_all_contributors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _register_bundle(tmp_path, split=True)

    def fake_replay(**_kwargs):
        return [
            ValidationFinding(
                validator="trajectory_phenotyping_contract",
                severity="error",
                message="Numeric replay mismatch.",
                detail={"kind": "numeric_mismatch", "step_id": "synthetic"},
            )
        ]

    monkeypatch.setattr(
        trajectory_bundle,
        "trajectory_phenotyping_artifact_findings",
        fake_replay,
    )

    findings = _call(tmp_path, fixture)
    assert len(findings) == 1
    assert "step_id" not in findings[0].detail
    assert findings[0].detail["contributor_step_ids"] == [
        "representation",
        "selection",
        "stability",
        "profiles",
    ]
    assert len(findings[0].evidence_ids) == len(_FILE_OWNER)


def test_duplicate_current_evidence_fails_closed(tmp_path: Path) -> None:
    context, plan, evidence, records = _register_bundle(tmp_path, split=True)
    source = tmp_path / "duplicate" / "cluster_sizes.csv"
    source.parent.mkdir()
    source.write_text("cluster,n\n0,1\n")
    duplicate = evidence.register_file(
        kind="table",
        description="Duplicate current cluster sizes.",
        source_path=source,
        produced_by_step="stability",
        evidence_id="duplicate_cluster_sizes",
    )
    next(record for record in records if record["step_id"] == "stability")[
        "evidence_ids"
    ].append(duplicate.evidence_id)

    findings = _call(tmp_path, (context, plan, evidence, records))
    assert "duplicate_current_evidence" in {
        finding.detail["kind"] for finding in findings
    }


def test_later_failed_owner_cannot_reuse_prior_success_evidence(tmp_path: Path) -> None:
    context, plan, evidence, records = _register_bundle(tmp_path, split=True)
    records.append(
        {
            "step_id": "stability",
            "status": "contract_failed",
            "evidence_ids": [],
        }
    )

    findings = _call(tmp_path, (context, plan, evidence, records))
    missing = {
        finding.detail.get("canonical_file")
        for finding in findings
        if finding.detail["kind"] == "missing_current_evidence"
    }
    assert {
        "cluster_assignments.csv",
        "cluster_stability.csv",
        "cluster_stability_assignments.csv",
    } <= missing


def test_tampered_registered_file_fails_digest_authority(tmp_path: Path) -> None:
    context, plan, evidence, records = _register_bundle(tmp_path, split=True)
    record = evidence.get("evidence_cluster_sizes")
    assert record is not None
    (tmp_path / record.relative_path).write_text("tampered\n")

    findings = _call(tmp_path, (context, plan, evidence, records))
    assert any(
        finding.detail["kind"] == "evidence_digest_or_path_invalid"
        and finding.detail["canonical_file"] == "cluster_sizes.csv"
        for finding in findings
    )


def test_registered_file_from_the_wrong_plan_owner_fails_closed(
    tmp_path: Path,
) -> None:
    context, plan, evidence, records = _register_bundle(
        tmp_path,
        split=True,
        omit="trajectory_profiles.csv",
    )
    wrong_owner = evidence.register_file(
        kind="table",
        description="Profile table registered by the wrong step.",
        source_path=tmp_path / "source" / "trajectory_profiles.csv",
        produced_by_step="selection",
        evidence_id="wrong_owner_profiles",
    )
    next(record for record in records if record["step_id"] == "selection")[
        "evidence_ids"
    ].append(wrong_owner.evidence_id)

    findings = _call(tmp_path, (context, plan, evidence, records))
    assert any(
        finding.detail["kind"] == "evidence_owner_mismatch"
        and finding.detail["canonical_file"] == "trajectory_profiles.csv"
        for finding in findings
    )


def test_unregistered_or_missing_canonical_file_does_not_count(tmp_path: Path) -> None:
    fixture = _register_bundle(
        tmp_path,
        split=True,
        omit="trajectory_profiles.csv",
    )
    findings = _call(tmp_path, fixture)
    assert any(
        finding.detail["kind"] == "missing_current_evidence"
        and finding.detail["canonical_file"] == "trajectory_profiles.csv"
        for finding in findings
    )


def test_existing_numeric_replay_is_used_after_authority_passes(tmp_path: Path) -> None:
    fixture = _register_bundle(tmp_path, split=True)
    findings = _call(tmp_path, fixture)
    replay = next(
        finding
        for finding in findings
        if finding.validator == "trajectory_phenotyping_contract"
    )
    assert replay.detail["kind"] == "trajectory_cohort_unreadable"
    assert "step_id" not in replay.detail
    assert replay.detail["contributor_step_ids"] == [
        "representation",
        "selection",
        "stability",
        "profiles",
    ]
