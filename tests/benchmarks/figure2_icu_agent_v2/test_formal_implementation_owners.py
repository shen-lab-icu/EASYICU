from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_icu_agent_v2.blinded_evaluator import (
    BlindedEvaluationError,
    instantiate_review_sheet,
    lock_blinded_scores,
)
from benchmarks.figure2_icu_agent_v2.easyicu_review_bundle_adapter import (
    EasyICUReviewMaterial,
    write_easyicu_review_bundle,
)
from benchmarks.figure2_icu_agent_v2.formal_scheduler import (
    FormalScheduleError,
    build_core_schedule_dry_run,
    build_qualification_schedule_dry_run,
    claim_trajectory_lease,
    consume_trajectory_lease,
    validate_trajectory_lease,
)
from benchmarks.figure2_icu_agent_v2.formal_easyicu_runner import FormalEasyICURunner
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    CANONICAL_FILES,
    normalize_review_bundle,
)
from easyicu.research_agent.schema import PipelineResult
from benchmarks.figure2_icu_agent_v2.multi_host_acceptance import (
    MultiHostAcceptanceError,
    validate_two_host_preflight,
)


def _native_pipeline_result(root: Path) -> PipelineResult:
    root.mkdir()
    artifacts = {
        "plan.json": {"population": "adult ICU"},
        "research_context.json": {"cohort": {"n": 42}},
        "manifest.json": {
            "evidence": [{"evidence_id": "result-1", "sha256": "a" * 64}],
            "findings": [],
        },
        "run_status.json": {"terminal": True, "gates_passed": True},
    }
    for name, value in artifacts.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "results_report.md").write_text("The estimate was 1.2.\n")
    (root / "manuscript.md").write_text("Manuscript.\n")
    return PipelineResult(
        run_id="run-1",
        workdir=str(root),
        context_path=str(root / "research_context.json"),
        plan_path=str(root / "plan.json"),
        manifest_path=str(root / "manifest.json"),
        report_path=str(root / "results_report.md"),
        manuscript_path=str(root / "manuscript.md"),
        evidence_count=1,
        findings_count=0,
    )
def test_easyicu_adapter_emits_normalizable_arm_neutral_bundle(tmp_path: Path) -> None:
    material = EasyICUReviewMaterial(
        plan={"population": "adult ICU"},
        cohort={"n": 42},
        results={"estimate": 1.2},
        diagnostics={"complete": True},
        report="The estimate was 1.2.",
        headline_evidence=({"claim": "estimate", "file": "03_results.json"},),
        artifact_inventory={"main result": ["03_results.json", "06_report.md"]},
    )
    output = write_easyicu_review_bundle(
        material,
        output_dir=tmp_path / "bundle",
        mandatory_artifacts=("main result",),
        resource_receipt={"within_frozen_budget": True, "provider_tokens": 10},
    )

    assert {path.name for path in output.iterdir()} == set(CANONICAL_FILES)
    normalized = normalize_review_bundle(output)
    receipt = json.loads(normalized.files["07_run_receipt.json"])
    assert receipt["substantive_output_files"]["03_results.json"] is True
    assert "provider_tokens" not in receipt


def test_easyicu_runner_projects_native_terminal_outputs_without_postrun_seam(
    tmp_path: Path,
) -> None:
    result = _native_pipeline_result(tmp_path / "native-run")

    class _Pipeline:
        def run(self, **kwargs):
            assert kwargs == {"cohort": "sealed-input.parquet"}
            return result

    class _HardStop:
        def assert_active(self):
            return 12.0

        def accounting_summary(self):
            return {"provider_reported": {"n_calls": 3}}

    runner = object.__new__(FormalEasyICURunner)
    runner._pipeline = _Pipeline()
    runner._provider_hard_stop = _HardStop()
    output = tmp_path / "review-bundle"

    returned = runner.run_and_write_review_bundle(
        output_dir=output,
        mandatory_artifacts=("main result",),
        artifact_inventory={
            "main result": ["03_results.json", "06_report.md"]
        },
        cohort="sealed-input.parquet",
    )

    assert returned is result
    assert {path.name for path in output.iterdir()} == set(CANONICAL_FILES)
    normalized = normalize_review_bundle(output)
    assert json.loads(normalized.files["03_results.json"])["evidence"][0][
        "evidence_id"
    ] == "result-1"


def test_easyicu_runner_writes_neutral_terminal_bundle_on_execution_failure(
    tmp_path: Path,
) -> None:
    class _Pipeline:
        def run(self, **kwargs):
            raise RuntimeError("private implementation detail")

    class _HardStop:
        def accounting_summary(self):
            return {"provider_reported": {"n_calls": 1}}

    runner = object.__new__(FormalEasyICURunner)
    runner._pipeline = _Pipeline()
    runner._provider_hard_stop = _HardStop()
    output = tmp_path / "failed-review-bundle"

    with pytest.raises(RuntimeError, match="private implementation detail"):
        runner.run_and_write_review_bundle(
            output_dir=output,
            mandatory_artifacts=("main result",),
            artifact_inventory={"main result": ()},
        )

    assert {path.name for path in output.iterdir()} == set(CANONICAL_FILES)
    normalized = normalize_review_bundle(output)
    receipt = json.loads(normalized.files["07_run_receipt.json"])
    assert receipt["terminal_status"] == "failed"
    assert receipt["failure_category"] == "execution_failure"
    assert b"private implementation detail" not in b"".join(
        normalized.files.values()
    )


def test_core_scheduler_projects_78_unique_trajectories_without_writes(
    tmp_path: Path,
) -> None:
    roots = {
        "server": tmp_path / "server-output",
        "laptop": tmp_path / "laptop-output",
    }
    dry_run = build_core_schedule_dry_run(roots)

    assert dry_run.scope == "core_wp2_wp3"
    assert dry_run.trajectory_count == 78
    assert dry_run.provider_accessed is False
    assert dry_run.site_pair_counts == {"server": 20, "laptop": 19}
    assert len({(row.task_id, row.arm) for row in dry_run.trajectories}) == 78
    assert not any(root.exists() for root in roots.values())
    for index in range(0, 78, 2):
        first, second = dry_run.trajectories[index : index + 2]
        assert first.task_id == second.task_id
        assert first.execution_site == second.execution_site
        assert second.predecessor_output_dir == first.output_dir

    occupied = Path(dry_run.trajectories[0].output_dir)
    occupied.mkdir(parents=True)
    with pytest.raises(FormalScheduleError):
        build_core_schedule_dry_run(roots)


def test_qualification_scheduler_is_deterministic_balanced_and_pair_local(
    tmp_path: Path,
) -> None:
    roots = {
        "server": tmp_path / "server-qualification",
        "laptop": tmp_path / "laptop-qualification",
    }
    task_ids = tuple(f"qualification_task_{index:02d}" for index in range(1, 13))

    first = build_qualification_schedule_dry_run(task_ids, roots)
    second = build_qualification_schedule_dry_run(tuple(reversed(task_ids)), roots)

    assert first.scope == "qualification12"
    assert first.trajectory_count == 24
    assert first.site_pair_counts == {"server": 6, "laptop": 6}
    assert first.site_assignment_sha256 == second.site_assignment_sha256
    assert first.trajectories == second.trajectories
    for site in roots:
        first_arms = [
            row.arm
            for row in first.trajectories[::2]
            if row.execution_site == site
        ]
        assert first_arms.count("easyicu_full") == 3
        assert first_arms.count("generic_code_agent") == 3


def test_site_assignment_lease_is_single_use_and_pair_ordered(tmp_path: Path) -> None:
    roots = {
        "server": tmp_path / "server-output",
        "laptop": tmp_path / "laptop-output",
    }
    dry_run = build_core_schedule_dry_run(roots)
    first, second = dry_run.trajectories[:2]
    lease_root = tmp_path / "server-leases"
    lease_root.mkdir()

    first_lease = claim_trajectory_lease(
        first,
        logical_site="server",
        lease_root=lease_root,
    )
    assert first_lease.is_file()
    validated = validate_trajectory_lease(
        first_lease,
        scope=first.scope,
        task_id=first.task_id,
        arm=first.arm,
        execution_site=first.execution_site,
    )
    assert validated["output_dir"] == first.output_dir
    started = consume_trajectory_lease(
        first_lease,
        scope=first.scope,
        task_id=first.task_id,
        arm=first.arm,
        execution_site=first.execution_site,
    )
    assert started.is_file()
    with pytest.raises(FileExistsError):
        consume_trajectory_lease(
            first_lease,
            scope=first.scope,
            task_id=first.task_id,
            arm=first.arm,
            execution_site=first.execution_site,
        )
    with pytest.raises(FormalScheduleError, match="execution_site mismatch"):
        validate_trajectory_lease(
            first_lease,
            scope=first.scope,
            task_id=first.task_id,
            arm=first.arm,
            execution_site="laptop",
        )
    with pytest.raises(FileExistsError):
        claim_trajectory_lease(
            first,
            logical_site="server",
            lease_root=lease_root,
        )
    with pytest.raises(FormalScheduleError, match="first arm"):
        claim_trajectory_lease(
            second,
            logical_site="server",
            lease_root=lease_root,
        )
    with pytest.raises(FormalScheduleError, match="another logical site"):
        claim_trajectory_lease(
            first,
            logical_site="laptop",
            lease_root=lease_root,
        )

    predecessor = Path(first.output_dir)
    predecessor.mkdir(parents=True)
    for name in CANONICAL_FILES:
        payload = {"terminal_status": "completed"} if name == "07_run_receipt.json" else {}
        (predecessor / name).write_text(json.dumps(payload), encoding="utf-8")
    second_lease = claim_trajectory_lease(
        second,
        logical_site="server",
        lease_root=lease_root,
    )
    assert second_lease.is_file()


def _site_receipt(site: str) -> dict:
    return {
        "schema_version": "easyicu.figure2_site_preflight/1",
        "logical_site": site,
        "host_fingerprint_sha256": ("a" if site == "server" else "b") * 64,
        "clock_offset_ms": 25,
        "design_commit": "c" * 40,
        "annotated_tag": "figure2-v2.1-test",
        "container_image_digest": "sha256:" + "d" * 64,
        "package_lock_sha256": "e" * 64,
        "provider_route_sha256": "f" * 64,
        "immutable_model_identifier": "shared-model-immutable-v1",
        "sampling_policy_sha256": "1" * 64,
        "runtime_budget_sha256": "2" * 64,
        "network_policy_sha256": "3" * 64,
        "input_manifest_set_sha256": "4" * 64,
        "cpu_limit": 2,
        "memory_limit_bytes": 8 * 1024**3,
        "pids_limit": 256,
        "clean_exact_head": True,
        "container_limits_enforced": True,
        "repository_access_denied": True,
        "undeclared_network_denied": True,
        "output_root_empty": True,
        "clock_synchronized": True,
        "provider_accessed": False,
    }


def test_two_host_preflight_requires_exact_runtime_parity() -> None:
    receipts = [_site_receipt("server"), _site_receipt("laptop")]

    result = validate_two_host_preflight(
        receipts,
        expected_design_commit="c" * 40,
        expected_annotated_tag="figure2-v2.1-test",
    )

    assert result["status"] == "passed"
    assert result["provider_accessed"] is False
    assert result["logical_sites"] == ["server", "laptop"]


def test_two_host_preflight_fails_on_resource_or_provider_drift() -> None:
    resource_drift = [_site_receipt("server"), _site_receipt("laptop")]
    resource_drift[1]["cpu_limit"] = 4
    with pytest.raises(MultiHostAcceptanceError, match="cpu_limit"):
        validate_two_host_preflight(
            resource_drift,
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )

    provider_access = [_site_receipt("server"), _site_receipt("laptop")]
    provider_access[0]["provider_accessed"] = True
    with pytest.raises(MultiHostAcceptanceError, match="provider_accessed"):
        validate_two_host_preflight(
            provider_access,
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )


def _score(task_id: str, reviewer: str, bundle: str, success: bool) -> dict:
    return {
        "reviewer_id": reviewer,
        "task_id": task_id,
        "bundle_id": bundle,
        "primary_success": success,
        "hard_gates_passed": {
            f"HG{index:02d}_{suffix}": success
            for index, suffix in (
                (1, "TERMINAL_INTEGRITY"),
                (2, "POPULATION_TIME_AUTHORITY"),
                (3, "TASK_SEMANTIC_GUARDRAILS"),
                (4, "MANDATORY_OUTPUTS"),
                (5, "DIAGNOSTIC_AUTHORITY"),
                (6, "EVIDENCE_BINDING"),
                (7, "INTERPRETATION_CEILING"),
                (8, "CONTAMINATION_AND_REPAIR"),
                (9, "TASK_QUESTION_ANSWERED"),
            )
        },
        "dimension_scores": {
            name: 2 if success else 0
            for name in (
                "problem_formulation",
                "literature_grounding",
                "data_concept_cohort_authority",
                "estimand_method_selection",
                "execution_validity",
                "diagnostics_sensitivity",
                "evidence_artifact_binding",
                "interpretation_safety",
                "reproducibility_efficiency",
            )
        },
        "arm_guess": "cannot_tell",
        "rationale": "Frozen criteria were applied independently.",
    }


def test_blinded_scores_lock_before_arm_mapping_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    task_id = "icu27_t01"
    sheet = instantiate_review_sheet(task_id)
    reviews = [
        _score(task_id, reviewer, bundle, True)
        for reviewer in ("clinical-r1", "methods-r2")
        for bundle in ("bundle_1", "bundle_2")
    ]
    destination = tmp_path / "locked.json"
    receipt = lock_blinded_scores(
        task_id=task_id,
        sheet_sha256=sheet["sheet_sha256"],
        reviews=reviews,
        eligible_reviewer_ids=("clinical-r1", "methods-r2"),
        reviewer_eligibility_receipt_sha256="a" * 64,
        destination=destination,
    )

    assert receipt["arm_mapping_present"] is False
    assert receipt["adjudication_required"] == {
        "bundle_1": False,
        "bundle_2": False,
    }
    with pytest.raises(FileExistsError):
        lock_blinded_scores(
            task_id=task_id,
            sheet_sha256=sheet["sheet_sha256"],
            reviews=reviews,
            eligible_reviewer_ids=("clinical-r1", "methods-r2"),
            reviewer_eligibility_receipt_sha256="a" * 64,
            destination=destination,
        )


def test_blinded_primary_cannot_disagree_with_hard_gates(tmp_path: Path) -> None:
    task_id = "icu27_t01"
    sheet = instantiate_review_sheet(task_id)
    reviews = [
        _score(task_id, reviewer, bundle, True)
        for reviewer in ("clinical-r1", "methods-r2")
        for bundle in ("bundle_1", "bundle_2")
    ]
    reviews[0]["hard_gates_passed"]["HG04_MANDATORY_OUTPUTS"] = False
    with pytest.raises(BlindedEvaluationError):
        lock_blinded_scores(
            task_id=task_id,
            sheet_sha256=sheet["sheet_sha256"],
            reviews=reviews,
            eligible_reviewer_ids=("clinical-r1", "methods-r2"),
            reviewer_eligibility_receipt_sha256="a" * 64,
            destination=tmp_path / "invalid.json",
        )
