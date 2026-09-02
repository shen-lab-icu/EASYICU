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
    expected_site_assignment,
    expected_site_assignment_sha256,
    validate_authorized_site_coordinates,
    validate_trajectory_lease,
)
from benchmarks.figure2_icu_agent_v2.formal_easyicu_runner import FormalEasyICURunner
from benchmarks.figure2_icu_agent_v2.formal_generic_runner import (
    FormalGenericCodeAgentRunner,
)
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    CANONICAL_FILES,
    ReviewBlindingContext,
    normalize_review_bundle,
)
from easyicu.research_agent.schema import PipelineResult
from benchmarks.figure2_icu_agent_v2.multi_host_acceptance import (
    MultiHostAcceptanceError,
    validate_two_host_preflight,
)


_BLINDING_CONTEXT = ReviewBlindingContext(
    host_markers=("fig2-server-01", "fig2-laptop-01"),
    output_roots=("/formal/server", "/formal/laptop"),
)


def _normalize(source_dir: Path):
    return normalize_review_bundle(
        source_dir,
        blinding_context=_BLINDING_CONTEXT,
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
    normalized = _normalize(output)
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
    runner._leased_output_dir = output.resolve()

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
    normalized = _normalize(output)
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
    runner._leased_output_dir = output.resolve()

    with pytest.raises(RuntimeError, match="private implementation detail"):
        runner.run_and_write_review_bundle(
            output_dir=output,
            mandatory_artifacts=("main result",),
            artifact_inventory={"main result": ()},
        )

    assert {path.name for path in output.iterdir()} == set(CANONICAL_FILES)
    normalized = _normalize(output)
    receipt = json.loads(normalized.files["07_run_receipt.json"])
    assert receipt["terminal_status"] == "failed"
    assert receipt["failure_category"] == "execution_failure"
    assert b"private implementation detail" not in b"".join(
        normalized.files.values()
    )


def test_formal_runners_reject_output_directory_outside_consumed_lease(
    tmp_path: Path,
) -> None:
    leased = (tmp_path / "leased-output").resolve()
    wrong = tmp_path / "wrong-output"

    easyicu = object.__new__(FormalEasyICURunner)
    easyicu._leased_output_dir = leased
    with pytest.raises(ValueError, match="consumed lease"):
        easyicu.run_and_write_review_bundle(
            output_dir=wrong,
            mandatory_artifacts=("result",),
            artifact_inventory={"result": ["03_results.json"]},
        )

    generic = object.__new__(FormalGenericCodeAgentRunner)
    generic._leased_output_dir = leased
    with pytest.raises(ValueError, match="consumed lease"):
        generic.run(
            task_prompt="offline",
            neutral_input_description="offline",
            mandatory_artifacts=("result",),
            output_dir=wrong,
            review_plan=lambda _plan: None,  # type: ignore[arg-type,return-value]
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
    assignment = expected_site_assignment("core_wp2_wp3")
    first, second = dry_run.trajectories[:2]
    lease_root = tmp_path / "server-leases"
    lease_root.mkdir()

    first_lease = claim_trajectory_lease(
        first,
        schedule=dry_run,
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
        site_assignment=assignment,
        expected_output_root=str(roots[first.execution_site].resolve()),
    )
    assert validated["output_dir"] == first.output_dir
    consumed = consume_trajectory_lease(
        first_lease,
        scope=first.scope,
        task_id=first.task_id,
        arm=first.arm,
        execution_site=first.execution_site,
        site_assignment=assignment,
        expected_output_root=str(roots[first.execution_site].resolve()),
    )
    assert consumed["output_dir"] == first.output_dir
    assert Path(f"{first_lease}.started").is_file()
    with pytest.raises(FileExistsError):
        consume_trajectory_lease(
            first_lease,
            scope=first.scope,
            task_id=first.task_id,
            arm=first.arm,
            execution_site=first.execution_site,
            site_assignment=assignment,
            expected_output_root=str(roots[first.execution_site].resolve()),
        )
    with pytest.raises(FormalScheduleError, match="execution_site mismatch"):
        validate_trajectory_lease(
            first_lease,
            scope=first.scope,
            task_id=first.task_id,
            arm=first.arm,
            execution_site="laptop",
            site_assignment=assignment,
            expected_output_root=str(roots["laptop"].resolve()),
        )
    with pytest.raises(FileExistsError):
        claim_trajectory_lease(
            first,
            schedule=dry_run,
            logical_site="server",
            lease_root=lease_root,
        )
    with pytest.raises(FormalScheduleError, match="first arm"):
        claim_trajectory_lease(
            second,
            schedule=dry_run,
            logical_site="server",
            lease_root=lease_root,
        )
    with pytest.raises(FormalScheduleError, match="another logical site"):
        claim_trajectory_lease(
            first,
            schedule=dry_run,
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
        schedule=dry_run,
        logical_site="server",
        lease_root=lease_root,
    )
    assert second_lease.is_file()


def test_manual_lease_cannot_override_frozen_core_assignment(tmp_path: Path) -> None:
    roots = {
        "server": tmp_path / "server-output",
        "laptop": tmp_path / "laptop-output",
    }
    dry_run = build_core_schedule_dry_run(roots)
    assignment = expected_site_assignment("core_wp2_wp3")
    trajectory = dry_run.trajectories[0]
    lease_root = tmp_path / "leases"
    lease_root.mkdir()
    legitimate = claim_trajectory_lease(
        trajectory,
        schedule=dry_run,
        logical_site=trajectory.execution_site,
        lease_root=lease_root,
    )
    payload = json.loads(legitimate.read_text(encoding="utf-8"))

    payload["execution_site"] = "laptop"
    forged_site = lease_root / "forged-site.lease.json"
    forged_site.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalScheduleError, match="frozen site mismatch"):
        validate_trajectory_lease(
            forged_site,
            scope=trajectory.scope,
            task_id=trajectory.task_id,
            arm=trajectory.arm,
            execution_site="laptop",
            site_assignment=assignment,
            expected_output_root=str(roots["laptop"].resolve()),
        )

    payload["task_id"] = "not_a_registered_core_task"
    payload["output_dir"] = "/dev/null"
    forged_task = lease_root / "forged-task.lease.json"
    forged_task.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalScheduleError, match="absent from core assignment"):
        validate_trajectory_lease(
            forged_task,
            scope=trajectory.scope,
            task_id="not_a_registered_core_task",
            arm=trajectory.arm,
            execution_site="laptop",
            site_assignment=assignment,
            expected_output_root=str(roots["laptop"].resolve()),
        )

    payload = json.loads(legitimate.read_text(encoding="utf-8"))
    payload["sequence_number"] = 9999
    forged_sequence = lease_root / "forged-sequence.lease.json"
    forged_sequence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalScheduleError, match="sequence number mismatch"):
        validate_trajectory_lease(
            forged_sequence,
            scope=trajectory.scope,
            task_id=trajectory.task_id,
            arm=trajectory.arm,
            execution_site=trajectory.execution_site,
            site_assignment=assignment,
            expected_output_root=str(roots[trajectory.execution_site].resolve()),
        )

    payload = json.loads(legitimate.read_text(encoding="utf-8"))
    payload["output_dir"] = str(
        tmp_path / "rogue" / trajectory.task_id / trajectory.arm
    )
    forged_output = lease_root / "forged-output.lease.json"
    forged_output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FormalScheduleError, match="output directory mismatch"):
        validate_trajectory_lease(
            forged_output,
            scope=trajectory.scope,
            task_id=trajectory.task_id,
            arm=trajectory.arm,
            execution_site=trajectory.execution_site,
            site_assignment=assignment,
            expected_output_root=str(roots[trajectory.execution_site].resolve()),
        )


def test_manual_qualification_lease_cannot_override_registered_assignment(
    tmp_path: Path,
) -> None:
    roots = {
        "server": tmp_path / "server-qualification",
        "laptop": tmp_path / "laptop-qualification",
    }
    task_ids = tuple(f"qualification_task_{index:02d}" for index in range(1, 13))
    dry_run = build_qualification_schedule_dry_run(task_ids, roots)
    assignment = expected_site_assignment("qualification12", task_ids=task_ids)
    trajectory = dry_run.trajectories[0]
    lease_root = tmp_path / "qualification-leases"
    lease_root.mkdir()
    legitimate = claim_trajectory_lease(
        trajectory,
        schedule=dry_run,
        logical_site=trajectory.execution_site,
        lease_root=lease_root,
    )
    payload = json.loads(legitimate.read_text(encoding="utf-8"))
    wrong_site = "laptop" if trajectory.execution_site == "server" else "server"
    payload["execution_site"] = wrong_site
    forged = lease_root / "forged-qualification-site.lease.json"
    forged.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FormalScheduleError, match="frozen site mismatch"):
        validate_trajectory_lease(
            forged,
            scope=trajectory.scope,
            task_id=trajectory.task_id,
            arm=trajectory.arm,
            execution_site=wrong_site,
            site_assignment=assignment,
            expected_output_root=str(roots[wrong_site].resolve()),
        )


def test_signed_core_coordinates_must_match_every_frozen_task_site_and_arm() -> None:
    assignment = expected_site_assignment("core_wp2_wp3")
    digest = expected_site_assignment_sha256("core_wp2_wp3")
    coordinates = [
        {
            "scope": "core_wp2_wp3",
            "task_id": item["task_id"],
            "arm": arm,
            "execution_site": item["execution_site"],
            "call_id": f"{item['task_id']}_{arm}",
        }
        for item in assignment
        for arm in ("easyicu_full", "generic_code_agent")
    ]

    assert validate_authorized_site_coordinates(
        "core_wp2_wp3",
        coordinates,
        declared_site_assignment_sha256=digest,
    ) == digest

    wrong_site = [dict(coordinate) for coordinate in coordinates]
    wrong_site[0]["execution_site"] = (
        "laptop" if wrong_site[0]["execution_site"] == "server" else "server"
    )
    with pytest.raises(FormalScheduleError, match="site assignment mismatch"):
        validate_authorized_site_coordinates(
            "core_wp2_wp3",
            wrong_site,
            declared_site_assignment_sha256=digest,
        )

    with pytest.raises(FormalScheduleError, match="both arms"):
        validate_authorized_site_coordinates(
            "core_wp2_wp3",
            coordinates[1:],
            declared_site_assignment_sha256=digest,
        )


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


def _receipt_bytes(receipt: dict) -> bytes:
    return json.dumps(
        receipt,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_two_host_preflight_requires_exact_runtime_parity() -> None:
    receipts = [
        _receipt_bytes(_site_receipt("server")),
        _receipt_bytes(_site_receipt("laptop")),
    ]

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
            [_receipt_bytes(receipt) for receipt in resource_drift],
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )

    provider_access = [_site_receipt("server"), _site_receipt("laptop")]
    provider_access[0]["provider_accessed"] = True
    with pytest.raises(MultiHostAcceptanceError, match="provider_accessed"):
        validate_two_host_preflight(
            [_receipt_bytes(receipt) for receipt in provider_access],
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )


def test_two_host_preflight_parses_raw_json_and_rejects_duplicate_keys() -> None:
    server = _receipt_bytes(_site_receipt("server"))
    laptop = _receipt_bytes(_site_receipt("laptop"))
    duplicate = server[:-1] + b',"logical_site":"laptop"}'

    with pytest.raises(MultiHostAcceptanceError, match="duplicate"):
        validate_two_host_preflight(
            [duplicate, laptop],
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )
    with pytest.raises(MultiHostAcceptanceError, match="unparsed JSON bytes"):
        validate_two_host_preflight(
            [_site_receipt("server"), laptop],  # type: ignore[list-item]
            expected_design_commit="c" * 40,
            expected_annotated_tag="figure2-v2.1-test",
        )

    nonfinite = server.replace(b'"clock_offset_ms":25', b'"clock_offset_ms":NaN')
    with pytest.raises(MultiHostAcceptanceError, match="nonfinite"):
        validate_two_host_preflight(
            [nonfinite, laptop],
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
