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
)
from benchmarks.figure2_icu_agent_v2.review_bundle_normalizer import (
    CANONICAL_FILES,
    normalize_review_bundle,
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


def test_core_scheduler_projects_78_unique_trajectories_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-output"
    dry_run = build_core_schedule_dry_run(root)

    assert dry_run.core_trajectory_count == 78
    assert dry_run.provider_accessed is False
    assert len({(row.task_id, row.arm) for row in dry_run.trajectories}) == 78
    assert not root.exists()

    occupied = Path(dry_run.trajectories[0].output_dir)
    occupied.mkdir(parents=True)
    with pytest.raises(FormalScheduleError):
        build_core_schedule_dry_run(root)


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
