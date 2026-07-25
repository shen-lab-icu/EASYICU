from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest

import easyicu.research_agent.orchestration.resume as pipeline_resume
from easyicu.research_agent.orchestration.resume import (
    ResumeController,
    clear_quarantined_concept_draft,
    load_quarantined_concept_draft,
    store_quarantined_concept_draft,
    upsert_step_record,
)
from easyicu.research_agent.contracts.runtime import ValidationFinding
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Resume policy test.",
        steps=[
            AnalysisStep(
                step_id="01_define",
                intent="Define the cohort.",
                expected_outputs=["table:cohort"],
            ),
            AnalysisStep(
                step_id="02_model",
                intent="Fit the model.",
                expected_outputs=["table:model"],
            ),
            AnalysisStep(
                step_id="03_figure",
                intent="Render the figure.",
                expected_outputs=["figure:model"],
            ),
        ],
    )


def test_resume_controller_drops_requested_step_and_stale_findings(tmp_path: Path):
    state = {
        "per_step_records": [
            {
                "step_id": "00_probe",
                "status": "ok",
                "step_summary": {"n_rows": 10},
            },
            {"step_id": "01_define", "status": "ok"},
            {"step_id": "02_model", "status": "ok"},
            {"step_id": "03_figure", "status": "ok"},
            {"step_id": "04_unplanned", "status": "ok"},
            {"step_id": "02_model", "status": "failed"},
        ],
        "findings": [
            {
                "validator": "step_contract",
                "severity": "error",
                "message": "stale issue for 02_model",
                "detail": {"step_id": "02_model"},
            },
            {
                "validator": "manuscript_gate",
                "severity": "error",
                "message": "stale downstream issue",
                "detail": {"failed_steps": [{"step_id": "03_figure"}]},
            },
            {
                "validator": "clinical",
                "severity": "warning",
                "message": "keep upstream warning",
                "detail": {"step_id": "01_define"},
            },
            {
                "validator": "cohort_auditor",
                "severity": "warning",
                "message": "legacy cohort warning is recomputed on resume",
            },
            {
                "validator": "runner",
                "severity": "error",
                "message": "prior runner failure for 01_define should clear",
            },
        ],
    }

    applied = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
        stop_after_step_id="02_model",
    ).apply()

    assert applied.resumed_step_ids == {"00_probe", "01_define"}
    assert [record["step_id"] for record in applied.per_step_records] == [
        "00_probe",
        "01_define",
    ]
    assert applied.probe_summary == {"n_rows": 10}
    messages = [finding.message for finding in applied.findings]
    assert "keep upstream warning" in messages
    assert not any("stale" in message for message in messages)
    resume_findings = [
        finding for finding in applied.findings if finding.validator == "resume"
    ]
    assert resume_findings
    assert resume_findings[-1].detail["dropped_completed_step_ids"] == [
        "03_figure",
        "04_unplanned",
    ]


def test_pipeline_resume_entrypoints_are_importable() -> None:
    from easyicu.research_agent.orchestration.resume import ResumeApplication

    assert ResumeController.__name__ == "ResumeController"
    assert ResumeApplication.__name__ == "ResumeApplication"
    assert callable(upsert_step_record)


def test_resume_exposes_latest_negative_critic_report_for_selected_step(tmp_path):
    state = {
        "per_step_records": [
            {
                "step_id": "02_model",
                "status": "critic_failed",
                "critique_report": {
                    "status": "needs_revision",
                    "concerns": ["Preserve the typed cohort binding."],
                },
            },
            {
                "step_id": "02_model",
                "status": "critic_failed",
                "critique_report": {
                    "status": "needs_revision",
                    "concerns": ["Keep observation counts audit-only."],
                },
            },
        ]
    }
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
    )

    report = controller.prior_negative_critic_report_for_step("02_model")

    assert report is not None
    assert report["concerns"] == ["Keep observation counts audit-only."]
    assert controller.prior_negative_critic_report_for_step("01_define") is None


def test_finding_step_match_uses_exact_identifier_boundaries() -> None:
    finding = ValidationFinding(
        validator="runner",
        severity="error",
        message="Failure belongs to 02_model_figure, not its parent.",
    )

    assert not ResumeController._finding_mentions_step(finding, {"02_model"})
    assert ResumeController._finding_mentions_step(finding, {"02_model_figure"})


def test_resume_runner_finding_uses_exact_step_identifier_boundary(
    tmp_path: Path,
) -> None:
    plan = AnalysisPlan(
        research_question="Keep distinct step identifiers distinct.",
        steps=[
            AnalysisStep(step_id="step1", intent="First step."),
            AnalysisStep(step_id="step10", intent="Tenth step."),
        ],
    )
    state = {
        "per_step_records": [{"step_id": "step1", "status": "ok"}],
        "findings": [
            ValidationFinding(
                validator="runner",
                severity="error",
                message="Runner failed for step10.",
            ).model_dump(mode="json")
        ],
    }

    application = ResumeController(
        plan=plan,
        run_dir=tmp_path,
        resume_state=state,
    ).apply()

    assert [
        (finding.validator, finding.message) for finding in application.findings
    ] == [("runner", "Runner failed for step10.")]


def test_resume_controller_does_not_reuse_ok_superseded_by_failure(
    tmp_path: Path,
) -> None:
    applied = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={
            "per_step_records": [
                {"step_id": "01_define", "status": "ok"},
                {
                    "step_id": "01_define",
                    "status": "contract_failed",
                    "step_summary": {"status": "rejected"},
                },
            ]
        },
    ).apply()

    assert applied.resumed_step_ids == set()
    assert applied.per_step_records == []


def test_resume_recomputes_plan_revision_findings_instead_of_carrying_them(
    tmp_path: Path,
) -> None:
    state = {
        "findings": [
            ValidationFinding(
                validator="plan_contract_pending",
                severity="warning",
                message="Old trajectory role was pending before probe replan.",
                detail={"kind": "trajectory_role_missing", "role": "representation"},
            ).model_dump(mode="json"),
            ValidationFinding(
                validator="plan_contract",
                severity="error",
                message="Old trajectory plan was missing a role.",
                detail={
                    "kind": "trajectory_role_missing",
                    "role": "candidate_selection",
                },
            ).model_dump(mode="json"),
            ValidationFinding(
                validator="plan_typed_dag",
                severity="error",
                message="An older plan revision had an ambiguous typed producer.",
                detail={
                    "reason": "typed_input_producer_ambiguous",
                    "typed_product": "table:result",
                },
            ).model_dump(mode="json"),
            ValidationFinding(
                validator="statistical_validator",
                severity="warning",
                message="Independent analytical warning remains relevant.",
            ).model_dump(mode="json"),
        ]
    }

    applied = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
    ).apply()

    assert [finding.validator for finding in applied.findings] == [
        "statistical_validator"
    ]


def test_pipeline_resume_is_a_leaf_module() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "orchestration"
        / "resume.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "pipeline",
        "pipeline_execute",
        "easyicu.research_agent.pipeline",
        "easyicu.research_agent.execution.phase",
    }
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module in forbidden
        for node in tree.body
    )


def test_execution_phase_delegates_resume_policy_to_resume_controller() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "easyicu"
        / "research_agent"
        / "execution/phase.py"
    )
    source = path.read_text(encoding="utf-8")

    assert "ResumeController(" in source
    assert "def _resume_code_for_step" not in source
    assert "def _remaining_steps" not in source
    assert "def _initial_step_index" not in source


def test_resume_controller_reuses_latest_valid_code_evidence(tmp_path: Path):
    bad_path = tmp_path / "evidence" / "bad.py"
    good_path = tmp_path / "evidence" / "good.py"
    bad_path.parent.mkdir()
    bad_path.write_text("{}", encoding="utf-8")
    good_path.write_text(
        "import os\nprint(os.environ['COHORT_PARQUET'])\n",
        encoding="utf-8",
    )
    state = {
        "evidence": [
            {
                "evidence_id": "code_good",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/good.py",
                "sha256": _sha256(good_path),
                "generation_mode": "llm",
            },
            {
                "evidence_id": "code_bad",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/bad.py",
                "sha256": _sha256(bad_path),
                "generation_mode": "llm",
            },
        ]
    }

    code, record = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert "COHORT_PARQUET" in code
    assert record["evidence_id"] == "code_good"


def test_resume_controller_reads_newer_repair_code_from_evidence_index(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    old_path = evidence_dir / "old.py"
    new_path = evidence_dir / "new.py"
    old_path.write_text(
        "import os\nprint('old', os.environ['COHORT_PARQUET'])\n",
        encoding="utf-8",
    )
    new_path.write_text(
        "import os\nprint('new repair', os.environ['COHORT_PARQUET'])\n",
        encoding="utf-8",
    )
    old = {
        "evidence_id": "code_old",
        "kind": "code",
        "produced_by_step": "02_model",
        "relative_path": "evidence/old.py",
        "sha256": _sha256(old_path),
        "generation_mode": "llm",
    }
    new = {
        "evidence_id": "code_new_repair",
        "kind": "code",
        "produced_by_step": "02_model",
        "relative_path": "evidence/new.py",
        "sha256": _sha256(new_path),
        "generation_mode": "repaired",
    }
    (evidence_dir / "evidence_index.json").write_text(
        json.dumps([old, new]),
        encoding="utf-8",
    )
    (evidence_dir / "evidence_aliases.json").write_text("{}", encoding="utf-8")

    code, record = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={"evidence": [old]},
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert "new repair" in code
    assert record["evidence_id"] == "code_new_repair"


def test_modern_evidence_authority_rejects_unregistered_resume_code(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    store.register_text(
        kind="log",
        description="Selected modern authority seed.",
        text="seed\n",
        filename="authority_seed.txt",
        evidence_id="authority_seed",
    )
    assert store.authority_head_path.is_file()

    unregistered_path = tmp_path / "evidence" / "unregistered_resume.py"
    unregistered_path.write_text(
        "import os\nprint(os.environ['COHORT_PARQUET'])\n",
        encoding="utf-8",
    )
    unregistered_record = {
        "evidence_id": "unregistered_resume_code",
        "kind": "code",
        "produced_by_step": "02_model",
        "relative_path": "evidence/unregistered_resume.py",
        "sha256": _sha256(unregistered_path),
        "generation_mode": "llm",
    }
    state = {
        "evidence": [unregistered_record],
        "per_step_records": [{"step_id": "02_model", "status": "contract_failed"}],
    }
    (tmp_path / "manifest_partial.json").write_text(
        json.dumps({"evidence": [unregistered_record]}),
        encoding="utf-8",
    )

    reused = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=state,
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert reused is None


def test_quarantined_concept_draft_is_isolated_and_digest_checked(
    tmp_path: Path,
) -> None:
    finding = {
        "validator": "llm_concept_auditor",
        "severity": "error",
        "message": "The plotted percentage does not reconcile to its denominator.",
        "detail": {"step_id": "02_model"},
    }
    draft = store_quarantined_concept_draft(
        run_dir=tmp_path,
        step_id="02_model",
        code="import os\nprint(os.environ['COHORT_PARQUET'])\n",
        findings=[finding],
    )

    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={
            "evidence": [],
            "step_attempt_history": [
                {
                    "step_id": "02_model",
                    "status": "blocked_by_concept_audit",
                    "quarantined_requires_repair": True,
                    "quarantined_draft_sha256": draft.sha256,
                    "quarantined_draft_relative_path": draft.relative_path,
                }
            ],
        },
        resume_from_step_id="02_model",
    )
    assert controller.prior_code_for_step("02_model") is None
    loaded = controller.quarantined_concept_draft_for_step("02_model")
    assert loaded == draft
    assert loaded.findings[0]["message"] == finding["message"]
    assert not (tmp_path / "evidence").exists()

    code_path = tmp_path / loaded.relative_path
    code_path.write_text("import os\nprint('tampered')\n", encoding="utf-8")
    assert (
        load_quarantined_concept_draft(
            run_dir=tmp_path,
            step_id="02_model",
        )
        is None
    )

    clear_quarantined_concept_draft(run_dir=tmp_path, step_id="02_model")
    assert not code_path.parent.exists()


def test_provider_budget_failure_is_not_resumed_as_code_defect(tmp_path: Path) -> None:
    step_id = "02_model"
    checkpoint = store_quarantined_concept_draft(
        run_dir=tmp_path,
        step_id=step_id,
        code="import os\nprint(os.environ['COHORT_PARQUET'])\n",
        findings=[
            {
                "validator": "provider_call_budget",
                "severity": "error",
                "message": "provider allowance exhausted before concept approval",
                "evidence_ids": [],
                "detail": {"step_id": step_id},
            }
        ],
    )
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={
            "step_attempt_history": [
                {
                    "step_id": step_id,
                    "status": "blocked_by_concept_audit",
                    "quarantined_requires_repair": True,
                    "quarantined_draft_sha256": checkpoint.sha256,
                    "quarantined_draft_relative_path": checkpoint.relative_path,
                }
            ]
        },
        resume_from_step_id=step_id,
    )

    assert controller.quarantined_concept_draft_for_step(step_id) is None


def test_stale_quarantine_file_is_not_reused_after_successful_repair(
    tmp_path: Path,
) -> None:
    draft = store_quarantined_concept_draft(
        run_dir=tmp_path,
        step_id="02_model",
        code="import os\nprint(os.environ['COHORT_PARQUET'])\n",
        findings=[
            {
                "validator": "llm_concept_auditor",
                "severity": "error",
                "message": "Repair this draft.",
                "detail": {"step_id": "02_model"},
            }
        ],
    )
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={
            "step_attempt_history": [
                {
                    "step_id": "02_model",
                    "status": "execution_failed",
                    "quarantined_requires_repair": False,
                    "quarantined_draft_sha256": draft.sha256,
                    "quarantined_draft_relative_path": draft.relative_path,
                }
            ]
        },
        resume_from_step_id="02_model",
    )

    assert controller.quarantined_concept_draft_for_step("02_model") is None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
@pytest.mark.parametrize(
    "symlink_component",
    ["steps", "step", "quarantine", "code", "metadata"],
)
def test_quarantine_store_rejects_symlinked_path_components(
    tmp_path: Path, symlink_component: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    steps = tmp_path / "run" / "steps"
    step_dir = steps / "02_model"
    quarantine_dir = step_dir / ".quarantine"
    if symlink_component == "steps":
        steps.parent.mkdir(parents=True)
        steps.symlink_to(outside, target_is_directory=True)
    elif symlink_component == "step":
        steps.mkdir(parents=True)
        step_dir.symlink_to(outside, target_is_directory=True)
    elif symlink_component == "quarantine":
        step_dir.mkdir(parents=True)
        quarantine_dir.symlink_to(outside, target_is_directory=True)
    else:
        quarantine_dir.mkdir(parents=True)
        target = outside / f"{symlink_component}.txt"
        target.write_text("keep target", encoding="utf-8")
        name = (
            "concept_draft.py" if symlink_component == "code" else "concept_draft.json"
        )
        (quarantine_dir / name).symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        store_quarantined_concept_draft(
            run_dir=tmp_path / "run",
            step_id="02_model",
            code="import os\nprint(os.environ['COHORT_PARQUET'])\n",
            findings=[{"severity": "error", "message": "blocked"}],
        )

    assert (
        load_quarantined_concept_draft(run_dir=tmp_path / "run", step_id="02_model")
        is None
    )
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
@pytest.mark.parametrize(
    "symlink_component", ["steps", "step", "quarantine", "code", "metadata"]
)
def test_quarantine_clear_rejects_symlinked_path_without_deleting_outside(
    tmp_path: Path, symlink_component: str
) -> None:
    run_dir = tmp_path / "run"
    steps = run_dir / "steps"
    step_dir = steps / "02_model"
    quarantine_dir = step_dir / ".quarantine"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    if symlink_component == "steps":
        steps.parent.mkdir(parents=True)
        steps.symlink_to(outside, target_is_directory=True)
    elif symlink_component == "step":
        steps.mkdir(parents=True)
        step_dir.symlink_to(outside, target_is_directory=True)
    elif symlink_component == "quarantine":
        steps.mkdir(parents=True)
        step_dir.mkdir()
        quarantine_dir.symlink_to(outside, target_is_directory=True)
    else:
        quarantine_dir.mkdir(parents=True)
        target = outside / f"{symlink_component}.txt"
        target.write_text("keep target", encoding="utf-8")
        name = (
            "concept_draft.py" if symlink_component == "code" else "concept_draft.json"
        )
        (quarantine_dir / name).symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        clear_quarantined_concept_draft(run_dir=run_dir, step_id="02_model")

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert outside.is_dir()


def test_quarantine_clear_does_not_silently_ignore_removal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft = store_quarantined_concept_draft(
        run_dir=tmp_path,
        step_id="02_model",
        code="import os\nprint(os.environ['COHORT_PARQUET'])\n",
        findings=[
            {
                "validator": "llm_concept_auditor",
                "severity": "error",
                "message": "blocked",
            }
        ],
    )

    def fail_remove(_path):
        raise OSError("simulated permission failure")

    monkeypatch.setattr(pipeline_resume.shutil, "rmtree", fail_remove)
    with pytest.raises(ValueError, match="could not be removed safely"):
        clear_quarantined_concept_draft(run_dir=tmp_path, step_id="02_model")

    assert (tmp_path / draft.relative_path).is_file()


@pytest.mark.parametrize("step_id", ["", "..", "../escape", "a/b", "a\\b"])
def test_quarantined_concept_draft_rejects_unsafe_step_id(
    tmp_path: Path, step_id: str
) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        store_quarantined_concept_draft(
            run_dir=tmp_path,
            step_id=step_id,
            code="import os\n",
            findings=[{"severity": "error", "message": "blocked"}],
        )


def test_resume_controller_rejects_code_evidence_outside_run_dir(tmp_path: Path):
    outside = tmp_path / "outside.py"
    outside.write_text("import os\nprint(os.environ['COHORT_PARQUET'])\n")
    state = {
        "evidence": [
            {
                "evidence_id": "code_escape",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "../outside.py",
                "sha256": _sha256(outside),
                "generation_mode": "llm",
            },
            {
                "evidence_id": "code_absolute",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": str(outside),
                "sha256": _sha256(outside),
                "generation_mode": "llm",
            },
        ]
    }

    reused = ResumeController(
        plan=_plan(),
        run_dir=tmp_path / "run",
        resume_state=state,
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert reused is None


def test_resume_controller_rejects_tampered_or_non_evidence_code(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    code_path = evidence_dir / "analysis.py"
    code_path.write_text("import os\nprint('original')\n", encoding="utf-8")
    original_sha256 = _sha256(code_path)
    code_path.write_text("import os\nprint('tampered but valid')\n", encoding="utf-8")
    outside_evidence = tmp_path / "ordinary.py"
    outside_evidence.write_text("import os\nprint('ordinary')\n", encoding="utf-8")
    state = {
        "evidence": [
            {
                "evidence_id": "code_outside_evidence",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "ordinary.py",
                "sha256": _sha256(outside_evidence),
                "generation_mode": "llm",
            },
            {
                "evidence_id": "code_tampered",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/analysis.py",
                "sha256": original_sha256,
                "generation_mode": "llm",
            },
        ]
    }

    assert (
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=state,
            resume_from_step_id="02_model",
        ).prior_code_for_step("02_model")
        is None
    )


def test_resume_controller_rejects_symlinked_evidence_code(tmp_path: Path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("import os\nprint('outside')\n", encoding="utf-8")
    link = evidence_dir / "code_agent__analysis.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    state = {
        "evidence": [
            {
                "evidence_id": "code_agent",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": str(link.relative_to(tmp_path)),
                "sha256": _sha256(outside),
                "generation_mode": "llm",
            }
        ]
    }

    assert (
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=state,
            resume_from_step_id="02_model",
        ).prior_code_for_step("02_model")
        is None
    )


@pytest.mark.parametrize(
    "generation_mode",
    ["fallback", "system", "deterministic_fallback", "deterministic_probe", ""],
)
def test_resume_controller_rejects_non_agent_code_generation_modes(
    tmp_path: Path, generation_mode: str
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    code_path = evidence_dir / "analysis.py"
    code_path.write_text("import os\nprint('not agent code')\n", encoding="utf-8")
    state = {
        "evidence": [
            {
                "evidence_id": "code_non_agent",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/analysis.py",
                "sha256": _sha256(code_path),
                "generation_mode": generation_mode,
            }
        ]
    }

    assert (
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=state,
            resume_from_step_id="02_model",
        ).prior_code_for_step("02_model")
        is None
    )


@pytest.mark.parametrize(
    "resumed_from_generation_mode",
    [None, "", "fallback", "system", "deterministic_fallback", "resumed_code_reuse"],
)
def test_resume_controller_rejects_reused_code_without_root_agent_origin(
    tmp_path: Path, resumed_from_generation_mode: str | None
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    code_path = evidence_dir / "analysis.py"
    code_path.write_text("import os\nprint('reused code')\n", encoding="utf-8")
    metadata = {}
    if resumed_from_generation_mode is not None:
        metadata["resumed_from_generation_mode"] = resumed_from_generation_mode
    state = {
        "evidence": [
            {
                "evidence_id": "code_reused_untrusted_origin",
                "kind": "code",
                "produced_by_step": "02_model",
                "relative_path": "evidence/analysis.py",
                "sha256": _sha256(code_path),
                "generation_mode": "resumed_code_reuse",
                "metadata": metadata,
            }
        ]
    }

    assert (
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=state,
            resume_from_step_id="02_model",
        ).prior_code_for_step("02_model")
        is None
    )


def test_resume_controller_accepts_reused_code_with_root_agent_origin(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    code_path = evidence_dir / "analysis.py"
    code_path.write_text("import os\nprint('reused repair')\n", encoding="utf-8")
    record = {
        "evidence_id": "code_reused_repaired_origin",
        "kind": "code",
        "produced_by_step": "02_model",
        "relative_path": "evidence/analysis.py",
        "sha256": _sha256(code_path),
        "generation_mode": "resumed_code_reuse",
        "metadata": {"resumed_from_generation_mode": "repaired"},
    }

    reused = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={"evidence": [record]},
        resume_from_step_id="02_model",
    ).prior_code_for_step("02_model")

    assert reused is not None
    assert reused[1]["evidence_id"] == record["evidence_id"]


def test_resume_controller_remaining_steps_respects_stop_point(tmp_path: Path):
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state=None,
        stop_after_step_id="02_model",
    )

    remaining = controller.remaining_steps(
        plan=_plan(),
        executed_step_ids={"01_define"},
    )

    assert [step.step_id for step in remaining] == ["02_model"]


def test_resume_controller_remaining_steps_respects_explicit_start_point(
    tmp_path: Path,
):
    controller = ResumeController(
        plan=_plan(),
        run_dir=tmp_path,
        resume_state={"per_step_records": []},
        resume_from_step_id="02_model",
    )

    remaining = controller.remaining_steps(
        plan=_plan(),
        executed_step_ids=set(),
    )

    assert [step.step_id for step in remaining] == ["02_model", "03_figure"]


def test_resume_controller_rejects_unknown_resume_or_stop_step(tmp_path: Path):
    with pytest.raises(ValueError, match="resume_from_step_id"):
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=None,
            resume_from_step_id="missing",
        )

    with pytest.raises(ValueError, match="stop_after_step_id"):
        ResumeController(
            plan=_plan(),
            run_dir=tmp_path,
            resume_state=None,
            stop_after_step_id="missing",
        )


def test_upsert_step_record_replaces_pending_and_preserves_completed_records():
    records = [
        {"step_id": "02_model", "status": "executed_pending_review"},
        {"step_id": "03_figure", "status": "ok"},
    ]

    upsert_step_record(
        records,
        {"step_id": "02_model", "status": "ok"},
        replace_statuses={"executed_pending_review"},
    )
    upsert_step_record(
        records,
        {"step_id": "03_figure", "status": "contract_failed"},
        replace_statuses={"executed_pending_review"},
    )
    upsert_step_record(records, {"step_id": "04_extra", "status": "ok"})

    assert records == [
        {"step_id": "02_model", "status": "ok"},
        {"step_id": "03_figure", "status": "ok"},
        {"step_id": "03_figure", "status": "contract_failed"},
        {"step_id": "04_extra", "status": "ok"},
    ]
