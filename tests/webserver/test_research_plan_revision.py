"""Prepared plan repair composes existing authorities; it grants no approval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.run_input import RunInputIdentityError
from easyicu.research_agent.planning.scientific_review import (
    PlanScientificFinding,
    PlanScientificReview,
)
from easyicu.webserver import research_plan_revision as owner


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    study = {
        "id": "study-1",
        "question": "Describe sepsis",
        "data_source": {"path": "/prepared", "database": "miiv"},
    }
    digest = owner.study_contexts.scientific_configuration_sha256(study)
    wrapper = tmp_path / "study-1" / "run_source"
    run_dir = wrapper / "pipeline" / "run_inner"
    run_dir.mkdir(parents=True)
    capsule = run_dir / "run_input_capsule.json"
    capsule.write_text(json.dumps({"scientific_identity": {"target_outcome": "death"}}))
    review = PlanScientificReview(
        status="changes_required",
        approval_allowed=False,
        top_journal_candidate=False,
        score=74,
        dimension_scores={"content_completeness": 45},
        findings=[
            PlanScientificFinding(
                code="ACCEPTED_BASELINE_CONTENT_MISSING",
                severity="blocker",
                dimension="content_completeness",
                message="Accepted baseline is missing.",
                remediation="Restore the bound content.",
                remediation_route="agent_plan_revision",
            )
        ],
        context_sha256="a" * 64,
        plan_sha256="b" * 64,
        literature_sha256="c" * 64,
        figure_strategy_sha256="d" * 64,
        generated_at="2026-09-07T00:00:00Z",
    ).model_dump(mode="json")
    row = {
        "run_id": "run_inner",
        "project_dir": str(wrapper),
        "scientific_configuration_sha256": digest,
        "research_input_state": "prepared",
    }
    manifest = {"research_input_state": "prepared"}
    record = SimpleNamespace(
        artifact_payloads={
            "source_run_manifest.json": manifest,
            "scientific_plan_review.json": review,
        }
    )
    seed = SimpleNamespace(
        schema_version="easyicu.web-review-recovery-seed/4",
        budget_mode="full_reviewed",
        prepared_package_binding={"sha256": "e" * 64},
        study=study,
        scientific_configuration_sha256=digest,
        pipeline_config={},
        pipeline_config_sha256="f" * 64,
    )
    config = SimpleNamespace(
        require_human_plan_review=True,
        workdir=run_dir.parent,
        bound_plan_revision_contract="Preserve age, sex, admission type and Charlson.",
        required_primary_cohort_selection_mode="all_input_rows",
    )
    checkpoint = SimpleNamespace(
        run_id="run_inner",
        pipeline_config_sha256="f" * 64,
        approved_decisions=[],
        execution_start_receipt=None,
        run_input_capsule_sha256=hashlib.sha256(capsule.read_bytes()).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        owner.agent_runs, "list_run_history", lambda **kw: {"runs": [row]}
    )
    monkeypatch.setattr(owner.agent_runs, "read_run_record", lambda path: record)
    monkeypatch.setattr(owner, "load_recovery_seed", lambda path: seed)
    monkeypatch.setattr(
        owner.PipelineConfig, "from_recovery_payload", lambda *a, **kw: config
    )
    monkeypatch.setattr(owner, "load_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(
        owner,
        "load_verified_run_input_capsule",
        lambda **kw: calls.append(("input", kw)),
    )
    monkeypatch.setattr(owner, "research_input_state", lambda path: "prepared")
    monkeypatch.setattr(
        owner.dataio,
        "validate_research_pipeline_source",
        lambda *a, **kw: calls.append(("package", kw)),
    )
    return SimpleNamespace(
        study=study,
        root=tmp_path,
        wrapper=wrapper,
        run_dir=run_dir,
        capsule=capsule,
        row=row,
        record=record,
        manifest=manifest,
        seed=seed,
        config=config,
        checkpoint=checkpoint,
        review=review,
        calls=calls,
    )


def load(source: SimpleNamespace):
    return owner.load_prepared_plan_revision(
        study=source.study, project_root=str(source.root), source_run_id="run_inner"
    )


def test_prepared_repair_reuses_bound_scope_without_consuming_review(source) -> None:
    before = source.capsule.read_bytes()
    result = load(source)
    assert result.run_dir == source.run_dir
    assert result.budget_mode == "full_reviewed"
    assert result.required_primary_cohort_selection_mode == "all_input_rows"
    assert "Charlson" in result.prior_plan_contract
    assert source.calls == [
        (
            "input",
            {
                "run_dir": source.run_dir,
                "scientific_identity": {"target_outcome": "death"},
            },
        ),
        (
            "package",
            {
                "database": "miiv",
                "expected_binding": source.seed.prepared_package_binding,
            },
        ),
    ]
    assert source.checkpoint.approved_decisions == []
    assert source.checkpoint.execution_start_receipt is None
    assert source.capsule.read_bytes() == before


@pytest.mark.parametrize("state", ["metadata_only", "unavailable", None])
def test_unprepared_history_cannot_promote_a_candidate(source, state) -> None:
    source.row["research_input_state"] = state
    source.manifest["research_input_state"] = state
    assert load(source) is None
    assert source.calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        "study",
        "legacy_seed",
        "budget",
        "package",
        "seed_study",
        "seed_digest",
        "no_review",
        "workdir",
        "checkpoint_run",
        "checkpoint_config",
        "approval",
        "execution",
        "capsule",
        "receipt",
        "scope",
        "approvable",
        "owner",
    ],
)
def test_prepared_repair_rejects_drift_instead_of_falling_back(
    source, mutation
) -> None:
    if mutation == "study":
        source.row["scientific_configuration_sha256"] = "x" * 64
    elif mutation == "legacy_seed":
        source.seed.schema_version = "easyicu.web-review-recovery-seed/1"
    elif mutation == "budget":
        source.seed.budget_mode = "planner_canary"
    elif mutation == "package":
        source.seed.prepared_package_binding = None
    elif mutation == "seed_study":
        source.seed.study = {**source.study, "id": "another-study"}
    elif mutation == "seed_digest":
        source.seed.scientific_configuration_sha256 = "x" * 64
    elif mutation == "no_review":
        source.config.require_human_plan_review = False
    elif mutation == "workdir":
        source.config.workdir = source.root / "another-run"
    elif mutation == "checkpoint_run":
        source.checkpoint.run_id = "another-run"
    elif mutation == "checkpoint_config":
        source.checkpoint.pipeline_config_sha256 = "x" * 64
    elif mutation == "approval":
        source.checkpoint.approved_decisions = [{"decision": "approved"}]
    elif mutation == "execution":
        source.checkpoint.execution_start_receipt = {"run_id": "run_inner"}
    elif mutation == "capsule":
        source.capsule.write_text("{}")
    elif mutation == "receipt":
        source.manifest.clear()
    elif mutation == "scope":
        source.root = source.root / "another-project"
    elif mutation == "approvable":
        source.review["approval_allowed"] = True
    elif mutation == "owner":
        source.review["findings"][0]["remediation_route"] = "runtime_capability"
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        load(source)
    assert raised.value.code == "prepared_plan_revision_source_invalid"
    assert source.calls == []


@pytest.mark.parametrize("boundary", ["checkpoint", "input", "package", "physical"])
def test_owner_validation_failure_cannot_be_downgraded_to_metadata(
    source, monkeypatch, boundary
) -> None:
    def invalid(*a, **kw):
        raise RunInputIdentityError("private input path must not escape")

    if boundary == "checkpoint":
        monkeypatch.setattr(owner, "load_checkpoint", invalid)
    elif boundary == "input":
        monkeypatch.setattr(owner, "load_verified_run_input_capsule", invalid)
    elif boundary == "package":
        monkeypatch.setattr(owner.dataio, "validate_research_pipeline_source", invalid)
    else:
        monkeypatch.setattr(owner, "research_input_state", lambda path: "metadata_only")
    with pytest.raises(owner.ResearchPipelineRunError) as raised:
        load(source)
    assert "private input path" not in str(raised.value)


def test_symlinked_capsule_is_not_restoration_authority(source) -> None:
    real = source.run_dir / "capsule_real.json"
    source.capsule.rename(real)
    source.capsule.symlink_to(real)
    with pytest.raises(owner.ResearchPipelineRunError):
        load(source)


def test_no_source_never_inherits_a_neighbouring_prepared_run(source) -> None:
    assert (
        owner.load_prepared_plan_revision(
            study=source.study, project_root=str(source.root), source_run_id=""
        )
        is None
    )
    assert source.calls == []
