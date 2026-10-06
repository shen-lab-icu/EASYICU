"""Copilot offers no plan that the launch refuses for the bound export.

A new plan or run analyzes the bound export's rows as the study's own, and the
launch refuses an export extracted for another cohort or under an earlier
rule.  The workflow did not read that answer: it called such a study ready to
plan, the host started the plan, and the researcher saw only the refusal's
code.  The workflow now reads the launch's own answer.  A fresh plan is not
offered and the plan step says what comes first; a resumed run keeps the
package its sealed plan bound and is still offered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from easyicu.webserver import dataio
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.pi_copilot import workflow as workflow_owner
from easyicu.webserver.pi_copilot.extraction_handoff import compile_study_cohort
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.research_launch_scientific import (
    ExportCohortRefusal,
    _require_export_holds_study_cohort,
    bound_export_cohort_refusal,
)
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from tests.webserver.copilot.research_workflow_fixtures import complete_study

STATIC = Path(workflow_owner.__file__).resolve().parents[1] / "static" / "js"

_OUTDATED = ExportCohortRefusal(
    code="research_pipeline_export_cohort_mismatch",
    reason_codes=("registered_export_cohort_execution_outdated",),
)
_INVALID = ExportCohortRefusal(
    code="research_pipeline_export_cohort_invalid",
    reason_codes=("cohort_contract_invalid",),
)


def _planning_study(export_path: Path | str, **cohort: Any) -> dict[str, Any]:
    """A question and an identified data source, which is ready to plan."""

    study: dict[str, Any] = {
        "id": "study-export-cohort",
        "revision": 2,
        "question": "Is early lactate clearance associated with hospital mortality?",
        "data_source": {"path": str(export_path), "database": "miiv"},
    }
    if cohort:
        study["cohort"] = dict(cohort)
    return study


def _write_export(
    export_path: Path, contract: dict[str, Any] | None, *, current_rule: bool
) -> None:
    export_path.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "easyicu_native_export_v2",
        "database": "miiv",
        "data_path": str(export_path.parent / "raw"),
        "format": "parquet",
        "files": [{"file": "demographics.parquet", "module": "demographics"}],
    }
    if contract is not None:
        manifest["cohort_contract"] = contract
        if current_rule:
            manifest["cohort_execution"] = dataio.export_cohort_execution(contract)
    (export_path / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _snapshot(study: dict[str, Any], refusal: ExportCohortRefusal | None, **kwargs: Any):
    return build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run=kwargs.pop("latest_run", None),
        export_cohort_refusal=refusal,
        **kwargs,
    )


def _plan_stage(snapshot: Any) -> Any:
    return next(stage for stage in snapshot.stages if stage.id == "plan")


def test_the_workflow_reads_the_refusal_the_launch_raises(tmp_path: Path) -> None:
    export_path = tmp_path / "export"
    all_icu = _planning_study(export_path)
    _write_export(export_path, compile_study_cohort(all_icu), current_rule=True)

    # The export holds the study's cohort under the current rule.
    assert bound_export_cohort_refusal(all_icu, str(export_path)) is None
    _require_export_holds_study_cohort(all_icu, str(export_path))

    # A study stating another cohort meets the same refusal in both places.
    sepsis = _planning_study(export_path, preset="sepsis3")
    refusal = bound_export_cohort_refusal(sepsis, str(export_path))
    assert refusal == ExportCohortRefusal(
        code="research_pipeline_export_cohort_mismatch",
        reason_codes=("registered_export_cohort_mismatch",),
    )
    with pytest.raises(ResearchPipelineRunError) as caught:
        _require_export_holds_study_cohort(sepsis, str(export_path))
    assert caught.value.code == refusal.code
    assert caught.value.details == {"mismatch_codes": list(refusal.reason_codes)}

    # An export of the same cohort extracted before the current rule.
    outdated_path = tmp_path / "outdated"
    _write_export(outdated_path, compile_study_cohort(all_icu), current_rule=False)
    assert bound_export_cohort_refusal(all_icu, str(outdated_path)) == _OUTDATED

    # A cohort that is not an executable contract is refused by its reason,
    # and the launch keeps the contract's own error as the cause.
    unexecutable = _planning_study(export_path, preset="icd")
    refusal = bound_export_cohort_refusal(unexecutable, str(export_path))
    assert refusal is not None
    assert refusal.code == "research_pipeline_export_cohort_invalid"
    with pytest.raises(ResearchPipelineRunError) as caught:
        _require_export_holds_study_cohort(unexecutable, str(export_path))
    assert caught.value.code == refusal.code
    assert caught.value.details == {"reason_code": refusal.reason_codes[0]}
    assert isinstance(caught.value.__cause__, dataio.ExportCohortError)

    # Nothing to compare: a raw folder, a missing path, no bound path.
    raw_folder = tmp_path / "raw_database"
    raw_folder.mkdir()
    for path in (str(raw_folder), str(tmp_path / "missing"), ""):
        assert bound_export_cohort_refusal(sepsis, path) is None


def test_a_study_is_not_offered_a_plan_its_launch_refuses() -> None:
    study = _planning_study("/prepared/export")

    ready = _snapshot(study, None)
    assert ready.next_action_code == "provider_ready_to_generate_plan"
    assert _plan_stage(ready).status == "ready"
    assert ready.bound_export_cohort_refusal is None

    refused = _snapshot(study, _OUTDATED)
    assert refused.current_stage == "plan"
    assert refused.next_action_code == "bound_export_cohort_mismatch"
    assert _plan_stage(refused).status == "blocked"
    assert _plan_stage(refused).reason_code == "bound_export_cohort_mismatch"
    assert refused.bound_export_cohort_refusal == {
        "code": "research_pipeline_export_cohort_mismatch",
        "reason_codes": ["registered_export_cohort_execution_outdated"],
    }

    invalid = _snapshot(study, _INVALID)
    assert invalid.next_action_code == "bound_export_cohort_invalid"
    assert _plan_stage(invalid).status == "blocked"
    assert invalid.bound_export_cohort_refusal == {
        "code": "research_pipeline_export_cohort_invalid",
        "reason_codes": ["cohort_contract_invalid"],
    }


def _failed_run(study: dict[str, Any], **row: Any) -> dict[str, Any]:
    return {
        "run_id": "run-failed",
        "study_id": study["id"],
        "scientific_configuration_sha256": (
            study_context_owner.scientific_configuration_sha256(study)
        ),
        "run_type": "full",
        "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked",
        "artifact_names": [
            "agent_plan.json",
            "evidence_ledger.json",
            "source_run_manifest.json",
        ],
        **row,
    }


@pytest.mark.parametrize(
    ("row", "offered"),
    [
        ({"run_status": "completed"}, "failed_pipeline_requires_fresh_plan"),
        (
            {
                "gate_reason": "research_agent_pipeline_failed_closed",
                "run_status": "blocked",
            },
            "failed_pipeline_execution_retry_available",
        ),
        (
            {
                "gate_reason": "research_pipeline_planner_efficiency_budget_exhausted",
                "run_status": "failed",
                "development_planner_checkpoint_available": True,
                "artifact_names": ["evidence_ledger.json", "source_run_manifest.json"],
            },
            "planner_checkpoint_resume_available",
        ),
    ],
)
def test_a_fresh_plan_after_a_failed_run_is_refused_and_a_resume_is_not(
    row: dict[str, Any], offered: str
) -> None:
    study = complete_study()
    latest_run = _failed_run(study, **row)

    assert _snapshot(study, None, latest_run=latest_run).next_action_code == offered
    refused = _snapshot(study, _OUTDATED, latest_run=latest_run)

    if offered == "failed_pipeline_requires_fresh_plan":
        assert refused.next_action_code == "bound_export_cohort_mismatch"
        assert _plan_stage(refused).status == "blocked"
    else:
        # A resume continues on the package its sealed plan bound.
        assert refused.next_action_code == offered
        assert _plan_stage(refused).status == "ready"
    assert refused.bound_export_cohort_refusal is not None


def test_the_project_projection_reads_the_studys_bound_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workflow_owner.sources, "load_registry", lambda: {})
    monkeypatch.setattr(workflow_owner, "list_bound_run_history", lambda **_kwargs: [])

    def project(export_path: Path) -> Any:
        study = _planning_study(export_path)
        return workflow_owner.build_project_workflow_projection(
            study_context_id=study["id"], study_override=study
        ).workflow

    current = tmp_path / "current"
    _write_export(
        current, compile_study_cohort(_planning_study(current)), current_rule=True
    )
    assert project(current).next_action_code == "provider_ready_to_generate_plan"

    outdated = tmp_path / "outdated"
    _write_export(
        outdated, compile_study_cohort(_planning_study(outdated)), current_rule=False
    )
    workflow = project(outdated)
    assert workflow.next_action_code == "bound_export_cohort_mismatch"
    assert workflow.bound_export_cohort_refusal == {
        "code": "research_pipeline_export_cohort_mismatch",
        "reason_codes": ["registered_export_cohort_execution_outdated"],
    }


def test_the_blocked_plan_step_and_a_refused_launch_read_in_chinese() -> None:
    aside = (STATIC / "screens-guided-pi-aside.js").read_text(encoding="utf-8")
    for code, phrase in (
        ("bound_export_cohort_mismatch", "请先为本研究的人群提取数据，或改用其他数据源"),
        ("bound_export_cohort_invalid", "请检查人群设置，或改用其他数据源"),
    ):
        line = next(row for row in aside.splitlines() if f"{code}: tr(" in row)
        assert phrase in line
    errors = (STATIC / "screens-guided-pi-error-text.js").read_text(encoding="utf-8")
    head = errors[: errors.index("function runFailureDetailText")]
    for code, phrase in (
        ("research_pipeline_export_cohort_mismatch", "请先为本研究的人群提取数据"),
        ("research_pipeline_export_cohort_invalid", "请检查人群设置"),
    ):
        branch = head.split(f"error.code === '{code}'", 1)[1].split("if (error.code", 1)[0]
        assert phrase in branch
