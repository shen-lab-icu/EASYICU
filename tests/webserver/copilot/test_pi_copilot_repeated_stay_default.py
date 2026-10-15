"""The plan card names a repeated-stay route the system chose on its own.

When a reviewed plan's configuration keeps every ICU stay and clusters by
patient without the researcher having chosen it, the host records that the
handling is the system's default (``plan_decisions.REPEATED_STAYS_SYSTEM_DEFAULT``,
DESIGN_v2 F.7), and the cards that move the plan towards its run say so, with
the way to choose otherwise.  The researcher's own choice records that it is
theirs: the plan decision (in ``test_pi_copilot_plan_decisions.py``), a cohort
option that keeps each patient's first stay, or a conversation turn that names
clustering or the first stay in their own words.  A model cannot write the
key, and a study whose design no longer follows the route names none.

Synthetic studies, sessions and cards only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.webserver import research_launch_scientific, study_contexts
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot import tools as tool_module
from easyicu.webserver.pi_copilot import workflow as workflow_module
from easyicu.webserver.pi_copilot.contracts import (
    AuthorityBinding,
    PiSessionRecord,
    ToolExecutionContext,
)
from easyicu.webserver.pi_copilot.plan_decisions import REPEATED_STAYS_SYSTEM_DEFAULT
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.routes import agent as agent_routes
from tests.support.node import run_node

KEY = REPEATED_STAYS_SYSTEM_DEFAULT
STATIC = Path(agent_routes.__file__).resolve().parents[1] / "static"
_CLUSTERED = {
    "analysis_family": "association_study",
    "analysis_unit": "icu_stay",
    "variance_estimator": "cluster_robust",
    "cluster_unit": "patient",
}


# --- the conversation ----------------------------------------------------------------


def _current(**fields: Any) -> dict[str, Any]:
    return {
        "id": "study-repeated-stay-default",
        "revision": 4,
        "question": "Is the first-day maximum lactate associated with hospital death?",
        "active_job_id": None,
        "cohort": {"preset": "all_icu", "exclude_readmissions": False},
        "analysis_design": dict(_CLUSTERED),
        "confirmations": {"plan_repeated_stays_clustered": True, KEY: True},
        **fields,
    }


def _update(
    monkeypatch: pytest.MonkeyPatch,
    current: dict[str, Any],
    proposal: dict[str, Any],
    message: str,
    *,
    patient_grouping_available: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the conversation's study update with the researcher's message."""

    writes: list[dict[str, Any]] = []
    if patient_grouping_available:
        monkeypatch.setattr(
            research_launch_scientific,
            "_patient_grouping_for_analysis_design",
            lambda _study: SimpleNamespace(output_identity_column="patient_id"),
        )
    monkeypatch.setattr(tool_module, "_bound_context", lambda _binding: dict(current))
    monkeypatch.setattr(
        tool_module.study_contexts,
        "upsert_context",
        lambda raw, **_kwargs: writes.append(dict(raw))
        or {**current, **raw, "revision": current["revision"] + 1},
    )
    session = PiSessionRecord(
        session_id="pi-repeated-stay-default",
        binding=AuthorityBinding(
            study_context_id=current["id"], study_revision=current["revision"]
        ),
    )
    result = tool_module.execute_tool(
        "easyicu_update_study_context",
        proposal,
        ToolExecutionContext(
            session=session, user_message=message, allowed_actions={"configure"}
        ),
    )
    return result, writes


def _written_key(writes: list[dict[str, Any]]) -> Any:
    assert writes, "the update saved nothing"
    return (writes[-1].get("confirmations") or {}).get(KEY, "unwritten")


@pytest.mark.parametrize(
    "message",
    ["保留每次 ICU 入住，按患者聚类处理同一患者的多次入住。",
     "Keep every stay, clustered by patient.",
     "不要再问了，按患者聚类处理。",
     "Don't ask again, keep every stay clustered by patient."],
)
def test_a_turn_that_names_clustering_makes_the_handling_the_researchers(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    result, writes = _update(
        monkeypatch, _current(), {"analysis_design": dict(_CLUSTERED)}, message,
        patient_grouping_available=True,
    )

    assert result["code"] == "study_context_updated"
    assert _written_key(writes) is False


@pytest.mark.parametrize("message", [
    "按患者聚类是什么意思？", "不要按患者聚类处理。", "如果按患者聚类会怎样",
    "请解释按患者聚类处理", "How does patient-clustered uncertainty work?",
    "Do not use patient-clustered inference.", "Explain patient-clustered inference",
])
def test_a_mention_does_not_make_the_system_default_a_researcher_choice(
    monkeypatch: pytest.MonkeyPatch, message: str,
) -> None:
    result, writes = _update(
        monkeypatch, _current(), {"analysis_design": dict(_CLUSTERED)}, message,
        patient_grouping_available=True,
    )
    assert result["code"] == "study_context_updated"
    assert _written_key(writes) is not False


def test_a_clustering_request_without_grouping_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, writes = _update(
        monkeypatch, _current(), {"analysis_design": dict(_CLUSTERED)},
        "Keep every stay, clustered by patient.",
    )
    assert result["code"] == "research_pipeline_cluster_variance_unsupported"
    assert writes == []


def test_a_turn_that_names_the_first_stay_makes_the_handling_the_researchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, writes = _update(
        monkeypatch,
        _current(),
        {"cohort": {"preset": "all_icu", "exclude_readmissions": True}},
        "只纳入每位患者的首次 ICU 入住。",
    )

    assert result["code"] == "study_context_updated"
    assert _written_key(writes) is False


def test_a_turn_that_names_neither_leaves_who_chose_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, writes = _update(
        monkeypatch,
        _current(),
        {"analysis_design": dict(_CLUSTERED), "title": "Lactate and death"},
        "把标题改成 Lactate and death。",
    )

    assert result["code"] == "study_context_updated"
    assert _written_key(writes) is not False


def test_a_model_cannot_write_who_chose_the_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _current(confirmations={"plan_repeated_stays_clustered": True})

    result, writes = _update(
        monkeypatch,
        current,
        {"title": "Lactate and death", "confirmations": {KEY: True}},
        "把标题改成 Lactate and death。",
    )

    assert result["code"] == "study_context_updated"
    assert _written_key(writes) == "unwritten"


# --- a cohort option -----------------------------------------------------------------


def _confirm_option(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, option_id: str
) -> dict[str, Any]:
    from easyicu.webserver.pi_copilot.service import PiCopilotService

    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "study-contexts.json")
    initial = study_contexts.upsert_context(
        {
            "id": "study-repeated-stay-option",
            "cohort": {"preset": "all_icu"},
            "analysis_design": dict(_CLUSTERED),
            "confirmations": {"plan_repeated_stays_clustered": True, KEY: True},
        }
    )
    service = PiCopilotService(store_path=tmp_path / "sessions.json")
    service.project_store.bind("project-repeated-stay-option", initial["id"])
    record = PiSessionRecord(
        session_id="pi-repeated-stay-option",
        project_id="project-repeated-stay-option",
        binding=service._binding_for_context(initial),  # noqa: SLF001
    )
    service._save_record(record)  # noqa: SLF001
    selection = service._cohort_eligibility_selection_projection(record)  # noqa: SLF001
    option = next(row for row in selection["options"] if row["id"] == option_id)
    result = service.confirm_cohort_eligibility(
        record.session_id,
        project_id="project-repeated-stay-option",
        option_id=option["id"],
        expected_revision=option["expected_revision"],
        primary_cohort_contract_sha256=option["primary_cohort_contract_sha256"],
        selection_event_id=option["selection_event_id"],
    )
    assert result["code"] == "cohort_eligibility_confirmed"
    updated = study_contexts.get_context(initial["id"])
    assert updated is not None
    return updated


def test_a_cohort_option_that_keeps_the_first_stay_is_the_researchers_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    updated = _confirm_option(monkeypatch, tmp_path, "first_admission_only")

    assert updated["cohort"]["exclude_readmissions"] is True
    assert updated["confirmations"][KEY] is False
    assert updated["confirmations"]["plan_repeated_stays_clustered"] is True
    assert cohort_eligibility.eligibility_stated(updated) is True


def test_a_cohort_option_that_keeps_every_stay_leaves_the_clustering_the_systems(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    updated = _confirm_option(monkeypatch, tmp_path, "adults_all_admissions")

    assert updated["cohort"]["exclude_readmissions"] is False
    assert updated["confirmations"][KEY] is True


# --- the workflow states it for the plan under review ---------------------------------


def _summary(study: dict[str, Any]) -> dict[str, Any]:
    review = {
        "status": "approvable",
        "findings": [],
    }
    snapshot = build_research_workflow_snapshot(
        study=study,
        active_export_present=True,
        active_job=None,
        latest_run={
            "run_id": "run-repeated-stay",
            "run_type": "full",
            "engine": "easyicu.research_agent.pipeline",
            "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["operator_plan_approval_required"],
            "scientific_configuration_sha256": "a" * 64,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json"],
        },
        plan_review_authority={
            "run_id": "run-repeated-stay",
            "resumable_here": True,
            "scientific_configuration_sha256": "a" * 64,
            "scientific_plan_review": review,
        },
    )
    enriched = workflow_module._enrich_plan_review(  # noqa: SLF001
        snapshot, study=study, review={"artifact_payloads": {"agent_plan.json": {}}}
    )
    return dict(enriched.plan_review_summary or {})


def test_the_plan_under_review_states_the_systems_route() -> None:
    study = _current()

    assert _summary(study)["repeated_stays_system_default"] == "patient_clustered"
    chosen = _current(confirmations={"plan_repeated_stays_clustered": True, KEY: False})
    assert "repeated_stays_system_default" not in _summary(chosen)
    moved_on = _current(cohort={"preset": "all_icu", "exclude_readmissions": True})
    assert "repeated_stays_system_default" not in _summary(moved_on)


# --- the cards ---------------------------------------------------------------------


def _card(code: str, summary: dict[str, Any]) -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = r"""
let declared = null;
global.window = {
  EU_LANG: 'zh',
  EasyICU: { guidedPi: { declare: (_name, api) => { declared = api; }, optional: () => null } },
};
require(process.argv[1]);
const code = process.argv[2];
const summary = JSON.parse(process.argv[3]);
const owner = declared.create({
  tr: (en, zh) => zh || en,
  esc: value => String(value),
  iconHtml: () => '',
  resourceButton: resource => `<a data-resource="${resource.artifact}"></a>`,
  sessionIsStale: () => false,
  busy: () => false,
  session: () => ({ binding: { run_id: 'run-repeated-stay' } }),
  workflow: () => ({ next_action_code: code, plan_review_summary: { run_id: 'run-repeated-stay', ...summary } }),
});
process.stdout.write(owner.workflowConfirmationHtml());
"""
    owner = STATIC / "js" / "screens-guided-pi-confirmation.js"
    result = run_node(
        node, script, str(owner.resolve()), code, json.dumps(summary), check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


DEFAULT_LINE = "重复 ICU 入住 · 系统默认"


@pytest.mark.parametrize(
    "code", ["operator_plan_approval_required", "plan_execution_upgrade_required"]
)
def test_the_cards_that_move_the_plan_name_the_systems_route(code: str) -> None:
    html = _card(code, {"repeated_stays_system_default": "patient_clustered"})

    assert DEFAULT_LINE in html
    assert "按患者聚类计算不确定性" in html and "请在对话里说明" in html
    assert DEFAULT_LINE not in _card(code, {})
    assert DEFAULT_LINE not in _card(code, {"repeated_stays_system_default": "first_stay"})


def test_a_card_that_does_not_move_the_plan_names_no_route() -> None:
    html = _card(
        "failed_pipeline_requires_fresh_plan",
        {"repeated_stays_system_default": "patient_clustered"},
    )

    assert DEFAULT_LINE not in html
