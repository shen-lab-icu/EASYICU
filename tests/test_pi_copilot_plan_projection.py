"""The plan preview reads compiled semantics; it does not re-derive them.

The Copilot plan reader needs each step's run phase and each design
candidate's analysis family. Both are owned by ``research_agent.planning``.
Stamping them onto the browser projection keeps the renderer from rebuilding
study semantics out of free-text method names -- the habit that produced the
reader's canned-prose defects.

The values must never reach the persisted artefact: ``plan_sha256`` is a digest
over the whole plan dump, so a new field would invalidate the stored digest of
every run already on disk and break digest-verified resume.
"""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver.pi_copilot.plan_projection import (
    project_plan_conversation_preview,
    project_plan_reader_fields,
)


def _plan() -> dict:
    return {
        "research_question": "Q",
        "analysis_type": "association_study",
        "design_selection": {
            "candidates": [
                {"disposition": "selected", "analysis_type": "association_study"},
                {"disposition": "rejected", "analysis_type": "descriptive_epidemiology"},
            ]
        },
        "steps": [
            {
                "step_id": "cohort_accounting",
                "method": "cohort_definition_and_attrition",
                "planned_analysis_role": "auxiliary",
                "expected_outputs": ["table:cohort_flow"],
            },
            {
                "step_id": "primary_association",
                "method": "signed_landmark_restricted_cubic_spline",
                "planned_analysis_role": "primary",
                "expected_outputs": ["table:estimates"],
            },
            {
                "step_id": "spline_sensitivity",
                "method": "restricted_cubic_spline_sensitivity",
                "planned_analysis_role": "sensitivity",
                "expected_outputs": ["table:sensitivity"],
            },
            {
                "step_id": "overview_figure",
                "method": "visualization",
                "planned_analysis_role": "auxiliary",
                "expected_outputs": ["figure:overview"],
            },
        ],
    }


def test_conversation_preview_uses_only_the_selected_reviewable_plan() -> None:
    source = _plan()
    selected = source["design_selection"]["candidates"][0]
    selected["reviewable_plan"] = [
        "Use the sealed ICU cohort.",
        "Use the prespecified exposure window.",
        "Follow the declared hospital outcome.",
        "Fit the adjusted association model.",
        "Report missingness and prespecify handling.",
        "Check feasibility and sensitivity analyses before execution.",
    ]
    selected.update(
        {
            "estimand": "Adjusted association after the exposure landmark.",
            "time_zero": "24 hours after ICU admission.",
            "observation_window": "From 24 hours to hospital discharge.",
            "primary_method": "Restricted cubic spline logistic model.",
            "required_variables": ["stay_id", "lact_max", "death", "age"],
        }
    )

    preview = project_plan_conversation_preview(source)

    assert preview is not None
    assert [item["key"] for item in preview["items"]] == [
        "population_and_unit",
        "exposure_and_timing",
        "outcome_and_followup",
        "adjustment_and_model",
        "missing_data",
        "sensitivity_and_feasibility",
    ]
    assert preview["items"][0]["text"] == "Use the sealed ICU cohort."
    assert preview["step_count"] == 4
    assert preview["analysis_step_count"] == 3
    assert preview["output_step_count"] == 1
    assert preview["table_count"] == 3
    assert preview["figure_count"] == 1
    assert preview["design"] == {
        "estimand": "Adjusted association after the exposure landmark.",
        "time_zero": "24 hours after ICU admission.",
        "observation_window": "From 24 hours to hospital discharge.",
        "primary_method": "Restricted cubic spline logistic model.",
        "required_variables": ["stay_id", "lact_max", "death", "age"],
    }


def test_conversation_preview_refuses_incomplete_or_unselected_designs() -> None:
    source = _plan()
    assert project_plan_conversation_preview(source) is None
    source["design_selection"]["candidates"][0]["reviewable_plan"] = ["short"]
    assert project_plan_conversation_preview(source) is None
    source["design_selection"]["candidates"][0]["disposition"] = "rejected"
    assert project_plan_conversation_preview(source) is None


def test_plan_preview_carries_the_owner_compiled_phase_and_family() -> None:
    projected = project_plan_reader_fields("agent_plan.json", _plan())

    assert [step["planned_phase"] for step in projected["steps"]] == [
        "cohort",
        "analysis",
        "robustness",
        "reporting",
    ]
    candidates = projected["design_selection"]["candidates"]
    assert candidates[0]["analysis_family"] == "association"
    assert candidates[1]["analysis_family"] == "descriptive"
    # the projection says what it is, so a reader cannot mistake it for
    # something the plan itself declared
    assert projected["reader_projection"] == {
        "owner": "easyicu.research_agent.planning.step_phase",
        "fields": ["planned_phase", "analysis_family"],
        "persisted": False,
    }


def test_the_projection_never_mutates_or_re_digests_the_plan() -> None:
    """The safety invariant: a reading aid must not cost digest continuity."""

    source = _plan()
    before = json.dumps(source, sort_keys=True)
    project_plan_reader_fields("agent_plan.json", source)
    assert json.dumps(source, sort_keys=True) == before
    assert "planned_phase" not in source["steps"][0]


def test_a_persisted_plan_keeps_its_stored_digest(tmp_path: Path) -> None:
    """Reading a plan through the preview must leave plan_sha256 verifiable."""

    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Q",
            "analysis_type": "association_study",
            "steps": [
                {
                    "step_id": "primary_association",
                    "intent": "Estimate the prespecified adjusted association.",
                    "planned_analysis_role": "primary",
                    "method": "logistic",
                    "expected_outputs": ["table:adjusted_association_estimates"],
                }
            ],
        }
    )
    payload = plan.model_dump(mode="json")
    stored = canonical_sha256(payload)
    artifact = tmp_path / "agent_plan.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    project_plan_reader_fields("agent_plan.json", json.loads(artifact.read_text(encoding="utf-8")))

    reloaded = AnalysisPlan.model_validate(json.loads(artifact.read_text(encoding="utf-8")))
    assert canonical_sha256(reloaded.model_dump(mode="json")) == stored
    # and the compiled field is not part of the typed plan at all
    assert "planned_phase" not in reloaded.model_dump(mode="json")["steps"][0]


def test_other_artifacts_and_broken_payloads_pass_through() -> None:
    other = {"tables": [1, 2]}
    assert project_plan_reader_fields("result_tables.json", other) is other
    assert project_plan_reader_fields("agent_plan.json", None) is None
    assert project_plan_reader_fields("agent_plan.json", []) == []
    # a preview degrades to the un-annotated plan rather than failing
    ragged = {"steps": "not-a-list", "design_selection": "not-a-mapping"}
    assert project_plan_reader_fields("agent_plan.json", ragged) == ragged


def test_the_artifact_service_stamps_the_plan_preview() -> None:
    """The wiring, not just the helper: the plan artifact path applies it."""

    source = Path("src/easyicu/webserver/pi_copilot/service.py").read_text(
        encoding="utf-8"
    )
    assert "from .plan_projection import (" in source
    assert "project_plan_reader_fields," in source
    head = source.index("    def get_research_artifact(")
    tail = source.index("    def get_research_evidence_preview(", head)
    assert "project_plan_reader_fields(clean_artifact, payload)" in source[head:tail]


def test_project_workflow_projects_the_same_plan_into_the_conversation() -> None:
    source = Path("src/easyicu/webserver/pi_copilot/service.py").read_text(
        encoding="utf-8"
    )
    head = source.index("    def get_project_workflow(")
    tail = source.index("    def get_workspace_preview(", head)
    owner = source[head:tail]

    assert "project_plan_conversation_preview" in owner
    assert 'payloads.get("agent_plan.json")' in owner
    assert 'update={"plan_conversation_preview": plan_preview}' in owner
