"""A failed run names the EasyICU type that stopped it.

A planning run was stopped by the signed trajectory design: the compiled plan
did not name one step for each signed execution role, and ``validate_plan``
raised ``TrajectoryScientificAuthorityError``.  The run was recorded as
``research_pipeline_execution_failed`` with an empty ``exception_types``,
because the failure diagnostic recorded only the type names on a hand-kept
list, and this type was not on it.  Its code frames were the only trace of
which owner refused, and the researcher read the generic sentence.

The diagnostic now records every exception class EasyICU defines, by name
only; builtin and third-party classes stay out, and exception text is still
never persisted.  The signed trajectory refusal has its own code and sentence.
Synthetic plans and designs only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from easyicu.research_agent.contracts.trajectory_design import (
    load_trajectory_design,
    normalize_trajectory_design,
    sealed_trajectory_authority_body,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.research_agent.trajectory.scientific_runtime_authority import (
    TrajectoryScientificAuthorityError,
    build_trajectory_scientific_runtime_authority,
)
from easyicu.webserver import agent_pipeline_runs, provider_adapter
from easyicu.webserver import study_contexts as study_context_owner
from easyicu.webserver.dataio import ExportCohortError
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError

_CODE = "research_pipeline_trajectory_authority_refused"
_DESIGN = {
    "coordinate_concepts": ["sofa2_resp", "sofa2_cardio"],
    "window_end_hours": 24,
    "grid_width_hours": 4,
}


def _refusal() -> TrajectoryScientificAuthorityError:
    """The refusal the signed design raises for a plan without its owners."""

    authority = build_trajectory_scientific_runtime_authority(
        sealed_trajectory_authority_body(
            load_trajectory_design(normalize_trajectory_design(_DESIGN)),
            protocol_content_sha256="5" * 64,
        )
    )
    plan = AnalysisPlan(
        research_question="Do organ-support trajectories form distinct groups?",
        steps=[
            AnalysisStep(
                step_id="describe",
                intent="Describe the cohort",
                method="descriptive",
                inputs=[],
                expected_outputs=["table:describe"],
            )
        ],
    )
    with pytest.raises(TrajectoryScientificAuthorityError) as raised:
        authority.validate_plan(plan)
    return raised.value


def _diagnostic(tmp_path: Path, exc: BaseException) -> dict[str, Any]:
    relative = agent_pipeline_runs._write_pipeline_failure_diagnostic(
        wrapper_dir=tmp_path,
        exc=exc,
        code=agent_pipeline_runs._pipeline_failure_code(exc),
    )
    return json.loads((tmp_path / str(relative)).read_text(encoding="utf-8"))


def _raised(exc: Exception, cause: BaseException) -> Exception:
    try:
        raise exc from cause
    except Exception as raised:
        return raised


def test_the_signed_trajectory_refusal_has_its_own_code(tmp_path: Path) -> None:
    refusal = _refusal()

    assert agent_pipeline_runs._pipeline_failure_code(refusal) == _CODE
    payload = _diagnostic(tmp_path, refusal)
    assert payload["code"] == _CODE
    assert payload["exception_types"] == ["TrajectoryScientificAuthorityError"]
    raised_at = payload["traceback_frames"][-1]
    assert (raised_at["file"], raised_at["function"]) == (
        "scientific_runtime_authority.py",
        "validate_plan",
    )
    # The class name crosses the boundary; the refusal's own wording does not.
    assert str(refusal) not in json.dumps(payload)
    assert payload["message"] == "The governed Research Agent operation failed."


def test_the_refusal_keeps_its_code_beneath_an_untyped_wrapper(tmp_path: Path) -> None:
    wrapped = _raised(RuntimeError("plan validation failed"), _refusal())

    assert agent_pipeline_runs._pipeline_failure_code(wrapped) == _CODE
    assert _diagnostic(tmp_path, wrapped)["exception_types"] == [
        "TrajectoryScientificAuthorityError"
    ]


def test_a_typed_owner_in_the_chain_still_decides_the_code() -> None:
    class _CompileStop(ValueError):
        easyicu_safe_diagnostic = {
            "owner": "easyicu.planning.progressive_compiler_v1",
            "reason_code": "progressive_example_stop",
        }

    typed = _raised(_CompileStop("compile stop"), _refusal())

    assert (
        agent_pipeline_runs._pipeline_failure_code(typed)
        == "research_pipeline_progressive_compile_failed"
    )


def test_a_value_error_with_the_same_words_is_not_the_refusal() -> None:
    lookalike = ValueError(
        "signed trajectory authority requires one owner per execution role"
    )

    assert (
        agent_pipeline_runs._pipeline_failure_code(lookalike)
        == "research_pipeline_execution_failed"
    )


def test_every_easyicu_type_in_the_chain_is_recorded_in_order(tmp_path: Path) -> None:
    innermost = _raised(ExportCohortError("cohort_unavailable"), KeyError("column"))
    middle = _raised(_refusal(), innermost)
    outer = _raised(
        ResearchPipelineRunError("research_pipeline_example", "stopped"), middle
    )

    assert _diagnostic(tmp_path, outer)["exception_types"] == [
        "ResearchPipelineRunError",
        "TrajectoryScientificAuthorityError",
        "ExportCohortError",
    ]


def test_builtin_third_party_and_foreign_types_are_not_recorded() -> None:
    class _Model(BaseModel):
        value: int

    with pytest.raises(ValidationError) as invalid:
        _Model(value="not a number")

    # A class outside EasyICU is not recorded even under an EasyICU class name.
    class StructuredResponseFailure(RuntimeError):
        pass

    claimed = type("ExampleRefusal", (ValueError,), {"__module__": "easyicu.example"})
    unnamed = type("not a name", (ValueError,), {"__module__": "easyicu.example"})

    recorded = agent_pipeline_runs._recorded_exception_type
    assert recorded(RuntimeError("x")) is None
    assert recorded(KeyError("x")) is None
    assert recorded(invalid.value) is None
    assert recorded(StructuredResponseFailure("x")) is None
    assert recorded(unnamed("x")) is None
    assert recorded(claimed("x")) == "ExampleRefusal"
    assert recorded(_refusal()) == "TrajectoryScientificAuthorityError"


# -- the run that the refusal stops ------------------------------------------


def _export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    (root / "_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "format": "parquet",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {"demographics": ["age"]},
                },
                "feature_definitions": {"included": False},
                "files": [
                    {
                        "file": "demographics.parquet",
                        "module": "demographics",
                        "concepts": 1,
                        "concept_ids": ["age"],
                        "rows": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class _Job:
    id = "job-signed-design-refusal"
    cancel_requested = False

    def emit(self, _event: dict[str, Any]) -> None:
        return None


def test_the_run_stops_with_the_refusal_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient

    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda *_args, **_kwargs: (
            ScriptedMockLLMClient([]),
            {"provider": "mock", "model": "signed-design-refusal"},
        ),
    )

    def refuse(**_kwargs: Any) -> Any:
        raise _refusal()

    monkeypatch.setattr(
        agent_pipeline_runs, "_metadata_only_planning_acquisition", refuse
    )
    export = _export(tmp_path / "export")
    study = {
        "id": "study-signed-design-refusal",
        "revision": 1,
        "question": "Do organ-support trajectories over the first ICU day form groups?",
        "data_source": {"path": str(export), "database": "miiv"},
        "outcome": "death",
        "analysis_design": {
            "analysis_family": "trajectory_clustering",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
        "trajectory_design": study_context_owner.normalize_trajectory_design(_DESIGN),
    }
    projects = tmp_path / "projects"
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=study,
        project_root=str(projects),
        provider={"provider": "openai", "external": True},
        provider_environment={"OPENAI_API_KEY": "test-key"},
        credential_source="pi_verified",
        budget_mode="planner_canary",
    )

    with pytest.raises(ResearchPipelineRunError) as stopped:
        runner(_Job())

    assert stopped.value.code == _CODE
    assert "trajectory design EasyICU signed" in str(stopped.value)
    assert isinstance(stopped.value.__cause__, TrajectoryScientificAuthorityError)
    (diagnostic,) = projects.rglob("diagnostics/research_pipeline_failure.json")
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert payload["code"] == _CODE
    assert payload["exception_types"] == ["TrajectoryScientificAuthorityError"]
    gate = json.loads(
        (diagnostic.parent.parent / "quality_gate.json").read_text(encoding="utf-8")
    )["gate"]
    assert gate["reason"] == _CODE
