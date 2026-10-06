"""A research context declares the selection its export records, as it ran.

The Planner reads a context's inclusion and exclusion criteria as already
applied to its input rows, and the manuscript's population block states them.
The Web caller read the raw cohort fields, so a legacy ``adult_first`` preset,
which Data Extraction executes as an adult age bound and the host as a first
ICU stay, declared neither: the block then said no criterion was applied, and
asked the Writer to say the analysis was not restricted to adults.  It also
declared the typed fields of every bound package as applied, although only an
export whose manifest records the contract it was extracted for is known to
hold them, and it compiled a concept-derived population's window from the
neutral materialization scope (24 h) instead of the study the export was
extracted for (720 h when the study states no window).

The criteria are now read as Data Extraction executes them; they are declared
as applied only for an export that records the study's contract, the first
stay always (the host applies it); ``data_constraints.source_selection`` says
how the host knows the export's selection (its ``basis``: the export's
contract, a prepared package's declaration that it is the study's cohort, or
nothing) and which criteria the host applies; and the window is the one the
export executed.  Fixtures are generic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.export_selection import (
    AppliedContracts,
    export_applied_selection,
)
from easyicu.research_agent.research_context.outbound import (
    outbound_safe_context_payload,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    dataio,
    primary_cohort,
    provider_adapter,
    research_pipeline_run_preparation,
)
from easyicu.webserver.agent_pipeline_runs import (
    _exclusion_criteria,
    _inclusion_criteria,
    _research_user_preferences,
)
from easyicu.webserver.pi_copilot.extraction_handoff import compile_study_cohort
from easyicu.webserver.research_launch_scientific import (
    bound_export_selection_basis,
    launch_materialization_window,
)
from tests.webserver.copilot.research_workflow_fixtures import (
    _acquisition_receipt,
    _foundation_profile,
    _write_real_pipeline_fixture,
    complete_study,
    confirmed_cohort_decision,
)

_FIRST_STAY = (
    "each patient's later ICU stays: the host keeps only the first ICU stay "
    "per patient across the bound source, before planning"
)


def _study(**cohort: Any) -> dict[str, Any]:
    return {"cohort": cohort}


def _export(root: Path, study: dict[str, Any] | None, **manifest: Any) -> Path:
    """A prepared package; with ``study``, it records the contract it was extracted for."""

    root.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "age": [65]}).to_parquet(
        root / "demographics.parquet", index=False
    )
    record: dict[str, Any] = {
        "database": "miiv",
        "format": "parquet",
        "concept_selection": {"mode": "explicit", "modules": {"demographics": ["age"]}},
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
        **manifest,
    }
    if study is not None:
        contract = compile_study_cohort(study)
        record["cohort_contract"] = contract
        record["cohort_execution"] = dataio.export_cohort_execution(contract)
    (root / "_manifest.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def test_a_preset_declares_the_criteria_it_executes() -> None:
    adult_first = _study(preset="adult_first")

    assert _inclusion_criteria(adult_first) == ["age range: 18 to *"]
    assert _exclusion_criteria(adult_first) == [_FIRST_STAY]
    # The same declaration as the fields the preset stands for.
    explicit = _study(age_min=18, exclude_readmissions=True)
    assert _inclusion_criteria(explicit) == _inclusion_criteria(adult_first)
    assert _exclusion_criteria(explicit) == _exclusion_criteria(adult_first)
    # A field the researcher set is declared as the preset executes it.
    assert _inclusion_criteria(_study(preset="adult_first", age_min=10)) == [
        "age range: 18 to *"
    ]
    assert _inclusion_criteria(_study(preset="adult_first", age_min=65, age_max=80)) == [
        "age range: 65 to 80"
    ]
    assert _exclusion_criteria(_study(preset="adult_first", exclude_readmissions=False)) == []


def test_a_bound_that_executes_nothing_is_declared_as_nothing() -> None:
    for cohort in (
        {"age_min": 0, "age_max": 100, "min_icu_los_hours": 0},
        {"age_max": 120},
        {"preset": "all_icu"},
    ):
        assert _inclusion_criteria({"cohort": cohort}) == [], cohort
        assert _exclusion_criteria({"cohort": cohort}) == [], cohort
    assert _inclusion_criteria(_study(age_min="18", age_max=80, min_icu_los_hours=24.0)) == [
        "age range: 18 to 80",
        "minimum ICU length of stay: 24 hours",
    ]


def test_an_unrecorded_export_declares_only_what_the_host_applies() -> None:
    study = _study(
        age_min=18,
        min_icu_los_hours=24,
        exclude_readmissions=True,
        icd_enabled=True,
        include_diagnoses=["condition-a"],
        exclude_diagnoses=["condition-b"],
    )

    assert _inclusion_criteria(study, export_recorded=False) == []
    assert _exclusion_criteria(study, export_recorded=False) == [_FIRST_STAY]
    assert _inclusion_criteria(study) == [
        "age range: 18 to *",
        "minimum ICU length of stay: 24 hours",
        "include diagnoses: condition-a",
    ]
    assert _exclusion_criteria(study) == [_FIRST_STAY, "exclude diagnoses: condition-b"]


def test_the_basis_says_how_the_host_knows_the_export_selection(tmp_path: Path) -> None:
    study = _study(preset="all_icu", age_min=18)

    def basis(path: Path | str | None, of: dict[str, Any] = study) -> str:
        return bound_export_selection_basis(of, None if path is None else str(path))

    assert basis(_export(tmp_path / "held", study)) == "export_contract"
    # Extracted for another cohort.
    assert basis(_export(tmp_path / "other", _study(preset="all_icu", age_min=65))) == (
        "unrecorded"
    )
    # A study-local prepared cohort declares itself the study's input; a
    # package that records a contract is judged by its contract.
    prepared = _export(tmp_path / "prepared", None, entry_mode="study_local_prepared_cohort")
    assert basis(prepared) == "package_declaration"
    contracted = _export(
        tmp_path / "contracted", _study(preset="all_icu", age_min=65),
        entry_mode="study_local_prepared_cohort",
    )
    assert basis(contracted) == "unrecorded"
    # An export from before contracts were recorded, or one whose recorded
    # contract is not an executable extraction contract.
    assert basis(_export(tmp_path / "older", None)) == "unrecorded"
    unreadable = _export(
        tmp_path / "unreadable", None, cohort_contract={"preset": "no_such_preset"}
    )
    assert basis(unreadable) == "unrecorded"
    # No manifest, no folder, no path.
    (tmp_path / "bare").mkdir()
    assert basis(tmp_path / "bare") == "unrecorded"
    assert basis(tmp_path / "missing") == "unrecorded"
    assert basis(None) == "unrecorded"
    # A concept population executed under the earlier rule (no execution record).
    concept = _study(preset="sepsis3")
    assert basis(_export(tmp_path / "now", concept), concept) == "export_contract"
    earlier = _export(tmp_path / "earlier", concept)
    manifest = json.loads((earlier / "_manifest.json").read_text(encoding="utf-8"))
    del manifest["cohort_execution"]
    (earlier / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert basis(earlier, concept) == "unrecorded"


def test_the_source_selection_reaches_the_planner_context() -> None:
    study = _study(preset="adult_first", label="Adults with condition-x")

    for basis in ("export_contract", "unrecorded", "package_declaration"):
        recorded = basis == "export_contract"
        context = build_research_context(
            research_question="Among adults with condition-x, is an early marker associated with an outcome?",
            cohort=pd.DataFrame({"stay_id": [1, 2], "age": [34.0, 61.0]}),
            cohort_name="synthetic",
            database="synthetic",
            inclusion_criteria=_inclusion_criteria(study, export_recorded=recorded),
            exclusion_criteria=_exclusion_criteria(study, export_recorded=recorded),
            user_preferences=_research_user_preferences(
                study, cohort_study=study, source_selection_basis=basis
            ),
        )
        payload = outbound_safe_context_payload(context)
        stated = json.loads(payload["study_preferences"]["data_constraints"])

        assert stated["source_selection"] == {
            "basis": basis,
            "host_applied": [_FIRST_STAY],
        }
        # Each host-applied criterion is one the context declares, verbatim.
        assert payload["cohort"]["exclusion_contract"] == [_FIRST_STAY]
        assert payload["cohort"].get("inclusion_contract") == (
            ["age range: 18 to *"] if recorded else None
        )
        # The research-context reader knows the host's criterion is applied
        # either way, and the export's only when its selection is recorded.
        selection = export_applied_selection(context)
        assert selection.basis == basis
        assert selection.recorded is recorded
        assert selection.host_applied == AppliedContracts(exclusion=(_FIRST_STAY,))
        assert selection.known_applied == AppliedContracts(
            inclusion=("age range: 18 to *",) if recorded else (),
            exclusion=(_FIRST_STAY,),
        )
        assert selection.unverified == AppliedContracts()
    # A study without the first stay has nothing the host applies.
    plain = _study(age_min=18)
    stated = json.loads(
        _research_user_preferences(
            plain, cohort_study=plain, source_selection_basis="export_contract"
        )["data_constraints"]
    )
    assert stated["source_selection"] == {"basis": "export_contract", "host_applied": []}
    # A caller that does not know the bound export states nothing about it.
    assert "source_selection" not in json.loads(
        _research_user_preferences(study)["data_constraints"]
    )


def test_a_concept_population_keeps_the_window_its_export_executed() -> None:
    study = _study(preset="sepsis3")  # states no time window
    executed = dataio.export_cohort_execution(compile_study_cohort(study))
    neutral = {**study, "time_window": launch_materialization_window(study)}

    stated = json.loads(
        _research_user_preferences(
            neutral, cohort_study=study, source_selection_basis="export_contract"
        )["data_constraints"]
    )

    assert executed["concept_cohort_window"]["window_end_hours"] == (
        primary_cohort.DEFAULT_OBSERVATION_WINDOW_HOURS
    )
    assert stated["concept_cohort_window"] == {
        "definition": "sepsis3",
        "window_end_hours": executed["concept_cohort_window"]["window_end_hours"],
    }
    # The neutral scope's outer window selected no one.
    assert neutral["time_window"]["hours"] == 24
    # A study that states its window keeps it.
    windowed = {**study, "time_window": {"observation_hours": 48, "anchor": "ICU admission"}}
    stated = json.loads(
        _research_user_preferences(
            windowed, cohort_study=windowed, source_selection_basis="export_contract"
        )["data_constraints"]
    )
    assert stated["concept_cohort_window"]["window_end_hours"] == 48


_PROVIDER_ENVIRONMENT = {
    "OPENAI_API_KEY": "test-private-provider-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
    "OPENAI_MODEL": "test-local-model",
    "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
}


def _adult_study() -> dict[str, Any]:
    study = complete_study()
    cohort, authority = confirmed_cohort_decision(
        "confirm_current_cohort",
        study_context_id=study["id"],
        study_context_revision=study["revision"],
        current_cohort={"max_patients": 2000, "age_min": 18},
    )
    study["cohort"] = cohort
    study["cohort_eligibility_authority"] = authority
    return study


@pytest.mark.parametrize("basis", ["export_contract", "unrecorded", "package_declaration"])
def test_the_web_runner_declares_what_its_bound_export_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, basis: str
) -> None:
    from easyicu.research_agent.acquisition import foundation
    from easyicu.research_agent.execution import runner as runner_module
    import easyicu.research_agent as research_agent

    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
    )
    actual_run = tmp_path / "actual-pipeline-run"
    _write_real_pipeline_fixture(actual_run, manuscript="# Results\nAnalysis-only.")
    universe = tmp_path / "universe.parquet"
    universe.write_bytes(b"typed-universe-placeholder")
    acquisition = _acquisition_receipt()
    acquisition.blocked = False
    acquisition.universe_path = universe
    acquisition.cohort_authority_path = None
    acquisition.cohort_authority_ref = None
    acquisition.trajectory_path = None
    acquisition.trajectory_authority_path = None
    acquisition.trajectory_authority_ref = None
    calls: dict[str, Any] = {}
    window_studies: list[dict[str, Any]] = []
    real_window = agent_pipeline_runs._study_concept_cohort_window

    def spy_window(study: Any) -> Any:
        window_studies.append(dict(study))
        return real_window(study)

    class FakePipeline:
        def run(self, **kwargs: Any) -> Any:
            calls["run"] = kwargs
            return type("Result", (), {"manifest_path": actual_run / "manifest.json"})()

    monkeypatch.setattr(agent_pipeline_runs, "_study_concept_cohort_window", spy_window)
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda provider, **kwargs: (object(), {"provider": "openai", "model": "test-model"}),
    )
    monkeypatch.setattr(foundation, "acquire_universe_for_question", lambda **kwargs: acquisition)
    monkeypatch.setattr(
        research_pipeline_run_preparation,
        "_data_foundation_profile",
        lambda **_kwargs: _foundation_profile(),
    )
    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda config, *, services: FakePipeline(),
    )
    study = _adult_study()
    del study["time_window"]  # the launch's neutral scope fills an outer window
    recorded = basis == "export_contract"
    export = _export(
        tmp_path / "export",
        study if recorded else None,
        **(
            {"entry_mode": "study_local_prepared_cohort"}
            if basis == "package_declaration"
            else {}
        ),
    )
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=str(export),
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PROVIDER_ENVIRONMENT,
        budget_mode="full_reviewed",
        runner_image="easyicu-research-agent:test",
    )

    class Job:
        id = "job-source-selection"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    runner(Job())

    run = calls["run"]
    stated = json.loads(run["user_preferences"]["data_constraints"])
    assert stated["source_selection"] == {"basis": basis, "host_applied": []}
    assert run["inclusion_criteria"] == (["age range: 18 to *"] if recorded else [])
    assert run["exclusion_criteria"] == []
    # The planning study carries the neutral outer window; the population's
    # window is compiled from the study the export was extracted for.
    assert stated["materialization_window"]["hours"] == 24
    assert window_studies and all("time_window" not in item for item in window_studies)
