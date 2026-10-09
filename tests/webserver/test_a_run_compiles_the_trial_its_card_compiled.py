"""A run of a causal study compiles the target trial its card compiled.

The host's compile for a trial's card and the run of the study once its
researcher approved the card prepare one launch, acquire one universe and
declare one research context (``agent_pipeline_runs``).  So the run's binder
(``bind_confirmed_target_trial``) compiles the approved record again on the
context the pipeline builds from what the run passes, once the signed suite
bound its endpoint, and finds it.  The run reads the trial's windows and
names the trial's outcome, no exposure and no endpoint of its own.  A trial
the run compiles to another record -- the host's rules changed after the
approval -- stops the run before planning, and an approved trial is never
planned as a metadata-only candidate.  Synthetic export rows only: no
finding is drawn from them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import easyicu.research_agent as research_agent
from easyicu.research_agent.acquisition.first_icu_stay import (
    COORDINATE_FLAG_COLUMN,
    COORDINATE_STAY_COLUMN,
    FirstIcuStayBinding,
)
from easyicu.research_agent.execution import runner as runner_module
from easyicu.research_agent.orchestration.reviewed_requirements import (
    bind_reviewed_requirements,
)
from easyicu.research_agent.orchestration.scientific_runtime import (
    ScientificRuntimeAuthorities,
)
from easyicu.research_agent.planning import target_trial_compile
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.target_trial_compile import (
    compile_target_trial,
    target_trial_context_endpoint,
)
from easyicu.research_agent.planning.target_trial_configuration import (
    TARGET_TRIAL_COMPILE_DRIFTED,
    TargetTrialConfirmationError,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.webserver import (
    agent_pipeline_runs,
    dataio,
    jobs,
    provider_adapter,
    source_identity_authority,
    target_trial_records,
)
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver import target_trial_setup as setup
from easyicu.webserver.pi_copilot.extraction_handoff import compile_study_cohort
from easyicu.webserver.research_pipeline_run_errors import ResearchPipelineRunError
from easyicu.webserver.target_trial_card import approve_target_trial
from tests.support.target_trial import STUDY_ID, TIME_ZERO
from tests.support.target_trial_export import (
    QUESTION,
    export_trial_population,
    export_trial_spec,
    target_trial_export,
)

#: Each stay is its patient's first, as the host's coordinate states.
_COHORT = {"preset": "all_icu", "exclude_readmissions": True}
_PROVIDER_ENVIRONMENT = {
    "OPENAI_API_KEY": "test-private-provider-key",
    "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
    "OPENAI_MODEL": "test-local-model",
    "EASYICU_DISABLE_PROVIDER_ENV_FILE": "1",
}


class _Captured(BaseException):
    """The pipeline was handed its run: nothing past it is this test's."""


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )
    monkeypatch.setattr(
        target_trial_records, "records_root", lambda: tmp_path / "records"
    )
    monkeypatch.setattr(jobs, "MANAGER", jobs.JobManager())


def _export(root: Path, study: dict[str, Any], *, seed: int) -> Path:
    """The synthetic export, recording the cohort contract it was extracted for."""

    if root.exists():
        shutil.rmtree(root)
    target_trial_export(root, n=600, seed=seed)
    manifest_path = root / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = compile_study_cohort(study)
    manifest["cohort_contract"] = contract
    manifest["cohort_execution"] = dataio.export_cohort_execution(contract)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _first_stays(
    export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host's first-stay coordinate: every synthetic stay is a first one."""

    stays = pd.read_parquet(export / "demographics.parquet")["stay_id"]
    path = tmp_path / "first_icu_stay.parquet"
    pd.DataFrame(
        {COORDINATE_STAY_COLUMN: stays.astype(int), COORDINATE_FLAG_COLUMN: True}
    ).to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = FirstIcuStayBinding(
        coordinate_path=path,
        coordinate_sha256=digest,
        authority_coordinates={
            "authority_ref": "synthetic/first_icu_stay",
            "order_rule": "earliest_order_time_per_patient",
            "stays": int(len(stays)),
            "patients": int(len(stays)),
            "non_first_icu_stays": 0,
        },
    )
    monkeypatch.setattr(
        source_identity_authority,
        "resolve_study_first_icu_stay",
        lambda **_kwargs: binding,
    )


def _finished(job: Any, seconds: float = 120.0) -> Any:
    deadline = time.monotonic() + seconds
    while job.status == "running":
        if time.monotonic() > deadline:
            raise AssertionError("the compile job did not finish")
        time.sleep(0.02)
    return job


def _approved_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, str]:
    """The study as the host leaves it: stated, compiled for its card, approved."""

    study = {
        "id": STUDY_ID,
        "question": QUESTION,
        "cohort": dict(_COHORT),
        "analysis_design": {
            "analysis_family": "causal_inference",
            "analysis_unit": "icu_stay",
            "variance_estimator": "model_based",
        },
    }
    export = _export(tmp_path / "export", study, seed=20261009)
    _first_stays(export, tmp_path, monkeypatch)
    stored = context_store.upsert_context(
        {**study, "data_source": {"path": str(export), "database": "miiv"}}
    )
    statement = setup.read_target_trial_statement(
        {
            "spec": export_trial_spec().model_dump(mode="json"),
            "population_spec": export_trial_population().model_dump(mode="json"),
        }
    )
    job = _finished(setup.submit_target_trial_compile(stored, statement))
    compiled = job.result["target_trial_compile"]
    assert (compiled["status"], compiled["reason_code"]) == ("compiled", None), compiled
    assert compiled["approvable"] is True
    stated = context_store.get_context(STUDY_ID)
    design = stated["target_trial_design"]
    approve_target_trial(
        STUDY_ID,
        expected_revision=stated["revision"],
        compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
    )
    return context_store.get_context(STUDY_ID), compiled["compile_sha256"]


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, study: dict) -> dict:
    """Launch the study's run as the host does; capture what the pipeline is handed."""

    captured: dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, config: Any) -> None:
            captured["config"] = config

        def run(self, **kwargs: Any) -> Any:
            captured["run"] = kwargs
            raise _Captured()

    monkeypatch.setattr(
        runner_module,
        "probe_runner_availability",
        lambda kind, **_kwargs: runner_module.RunnerAvailability(
            kind=kind, available=True, image="easyicu-research-agent:test"
        ),
    )
    monkeypatch.setattr(
        provider_adapter,
        "build_research_agent_provider_client",
        lambda provider, **kwargs: (object(), {"provider": "openai", "model": "test-model"}),
    )
    monkeypatch.setattr(
        research_agent.ResearchAgentPipeline,
        "from_config",
        lambda config, *, services: FakePipeline(config),
    )
    runner = agent_pipeline_runs.make_research_pipeline_run_runner(
        export_path=study["data_source"]["path"],
        study_context=study,
        project_root=str(tmp_path / "projects"),
        provider={"provider": "openai", "external": True},
        provider_environment=_PROVIDER_ENVIRONMENT,
        budget_mode="full_reviewed",
        runner_image="easyicu-research-agent:test",
    )

    class Job:
        id = "job-target-trial-run"
        cancel_requested = False
        events: list[dict[str, Any]] = []

        def emit(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))

    with pytest.raises(_Captured):
        runner(Job())
    return captured


def _run_context(captured: dict) -> Any:
    """The research context the pipeline builds from the run's arguments.

    As ``ResearchAgentPipeline._run_plan_phase`` builds it: the sealed
    authorities bind their declarations onto the run's, then the context is
    built on the universe's rows (the pipeline reads a staged copy; the
    record names no path).
    """

    config, run = captured["config"], captured["run"]
    endpoint, exposure, preferences = ScientificRuntimeAuthorities.load(
        trajectory=config.trajectory_scientific_runtime_authority,
        current_case=config.current_case_scientific_runtime_authority,
    ).bind_run_inputs(
        endpoint=run["endpoint"],
        primary_exposure=run["primary_exposure"],
        user_preferences=run["user_preferences"],
    )
    assert run["trajectory_path"] is None
    return build_research_context(
        research_question=run["question"],
        cohort=Path(run["cohort"]),
        cohort_name=run["cohort_name"],
        database=run["database"],
        target_outcome=run["target_outcome"],
        endpoint=endpoint,
        primary_exposure=exposure,
        inclusion_criteria=run["inclusion_criteria"],
        exclusion_criteria=run["exclusion_criteria"],
        id_columns=run["id_columns"],
        outcome_columns=run["outcome_columns"],
        concept_descriptions=run["concept_descriptions"],
        time_windows=run["time_windows"],
        user_preferences=preferences,
        notes=run["notes"],
        trajectory_binding=None,
    )


def test_a_run_compiles_the_record_its_card_compiled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, approved_sha256 = _approved_study(tmp_path, monkeypatch)

    captured = _run(tmp_path, monkeypatch, study)

    run = captured["run"]
    # The trial's outcome, no exposure, no endpoint of the run's own.
    assert (run["primary_exposure"], run["endpoint"]) == (None, None)
    assert run["outcome_columns"] == (run["target_outcome"],)
    # The window the run summarized its columns over is the trial's [0, T0).
    assert [(w.start_hours, w.end_hours) for w in run["time_windows"]] == [
        (0.0, float(TIME_ZERO))
    ]
    constraints = json.loads(run["user_preferences"]["data_constraints"])
    assert constraints["materialization_window"]["hours"] == float(TIME_ZERO)
    context = _run_context(captured)
    # The endpoint the suite binds is the one the card's compile stated.
    assert context.endpoint == target_trial_context_endpoint(export_trial_spec())
    spec = export_trial_spec()
    population = compile_population(
        export_trial_population(),
        context,
        time_zero_hours=spec.time_zero.hours_after_icu_admission,
    )
    assert (
        compile_target_trial(spec, context, population=population).sha256()
        == approved_sha256
    )
    # The run binds the approved trial, and the pipeline's binder finds its
    # record on the run's context.
    bound = captured["config"].bound_target_trial
    assert bound is not None and bound["compile_sha256"] == approved_sha256
    bind_reviewed_requirements(context, captured["config"])


def test_an_approved_trial_is_not_planned_as_a_metadata_only_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _approved = _approved_study(tmp_path, monkeypatch)

    with pytest.raises(ResearchPipelineRunError) as refused:
        agent_pipeline_runs.make_research_pipeline_run_runner(
            export_path=study["data_source"]["path"],
            study_context=study,
            project_root=str(tmp_path / "projects"),
            provider={"provider": "openai", "external": True},
            provider_environment=_PROVIDER_ENVIRONMENT,
            budget_mode="planner_canary",
            runner_image="easyicu-research-agent:test",
        )

    assert refused.value.code == "research_pipeline_target_trial_plans_on_data"


def test_a_record_its_run_compiles_otherwise_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, _approved = _approved_study(tmp_path, monkeypatch)
    # After the approval the host no longer offers the trial's time zero, as
    # an update could change it: the run compiles the trial to another record.
    monkeypatch.setattr(
        target_trial_compile,
        "TIME_ZERO_MENU_HOURS",
        tuple(
            hours
            for hours in target_trial_compile.TIME_ZERO_MENU_HOURS
            if hours != TIME_ZERO
        ),
    )

    captured = _run(tmp_path, monkeypatch, study)

    # The run stops before its plan: the approved record is not executed.
    with pytest.raises(TargetTrialConfirmationError) as stopped:
        bind_reviewed_requirements(_run_context(captured), captured["config"])
    assert stopped.value.reason_code == TARGET_TRIAL_COMPILE_DRIFTED
