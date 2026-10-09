"""A confirmed target trial runs from its study to the Writer's digest.

One synthetic export goes the whole way a causal study goes once its trial is
confirmed.  Data Extraction acquires it with the trial's windows; the host
compiles the stated trial on the context it builds and keeps the record for
the card; the researcher's click approves it; at run start the projection
signs the suite from the approved record and the universe's metadata; the run
compiles the trial again on its own context and finds the approved record;
the signed suite owns the plan; the plan's cohort is the trial's population
as the host's evaluator selects it; the suite estimates the emulation; and
the Writer's evidence digest carries its reportable envelope.  Synthetic
export rows only: the estimate is the simulation's, not a finding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    load_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.cohort.schema import build_cohort
from easyicu.research_agent.execution.runners.target_trial_executor import (
    run_target_trial_suite,
)
from easyicu.research_agent.methods import clone_censor_weight
from easyicu.research_agent.planning.cohort_contract import (
    cohort_concept_id_scope,
    sealed_cohort_concept_ids,
)
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.target_trial_configuration import (
    TargetTrialConfirmationError,
    bind_confirmed_target_trial,
)
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import study_contexts as context_store
from easyicu.webserver.scientific_runtime_projection import (
    compile_web_scientific_runtime_projection,
)
from tests.support.target_trial import STUDY_ID, compiled_target_trial
from tests.support.target_trial_export import (
    ONSET_WINDOWS,
    QUESTION,
    acquire_target_trial,
    export_context,
    export_trial_population,
    export_trial_spec,
    target_trial_export,
)

pytest.importorskip("lifelines")
pytest.importorskip("statsmodels")

#: Fewer resamples than the host's 500 keep the test short; the executed
#: design states the number the bootstrap ran.
RESAMPLES = 20
STEP = "01_primary"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )


def _approved_study(universe: Path) -> dict:
    """The study as the host writes it: the trial, then the researcher's click."""

    population = export_trial_population()
    compiled = compiled_target_trial(
        export_context(universe), spec=export_trial_spec(), population=population
    )
    assert compiled.approvable
    design = {
        "schema_version": "easyicu.target_trial_design/1",
        "spec": compiled.spec.model_dump(mode="json"),
        "population_spec": population.model_dump(mode="json"),
        "compile_record": compiled.record(),
        "compile_sha256": compiled.sha256(),
        "confirmation_lines": len(compiled.confirmations),
    }
    created = context_store.upsert_context(
        {
            "id": STUDY_ID,
            "question": QUESTION,
            # Each stay is its patient's first; the suite's own inference is the
            # host's bootstrap, whatever the design's estimator says.
            "analysis_design": {
                "analysis_family": "causal_inference",
                "analysis_unit": "icu_stay",
                "variance_estimator": "model_based",
            },
        }
    )
    stated = context_store.bind_target_trial_design(
        STUDY_ID, design, expected_revision=created["revision"]
    )
    return context_store.record_target_trial_approval(
        STUDY_ID,
        confirmed_compile_sha256=design["compile_sha256"],
        n_lines_confirmed=design["confirmation_lines"],
        expected_revision=stated["revision"],
    )


def _plan(authority, context) -> AnalysisPlan:
    """The plan the signed suite owns, over the trial's population."""

    population = compile_population(
        export_trial_population(),
        context,
        time_zero_hours=authority.time_zero_hours,
    )
    with cohort_concept_id_scope(sealed_cohort_concept_ids(context)):
        draft = AnalysisPlan.model_validate(
            {
                "research_question": QUESTION,
                "analysis_type": "causal_inference",
                "cohort": population.cohort_definition().plan_dict(),
                "steps": [
                    {
                        "step_id": STEP,
                        "planned_analysis_role": "primary",
                        "intent": "The sealed suite owns this step.",
                        "inputs": ["table:analysis_cohort"],
                        "expected_outputs": ["table:draft"],
                        "method": "signed_target_trial_suite",
                    }
                ],
            }
        )
    return authority.bind_plan(draft)


def test_a_confirmed_target_trial_runs_from_its_study(tmp_path, monkeypatch) -> None:
    export = target_trial_export(tmp_path / "export", n=3000)
    universe = acquire_target_trial(
        export, tmp_path / "trial", onset_windows=ONSET_WINDOWS
    )
    study = _approved_study(universe)

    projection = compile_web_scientific_runtime_projection(
        study=study,
        sensitivity_specs=[],
        primary_exposure=None,
        primary_exposure_source=None,
        target_outcome="mort_28d",
        declared_covariates=[],
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe,
        scientific_configuration_sha256=context_store.scientific_configuration_sha256(
            study
        ),
        dependence=None,
    )
    authority = load_current_case_scientific_runtime_authority(projection.authority)
    approval = study["target_trial_design"]["approval"]
    assert authority.confirmation.approval_event_id == approval["approval_event_id"]

    # The run's own context compiles to the approved record.
    run_context = export_context(universe)
    assert (
        bind_confirmed_target_trial(run_context, projection.bound_target_trial)
        is run_context
    )
    # Extracted again over one window, it would not.
    one_window = acquire_target_trial(
        export, tmp_path / "one_window", cohort_window=(0.0, 24.0)
    )
    with pytest.raises(TargetTrialConfirmationError):
        bind_confirmed_target_trial(
            export_context(one_window), projection.bound_target_trial
        )

    plan = _plan(authority, run_context)
    assert [step.method for step in plan.steps] == [
        "host_materialized_locked_cohort",
        "signed_target_trial_suite",
        "signed_target_trial_figure",
    ]
    frame = pd.read_parquet(universe)
    with cohort_concept_id_scope(sealed_cohort_concept_ids(run_context)):
        cohort = build_cohort(plan.cohort, frame).reset_index(drop=True)
    assert 0 < len(cohort) < len(frame)

    original = clone_censor_weight.bootstrap_clone_censor_weight
    monkeypatch.setattr(
        clone_censor_weight,
        "bootstrap_clone_censor_weight",
        lambda estimate: original(estimate, resamples=RESAMPLES, seed=20261009),
    )
    summary = run_target_trial_suite(
        frame=cohort,
        authority=projection.authority,
        runtime_projection_sha256=projection.projection_sha256,
        out_dir=tmp_path / "suite",
        input_product="table:analysis_cohort",
        input_evidence_id="cohort_evidence",
        input_sha256=hashlib.sha256(b"synthetic analysis cohort").hexdigest(),
    )
    summary = json.loads(json.dumps(summary, allow_nan=False))
    envelope = summary["reportable_target_trial_results"]
    assert envelope["n_eligible"] <= len(cohort)
    assert envelope["adjustment_columns"] == ["age", "lact_max"]
    assert envelope["evidence_ceiling"] == "analysis_only"

    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": STEP,
                "status": "ok",
                "generation_mode": "deterministic_standard",
                "step_summary": summary,
            }
        ],
        run_dir=tmp_path,
        evidence=None,
    )
    lines = digest.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith(f"- {STEP} ["))
    row = json.loads(lines[head + 1])
    assert (
        row["reportable_target_trial_results"]["risk_difference_percentage_points"]
        == (envelope["risk_difference_percentage_points"])
    )
