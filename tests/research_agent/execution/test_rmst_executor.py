"""Standalone executor and authority tests for the RMST contrast owner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.current_case_scientific_runtime import (
    build_current_case_scientific_runtime_authority,
)
from easyicu.research_agent.authority.rmst_runtime import RmstRuntimeAuthority
from easyicu.research_agent.contracts.rmst import RMSTSpec
from easyicu.research_agent.execution.runners.rmst_executor import (
    rmst_executor_code,
    run_rmst_contrast,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.methods.rmst import rmst_difference
from easyicu.research_agent.planning.sensitivity_authority import (
    PrespecifiedSensitivitySpec,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver.rmst_runtime_projection import compile_rmst_runtime_projection
from easyicu.webserver.scientific_runtime_projection import (
    WebScientificRuntimeProjectionError,
)


def _spec(**overrides) -> RMSTSpec:
    payload = {
        "time_column": "time_days",
        "event_column": "death",
        "event_code": 1.0,
        "group_column": "lact_group",
        "group_levels": ["high", "low"],
        "tau": 28.0,
        "time_unit": "days",
    }
    payload.update(overrides)
    return RMSTSpec(**payload)


def _authority(spec: RMSTSpec | None = None) -> RmstRuntimeAuthority:
    authority = build_current_case_scientific_runtime_authority(
        {
            "schema_version": "easyicu.rmst_runtime_authority/1",
            "authority_kind": "restricted_mean_survival_difference",
            "protocol_content_sha256": "a" * 64,
            "specification": (spec or _spec()).model_dump(mode="json"),
            "sensitivity_spec_id": "rmst_28d",
            "identity_column": "stay_id",
            "primary_cohort_selection_mode": "all_input_rows",
            "development_execution_only_allowed": True,
            "plan_method": "rmst",
            "plan_intent": "Estimate the prespecified 28-day RMST contrast.",
            "plan_outputs": [
                "table:rmst_summary",
                "log:rmst_runtime_receipt",
            ],
        }
    )
    assert isinstance(authority, RmstRuntimeAuthority)
    return authority


def _frame(n: int = 120) -> pd.DataFrame:
    times = 1.0 + 59.0 * (np.arange(n) / max(n - 1, 1))
    events = (np.arange(n) % 3 != 0).astype(float)
    groups = np.where(np.arange(n) % 2 == 0, "high", "low")
    return pd.DataFrame(
        {
            "stay_id": [f"s{i}" for i in range(n)],
            "time_days": times,
            "death": events,
            "lact_group": groups,
        }
    )


def _step(**overrides) -> AnalysisStep:
    payload = {
        "step_id": "rmst_sensitivity",
        "planned_analysis_role": "sensitivity",
        "intent": "Estimate the prespecified 28-day RMST contrast.",
        "inputs": ["artifact:analysis_cohort"],
        "expected_outputs": ["table:rmst_summary", "log:rmst_runtime_receipt"],
        "method": "rmst",
        "sensitivity_spec_ids": ["rmst_28d"],
    }
    payload.update(overrides)
    return AnalysisStep(**payload)


def test_rmst_contrast_runs_and_matches_the_reviewed_kernel(tmp_path) -> None:
    authority = _authority()
    frame = _frame()

    summary = run_rmst_contrast(
        frame=frame,
        authority=authority,
        runtime_projection_sha256="b" * 64,
        out_dir=tmp_path,
        source_cohort=tmp_path / "cohort.parquet",
    )

    assert summary["status"] == "ok"
    assert (tmp_path / "rmst_summary.csv").is_file()
    receipt = json.loads((tmp_path / "rmst_runtime_receipt.json").read_text())
    assert receipt["claim_ceiling"] == "analysis_only"
    assert receipt["publication_ready"] is False

    kernel = rmst_difference(
        frame["time_days"], frame["death"].astype(bool), frame["lact_group"], 28.0
    )
    expected = float(kernel["diff"])
    if str(kernel["group_a"]) != "high":
        expected = -expected
    assert receipt["estimands"]["difference"] == pytest.approx(expected, abs=1e-9)

    rows = pd.read_csv(tmp_path / "rmst_summary.csv")
    assert set(rows["row_type"]) == {"group", "difference"}
    difference = rows.loc[rows["row_type"] == "difference"].iloc[0]
    assert difference["group"] == "high - low"
    assert difference["tau"] == 28.0
    assert summary["n_events"] == int(frame["death"].sum())


def test_rmst_contrast_reuses_new_columns_levels_and_horizon(tmp_path) -> None:
    spec = _spec(
        time_column="followup_hours",
        event_column="observed_event",
        group_column="arm",
        group_levels=["active", "control"],
        tau=20.0,
        time_unit="hours",
    )
    frame = _frame().rename(
        columns={
            "time_days": "followup_hours",
            "death": "observed_event",
            "lact_group": "arm",
        }
    )
    frame["arm"] = np.where(frame["arm"].eq("high"), "active", "control")

    summary = run_rmst_contrast(
        frame=frame,
        authority=_authority(spec),
        runtime_projection_sha256="c" * 64,
        out_dir=tmp_path,
    )

    assert summary["status"] == "ok"
    receipt = json.loads((tmp_path / "rmst_runtime_receipt.json").read_text())
    assert receipt["estimands"]["reference"] == "active"
    assert receipt["estimands"]["tau"] == 20.0
    assert receipt["estimands"]["time_unit"] == "hours"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda frame: frame.drop(columns=["death"]), "missing declared columns"),
        (
            lambda frame: frame.assign(lact_group="medium"),
            "group levels differ",
        ),
        (
            lambda frame: frame.assign(time_days=1.0),
            "horizon must lie inside observed follow-up",
        ),
        (
            lambda frame: frame.assign(time_days=np.nan),
            "missing or non-finite",
        ),
        (
            lambda frame: frame.assign(
                death=np.zeros(len(frame))
            ),
            "records no observed events",
        ),
        (
            lambda frame: frame.assign(
                death=np.where(np.arange(len(frame)) == 0, 2.0, frame["death"])
            ),
            "outside censor code 0",
        ),
        (
            lambda frame: frame.assign(stay_id=None),
            "identity column has missing or blank",
        ),
    ],
)
def test_rmst_contrast_fails_closed_on_invalid_inputs(tmp_path, mutate, message) -> None:
    frame = mutate(_frame())

    with pytest.raises(ValueError, match=message):
        run_rmst_contrast(
            frame=frame,
            authority=_authority(),
            runtime_projection_sha256="d" * 64,
            out_dir=tmp_path,
        )


def test_rmst_contrast_records_repeated_identity_as_a_limitation(tmp_path) -> None:
    frame = _frame()
    frame.loc[1, "stay_id"] = frame.loc[0, "stay_id"]

    run_rmst_contrast(
        frame=frame,
        authority=_authority(),
        runtime_projection_sha256="e" * 64,
        out_dir=tmp_path,
    )

    receipt = json.loads((tmp_path / "rmst_runtime_receipt.json").read_text())
    assert receipt["duplicate_identity_rows"] == 1
    assert any("not clustered" in item for item in receipt["limitations"])


def test_rmst_input_digest_covers_the_identity_assignment(tmp_path) -> None:
    first = _frame()
    second = first.copy()
    second["stay_id"] = [f"alternate-{index}" for index in range(len(second))]

    run_rmst_contrast(
        frame=first,
        authority=_authority(),
        runtime_projection_sha256="e" * 64,
        out_dir=tmp_path / "first",
    )
    run_rmst_contrast(
        frame=second,
        authority=_authority(),
        runtime_projection_sha256="e" * 64,
        out_dir=tmp_path / "second",
    )

    first_receipt = json.loads(
        (tmp_path / "first" / "rmst_runtime_receipt.json").read_text()
    )
    second_receipt = json.loads(
        (tmp_path / "second" / "rmst_runtime_receipt.json").read_text()
    )
    assert (
        first_receipt["analysis_input_sha256"]
        != second_receipt["analysis_input_sha256"]
    )


def test_rmst_executor_code_seals_the_kernel_and_requires_a_cohort() -> None:
    code = rmst_executor_code(
        _step(),
        authority=_authority(),
        runtime_projection_sha256="f" * 64,
    )
    assert "run_rmst_contrast" in code
    assert "load_step_cohort_frame" in code

    with pytest.raises(ValueError, match="typed cohort input"):
        rmst_executor_code(
            _step(inputs=[]),
            authority=_authority(),
            runtime_projection_sha256="f" * 64,
        )


def test_rmst_authority_governs_exactly_one_matching_step() -> None:
    authority = _authority()
    draft = AnalysisPlan(
        research_question="Estimate RMST.",
        steps=[_step()],
    )
    plan = authority.bind_plan(draft)
    assert authority.governed_step(plan).step_id == "rmst_sensitivity"
    governed = authority.governed_step(plan)
    assert governed.inputs == [
        "artifact:analysis_cohort",
        "stay_id",
        "time_days",
        "death",
        "lact_group",
    ]
    assert governed.icu_rule_refs == [authority.plan_rule_ref]
    assert governed.runtime_outcome_contract is not None
    assert governed.runtime_outcome_contract.owner_ref == authority.plan_rule_ref

    drifted_intent = AnalysisPlan(
        research_question="Estimate RMST.",
        steps=[
            governed.model_copy(
                update={"intent": "A different intent that drifted from the authority."}
            )
        ],
    )
    with pytest.raises(ValueError, match="intent"):
        authority.governed_step(drifted_intent)

    wrong_outputs = AnalysisPlan(
        research_question="Estimate RMST.",
        steps=[
            governed.model_copy(
                update={"expected_outputs": ["table:rmst_summary"]}
            )
        ],
    )
    with pytest.raises(ValueError, match="expected_outputs"):
        authority.governed_step(wrong_outputs)

    unbound = AnalysisPlan(
        research_question="Estimate RMST.",
        steps=[_step(sensitivity_spec_ids=[])],
    )
    with pytest.raises(ValueError, match="exactly one step"):
        authority.governed_step(unbound)


def test_rmst_authority_rejects_a_tampered_specification() -> None:
    payload = _authority().model_dump(mode="json")
    payload["specification"]["tau"] = 29.0

    with pytest.raises(ValueError, match="digest mismatch"):
        RmstRuntimeAuthority.model_validate_json(json.dumps(payload))


def test_rmst_runtime_projection_and_registered_executor_close_the_chain(
    tmp_path,
) -> None:
    universe_path = tmp_path / "analysis_cohort.parquet"
    _frame().rename(columns={"stay_id": "patient_stay_id"}).to_parquet(
        universe_path, index=False
    )
    sensitivity = PrespecifiedSensitivitySpec(
        spec_id="rmst_28d",
        axis="estimand",
        strategy="restricted_mean_survival",
        rmst_execution=_spec(),
    )

    projection = compile_rmst_runtime_projection(
        study={"cohort": {"selection_mode": "all_input_rows"}},
        sensitivity_specs=[sensitivity],
        primary_exposure="lact_group",
        primary_exposure_source="lact_group",
        target_outcome="death",
        declared_covariates=(),
        covariate_operationalizations={},
        target_is_event_status=True,
        universe_path=universe_path,
        scientific_configuration_sha256="a" * 64,
    )
    assert projection is not None
    authority = RmstRuntimeAuthority.model_validate_json(
        json.dumps(projection.authority)
    )
    draft = AnalysisPlan(
        research_question="Estimate RMST.",
        steps=[_step(intent="Planner wording may differ before host binding.")],
    )
    plan = authority.bind_plan(draft)
    step = authority.governed_step(plan)
    selection = select_standard_executor(
        step,
        plan=plan,
        current_case_scientific_runtime_authority=projection.authority,
        scientific_runtime_projection_sha256=projection.projection_sha256,
    )

    assert projection.analysis_only_execution is True
    assert selection.analysis_kind == "signed_rmst_contrast"
    assert authority.execution_contract_sha256 in selection.code
    compile(selection.code, "registered_rmst.py", "exec")


def test_rmst_projection_fails_before_planning_when_a_bound_column_is_missing(
    tmp_path,
) -> None:
    universe_path = tmp_path / "analysis_cohort.parquet"
    _frame().drop(columns=["death"]).rename(
        columns={"stay_id": "patient_stay_id"}
    ).to_parquet(universe_path, index=False)
    sensitivity = PrespecifiedSensitivitySpec(
        spec_id="rmst_28d",
        axis="estimand",
        strategy="restricted_mean_survival",
        rmst_execution=_spec(),
    )

    with pytest.raises(WebScientificRuntimeProjectionError) as caught:
        compile_rmst_runtime_projection(
            study={"cohort": {"selection_mode": "all_input_rows"}},
            sensitivity_specs=[sensitivity],
            primary_exposure="lact_group",
            primary_exposure_source="lact_group",
            target_outcome="death",
            declared_covariates=(),
            covariate_operationalizations={},
            target_is_event_status=True,
            universe_path=universe_path,
            scientific_configuration_sha256="a" * 64,
        )

    assert caught.value.code == "web_rmst_columns_missing"
    assert caught.value.details["missing_columns"] == ["death"]
