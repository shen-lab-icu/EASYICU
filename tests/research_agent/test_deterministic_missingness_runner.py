"""Regression: the deterministic missingness/measurement audit runner produces a
per-concept audit table without an LLM coder call.

Built 2026-07-08 for E3 + M1. The missingness/measurement audit is a pure count
(measured vs missing fraction per concept + a structural-vs-measurement split),
yet the LLM coder reliably exhausted its retry budget on it (~27.6 min, IDENTICAL
across two real E3 runs) and failed with no code, blocking the whole run on
``execution_complete``. The deterministic runner removes both the flakiness and
the dominant coder round-trip.

The tests exec the runner's code string against synthetic cohorts (no real data),
asserting: it uses the ``<concept>_measured`` indicator as the authoritative
measurement signal, narrowly distinguishes a complete binary event-status flag
from measurement availability, never imputes, distinguishes structural-no-source
from measurement missingness, and blocks gracefully on an empty cohort.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.plausibility import (
    FlagOnlyPlausibilityScope,
)
from easyicu.research_agent.execution.runners.deterministic_missingness import (
    is_missingness_complete_case_contract,
    is_missingness_measurement_availability_contract,
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
    source_availability_audit_executor_owns_step,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.gates.plausibility_receipt import (
    plausibility_audit_receipt_findings,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _exec_runner(
    run_dir: Path,
    cohort: pd.DataFrame,
    context: dict,
    *,
    requested_inputs: list[str] | None = None,
    requested_outputs: list[str] | None = None,
    current_plan_name: str | None = None,
):
    out_dir = run_dir / "steps" / "02_missingness_measurement_audit" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "research_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    if requested_inputs is not None:
        plan_name = current_plan_name or "analysis_plan.json"
        plan_path = run_dir / plan_name
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "step_id": "02_missingness_measurement_audit",
                            "inputs": requested_inputs,
                            "expected_outputs": [
                                *(
                                    requested_outputs
                                    or [
                                        "table:missingness_measurement_audit",
                                        "table:analytic_denominators",
                                    ]
                                ),
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        if current_plan_name is not None:
            (run_dir / "manifest_partial.json").write_text(
                json.dumps({"plan_path": current_plan_name}),
                encoding="utf-8",
            )
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = missingness_measurement_audit_code()
        try:
            exec(compile(code, "<det_missingness>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return (
        json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8")),
        out_dir,
    )


def _cohort(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # lactate: measured for 60% of stays; crea: measured for 90%; rrt: measured
    # for NO stay (structural no-source in this cohort).
    lact_measured = (rng.random(n) < 0.60).astype(int)
    crea_measured = (rng.random(n) < 0.90).astype(int)
    rrt_measured = np.zeros(n, dtype=int)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "age": rng.integers(40, 90, n),
            "sex": rng.choice(["Male", "Female"], n),
            "death": (rng.random(n) < 0.12).astype(int),
            "los_icu": rng.gamma(2.0, 2.0, n),
            "lactate": np.where(lact_measured == 1, rng.gamma(2.0, 1.5, n), np.nan),
            "lactate_measured": lact_measured,
            "lactate_n": lact_measured,
            "crea": np.where(crea_measured == 1, rng.gamma(2.0, 0.6, n), np.nan),
            "crea_measured": crea_measured,
            "crea_n": crea_measured,
            "rrt": np.full(n, np.nan),
            "rrt_measured": rrt_measured,
            "rrt_n": rrt_measured,
        }
    )


@pytest.mark.parametrize(
    ("method", "availability_product"),
    [
        (
            "missingness_and_source_availability_audit",
            "table:measurement_source_audit",
        ),
        (
            "missingness_and_measurement_frequency_audit",
            "table:measurement_availability",
        ),
        (
            "missingness_and_informative_measurement_audit",
            "table:measurement_availability_audit",
        ),
    ],
)
def test_structured_availability_contract_is_selected_before_any_coder_path(
    method: str,
    availability_product: str,
):
    step = AnalysisStep(
        step_id="03_missingness_measurement_audit",
        intent="Audit missingness and source availability without imputing zero.",
        inputs=[
            "artifact:analysis_cohort",
            "aki_stage_max",
            "aki_stage_measured",
            "aki_stage_n",
        ],
        expected_outputs=[
            "table:missingness_audit",
            availability_product,
        ],
        method=method,
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])

    assert is_missingness_measurement_availability_contract(
        step.method,
        step.expected_outputs,
    )
    assert source_availability_audit_executor_owns_step(step)
    selection = select_standard_executor(step, plan=plan)
    assert selection is not None
    assert selection.analysis_kind == "missingness_source_availability_audit"
    assert "missingness_measurement_audit.csv" in selection.code
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_structured_missingness_complete_case_contract_is_selected():
    step = AnalysisStep(
        step_id="05_missingness_audit",
        intent="Audit missingness and complete-case attrition without imputation.",
        inputs=[
            "artifact:analysis_cohort",
            "exposure",
            "outcome",
            "age",
        ],
        expected_outputs=[
            "table:missingness_profile",
            "table:complete_case_attrition",
        ],
        method="missingness_and_complete_case_audit",
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])

    assert is_missingness_complete_case_contract(
        step.method,
        step.expected_outputs,
    )
    assert missingness_audit_executor_owns_step(step)
    selection = select_standard_executor(step, plan=plan)
    assert selection is not None
    assert selection.analysis_kind == "missingness_complete_case_audit"
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_structured_missingness_complete_case_contract_rejects_wider_scope():
    extra_output = AnalysisStep(
        step_id="05_missingness_audit",
        intent="Audit missingness and fit an effect model.",
        inputs=["artifact:analysis_cohort", "exposure", "outcome"],
        expected_outputs=[
            "table:missingness_profile",
            "table:complete_case_attrition",
            "table:adjusted_association_estimates",
        ],
        method="missingness_and_complete_case_audit",
    )
    foreign_input = AnalysisStep(
        step_id="05_missingness_audit",
        intent="Reconcile missingness from an unreviewed table.",
        inputs=["table:unreviewed_reconciliation", "age"],
        expected_outputs=[
            "table:missingness_profile",
            "table:complete_case_attrition",
        ],
        method="missingness_and_complete_case_audit",
    )

    assert not missingness_audit_executor_owns_step(extra_output)
    assert not missingness_audit_executor_owns_step(foreign_input)


@pytest.mark.parametrize(
    ("method", "expected_outputs"),
    [
        (
            "longitudinal_missingness_and_measurement_availability_audit",
            [
                "table:missingness_audit",
                "table:measurement_availability_audit",
            ],
        ),
        (
            "missingness_and_measurement_model_audit",
            [
                "table:missingness_audit",
                "table:measurement_availability_audit",
            ],
        ),
        (
            "missingness_and_informative_measurement_audit",
            [
                "table:missingness_audit",
                "table:measurement_availability_audit",
                "table:adjusted_association_estimates",
            ],
        ),
        (
            "missingness_and_informative_measurement_audit",
            [
                "table:missingness_audit",
                "test:missingness_mechanism",
            ],
        ),
        (
            "missingness_and_informative_measurement_audit",
            [
                "table:missingness_audit",
                "table:score_quality_audit",
            ],
        ),
    ],
)
def test_structured_availability_contract_rejects_richer_or_unknown_science(
    method: str,
    expected_outputs: list[str],
):
    step = AnalysisStep(
        step_id="03_missingness_measurement_audit",
        intent="Audit missingness and source availability.",
        inputs=["artifact:analysis_cohort", "aki_stage_max"],
        expected_outputs=expected_outputs,
        method=method,
    )

    assert not is_missingness_measurement_availability_contract(
        method,
        expected_outputs,
    )
    assert not source_availability_audit_executor_owns_step(step)


def test_structured_availability_executor_requires_exact_analysis_cohort_scope():
    step = AnalysisStep(
        step_id="03_missingness_measurement_audit",
        intent="Audit missingness and source availability.",
        inputs=[
            "artifact:analysis_cohort",
            "table:unreviewed_source_reconciliation",
            "aki_stage_max",
        ],
        expected_outputs=[
            "table:missingness_audit",
            "table:measurement_availability_audit",
        ],
        method="missingness_and_informative_measurement_audit",
    )

    assert is_missingness_measurement_availability_contract(
        step.method,
        step.expected_outputs,
    )
    assert not source_availability_audit_executor_owns_step(step)


def test_structured_availability_executor_accepts_implicit_locked_cohort_scope():
    step = AnalysisStep(
        step_id="04_missingness_and_measurement_audit",
        intent="Audit missingness and measurement availability.",
        inputs=[
            "sep3_sofa2_max",
            "sep3_sofa2_measured",
            "death",
            "age",
            "lact_max",
            "lact_measured",
        ],
        expected_outputs=[
            "table:missingness_audit",
            "table:measurement_availability_audit",
        ],
        method="missingness_and_measurement_frequency_audit",
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])

    assert source_availability_audit_executor_owns_step(step)
    selection = select_standard_executor(step, plan=plan)
    assert selection is not None
    assert selection.analysis_kind == "missingness_source_availability_audit"
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_compact_missingness_executor_consumes_declared_cohort_product():
    step = AnalysisStep(
        step_id="03_missingness_and_measurement_audit",
        intent="Audit missingness and measurement availability.",
        inputs=[
            "cohort:analysis_set",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        expected_outputs=["table:missingness_measurement_audit"],
        method="missingness_measurement_audit",
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])

    selection = select_standard_executor(step, plan=plan)

    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_audit"
    assert selection.consumed_input_keys == ("cohort:analysis_set",)
    assert audit_mechanical_code_contracts(selection.code, step) == []


def test_compact_missingness_executor_emits_exact_plausibility_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step = AnalysisStep(
        step_id="04_missingness_and_measurement_audit",
        planned_analysis_role="auxiliary",
        intent="Audit missingness without changing the analysis cohort.",
        inputs=["artifact:analysis_cohort", "age"],
        expected_outputs=["table:missingness_measurement_audit"],
        method="missingness_measurement_audit",
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])
    frame = pd.DataFrame({"age": [-1.0, 50.0, 101.0]})
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    (run_dir / "research_context.json").write_text("{}", encoding="utf-8")
    (run_dir / "analysis_plan.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )

    raw_contracts: dict[str, object] = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "authority_scope": (
            "host_verified_physical_representation_and_domain_constraints"
        ),
        "scientific_ownership": "Planner retains scientific decisions",
        "contracts": {
            "age": {
                "column": "age",
                "analysis_plausibility_range": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
        },
    }
    encoded_contracts = json.dumps(
        raw_contracts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    contracts_sha256 = hashlib.sha256(encoded_contracts).hexdigest()
    raw_contracts["contracts_sha256"] = contracts_sha256
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(
        json.dumps(
            {
                "step_id": step.step_id,
                "inputs": {
                    "artifact:analysis_cohort": {
                        "relative_path": cohort_path.name,
                        "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": len(frame),
                        },
                    }
                },
                "raw_input_contracts": raw_contracts,
            }
        ),
        encoding="utf-8",
    )
    out_dir = run_dir / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_STEP_ID", step.step_id)
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=("age",),
        source_contracts_sha256=contracts_sha256,
        authority_kind="resolved_raw_input_contracts",
    )

    selection = select_standard_executor(
        step,
        plan=plan,
        plausibility_scope=scope,
    )
    assert selection is not None
    assert selection.analysis_kind == "missingness_measurement_audit"
    assert (
        flag_only_plausibility_obligation_findings(
            None,
            script_text=selection.code,
            step=step,
            scope=scope,
        )
        == []
    )

    exec(compile(selection.code, "<missingness-executor>", "exec"), {})

    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    assert summary["n_total"] == len(frame)
    assert summary["plausibility_audit"] == {
        "age": {
            "policy": "retain_and_flag",
            "below_minimum_n": 1,
            "above_maximum_n": 1,
            "out_of_range_n": 2,
        }
    }
    assert (
        plausibility_audit_receipt_findings(
            step_summary=summary,
            step=step,
            script_text=selection.code,
            scope=scope,
        )
        == []
    )


def test_compact_missingness_executor_reads_exact_bound_cohort(
    tmp_path: Path,
    monkeypatch,
):
    step = AnalysisStep(
        step_id="03_missingness_and_measurement_audit",
        intent="Audit missingness and measurement availability.",
        inputs=[
            "cohort:analysis_set",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        expected_outputs=["table:missingness_measurement_audit"],
        method="missingness_measurement_audit",
    )
    plan = AnalysisPlan(research_question="Test", steps=[step])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bound = _cohort(n=50, seed=41)
    raw = _cohort(n=100, seed=42)
    bound_path = run_dir / "bound.parquet"
    raw_path = run_dir / "raw.parquet"
    bound.to_parquet(bound_path, index=False)
    raw.to_parquet(raw_path, index=False)
    (run_dir / "research_context.json").write_text("{}", encoding="utf-8")
    (run_dir / "analysis_plan.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "cohort:analysis_set": {
                        "relative_path": bound_path.name,
                        "sha256": hashlib.sha256(bound_path.read_bytes()).hexdigest(),
                        "product_contract": {
                            "columns": list(bound.columns),
                            "row_count": len(bound),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out_dir = run_dir / "outputs"
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_STEP_ID", step.step_id)
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))
    monkeypatch.setenv("COHORT_PARQUET", str(raw_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    selection = select_standard_executor(step, plan=plan)
    assert selection is not None

    exec(compile(selection.code, "<missingness-executor>", "exec"), {})

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["cohort_input_key"] == "cohort:analysis_set"
    assert summary["n_total"] == len(bound)
    assert summary["measurement_provenance_audit"]["checks"][0]["status"] == "checked"


def test_missingness_audit_counts_from_measured_indicator(tmp_path: Path):
    cohort = _cohort(n=1000, seed=1)
    summary, out_dir = _exec_runner(tmp_path, cohort, {})
    assert summary["status"] == "ok"
    assert summary["analysis_family"] == "data_quality"
    assert summary["adjusted_effect"] is None  # a descriptive audit, never an effect

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    concepts = set(audit["concept"])
    assert {"lactate", "crea", "rrt"} <= concepts
    # id / demographic / outcome columns are NOT audited as concepts.
    assert concepts.isdisjoint({"stay_id", "age", "sex", "death", "los_icu"})

    lact = audit[audit["concept"] == "lactate"].iloc[0]
    exp_measured = int((cohort["lactate_measured"] == 1).sum())
    assert lact["measured_one_n"] == exp_measured
    assert lact["value_missing_n"] == 1000 - exp_measured
    # counts and percentages are consistent (never imputed to 0 -> full denom).
    assert lact["measured_one_n"] + lact["value_missing_n"] == 1000
    assert abs(lact["value_missing_pct"] - 100.0 * (1000 - exp_measured) / 1000) < 1e-6
    # schema aliases the figure renderer resolves are all present.
    for col in (
        "n_total",
        "measured_n",
        "n_nonmissing",
        "missing_n",
        "missing_pct",
        "measured_pct",
    ):
        assert col in audit.columns


def test_missingness_and_source_outputs_are_distinct_declared_products(tmp_path: Path):
    cohort = _cohort(n=100, seed=11)
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "artifact:analysis_cohort",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        requested_outputs=[
            "table:missingness_audit",
            "table:measurement_source_audit",
        ],
    )

    missingness = pd.read_csv(out_dir / "missingness_audit.csv")
    source = pd.read_csv(out_dir / "measurement_source_audit.csv")
    lactate_missingness = missingness.loc[missingness["concept"] == "lactate"].iloc[0]
    lactate_source = source.loc[source["concept"] == "lactate"].iloc[0]
    assert lactate_missingness["missing_n"] == int(cohort["lactate"].isna().sum())
    assert lactate_missingness["n_nonmissing"] == int(cohort["lactate"].notna().sum())
    assert lactate_source["indicator_semantics"] == "measurement_availability"
    assert summary["status"] == "ok"
    assert summary["missing_declared_inputs"] == []
    assert summary["measurement_provenance_audit"] == {
        "source": "COHORT_PARQUET",
        "checks": [
            {
                "measured_column": "lactate_measured",
                "count_column": "lactate_n",
                "status": "checked",
                "comparison_n": 100,
                "invalid_pair_n": 0,
                "discordant_n": 0,
                "role": "audit_only",
            }
        ],
    }
    assert summary["output_files"] == {
        "table:missingness_audit": "missingness_audit.csv",
        "table:measurement_source_audit": "measurement_source_audit.csv",
    }


def test_missingness_profile_and_complete_case_outputs_are_bound(tmp_path: Path):
    cohort = _cohort(n=100, seed=15)
    cohort.loc[:9, "age"] = np.nan
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["age", "death"],
        requested_outputs=[
            "table:missingness_profile",
            "table:complete_case_attrition",
        ],
    )

    profile = pd.read_csv(out_dir / "missingness_audit.csv")
    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    complete = denominators.loc[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    assert profile.loc[profile["concept"] == "age", "missing_n"].item() == 10
    assert complete["n_complete"] == 90
    assert summary["output_files"] == {
        "table:missingness_profile": "missingness_audit.csv",
        "table:complete_case_attrition": "analytic_denominators.csv",
    }


def test_measurement_availability_is_a_concrete_declared_product(tmp_path: Path):
    cohort = _cohort(n=100, seed=12)
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "artifact:analysis_cohort",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        requested_outputs=[
            "table:missingness_audit",
            "table:measurement_availability",
        ],
    )

    availability = pd.read_csv(out_dir / "measurement_availability.csv")
    lactate = availability.loc[availability["concept"] == "lactate"].iloc[0]
    assert lactate["indicator_semantics"] == "measurement_availability"
    assert summary["status"] == "ok"
    assert summary["output_files"] == {
        "table:missingness_audit": "missingness_audit.csv",
        "table:measurement_availability": "measurement_availability.csv",
    }


def test_measurement_availability_audit_is_a_concrete_declared_product(
    tmp_path: Path,
):
    cohort = _cohort(n=100, seed=14)
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "artifact:analysis_cohort",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        requested_outputs=[
            "table:missingness_audit",
            "table:measurement_availability_audit",
        ],
    )

    availability = pd.read_csv(out_dir / "measurement_availability_audit.csv")
    lactate = availability.loc[availability["concept"] == "lactate"].iloc[0]
    assert lactate["indicator_semantics"] == "measurement_availability"
    assert summary["status"] == "ok"
    assert summary["output_files"] == {
        "table:missingness_audit": "missingness_audit.csv",
        "table:measurement_availability_audit": ("measurement_availability_audit.csv"),
    }


def test_missingness_runner_uses_manifest_selected_current_plan(tmp_path: Path):
    cohort = _cohort(n=100, seed=13)
    summary, _out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "artifact:analysis_cohort",
            "lactate",
            "lactate_measured",
            "lactate_n",
        ],
        requested_outputs=[
            "table:missingness_audit",
            "table:measurement_availability",
        ],
        current_plan_name="analysis_plan_revision_2.json",
    )

    assert summary["requested_input_count"] == 3
    assert summary["output_files"] == {
        "table:missingness_audit": "missingness_audit.csv",
        "table:measurement_availability": "measurement_availability.csv",
    }


def test_missingness_runner_accepts_host_evidence_plan_path(tmp_path: Path):
    cohort = _cohort(n=100, seed=16)
    summary, _out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["age", "death"],
        requested_outputs=[
            "table:missingness_profile",
            "table:complete_case_attrition",
        ],
        current_plan_name="evidence/analysis_plan_input_closure.json",
    )

    assert summary["status"] == "ok"
    assert summary["requested_input_count"] == 2


def test_missingness_runner_rejects_plan_path_escape(tmp_path: Path):
    cohort = _cohort(n=10, seed=17)

    with pytest.raises(ValueError, match="unsafe plan_path"):
        _exec_runner(
            tmp_path,
            cohort,
            {},
            requested_inputs=["age", "death"],
            requested_outputs=[
                "table:missingness_profile",
                "table:complete_case_attrition",
            ],
            current_plan_name="../analysis_plan.json",
        )


def test_structural_no_source_distinguished_from_measurement_missing(tmp_path: Path):
    cohort = _cohort(n=800, seed=2)
    summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    # rrt is measured for NO stay -> structural no-source, not measurement-missing.
    rrt = audit[audit["concept"] == "rrt"].iloc[0]
    assert rrt["missingness_kind"] == "structural_no_source"
    assert rrt["measured_one_n"] == 0
    # lactate/crea are sourced (some measured) -> measurement missingness.
    lact = audit[audit["concept"] == "lactate"].iloc[0]
    assert lact["missingness_kind"] == "measurement_missing"
    assert summary["n_structural_no_source"] >= 1


def test_never_imputes_partial_concept(tmp_path: Path):
    # A concept measured for exactly 3/5 stays must report measured=3, missing=2
    # -- the two unmeasured stays are NOT counted as measured-zero.
    cohort = pd.DataFrame(
        {
            "stay_id": [0, 1, 2, 3, 4],
            "age": [60, 61, 62, 63, 64],
            "sex": ["Male"] * 5,
            "death": [0, 1, 0, 0, 1],
            "bili": [1.2, np.nan, 3.4, np.nan, 0.9],
            "bili_measured": [1, 0, 1, 0, 1],
        }
    )
    _summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    bili = audit[audit["concept"] == "bili"].iloc[0]
    assert bili["measured_one_n"] == 3
    assert bili["value_missing_n"] == 2
    assert bili["value_missing_pct"] == 40.0


def test_complete_binary_event_flag_is_not_misread_as_measurement_missingness(
    tmp_path: Path,
):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "sex": ["F", "M", "F", "M", "F"],
            "death": [0, 1, 0, 0, 1],
            "rrt_first": [0, 0, 1, 0, 1],
            "rrt_measured": [0, 0, 1, 0, 1],
            "rrt_n": [0, 0, 1, 0, 2],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["rrt_first", "rrt_measured"],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    rrt = audit[audit["concept"] == "rrt"].iloc[0]
    assert summary["n_binary_event_status"] == 1
    assert rrt["indicator_semantics"] == "binary_event_presence"
    assert rrt["missingness_kind"] == "binary_event_status_complete"
    assert rrt["raw_indicator_one_n"] == 2
    assert rrt["event_count_column"] == "rrt_n"
    assert rrt["measured_one_n"] == 5
    assert rrt["value_missing_n"] == 0
    assert rrt["value_present_but_measured_zero_n"] == 0


def test_typed_positive_only_event_absence_is_not_missingness(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "death": [0, 1, 0, 0, 1],
            "susp_inf_n": [0, 1, 2, 0, 1],
            "susp_inf_measured": [0, 1, 1, 0, 1],
            "susp_inf_first": [np.nan, 1.0, 1.0, np.nan, 1.0],
        }
    )
    context = {
        "variables": [
            {
                "name": "susp_inf_first",
                "observation_semantics": {
                    "kind": "positive_only_event",
                    "event_count_column": "susp_inf_n",
                    "measured_column": "susp_inf_measured",
                    "representative_column": "susp_inf_first",
                },
            }
        ]
    }

    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        context,
        requested_inputs=[
            "susp_inf_first",
            "susp_inf_measured",
            "susp_inf_n",
        ],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    row = audit.loc[audit["concept"] == "susp_inf"].iloc[0]
    assert summary["status"] == "ok"
    assert row["indicator_semantics"] == "binary_event_presence"
    assert row["missingness_kind"] == "binary_event_status_complete"
    assert row["raw_value_missing_n"] == 2
    assert row["event_present_n"] == 3
    assert row["event_absent_n"] == 2
    assert row["measured_one_n"] == len(cohort)
    assert row["value_missing_n"] == 0
    assert summary["observation_semantics_audit"]["susp_inf_first"][
        "event_absent_n"
    ] == 2


def test_typed_conditional_event_time_uses_event_positive_denominator(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "death": [0, 1, 1, 0, 1],
            "death_time": [np.nan, 12.0, np.nan, np.nan, 48.0],
        }
    )
    context = {
        "variables": [
            {
                "name": "death_time",
                "observation_semantics": {
                    "kind": "conditional_event_time",
                    "event_status_column": "death",
                    "representative_column": "death_time",
                    "time_origin": "icu_admission",
                    "time_unit": "h",
                },
            }
        ]
    }

    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        context,
        requested_inputs=["death_time"],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    row = audit.loc[audit["concept"] == "death_time"].iloc[0]
    missingness = pd.read_csv(out_dir / "missingness_audit.csv").iloc[0]
    denominator = pd.read_csv(out_dir / "analytic_denominators.csv")
    observed = denominator.loc[
        denominator["analysis_set"] == "observed:death_time"
    ].iloc[0]
    assert summary["status"] == "ok"
    assert row["indicator_semantics"] == "conditional_event_time"
    assert row["eligible_n"] == 3
    assert row["not_applicable_n"] == 2
    assert row["raw_value_missing_n"] == 3
    assert row["value_missing_n"] == 1
    assert missingness["missing_pct"] == pytest.approx(100.0 / 3.0)
    assert observed["n_complete"] == 4


def test_typed_conditional_event_time_before_origin_is_reported_for_protocol(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "death": [0, 1, 1],
            "death_time": [np.nan, -2.0, 48.0],
        }
    )
    context = {
        "variables": [
            {
                "name": "death_time",
                "observation_semantics": {
                    "kind": "conditional_event_time",
                    "event_status_column": "death",
                    "representative_column": "death_time",
                    "time_origin": "icu_admission",
                    "time_unit": "h",
                },
            }
        ]
    }

    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        context,
        requested_inputs=["death_time"],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    row = audit.loc[audit["concept"] == "death_time"].iloc[0]
    assert summary["status"] == "ok"
    assert summary["temporal_validity_audit"] == {
        "status": "flagged_requires_downstream_protocol",
        "reason_codes": ["event_time_before_declared_origin:death_time:1"],
    }
    assert summary["blocking_reason"] is None
    assert row["before_origin_n"] == 1


def test_binary_value_flag_without_event_count_remains_a_conflict(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [60, 61, 62, 63],
            "sex": ["F", "M", "F", "M"],
            "death": [0, 1, 0, 0],
            "screen_first": [0, 1, 0, 1],
            "screen_measured": [0, 1, 0, 1],
        }
    )
    _summary, out_dir = _exec_runner(tmp_path, cohort, {})
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    screen = audit[audit["concept"] == "screen"].iloc[0]
    assert screen["indicator_semantics"] == "measurement_availability"
    assert screen["missingness_kind"] == "measurement_flag_conflict"
    assert screen["measured_one_n"] == 2
    assert screen["value_present_but_measured_zero_n"] == 2


def test_family_aggregate_and_declared_analytic_denominator(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "sex": ["F", "M", "F", "M", "F"],
            "death": [0, 1, 0, 0, 1],
            "aki_stage_first": [0, 0, 1, 1, 2],
            "aki_stage_max": [0, 1, 2, np.nan, 3],
            "aki_stage_measured": [1, 1, 1, 0, 1],
            "crea_first": [0.8, 1.1, np.nan, 2.0, 0.9],
            "crea_measured": [1, 1, 0, 1, 1],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=[
            "aki_stage_max",
            "aki_stage_measured",
            "crea_first",
            "crea_measured",
            "age",
            "sex",
            "death",
        ],
    )

    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    stage = audit[audit["concept"] == "aki_stage"].iloc[0]
    crea = audit[audit["concept"] == "crea"].iloc[0]
    assert stage["value_column"] == "aki_stage_max"
    assert crea["value_column"] == "crea_first"
    assert stage["raw_value_missing_n"] == 1
    assert stage["measured_but_value_missing_n"] == 0
    assert {"age", "sex", "death"} <= set(audit["concept"])

    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    complete = denominators[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    assert complete["n_total"] == 5
    assert complete["n_complete"] == 3
    assert complete["n_excluded_missing"] == 2
    assert summary["all_requested_inputs_complete_n"] == 3
    assert summary["missing_declared_inputs"] == []


def test_bare_concept_declared_input_resolves_to_value_column_not_blocked(
    tmp_path: Path,
):
    # Regression: a plan may declare a time-series concept by its BARE name
    # (``crea``) while the cohort materialises it only as aggregates
    # (``crea_first``/``crea_measured``). The audit resolves it via
    # _representative_value_column, so the analytic-denominator loop must resolve
    # it the SAME way instead of flagging it as a missing declared input and
    # spuriously blocking an otherwise-complete audit.
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [60, 61, 62, 63, 64],
            "crea_first": [0.8, 1.1, np.nan, 2.0, 0.9],
            "crea_measured": [1, 1, 0, 1, 1],
        }
    )
    summary, out_dir = _exec_runner(
        tmp_path, cohort, {}, requested_inputs=["crea", "age"]
    )
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    assert "crea" in set(audit["concept"])  # concept was genuinely audited
    assert summary["status"] == "ok"  # ... so it must not be blocked
    assert summary["missing_declared_inputs"] == []
    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    complete = denominators[
        denominators["analysis_set"] == "all_requested_inputs"
    ].iloc[0]
    # crea_first missing on row 3 only -> 4 complete on {crea, age}
    assert complete["n_complete"] == 4


def test_genuinely_absent_declared_input_still_blocks(tmp_path: Path):
    # The bare-name resolution must NOT mask a truly-missing declared input.
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "crea_first": [0.8, np.nan, 2.0],
            "crea_measured": [1, 0, 1],
        }
    )
    summary, _ = _exec_runner(
        tmp_path, cohort, {}, requested_inputs=["crea", "nonexistent_var"]
    )
    assert summary["status"] == "blocked"
    assert summary["missing_declared_inputs"] == ["nonexistent_var"]


def test_declared_inputs_scope_audit_instead_of_scanning_unrelated_wide_columns(
    tmp_path: Path,
):
    cohort = _cohort(n=20, seed=7)
    cohort["sofa_liver"] = np.arange(20, dtype=float)
    cohort["sofa_liver_measured"] = 1
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["lactate", "lactate_measured", "age", "death"],
    )
    audit = pd.read_csv(out_dir / "missingness_measurement_audit.csv")
    assert summary["n_concepts_audited"] == 3
    assert set(audit["concept"]) == {"lactate", "age", "death"}
    assert "sofa_liver" not in set(audit["concept"])


def test_missing_declared_input_blocks_joint_denominator(tmp_path: Path):
    cohort = _cohort(n=20, seed=3)
    summary, out_dir = _exec_runner(
        tmp_path,
        cohort,
        {},
        requested_inputs=["lactate", "column_not_in_cohort"],
    )
    assert summary["status"] == "blocked"
    assert summary["all_requested_inputs_complete_n"] is None
    assert summary["missing_declared_inputs"] == ["column_not_in_cohort"]

    denominators = pd.read_csv(out_dir / "analytic_denominators.csv")
    joint = denominators[denominators["analysis_set"] == "all_requested_inputs"].iloc[0]
    assert pd.isna(joint["n_complete"])


def test_blocks_on_empty_cohort(tmp_path: Path):
    cohort = _cohort(n=0)
    summary, _out = _exec_runner(tmp_path, cohort, {})
    assert summary["status"] == "blocked"
    assert summary["adjusted_effect"] is None
    assert "empty" in summary["blocking_reason"].lower()


def test_corrupt_analysis_plan_fails_loudly_instead_of_widening_scope(
    tmp_path: Path,
) -> None:
    """A present-but-unparseable host plan must fail the step, not silently
    fall back to auditing every paired concept (scope drift)."""
    import pytest

    out_dir = tmp_path / "steps" / "02_missingness_measurement_audit" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "research_context.json").write_text("{}", encoding="utf-8")
    (tmp_path / "analysis_plan.json").write_text("{not json", encoding="utf-8")
    cohort_path = tmp_path / "cohort_analysis.parquet"
    _cohort(n=10).to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = missingness_measurement_audit_code()
        with pytest.raises(json.JSONDecodeError):
            exec(compile(code, "<det_missingness>", "exec"), {"__name__": "__main__"})
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert not (out_dir / "step_summary.json").exists()


def test_absent_plan_and_context_keep_legacy_discovery_mode(tmp_path: Path) -> None:
    """Missing host files stay a legitimate legacy state (broad discovery)."""
    out_dir = tmp_path / "steps" / "02_missingness_measurement_audit" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = tmp_path / "cohort_analysis.parquet"
    _cohort(n=50).to_parquet(cohort_path, index=False)

    saved = dict(os.environ)
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    os.environ["COHORT_PARQUET"] = str(cohort_path)
    try:
        code = missingness_measurement_audit_code()
        try:
            exec(compile(code, "<det_missingness>", "exec"), {"__name__": "__main__"})
        except SystemExit:
            pass
    finally:
        os.environ.clear()
        os.environ.update(saved)
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["n_concepts_audited"] >= 3
