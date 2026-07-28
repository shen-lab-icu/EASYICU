"""The enriched three-product measurement-bias audit contract.

A replanner that adds the observation-process and component-completeness
products is asking for more science, not different science. Before this
contract existed, that enrichment silently cost the step its deterministic
owner and dropped it onto the stochastic Coder — where the E1 canary's
Step 04 died, taking with it the only audit that guards the headline result
against differential ascertainment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.deterministic_missingness import (
    is_compact_missingness_measurement_contract,
    is_measurement_bias_audit_contract,
    missingness_audit_executor_owns_step,
    missingness_measurement_audit_code,
)
from easyicu.research_agent.execution.runners.selection import (
    select_standard_executor,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

_ENRICHED_OUTPUTS = [
    "table:missingness_measurement_audit",
    "table:measurement_process_audit",
    "table:exposure_component_completeness_audit",
]


def _step(**overrides: Any) -> AnalysisStep:
    payload: Dict[str, Any] = {
        "step_id": "04_missingness_and_event_timing_audit",
        "intent": "Audit component availability, measurement frequency and timing.",
        "method": "measurement_bias_audit",
        "planned_analysis_role": "auxiliary",
        "inputs": [
            "artifact:analysis_cohort",
            "sep3_sofa2_max",
            "sep3_sofa2_n",
            "sep3_sofa2_measured",
            "sofa2_liver_max",
            "sofa2_liver_n",
            "sofa2_liver_measured",
        ],
        "expected_outputs": list(_ENRICHED_OUTPUTS),
    }
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def test_the_enriched_contract_is_owned() -> None:
    step = _step()

    assert is_measurement_bias_audit_contract(step.method, step.expected_outputs)
    assert missingness_audit_executor_owns_step(step)


def test_the_selector_reports_the_enriched_kind_not_the_compact_one() -> None:
    plan = AnalysisPlan(research_question="q", robustness_specs=[], steps=[])

    selection = select_standard_executor(_step(), plan=plan)

    assert selection is not None
    assert selection.analysis_kind == "measurement_bias_audit"
    assert selection.selection_reason == "measurement_bias_contract_preflight"


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("a fourth product", {"expected_outputs": _ENRICHED_OUTPUTS + ["table:x"]}),
        ("only two products", {"expected_outputs": _ENRICHED_OUTPUTS[:2]}),
        (
            "a duplicated product",
            {"expected_outputs": [_ENRICHED_OUTPUTS[0]] * 3},
        ),
        (
            "a substituted product",
            {
                "expected_outputs": [
                    _ENRICHED_OUTPUTS[0],
                    _ENRICHED_OUTPUTS[1],
                    "table:adjusted_association_estimates",
                ]
            },
        ),
        (
            "a non-table product",
            {
                "expected_outputs": [
                    _ENRICHED_OUTPUTS[0],
                    _ENRICHED_OUTPUTS[1],
                    "figure:exposure_component_completeness_audit",
                ]
            },
        ),
        ("a modelling method", {"method": "adjusted_association_models"}),
        ("an estimation method", {"method": "measurement_bias_audit_and_estimation"}),
        ("an empty method", {"method": ""}),
        (
            "a second typed input",
            {
                "inputs": [
                    "artifact:analysis_cohort",
                    "table:adjusted_association_estimates",
                ]
            },
        ),
    ],
)
def test_the_contract_fails_closed(label: str, overrides: Dict[str, Any]) -> None:
    """This count-only runner must never swallow a model, test or figure."""

    plan = AnalysisPlan(research_question="q", robustness_specs=[], steps=[])

    assert select_standard_executor(_step(**overrides), plan=plan) is None or (
        select_standard_executor(_step(**overrides), plan=plan).analysis_kind
        != "measurement_bias_audit"
    )


def test_the_compact_contract_still_means_one_product() -> None:
    """Widening must not have weakened the narrower contract."""

    assert not is_compact_missingness_measurement_contract(
        "measurement_bias_audit", _ENRICHED_OUTPUTS
    )
    assert is_compact_missingness_measurement_contract(
        "missingness_measurement_audit", ["table:missingness_measurement_audit"]
    )


def _run_audit(tmp_path: Path, step: AnalysisStep, frame: pd.DataFrame,
               context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the generated runner exactly as the sandbox would."""

    run_dir = tmp_path / "run"
    (run_dir / "evidence").mkdir(parents=True)
    (run_dir / "resolved_inputs").mkdir()
    cohort_rel = "evidence/table_cohort__analysis_cohort.parquet"
    frame.to_parquet(run_dir / cohort_rel, index=False)
    (run_dir / "research_context.json").write_text(json.dumps(context))
    plan_rel = "evidence/plan__analysis_plan.json"
    (run_dir / plan_rel).write_text(
        json.dumps({"revision": 1, "steps": [step.model_dump(mode="json")]})
    )
    (run_dir / "manifest_partial.json").write_text(json.dumps({"plan_path": plan_rel}))
    import hashlib

    digest = hashlib.sha256((run_dir / cohort_rel).read_bytes()).hexdigest()
    (run_dir / f"resolved_inputs/{step.step_id}.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "artifact:analysis_cohort": {
                        "relative_path": cohort_rel,
                        "absolute_path": str(run_dir / cohort_rel),
                        "sha256": digest,
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": int(len(frame)),
                        },
                    }
                }
            }
        )
    )
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    out_dir.mkdir(parents=True)
    script = tmp_path / "analysis.py"
    script.write_text(missingness_measurement_audit_code(step))
    result = subprocess.run(
        [sys.executable, str(script)],
        env={
            **os.environ,
            "STEP_OUT_DIR": str(out_dir),
            "EASYICU_RUN_DIR": str(run_dir),
            "EASYICU_STEP_ID": step.step_id,
            "EASYICU_RESOLVED_INPUTS_JSON": str(
                run_dir / f"resolved_inputs/{step.step_id}.json"
            ),
            "COHORT_PARQUET": str(run_dir / cohort_rel),
        },
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return {
        "summary": json.loads((out_dir / "step_summary.json").read_text()),
        "out_dir": out_dir,
    }


def _frame() -> pd.DataFrame:
    # Two components with deliberately DIFFERENT availability between the two
    # exposure strata: exactly the shape that inflates an apparent association.
    return pd.DataFrame(
        {
            "stay_id": list(range(1, 11)),
            "sep3_sofa2_max": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "sep3_sofa2_measured": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "sep3_sofa2_n": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "sofa2_liver_max": [1.0, None, None, None, None, 2.0, 3.0, 1.0, None, 2.0],
            "sofa2_liver_measured": [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            "sofa2_liver_n": [2, 0, 0, 0, 0, 3, 1, 4, 0, 2],
        }
    )


_CONTEXT = {
    "primary_exposure": "sep3_sofa2_max",
    "target_outcome": "death",
    "variables": [],
    "user_preferences": {},
}


def test_all_three_declared_products_are_written(tmp_path: Path) -> None:
    step = _step()

    result = _run_audit(tmp_path, step, _frame(), _CONTEXT)

    assert result["summary"]["status"] == "ok"
    files = result["summary"]["output_files"]
    assert set(files) == set(_ENRICHED_OUTPUTS)
    # Distinct files: two declared products resolving to one table would
    # satisfy the contract without satisfying a reader.
    assert len(set(files.values())) == 3
    for relative in files.values():
        assert (result["out_dir"] / relative).is_file()


def test_completeness_is_stratified_by_the_declared_exposure(
    tmp_path: Path,
) -> None:
    result = _run_audit(tmp_path, _step(), _frame(), _CONTEXT)

    block = result["summary"]["exposure_component_completeness"]
    assert block["stratified"] is True
    assert block["exposure_variable"] == "sep3_sofa2_max"
    table = pd.read_csv(
        result["out_dir"] / "exposure_component_completeness_audit.csv"
    )
    liver = table[table["concept"] == "sofa2_liver"].set_index("exposure_category")
    assert int(liver.loc["0", "n_stratum"]) == 5
    assert int(liver.loc["1", "n_stratum"]) == 5
    # 1/5 vs 4/5: the differential a reader must see.
    assert float(liver.loc["0", "measured_pct"]) == pytest.approx(20.0)
    assert float(liver.loc["1", "measured_pct"]) == pytest.approx(80.0)
    assert float(liver.loc["__all__", "measured_pct"]) == pytest.approx(50.0)


def test_an_undeclared_exposure_degrades_instead_of_failing(
    tmp_path: Path,
) -> None:
    """No declared exposure is a legitimate state, not a dead step."""

    result = _run_audit(
        tmp_path, _step(), _frame(), {**_CONTEXT, "primary_exposure": ""}
    )

    assert result["summary"]["status"] == "ok"
    block = result["summary"]["exposure_component_completeness"]
    assert block["stratified"] is False
    assert "no primary exposure declared" in block["note"]
    table = pd.read_csv(
        result["out_dir"] / "exposure_component_completeness_audit.csv"
    )
    assert set(table["exposure_category"]) == {"__all__"}


def test_an_exposure_that_is_not_a_cohort_column_degrades(tmp_path: Path) -> None:
    result = _run_audit(
        tmp_path, _step(), _frame(), {**_CONTEXT, "primary_exposure": "not_a_column"}
    )

    block = result["summary"]["exposure_component_completeness"]
    assert block["stratified"] is False
    assert "is not a cohort column" in block["note"]


def test_the_raw_indicator_is_reported_beside_semantic_completeness(
    tmp_path: Path,
) -> None:
    """Semantic completeness alone would hide a differential event process."""

    # Under the declared ICU rule an absent event row is a COMPLETE negative
    # observation, so an event concept reads 100 % complete in every stratum.
    # That is where a differential observation process hides, so the raw
    # indicator rate has to travel beside it.
    result = _run_audit(tmp_path, _step(), _frame(), _CONTEXT)

    table = pd.read_csv(
        result["out_dir"] / "exposure_component_completeness_audit.csv"
    )
    event = table[
        (table["concept"] == "sep3_sofa2")
        & (table["exposure_category"] != "__all__")
    ].set_index("exposure_category")
    assert float(event.loc["0", "measured_pct"]) == pytest.approx(100.0)
    assert float(event.loc["1", "measured_pct"]) == pytest.approx(100.0)
    assert float(event.loc["0", "raw_indicator_one_pct"]) == pytest.approx(0.0)
    assert float(event.loc["1", "raw_indicator_one_pct"]) == pytest.approx(100.0)


def test_a_concept_without_an_indicator_reports_empty_not_zero(
    tmp_path: Path,
) -> None:
    """A 0 there reads as 'never measured' when it means 'not applicable'."""

    frame = _frame()
    frame["age"] = [40.0, 50, 60, 70, 80, 45, 55, 65, 75, 85]
    step = _step(inputs=list(_step().inputs) + ["age"])

    result = _run_audit(tmp_path, step, frame, _CONTEXT)

    table = pd.read_csv(
        result["out_dir"] / "exposure_component_completeness_audit.csv"
    )
    age = table[table["concept"] == "age"]
    assert not age.empty
    assert bool(age["has_measured_indicator"].eq(False).all())
    # Both columns, not just the percentage: a count of 0 misleads on its own.
    assert bool(age["raw_indicator_one_n"].isna().all())
    assert bool(age["raw_indicator_one_pct"].isna().all())


def test_the_process_audit_carries_the_observation_process(
    tmp_path: Path,
) -> None:
    result = _run_audit(tmp_path, _step(), _frame(), _CONTEXT)

    process = pd.read_csv(result["out_dir"] / "measurement_process_audit.csv")
    liver = process[process["concept"] == "sofa2_liver"].iloc[0]
    assert liver["measurement_count_column"] == "sofa2_liver_n"
    assert int(liver["measurement_total_n"]) == 12
    assert int(liver["measurement_count_max"]) == 4
    assert int(liver["repeat_measured_n"]) == 4  # counts 2, 3, 4 and 2
    for column in ("eligible_n", "not_applicable_n", "before_origin_n"):
        assert column in process.columns


def test_missingness_and_process_products_are_not_the_same_table(
    tmp_path: Path,
) -> None:
    result = _run_audit(tmp_path, _step(), _frame(), _CONTEXT)

    missingness = pd.read_csv(
        result["out_dir"] / "missingness_measurement_audit.csv"
    )
    process = pd.read_csv(result["out_dir"] / "measurement_process_audit.csv")
    assert "measurement_count_max" in process.columns
    assert "measurement_count_max" not in missingness.columns
    exclusive: List[str] = [
        column for column in process.columns if column not in missingness.columns
    ]
    assert exclusive, "the process audit must answer a question of its own"
