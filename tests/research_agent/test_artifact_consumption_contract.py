from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.typed_binding import (
    _attach_verified_consumption_contract,
    _typed_parent_schema_context_block,
)
from easyicu.research_agent.contracts.artifact_consumption import (
    ArtifactConsumptionError,
    verify_artifact_consumption,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ArtifactConsumptionContract,
)
from easyicu.research_agent.plan_utils import _split_table_and_figure_outputs_in_plan
from easyicu.research_agent.agents.core import _normalise_plan_payload
from easyicu.research_agent.authority.plan_scope import _step_scientific_signature


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path, *, input_key: str = "table:trend_results") -> dict:
    frame = pd.read_csv(path)
    digest = _sha(path)
    return {
        "absolute_path": str(path),
        "sha256": digest,
        "identity_row": {"input_key": input_key},
        "product_contract": {
            "schema_version": "easyicu.host_typed_product.v4",
            "tabular_format": "csv",
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
    }


def test_all_rows_receipt_preserves_verified_multirow_cardinality(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trends.csv"
    pd.DataFrame(
        {
            "outcome": ["binary_endpoint", "continuous_endpoint"],
            "statistic": [2.1, 3.2],
        }
    ).to_csv(path, index=False)
    contract = ArtifactConsumptionContract(
        input_key="table:trend_results",
        mode="all_rows",
    )

    receipt = verify_artifact_consumption(
        contract=contract,
        binding=_binding(path),
    )

    assert receipt["mode"] == "all_rows"
    assert receipt["verified_row_count"] == 2
    assert receipt["artifact_sha256"] == _sha(path)


def test_single_row_contract_rejects_multirow_input(tmp_path: Path) -> None:
    path = tmp_path / "trends.csv"
    pd.DataFrame({"role": ["a", "b"], "value": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ArtifactConsumptionError, match="exactly one row"):
        verify_artifact_consumption(
            contract=ArtifactConsumptionContract(
                input_key="table:trend_results",
                mode="single_row",
            ),
            binding=_binding(path),
        )


def test_one_per_role_requires_exact_complete_role_roster(tmp_path: Path) -> None:
    path = tmp_path / "trends.csv"
    pd.DataFrame(
        {
            "outcome": ["binary_endpoint", "continuous_endpoint"],
            "statistic": [2.1, 3.2],
        }
    ).to_csv(path, index=False)
    contract = ArtifactConsumptionContract(
        input_key="table:trend_results",
        mode="one_per_role",
        role_column="outcome",
        expected_roles=["binary_endpoint", "continuous_endpoint"],
    )

    receipt = verify_artifact_consumption(
        contract=contract,
        binding=_binding(path),
    )

    assert receipt["verified_role_counts"] == {
        "binary_endpoint": 1,
        "continuous_endpoint": 1,
    }

    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {"outcome": ["binary_endpoint", "binary_endpoint"], "statistic": [1, 2]}
    ).to_csv(duplicate, index=False)
    with pytest.raises(ArtifactConsumptionError, match="exactly one row"):
        verify_artifact_consumption(
            contract=contract,
            binding=_binding(duplicate),
        )


def test_contract_must_target_an_exact_same_step_input() -> None:
    with pytest.raises(ValidationError, match="must target exact inputs"):
        AnalysisStep(
            step_id="figure",
            intent="Render the declared result.",
            inputs=["table:other"],
            expected_outputs=["figure:result"],
            input_consumption_contracts=[
                ArtifactConsumptionContract(
                    input_key="table:trend_results",
                    mode="all_rows",
                )
            ],
        )


def test_verified_consumption_receipt_reaches_coder_context(tmp_path: Path) -> None:
    path = tmp_path / "trends.csv"
    pd.DataFrame(
        {"outcome": ["binary_endpoint", "continuous_endpoint"], "value": [1, 2]}
    ).to_csv(path, index=False)
    step = AnalysisStep(
        step_id="figure",
        intent="Render all declared results.",
        inputs=["table:trend_results"],
        expected_outputs=["figure:result"],
        method="visualization",
        input_consumption_contracts=[
            ArtifactConsumptionContract(
                input_key="table:trend_results",
                mode="all_rows",
            )
        ],
    )
    binding = _attach_verified_consumption_contract(
        step=step,
        input_name="table:trend_results",
        binding=_binding(path),
    )

    block = _typed_parent_schema_context_block({"table:trend_results": binding})

    assert '"row_count":2' in block
    assert '"mode":"all_rows"' in block
    assert "all_rows means preserve every row" in block


def test_existing_multi_table_visualization_gets_all_rows_contracts() -> None:
    plan = AnalysisPlan(
        research_question="Render two exact upstream result tables.",
        steps=[
            AnalysisStep(
                step_id="figure",
                intent="Render the declared tables without selecting rows.",
                inputs=["table:stratified_results", "table:trend_results"],
                expected_outputs=["figure:summary"],
                method="visualization",
            )
        ],
    )

    revised, findings = _split_table_and_figure_outputs_in_plan(plan)

    assert [
        (contract.input_key, contract.mode)
        for contract in revised.steps[0].input_consumption_contracts
    ] == [
        ("table:stratified_results", "all_rows"),
        ("table:trend_results", "all_rows"),
    ]
    assert any(
        finding.detail.get("reason") == "visualization_all_rows_consumption_default"
        for finding in findings
    )


def test_split_figure_contracts_cover_tables_without_claiming_statistic_inputs() -> None:
    plan = AnalysisPlan(
        research_question="Fit a model and render its result table.",
        steps=[
            AnalysisStep(
                step_id="model",
                intent="Fit the model and render its declared figure.",
                inputs=["exposure", "outcome"],
                expected_outputs=[
                    "table:model_results",
                    "statistic:primary_effect",
                    "figure:effect_summary",
                ],
                method="logistic_regression",
            )
        ],
    )

    revised, _ = _split_table_and_figure_outputs_in_plan(plan)

    figure_step = revised.steps[1]
    assert figure_step.inputs == ["table:model_results", "statistic:primary_effect"]
    assert [
        contract.input_key for contract in figure_step.input_consumption_contracts
    ] == ["table:model_results"]


def test_planner_normalizer_preserves_only_closed_consumption_fields() -> None:
    payload, dropped = _normalise_plan_payload(
        {
            "research_question": "Render a verified table.",
            "steps": [
                {
                    "step_id": "figure",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Render all rows.",
                    "inputs": ["table:result"],
                    "expected_outputs": ["figure:result"],
                    "method": "visualization",
                    "input_consumption_contracts": [
                        {
                            "schema_version": "easyicu.artifact_consumption/1",
                            "input_key": "table:result",
                            "mode": "all_rows",
                            "invented_selector": "first",
                        }
                    ],
                }
            ],
        }
    )

    contract = payload["steps"][0]["input_consumption_contracts"][0]
    assert contract == {
        "schema_version": "easyicu.artifact_consumption/1",
        "input_key": "table:result",
        "mode": "all_rows",
    }
    assert dropped["input_consumption_contracts"] == ["table:result:invented_selector"]


def test_consumption_contract_change_is_scientific_scope_change() -> None:
    base = AnalysisStep(
        step_id="figure",
        intent="Render the result.",
        inputs=["table:result"],
        expected_outputs=["figure:result"],
        method="visualization",
        input_consumption_contracts=[
            ArtifactConsumptionContract(input_key="table:result", mode="all_rows")
        ],
    )
    changed = base.model_copy(
        update={
            "input_consumption_contracts": [
                ArtifactConsumptionContract(
                    input_key="table:result",
                    mode="single_row",
                )
            ]
        }
    )

    assert _step_scientific_signature(base) != _step_scientific_signature(changed)
