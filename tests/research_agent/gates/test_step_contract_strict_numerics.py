from __future__ import annotations

import pytest

from easyicu.research_agent.gates.step_contract import _step_contract_findings
from easyicu.research_agent.schema import AnalysisStep


@pytest.mark.parametrize("bad_value", [True, "0.81", float("nan"), float("inf")])
def test_prediction_contract_rejects_non_numeric_or_nonfinite_metrics(bad_value) -> None:
    step = AnalysisStep(
        step_id="03_model_training",
        method="prediction_model_evaluation",
        intent="Evaluate held-out prediction performance and calibration.",
        expected_outputs=["statistic:auroc", "statistic:brier_score"],
    )

    findings = _step_contract_findings(
        step=step,
        step_summary={"auroc": bad_value, "brier_score": bad_value},
    )

    messages = " ".join(finding.message for finding in findings)
    assert "AUROC" in messages
    assert "calibration" in messages.lower() or "Brier" in messages
