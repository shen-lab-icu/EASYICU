"""Default statsmodels confidence intervals must be labeled as Wald."""

from __future__ import annotations

import ast

from easyicu.research_agent.gates.interval_method import (
    confidence_interval_method_findings,
)
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repair_registry import RepairClass, repair_metadata_for
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def _step(ra):
    return ra.AnalysisStep(
        step_id="primary_model",
        intent="Estimate the declared adjusted association.",
        inputs=["table:analysis_cohort"],
        expected_outputs=["table:model_summary"],
        method="multivariable_logistic_regression",
    )


def _script(*, interval_method: str = "profile_normal") -> str:
    return f'''
import statsmodels.api as sm

model = sm.Logit(y, X)
result = model.fit(disp=False)
ci = result.conf_int()
interval_method = "{interval_method}"
model_method = "statsmodels_Logit_{interval_method}"
'''


def _findings(script: str, ra):
    return [
        finding
        for finding in audit_mechanical_code_contracts(script, _step(ra))
        if (finding.detail or {}).get("reason")
        == "confidence_interval_method_mislabeled"
    ]


def test_default_statsmodels_conf_int_profile_label_fails_preflight(ra):
    findings = _findings(_script(), ra)

    assert len(findings) == 1
    assert findings[0].detail["occurrence_count"] == 2
    assert repair_reason_for_finding(findings[0]) is (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    )


def test_interval_label_repair_is_coordinate_bound_and_idempotent(ra):
    script = _script()
    findings = _findings(script, ra)

    repaired, names = deterministic_concept_audit_repair(
        script,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )
    repaired_again, second_names = deterministic_concept_audit_repair(
        repaired,
        [finding.message for finding in findings],
        repair_reasons=[repair_reason_for_finding(finding) for finding in findings],
        repair_findings=findings,
    )

    assert names == ["statsmodels_interval_method_label_v1"]
    assert 'interval_method = "profile_normal"' not in repaired
    assert "'wald_95_percent'" in repaired
    assert "'statsmodels_Logit_wald_95_percent'" in repaired
    assert _findings(repaired, ra) == []
    assert repaired_again == repaired
    assert second_names == []
    ast.parse(repaired)


def test_already_wald_or_unbound_profile_text_is_not_flagged(ra):
    assert _findings(_script(interval_method="wald_95_percent"), ra) == []
    unrelated = '''
label = "profile_normal"
result = custom_profile_likelihood_fit(data)
ci = result.conf_int()
'''
    assert confidence_interval_method_findings(ast.parse(unrelated)) == []


def test_interval_label_repair_registry_is_syntactic():
    metadata = repair_metadata_for("statsmodels_interval_method_label_v1")

    assert metadata.repair_class is RepairClass.SYNTACTIC
    assert metadata.introduces_numbers is False
