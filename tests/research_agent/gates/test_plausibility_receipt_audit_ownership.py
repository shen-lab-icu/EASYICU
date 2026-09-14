"""Host plausibility-receipt and percentage finding ownership tests."""

import json

import pandas as pd
import pytest

from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient


def _offline_concept_auditor(ra, fixed_client):  # noqa: ANN001
    try:
        response = fixed_client.complete([], max_tokens=1024, temperature=0.0)
    except Exception as exc:
        response = exc
    return ra.LLMConceptAuditor(ScriptedMockLLMClient([response]))


def _flag_only_receipt_script(*, body_prefix: str = "") -> str:
    from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
    from easyicu.research_agent.execution.runners.plausibility_receipt import (
        host_plausibility_receipt_injected,
    )

    body = body_prefix + (
        "import pandas as pd\n\n"
        "frame = pd.read_parquet('/cohort.parquet')\n"
        "numeric = numeric_series_fail_closed(frame['age'], 'age')\n"
        "print(len(numeric))\n"
    )
    return host_plausibility_receipt_injected(
        body,
        scope=FlagOnlyPlausibilityScope(
            step_id="repeated_stay_dependence_audit",
            expected_columns=("age",),
            source_contracts_sha256="a" * 64,
            authority_kind="resolved_raw_input_contracts",
        ),
        already_satisfied=False,
    )


def _strict_nonfinite_finding(variable: str):
    class _FindingLLM:
        def complete(self, messages, *, max_tokens=1024, temperature=0.0):
            return json.dumps(
                {
                    "findings": [
                        {
                            "severity": "error",
                            "message": (
                                "The appended plausibility audit converts invalid or "
                                "non-finite values with errors='coerce' and treats "
                                "them as unavailable."
                            ),
                            "detail": {
                                "issue_code": "strict_numeric_nonfinite_guard_required",
                                "variables": [variable],
                            },
                        }
                    ]
                }
            )

    return _FindingLLM()


def _audit_flag_only_receipt_script(ra, llm, *, script=None):
    return _offline_concept_auditor(ra, llm).audit(
        context=ra.build_research_context(
            research_question="Association of peak lactate with death.",
            cohort=pd.DataFrame({"stay_id": [1, 2], "age": [64.0, 71.0]}),
            cohort_name="c",
            database="synthetic",
        ),
        script_text=_flag_only_receipt_script() if script is None else script,
        step=None,
    )


def test_llm_concept_auditor_does_not_charge_the_coder_for_the_host_receipt(ra):
    findings = _audit_flag_only_receipt_script(
        ra, _strict_nonfinite_finding("_easyicu_plausibility_numeric_v1")
    )
    assert findings[0].severity == "warning"
    assert findings[0].detail["host_owned_source_region"] == (
        "flag_only_plausibility_receipt"
    )
    assert "host appended" in findings[0].detail["downgraded_reason"]


def test_llm_concept_auditor_still_charges_the_coder_for_its_own_coercion(ra):
    findings = _audit_flag_only_receipt_script(ra, _strict_nonfinite_finding("numeric"))
    assert findings[0].severity == "error"
    assert "host_owned_source_region" not in findings[0].detail


@pytest.mark.parametrize(
    "body_prefix",
    ['quality_fields = ["coercion_loss_n"]\n', "plausibility_expected_columns = ('age',)\n"],
)
def test_receipt_fields_in_agent_body_do_not_expand_host_exemption(ra, body_prefix):
    script = _flag_only_receipt_script(body_prefix=body_prefix).replace(
        "numeric_series_fail_closed(frame['age'], 'age')",
        "pd.to_numeric(frame['age'], errors='coerce')",
    )
    findings = _audit_flag_only_receipt_script(
        ra, _strict_nonfinite_finding("numeric"), script=script
    )
    assert findings[0].severity == "error"
    assert "host_owned_source_region" not in findings[0].detail


@pytest.mark.parametrize(
    "mutation",
    [
        lambda script: script.replace(
            'plausibility_frame[column], errors="coerce"',
            'plausibility_frame[column] * 100, errors="coerce"',
        ),
        lambda script: script + "model.fit(_easyicu_plausibility_numeric_v1)\n",
        lambda script: "for repeated in range(2):\n"
        + "\n".join("    " + line for line in script.splitlines()),
    ],
)
def test_copied_or_modified_receipt_cannot_exempt_analysis_code(ra, mutation):
    script = mutation(_flag_only_receipt_script())
    findings = _audit_flag_only_receipt_script(
        ra,
        _strict_nonfinite_finding("_easyicu_plausibility_numeric_v1"),
        script=script,
    )
    assert findings[0].severity == "error"
    assert "host_owned_source_region" not in findings[0].detail


@pytest.mark.parametrize(
    "body_prefix",
    [
        "def analyze(_easyicu_plausibility_numeric_v1):\n"
        "    return pd.to_numeric(_easyicu_plausibility_numeric_v1, errors='coerce')\n",
        "def analyze(_easyicu_plausibility_numeric_v1, /):\n    pass\n",
        "def analyze(*, _easyicu_plausibility_numeric_v1):\n    pass\n",
        "def analyze(*_easyicu_plausibility_numeric_v1):\n    pass\n",
        "def analyze(**_easyicu_plausibility_numeric_v1):\n    pass\n",
        "analyze = lambda _easyicu_plausibility_numeric_v1: "
        "pd.to_numeric(_easyicu_plausibility_numeric_v1, errors='coerce')\n",
        "match frame:\n    case {'metric': _easyicu_plausibility_numeric_v1}:\n"
        "        print(pd.to_numeric(_easyicu_plausibility_numeric_v1, errors='coerce'))\n",
        "match frame:\n    case [*_easyicu_plausibility_numeric_v1]:\n        pass\n",
        "match frame:\n    case {**_easyicu_plausibility_numeric_v1}:\n        pass\n",
        "import _easyicu_plausibility_numeric_v1.analysis\n",
    ],
)
def test_agent_lexical_bindings_cannot_be_claimed_by_host_receipt(ra, body_prefix):
    findings = _audit_flag_only_receipt_script(
        ra,
        _strict_nonfinite_finding("_easyicu_plausibility_numeric_v1"),
        script=_flag_only_receipt_script(body_prefix=body_prefix),
    )
    assert findings[0].severity == "error"
    assert "host_owned_source_region" not in findings[0].detail


def test_host_plausibility_receipt_region_matches_the_complete_generated_tail():
    from easyicu.research_agent.audits.validators import (
        _host_plausibility_receipt_region,
    )

    source = _flag_only_receipt_script(body_prefix='fields = ["coercion_loss_n"]\n')
    first, last = _host_plausibility_receipt_region(source)
    assert source.splitlines()[first - 1] == "import json"
    assert first > source[: source.index("print(len(numeric))")].count("\n") + 1
    assert last == len(source.splitlines())


def test_llm_concept_auditor_accepts_the_registered_percentage_issue_code(ra):
    from easyicu.research_agent.audits.validators import (
        parse_llm_concept_audit_response,
    )

    raw = (
        '{"findings":[{"severity":"error","message":"Rendered percentages are '
        'not reconciled to their count numerators and denominator.",'
        '"detail":{"issue_code":'
        '"registered_percentage_count_reconciliation_required",'
        '"step_id":"display_package","variables":["measurement_missingness"]}}]}'
    )
    findings = parse_llm_concept_audit_response(raw, step_id="display_package")
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue_code"] == (
        "registered_percentage_count_reconciliation_required"
    )


def test_registered_percentage_issue_code_routes_to_scientific_semantics():
    from easyicu.research_agent.repairs.reasons import (
        RepairReason,
        repair_reason_for_finding,
    )
    from easyicu.research_agent.schema import ValidationFinding

    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="Rendered percentages are not reconciled to counts.",
        detail={
            "issue_code": "registered_percentage_count_reconciliation_required",
            "step_id": "display_package",
        },
    )
    assert repair_reason_for_finding(finding) == (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    )

    deterministic = ValidationFinding(
        validator="analysis_pattern_auditor",
        severity="error",
        message="Rendered percentages are not reconciled to counts.",
        detail={
            "kind": "registered_percentage_count_reconciliation_required",
            "step_id": "display_package",
        },
    )
    assert repair_reason_for_finding(deterministic) == (
        RepairReason.SCIENTIFIC_SEMANTICS_VIOLATION
    )
