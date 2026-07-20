from __future__ import annotations

import ast

from easyicu.research_agent.gates.typed_schema import (
    host_schema_numeric_alias_findings,
)
from easyicu.research_agent.gates.concept import deterministic_code_gate_findings
from easyicu.research_agent.repairs.reasons import (
    RepairReason,
    repair_reason_for_finding,
)
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.repairs.typed_schema import patch_host_schema_numeric_alias
from easyicu.research_agent.schema import AnalysisStep, ResearchContext


def _bindings() -> dict[str, object]:
    return {
        "table:result": {
            "product_contract": {
                "schema_version": "easyicu.host_typed_product.v3",
                "columns": ["group", "event_n", "population_n", "population_note"],
                "column_dtypes": {
                    "group": "object",
                    "event_n": "int64",
                    "population_n": "int64",
                    "population_note": "object",
                },
                "numeric_columns": ["event_n", "population_n"],
            }
        }
    }


def _code(*, reconciliation: str | None = None) -> str:
    check = reconciliation or """
    if not np.allclose(
        numeric["population_note"].to_numpy(dtype=float),
        numeric["population_n"].to_numpy(dtype=float),
    ):
        raise ValueError("denominator mismatch")
"""
    return f"""import numpy as np

def validate(frame):
    numeric_fields = [
        "event_n",
        "population_n",
        "population_note",
    ]
    numeric = {{}}
    for field in numeric_fields:
        numeric[field] = strict_numeric(frame[field], field)
{check}
    return numeric
"""


def test_host_schema_flags_only_candidate_authored_unique_numeric_alias() -> None:
    findings = host_schema_numeric_alias_findings(ast.parse(_code()), _bindings())

    assert len(findings) == 1
    assert findings[0].detail == {
        "reason": "host_schema_nonnumeric_numeric_alias",
        "input_key": "table:result",
        "sequence_name": "numeric_fields",
        "occurrences": [
            {
                "line": 7,
                "column": 8,
                "end_line": 7,
                "end_column": 25,
                "mapping_name": "numeric",
                "source_column": "population_note",
                "numeric_alias_column": "population_n",
            }
        ],
    }
    assert repair_reason_for_finding(findings[0]) is (
        RepairReason.TYPED_PRODUCT_BINDING_INVALID
    )


def test_host_schema_numeric_alias_repair_is_exact_and_convergent() -> None:
    code = _code()
    findings = host_schema_numeric_alias_findings(ast.parse(code), _bindings())

    repaired = patch_host_schema_numeric_alias(code, repair_findings=findings)

    assert repaired != code
    assert "numeric['population_note'] = numeric['population_n']" in repaired
    assert host_schema_numeric_alias_findings(ast.parse(repaired), _bindings()) == []
    assert (
        patch_host_schema_numeric_alias(repaired, repair_findings=findings) == repaired
    )
    ast.parse(repaired)


def test_host_schema_numeric_alias_routes_through_central_repair() -> None:
    code = _code()
    findings = host_schema_numeric_alias_findings(ast.parse(code), _bindings())

    repaired, names = deterministic_concept_audit_repair(
        code,
        [findings[0].message],
        repair_reasons=[RepairReason.TYPED_PRODUCT_BINDING_INVALID],
        repair_findings=findings,
    )

    assert names == ["host_schema_numeric_alias_v1"]
    assert repaired != code


def test_shared_deterministic_gate_receives_resolved_host_schema() -> None:
    context = ResearchContext(
        research_question="Report a generic grouped event rate.",
        cohort={
            "cohort_name": "synthetic",
            "database": "synthetic",
            "n_patients": 4,
            "n_stays": 4,
        },
        variables=[],
    )
    step = AnalysisStep(
        step_id="render_result",
        intent="Render an already-computed grouped result.",
        inputs=["table:result"],
        expected_outputs=["figure:result"],
        method="publication_figure_generation",
    )

    findings = deterministic_code_gate_findings(
        context=context,
        step=step,
        script_text=_code(),
        resolved_input_bindings=_bindings(),
    )

    assert any(
        (finding.detail or {}).get("reason") == "host_schema_nonnumeric_numeric_alias"
        for finding in findings
    )


def test_host_schema_declines_without_candidate_authored_reconciliation() -> None:
    code = _code(reconciliation="    pass\n")

    assert host_schema_numeric_alias_findings(ast.parse(code), _bindings()) == []


def test_host_schema_declines_ambiguous_numeric_aliases() -> None:
    code = _code(reconciliation="""
    assert np.allclose(numeric["population_note"], numeric["population_n"])
    assert np.allclose(numeric["population_note"], numeric["event_n"])
""")

    assert host_schema_numeric_alias_findings(ast.parse(code), _bindings()) == []


def test_host_schema_declines_multiple_typed_inputs() -> None:
    bindings = _bindings()
    bindings["table:other"] = bindings["table:result"]

    assert host_schema_numeric_alias_findings(ast.parse(_code()), bindings) == []


def test_host_schema_declines_untyped_or_dynamic_numeric_sequences() -> None:
    v2 = _bindings()
    v2["table:result"]["product_contract"]["schema_version"] = (  # type: ignore[index]
        "easyicu.host_typed_product.v2"
    )
    dynamic = _code().replace(
        'numeric_fields = [\n        "event_n",\n        "population_n",\n        "population_note",\n    ]',
        "numeric_fields = list(frame.columns)",
    )

    assert host_schema_numeric_alias_findings(ast.parse(_code()), v2) == []
    assert host_schema_numeric_alias_findings(ast.parse(dynamic), _bindings()) == []


def test_host_schema_declines_when_host_declares_field_numeric() -> None:
    bindings = _bindings()
    contract = bindings["table:result"]["product_contract"]  # type: ignore[index]
    contract["numeric_columns"].append("population_note")  # type: ignore[union-attr]

    assert host_schema_numeric_alias_findings(ast.parse(_code()), bindings) == []
