"""Generated code reads host-runtime coordinates, not values baked into it.

A hard-coded execution-cohort row count is repaired to the host's runtime row
count, and categorical levels read from the outbound context's opaque
projection are repaired to the digest-verified local context's levels; a
dynamic count, unrelated table constants and unrelated shape metadata are left
alone.
"""

from __future__ import annotations

from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair


def test_hardcoded_execution_cohort_count_uses_host_runtime_coordinate(ra):
    step = ra.AnalysisStep(
        step_id="cohort_summary",
        intent="Summarize the exact locked execution cohort.",
        inputs=["age"],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )
    code = """\
import os
import pandas as pd

cohort_path = os.environ["COHORT_PARQUET"]
df = pd.read_parquet(cohort_path)
locked_n = int(len(df))
if locked_n != 94458:
    raise ValueError("locked cohort row count mismatch")
"""

    findings = audit_mechanical_code_contracts(code, step)
    row_findings = [
        finding
        for finding in findings
        if (finding.detail or {}).get("reason")
        == "execution_cohort_row_count_hardcoded"
    ]

    assert len(row_findings) == 1
    repaired, names = deterministic_concept_audit_repair(
        code,
        [row_findings[0].message],
        repair_findings=row_findings,
    )
    assert names == ["execution_cohort_runtime_row_count_v1"]
    assert "94458" not in repaired
    assert 'environ["EASYICU_COHORT_ROWS"]' in repaired
    assert not any(
        (finding.detail or {}).get("reason") == "execution_cohort_row_count_hardcoded"
        for finding in audit_mechanical_code_contracts(repaired, step)
    )


def test_dynamic_execution_cohort_count_and_unrelated_table_constants_are_allowed(ra):
    step = ra.AnalysisStep(
        step_id="cohort_summary",
        intent="Summarize the exact locked execution cohort.",
        inputs=["age"],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )
    code = """\
import os
import pandas as pd

df = pd.read_parquet(os.environ["COHORT_PARQUET"])
expected_n = int(os.environ["EASYICU_COHORT_ROWS"])
if len(df) != expected_n:
    raise ValueError("locked cohort row count mismatch")
display_rows = [{"label": "n"}, {"label": "missing"}]
if len(display_rows) != 2:
    raise ValueError("display schema changed")
"""

    assert not any(
        (finding.detail or {}).get("reason") == "execution_cohort_row_count_hardcoded"
        for finding in audit_mechanical_code_contracts(code, step)
    )


def test_outbound_opaque_levels_bind_to_digest_verified_local_context(ra):
    step = ra.AnalysisStep(
        step_id="cohort_summary",
        intent="Summarize a closed categorical distribution.",
        inputs=["sex"],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )
    # The two prologue lines are part of the fixture, not decoration: the gate
    # runs on the assembled script, and a module-level read of a name nothing
    # binds -- `context` here -- is itself a blocking finding.
    code = """\
import json
import pathlib

context = json.loads(pathlib.Path("research_context.json").read_text())
sex_metadata = next(
    variable for variable in context["variables"] if variable["name"] == "sex"
)
sex_levels = sex_metadata["observed_shape"]["opaque_levels"]
"""

    findings = audit_mechanical_code_contracts(code, step)
    level_findings = [
        finding
        for finding in findings
        if (finding.detail or {}).get("reason")
        == "runtime_context_opaque_levels_projection"
    ]

    assert len(level_findings) == 1
    repaired, names = deterministic_concept_audit_repair(
        code,
        [level_findings[0].message],
        repair_findings=level_findings,
    )
    assert names == ["runtime_context_private_levels_v1"]
    assert "sex_metadata['observed_domain']['levels']" in repaired
    assert "Female" not in repaired
    assert "Male" not in repaired
    assert not audit_mechanical_code_contracts(repaired, step)


def test_runtime_context_level_bridge_does_not_rewrite_unrelated_shape_metadata(ra):
    step = ra.AnalysisStep(
        step_id="cohort_summary",
        intent="Inspect a safe metadata projection.",
        inputs=["sex"],
        expected_outputs=["table:cohort_summary"],
        method="descriptive_summary",
    )
    code = """\
shape = variable["observed_shape"]["shape"]
levels = variable["observed_domain"]["levels"]
"""

    assert not any(
        (finding.detail or {}).get("reason")
        == "runtime_context_opaque_levels_projection"
        for finding in audit_mechanical_code_contracts(code, step)
    )
