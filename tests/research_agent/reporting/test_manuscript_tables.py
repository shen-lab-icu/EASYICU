"""Reader tables come only from the exact current Table 1 owner and contract."""

import hashlib

import pandas as pd
import pytest

from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRecord, TableOneSpec


def _source(tmp_path):
    spec = TableOneSpec(
        schema_version="easyicu.table_one/2", p_values_required=False,
        p_value_adjustment="not_applicable_repeated_units",
        group_by="exposure", group_levels=[0, 1],
        variables=[{
            "name": "value", "variable_kind": "continuous", "summary": "median_iqr",
            "test": "none_descriptive_smd_only",
        }],
    )
    frame = pd.DataFrame({"exposure": [0, 0, 1, 1], "value": [1.0, 3.0, 8.0, None]})
    table = build_grouped_table_one(frame, spec)
    path = tmp_path / "evidence" / "baseline_table__table_one.csv"
    path.parent.mkdir()
    table.to_csv(path, index=False)
    record = EvidenceRecord(
        evidence_id="baseline_table", kind="table", description="Table 1",
        relative_path="evidence/" + path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), produced_by_step="baseline",
    )
    plan = AnalysisPlan(
        research_question="Describe baseline values by exposure.",
        steps=[AnalysisStep(
            step_id="baseline", intent="Describe baseline values", inputs=["exposure", "value"],
            expected_outputs=["table:table_one"], method="grouped_table_one",
            table_one_spec=spec,
        )],
    )
    return plan, record, table


def test_reader_table_uses_planned_summary_and_preserves_missingness(tmp_path):
    from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables

    plan, record, _ = _source(tmp_path)
    before = (tmp_path / record.relative_path).read_bytes()
    tables = build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)

    assert len(tables) == 1
    assert tables[0].columns == ("Variable", "Group", "N", "Median [Q1, Q3]", "Missing n (%)", "SMD")
    assert tables[0].rows[1][:5] == ("value", "0", "2", "2.00 [1.50, 2.50]", "0 (0.0%)")
    assert tables[0].rows[2][4] == "1 (50.0%)"
    assert record.sha256 in " ".join(tables[0].notes)
    assert (tmp_path / record.relative_path).read_bytes() == before


def test_reader_group_labels_are_exact_authorized_coordinates(tmp_path):
    from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables

    plan, record, _ = _source(tmp_path)
    plan = plan.model_copy(update={"display_labels": {"exposure=0": "Reference category", "exposure=1": "Comparison category"}})
    table = build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)[0]
    assert {row[1] for row in table.rows} == {"Overall", "Reference category", "Comparison category"}


@pytest.mark.parametrize("mutation", ["wrong_owner", "drift", "missing", "schema", "contract"])
def test_reader_table_fails_closed_at_its_source_boundary(tmp_path, mutation):
    from easyicu.research_agent.reporting.manuscript_tables import (
        ManuscriptTableProjectionError, build_manuscript_tables,
    )

    plan, record, table = _source(tmp_path)
    path = tmp_path / record.relative_path
    if mutation == "wrong_owner":
        record = record.model_copy(update={"produced_by_step": "unrelated"})
    elif mutation == "missing":
        path.unlink()
    elif mutation == "drift":
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        table["schema_version" if mutation == "schema" else "contract_sha256"] = "unsupported"
        table.to_csv(path, index=False)
        record = record.model_copy(update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    with pytest.raises(ManuscriptTableProjectionError):
        build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)


def test_latex_embeds_reader_table_with_escaped_cells(tmp_path):
    from easyicu.research_agent.reporting.latex import scaffold_to_latex
    from easyicu.research_agent.reporting.manuscript_tables import ManuscriptTable

    table = ManuscriptTable(
        caption="Baseline characteristics", columns=("Variable", "Value"),
        rows=(("R&D", "2 (50%)"),), notes=("Analysis only.",),
    )
    rendered = scaffold_to_latex(markdown="# Draft\n\nSee Table 1.", tables=[table])

    assert r"\begin{longtable}" in rendered
    assert r"\caption{Baseline characteristics}" in rendered
    assert r"R\&D & 2 (50\%)" in rendered
    assert r"\endhead" in rendered
    assert "Analysis only." in rendered


def test_write_phase_passes_the_executed_plan_to_the_table_owner():
    import ast
    import inspect
    from easyicu.research_agent.reporting import write_phase

    tree = ast.parse(inspect.getsource(write_phase.run_write_phase))
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_publish_and_audit_manuscript")
    keyword = next(item for item in call.keywords if item.arg == "plan")
    assert ast.unparse(keyword.value) == "execute_result.plan"
    assert "build_manuscript_tables(" in inspect.getsource(write_phase._publish_and_audit_manuscript)


@pytest.mark.parametrize("raw,expected", [("0.002", "<0.1"), ("99.999", ">99.9"), ("0", "0.0"), ("", "N/A")])
def test_reader_preserves_nonzero_rare_missingness(raw, expected):
    from easyicu.research_agent.reporting.manuscript_tables import _percent

    assert _percent(raw) == expected


@pytest.mark.parametrize("field,value", [
    ("denominator_n", "1.5"), ("missing_n", "-1"),
    ("missing_pct", "101"), ("missing_pct", "-0.01"),
])
def test_reader_does_not_round_invalid_counts_or_percentages_into_valid_values(tmp_path, field, value):
    from easyicu.research_agent.reporting.manuscript_tables import (
        ManuscriptTableProjectionError, build_manuscript_tables,
    )

    plan, record, table = _source(tmp_path)
    table[field] = value
    path = tmp_path / record.relative_path
    table.to_csv(path, index=False)
    record = record.model_copy(update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ManuscriptTableProjectionError):
        build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)


def test_reader_does_not_append_percent_unit_to_unavailable_values(tmp_path):
    from easyicu.research_agent.reporting.manuscript_tables import build_manuscript_tables

    plan, record, table = _source(tmp_path)
    table["missing_pct"] = None
    path = tmp_path / record.relative_path
    table.to_csv(path, index=False)
    record = record.model_copy(update={"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    rendered = build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)
    assert rendered[0].rows[1][4] == "0 (N/A)"


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_reader_rejects_out_of_range_p_values(value):
    from easyicu.research_agent.reporting.manuscript_tables import (
        ManuscriptTableProjectionError, _p_value,
    )

    with pytest.raises(ManuscriptTableProjectionError):
        _p_value(value)


@pytest.mark.parametrize("malformed", [b"\xff\xfe", b"schema_version,contract_sha256\neasyicu.table_one_result/3\n"])
def test_reader_wraps_unreadable_or_truncated_source_as_projection_error(tmp_path, malformed):
    from easyicu.research_agent.reporting.manuscript_tables import (
        ManuscriptTableProjectionError, build_manuscript_tables,
    )

    plan, record, _ = _source(tmp_path)
    (tmp_path / record.relative_path).write_bytes(malformed)
    record = record.model_copy(update={"sha256": hashlib.sha256(malformed).hexdigest()})
    with pytest.raises(ManuscriptTableProjectionError):
        build_manuscript_tables(plan=plan, evidence_records=[record], run_dir=tmp_path)
