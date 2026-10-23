"""Step-scoped auxiliary summaries keep the closed deterministic executor."""

import json

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.cohort_summary_executor import (
    cohort_summary_executor_code,
    cohort_summary_executor_owns_step,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.baseline_requirements import (
    baseline_requirement_coverage,
    bind_baseline_requirements,
    candidate_baseline_requirements,
)
from easyicu.research_agent.schema import (
    AnalysisPlan, AnalysisStep, CohortDescriptor, ConceptDescriptor, ResearchContext,
)
from easyicu.research_agent.reporting.article_contract import (
    build_article_analysis_contract,
    roles_covered_by_plan,
)
from easyicu.research_agent.reporting.writer_evidence import _preferred_writer_evidence_names


def _step():
    return AnalysisStep(
        step_id="baseline", planned_analysis_role="auxiliary",
        method="descriptive_cohort_summary",
        intent="Describe the closed baseline columns in the declared cohort.",
        inputs=["artifact:analysis_cohort", "age"],
        expected_outputs=["table:cohort_summary_baseline"],
    )


def test_scoped_summary_retains_the_deterministic_executor_and_registered_product(tmp_path, monkeypatch):
    import easyicu.research_agent.execution.runners.cohort_summary_executor as owner

    step = _step()
    assert cohort_summary_executor_owns_step(step)
    plan = AnalysisPlan(research_question="Synthetic baseline context", steps=[step])
    assert select_standard_executor(step, plan=plan).analysis_kind == "descriptive_cohort_summary"
    source = tmp_path / "synthetic.parquet"
    frame = pd.DataFrame({"age": [30.0, 50.0, None]})
    (tmp_path / "research_context.json").write_text(json.dumps({"variables": [{
        "name": "age", "role": "demographic", "dtype": "float64", "unit": "years",
    }]}))
    monkeypatch.setattr(owner, "load_cohort_summary_frame", lambda **kwargs: (frame, source))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("STEP_OUT_DIR", str(tmp_path / "out"))
    exec(cohort_summary_executor_code(step), {})
    summary = json.loads((tmp_path / "out/step_summary.json").read_text())
    assert summary["output_files"] == {"table:cohort_summary_baseline": "cohort_summary_baseline.csv"}
    assert summary["cohort_n"] == 3
    table = pd.read_csv(tmp_path / "out/cohort_summary_baseline.csv")
    assert set(table["variable"]) == {"__cohort__", "age"}
    values = table.loc[table["variable"] == "age"].set_index("statistic")["value"]
    assert values["mean"] == 40.0
    assert values["missing_n"] == 1
    # Register the auxiliary first: its basename must never seize the primary
    # summary's compatibility alias, regardless of plan ordering.
    evidence = EvidenceStore(tmp_path / "registered")
    evidence.register_file(
        kind="table", description="Synthetic baseline summary",
        source_path=tmp_path / "out/cohort_summary_baseline.csv", produced_by_step=step.step_id,
    )
    primary = tmp_path / "primary/cohort_summary.csv"
    primary.parent.mkdir()
    primary.write_text("variable,statistic,value\nmeasurement,mean,7\n")
    evidence.register_file(
        kind="table", description="Synthetic primary summary",
        source_path=primary, produced_by_step="primary_summary",
    )
    assert "cohort_summary" in _preferred_writer_evidence_names(evidence)
    assert evidence.get("cohort_summary").produced_by_step == "primary_summary"


def test_primary_summary_cannot_change_its_registered_product():
    from easyicu.research_agent.execution.runners.cohort_summary_executor import run_cohort_summary_from_env

    with pytest.raises(RuntimeError, match="retain its registered product"):
        run_cohort_summary_from_env(
            declared_columns=["age"], typed_cohort_input="artifact:analysis_cohort",
            primary_summary=True, summary_product_id="table:cohort_summary_baseline",
        )


@pytest.mark.parametrize("change", [
    {"step_id": "different_step"},
    {"expected_outputs": ["table:cohort_summary_someone_else"]},
    {"planned_analysis_role": "primary"},
    {"method": "unregistered_analysis"},
    {"inputs": ["artifact:analysis_cohort"]},
    {"step_id": "../baseline", "expected_outputs": ["table:cohort_summary_../baseline"]},
])
def test_scoped_product_never_relaxes_the_closed_summary_contract(change):
    assert not cohort_summary_executor_owns_step(_step().model_copy(update=change))


def test_scoped_summary_retains_the_accepted_baseline_roster():
    requirements = candidate_baseline_requirements(
        plan={"steps": [_step().model_dump(mode="json")]},
        source_plan_sha256="a" * 64,
        selected_concepts=["age"], catalog_columns=["age"],
    )
    assert requirements is not None
    assert requirements.schema_version == "easyicu.accepted_baseline_requirements/2"
    assert requirements.tables[0].source_step_id == "baseline"
    assert requirements.tables[0].group_by is None
    assert [item.name for item in requirements.tables[0].variables] == ["age"]
    context = ResearchContext(
        research_question="Describe synthetic baseline characteristics.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=0),
        variables=[ConceptDescriptor(name="age", role="demographic", dtype="float64")],
    )
    context = bind_baseline_requirements(context, requirements.model_dump(mode="json"))
    plan = AnalysisPlan(research_question=context.research_question, steps=[_step()])
    coverage = baseline_requirement_coverage(context, plan)
    assert coverage["status"] == "complete"
    assert coverage["tables"][0]["matched_step_id"] == "baseline"
    contract = build_article_analysis_contract(context, analysis_type="association_study")
    assert "baseline_context" in roles_covered_by_plan(plan, contract)
    wrong_owner = plan.model_copy(update={"steps": [
        _step().model_copy(update={"step_id": "another_step"}),
    ]})
    assert "baseline_context" not in roles_covered_by_plan(wrong_owner, contract)


@pytest.mark.parametrize("product", ["table:../summary", "artifact:summary"])
def test_runtime_product_identity_must_remain_a_canonical_table(product):
    from easyicu.research_agent.execution.runners.cohort_summary_executor import run_cohort_summary_from_env

    with pytest.raises(RuntimeError, match="canonical table product"):
        run_cohort_summary_from_env(
            declared_columns=["age"], typed_cohort_input="artifact:analysis_cohort",
            summary_product_id=product,
        )
