from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_workflow_graph_and_replay_bundle_build(ra, tmp_path: Path):
    df = pd.DataFrame({
        "stay_id": [1, 2, 3],
        "age": [60, 70, 80],
        "death": [0, 1, 0],
    })
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path, index=False)
    ctx = ra.build_research_context(
        research_question="Predict death",
        cohort=df,
        cohort_name="demo",
        database="synthetic",
        target_outcome="death",
    )
    plan = ra.schema.AnalysisPlan(
        research_question="Predict death",
        steps=[ra.schema.AnalysisStep(step_id="01_table_one", intent="Describe cohort")],
    )
    records = [{
        "step_id": "01_table_one",
        "status": "ok",
        "generation_mode": "llm",
        "evidence_ids": ["table_one"],
    }]
    graph = ra.build_workflow_graph(
        run_id="run_demo",
        context=ctx,
        plan=plan,
        per_step_records=records,
        paused_after_analysis=True,
    )
    assert any(n.node_id == "context" for n in graph.nodes)
    mermaid = ra.render_workflow_graph_mermaid(graph)
    assert "flowchart TD" in mermaid

    replay = ra.build_execution_replay(
        run_id="run_demo",
        cohort_path=cohort_path,
        context_path="research_context.json",
        plan_path="analysis_plan.json",
        llm_signature="mock",
        prompt_pack_version="v1",
        per_step_records=records,
        findings=[],
        evidence_ids=["table_one"],
    )
    assert replay.run_id == "run_demo"
    assert replay.steps[0].step_id == "01_table_one"
