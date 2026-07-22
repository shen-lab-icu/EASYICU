"""Parity tests for the default LangGraph phase dispatcher.

``pipeline.run_with_graph(...)`` routes the existing
``plan → execute → write → finalise`` phases through a
``langgraph.graph.StateGraph``. The wrapper is intended to have
identical behaviour to the default sequential ``pipeline.run(...)``
path; this test pins that contract.

LangGraph is a core research-agent dependency.
"""

from __future__ import annotations

from pathlib import Path


def _run_args(ra, cohort):
    return dict(
        question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=cohort,
        cohort_name="graph_poc",
        database="synthetic",
        target_outcome="death",
    )


def test_run_with_graph_returns_pipeline_result(ra, synthetic_cohort, tmp_path: Path):
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "graph_run",
        llm=ra.MockLLMClient(),
        runner_kind="subprocess",
    )
    result = pipeline.run_with_graph(**_run_args(ra, synthetic_cohort))
    assert result.run_id
    assert Path(result.manifest_path).exists()
    assert result.evidence_count >= 1
    receipt = Path(result.workdir) / "orchestration_runtime.json"
    assert '"backend": "langgraph"' in receipt.read_text(encoding="utf-8")


def test_run_with_graph_parity_with_run(ra, synthetic_cohort, tmp_path: Path):
    """The graph path must produce a result whose published fields
    match the sequential path on the same inputs.

    Run ids and timestamps will differ (each run gets a fresh
    directory), so we compare the *structural* outputs: evidence
    count, finding count, the existence of canonical artefacts, and
    the manuscript scaffold path.
    """

    seq_pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "seq",
        llm=ra.MockLLMClient(),
        runner_kind="subprocess",
    )
    graph_pipe = ra.ResearchAgentPipeline(
        workdir=tmp_path / "graph",
        llm=ra.MockLLMClient(),
        runner_kind="subprocess",
    )

    seq = seq_pipe.run(_use_graph=False, **_run_args(ra, synthetic_cohort))
    graph = graph_pipe.run(**_run_args(ra, synthetic_cohort))

    assert seq.evidence_count == graph.evidence_count
    assert seq.findings_count == graph.findings_count
    assert Path(seq.manifest_path).exists() and Path(graph.manifest_path).exists()
    assert Path(seq.manuscript_path).exists() and Path(graph.manuscript_path).exists()
    assert Path(seq.report_path).exists() and Path(graph.report_path).exists()
    assert '"backend": "legacy_sequential"' in (
        Path(seq.workdir) / "orchestration_runtime.json"
    ).read_text(encoding="utf-8")


def test_build_pipeline_graph_is_compiled_runnable(ra):
    """The graph builder returns a compiled runnable so that callers
    can ``invoke({})`` it directly without re-running ``compile()``.
    """
    from easyicu.research_agent.graph import build_pipeline_graph

    def _noop_plan():
        class _R:
            aborted_result = "stub"
            findings = []
            evidence = None

        return _R()

    graph = build_pipeline_graph(
        plan_invoker=_noop_plan,
        execute_invoker=lambda p: None,
        write_invoker=lambda p, e: None,
        finalise_invoker=lambda p, e, w: None,
    )
    assert hasattr(
        graph, "invoke"
    ), "build_pipeline_graph must return a compiled runnable"

    final_state = graph.invoke({})
    assert (
        final_state["final_result"] == "stub"
    ), "abort route must surface the aborted_result"
