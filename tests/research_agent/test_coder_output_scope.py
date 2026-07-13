from __future__ import annotations

from easyicu.research_agent.agents import CoderAgent
from easyicu.research_agent.llm import LLMMessage


class _RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[LLMMessage] = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.messages = list(messages)
        return "import os\n"


def _context(ra):
    return ra.ResearchContext(
        research_question="Describe an ICU cohort.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=10, n_patients=10
        ),
        variables=[],
    )


def test_coder_prompt_forbids_figure_when_not_declared(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="summary",
        intent="Compute a summary table.",
        inputs=["x"],
        expected_outputs=["table:summary"],
        method="descriptive_summary",
    )

    CoderAgent(llm).run(context=_context(ra), step=step)

    prompt = llm.messages[-1].content
    assert "DECLARED OUTPUT SCOPE (binding)" in prompt
    assert "declares no figure product" in prompt
    assert "Do not render, save, or register figures" in prompt


def test_coder_prompt_allows_only_declared_figure_products(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="render",
        intent="Render the declared figure.",
        inputs=["artifact:summary"],
        expected_outputs=["figure:summary"],
        method="descriptive_summary",
    )

    CoderAgent(llm).run(context=_context(ra), step=step)

    prompt = llm.messages[-1].content
    assert "Figure rendering is allowed only for the explicitly declared" in prompt
    assert "declares no figure product" not in prompt


def test_coder_prompt_binds_typed_inputs_to_resolved_manifest(ra):
    llm = _RecordingLLM()
    step = ra.AnalysisStep(
        step_id="consume",
        intent="Consume the declared upstream table.",
        inputs=["table:scaling_summary"],
        expected_outputs=["table:result"],
        method="descriptive_summary",
    )

    CoderAgent(llm).run(context=_context(ra), step=step)

    prompt = llm.messages[-1].content
    assert "TYPED INPUT BINDING (binding)" in prompt
    assert "EASYICU_RESOLVED_INPUTS_JSON" in prompt
    assert "Do not glob EASYICU_EVIDENCE_DIR" in prompt
    assert "reconstruct a declared upstream product" in prompt


def test_runtime_only_builds_visualization_request_for_figure_step(ra):
    context = _context(ra)
    supervisor = ra.RuntimeSupervisor()
    state = supervisor.bootstrap_state(run_id="run", context=context)
    table_step = ra.AnalysisStep(
        step_id="summary",
        intent="Compute a summary table.",
        expected_outputs=["table:summary"],
    )
    figure_step = ra.AnalysisStep(
        step_id="render",
        intent="Render the summary.",
        expected_outputs=["figure:summary"],
    )

    table_state = supervisor.prepare_step_state(
        state=state, context=context, step=table_step, evidence_refs=[]
    )
    figure_state = supervisor.prepare_step_state(
        state=state, context=context, step=figure_step, evidence_refs=[]
    )

    assert table_state.visualization_request is None
    assert figure_state.visualization_request is not None
