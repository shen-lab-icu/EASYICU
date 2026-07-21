"""Regression locks for lossless wide-trajectory coder prompt transport."""

from __future__ import annotations

from easyicu.research_agent.agents.core import CoderAgent
from easyicu.research_agent.research_context.prompt_variables import (
    compact_fixed_window_trajectory_prompt,
)
from easyicu.research_agent.schema import (
    AggregationRule,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    FixedWindowTrajectoryMetadata,
    MissingnessProfile,
    ResearchContext,
    VariableRole,
)


class _CaptureLLM:
    def __init__(self) -> None:
        self.calls = []

    def complete(self, messages, **_kwargs):  # noqa: ANN001
        self.calls.append(list(messages))
        return "import os\nvalue = 1\n"


def _trajectory_context() -> ResearchContext:
    variables = []
    families = ("total", "resp", "cardio", "cns", "coag", "liver", "renal")
    for family in families:
        is_total = family == "total"
        for window in range(12):
            start = window * 6
            end = start + 6
            variables.append(
                ConceptDescriptor(
                    name=f"trajectory_{family}_h{start}_{end}",
                    role=(
                        VariableRole.COMPOSITE_SCORE
                        if is_total
                        else VariableRole.ORDINAL_SCORE
                    ),
                    dtype="float32",
                    valid_range=[0.0, 24.0] if is_total else [0.0, 4.0],
                    observed_domain={
                        "n_unique": 25 if is_total else 5,
                        "is_constant": False,
                        "is_binary": False,
                        "min": 0.0,
                        "max": 20.0 if is_total else 4.0,
                    },
                    aggregation_default=AggregationRule.MAX_LAST,
                    is_ordinal=True,
                    ordinal_levels=None if is_total else [0, 1, 2, 3, 4],
                    temporal_resolution="fixed 6-hour relative windows",
                    fixed_window_trajectory=FixedWindowTrajectoryMetadata(
                        family=family,
                        window_start_hours=float(start),
                        window_end_hours=float(end),
                        window_width_hours=6.0,
                        source_scale="ordinal",
                        representation_kind="fractional_window_summary",
                        observed_fractional_values=True,
                    ),
                    pitfalls=(
                        ["Audit component completeness before interpretation."]
                        if is_total
                        else ["Zero and unavailable inputs must not be conflated."]
                    ),
                    missingness=MissingnessProfile(
                        fraction_missing=window / 100,
                        n_missing=window,
                        n_total=100,
                        missingness_severity="low" if window < 10 else "medium",
                    ),
                )
            )
    variables.append(
        ConceptDescriptor(
            name="death",
            role=VariableRole.OUTCOME,
            dtype="int64",
            observed_domain={"is_binary": True, "n_unique": 2},
        )
    )
    return ResearchContext(
        research_question=(
            "Discover fixed-window ICU trajectories, preserve missingness, "
            "assess stability, and describe their relationship to mortality."
        ),
        cohort=CohortDescriptor(
            cohort_name="wide_trajectory",
            database="synthetic",
            n_stays=1000,
            n_patients=1000,
            inclusion_criteria=["Adult ICU stays with a prespecified minimum coverage"],
        ),
        variables=variables,
        target_outcome="death",
    )


def _trajectory_step(context: ResearchContext, *, representation: bool) -> AnalysisStep:
    inputs = [
        variable.name
        for variable in context.variables
        if variable.fixed_window_trajectory is not None
    ]
    if representation:
        return AnalysisStep(
            step_id="03_build_trajectory_representation",
            intent="Build the missingness-aware ordered fixed-window representation.",
            inputs=["artifact:analysis_cohort", *inputs],
            expected_outputs=[
                "artifact:trajectory_representation",
                "table:trajectory_membership",
                "manifest:trajectory_representation_schema",
            ],
            method="ordinal_aware_observed_data_trajectory_representation",
        )
    return AnalysisStep(
        step_id="02_longitudinal_missingness_audit",
        intent="Audit window- and component-level longitudinal missingness.",
        inputs=["artifact:analysis_cohort", *inputs],
        expected_outputs=[
            "table:trajectory_missingness",
            "manifest:trajectory_missingness_audit",
        ],
        method="longitudinal_missingness_and_measurement_availability_audit",
    )


def _payload_bytes(messages) -> int:  # noqa: ANN001
    return sum(len(str(message.content or "").encode("utf-8")) for message in messages)


def test_wide_trajectory_projection_preserves_every_exact_window_coordinate() -> None:
    context = _trajectory_context()
    trajectory_variables = [
        variable
        for variable in context.variables
        if variable.fixed_window_trajectory is not None
    ]

    projection = compact_fixed_window_trajectory_prompt(trajectory_variables)
    rendered = dict(projection.variable_lines)

    assert len(rendered) == 84
    assert set(rendered) == {variable.name for variable in trajectory_variables}
    assert len(projection.shared_lines) == 2
    assert "plausibility_range(flag_only;never_exclude_rows)=[0.0, 24.0]" in (
        projection.shared_lines[0]
    )
    assert "representation=fractional_window_summary" in "\n".join(
        projection.shared_lines
    )
    assert (
        "f='resp' t=[18,24)h obs=0:4/u5 m=3.0%/low"
        in rendered["trajectory_resp_h18_24"]
    )


def test_wide_trajectory_generation_prompts_stay_below_transport_gate() -> None:
    context = _trajectory_context()

    for representation in (False, True):
        step = _trajectory_step(context, representation=representation)
        llm = _CaptureLLM()

        CoderAgent(llm).run(context=context, step=step)

        messages = llm.calls[0]
        payload = "\n".join(str(message.content or "") for message in messages)
        assert _payload_bytes(messages) <= 42_000
        assert "Shared fixed-window trajectory policies" in payload
        for variable in context.variables:
            if variable.fixed_window_trajectory is not None:
                assert variable.name in payload
