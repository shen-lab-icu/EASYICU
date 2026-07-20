from __future__ import annotations

import ast
from datetime import timedelta
import json

import pytest

from easyicu.research_agent.agents.core import CoderAgent, _looks_like_python_script
from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts
from easyicu.research_agent.repairs.source import deterministic_concept_audit_repair
from easyicu.research_agent.research_context.prompt_scope import (
    coder_guide_for_step,
    scoped_coder_context,
)
from easyicu.research_agent.execution.concept_audit_cache import LLMConceptAuditCache
from easyicu.research_agent.orchestration.resume import _looks_like_generated_python
from easyicu.research_agent.authority.provider_budget import (
    ProviderCallBudgetExhausted,
    StepProviderCallBudget,
)
from easyicu.research_agent.schema import ValidationFinding


class _SequenceLLM:
    def __init__(self, responses):  # noqa: ANN001
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append((list(messages), dict(kwargs)))
        return self.responses.pop(0)


def _context(ra, *, variable_count: int = 0):
    variables = [
        ra.ConceptDescriptor(
            name=f"unused_{index}",
            description=f"Unrelated registered column {index}",
            role="other",
            dtype="float64",
        )
        for index in range(variable_count)
    ]
    variables.extend(
        [
            ra.ConceptDescriptor(
                name="selected_first",
                description="Planner-selected exposure representation",
                role="intervention",
                dtype="float64",
                source_concept="selected",
            ),
            ra.ConceptDescriptor(
                name="selected_measured",
                description="Registered measurement companion",
                role="meta",
                dtype="bool",
                source_concept="selected",
            ),
            ra.ConceptDescriptor(
                name="outcome",
                description="Registered target outcome",
                role="outcome",
                dtype="float64",
            ),
        ]
    )
    return ra.ResearchContext(
        research_question="Run the planner-owned ICU analysis.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=20, n_patients=18
        ),
        variables=variables,
        primary_exposure="selected_first",
        target_outcome="outcome",
    )


def _figure_step(ra):
    return ra.AnalysisStep(
        step_id="render",
        intent="Render validated structural accounting inputs.",
        inputs=["table:cohort_flow", "selected_first"],
        expected_outputs=["figure:cohort_flow"],
        method="visualization",
    )


def test_step_scoped_coder_context_keeps_declared_family_and_drops_unrelated(ra):
    context = _context(ra, variable_count=70)
    scoped = scoped_coder_context(context, _figure_step(ra))
    names = {variable.name for variable in scoped.variables}

    assert {"selected_first", "selected_measured", "outcome"} <= names
    assert "unused_69" not in names
    assert len(scoped.variables) <= 36


def test_step_scoped_context_does_not_pad_unused_capacity(ra):
    context = _context(ra, variable_count=70)

    scoped = scoped_coder_context(context, _figure_step(ra))

    assert {variable.name for variable in scoped.variables} == {
        "selected_first",
        "selected_measured",
        "outcome",
    }


def test_step_scoped_context_keeps_all_declared_source_concept_companions(ra):
    variables = [
        ra.ConceptDescriptor(name=f"filler_{index}", dtype="float64")
        for index in range(45)
    ]
    variables.extend(
        [
            ra.ConceptDescriptor(
                name="event_first", dtype="float64", source_concept="event"
            ),
            ra.ConceptDescriptor(name="event_n", dtype="int64", source_concept="event"),
            ra.ConceptDescriptor(
                name="event_measured_6h", dtype="bool", source_concept="event"
            ),
        ]
    )
    context = ra.ResearchContext(
        research_question="Describe the planner-selected event.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=20, n_patients=18
        ),
        variables=variables,
    )
    step = ra.AnalysisStep(
        step_id="event",
        intent="Use the declared event representation.",
        inputs=["event_first"],
        expected_outputs=["table:event_summary"],
        method="descriptive_summary",
    )

    scoped = scoped_coder_context(context, step, max_variables=36)
    names = {variable.name for variable in scoped.variables}

    assert {"event_first", "event_n", "event_measured_6h"} <= names
    assert len(scoped.variables) <= 36


def test_step_scoped_context_drops_unrequested_sibling_summaries(ra):
    variables = [
        ra.ConceptDescriptor(
            name=f"event_companion_{index}",
            dtype="float64",
            source_concept="event",
        )
        for index in range(40)
    ]
    context = ra.ResearchContext(
        research_question="Describe the planner-selected event.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=20, n_patients=18
        ),
        variables=variables,
        primary_exposure="event_companion_0",
        target_outcome="outcome",
    )
    step = ra.AnalysisStep(
        step_id="event_summary",
        intent="Summarise the declared event.",
        inputs=["event_companion_0"],
        expected_outputs=["table:event_summary"],
        method="descriptive_summary",
    )

    scoped = scoped_coder_context(context, step, max_variables=36)

    assert [variable.name for variable in scoped.variables] == ["event_companion_0"]


def test_step_scoped_context_never_truncates_explicit_planner_inputs(ra):
    variables = [
        ra.ConceptDescriptor(name=f"declared_{index}", dtype="float64")
        for index in range(40)
    ]
    context = ra.ResearchContext(
        research_question="Summarise every Planner-declared variable.",
        cohort=ra.CohortDescriptor(
            cohort_name="demo", database="synthetic", n_stays=20, n_patients=18
        ),
        variables=variables,
    )
    step = ra.AnalysisStep(
        step_id="wide_summary",
        intent="Summarise the declared variables.",
        inputs=[variable.name for variable in variables],
        expected_outputs=["table:wide_summary"],
        method="descriptive_summary",
    )

    scoped = scoped_coder_context(context, step, max_variables=36)

    assert len(scoped.variables) == 40
    assert {variable.name for variable in scoped.variables} == {
        f"declared_{index}" for index in range(40)
    }


def test_figure_coder_guide_excludes_unrelated_method_families(ra):
    from easyicu.research_agent.agents.core import _CODER_GUIDE

    guide = coder_guide_for_step(_CODER_GUIDE, _figure_step(ra))

    assert "For rendering-only figure steps" in guide
    assert 'Use matplotlib\'s "Agg" backend' in guide
    assert "TABLE-ONE / DESCRIPTIVE SUMMARIES:" not in guide
    assert "ROBUSTNESS:" not in guide
    assert len(guide) < len(_CODER_GUIDE) * 0.7


def test_render_only_descriptive_method_does_not_load_table_contract(ra):
    from easyicu.research_agent.agents.core import _CODER_GUIDE

    step = ra.AnalysisStep(
        step_id="render",
        intent="Render the declared descriptive figure.",
        inputs=["artifact:summary"],
        expected_outputs=["figure:summary"],
        method="descriptive_summary",
    )

    guide = coder_guide_for_step(_CODER_GUIDE, step)

    assert "For rendering-only figure steps" in guide
    assert 'Use matplotlib\'s "Agg" backend' in guide
    assert "TABLE-ONE / DESCRIPTIVE SUMMARIES:" not in guide
    assert "CLINICAL SCORE AND MISSINGNESS SEMANTICS:" not in guide


def test_table_coder_guide_loads_host_owned_descriptive_input_contract(ra):
    from easyicu.research_agent.agents.core import _CODER_GUIDE

    step = ra.AnalysisStep(
        step_id="describe",
        intent="Describe the Agent-selected cohort variables.",
        inputs=["selected_variable"],
        expected_outputs=["table:table_one"],
        method="table_one",
    )

    guide = coder_guide_for_step(_CODER_GUIDE, step)

    assert "methods.descriptive_inputs" in guide
    assert "strict_numeric_input" in guide
    assert "closed_categorical_counts" in guide
    assert "measurement_provenance_receipt" in guide
    assert (
        "only when an\n  exact declared measured/count companion pair exists" in guide
    )
    assert "Never invent a provenance pair" in guide
    assert "level/count-only `.table`" in guide
    assert "percentage denominator" in guide


def test_mechanical_preflight_rejects_closed_counts_runtime_introspection(ra):
    step = ra.AnalysisStep(
        step_id="describe",
        intent="Describe an Agent-selected closed category.",
        inputs=["selected_variable"],
        expected_outputs=["table:table_one"],
        method="table_one",
    )
    findings = audit_mechanical_code_contracts(
        """
import inspect
from easyicu.research_agent.methods.descriptive_inputs import closed_categorical_counts

def invoke_counts(series, levels):
    signature = inspect.signature(closed_categorical_counts)
    return closed_categorical_counts(series, declared_levels=levels)
""".lstrip(),
        step,
    )

    assert any(
        finding.validator == "mechanical_code_preflight"
        and finding.severity == "error"
        and finding.detail
        and finding.detail.get("reason") == "host_helper_runtime_introspection"
        and finding.detail.get("helper_name") == "closed_categorical_counts"
        for finding in findings
    )


def test_coder_repair_applies_minimal_patch_without_full_rewrite(ra):
    patch = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [{"old": "value = 1", "new": "value = 2", "expected_count": 1}],
        }
    )
    llm = _SequenceLLM([patch])
    repaired = CoderAgent(llm).repair(
        context=_context(ra),
        step=_figure_step(ra),
        code="import os\nvalue = 1\n",
        run_log="ERROR: local value is invalid",
    )

    assert repaired == "import os\nvalue = 2\n"
    assert len(llm.calls) == 1
    assert llm.calls[0][1]["max_tokens"] == 2048
    assert "Do not return a complete script" in llm.calls[0][0][-1].content


def test_coder_repair_requests_full_rewrite_only_after_patch_failure(ra):
    llm = _SequenceLLM(["not json", "import os\nvalue = 3\n"])
    budget = StepProviderCallBudget(2, step_id="render")
    repaired = CoderAgent(llm).repair(
        context=_context(ra),
        step=_figure_step(ra),
        code="import os\nvalue = 1\n",
        run_log="ERROR: local value is invalid",
        provider_budget=budget,
    )

    assert repaired.endswith("value = 3")
    assert len(llm.calls) == 2
    assert budget.used == 2
    assert budget.categories == ("repair_patch", "repair_full_rewrite")
    assert "FULL-REWRITE FALLBACK" in llm.calls[1][0][-1].content


def test_coder_repair_budget_exhaustion_prevents_full_rewrite(ra):
    llm = _SequenceLLM(["not json", "import os\nvalue = 3\n"])
    budget = StepProviderCallBudget(1, step_id="render")

    with pytest.raises(ProviderCallBudgetExhausted) as exc_info:
        CoderAgent(llm).repair(
            context=_context(ra),
            step=_figure_step(ra),
            code="import os\nvalue = 1\n",
            run_log="ERROR: local value is invalid",
            provider_budget=budget,
        )

    assert exc_info.value.category == "repair_full_rewrite"
    assert len(llm.calls) == 1
    assert budget.categories == ("repair_patch",)


def test_coder_repair_uses_last_non_audit_slot_for_direct_rewrite(ra):
    """Do not strand a repair by spending its sole slot on a bad patch."""
    llm = _SequenceLLM(["import os\nvalue = 3\n"])
    budget = StepProviderCallBudget(
        3,
        step_id="render",
        reserved_final_category="concept_audit",
    )
    budget.consume("initial_generation")

    coder = CoderAgent(llm)
    repaired = coder.repair(
        context=_context(ra),
        step=_figure_step(ra),
        code="import os\nvalue = 1\n",
        run_log="ERROR: local value is invalid",
        provider_budget=budget,
    )

    assert repaired.endswith("value = 3")
    assert len(llm.calls) == 1
    assert budget.categories == ("initial_generation", "repair_full_rewrite")
    assert coder.last_repair_transport == "full_rewrite"
    assert coder.last_repair_provider_calls == 1
    assert budget.can_consume("concept_audit") is True
    assert budget.can_consume("repair_patch") is False


def test_patch_json_is_never_accepted_as_complete_python_script():
    payload = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {
                    "old": "def choose(frame):\n    return frame.columns[0]",
                    "new": "def choose(frame):\n    return None",
                    "expected_count": 1,
                }
            ],
        }
    )
    assert not _looks_like_python_script(payload)
    assert not _looks_like_generated_python(payload)


def test_mechanical_preflight_blocks_arbitrary_numeric_column_fallback(ra):
    code = """
def find_column(frame, candidates, numeric=False):
    for column in candidates:
        if column in frame.columns:
            return column
    if numeric:
        for column in frame.columns:
            if frame[column].notna().any():
                return column
    return None
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail and finding.detail.get("reason") == "arbitrary_column_fallback"
        for finding in findings
    )


def test_mechanical_preflight_blocks_filter_then_plot_for_accounting(ra):
    code = """
def render(frame):
    valid_rows = frame['n'].notna()
    plotted = frame.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_blocks_alias_row_filter_for_accounting(ra):
    code = """
def render(frame):
    accounting = frame
    valid_rows = frame['n'].notna()
    plotted = accounting[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_blocks_sibling_derived_loc_filter(ra):
    code = """
def render(frame, audit):
    valid_rows = audit['n'].notna()
    plotted = frame.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_allows_fail_closed_accounting_guard(ra):
    code = """
def render(frame):
    valid_rows = frame['n'].notna()
    if not valid_rows.all():
        raise ValueError('invalid accounting row')
    plotted = frame.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


@pytest.mark.parametrize(
    "guard",
    [
        "if valid_rows.sum() != len(unrelated):\n        raise ValueError('invalid')",
        "assert valid_rows.sum() == len(unrelated), 'invalid'",
    ],
)
def test_mechanical_preflight_rejects_accounting_guard_against_unrelated_frame(
    ra, guard
):
    code = f"""
def render(frame, unrelated):
    valid_rows = frame['n'].notna()
    {guard}
    plotted = frame.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_allows_accounting_guard_against_frame_alias(ra):
    code = """
def render(frame):
    accounting = frame
    valid_rows = accounting['n'].notna()
    if valid_rows.sum() != len(frame):
        raise ValueError('invalid')
    plotted = accounting.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


@pytest.mark.parametrize(
    "guard",
    [
        "if (~valid_rows).any():\n        raise ValueError('invalid')",
        "if valid_rows.sum() != len(frame):\n        return None",
        "if valid_rows.all() == False:\n        raise ValueError('invalid')",  # noqa: E712
        "assert valid_rows.all(), 'invalid'",
        "assert valid_rows.sum() == len(frame), 'invalid'",
    ],
)
def test_mechanical_preflight_accepts_equivalent_fail_closed_guards(ra, guard):
    code = f"""
def render(frame):
    valid_rows = frame['n'].notna()
    {guard}
    plotted = frame.loc[valid_rows].copy()
    return plotted
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_ignores_unrelated_mapping_subscript(ra):
    code = """
def render(frame, labels):
    valid_rows = frame['n'].notna()
    selected_label = labels[valid_rows]
    return selected_label
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_filter"
        for finding in findings
    )


def test_mechanical_preflight_blocks_silent_accounting_count_rounding(ra):
    code = """
def render(frame):
    counts = frame['n']
    if counts.isna().any():
        raise ValueError('missing count')
    labels = [f"{value:,.0f}" for value in counts]
    return labels
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_integer_validation"
        for finding in findings
    )


def test_mechanical_preflight_accepts_integer_like_accounting_validation(ra):
    code = """
import numpy as np

def render(frame):
    counts = frame['n']
    if not np.allclose(counts, np.round(counts)):
        raise ValueError('fractional count')
    labels = [f"{value:,.0f}" for value in counts]
    return labels
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_integer_validation"
        for finding in findings
    )


def test_mechanical_preflight_rejects_unrelated_integer_guard(ra):
    code = """
import numpy as np

def render(frame):
    heights = frame['height']
    if not np.allclose(heights, np.round(heights)):
        raise ValueError('fractional height')
    counts = frame['score_count']
    return [f"{value:,.0f}" for value in counts]
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "structural_accounting_integer_validation"
        for finding in findings
    )


def test_mechanical_preflight_repairs_scalar_cast_before_sum(ra):
    code = """
import numpy as np

left = np.array([True, False, True])
right = np.array([True, True, False])
count = int(
    left & right
).sum()
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        value
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for value in (finding.message, finding.detail.get("reason"))
        if value
    ]

    assert messages
    repaired, repair_names = deterministic_concept_audit_repair(code, messages)
    repaired_again, repair_names_again = deterministic_concept_audit_repair(
        repaired, messages
    )
    namespace = {}
    exec(repaired, namespace)

    assert repair_names == ["scalar_cast_before_reduction_v1"]
    assert repair_names_again == []
    assert repaired_again == repaired
    assert namespace["count"] == 1
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in audit_mechanical_code_contracts(repaired, _figure_step(ra))
    )


def test_mechanical_preflight_repairs_unreduced_boolean_mask_count(ra):
    code = """
import pandas as pd

original = pd.Series([1.0, float("inf"), None])
coerced = pd.to_numeric(original, errors="coerce").where(lambda s: s != float("inf"))
invalid_n = int(
    original.notna() & coerced.isna()
)
if invalid_n > 0:
    detected = True
else:
    detected = False
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        finding.detail.get("reason")
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
    ]

    repaired, repair_names = deterministic_concept_audit_repair(code, messages)
    repaired_again, repair_names_again = deterministic_concept_audit_repair(
        repaired, messages
    )
    namespace = {}
    exec(repaired, namespace)

    assert messages == ["scalar_cast_before_reduction"]
    assert repair_names == ["scalar_cast_before_reduction_v1"]
    assert repair_names_again == []
    assert repaired_again == repaired
    assert namespace["invalid_n"] == 1
    assert namespace["detected"] is True


def test_mechanical_preflight_repairs_inverted_right_mask_reduction(ra):
    code = """
import numpy as np
import pandas as pd

frame = pd.DataFrame({"stage": [0.0, np.inf, np.nan]})
stage = pd.to_numeric(frame["stage"], errors="coerce")
invalid_stage_counts = {
    "nonfinite_n": int(
        frame["stage"].notna()
        & ~np.isfinite(stage.to_numpy(dtype=float))
        .sum()
    )
}
if invalid_stage_counts["nonfinite_n"] > 0:
    detected = True
else:
    detected = False
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        finding.detail.get("reason")
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
    ]

    repaired, repair_names = deterministic_concept_audit_repair(code, messages)
    namespace = {}
    exec(repaired, namespace)

    assert messages == ["scalar_cast_before_reduction"]
    assert repair_names == ["scalar_cast_before_reduction_v1"]
    assert namespace["invalid_stage_counts"]["nonfinite_n"] == 1
    assert namespace["detected"] is True
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in audit_mechanical_code_contracts(repaired, _figure_step(ra))
    )


def test_mechanical_preflight_does_not_move_scalar_right_reduction(ra):
    code = """
left = True
right = 2
count = int(left & ~right.sum())
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_does_not_rewrite_scalar_bitwise_integer_guard(ra):
    code = """
left = True
right = False
invalid_n = int(left & right)
if invalid_n > 0:
    raise RuntimeError("invalid")
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_does_not_guess_unproven_boolean_mask_intent(ra):
    code = """
def encode(original, coerced):
    return int(original.notna() & coerced.isna())
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_preserves_shadowed_int_before_sum(ra):
    code = """
def int(value):
    return value

count = int(mask).sum()
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


@pytest.mark.parametrize("unused_import", ["builtins", "operator", "sys"])
def test_mechanical_preflight_allows_unused_namespace_import_before_scalar_repair(
    ra, unused_import
):
    code = f"""
import {unused_import}
count = int(mask).sum()
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        finding.detail.get("reason")
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
    ]
    repaired, repair_names = deterministic_concept_audit_repair(code, messages)

    assert messages == ["scalar_cast_before_reduction"]
    assert repair_names == ["scalar_cast_before_reduction_v1"]
    assert "int((mask).sum())" in repaired


@pytest.mark.parametrize(
    "shadowing",
    [
        "match value:\n    case int:\n        pass",
        "match values:\n    case [*int]:\n        pass",
        "from custom_numeric import *",
        "import builtins\nbuiltins.int = custom_int",
        "__builtins__['int'] = custom_int",
        "__builtins__.int = custom_int",
        "import builtins\nsetattr(builtins, 'int', custom_int)",
        "import builtins as b\nb.int = custom_int",
        "globals()['int'] = custom_int",
        "locals()['int'] = custom_int",
        "exec('int = custom_int')",
        "match value:\n    case {'a': a, **int}:\n        pass",
        "def typed[int](value):\n    return value",
    ],
    ids=[
        "match-as",
        "match-star",
        "import-star",
        "builtins-attribute",
        "builtins-mapping",
        "builtins-object-attribute",
        "builtins-setattr",
        "builtins-alias",
        "globals-mutation",
        "locals-mutation",
        "exec-binding",
        "match-mapping-rest",
        "type-parameter",
    ],
)
def test_mechanical_preflight_preserves_dynamically_shadowed_int_before_sum(
    ra, shadowing
):
    code = f"""
{shadowing}

count = int(mask).sum()
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


@pytest.mark.parametrize(
    "shadowing",
    [
        "runner = exec\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = builtins.exec\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = builtins.__dict__['exec']\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = builtins.__dict__.get('exec')\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = builtins.__getattribute__('exec')\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = getattr(builtins, 'exec')\nrunner('global int; int = custom_int')",
        "import builtins\nsetter = getattr(builtins, 'setattr')\nsetter(builtins, 'int', custom_int)",
        "import builtins\nrunner = builtins.__dict__.copy()['exec']\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = dict(builtins.__dict__)['exec']\nrunner('global int; int = custom_int')",
        "import builtins\nrunner = next(v for k, v in builtins.__dict__.items() if k == 'exec')\nrunner('global int; int = custom_int')",
        "import builtins\ndict.update(builtins.__dict__, {'int': custom_int})",
        "import builtins\nd = builtins.__dict__\nd |= {'int': custom_int}",
        "import builtins, operator\noperator.setitem(builtins.__dict__, 'int', custom_int)",
        "import builtins, operator\noperator.ior(builtins.__dict__, {'int': custom_int})",
        "import builtins\nfrom operator import ior\nior(builtins.__dict__, {'int': custom_int})",
        "import builtins\nbuiltins.__setattr__('int', custom_int)",
        "import builtins, operator\nrunner = operator.getitem(builtins.__dict__, 'exec')\nrunner('global int; int = custom_int')",
        "import sys\nrunner = sys.modules['builtins'].__dict__['exec']\nrunner('global int; int = custom_int')",
        "import sys\ns = sys\nrunner = s.modules['builtins'].exec\nrunner('global int; int = custom_int')",
        "import __main__\n__main__.__dict__.update({'int': custom_int})",
        "from __main__ import __dict__ as scope\nscope.update({'int': custom_int})",
        "import sys\nsys._getframe().f_builtins.update({'int': custom_int})",
        "def anchor():\n    pass\nscope = anchor.__getattribute__('__globals__')\nscope.update({'int': custom_int})",
        "def anchor():\n    pass\nscope = object.__getattribute__(anchor, '__globals__')\nscope.update({'int': custom_int})",
        "import inspect\ndef anchor():\n    pass\nscope = dict(inspect.getmembers(anchor))['__globals__']\nscope.update({'int': custom_int})",
        "import gc\ndef anchor():\n    pass\nscope = gc.get_referents(anchor)[0]\nscope.update({'int': custom_int})",
        "from importlib import import_module as load\nrunner = load('builtins').exec\nrunner('global int; int = custom_int')",
        "import pydoc\nrunner = pydoc.locate('builtins.exec')\nrunner('global int; int = custom_int')",
        "import pkgutil\nrunner = pkgutil.resolve_name('builtins:exec')\nrunner('global int; int = custom_int')",
        "from unittest.mock import patch\nwith patch('builtins.int', new=custom_int):\n    count = int(mask).sum()",
    ],
    ids=[
        "exec-alias",
        "builtins-attribute",
        "builtins-subscript",
        "builtins-mapping-get",
        "builtins-getattribute",
        "getattr-exec",
        "getattr-setattr",
        "builtins-copy",
        "builtins-dict-copy",
        "builtins-items",
        "dict-update-builtins",
        "builtins-alias-ior",
        "operator-setitem",
        "operator-ior",
        "imported-operator-ior",
        "builtins-magic-setattr",
        "operator-getitem",
        "sys-modules-builtins",
        "sys-alias-modules-builtins",
        "main-module-dict",
        "main-module-dict-from-import",
        "sys-frame-builtins",
        "function-getattribute-globals",
        "object-getattribute-globals",
        "inspect-getmembers",
        "gc-get-referents",
        "importlib-alias",
        "pydoc-locate",
        "pkgutil-resolve-name",
        "unittest-mock-patch",
    ],
)
def test_mechanical_preflight_preserves_indirectly_shadowed_int_before_sum(
    ra, shadowing
):
    code = f"""
{shadowing}

count = int(mask).sum()
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code,
        ["scalar_cast_before_reduction"],
    )

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "scalar_cast_before_reduction"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_blocks_unpersisted_binding_metadata(ra):
    code = """
def load(binding):
    relative_path = binding.get('relative_path')
    record = {'evidence_id': binding.get('evidence_id'), 'loaded': True}
    return {'binding': record, 'path': relative_path}

def render(bindings):
    return bindings['table:summary']['binding']['relative_path']
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "unpersisted_binding_metadata"
        for finding in findings
    )


def test_mechanical_preflight_accepts_persisted_binding_metadata(ra):
    code = """
def load(binding):
    relative_path = binding.get('relative_path')
    record = {
        'evidence_id': binding.get('evidence_id'),
        'relative_path': relative_path,
        'loaded': True,
    }
    return {'binding': record, 'path': relative_path}

def render(bindings):
    return bindings['table:summary']['binding']['relative_path']
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "unpersisted_binding_metadata"
        for finding in findings
    )


def test_mechanical_preflight_blocks_nonterminating_provenance_audit(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    failed = any(
        check.get('invalid_pair_n', 0) > 0
        or check.get('discordant_n', 0) > 0
        for check in checks
    )
    return {
        'fail_closed': failed,
        'completed_step_allowed': not failed,
        'checks': checks,
    }

audit = provenance_audit(frame)
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_terminating_provenance_guard(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 0,
        'discordant_n': 0,
    }]
    failed = any(
        check.get('invalid_pair_n', 0) > 0
        or check.get('discordant_n', 0) > 0
        for check in checks
    )
    return {
        'fail_closed': failed,
        'completed_step_allowed': not failed,
        'checks': checks,
    }

audit = provenance_audit(frame)
if audit['fail_closed']:
    raise ValueError('invalid measurement provenance')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_self_raising_provenance_helper_assignment(ra):
    code = """
def fail(message):
    raise RuntimeError(message)

def provenance_audit(frame):
    invalid_pair_n = int(frame['invalid_pair'].sum())
    discordant_n = int(frame['discordant'].sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    if invalid_pair_n > 0 or discordant_n > 0:
        fail('invalid measurement provenance')
    return {'checks': checks}

def main(frame):
    audit = provenance_audit(frame)
    if not audit or not audit.get('checks'):
        fail('measurement provenance audit returned no checks')
    model.fit(frame)

main(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_immediately_returned_provenance_audit_row(ra):
    code = """
def fail(message):
    raise RuntimeError(message)

def provenance_audit(frame):
    invalid_pair_n = int(frame['invalid_pair'].sum())
    discordant_n = int(frame['discordant'].sum())
    if invalid_pair_n > 0 or discordant_n > 0:
        fail('invalid measurement provenance')
    return {'checks': [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]}

def main(frame):
    audit = provenance_audit(frame)
    if not audit or not audit.get('checks'):
        fail('measurement provenance audit returned no checks')
    model.fit(frame)

main(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_rebound_returned_provenance_count(ra):
    code = """
def fail(message):
    raise RuntimeError(message)

def provenance_audit(frame):
    invalid_pair_n = int(frame['invalid_pair'].sum())
    discordant_n = int(frame['discordant'].sum())
    invalid_pair_n = 0
    if invalid_pair_n > 0 or discordant_n > 0:
        fail('invalid measurement provenance')
    return {'checks': [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]}

audit = provenance_audit(frame)
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_nonterminal_provenance_fail_helper(ra):
    code = """
def fail(message):
    if strict:
        raise RuntimeError(message)
    return None

def provenance_audit(frame):
    invalid_pair_n = int(frame['invalid_pair'].sum())
    discordant_n = int(frame['discordant'].sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    if invalid_pair_n > 0 or discordant_n > 0:
        fail('invalid measurement provenance')
    return {'checks': checks}

audit = provenance_audit(frame)
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_provenance_guard_swallowed_by_handler(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    failed = any(
        check['invalid_pair_n'] or check['discordant_n'] for check in checks
    )
    return {
        'checks': checks,
        'fail_closed': failed,
        'completed_step_allowed': not failed,
    }

def main(frame):
    try:
        audit = provenance_audit(frame)
        if audit['fail_closed']:
            raise RuntimeError('invalid measurement provenance')
        model.fit(frame)
    except Exception as exc:
        write_failure_summary(str(exc))

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    finding = next(
        finding
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
    )

    assert any(
        issue.get("failure_mode") == "provenance_guard_swallowed_by_handler"
        and issue.get("handler_line")
        for issue in finding.detail.get("issues", [])
    )


def test_mechanical_preflight_accepts_immediate_provenance_handler_reraise(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    failed = any(
        check['invalid_pair_n'] or check['discordant_n'] for check in checks
    )
    return {
        'checks': checks,
        'fail_closed': failed,
        'completed_step_allowed': not failed,
    }

def main(frame):
    try:
        audit = provenance_audit(frame)
        if audit['fail_closed']:
            raise RuntimeError('invalid measurement provenance')
        model.fit(frame)
    except (Exception, ValueError):
        raise

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("exit_code", ["0", "None"])
def test_mechanical_preflight_rejects_successful_exit_from_provenance_handler(
    ra, exit_code
):
    code = f"""
def provenance_audit(frame):
    checks = [{{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }}]
    failed = any(
        check['invalid_pair_n'] or check['discordant_n'] for check in checks
    )
    return {{
        'checks': checks,
        'fail_closed': failed,
        'completed_step_allowed': not failed,
    }}

def main(frame):
    try:
        audit = provenance_audit(frame)
        if audit['fail_closed']:
            raise RuntimeError('invalid measurement provenance')
        model.fit(frame)
    except Exception:
        raise SystemExit({exit_code})

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_reports_each_unsafe_provenance_handler(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    failed = any(
        check['invalid_pair_n'] or check['discordant_n'] for check in checks
    )
    return {
        'checks': checks,
        'fail_closed': failed,
        'completed_step_allowed': not failed,
    }

def main(frame):
    try:
        audit = provenance_audit(frame)
        if audit['fail_closed']:
            raise RuntimeError('invalid measurement provenance')
        model.fit(frame)
    except KeyError:
        raise
    except Exception:
        write_failure_summary()

main(frame)
"""
    tree = ast.parse(code)
    unsafe_line = next(
        handler.lineno
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        and not (
            handler.body
            and isinstance(handler.body[0], ast.Raise)
            and handler.body[0].exc is None
        )
    )

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    finding = next(
        finding
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
    )
    handler_lines = {
        issue.get("handler_line")
        for issue in finding.detail.get("issues", [])
        if issue.get("failure_mode") == "provenance_guard_swallowed_by_handler"
    }

    assert handler_lines == {unsafe_line}


def test_mechanical_preflight_reports_helper_call_and_handler_locations(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    return {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }

def main(frame):
    try:
        audit = provenance_audit(frame)
        failures = []
        if audit['invalid_pair_n'] or audit['discordant_n']:
            failures.append('invalid measurement provenance')
        if failures:
            raise RuntimeError('; '.join(failures))
        model.fit(frame)
    except Exception as exc:
        write_failure_summary(str(exc))

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    finding = next(
        finding
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
    )
    issues = finding.detail.get("issues", [])

    assert any(
        issue.get("failure_mode") == "provenance_guard_swallowed_by_handler"
        and issue.get("handler_line")
        for issue in issues
    )
    assert any(
        issue.get("failure_mode") == "provenance_helper_result_not_immediately_guarded"
        and issue.get("helper_name") == "provenance_audit"
        and issue.get("call_line")
        and issue.get("following_guard_line")
        for issue in issues
    )


@pytest.mark.parametrize(
    "handler",
    [
        "except Exception:\n        if strict:\n            raise\n        write_failure_summary()",
        "except Exception:\n        write_failure_summary()\n    finally:\n        publish_partial_output()",
    ],
)
def test_mechanical_preflight_does_not_accept_partial_provenance_reraise(ra, handler):
    code = f"""
def provenance_audit(frame):
    checks = [{{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }}]
    failed = any(
        check['invalid_pair_n'] or check['discordant_n'] for check in checks
    )
    return {{
        'checks': checks,
        'fail_closed': failed,
        'completed_step_allowed': not failed,
    }}

def main(frame, strict=True):
    try:
        audit = provenance_audit(frame)
        if audit['fail_closed']:
            raise RuntimeError('invalid measurement provenance')
        model.fit(frame)
    {handler}

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_inline_failure_collection_then_raise(ra):
    code = """
def main(frame):
    valid_pairs = frame['measured'].notna() & frame['count'].notna()
    discordant = valid_pairs & (frame['measured'] != (frame['count'] > 0))
    invalid_pair_n = int((~valid_pairs).sum())
    discordant_n = int(discordant.sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    if failures:
        reason = '; '.join(failures)
        write_failed_summary(checks, reason)
        raise RuntimeError(reason)
    model.fit(frame)

main(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_direct_host_shaped_provenance_guard(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame['invalid_pair_n'])
    discordant_n = int(frame['discordant_n'])
    audit = {
        'source': 'COHORT_PARQUET',
        'checks': [{
            'role': 'audit_only',
            'status': 'checked',
            'invalid_pair_n': invalid_pair_n,
            'discordant_n': discordant_n,
        }],
    }
    if invalid_pair_n > 0 or discordant_n > 0:
        raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "call_site",
    [
        "checks.append(validate_provenance(frame))",
        "record_check(result=validate_provenance(frame))",
    ],
    ids=["positional-argument", "keyword-argument"],
)
def test_mechanical_preflight_accepts_self_raising_provenance_call_as_eager_argument(
    ra, call_site
):
    code = f"""
def validate_provenance(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    check = {{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }}
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')
    return check

checks = []
{call_site}
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "call_site",
    [
        "if enabled:\n    checks.append(validate_provenance(frame))",
        "callback = lambda: checks.append(validate_provenance(frame))",
        (
            "try:\n    checks.append(validate_provenance(frame))\n"
            "except RuntimeError:\n    pass"
        ),
    ],
    ids=["conditional", "lazy-lambda", "swallowed"],
)
def test_mechanical_preflight_rejects_non_direct_eager_argument_provenance_call(
    ra, call_site
):
    code = f"""
def validate_provenance(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    check = {{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }}
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')
    return check

checks = []
{call_site}
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_self_raising_provenance_call_in_covered_loop(
    ra,
):
    code = """
def validate_provenance(frame, measured_column, count_column):
    invalid_pair_n = int(frame[measured_column].isna().sum())
    discordant_n = int(
        (frame[measured_column] != (frame[count_column] > 0)).sum()
    )
    check = {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')
    return check

def main(frame, stems, declared_columns):
    checks = []
    for stem in stems:
        measured_column = f'{stem}_measured'
        count_column = f'{stem}_n'
        if measured_column not in declared_columns or count_column not in declared_columns:
            raise ValueError('incomplete provenance pair')
        checks.append(
            validate_provenance(
                frame,
                measured_column=measured_column,
                count_column=count_column,
            )
        )
    if not checks:
        raise RuntimeError('no completed provenance checks')
    model.fit(frame)

main(frame, stems, declared_columns)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "loop_or_guard",
    [
        """for stem in stems:
        if enabled:
            checks.append(validate_provenance(frame))
    if not checks:
        raise RuntimeError('no checks')""",
        """for stem in stems:
        checks.append(validate_provenance(frame))""",
        """checks.append({'role': 'decoy'})
    for stem in stems:
        checks.append(validate_provenance(frame))
    if not checks:
        raise RuntimeError('no checks')""",
        """for stem in stems:
        checks.append(validate_provenance(frame))
        model.fit(frame)
    if not checks:
        raise RuntimeError('no checks')""",
        """for stem in stems:
        checks.append(validate_provenance(frame))
        if stop:
            break
    if not checks:
        raise RuntimeError('no checks')""",
    ],
    ids=[
        "conditional-append",
        "missing-empty-guard",
        "prepopulated-decoy",
        "result-sink-before-full-coverage",
        "break-before-full-coverage",
    ],
)
def test_mechanical_preflight_rejects_uncovered_eager_argument_provenance_loop(
    ra, loop_or_guard
):
    code = f"""
def validate_provenance(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    check = {{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }}
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')
    return check

def main(frame, stems, enabled):
    checks = []
    {loop_or_guard}
    model.fit(frame)

main(frame, stems, enabled)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_proves_aggregate_provenance_loop_coverage(ra):
    code = """
def main(frame, stems):
    checks = []
    failures = []
    for stem in stems:
        if stem == 'unavailable':
            checks.append({
                'role': 'audit_only',
                'invalid_pair_n': None,
                'discordant_n': None,
            })
            failures.append('unavailable pair')
            continue
        invalid_pair_n = int(frame['measured'].isna().sum())
        discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
        checks.append({
            'role': 'audit_only',
            'invalid_pair_n': invalid_pair_n,
            'discordant_n': discordant_n,
        })
        if invalid_pair_n or discordant_n:
            failures.append('invalid measurement provenance')
    if not checks:
        raise RuntimeError('no provenance checks')
    if failures:
        raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame, stems)
"""
    initial = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        value
        for finding in initial
        if finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for value in (finding.message, finding.detail.get("reason"))
        if value
    ]

    repaired, repair_names = deterministic_concept_audit_repair(code, messages)
    repaired_again, repair_names_again = deterministic_concept_audit_repair(
        repaired, messages
    )
    final = audit_mechanical_code_contracts(repaired, _figure_step(ra))

    assert messages
    assert repair_names == ["provenance_fail_closed_guard_v1"]
    assert repair_names_again == []
    assert repaired_again == repaired
    assert "_easyicu_provenance_loop_observed = False" in repaired
    assert "if not _easyicu_provenance_loop_observed:" in repaired
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in final
    )


def test_mechanical_preflight_rejects_uncovered_aggregate_provenance_loop(ra):
    code = """
def main(frame):
    failures = []
    for item in []:
        invalid_pair_n = int(frame['measured'].isna().sum())
        discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
        checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
                   'discordant_n': discordant_n}]
        if invalid_pair_n or discordant_n:
            failures.append('failed')
    if failures:
        raise RuntimeError('failed')
    model.fit(frame)

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    repaired, repair_names = deterministic_concept_audit_repair(
        code, ["provenance_audit_not_fail_closed"]
    )

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )
    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_rejects_uncovered_continue_with_loop_sentinel(ra):
    code = """
def main(frame, stems, skip):
    checks = []
    failures = []
    _easyicu_provenance_loop_observed = False
    for stem in stems:
        _easyicu_provenance_loop_observed = True
        if skip:
            continue
        invalid_pair_n = int(frame['measured'].isna().sum())
        discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
        checks.append({'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
                       'discordant_n': discordant_n})
        if invalid_pair_n or discordant_n:
            failures.append('failed')
    if not _easyicu_provenance_loop_observed:
        raise RuntimeError('no iterations')
    if failures:
        raise RuntimeError('failed')
    model.fit(frame)

main(frame, stems, skip)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "call_site",
    [
        "if False:\n    provenance_audit(frame)",
        "if enabled:\n    provenance_audit(frame)",
        "for _ in []:\n    provenance_audit(frame)",
        "match mode:\n    case 'audit':\n        provenance_audit(frame)",
        "callback = lambda: provenance_audit(frame)",
        "with contextlib.suppress(RuntimeError):\n    provenance_audit(frame)",
        (
            "try:\n    provenance_audit(frame)\n"
            "except RuntimeError:\n    raise\n"
            "finally:\n    model.fit(frame)"
        ),
    ],
    ids=[
        "false-branch",
        "runtime-branch",
        "empty-loop",
        "match",
        "dead-lambda",
        "suppressed",
        "finally-sink",
    ],
)
def test_mechanical_preflight_rejects_non_direct_self_raising_provenance_call(
    ra, call_site
):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

{call_site}
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_early_returning_self_raising_helper(ra):
    code = """
def provenance_audit(frame, skip):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if skip:
        return
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

provenance_audit(frame, skip)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_direct_return_before_self_raising_guard(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    return {'checks': checks}
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_post_guard_success_returning_helper(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')
    return {'checks': checks}

provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("wrapper", ["module", "terminal_main"])
def test_mechanical_preflight_rejects_provenance_call_after_result_sink(ra, wrapper):
    call_site = "model.fit(frame)\nprovenance_audit(frame)"
    if wrapper == "terminal_main":
        call_site = (
            "def main():\n"
            "    model.fit(frame)\n"
            "    provenance_audit(frame)\n\n"
            "if __name__ == '__main__':\n"
            "    main()"
        )
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

{call_site}
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("definition_kind", ["decorated", "async", "generator"])
def test_mechanical_preflight_rejects_indirect_marker_runtime_binding(
    ra, definition_kind
):
    prefix = ""
    function_head = "def provenance_audit(frame):"
    extra_body = ""
    if definition_kind == "decorated":
        prefix = "@swallow\n"
    elif definition_kind == "async":
        function_head = "async def provenance_audit(frame):"
    else:
        extra_body = "\n    yield checks"
    code = f"""
{prefix}{function_head}
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance'){extra_body}

provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "rebind",
    [
        "class provenance_audit:\n    pass",
        "for provenance_audit in [noop]:\n    pass",
        "with manager() as provenance_audit:\n    pass",
    ],
    ids=["class", "for-target", "with-target"],
)
def test_mechanical_preflight_rejects_rebound_marker_helper(ra, rebind):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

{rebind}
provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "rebind",
    [
        "globals()['provenance_audit'] = noop",
        "vars()['provenance_audit'] = noop",
        "exec('provenance_audit = noop', globals())",
    ],
    ids=["globals", "vars", "exec"],
)
def test_mechanical_preflight_rejects_dynamic_marker_rebinding(ra, rebind):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

{rebind}
provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "rebind",
    [
        "runner = exec\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import builtins\nrunner = builtins.__dict__.get('exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import builtins\nrunner = builtins.__getattribute__('exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import builtins\nrunner = getattr(builtins, 'exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import builtins\nrunner = builtins.__dict__.copy()['exec']\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import builtins\nrunner = next(v for k, v in builtins.__dict__.items() if k == 'exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "name = 'provenance_audit'\nsetattr(sys.modules[__name__], name, noop)",
        "import operator\noperator.setitem(sys.modules[__name__].__dict__, 'provenance_audit', noop)",
        "sys.modules[__name__].__setattr__('provenance_audit', noop)",
        "import builtins, operator\nrunner = operator.getitem(builtins.__dict__, 'exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "runner = sys.modules['builtins'].__dict__['exec']\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "s = sys\nrunner = s.modules['builtins'].exec\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import __main__\n__main__.__dict__.update({'provenance_audit': noop})",
        "from __main__ import __dict__ as scope\nscope.update({'provenance_audit': noop})",
        "runner = sys._getframe().f_builtins['exec']\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "scope = provenance_audit.__getattribute__('__globals__')\nscope.update({'provenance_audit': noop})",
        "scope = object.__getattribute__(provenance_audit, '__globals__')\nscope.update({'provenance_audit': noop})",
        "import inspect\nscope = dict(inspect.getmembers(provenance_audit))['__globals__']\nscope.update({'provenance_audit': noop})",
        "import gc\nscope = gc.get_referents(provenance_audit)[0]\nscope.update({'provenance_audit': noop})",
        "from importlib import import_module as load\nrunner = load('builtins').exec\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import pydoc\nrunner = pydoc.locate('builtins.exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "import pkgutil\nrunner = pkgutil.resolve_name('builtins:exec')\nrunner('global provenance_audit; provenance_audit = lambda frame: None')",
        "from unittest.mock import patch\npatch('__main__.provenance_audit', new=noop).start()",
    ],
    ids=[
        "exec-alias",
        "builtins-mapping-get",
        "builtins-getattribute",
        "getattr-exec",
        "builtins-copy",
        "builtins-items",
        "dynamic-setattr-name",
        "operator-setitem",
        "module-magic-setattr",
        "operator-getitem",
        "sys-modules-builtins",
        "sys-alias-modules-builtins",
        "main-module-dict",
        "main-module-dict-from-import",
        "sys-frame-builtins",
        "function-getattribute-globals",
        "object-getattribute-globals",
        "inspect-getmembers",
        "gc-get-referents",
        "importlib-alias",
        "pydoc-locate",
        "pkgutil-resolve-name",
        "unittest-mock-patch",
    ],
)
def test_mechanical_preflight_rejects_indirect_marker_rebinding(ra, rebind):
    code = f"""
import sys

def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

{rebind}
provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_module_level_provenance_fallthrough(ra):
    code = """
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
provenance_gate_failed = invalid_pair_n > 0 or discordant_n > 0
if provenance_gate_failed:
    table_rows = []
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_module_level_provenance_raise(ra):
    code = """
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
provenance_gate_failed = invalid_pair_n > 0 or discordant_n > 0
if provenance_gate_failed:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_unchanged_host_provenance_receipts(ra):
    code = """
import pandas as pd
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

receipts = []
for measured_column, count_column in declared_pairs:
    receipts.append(measurement_provenance_receipt(
        frame,
        measured_column=measured_column,
        count_column=count_column,
    ))
pd.DataFrame.from_records(receipts).to_csv('provenance.csv', index=False)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_host_provenance_receipt_decoy_does_not_authorize_custom_audit(ra):
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

measurement_provenance_receipt(
    other_frame,
    measured_column='other_measured',
    count_column='other_n',
)
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "mutation",
    ["invalid_pair_n = 0", "discordant_n = 0"],
)
def test_mechanical_preflight_rejects_module_count_rebound_after_audit(ra, mutation):
    code = f"""
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}}]
{mutation}
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("scope", ["module", "function"])
def test_mechanical_preflight_rejects_mutable_provenance_count_binding(ra, scope):
    body = """
invalid_pair_n = np.array(1)
discordant_n = np.array(0)
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
invalid_pair_n[...] = 0
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
""".strip()
    if scope == "function":
        indented = "\n".join(f"    {line}" for line in body.splitlines())
        code = f"""
import numpy as np
def provenance_audit(frame):
{indented}

provenance_audit(frame)
model.fit(frame)
"""
    else:
        code = f"""
import numpy as np
{body}
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_module_last_loop_value_guard(ra):
    code = """
checks = []
for measured, count in pairs:
    invalid_pair_n = int(measured.isna().sum())
    discordant_n = int((measured != (count > 0)).sum())
    checks.append({
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    })
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_conditionally_reached_module_guard(ra):
    code = """
if enabled:
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    if invalid_pair_n > 0 or discordant_n > 0:
        raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_decoy_module_audit_row(ra):
    code = """
real_invalid_pair_n = int(frame['measured'].isna().sum())
real_discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
real_checks = [{
    'role': 'audit_only',
    'invalid_pair_n': real_invalid_pair_n,
    'discordant_n': real_discordant_n,
}]
invalid_pair_n = int(0)
discordant_n = int(0)
decoy_checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_module_output_before_guard(ra):
    code = """
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
frame.to_csv(output_path)
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "intervening",
    [
        "reset_counts()",
        "checks.clear()",
        "alias = checks\nalias.clear()",
    ],
    ids=["helper-resets-counts", "audit-container-clear", "audit-alias-clear"],
)
def test_mechanical_preflight_rejects_module_post_audit_side_effects(ra, intervening):
    code = f"""
def reset_counts():
    global invalid_pair_n, discordant_n
    invalid_pair_n = 0
    discordant_n = 0

invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}}]
{intervening}
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_module_helper_sink_before_guard(ra):
    code = """
def publish(result):
    result.to_csv(output_path)

publish(result)
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "helper_setup",
    [
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\nr = reset\nr()",
        "def publish(value):\n    value.to_csv(output_path)\npub = publish\npub(result)",
        "def first():\n    pass\ndef second():\n    pass\nalias = first\nalias = second\nalias()",
        "a = b\nb = a",
        "def invoke(fn):\n    fn()\ndef reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\ninvoke(reset)",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\ndef chooser():\n    return reset\nr = chooser()\nr()",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\nr: object = reset\nr()",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\nr = s = reset\nr()",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\n(r,) = (reset,)\nr()",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\nr = [reset][0]\nr()",
        "def reset():\n    global invalid_pair_n, discordant_n\n    invalid_pair_n = discordant_n = 0\nr = reset if enabled else safe\nr()",
    ],
    ids=[
        "reset-alias",
        "sink-alias",
        "ambiguous-rebind",
        "alias-cycle",
        "higher-order-invoke",
        "returned-helper",
        "annotated-alias",
        "multi-target-alias",
        "unpacked-alias",
        "subscript-alias",
        "conditional-alias",
    ],
)
def test_mechanical_preflight_rejects_ambiguous_module_helper_aliases(ra, helper_setup):
    code = f"""
{helper_setup}
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}}]
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_unproven_module_append_container(ra):
    code = """
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = external_noop
checks.append({
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
})
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "extra_field",
    [
        "**override",
        "key_factory(): side_effect()",
        "'invalid_pair_n': 0",
    ],
    ids=["dict-unpack", "computed-key", "duplicate-key"],
)
def test_mechanical_preflight_rejects_ambiguous_module_audit_dict(ra, extra_field):
    code = f"""
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
    {extra_field},
}}]
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "audit_statement",
    [
        "checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n, "
        "'discordant_n': discordant_n} for _ in []]",
        "checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n, "
        "'discordant_n': discordant_n}] if enabled else []",
        "class Audit:\n    checks = [{'role': 'audit_only', "
        "'invalid_pair_n': invalid_pair_n, 'discordant_n': discordant_n}]",
    ],
    ids=["empty-list-comprehension", "conditional-expression", "class-body"],
)
def test_mechanical_preflight_rejects_nonmaterialized_module_audit_row(
    ra, audit_statement
):
    code = f"""
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
{audit_statement}
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "definition",
    [
        "def unused(value=result.to_csv(output_path)):\n    pass",
        "@publish_result(result)\ndef unused():\n    pass",
    ],
    ids=["default-argument", "decorator"],
)
def test_mechanical_preflight_rejects_module_definition_time_sink(ra, definition):
    code = f"""
{definition}
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}}]
if invalid_pair_n or discordant_n:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_module_dynamic_count_rebinding(ra):
    code = """
invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
exec('invalid_pair_n = 0; discordant_n = 0')
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_module_provenance_ignores_sink_inside_unexecuted_helper(ra):
    code = """
def unused_helper(frame):
    model.fit(frame)

invalid_pair_n = int(frame['measured'].isna().sum())
discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
checks = [{
    'role': 'audit_only',
    'invalid_pair_n': invalid_pair_n,
    'discordant_n': discordant_n,
}]
if invalid_pair_n > 0 or discordant_n > 0:
    raise RuntimeError('invalid measurement provenance')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_declines_nested_module_provenance_branch(ra):
    code = """
if 'measured' in frame.columns and 'count' in frame.columns:
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    provenance_failed = invalid_pair_n > 0 or discordant_n > 0
    if provenance_failed:
        summary['status'] = 'failed_provenance_audit'
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [
        finding.detail.get("reason")
        for finding in findings
        if finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
    ]
    repaired, repair_names = deterministic_concept_audit_repair(code, messages)
    repaired_findings = audit_mechanical_code_contracts(repaired, _figure_step(ra))

    assert repair_names == []
    assert repaired == code
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in repaired_findings
    )


def test_mechanical_preflight_allows_unrelated_static_setattr_with_marker(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

setattr(config, 'display_label', 'audit')
provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_marker_code_replacement(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

provenance_audit.__code__ = noop.__code__
provenance_audit(frame)
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "owner_binding", ["decorated", "for-target", "code-replacement"]
)
def test_mechanical_preflight_rejects_indirect_terminal_owner(ra, owner_binding):
    decorator = "@replace\n" if owner_binding == "decorated" else ""
    rebind = "for main in [evil]:\n    pass\n" if owner_binding == "for-target" else ""
    if owner_binding == "code-replacement":
        rebind = "main.__code__ = evil.__code__\n"
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

def evil():
    model.fit(frame)

{decorator}def main():
    provenance_audit(frame)

{rebind}if __name__ == '__main__':
    main()
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_decorated_returned_marker_helper(ra):
    code = """
@swallow
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('failed')
    return {'checks': checks}, failures

audit, failures = provenance_audit(frame)
if failures:
    raise RuntimeError('failed')
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_conditionally_raising_failure_collection(ra):
    code = """
def main(frame, strict_mode):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    if failures:
        write_failed_summary(checks)
        if strict_mode:
            raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame, strict_mode)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_failure_collection_mutated_after_guard(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    if failures:
        raise RuntimeError('invalid measurement provenance')
    failures.clear()
    model.fit(frame)

main(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_custom_empty_collection_constructor(ra):
    code = """
class Noop:
    def append(self, value):
        pass
    def __bool__(self):
        return False

def list():
    return Noop()

def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = list()
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    if failures:
        raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_declines_custom_returned_collection_constructor(ra):
    code = """
class Noop:
    def append(self, value):
        pass
    def __bool__(self):
        return False

def list():
    return Noop()

def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = list()
    if invalid_pair_n or discordant_n:
        failures.append('failed')
    return {'checks': checks}, failures

audit, failures = provenance_audit(frame)
model.fit(frame)
"""

    repaired, repair_names = deterministic_concept_audit_repair(
        code, ["provenance_audit_not_fail_closed"]
    )

    assert repair_names == []
    assert repaired == code


def test_mechanical_preflight_rejects_inline_terminal_guard_swallowed_by_try(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    try:
        if failures:
            raise RuntimeError('invalid measurement provenance')
    except RuntimeError:
        pass
    model.fit(frame)

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_self_raising_helper_swallowed_by_caller(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if invalid_pair_n or discordant_n:
        raise RuntimeError('invalid measurement provenance')

try:
    provenance_audit(frame)
except RuntimeError:
    pass
model.fit(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "controlled_append",
    [
        """if collect_failures:
        if invalid_pair_n or discordant_n:
            failures.append('invalid measurement provenance')""",
        """match audit_mode:
        case 'provenance':
            if invalid_pair_n or discordant_n:
                failures.append('invalid measurement provenance')""",
    ],
    ids=["conditional", "match"],
)
def test_mechanical_preflight_rejects_control_nested_full_failure_append(
    ra, controlled_append
):
    code = f"""
def main(frame, collect_failures=True, audit_mode='provenance'):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    failures = []
    {controlled_append}
    if failures:
        raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_scientific_sink_inside_terminal_guard(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if invalid_pair_n or discordant_n:
        failures.append('invalid measurement provenance')
    if failures:
        model.fit(frame)
        raise RuntimeError('invalid measurement provenance')

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_swallowed_failure_append_payload(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    try:
        if invalid_pair_n or discordant_n:
            failures.append(build_failure_message(frame))
    except Exception:
        pass
    if failures:
        raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_closes_provenance_preflight_finding(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    return {
        'fail_closed': True,
        'completed_step_allowed': False,
        'checks': checks,
    }

audit = provenance_audit(frame)
model.fit(frame)
"""
    initial = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [finding.message for finding in initial if finding.severity == "error"]
    repaired, names = deterministic_concept_audit_repair(code, messages)
    final = audit_mechanical_code_contracts(repaired, _figure_step(ra))

    assert names == ["provenance_fail_closed_guard_v1"]
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in final
    )


def test_mechanical_preflight_rejects_unbound_provenance_failure_collection(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 0,
        'discordant_n': 0,
    }]
    return {'checks': checks}

audit = provenance_audit(frame)
provenance_failures = []
for check in audit['checks']:
    if check['invalid_pair_n'] or check['discordant_n']:
        provenance_failures.append(check)
if provenance_failures:
    raise ValueError('invalid measurement provenance')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_helper_returned_provenance_failures(ra):
    code = """
def run_measurement_provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('invalid or discordant provenance pair')
    return {'checks': checks}, failures

audit, audit_failures = run_measurement_provenance_audit(frame)
if len(audit_failures) > 0:
    raise RuntimeError('measurement provenance failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_provenance_preflight_does_not_treat_required_key_set_as_audit_row(ra):
    code = """
def normalise_receipt(receipt):
    required = {
        "invalid_pair_n",
        "discordant_n",
        "audit_only",
    }
    if not required.issubset(receipt):
        raise ValueError("receipt fields missing")
    return receipt

normalise_receipt(receipt)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_provenance_preflight_defers_direct_host_receipt_to_host_gate(ra):
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    measurement_provenance_receipt,
)

def main(frame):
    receipt = measurement_provenance_receipt(
        frame,
        measured_column="signal_measured",
        count_column="signal_n",
    )
    checks = [{
        "role": "audit_only",
        "invalid_pair_n": receipt["invalid_pair_n"],
        "discordant_n": receipt["discordant_n"],
    }]
    return {"checks": checks}

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_provenance_preflight_accepts_guard_immediately_before_audit_row(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame["invalid"].sum())
    discordant_n = int(frame["discordant"].sum())
    if invalid_pair_n > 0 or discordant_n > 0:
        raise RuntimeError("invalid provenance")
    audit = {"checks": [{
        "role": "audit_only",
        "invalid_pair_n": invalid_pair_n,
        "discordant_n": discordant_n,
    }]}
    model.fit(frame)
    return audit

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_provenance_preflight_requires_post_guard_audit_row_to_be_immediate(ra):
    code = """
def main(frame):
    invalid_pair_n = int(frame["invalid"].sum())
    discordant_n = int(frame["discordant"].sum())
    if invalid_pair_n > 0 or discordant_n > 0:
        raise RuntimeError("invalid provenance")
    mutate_or_publish(frame)
    audit = {"checks": [{
        "role": "audit_only",
        "invalid_pair_n": invalid_pair_n,
        "discordant_n": discordant_n,
    }]}
    return audit

main(frame)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_extra_monotonic_failure_appends(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if missing_pair:
        failures.append('missing pair')
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('invalid or discordant pair')
    if not checks:
        failures.append('no checks')
    return {'checks': checks}, failures

audit, failures = provenance_audit(frame)
if failures:
    raise RuntimeError('failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_helper_that_only_returns_provenance_failures(ra):
    code = """
def run_measurement_provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('invalid or discordant provenance pair')
    return {'checks': checks}, failures

audit, provenance_failures = run_measurement_provenance_audit(frame)
write_audit(audit, provenance_failures)
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_guards_returned_provenance_failures_once(ra):
    code = """
def run_measurement_provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('invalid or discordant provenance pair')
    return {'checks': checks}, failures

audit, provenance_failures = run_measurement_provenance_audit(frame)
model.fit(frame)
"""
    initial = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [finding.message for finding in initial if finding.severity == "error"]

    repaired, names = deterministic_concept_audit_repair(code, messages)
    repaired_again, names_again = deterministic_concept_audit_repair(repaired, messages)
    final = audit_mechanical_code_contracts(repaired, _figure_step(ra))

    assert names == ["provenance_fail_closed_guard_v1"]
    assert names_again == []
    assert repaired_again == repaired
    module = ast.parse(repaired)
    call_assignment = module.body[-3]
    guard = module.body[-2]
    assert isinstance(call_assignment, ast.Assign)
    assert isinstance(guard, ast.If)
    assert isinstance(guard.test, ast.Name)
    assert guard.test.id == "provenance_failures"
    assert isinstance(guard.body[0], ast.Raise)
    assert isinstance(module.body[-1], ast.Expr)
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in final
    )


@pytest.mark.parametrize(
    "failure_branch",
    [
        """if invalid_pair_n == 0 and discordant_n == 0:
        pass
    else:
        failures.append('failed')""",
        """if invalid_pair_n > 0 or discordant_n > 0:
        if verbose:
            failures.append('failed')""",
        """if invalid_pair_n > 0 or discordant_n > 0:
        return {'checks': checks}, failures
        failures.append('failed')""",
        """if invalid_pair_n > 0 or discordant_n > 0:
        failures.extend(['failed'])""",
        """if mode == 'invalid_pair_n' or mode == 'discordant_n':
        failures.append('failed')""",
    ],
)
def test_mechanical_preflight_rejects_nonexact_returned_failure_grammar(
    ra, failure_branch
):
    code = f"""
def provenance_audit(frame, verbose=False, mode=''):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    failures = []
    {failure_branch}
    return {{'checks': checks}}, failures

audit, failures = provenance_audit(frame)
if failures:
    raise RuntimeError('failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "failures = []",
        "failures.clear()",
        "failures.pop()",
        "failures.remove('failed')",
        "failures[:] = []",
        "del failures[:]",
        "wipe(failures)",
        "wipe(items=failures)",
        "alias = failures\n    alias.clear()",
        "container = [failures]",
    ],
)
def test_mechanical_preflight_rejects_mutated_returned_failure_collection(ra, mutation):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('failed')
    {mutation}
    return {{'checks': checks}}, failures

audit, failures = provenance_audit(frame)
if failures:
    raise RuntimeError('failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_conditional_failure_collection_init(ra):
    code = """
def provenance_audit(frame, setup):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    if setup:
        failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('failed')
    return {'checks': checks}, failures

audit, failures = provenance_audit(frame, setup)
if failures:
    raise RuntimeError('failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "call",
    ["provenance_audit(frame)", "consume(provenance_audit(frame))"],
)
def test_mechanical_preflight_rejects_unbound_marker_calls(ra, call):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}}]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('failed')
    return {{'checks': checks}}, failures

{call}
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_cross_scope_provenance_guard_decoy(ra):
    code = """
def provenance_audit(frame):
    checks = [{'role': 'audit_only', 'invalid_pair_n': 1, 'discordant_n': 0}]
    return {'fail_closed': True, 'checks': checks}

audit = provenance_audit(frame)
def unrelated():
    audit = {'fail_closed': True}
    if audit['fail_closed']:
        raise RuntimeError('unrelated')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("nested", [False, True])
def test_mechanical_preflight_rejects_ambiguous_same_name_marker_binding(ra, nested):
    replacement = """
def provenance_audit(frame):
    return {'fail_closed': False}

audit = provenance_audit(frame)
"""
    if nested:
        replacement = """
def wrapper():
    def provenance_audit(frame):
        return {'fail_closed': False}
    return provenance_audit(frame)

audit = wrapper()
"""
    code = f"""
def provenance_audit(frame):
    checks = [{{'role': 'audit_only', 'invalid_pair_n': 1, 'discordant_n': 0}}]
    return {{'fail_closed': True, 'checks': checks}}

{replacement}
if audit['fail_closed']:
    raise RuntimeError('failed')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_return_only_decision_wrapper(ra):
    code = """
def provenance_audit(frame):
    checks = [{'role': 'audit_only', 'invalid_pair_n': 1, 'discordant_n': 0}]
    return {'fail_closed': True, 'checks': checks}

def wrapper():
    audit = provenance_audit(frame)
    if audit['fail_closed']:
        return

wrapper()
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_return_only_wrapper_guard(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('failed')
    return {'checks': checks}, failures

def wrapper():
    audit, failures = provenance_audit(frame)
    if failures:
        return

wrapper()
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize("terminal", ["return", "raise RuntimeError('failed')"])
def test_mechanical_preflight_rejects_unused_marker_helper(ra, terminal):
    code = f"""
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    audit = {{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
             'discordant_n': discordant_n}}
    if invalid_pair_n > 0 or discordant_n > 0:
        {terminal}
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_deterministic_repair_guards_every_returned_failure_call(ra):
    code = """
def provenance_audit(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    checks = [{'role': 'audit_only', 'invalid_pair_n': invalid_pair_n,
               'discordant_n': discordant_n}]
    failures = []
    if invalid_pair_n > 0 or discordant_n > 0:
        failures.append('failed')
    return {'checks': checks}, failures

audit_one, failures_one = provenance_audit(frame)
audit_two, failures_two = provenance_audit(frame)
model.fit(frame)
"""
    initial = audit_mechanical_code_contracts(code, _figure_step(ra))
    messages = [finding.message for finding in initial if finding.severity == "error"]
    repaired, names = deterministic_concept_audit_repair(code, messages)
    repaired_again, names_again = deterministic_concept_audit_repair(repaired, messages)
    final = audit_mechanical_code_contracts(repaired, _figure_step(ra))

    assert names == ["provenance_fail_closed_guard_v1"]
    assert names_again == []
    assert repaired_again == repaired
    guards = [node for node in ast.parse(repaired).body if isinstance(node, ast.If)]
    assert [guard.test.id for guard in guards] == ["failures_one", "failures_two"]
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in final
    )


def test_mechanical_preflight_rejects_unused_inline_provenance_helper(ra):
    code = """
def run(frame):
    valid_pairs = frame['measured'].notna() & frame['count'].notna()
    discordant = valid_pairs & (frame['measured'] != (frame['count'] > 0))
    audit = {
        'role': 'audit_only',
        'invalid_pair_n': int((~valid_pairs).sum()),
        'discordant_n': int(discordant.sum()),
    }
    provenance_ok = valid_pairs.all() and not discordant.any()
    if not provenance_ok:
        write_failed_summary(audit)
        return
    model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_inline_provenance_guard_without_return(ra):
    code = """
def run(frame):
    invalid_pairs = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    audit = {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pairs,
        'discordant_n': discordant_n,
    }
    audit_status_ok = invalid_pairs == 0 and discordant_n == 0
    if not audit_status_ok:
        write_failed_summary(audit)
    model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_failed_status_then_product_registration(ra):
    code = """
def run(frame):
    invalid_pair_n = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    audit = {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pair_n,
        'discordant_n': discordant_n,
    }
    if invalid_pair_n > 0 or discordant_n > 0:
        final_mask = False
        summary['status'] = 'failed_provenance_audit'
    write_analysis_cohort(frame[final_mask])
    summary['registered_outputs'] = {'artifact:analysis_cohort': 'cohort.parquet'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_each_inline_provenance_row_guarded_in_loop(ra):
    code = """
def main(frame, pairs):
    checks = []
    for measured_column, count_column in pairs:
        invalid_pair_n = int(frame[measured_column].isna().sum())
        discordant_n = int(
            (frame[measured_column] != (frame[count_column] > 0)).sum()
        )
        checks.append({
            'role': 'audit_only',
            'invalid_pair_n': invalid_pair_n,
            'discordant_n': discordant_n,
        })
        if invalid_pair_n > 0 or discordant_n > 0:
            raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

if __name__ == '__main__':
    main(frame, pairs)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_inline_loop_guard_after_continue(ra):
    code = """
def main(frame, pairs):
    checks = []
    for measured_column, count_column in pairs:
        invalid_pair_n = int(frame[measured_column].isna().sum())
        discordant_n = int(
            (frame[measured_column] != (frame[count_column] > 0)).sum()
        )
        if should_skip(measured_column):
            continue
        checks.append({
            'role': 'audit_only',
            'invalid_pair_n': invalid_pair_n,
            'discordant_n': discordant_n,
        })
        if invalid_pair_n > 0 or discordant_n > 0:
            raise RuntimeError('invalid measurement provenance')
    model.fit(frame)

if __name__ == '__main__':
    main(frame, pairs)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_conditionally_nested_provenance_return(ra):
    code = """
def run(frame, should_stop):
    invalid_pairs = int(frame['measured'].isna().sum())
    discordant_n = int((frame['measured'] != (frame['count'] > 0)).sum())
    audit = {
        'role': 'audit_only',
        'invalid_pair_n': invalid_pairs,
        'discordant_n': discordant_n,
    }
    audit_status_ok = invalid_pairs == 0 and discordant_n == 0
    if not audit_status_ok:
        write_failed_summary(audit)
        if should_stop:
            return
    model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_success_polarity_provenance_return(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 0,
        'discordant_n': 0,
    }]
    return {'completed_step_allowed': True, 'checks': checks}

audit = provenance_audit(frame)
if audit['completed_step_allowed']:
    return
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_partial_failure_guard(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    return {'fail_closed': True, 'checks': checks}

audit = provenance_audit(frame)
if audit['fail_closed'] and strict_mode:
    return
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_unreachable_provenance_return(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    return {'fail_closed': True, 'checks': checks}

audit = provenance_audit(frame)
for _ in rows:
    if audit['fail_closed']:
        continue
        return
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_nonraising_nested_provenance_termination(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    return {'fail_closed': True, 'checks': checks}

audit = provenance_audit(frame)
if audit['fail_closed']:
    if write_summary:
        return
    else:
        raise ValueError('invalid measurement provenance')
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "wrapped_guard",
    [
        """if strict_mode:
    if audit['fail_closed']:
        return""",
        """for _ in rows:
    if audit['fail_closed']:
        return""",
    ],
)
def test_mechanical_preflight_rejects_conditionally_reached_provenance_guard(
    ra, wrapped_guard
):
    code = f"""
def provenance_audit(frame):
    checks = [{{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }}]
    return {{'fail_closed': True, 'checks': checks}}

audit = provenance_audit(frame)
{wrapped_guard}
model.fit(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_provenance_guard_after_result_sink(ra):
    code = """
def provenance_audit(frame):
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 1,
        'discordant_n': 0,
    }]
    return {'fail_closed': True, 'checks': checks}

audit = provenance_audit(frame)
model.fit(frame)
write_success_summary()
if audit['fail_closed']:
    return
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_audit_not_fail_closed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_measured_only_provenance_scan(ra):
    code = """
def provenance_audit(frame):
    measured_columns = [
        column for column in frame.columns
        if str(column).endswith('_measured')
    ]
    checks = []
    for measured_column in measured_columns:
        checks.append({
            'role': 'audit_only',
            'invalid_pair_n': 0,
            'discordant_n': 0,
        })
    return {'checks': checks}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_pair_scan_not_bidirectional"
        for finding in findings
    )


def test_mechanical_preflight_accepts_bidirectional_provenance_scan(ra):
    code = """
def provenance_audit(frame):
    measured_columns = [
        column for column in frame.columns
        if str(column).endswith('_measured')
    ]
    count_columns = [
        column for column in frame.columns
        if str(column).endswith('_n')
    ]
    checks = [{
        'role': 'audit_only',
        'invalid_pair_n': 0,
        'discordant_n': 0,
    }]
    return {'checks': checks, 'counts': count_columns, 'flags': measured_columns}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_pair_scan_not_bidirectional"
        for finding in findings
    )


def test_mechanical_preflight_blocks_swallowed_reconciliation_error(ra):
    code = """
def audit_event_presence(frame):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except Exception as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    return {'status': 'checked', 'audit': result.audit}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_tuple_caught_reconciliation_error(ra):
    code = """
def audit_event_presence(frame):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except (ValueError, TypeError) as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    return {'status': 'checked', 'audit': result.audit}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_import_aliased_tuple_swallow(ra):
    code = """
from easyicu.research_agent.methods.source_status import (
    reconcile_binary_event_presence as reconcile,
)

def audit_event_presence(frame):
    try:
        result = reconcile(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except (ValueError, TypeError) as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    return {'status': 'checked', 'audit': result.audit}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_conditionally_reraised_reconciliation_error(ra):
    code = """
def audit_event_presence(frame, strict):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except ValueError as exc:
        if strict:
            raise
        return {'status': 'unavailable', 'reason': str(exc)}
    return {'status': 'checked', 'audit': result.audit}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_unconditional_bare_reraise(ra):
    code = """
def audit_event_presence(frame):
    try:
        return reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except (ValueError, TypeError):
        raise
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_swallowed_descriptive_input_error(ra):
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    DescriptiveInputError,
    measurement_provenance_receipt,
)

def audit_measurement(frame):
    try:
        return measurement_provenance_receipt(
            frame,
            measured_column=measured_column,
            count_column=count_column,
        )
    except DescriptiveInputError:
        return {'status': 'unavailable'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "host_validation_helper_error_swallowed"
        and finding.detail.get("helper_names") == ["measurement_provenance_receipt"]
        for finding in findings
    )


def test_mechanical_preflight_accepts_descriptive_input_error_reraise(ra):
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    DescriptiveInputError,
    strict_numeric_input,
)

def summarize(values):
    try:
        return strict_numeric_input(values)
    except DescriptiveInputError:
        raise
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "host_validation_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_ignores_custom_helper_with_same_name(ra):
    code = """
def strict_numeric_input(values):
    return values

def summarize(values):
    try:
        return strict_numeric_input(values)
    except ValueError:
        return None
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "host_validation_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_contextlib_suppressed_host_helper(ra):
    code = """
import contextlib
from easyicu.research_agent.methods.descriptive_inputs import (
    DescriptiveInputError,
    strict_numeric_input,
)

def summarize(values):
    with contextlib.suppress(DescriptiveInputError):
        return strict_numeric_input(values)
    return None
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "host_validation_helper_error_swallowed"
        and finding.detail.get("helper_names") == ["strict_numeric_input"]
        for finding in findings
    )


def test_mechanical_preflight_blocks_reraise_suppressed_by_finally_return(ra):
    code = """
def audit_event_presence(frame):
    try:
        return reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except (ValueError, TypeError):
        raise
    finally:
        return {'status': 'unavailable'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


@pytest.mark.parametrize(
    "control_flow", ["return {'status': 'unavailable'}", "break", "continue"]
)
def test_mechanical_preflight_blocks_finally_control_flow_without_except(
    ra, control_flow
):
    if control_flow in {"break", "continue"}:
        body = f"""
def audit_event_presence(frame):
    while True:
        try:
            reconcile_binary_event_presence(
                frame,
                count_column=count_column,
                measured_column=measured_column,
                representative_column=representative_column,
            )
        finally:
            {control_flow}
"""
    else:
        body = f"""
def audit_event_presence(frame):
    try:
        reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    finally:
        {control_flow}
"""

    findings = audit_mechanical_code_contracts(body, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


@pytest.mark.parametrize("call_region", ["else", "handler"])
def test_mechanical_preflight_blocks_finally_suppression_after_try_body(
    ra, call_region
):
    if call_region == "else":
        region = """
    else:
        reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
"""
        code = f"""
def audit_event_presence(frame):
    try:
        prepare_audit()
    except LookupError:
        raise
{region}
    finally:
        return {{'status': 'unavailable'}}
"""
    else:
        region = """
    except LookupError:
        reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
"""
        code = f"""
def audit_event_presence(frame):
    try:
        prepare_audit()
{region}
    finally:
        return {{'status': 'unavailable'}}
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_simple_reconciliation_alias_in_finally(ra):
    code = """
def audit_event_presence(frame):
    reconcile = reconcile_binary_event_presence
    try:
        reconcile(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    finally:
        return {'status': 'unavailable'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_ignores_unexecuted_nested_reconciliation_body(ra):
    code = """
def audit_event_presence(frame):
    try:
        def deferred_audit():
            return reconcile_binary_event_presence(
                frame,
                count_column=count_column,
                measured_column=measured_column,
                representative_column=representative_column,
            )
        prepare_audit()
    except ValueError:
        return {'status': 'unavailable'}
    finally:
        cleanup_temporary_files()
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_does_not_borrow_alias_from_nested_scope(ra):
    code = """
def audit_event_presence(frame):
    reconcile = safe_lookup
    def deferred_audit():
        reconcile = reconcile_binary_event_presence
        return reconcile(frame)
    try:
        reconcile(frame)
    except ValueError:
        return {'status': 'lookup_unavailable'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_finally_cleanup_that_propagates(ra):
    code = """
def audit_event_presence(frame):
    try:
        return reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    finally:
        cleanup_temporary_files()
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_accepts_break_inside_finally_local_loop(ra):
    code = """
def audit_event_presence(frame):
    try:
        return reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    finally:
        for path in temporary_files:
            cleanup(path)
            break
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_ignores_lookup_only_handler(ra):
    code = """
def audit_event_presence(frame):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
        return lookup[result.row_status.name]
    except KeyError:
        return {'status': 'lookup_unavailable'}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_replaced_reconciliation_error(ra):
    code = """
def audit_event_presence(frame):
    try:
        return reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except ValueError as exc:
        raise RuntimeError('declared provenance triad is invalid') from exc
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_rejects_successful_exit_for_host_validation_error(ra):
    code = """
from easyicu.research_agent.methods.descriptive_inputs import (
    DescriptiveInputError,
    measurement_provenance_receipt,
)

def audit_measurement(frame):
    try:
        return measurement_provenance_receipt(
            frame,
            measured_column=measured_column,
            count_column=count_column,
        )
    except DescriptiveInputError:
        raise SystemExit(0)
"""

    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "host_validation_helper_error_swallowed"
        for finding in findings
    )


def test_deterministic_repair_re_raises_swallowed_reconciliation_error(ra):
    code = """
def audit_event_presence(frame):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except Exception as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    return result.audit
"""
    repaired, names = deterministic_concept_audit_repair(
        code, ["provenance_helper_error_swallowed"]
    )

    assert names == ["provenance_helper_reraise_v1"]
    assert "_easyicu_provenance_helper_reraise_v1" in repaired
    findings = audit_mechanical_code_contracts(repaired, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_deterministic_repair_makes_conditional_tuple_handler_fail_closed(ra):
    code = """
def audit_event_presence(frame, strict):
    try:
        result = reconcile_binary_event_presence(
            frame,
            count_column=count_column,
            measured_column=measured_column,
            representative_column=representative_column,
        )
    except (ValueError, TypeError) as exc:
        if strict:
            raise
        return {'status': 'unavailable', 'reason': str(exc)}
    return result.audit
"""
    repaired, names = deterministic_concept_audit_repair(
        code, ["provenance_helper_error_swallowed"]
    )

    assert names == ["provenance_helper_reraise_v1"]
    assert "_easyicu_provenance_helper_reraise_v1" in repaired
    findings = audit_mechanical_code_contracts(repaired, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
        for finding in findings
    )


def test_mechanical_preflight_blocks_double_first_time_companion(ra):
    code = """
def timing_audit(frame, candidates):
    rows = []
    for candidate in candidates:
        time_column = f"{candidate}_first_time"
        rows.append((frame[candidate], time_column))
    return rows

candidate_covariates = ["age", "gcs_first", "lact_first"]
timing_candidates = [value for value in candidate_covariates if value != "age"]
timing_audit(frame, timing_candidates)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "double_first_time_companion_suffix"
        for finding in findings
    )


def test_mechanical_preflight_accepts_stem_to_first_time_companion(ra):
    code = """
def timing_audit(frame, candidates):
    return [f"{candidate}_first_time" for candidate in candidates]

timing_candidates = ["gcs", "lact"]
timing_audit(frame, timing_candidates)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "double_first_time_companion_suffix"
        for finding in findings
    )


def test_mechanical_preflight_blocks_unused_authoritative_exposure(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=[
            "artifact:quality_checked_analysis_data",
            "artifact:primary_exposure_definition",
        ],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
def main():
    typed = load_typed_inputs()
    exposure_definition = typed.get('artifact:primary_exposure_definition')
    exposure_col = 'candidate_event_max'
    return frame[exposure_col].mean()
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_unused"
        for finding in findings
    )


def test_mechanical_preflight_does_not_count_print_as_exposure_consumption(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
def main():
    typed = load_typed_inputs()
    exposure_definition = typed['artifact:primary_exposure_definition']
    print(exposure_definition)
    return {'status': 'ok'}
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_unused"
        for finding in findings
    )


def test_mechanical_preflight_requires_resolved_exposure_to_reach_result(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
def main():
    typed = load_typed_inputs()
    exposure_definition = typed['artifact:primary_exposure_definition']
    exposure_col = resolve_declared_exposure(frame, exposure_definition)
    print(exposure_col)
    return {'status': 'ok'}
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_unused"
        for finding in findings
    )


def test_mechanical_preflight_accepts_consumed_authoritative_exposure(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
def main():
    typed = load_typed_inputs()
    exposure_definition = typed.get('artifact:primary_exposure_definition')
    exposure_col = resolve_declared_exposure(frame, exposure_definition)
    return frame[exposure_col].mean()
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_unused"
        for finding in findings
    )


def test_mechanical_preflight_does_not_treat_requested_product_list_as_binding(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
def load_typed_inputs():
    requested = ['artifact:primary_exposure_definition']
    return load_requested_products(requested)

def main():
    typed = load_typed_inputs()
    exposure_definition = typed.get('artifact:primary_exposure_definition')
    exposure_col = resolve_declared_exposure(frame, exposure_definition)
    return frame[exposure_col].mean()
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_unused"
        for finding in findings
    )


def test_mechanical_preflight_blocks_constructed_authoritative_exposure_fallback(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
exposure_definition = typed.get('artifact:primary_exposure_definition')
try:
    resolved = resolve_exposure_definition(exposure_definition, frame)
except RuntimeError:
    resolved = {
        'exposure_column': 'candidate_event',
        'source_concept': 'candidate',
        'role': 'intervention',
    }
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_fallback"
        for finding in findings
    )


def test_mechanical_preflight_accepts_fail_closed_authoritative_exposure_binding(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
exposure_definition = typed.get('artifact:primary_exposure_definition')
try:
    resolved = resolve_exposure_definition(exposure_definition, frame)
except RuntimeError as exc:
    raise RuntimeError('authoritative exposure unavailable') from exc
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "authoritative_primary_exposure_fallback"
        for finding in findings
    )


def test_mechanical_preflight_blocks_reconciliation_of_finalized_exposure_table(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
exposure_definition = typed['artifact:primary_exposure_definition']
if isinstance(exposure_definition, pd.DataFrame):
    finalized = exposure_definition[selected_exposure]
    helper_result = reconcile_binary_event_presence(
        frame,
        count_column=registered_count,
        measured_column=registered_measured,
        representative_column=registered_representative,
    )
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "finalized_exposure_reconciliation_fallback"
        for finding in findings
    )


def test_mechanical_preflight_accepts_direct_finalized_exposure_binding(ra):
    step = ra.AnalysisStep(
        step_id="diagnostics",
        intent="Run planner-owned exposure diagnostics.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["table:diagnostics"],
        method="diagnostic_analysis",
    )
    code = """
exposure_definition = typed['artifact:primary_exposure_definition']
if isinstance(exposure_definition, pd.DataFrame):
    finalized = pd.to_numeric(
        exposure_definition[selected_exposure], errors='coerce'
    )
    if finalized.isna().any() or not finalized.isin([0, 1]).all():
        raise RuntimeError('invalid finalized exposure')
    treatment = finalized.astype(int)
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "finalized_exposure_reconciliation_fallback"
        for finding in findings
    )


def test_mechanical_preflight_blocks_erasure_of_supported_dataframe_artifact(ra):
    step = ra.AnalysisStep(
        step_id="assignment",
        intent="Fit the Planner-owned assignment model.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["artifact:assignment_model"],
        method="confounder_selection_and_propensity_model",
    )
    code = """
def resolve_exposure(exposure_definition):
    if isinstance(exposure_definition, pd.DataFrame):
        return exposure_definition[selected_exposure]
    return None

exposure_definition = typed.get('artifact:primary_exposure_definition')
if not isinstance(exposure_definition, (dict, list, str)):
    exposure_definition = {}
resolved = resolve_exposure(exposure_definition)
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert any(
        finding.detail
        and finding.detail.get("reason") == "typed_dataframe_artifact_erased"
        for finding in findings
    )


def test_mechanical_preflight_accepts_preserved_dataframe_artifact(ra):
    step = ra.AnalysisStep(
        step_id="assignment",
        intent="Fit the Planner-owned assignment model.",
        inputs=["artifact:primary_exposure_definition"],
        expected_outputs=["artifact:assignment_model"],
        method="confounder_selection_and_propensity_model",
    )
    code = """
def resolve_exposure(exposure_definition):
    if isinstance(exposure_definition, pd.DataFrame):
        return exposure_definition[selected_exposure]
    if isinstance(exposure_definition, dict):
        return exposure_definition['exposure_column']
    raise RuntimeError('unsupported typed exposure artifact')

exposure_definition = typed.get('artifact:primary_exposure_definition')
resolved = resolve_exposure(exposure_definition)
"""
    findings = audit_mechanical_code_contracts(code, step)

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "typed_dataframe_artifact_erased"
        for finding in findings
    )


def test_mechanical_preflight_blocks_undefined_direct_helper_call(ra):
    code = """
def main(frame):
    return resolve_declared_product(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    finding = next(
        item
        for item in findings
        if item.detail and item.detail.get("reason") == "undefined_helper_call"
    )
    assert finding.detail["calls"] == [{"name": "resolve_declared_product", "line": 3}]


def test_mechanical_preflight_accepts_defined_and_imported_helper_calls(ra):
    code = """
from pathlib import Path

def resolve_declared_product(frame):
    return frame

def main(frame):
    return Path('output.csv'), resolve_declared_product(frame), len(frame)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail and finding.detail.get("reason") == "undefined_helper_call"
        for finding in findings
    )


def test_mechanical_preflight_blocks_invalid_local_helper_calls(ra):
    code = """
def resolve_finalized_exposure(definition, frame):
    return definition, frame

def validate_domains(frame, covariates):
    return frame, covariates

resolved = resolve_finalized_exposure(
    definition,
    frame,
    selected_exposure_column=selected_exposure,
)
audit = validate_domains(frame, covariates, metadata)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    finding = next(
        item
        for item in findings
        if item.detail and item.detail.get("reason") == "invalid_local_helper_call"
    )
    assert [call["name"] for call in finding.detail["calls"]] == [
        "resolve_finalized_exposure",
        "validate_domains",
    ]


def test_mechanical_preflight_accepts_flexible_local_helper_signatures(ra):
    code = """
def resolve(definition, frame=None, *args, selected=None, **kwargs):
    return definition

resolved = resolve(
    definition,
    frame,
    extra,
    selected=selected_exposure,
    provenance=metadata,
)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail and finding.detail.get("reason") == "invalid_local_helper_call"
        for finding in findings
    )


def test_mechanical_preflight_blocks_branch_local_read_after_merge(ra):
    code = """
def consume(definition):
    if not isinstance(definition, pd.DataFrame):
        context_source_concept = load_raw_metadata(definition)
    summary = {'source_concept': context_source_concept}
    return summary
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    finding = next(
        item
        for item in findings
        if item.detail and item.detail.get("reason") == "branch_local_unbound"
    )
    assert finding.detail["name"] == "context_source_concept"
    assert finding.detail["first_use_line"] == 5


def test_mechanical_preflight_blocks_local_read_before_later_initialization(ra):
    code = """
def build_table(provenance_failed):
    if provenance_failed:
        write_failed_table(table_rows)
        return
    table_rows = []
    table_rows.append({'label': 'ok'})
    return table_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        item.detail
        and item.detail.get("reason") == "local_read_before_assignment"
        and item.detail.get("name") == "table_rows"
        for item in findings
    )


def test_mechanical_preflight_accepts_local_initialized_before_failure_branch(ra):
    code = """
def build_table(provenance_failed):
    table_rows = []
    if provenance_failed:
        write_failed_table(table_rows)
        return
    table_rows.append({'label': 'ok'})
    return table_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        item.detail and item.detail.get("reason") == "local_read_before_assignment"
        for item in findings
    )


def test_mechanical_preflight_accepts_local_assigned_in_both_branches(ra):
    code = """
def consume(definition):
    if isinstance(definition, pd.DataFrame):
        source_concept = load_finalized_metadata(definition)
    else:
        source_concept = load_raw_metadata(definition)
    return {'source_concept': source_concept}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail and finding.detail.get("reason") == "branch_local_unbound"
        for finding in findings
    )


def test_mechanical_preflight_accepts_missing_branch_that_terminates(ra):
    code = """
def consume(definition):
    if isinstance(definition, pd.DataFrame):
        source_concept = load_finalized_metadata(definition)
    else:
        raise RuntimeError('unsupported input form')
    return {'source_concept': source_concept}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail and finding.detail.get("reason") == "branch_local_unbound"
        for finding in findings
    )


def test_mechanical_preflight_blocks_try_local_read_after_continuing_handler(ra):
    code = """
def build_receipt(frame):
    try:
        audit_rows = compute_audit(frame)
    except ValueError:
        record_failure()
    return audit_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


def test_mechanical_preflight_accepts_try_local_initialized_before_handler(ra):
    code = """
def build_receipt(frame):
    audit_rows = None
    try:
        audit_rows = compute_audit(frame)
    except ValueError:
        record_failure()
    if audit_rows is None:
        return failed_receipt()
    return audit_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


def test_mechanical_preflight_accepts_try_local_when_handler_terminates(ra):
    code = """
def build_receipt(frame):
    try:
        audit_rows = compute_audit(frame)
    except ValueError:
        raise
    return audit_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


@pytest.mark.parametrize(
    "handler_body",
    [
        "print(audit_rows)",
        "audit_rows += 1",
    ],
)
def test_mechanical_preflight_blocks_try_local_read_inside_handler(ra, handler_body):
    code = f"""
def build_receipt(frame):
    try:
        audit_rows = compute_audit(frame)
    except ValueError:
        {handler_body}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        and finding.detail.get("scope") == "build_receipt"
        for finding in findings
    )


def test_mechanical_preflight_blocks_try_local_read_inside_finally(ra):
    code = """
def build_receipt(frame):
    try:
        audit_rows = compute_audit(frame)
    except ValueError:
        pass
    finally:
        print(audit_rows)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


def test_mechanical_preflight_blocks_exception_alias_read_after_handler(ra):
    code = """
def build_receipt(frame):
    try:
        compute_audit(frame)
    except ValueError as audit_error:
        log(audit_error)
    return audit_error
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_error"
        for finding in findings
    )


def test_mechanical_preflight_accepts_rebound_exception_alias_in_later_handler(ra):
    code = """
def fit_model(frame):
    try:
        fit_primary(frame)
    except Exception as exc:
        log_primary_failure(str(exc))

    try:
        fit_fallback(frame)
    except Exception as exc:
        log_fallback_failure(str(exc))
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


def test_mechanical_preflight_blocks_exception_alias_read_between_handlers(ra):
    code = """
def fit_model(frame):
    try:
        fit_primary(frame)
    except Exception as exc:
        log_primary_failure(str(exc))

    log_between_attempts(exc)

    try:
        fit_fallback(frame)
    except Exception as exc:
        log_fallback_failure(str(exc))
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


def test_mechanical_preflight_blocks_rebound_exception_alias_after_later_handler(ra):
    code = """
def fit_model(frame):
    try:
        fit_primary(frame)
    except Exception as exc:
        log_primary_failure(str(exc))

    try:
        fit_fallback(frame)
    except Exception as exc:
        log_fallback_failure(str(exc))

    print(exc)
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


@pytest.mark.parametrize(
    "try_suffix",
    [
        """else:
        log(exc)""",
        """finally:
        log(exc)""",
    ],
)
def test_mechanical_preflight_blocks_exception_alias_in_try_sibling_suite(
    ra, try_suffix
):
    code = f"""
def fit_model(frame):
    try:
        fit_primary(frame)
    except Exception as exc:
        log(str(exc))
    {try_suffix}
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


def test_mechanical_preflight_blocks_exception_alias_in_handler_type(ra):
    code = """
def fit_model(frame):
    try:
        fit_primary(frame)
    except choose_exception_type(exc) as exc:
        log(str(exc))
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


def test_mechanical_preflight_accepts_outer_rebound_alias_in_nested_handler(ra):
    code = """
def fit_model(frame):
    try:
        fit_primary(frame)
    except Exception as exc:
        try:
            log(str(exc))
        except LoggingError:
            log_fallback(str(exc))
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))
    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "exc"
        for finding in findings
    )


def test_mechanical_preflight_accepts_nonthrowing_try_prefix_assignment(ra):
    code = """
def build_receipt(frame):
    try:
        audit_rows = 1
        compute_audit(frame)
    except ValueError:
        log_failure()
    return audit_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


@pytest.mark.parametrize("initializer", ["x: int", "x = {[]}", "x = {[]: 1}"])
def test_mechanical_preflight_does_not_treat_failing_prefix_as_assignment(
    ra, initializer
):
    code = f"""
def build_receipt():
    try:
        {initializer}
    except Exception:
        pass
    return x
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "x"
        for finding in findings
    )


def test_mechanical_preflight_does_not_treat_finally_annotation_as_binding(ra):
    code = """
def build_receipt():
    try:
        may_fail()
    except Exception:
        pass
    finally:
        x: int
    return x
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "x"
        for finding in findings
    )


@pytest.mark.parametrize(
    "handler_body",
    [
        "for x in [1, 2]:\n            consume(x)\n        raise",
        "with resource() as x:\n            consume(x)\n        raise",
    ],
)
def test_mechanical_preflight_accepts_handler_binding_before_body_use(ra, handler_body):
    code = f"""
def build_receipt():
    try:
        x = may_fail()
    except Exception:
        {handler_body}
    return x
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "x"
        and "inside an exception handler" in finding.message
        for finding in findings
    )


def test_mechanical_preflight_does_not_claim_nested_try_store_is_straight_line(ra):
    code = """
def build_receipt(frame, use_primary):
    try:
        if use_primary:
            audit_rows = primary_audit(frame)
        else:
            audit_rows = secondary_audit(frame)
    except ValueError:
        raise
    return audit_rows
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "branch_local_unbound"
        and finding.detail.get("name") == "audit_rows"
        for finding in findings
    )


def test_mechanical_preflight_blocks_lossy_ordinal_rounding(ra):
    code = """
def summarize(values, metadata):
    is_ordinal = bool(metadata.get('is_ordinal', False))
    if is_ordinal:
        levels = values.round().astype(int)
        return levels
    return values
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    finding = next(
        item
        for item in findings
        if item.detail and item.detail.get("reason") == "lossy_ordinal_rounding"
    )
    assert finding.detail["lines"] == [5]


def test_mechanical_preflight_accepts_exact_ordinal_level_validation(ra):
    code = """
def summarize(values, metadata):
    is_ordinal = bool(metadata.get('is_ordinal', False))
    if is_ordinal:
        invalid = values.notna() & ~values.isin(metadata['levels'])
        if invalid.any():
            raise ValueError('invalid ordinal level')
        return values.astype('Int64')
    return values
"""
    findings = audit_mechanical_code_contracts(code, _figure_step(ra))

    assert not any(
        finding.detail and finding.detail.get("reason") == "lossy_ordinal_rounding"
        for finding in findings
    )


def test_llm_concept_audit_cache_reuses_identical_digest(tmp_path, ra):
    context = _context(ra)
    step = _figure_step(ra)
    cache = LLMConceptAuditCache(tmp_path)
    cache_identity = {
        "environment_sha256": "environment-a",
        "auditor_identity": "auditor:model-a",
    }
    key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        **cache_identity,
    )
    finding = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="semantic issue",
        detail={"line": 2},
    )

    assert cache.get(key) is None
    cache.put(key, [finding])
    cached = LLMConceptAuditCache(tmp_path).get(key)

    assert cached is not None
    assert [item.model_dump() for item in cached] == [finding.model_dump()]
    changed_key = cache.key(
        context=context,
        step=step,
        script_text="import json\n",
        audit_prompt="auditor prompt v1",
        **cache_identity,
    )
    assert changed_key != key
    assert cache.get(changed_key) is None

    changed_prompt_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v2",
        **cache_identity,
    )
    assert changed_prompt_key != key
    assert cache.get(changed_prompt_key) is None

    changed_authority_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        **cache_identity,
        authority_bindings={
            "artifact:primary_exposure_definition": {"sha256": "changed"}
        },
    )
    assert changed_authority_key != key
    assert cache.get(changed_authority_key) is None

    changed_validator_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        **cache_identity,
        validator_implementation_sha256="b" * 64,
    )
    assert changed_validator_key != key
    assert cache.get(changed_validator_key) is None

    timestamp_only_key = cache.key(
        context=context.model_copy(
            update={"created_at": context.created_at + timedelta(hours=1)}
        ),
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        **cache_identity,
    )
    assert timestamp_only_key == key

    changed_environment_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        environment_sha256="environment-b",
        auditor_identity=cache_identity["auditor_identity"],
    )
    assert changed_environment_key != key
    assert cache.get(changed_environment_key) is None

    changed_auditor_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        environment_sha256=cache_identity["environment_sha256"],
        auditor_identity="auditor:model-b",
    )
    assert changed_auditor_key != key
    assert cache.get(changed_auditor_key) is None


@pytest.mark.parametrize(
    "issue_code",
    [
        "llm_concept_audit_provider_failure",
        "llm_concept_audit_response_invalid",
    ],
)
def test_llm_concept_audit_cache_does_not_persist_transient_failures(
    tmp_path,
    ra,
    issue_code,
):
    cache = LLMConceptAuditCache(tmp_path)
    key = cache.key(
        context=_context(ra),
        step=_figure_step(ra),
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
        environment_sha256="environment-a",
        auditor_identity="auditor:model-a",
    )
    failure = ValidationFinding(
        validator="llm_concept_auditor",
        severity="error",
        message="The audit did not produce a reusable semantic result.",
        detail={"issue_code": issue_code},
    )

    cache.put(key, [failure])

    assert cache.get(key) is None
    assert not cache.path.exists()
