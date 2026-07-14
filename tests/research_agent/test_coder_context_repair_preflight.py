from __future__ import annotations

import json

from easyicu.research_agent.agents import CoderAgent, _looks_like_python_script
from easyicu.research_agent.code_preflight import audit_mechanical_code_contracts
from easyicu.research_agent.coder_context import (
    coder_guide_for_step,
    scoped_coder_context,
)
from easyicu.research_agent.concept_audit_cache import LLMConceptAuditCache
from easyicu.research_agent.pipeline_resume import _looks_like_generated_python
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


def test_figure_coder_guide_excludes_unrelated_method_families(ra):
    from easyicu.research_agent.agents import _CODER_GUIDE

    guide = coder_guide_for_step(_CODER_GUIDE, _figure_step(ra))

    assert "For rendering-only figure steps" in guide
    assert "Use matplotlib's \"Agg\" backend" in guide
    assert "TABLE-ONE / DESCRIPTIVE SUMMARIES:" not in guide
    assert "ROBUSTNESS:" not in guide
    assert len(guide) < len(_CODER_GUIDE) * 0.7


def test_coder_repair_applies_minimal_patch_without_full_rewrite(ra):
    patch = json.dumps(
        {
            "format": "easyicu.code_patch/1",
            "edits": [
                {"old": "value = 1", "new": "value = 2", "expected_count": 1}
            ],
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
    repaired = CoderAgent(llm).repair(
        context=_context(ra),
        step=_figure_step(ra),
        code="import os\nvalue = 1\n",
        run_log="ERROR: local value is invalid",
    )

    assert repaired.endswith("value = 3")
    assert len(llm.calls) == 2
    assert "FULL-REWRITE FALLBACK" in llm.calls[1][0][-1].content


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
        finding.detail and finding.detail.get("reason") == "structural_accounting_filter"
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
        finding.detail and finding.detail.get("reason") == "structural_accounting_filter"
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


def test_llm_concept_audit_cache_reuses_identical_digest(tmp_path, ra):
    context = _context(ra)
    step = _figure_step(ra)
    cache = LLMConceptAuditCache(tmp_path)
    key = cache.key(context=context, step=step, script_text="import os\n")
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
    changed_key = cache.key(context=context, step=step, script_text="import json\n")
    assert changed_key != key
    assert cache.get(changed_key) is None
