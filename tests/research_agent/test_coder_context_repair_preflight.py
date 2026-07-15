from __future__ import annotations

from datetime import timedelta
import json

import pytest

from easyicu.research_agent.agents import CoderAgent, _looks_like_python_script
from easyicu.research_agent.code_preflight import audit_mechanical_code_contracts
from easyicu.research_agent.code_repair import deterministic_concept_audit_repair
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


def test_figure_coder_guide_excludes_unrelated_method_families(ra):
    from easyicu.research_agent.agents import _CODER_GUIDE

    guide = coder_guide_for_step(_CODER_GUIDE, _figure_step(ra))

    assert "For rendering-only figure steps" in guide
    assert 'Use matplotlib\'s "Agg" backend' in guide
    assert "TABLE-ONE / DESCRIPTIVE SUMMARIES:" not in guide
    assert "ROBUSTNESS:" not in guide
    assert len(guide) < len(_CODER_GUIDE) * 0.7


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


def test_mechanical_preflight_accepts_terminating_provenance_failure_collection(ra):
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

    assert not any(
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


def test_mechanical_preflight_accepts_re_raised_reconciliation_error(ra):
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

    assert not any(
        finding.detail
        and finding.detail.get("reason") == "provenance_helper_error_swallowed"
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
        finding.detail
        and finding.detail.get("reason") == "invalid_local_helper_call"
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
    key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
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
    )
    assert changed_key != key
    assert cache.get(changed_key) is None

    changed_prompt_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v2",
    )
    assert changed_prompt_key != key
    assert cache.get(changed_prompt_key) is None

    changed_authority_key = cache.key(
        context=context,
        step=step,
        script_text="import os\n",
        audit_prompt="auditor prompt v1",
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
    )
    assert timestamp_only_key == key
