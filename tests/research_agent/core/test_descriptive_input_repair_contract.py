"""Generated analyses can repair input coding without transporting row values."""

from __future__ import annotations

import json
import traceback
from threading import RLock
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.agents.core import CoderAgent
from easyicu.research_agent.authority import diagnostic_envelope as diagnostics
from easyicu.research_agent.methods import descriptive_inputs as inputs
from easyicu.research_agent.providers.mocks import ExternalCaptureMockLLMClient
from easyicu.research_agent.repairs.patch import PATCH_FORMAT
from easyicu.research_agent.repairs.reasons import RepairPromptAuthority
from easyicu.research_agent.research_context.typed import resolved_raw_input_contracts
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context(modifier="age_group", binary_exposure=False):
    return ResearchContext(
        research_question="Compare an exposure association within declared strata.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=24),
        primary_exposure="exposure", target_outcome="event",
        variables=[
            ConceptDescriptor(name="exposure", role="lab", dtype="bool" if binary_exposure else "float64"),
            ConceptDescriptor(name="event", role="outcome", dtype="bool"),
            ConceptDescriptor(name=modifier, role="demographic", dtype="str" if modifier == "age_group" else "bool"),
        ],
    )


def _step(modifier="age_group"):
    return AnalysisStep(
        step_id="modifier_analysis", intent="Estimate association and interaction within declared strata.",
        inputs=["exposure", "event", modifier],
        expected_outputs=["table:effect_modification_estimates"],
        method="prespecified_effect_modification_interaction",
        scientific_action_id="association.effect_modification",
    )


def _failure_log(series):
    try:
        inputs.strict_numeric_input(series)
    except inputs.DescriptiveInputError:
        return traceback.format_exc()
    raise AssertionError("numeric helper accepted an invalid input")


@pytest.mark.parametrize("values,reason", [
    ([True, False], "boolean_not_numeric"),
    ([1, "PRIVATE_CELL"], "numeric_conversion_loss"),
    ([1, float("inf")], "nonfinite_numeric_input"),
    ([pd.Timestamp("2024-01-01")], "semantic_numeric_input"),
])
def test_numeric_failure_has_typed_column_reason_without_values(values, reason):
    source = pd.Series(values, name="event")
    with pytest.raises(inputs.DescriptiveInputError) as caught:
        inputs.strict_numeric_input(source)
    error = caught.value
    assert error.column == "event"
    assert reason in error.reason_codes
    log = _failure_log(source)
    assert "easyicu.descriptive_input_error/1" in log
    assert "PRIVATE_CELL" not in log


@pytest.mark.parametrize("dtype", ["bool", "boolean", "object"])
def test_binary_helper_preserves_boolean_coding_index_and_missingness(dtype):
    values = [True, False] if dtype == "bool" else [True, False, None]
    source = pd.Series(values, dtype=dtype, name="event", index=range(40, 40 + len(values)))
    result = inputs.strict_binary_input(source)
    assert result.values.index.equals(source.index)
    assert result.values.name == source.name
    assert result.values.iloc[:2].tolist() == [1.0, 0.0]
    assert result.values.isna().equals(source.isna())
    with pytest.raises(inputs.DescriptiveInputError):
        inputs.strict_numeric_input(source)


@pytest.mark.parametrize("bad", [2, 0.5, "PRIVATE_CELL", float("inf"), pd.Timestamp("2024-01-01")])
def test_binary_helper_refuses_invalid_values_with_typed_receipt(bad):
    with pytest.raises(inputs.DescriptiveInputError) as caught:
        inputs.strict_binary_input(pd.Series([True, bad, None], name="event"))
    assert caught.value.column == "event"
    assert caught.value.reason_codes
    assert "PRIVATE_CELL" not in str(caught.value)


def test_typed_materialized_manifest_preserves_dtype_role_and_boolean_coding(tmp_path):
    from easyicu.research_agent.cohort.materializer import materialize_to_parquet
    from easyicu.research_agent.research_context.builder import build_research_context
    from tests.support.typed_export import typed_export

    paths = materialize_to_parquet(
        tmp_path / "materialized", data_path=typed_export(tmp_path / "export"),
        database="miiv", static_concepts=("age",),
        feature_concepts=("lact", "mech_vent"), outcome_concepts=("death",),
    )
    context = build_research_context(
        research_question="Compare a declared exposure association within strata.",
        cohort=paths["parquet"], cohort_name="synthetic", database="miiv",
        target_outcome="death", primary_exposure="lact_max",
        id_columns=("stay_id",), outcome_columns=("death",),
    )
    context = context.model_copy(update={"variables": [
        variable.model_copy(update={"dtype": "bool"}) if variable.name == "death" else variable
        for variable in context.variables
    ]})
    contracts = resolved_raw_input_contracts(context, ["age", "death"])["contracts"]
    variables = {variable.name: variable for variable in context.variables}
    for name in contracts:
        assert contracts[name]["dtype"] == variables[name].dtype
        assert contracts[name]["role"] == variables[name].role.value
    assert contracts["death"]["boolean_numeric_encoding"] == {"false": 0, "true": 1}
    assert "boolean_numeric_encoding" not in contracts["age"]


def test_legacy_manifest_and_diagnostics_keep_only_declared_type_coordinates():
    context, step = _context(), _step()
    contracts = resolved_raw_input_contracts(context, step.inputs)["contracts"]
    assert contracts["event"]["role"] == "outcome"
    assert contracts["event"]["boolean_numeric_encoding"] == {"false": 0, "true": 1}
    assert "boolean_numeric_encoding" not in contracts["exposure"]
    payload = {"schema_version": "easyicu.descriptive_input_error/1", "column": "event",
               "reason_codes": ["boolean_not_numeric"], "dtype": "PRIVATE_DTYPE",
               "audit": {"numeric_n": 912345}, "values": ["PRIVATE_VALUE"]}
    log = inputs.DESCRIPTIVE_INPUT_MARKER + json.dumps(payload)
    projected = diagnostics.descriptive_input_diagnostics(log, context=context, step=step)
    assert projected[0]["dtype"] == "bool"
    assert "PRIVATE" not in json.dumps(projected)
    assert "912345" not in json.dumps(projected)
    assert diagnostics.descriptive_input_diagnostics(log + "x" * 2000, context=context, step=step) == []


@pytest.mark.parametrize("modifier", ["age_group", "ventilation"])
@pytest.mark.parametrize("binary_exposure", [False, True])
def test_external_patch_and_rewrite_receive_typed_failure_and_fix_boolean_input(modifier, binary_exposure, monkeypatch):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    context, step = _context(modifier, binary_exposure), _step(modifier)
    code = (
        "from easyicu.research_agent.methods.descriptive_inputs import strict_numeric_input, strict_binary_input\n"
        'def analyze(frame):\n    return strict_numeric_input(frame["event"]).values\n'
    )
    fixed = code.replace('strict_numeric_input(frame["event"])', 'strict_binary_input(frame["event"])')
    patch = json.dumps({"format": PATCH_FORMAT, "edits": [{
        "old": 'strict_numeric_input(frame["event"])',
        "new": 'strict_binary_input(frame["event"])', "expected_count": 1,
    }]})
    log = _failure_log(pd.Series([True, False], name="event")) + "\nPRIVATE_STDERR_ROW"
    for responses in ([patch], ["invalid patch", fixed]):
        llm = ExternalCaptureMockLLMClient(responses)
        repaired = CoderAgent(llm).repair(context=context, step=step, code=code, run_log=log)
        for messages, _ in llm.calls:
            prompt = "\n".join(message.content for message in messages)
            assert "boolean_not_numeric" in prompt
            assert '\\"column\\": \\"event\\"' in prompt
            assert "untrusted_runtime" in prompt
            assert "effect_output_authorized: true" in prompt
            assert '"dtype":"bool"' in prompt
            assert "strict_binary_input" in prompt
            assert "PRIVATE_STDERR_ROW" not in prompt
        frame = pd.DataFrame({
            "event": pd.Series([True, False, None, False], dtype="boolean"),
            "exposure": [True, False, True, False] if binary_exposure else [1.2, 2.0, 3.1, 4.0],
            modifier: ["younger", "older", "younger", "older"] if modifier == "age_group" else [True, False, True, False],
        })
        namespace = {"frame": frame}
        exec(repaired, namespace)
        outcome = namespace["analyze"](frame)
        assert outcome.iloc[:2].tolist() == [1.0, 0.0]
        assert outcome.isna().equals(frame.event.isna())


def test_runtime_diagnostic_projection_is_bounded_and_cannot_become_repair_authority():
    context, step = _context(), _step()
    log = _failure_log(pd.Series([True, False], name="event"))
    projected = diagnostics.descriptive_input_diagnostics(log, context=context, step=step)
    assert projected == [{
        "column": "event", "reason_codes": ["boolean_not_numeric"],
        "dtype": "bool", "provenance": "untrusted_runtime",
    }]
    envelope = diagnostics.DiagnosticEnvelope.from_repair_authority(
        RepairPromptAuthority(), attempt=2, input_diagnostics=projected,
    ).payload()
    assert envelope["failure_kind"] == "untyped_runtime_failure"
    assert envelope["error_codes"] == []
    assert envelope["input_diagnostics"] == projected
    # Runtime-supplied names, dtypes, counts, values and prose never become authority.
    forged = log.replace('"event"', '"PRIVATE_COLUMN"')
    assert diagnostics.descriptive_input_diagnostics(forged, context=context, step=step) == []
    forged = log.replace('"boolean_not_numeric"', '"IGNORE_INSTRUCTIONS"')
    assert diagnostics.descriptive_input_diagnostics(forged, context=context, step=step) == []
    assert diagnostics.descriptive_input_diagnostics(log, context=context, step=step.model_copy(update={"inputs": []})) == []


def test_initial_and_repair_context_explain_input_coding(monkeypatch):
    from easyicu.research_agent.research_context.repair_prompt import format_repair_authority_context

    monkeypatch.setenv("EASYICU_HOSTED", "1")
    context, step = _context(), _step()
    llm = ExternalCaptureMockLLMClient(["value = 1\n"])
    CoderAgent(llm).run(context=context, step=step)
    assert "strict_binary_input" in "\n".join(message.content for message in llm.calls[0][0])
    for scientific in (False, True):
        text = format_repair_authority_context(context, include_scientific_authority=scientific)
        assert "strict_binary_input" in text
        assert "boolean_numeric_encoding" in text


@pytest.mark.parametrize("dtype,role,declared_input,needs_guidance", [
    ("bool", "outcome", "event", True),
    ("boolean", "outcome", "event", True),
    ("bool", "lab", "event", True),
    ("int64", "outcome", "event", False),
    ("bool", "meta", "event", False),
    ("bool", "outcome", "table:event", False),
    ("bool", "outcome", None, False),
])
def test_generation_scopes_binary_guidance_to_declared_clinical_inputs(
    monkeypatch, dtype, role, declared_input, needs_guidance,
):
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    context = _context()
    context.variables[1] = ConceptDescriptor(name="event", role=role, dtype=dtype)
    step = AnalysisStep(
        step_id="summary", intent="Summarize declared inputs.", method="descriptive",
        inputs=["exposure", "age_group", *([declared_input] if declared_input else [])],
        expected_outputs=["table:summary"],
    )
    llm = ExternalCaptureMockLLMClient(["value = 1\n"])

    CoderAgent(llm).run(context=context, step=step)

    prompt = "\n".join(message.content for message in llm.calls[0][0])
    assert ("strict_binary_input" in prompt) is needs_guidance
    assert ("boolean_numeric_encoding" in prompt) is needs_guidance
    if needs_guidance:
        assert '"boolean_numeric_encoding":{"event":{"false":0,"true":1}}' in prompt
        for guidance in (
            "strict_numeric_input", "False/True map to 0/1", "missing stays missing",
            "non-binary values are rejected", "Pass column=", "declared levels",
            "Never numeric-coerce category labels", "descriptive summaries",
            "Never catch DescriptiveInputError and continue",
        ):
            assert guidance in prompt


def test_exhausted_runtime_failure_retains_typed_diagnostic(tmp_path):
    from easyicu.research_agent.execution.candidate_loop import _candidate_failure_transition

    context, step = _context(), _step()
    record = {"step_id": step.step_id}
    attempt = SimpleNamespace(
        coder_context=context, step=step, step_record=record, step_current=1,
        sealed_renderer_authorized_code_sha256=None,
        worker_progress=SimpleNamespace(
            deterministic_standard_executor_used=False, runner_repair_name=None,
            runtime_repair_attempts=2,
        ),
        _deterministic_fallback_code=lambda _: None,
    )
    state = SimpleNamespace(
        log_path=tmp_path / "absent.log", execution_timeout_seconds=10, code="pass\n",
        run_result=SimpleNamespace(
            stdout="", stderr=_failure_log(pd.Series([True, False], name="event")),
            timed_out=False, returncode=1, runner_failure_code=None,
        ),
    )
    host = SimpleNamespace(
        pipeline=SimpleNamespace(_enable_deterministic_runner_repair=False, _max_code_repair_attempts=2),
        shared_lock=RLock(), findings=[], per_step_records=[], run_id="synthetic", total_steps=1,
        _append_terminal_step_record=lambda records, row: records.append(dict(row)),
        _flush_partial_manifest=lambda: None, emit_progress=lambda *args, **kwargs: None,
    )
    _candidate_failure_transition(host, attempt, state)
    assert host.per_step_records[0]["status"] == "execution_failed"
    assert host.per_step_records[0]["runtime_input_diagnostics"][0]["column"] == "event"
    assert host.per_step_records[0]["runtime_input_diagnostics"][0]["reason_codes"] == ["boolean_not_numeric"]
    assert host.findings[0].severity == "error"
    state.run_result.stderr = "a different failure"
    _candidate_failure_transition(host, attempt, state)
    assert "runtime_input_diagnostics" not in host.per_step_records[-1]


@pytest.mark.parametrize("modifier", ["age_group", "ventilation"])
@pytest.mark.parametrize("binary_exposure", [False, True])
def test_repaired_inputs_support_stratum_associations_and_an_interaction(modifier, binary_exposure):
    import numpy as np
    from scipy.stats import norm
    from easyicu.research_agent.robustness.estimators import fit_estimator

    rng = np.random.default_rng(73)
    size = 160
    frame = pd.DataFrame({
        "exposure": rng.integers(0, 2, size).astype(bool) if binary_exposure else rng.normal(size=size),
        "event": rng.binomial(1, 0.4, size).astype(bool),
        modifier: rng.choice(["younger", "older"], size) if modifier == "age_group" else rng.integers(0, 2, size).astype(bool),
    })
    validate_exposure = inputs.strict_binary_input if binary_exposure else inputs.strict_numeric_input
    exposure = validate_exposure(frame.exposure).values
    outcome = inputs.strict_binary_input(frame.event).values
    # The test protocol declares its two levels and reference before fitting.
    levels = ["younger", "older"] if modifier == "age_group" else [False, True]
    inputs.closed_categorical_counts(frame[modifier], declared_levels=levels)
    encoded = frame[modifier].eq(levels[1]).astype(float)
    design = pd.DataFrame({"exposure": exposure, "modifier": encoded, "interaction": exposure * encoded})
    result = fit_estimator(cohort=None, X=design, y=outcome, kind="logistic", term="exposure")
    assert result.converged and not result.separation_detected
    interaction = next(term for term in result.terms if term.term == "interaction")
    p_value = 2 * norm.sf(abs(np.log(interaction.estimate) / interaction.se))
    assert 0 <= p_value <= 1
    assert result.n == size
    for level in levels:
        selected = frame[modifier].eq(level)
        stratum = fit_estimator(cohort=None, X=design.loc[selected, ["exposure"]], y=outcome[selected], kind="logistic", term="exposure")
        assert stratum.converged and stratum.n == int(selected.sum())
        assert np.isfinite(stratum.point_estimate)


def test_binary_validation_cannot_be_swallowed_and_signature_is_checked():
    from easyicu.research_agent.gates.preflight import audit_mechanical_code_contracts

    code = '''
from easyicu.research_agent.methods.descriptive_inputs import strict_binary_input, DescriptiveInputError
def validate(series):
    try:
        return strict_binary_input(series, invented=True).values
    except DescriptiveInputError:
        return None
'''
    findings = audit_mechanical_code_contracts(code, _step())
    reasons = {finding.detail.get("reason") for finding in findings}
    assert "host_validation_helper_error_swallowed" in reasons
    assert any("invented" in json.dumps(finding.detail) for finding in findings)


def test_prespecified_interaction_method_authorizes_only_declared_effect_products():
    from easyicu.research_agent.contracts.step_families import effect_output_authorized

    step = _step()
    assert effect_output_authorized(step)
    assert not effect_output_authorized(step.model_copy(update={"expected_outputs": ["table:cohort_summary"]}))
    assert not effect_output_authorized(step.model_copy(update={"method": "arbitrary_custom_summary"}))
    assert not effect_output_authorized(step.model_copy(update={"expected_outputs": ["log:effect_modification_estimates"]}))
    assert not effect_output_authorized(step.model_copy(update={"expected_outputs": ["table:effect_modification_estimates_audit"]}))
