"""Synthetic quote failures stay private and can be corrected in one planner run."""

import json
import stat
from dataclasses import replace

import pytest

from easyicu.research_agent.agents import family_spec_planner, progressive_planner
from easyicu.research_agent.agents._support import LLM_PARSE_DEBUG_CHARS
from easyicu.research_agent.agents.progressive_attempt import progressive_planner_failure_facts
from easyicu.research_agent.planning.family_spec import build_landmark_association_skeleton
from easyicu.research_agent.planning.population_spec import quote_written_in
from easyicu.research_agent.planning.progressive_compiler import (
    compile_progressive_plan,
    validate_progressive_foundation,
)
from easyicu.research_agent.planning.progressive_contract import ProgressivePlanCompileError
from easyicu.research_agent.planning.progressive_resume import assemble_progressive_skeleton
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.providers.structured_retry import (
    StructuredResponseFailure,
    call_llm_with_structured_retry,
)
from easyicu.webserver.agent_pipeline_runs import _safe_pipeline_typed_failure

from .family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _request,
    _spec_payload,
)

_GOOD = "observed for at least 12 h"
_BAD = "observed for at least 36 h"
_PATH = "cohort.population_spec.criteria[1].quote"
_REASON = "progressive_population_quote_not_verbatim"


def _run(llm, context, **kwargs):
    return progressive_planner.ProgressivePlannerAgent(llm).run_attempt(
        context,
        planner_strategy=progressive_planner.FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        required_primary_cohort_selection_mode="predicate_filtered",
        **kwargs,
    )


@pytest.fixture
def population_projection(monkeypatch):
    """Exercise the real compiler with a typed population on the template seam.

    This base's family response transports legacy population criteria, so the
    synthetic template projects their words into a typed spec without adding a
    Planner-visible field. All parsing, retry, and compilation remain real.
    """
    builder = family_spec_planner.build_landmark_association_skeleton

    def project(request, spec, **kwargs):
        draft = builder(request, spec, **kwargs)
        foundation = draft.foundation.foundation
        population = {"criteria": [
            {"id": "c1", "source": "question", "role": "include", "kind": "age_years",
             "quote": "adult ICU stays", "min_years": 18},
            {"id": "c2", "source": "question", "role": "include", "kind": "icu_stay_hours",
             "quote": spec.population.criteria[0].criterion, "min_hours": 12},
        ]}
        return replace(draft, foundation=draft.foundation.model_copy(update={
            "foundation": foundation.model_copy(update={
                "cohort": foundation.cohort.model_copy(update={"population_spec": population}),
            }),
        }))

    monkeypatch.setattr(family_spec_planner, "build_landmark_association_skeleton", project)
    context = _context()
    return context.model_copy(update={
        "research_question": context.research_question + f" Only include stays {_GOOD}.",
    })


def _answer(context, quote):
    return json.dumps(_spec_payload(_request(context), population={
        "criteria": [{"criterion": quote, "concept_ids": []}],
    }))


def test_a_bad_quote_then_the_source_words_succeeds_within_the_same_budget(
    population_projection, monkeypatch, tmp_path,
):
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(tmp_path))
    context = population_projection
    llm = ScriptedMockLLMClient([_answer(context, _BAD), _answer(context, _GOOD)])
    events = []
    result = _run(llm, context, progress_callback=events.append)

    assert len(llm.calls) == 2
    assert result.facts.foundation.foundation.cohort.population_spec["criteria"][1]["quote"] == _GOOD
    feedback = llm.calls[1][0][-1].content
    assert _REASON in feedback and _PATH in feedback and _BAD in feedback
    assert context.research_question in feedback
    assert [event.phase for event in events if event.role == "family_spec_planner"] == [
        "started", "rejected", "started", "accepted",
    ]
    saved, = list(tmp_path.glob("family_spec_rejected_*.json"))
    assert _BAD in json.loads(saved.read_text())["response_head"]


def test_the_real_landmark_draft_retries_a_typed_quote_and_compiles_without_patching():
    """Integrate the real template, typed foundation, retry and compiler.

    The family spec transports legacy criteria only. Supply the existing
    typed foundation field as synthetic input, rather than replace the
    builder or add a Planner-visible field just for this integration.
    """
    context = _context()
    context = context.model_copy(update={
        "research_question": context.research_question + f" Only include stays {_GOOD}.",
    })
    request = _request(context)
    projected = []

    def parse(raw):
        spec = family_spec_planner.parse_family_plan_spec(raw, request)
        draft = build_landmark_association_skeleton(request, spec)
        foundation = draft.foundation.foundation
        # Typed synthetic input on the real foundation, preserving the
        # template's cohort predicates, labels, outline and materializations.
        foundation = foundation.model_copy(update={
            "cohort": foundation.cohort.model_copy(update={"population_spec": {"criteria": [
                {"id": "c1", "source": "question", "role": "include", "kind": "age_years",
                 "quote": "adult ICU stays", "min_years": 18},
                {"id": "c2", "source": "question", "role": "include", "kind": "icu_stay_hours",
                 "quote": spec.population.criteria[0].criterion, "min_hours": 12},
            ]}}),
        })
        projected.append(foundation)
        validate_progressive_foundation(
            foundation, context=context, analysis_type=draft.outline.analysis_type,
        )
        return replace(draft, foundation=draft.foundation.model_copy(update={
            "foundation": foundation,
        }))

    llm = ScriptedMockLLMClient([_answer(context, _BAD), _answer(context, _GOOD)])
    draft = call_llm_with_structured_retry(
        llm, [], parser=parse, role=family_spec_planner.FAMILY_SPEC_ROLE,
        max_retries=progressive_planner._MAX_STEP_PARSE_RETRIES,
        retryable_parse_error=lambda exc: isinstance(exc, ValueError),
    )
    assert [item.cohort.population_spec["criteria"][1]["quote"] for item in projected] == [
        _BAD, _GOOD,
    ]
    feedback = llm.calls[1][0][-1].content
    assert _PATH in feedback and _REASON in feedback and _GOOD in feedback
    skeleton = assemble_progressive_skeleton(
        outline=draft.outline, foundation=draft.foundation.foundation,
        steps=[item.step for item in draft.materializations],
    )
    plan, receipt = compile_progressive_plan(
        skeleton=skeleton, context=context, allowed_literature_citation_keys=ALLOWED_CITATIONS,
        adjustment_proposal=draft.adjustment_proposal,
    )
    assert receipt.analysis_plan_sha256 and plan.cohort.inclusion
    assert len(llm.calls) == 2


@pytest.mark.parametrize("raw", ["not-json", "[]", "{}"])
def test_structured_parse_errors_still_retry_and_keep_a_private_rejected_answer(
    population_projection, monkeypatch, tmp_path, raw,
):
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(tmp_path))
    context = population_projection
    llm = ScriptedMockLLMClient([raw, _answer(context, _GOOD)])
    result = _run(llm, context)
    assert result.facts.compile_receipt is not None and len(llm.calls) == 2
    saved, = list(tmp_path.glob("family_spec_rejected_*.json"))
    assert raw in json.loads(saved.read_text())["response_head"]


def test_always_wrong_exhausts_the_existing_budget_with_the_same_stop(
    population_projection, monkeypatch, tmp_path,
):
    debug_dir = tmp_path / "private_debug"
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(debug_dir))
    context = population_projection
    llm = ScriptedMockLLMClient([_answer(context, _BAD)], repeat_last=True)
    with pytest.raises(StructuredResponseFailure) as stopped:
        _run(llm, context)

    budget = progressive_planner._MAX_STEP_PARSE_RETRIES + 1
    assert len(llm.calls) == budget
    assert isinstance(stopped.value.__cause__, ProgressivePlanCompileError)
    assert stopped.value.__cause__.reason_code == _REASON
    assert stopped.value.__cause__.path == _PATH
    diagnostic = _safe_pipeline_typed_failure(stopped.value)
    assert diagnostic["reason_code"] == _REASON and diagnostic["path"] == _PATH
    assert _BAD not in json.dumps(diagnostic)
    records = stopped.value.easyicu_structured_attempt_metadata
    assert len(records) == budget and all(row["reason_code"] == _REASON for row in records)
    assert _BAD not in json.dumps(records)
    facts = progressive_planner_failure_facts(stopped.value)
    assert facts.foundation is None and facts.compile_receipt is None
    files = list(debug_dir.glob("*.json"))
    assert len(files) == budget
    for path in files:
        head = json.loads(path.read_text())["response_head"]
        assert _BAD in head and _PATH in head and _REASON in head


@pytest.mark.parametrize("error", [
    TypeError("synthetic validator bug"),
    ProgressivePlanCompileError("progressive_synthetic_host_refusal", "synthetic host refusal"),
], ids=["TypeError", "unrelated_compiler_refusal"])
def test_a_programming_error_or_host_refusal_is_not_a_rejected_model_answer(
    population_projection, monkeypatch, tmp_path, error,
):
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(tmp_path))
    cause = RuntimeError("synthetic implementation failure")

    def broken_validation(*args, **kwargs):
        raise error from cause

    monkeypatch.setattr(family_spec_planner, "validate_progressive_foundation", broken_validation)
    context = population_projection
    llm = ScriptedMockLLMClient([_answer(context, _GOOD)], repeat_last=True)
    events = []
    with pytest.raises(type(error)) as stopped:
        _run(llm, context, progress_callback=events.append)

    assert stopped.value is error and stopped.value.__cause__ is cause
    assert len(llm.calls) == 1
    assert not list(tmp_path.glob("family_spec_rejected_*.json"))
    assert not [event for event in events if event.phase == "rejected"]


def test_a_long_question_keeps_bounded_retry_feedback_near_the_offending_quote(
    population_projection,
):
    context = population_projection
    context = context.model_copy(update={
        "research_question": (
            "Synthetic background wording. " * 140
            + context.research_question
            + " Additional synthetic background. " * 20
        ),
    })
    llm = ScriptedMockLLMClient([_answer(context, _BAD), _answer(context, _GOOD)])
    result = _run(llm, context)

    assert result.facts.compile_receipt is not None and len(llm.calls) == 2
    feedback = llm.calls[1][0][-1].content
    assert _PATH in feedback and _REASON in feedback and _BAD in feedback
    assert _GOOD in feedback and "[truncated]" in feedback
    response_shape = family_spec_planner.family_spec_response_shape(_request(context))
    assert len(feedback) < len(response_shape) + 3500
    assert context.research_question not in feedback


def test_long_source_does_not_displace_the_private_raw_answer(
    population_projection, monkeypatch, tmp_path,
):
    monkeypatch.setattr(progressive_planner, "_MAX_STEP_PARSE_RETRIES", 0)
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1")
    monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(tmp_path))
    context = population_projection
    context = context.model_copy(update={
        "research_question": "Synthetic background wording. " * 140 + context.research_question,
        "cohort": context.cohort.model_copy(update={
            "inclusion_criteria": ["Synthetic study wording. " * 200],
        }),
    })
    raw = _answer(context, _BAD).replace('"request_sha256":', ' ' * 2400 + '"request_sha256":')
    with pytest.raises(StructuredResponseFailure):
        _run(ScriptedMockLLMClient([raw]), context)
    saved, = list(tmp_path.glob("family_spec_rejected_*.json"))
    head = json.loads(saved.read_text())["response_head"]
    assert _BAD in head and _PATH in head
    assert '"request_sha256"' in head
    assert len(head) <= LLM_PARSE_DEBUG_CHARS


@pytest.mark.parametrize("wording,quote", [
    ('Only include stays “observed for at least 12 h”.', '"observed for at least 12 h"'),
    ('仅纳入「观察≥１２小时」的住院。', '"观察≥12小时"'),
])
def test_normalized_quote_compiles_on_the_first_attempt(population_projection, wording, quote):
    context = population_projection
    context = context.model_copy(update={"research_question": context.research_question + wording})
    llm = ScriptedMockLLMClient([_answer(context, quote)])
    result = _run(llm, context)
    assert len(llm.calls) == 1
    assert result.facts.compile_receipt is not None
    assert result.facts.foundation.foundation.cohort.population_spec["criteria"][1]["quote"] == quote


@pytest.mark.parametrize("enabled,configured", [(False, True), (True, False), (True, True)])
def test_failed_question_answer_is_only_kept_in_opted_in_private_diagnostics(
    monkeypatch, tmp_path, enabled, configured,
):
    monkeypatch.setattr(progressive_planner, "_MAX_STEP_PARSE_RETRIES", 0)
    monkeypatch.setenv("EASYICU_LLM_DEBUG", "1" if enabled else "0")
    debug_dir = tmp_path / "private_debug"
    if configured:
        monkeypatch.setenv("EASYICU_LLM_DEBUG_DIR", str(debug_dir))
    else:
        monkeypatch.delenv("EASYICU_LLM_DEBUG_DIR", raising=False)
    context = _context()
    quote = "synthetic unrelated wording sk-123456789012345678901234"
    payload = _spec_payload(_request(context), question_requirements=[{
        "id": "r1", "kind": "definition", "coverage": "definition_only",
        "quote": quote, "concepts": ["age"], "note": "Defines the population age bound.",
    }])
    # The offending quote lies beyond the raw head's budget. The rejection
    # must still name it, while preserving a bounded part of the raw answer.
    raw = json.dumps(payload).replace('"question_requirements":',
                                      " " * LLM_PARSE_DEBUG_CHARS + '"question_requirements":')
    with pytest.raises(StructuredResponseFailure) as stopped:
        _run(ScriptedMockLLMClient([raw]), context)
    assert stopped.value.__cause__.reason_code == "question_requirement_quote_not_in_question"
    assert quote not in json.dumps(stopped.value.easyicu_structured_attempt_metadata)
    if not (enabled and configured):
        assert not debug_dir.exists()
        return
    files = list(debug_dir.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert "synthetic unrelated wording" in saved["response_head"]
    assert "question_requirement_quote_not_in_question" in saved["response_head"]
    assert "question_requirements[0].quote" in saved["response_head"]
    assert '"request_sha256"' in saved["response_head"]
    assert "123456789012345678901234" not in files[0].read_text()
    assert "[REDACTED]" in saved["response_head"]
    assert saved["truncated"] is True
    assert len(saved["response_head"]) <= LLM_PARSE_DEBUG_CHARS
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(debug_dir.stat().st_mode) == 0o700


@pytest.mark.parametrize("question,quote", [
    ("纳入“成人”（年龄≥１８岁），观察２４小时。", '纳入"成人"(年龄≥18岁),观察24小时。'),
    ("纳入「成人」；观察１２小时。", '纳入"成人";观察12小时。'),
    ("纳入『成人』，年龄≥１８岁。", '纳入"成人",年龄≥18岁。'),
    ("Include ‘adults’ aged 18 years or older.", "'adults' aged 18 years or older"),
    ('Include “adults” (age ≥ 18), observed for 12 h.', '"adults" (age ≥ 18), observed for 12 h'),
    ("Include café workers aged 18 years or older.", "cafe\u0301 workers aged 18 years or older"),
    ("纳入年龄 ≥ 18 岁的成人", "年龄≥18岁"),
    ("Include adults\n  aged 18 years or older.", "adults aged 18 years or older"),
])
def test_harmless_typography_is_accepted_in_both_quote_owners(question, quote):
    from easyicu.research_agent.planning.question_requirements import (
        QuestionRequirement, question_requirement_problems,
    )

    assert quote_written_in(quote, "question", (question,))
    requirement = QuestionRequirement(id="r1", kind="definition", quote=quote,
                                      concepts=["age"], coverage="definition_only",
                                      note="Defines the population age bound.")
    assert not question_requirement_problems([requirement], question=question, roster=("age",), named=())


@pytest.mark.parametrize("question,quote", [
    ("纳入年龄≥１８岁的成人", "年龄≥81岁"),
    ("Include adults aged 18 years or older.", "adults aged 21 years or older"),
    ("Include stays with concentration ≥ 1.8.", "concentration ≥ 18"),
    ("纳入年龄≥１８岁的成人", "年龄>18岁"),
    ("Include adults without the marker.", "adults with the marker"),
    ("纳入年龄≥１８岁的成人", "adults aged 18 years or older"),
])
def test_words_numbers_and_operators_cannot_be_rewritten(question, quote):
    assert not quote_written_in(quote, "question", (question,))
