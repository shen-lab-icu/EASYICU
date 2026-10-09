"""A Writer that failed on the provider's transport names that stop.

When the Writer raises before drafting, the write phase records a
``writer_agent`` error.  When the exception carries a typed server status
(5xx), the error's detail names the stop ``writer_provider_transport_unavailable``
with the status, the attempts the transport made and why it stopped retrying.
A message that only quotes a status, a client status, or a status that is not
an integer names nothing.  Readiness reports the stop of the current Writer
error, the last one supersession left active, as ``writer_stop``; a later
draft retires it.  The draft stage records the stop on the Writer's error,
and the conversation has a sentence for it.

Synthetic exceptions and findings; no provider is called.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.planning.capability_registry import (
    ScientificCapabilityAssessment,
)
from easyicu.research_agent.providers.transport_retry import (
    RETRY_ATTEMPTS_EXHAUSTED,
    RETRY_WINDOW_EXHAUSTED,
)
from easyicu.research_agent.reporting import write_phase
from easyicu.research_agent.reporting import writer_stop as owner
from easyicu.research_agent.reporting.manuscript_state import (
    ManuscriptState,
    render_not_generated,
)
from easyicu.research_agent.reporting.readiness import _compute_readiness_gates
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    ResearchContext,
    ValidationFinding,
)

_STOP = owner.WRITER_PROVIDER_TRANSPORT_UNAVAILABLE


def _failure(
    status: Any = 500, *, attempts: Any = 2, exhausted: Any = None, response=False
):
    failure = RuntimeError("Error code: 500 - proxy CONNECT returned status 503")
    if response:
        failure.response = SimpleNamespace(status_code=status, headers={})
    elif status is not None:
        failure.status_code = status
    if attempts is not None:
        failure.easyicu_transport_attempts = attempts
    if exhausted is not None:
        failure.easyicu_transport_retry_exhausted = exhausted
    return failure


def test_a_typed_server_status_names_the_stop_with_its_attempts():
    stop = owner.writer_transport_stop(_failure(exhausted=RETRY_ATTEMPTS_EXHAUSTED))

    assert stop == {
        "reason_code": _STOP,
        "provider_http_status": 500,
        "transport_attempts": 2,
        "transport_retry_exhausted": RETRY_ATTEMPTS_EXHAUSTED,
    }
    assert owner.writer_transport_stop(_failure(502, response=True, attempts=None)) == {
        "reason_code": _STOP,
        "provider_http_status": 502,
    }


@pytest.mark.parametrize(
    "failure",
    [
        # Only the message names a status.
        _failure(None),
        ValueError("proxy CONNECT returned status 503"),
        _failure("503"),
        _failure(True),
        # A client status is not the provider being unavailable.
        _failure(429),
        _failure(404),
    ],
)
def test_anything_else_names_no_stop(failure):
    assert owner.writer_transport_stop(failure) is None


def test_untyped_attempts_and_unregistered_reasons_are_left_out():
    stop = owner.writer_transport_stop(
        _failure(503, attempts=True, exhausted="the proxy said so")
    )

    assert stop == {"reason_code": _STOP, "provider_http_status": 503}


def test_the_finding_detail_keeps_the_write_phase_fields_beside_the_stop():
    detail = owner.writer_failure_detail(
        _failure(504, attempts=1, exhausted=RETRY_WINDOW_EXHAUSTED),
        exception_type="InternalServerError",
        rejected_candidate_evidence_id=None,
    )

    assert detail == {
        "exception_type": "InternalServerError",
        "rejected_candidate_evidence_id": None,
        "reason_code": _STOP,
        "provider_http_status": 504,
        "transport_attempts": 1,
        "transport_retry_exhausted": RETRY_WINDOW_EXHAUSTED,
    }
    assert owner.writer_failure_detail(
        ValueError("x"), exception_type="ValueError"
    ) == {"exception_type": "ValueError"}


@pytest.mark.parametrize(
    "detail",
    [
        None,
        {"reason_code": "writer_failed_before_draft", "provider_http_status": 500},
        {"reason_code": _STOP},
        {"reason_code": _STOP, "provider_http_status": 404},
        {"reason_code": _STOP, "provider_http_status": "500"},
        {"reason_code": _STOP, "provider_http_status": 600},
    ],
)
def test_a_recorded_stop_is_read_only_when_it_is_one(detail):
    assert owner.writer_stop(detail) is None


def test_a_recorded_stop_is_read_back_without_other_keys():
    assert owner.writer_stop(
        {
            "reason_code": _STOP,
            "provider_http_status": 503,
            "transport_attempts": 0,
            "transport_retry_exhausted": RETRY_ATTEMPTS_EXHAUSTED,
            "exception_type": "InternalServerError",
        }
    ) == {
        "reason_code": _STOP,
        "provider_http_status": 503,
        "transport_retry_exhausted": RETRY_ATTEMPTS_EXHAUSTED,
    }


def _writer_error(detail, severity="error"):
    return ValidationFinding(
        validator="writer_agent",
        severity=severity,
        message="WriterAgent failed before producing a manuscript scaffold: synthetic",
        detail=detail,
    )


_TRANSPORT = {
    "reason_code": _STOP,
    "provider_http_status": 500,
    "transport_attempts": 2,
}


def test_the_last_current_writer_error_decides():
    other = ValidationFinding(
        validator="evidence_bound_writer",
        severity="error",
        message="x",
        detail=_TRANSPORT,
    )

    assert owner.current_writer_stop([_writer_error(_TRANSPORT), other]) == _TRANSPORT
    assert (
        owner.current_writer_stop([_writer_error({"exception_type": "ValueError"})])
        is None
    )
    # A later failure that names nothing replaces an earlier transport stop.
    assert (
        owner.current_writer_stop(
            [_writer_error(_TRANSPORT), _writer_error({"exception_type": "ValueError"})]
        )
        is None
    )
    assert (
        owner.current_writer_stop([_writer_error(_TRANSPORT, severity="warning")])
        is None
    )
    assert owner.current_writer_stop([other]) is None


def _gates(tmp_path: Path, monkeypatch, *, manuscript: str, findings) -> dict:
    monkeypatch.setattr(
        "easyicu.research_agent.reporting.readiness.assess_scientific_capability",
        lambda **_kwargs: ScientificCapabilityAssessment(
            capability_id="test_capability",
            analysis_type="test",
            question_present=True,
            question_coordinates_resolved=True,
            input_contract_resolved=True,
            runtime_data_available=None,
            execution_backend_available=None,
            scientific_validator_available=True,
            claim_ceiling="reportable",
        ),
    )
    path = tmp_path / "manuscript_scaffold_bound.md"
    path.write_text(manuscript, encoding="utf-8")
    return _compute_readiness_gates(
        context=ResearchContext(
            research_question="synthetic",
            cohort={
                "cohort_name": "c",
                "database": "miiv",
                "n_patients": 10,
                "n_stays": 10,
            },
            variables=[],
        ),
        plan=AnalysisPlan(
            research_question="synthetic",
            analysis_type="descriptive_epidemiology",
            steps=[
                AnalysisStep(
                    step_id="01_prep",
                    intent="synthetic",
                    inputs=[],
                    expected_outputs=[],
                )
            ],
        ),
        per_step_records=[{"step_id": "01_prep", "status": "ok"}],
        findings=findings,
        evidence=EvidenceStore(root=tmp_path),
        run_dir=tmp_path,
        manuscript_path=path,
        stop_after_analysis=False,
    )


def test_readiness_reports_the_stop_until_a_draft_retires_it(tmp_path, monkeypatch):
    no_draft = render_not_generated(
        ManuscriptState.blocked("writer_failed_before_draft"), "No draft."
    )

    gates = _gates(
        tmp_path, monkeypatch, manuscript=no_draft, findings=[_writer_error(_TRANSPORT)]
    )
    assert gates["writer_stop"] == _TRANSPORT
    assert gates["evidence_complete"] is False

    gates = _gates(
        tmp_path,
        monkeypatch,
        manuscript="Bound manuscript content here.\n",
        findings=[_writer_error(_TRANSPORT)],
    )
    assert gates["writer_stop"] is None


class _StopAfterWriter(Exception):
    pass


@pytest.mark.parametrize(
    ("failure", "stop"),
    [
        (_failure(503, exhausted=RETRY_ATTEMPTS_EXHAUSTED), True),
        (ValueError("proxy CONNECT returned status 503"), False),
    ],
)
def test_the_draft_stage_records_the_stop_on_the_writer_s_error(
    tmp_path, monkeypatch, failure, stop
):
    def raise_failure(**_kwargs):
        raise failure

    def stop_after_writer(**_kwargs):
        raise _StopAfterWriter

    for name, value in {
        "_rehydrate_step_numeric_authority": lambda **_kwargs: None,
        "declared_table_callouts": lambda **_kwargs: (),
        "_preferred_writer_evidence_names": lambda *_args, **_kwargs: [],
        "analyzed_population": lambda **_kwargs: None,
        "_ensure_unsigned_novelty_positioning_packet": lambda **_kwargs: None,
        "ManuscriptAgent": lambda *_args, **_kwargs: SimpleNamespace(),
        "RegisteredOutputEnvelopeConsumer": lambda: SimpleNamespace(
            authoritative_writer_records=lambda *_args, **_kwargs: []
        ),
        "_render_writer_evidence_digest": lambda **_kwargs: "digest\n",
        "_render_or_resume_writer_scaffold": raise_failure,
        "_preserve_rejected_writer_candidate": lambda *_args, **_kwargs: None,
        "_repair_robustness_reader_prose": stop_after_writer,
    }.items():
        monkeypatch.setattr(write_phase, name, value)
    evidence = SimpleNamespace(
        current_verified_records=lambda _records: [],
        current_resolvable_names=lambda _records: [],
        register_text=lambda **_kwargs: None,
    )
    pipeline = SimpleNamespace(
        _enable_nature_writing_skill=False,
        _user_extension_activation=SimpleNamespace(writing_advisory=None),
        _writer_digest_widened=False,
        _writer_digest_secondary_cap_per_step=0,
    )
    findings: list[ValidationFinding] = []

    with pytest.raises(_StopAfterWriter):
        write_phase._draft_manuscript(
            pipeline,
            context=None,
            agent_context=None,
            evidence=evidence,
            findings=findings,
            literature=None,
            per_step_records=[],
            resume_state=None,
            prompt_version="test",
            role_resolver=lambda _role: None,
            runtime_state=SimpleNamespace(semantics=None),
            execute_result=SimpleNamespace(plan=SimpleNamespace(display_labels={})),
            run_dir=tmp_path,
            run_id="run",
            run_language="en",
            emit_progress=lambda *_args, **_kwargs: None,
        )

    (error,) = [item for item in findings if item.validator == "writer_agent"]
    assert error.severity == "error"
    assert error.detail["exception_type"] == type(failure).__name__
    if stop:
        assert owner.writer_stop(error.detail) == {
            "reason_code": _STOP,
            "provider_http_status": 503,
            "transport_attempts": 2,
            "transport_retry_exhausted": RETRY_ATTEMPTS_EXHAUSTED,
        }
    else:
        assert "reason_code" not in error.detail


def test_the_writer_s_stop_has_a_sentence_for_the_user():
    # The conversation states a failed run's stop by its reason code; a code
    # without a sentence there falls back to the generic failed-check line.
    copy = (
        Path(owner.__file__).resolve().parents[2]
        / "webserver/static/js/screens-guided-pi-error-text.js"
    ).read_text(encoding="utf-8")

    assert f"{_STOP}: " in copy
