"""A standard executor that stops for a reason it can name says which.

A deterministic standard executor whose declared result has no estimate on the
data it was given raises ``ExecutorStop`` and leaves a record of the stop in its
output directory: codes only.  The host reads the record for the executor that
owns the step, removes it with the executor's other private files, and carries
the reason and its lower-layer cause on the step record, the failure finding
and the run's failed steps.  A record from another executor, of another shape
or of an unregistered code names no stop, a timed-out attempt's record is not
read, and an earlier attempt's record is gone before the next attempt runs.  A
step without a record keeps the shape it always had.

Synthetic runs and records; no benchmark item.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts import executor_stop
from easyicu.research_agent.contracts.executor_stop import (
    EXECUTOR_STOP_RECORD_NAME,
    EXECUTOR_STOP_REASONS,
    EXECUTOR_STOP_SCHEMA_VERSION,
    ExecutorStop,
    ExecutorStopReason,
    executor_stop_codes,
    recorded_executor_stop,
    write_executor_stop_record,
)
from easyicu.research_agent.execution.executor_stop_record import (
    read_executor_stop_record,
)
from easyicu.research_agent.execution.standard_executor_diagnostics import (
    standard_executor_failure_finding,
)
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient

_INTERVAL = "continuous_survival_interval_result_not_estimable"
_SUITE = "signed_landmark_continuous_survival_suite"


def _record(**overrides):
    payload = {
        "schema_version": EXECUTOR_STOP_SCHEMA_VERSION,
        "owner": _SUITE,
        "reason_code": _INTERVAL,
        "cause_code": "interval_without_event",
    }
    payload.update(overrides)
    return payload


def _write(out_dir: Path, payload) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    (out_dir / EXECUTOR_STOP_RECORD_NAME).write_bytes(
        data if isinstance(data, bytes) else data.encode("utf-8")
    )


def test_the_suite_owns_its_stop_and_the_interval_model_names_its_causes():
    from easyicu.research_agent.authority.landmark_continuous_survival_runtime import (
        CONTINUOUS_SURVIVAL_PLAN_METHOD,
    )
    from easyicu.research_agent.methods.time_varying_cox import (
        TIME_VARYING_NOT_ESTIMABLE_REASONS,
    )

    reason = EXECUTOR_STOP_REASONS[_INTERVAL]
    assert reason.owner == CONTINUOUS_SURVIVAL_PLAN_METHOD
    assert reason.cause_codes == frozenset(TIME_VARYING_NOT_ESTIMABLE_REASONS)
    assert "invalid_input" not in reason.cause_codes


def test_a_stop_is_a_value_error_whose_message_starts_with_its_code():
    stop = ExecutorStop(_INTERVAL, cause_code="did_not_converge", detail="tau 3")

    assert isinstance(stop, ValueError)
    assert str(stop) == f"{_INTERVAL}: tau 3"
    assert (stop.owner, stop.reason_code, stop.cause_code) == (
        _SUITE,
        _INTERVAL,
        "did_not_converge",
    )


@pytest.mark.parametrize(
    ("reason_code", "cause_code"),
    [
        ("unregistered_stop", None),
        (_INTERVAL, None),
        (_INTERVAL, "invalid_input"),
        (_INTERVAL, "not_a_cause"),
    ],
)
def test_a_stop_outside_the_vocabulary_cannot_be_raised(reason_code, cause_code):
    with pytest.raises(ValueError) as excinfo:
        ExecutorStop(reason_code, cause_code=cause_code)

    assert not isinstance(excinfo.value, ExecutorStop)


def test_a_written_record_is_read_back_for_its_own_executor(tmp_path):
    stop = ExecutorStop(_INTERVAL, cause_code="interval_without_event")

    assert write_executor_stop_record(tmp_path, stop) is True
    recorded, rejection = read_executor_stop_record(tmp_path, expected_owner=_SUITE)

    assert rejection is None
    assert (recorded.reason_code, recorded.cause_code) == (
        _INTERVAL,
        "interval_without_event",
    )
    assert json.loads((tmp_path / EXECUTOR_STOP_RECORD_NAME).read_text()) == _record()
    assert sorted(path.name for path in tmp_path.iterdir()) == [EXECUTOR_STOP_RECORD_NAME]


def _rows_without_late_deaths(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    """Stays alive at the 24-hour landmark, a laboratory value and 28-day death.

    The value's effect reverses five days after the landmark, so the PH test
    rejects one hazard ratio for the whole follow-up, and no stay dies 13 or
    more days after it, so the last follow-up interval (days 14 to 27) has no
    death.
    """

    rng = np.random.default_rng(seed)
    age = rng.normal(65.0, 12.0, n)
    sex = rng.choice(["F", "M"], n)
    value = np.exp(rng.normal(0.7, 0.5, n))
    centred = value - float(np.mean(value))
    early = rng.exponential(1.0 / (0.03 * np.exp(0.9 * centred)))
    late = 5.0 + rng.exponential(1.0 / (0.03 * np.exp(-0.9 * centred)))
    after = np.where(early < 5.0, early, late)
    death = (after <= 27.0) & (after < 13.0)
    return pd.DataFrame(
        {
            "lab_max": value,
            "mort_28d": death.astype(int),
            "followup_days_28d": np.where(death, 1.0 + after, 28.0),
            "age": age,
            "sex": sex,
        }
    )


def test_the_suite_names_its_stop_when_its_interval_result_has_no_estimate(tmp_path):
    """The PH test rejects, and the last follow-up interval has no death."""

    pytest.importorskip("lifelines")
    from easyicu.research_agent.authority.current_case_scientific_runtime import (
        build_current_case_scientific_runtime_authority,
    )
    from easyicu.research_agent.execution.runners.landmark_continuous_survival_executor import (
        run_landmark_continuous_survival_suite,
    )
    from easyicu.research_agent.methods.time_varying_cox import TimeVaryingCoxError
    from tests.support.continuous_survival import continuous_authority_body

    authority = build_current_case_scientific_runtime_authority(continuous_authority_body())
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()  # the runner creates the step's output directory first
    with pytest.raises(ExecutorStop) as caught:
        run_landmark_continuous_survival_suite(
            frame=_rows_without_late_deaths(),
            authority=authority.model_dump(mode="json"),
            runtime_projection_sha256="b" * 64,
            out_dir=out_dir,
            input_product="table:analysis_cohort",
            input_evidence_id="cohort_evidence",
            input_sha256="c" * 64,
        )

    stop = caught.value
    assert (stop.reason_code, stop.cause_code) == (_INTERVAL, "interval_without_event")
    assert str(stop).startswith(f"{_INTERVAL}: the PH test rejected")
    assert isinstance(stop.__cause__, TimeVaryingCoxError)
    recorded, rejection = read_executor_stop_record(out_dir, expected_owner=_SUITE)
    assert rejection is None
    assert (recorded.reason_code, recorded.cause_code) == (
        _INTERVAL,
        "interval_without_event",
    )


def test_a_record_that_cannot_be_written_never_replaces_the_stop(tmp_path, capsys):
    stop = ExecutorStop(_INTERVAL, cause_code="interval_without_event")

    assert write_executor_stop_record(tmp_path / "missing", stop) is False
    assert "executor stop record not written" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("payload", "rejection"),
    [
        (_record(owner="grouped_table_one"), "record_owner_mismatch"),
        (_record(reason_code="unregistered_stop"), "record_reason_unregistered"),
        (_record(cause_code="not_a_cause"), "record_cause_unregistered"),
        (_record(cause_code=None), "record_cause_unregistered"),
        (_record(schema_version="easyicu.executor_stop/9"), "record_schema_unknown"),
        ({**_record(), "detail": "free text"}, "record_shape_invalid"),
        ([_record()], "record_shape_invalid"),
        ('{"schema_version": NaN}', "record_unreadable"),
        (b"\xff\xfe", "record_unreadable"),
        (b" " * 5000, "record_too_large"),
    ],
)
def test_a_record_of_another_shape_or_code_names_no_stop(tmp_path, payload, rejection):
    _write(tmp_path, payload)

    assert read_executor_stop_record(tmp_path, expected_owner=_SUITE) == (None, rejection)


def test_a_record_is_not_read_through_a_link(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere, _record())
    linked_record = tmp_path / "record_link"
    linked_record.mkdir()
    (linked_record / EXECUTOR_STOP_RECORD_NAME).symlink_to(
        elsewhere / EXECUTOR_STOP_RECORD_NAME
    )
    linked_directory = tmp_path / "directory_link"
    linked_directory.symlink_to(elsewhere, target_is_directory=True)

    assert read_executor_stop_record(linked_record, expected_owner=_SUITE) == (
        None,
        "record_unreadable",
    )
    assert read_executor_stop_record(linked_directory, expected_owner=_SUITE) == (
        None,
        "record_unreadable",
    )
    assert read_executor_stop_record(tmp_path / "absent", expected_owner=_SUITE) == (
        None,
        None,
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_special_file_at_the_record_name_is_refused_without_waiting(tmp_path):
    directory = tmp_path / "directory"
    (directory / EXECUTOR_STOP_RECORD_NAME).mkdir(parents=True)
    assert read_executor_stop_record(directory, expected_owner=_SUITE) == (
        None,
        "record_unreadable",
    )

    fifo = tmp_path / "fifo" / EXECUTOR_STOP_RECORD_NAME
    fifo.parent.mkdir()
    os.mkfifo(fifo)
    read: list = []
    reader = threading.Thread(
        target=lambda: read.append(
            read_executor_stop_record(fifo.parent, expected_owner=_SUITE)
        ),
        daemon=True,
    )
    reader.start()
    reader.join(timeout=5)
    if reader.is_alive():
        # Release the waiting open before failing.
        os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
        reader.join(timeout=5)
        pytest.fail("the read waited for a writer at the record's name")
    assert read == [(None, "record_unreadable")]


def test_a_step_record_carries_a_stop_only_for_its_own_executor():
    record = {
        "deterministic_standard_analysis": _SUITE,
        "executor_stop_reason_code": _INTERVAL,
        "executor_stop_cause_code": "non_finite_estimate",
    }

    assert recorded_executor_stop(record) == (_INTERVAL, "non_finite_estimate")
    assert recorded_executor_stop({**record, "deterministic_standard_analysis": "x"}) is None
    assert recorded_executor_stop({**record, "executor_stop_cause_code": "x"}) is None
    assert recorded_executor_stop({"deterministic_standard_analysis": _SUITE}) is None
    assert executor_stop_codes(record) == {
        "reason_code": _INTERVAL,
        "cause_code": "non_finite_estimate",
    }
    assert executor_stop_codes({**record, "deterministic_standard_analysis": "x"}) == {}


def test_the_failure_finding_names_the_stop_and_its_cause():
    record = {
        "deterministic_standard_analysis": _SUITE,
        "executor_stop_reason_code": _INTERVAL,
        "executor_stop_cause_code": "invalid_contrast_variance",
    }
    finding = standard_executor_failure_finding(
        step_record=record,
        step_id="04_suite",
        reason="executor_typed_stop",
        failure_phase="execution_or_output_validation",
    )
    plain = standard_executor_failure_finding(
        step_record={"deterministic_standard_analysis": _SUITE},
        step_id="04_suite",
        reason="executor_runtime_failure",
        failure_phase="execution_or_output_validation",
    )

    assert finding.detail["executor_stop_reason_code"] == _INTERVAL
    assert finding.detail["executor_stop_cause_code"] == "invalid_contrast_variance"
    assert "executor_stop_reason_code" not in plain.detail


def test_the_run_names_a_failed_step_s_stop_and_otherwise_keeps_its_shape():
    from easyicu.research_agent.reporting.readiness import execution_gate_status
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan.model_validate(
        {
            "research_question": "q",
            "steps": [
                {"step_id": step_id, "intent": "i", "inputs": [], "expected_outputs": []}
                for step_id in ("01_a", "02_b", "03_c")
            ],
        }
    )
    records = [
        {"step_id": "01_a", "status": "ok"},
        {
            "step_id": "02_b",
            "status": "deterministic_standard_blocked",
            "deterministic_standard_analysis": _SUITE,
            "executor_stop_reason_code": _INTERVAL,
            "executor_stop_cause_code": "interval_without_event",
        },
        {
            "step_id": "03_c",
            "status": "deterministic_standard_blocked",
            "deterministic_standard_analysis": "grouped_table_one",
            "executor_stop_reason_code": _INTERVAL,
        },
    ]

    gate = execution_gate_status(plan=plan, per_step_records=records)

    assert gate["failed_steps"] == [
        {
            "step_id": "02_b",
            "status": "deterministic_standard_blocked",
            "reason_code": _INTERVAL,
            "cause_code": "interval_without_event",
        },
        {"step_id": "03_c", "status": "deterministic_standard_blocked"},
    ]


# ---- through the execution phase -------------------------------------------

_TABLE_ONE = "grouped_table_one"
_TEST_STOP = "grouped_table_one_test_stop"


def _plan() -> str:
    return json.dumps(
        {
            "research_question": "Summarize the ICU cohort.",
            "steps": [
                {
                    "step_id": "01_summary",
                    "planned_analysis_role": "auxiliary",
                    "intent": "Summarize age by event status with the grouped Table 1 executor.",
                    "inputs": ["age", "death"],
                    "expected_outputs": ["table:table_one"],
                    "method": "table_one",
                    "table_one_spec": {
                        "schema_version": "easyicu.table_one/2",
                        "p_value_adjustment": "not_applicable_repeated_units",
                        "group_by": "death",
                        "group_levels": [0, 1],
                        "variables": [
                            {
                                "name": "age",
                                "variable_kind": "continuous",
                                "summary": "median_iqr",
                                "test": "none_descriptive_smd_only",
                            }
                        ],
                        "p_values_required": False,
                    },
                }
            ],
        }
    )


def _run(
    ra, tmp_path: Path, monkeypatch, *, record, timed_out: bool = False, stale: bool = False
):
    """One run whose standard executor fails after leaving ``record``.

    With ``stale``, a record is already in the step's output directory when the
    runner is built, as an earlier attempt would have left it.
    """

    from easyicu.research_agent.agents.core import PlannerAgent
    from easyicu.research_agent.contracts.runtime import RunResult

    original_run = PlannerAgent.run

    def run_without_article_suite(self, context, **kwargs):
        kwargs["enforce_article_contract"] = False
        return original_run(self, context, **kwargs)

    monkeypatch.setattr(PlannerAgent, "run", run_without_article_suite)
    monkeypatch.setattr(
        executor_stop,
        "EXECUTOR_STOP_REASONS",
        MappingProxyType(
            {
                **EXECUTOR_STOP_REASONS,
                _TEST_STOP: ExecutorStopReason(
                    owner=_TABLE_ONE,
                    cause_codes=frozenset({"test_cause"}),
                    repeats_on_unchanged_retry=True,
                ),
            }
        ),
    )
    seen_before_run: list[bool] = []

    class StoppingRunner:
        network_policy = "none"
        authority_identity_sha256 = "1" * 64

        def __init__(self, *, workdir: Path, timeout_seconds: float) -> None:
            self.workdir = Path(workdir)
            self.timeout_seconds = timeout_seconds

        @staticmethod
        def validate_runtime_capabilities() -> tuple[str, ...]:
            return ("pandas",)

        def run(self, *, step_id, code, resolved_inputs_path=None):
            del resolved_inputs_path
            step_dir = self.workdir / "steps" / step_id
            out_dir = step_dir / "outputs"
            seen_before_run.append((out_dir / EXECUTOR_STOP_RECORD_NAME).exists())
            out_dir.mkdir(parents=True, exist_ok=True)
            if record is not None:
                _write(out_dir, record)
            script_path = step_dir / "analysis.py"
            script_path.write_text(code, encoding="utf-8")
            log_path = step_dir / "run.log"
            log_path.write_text(f"ValueError: {_TEST_STOP}: synthetic\n", encoding="utf-8")
            return RunResult(
                step_id=step_id,
                script_path=script_path,
                cwd=step_dir,
                out_dir=out_dir,
                stdout="",
                stderr="synthetic stop",
                returncode=-9 if timed_out else 1,
                duration_seconds=0.02,
                artefacts=sorted(out_dir.iterdir()),
                timed_out=timed_out,
                effective_isolation="controlled_test",
                runner_log_path=log_path,
            )

    def runner_factory(*, workdir, timeout_seconds=300.0, **_kwargs):
        if stale:
            _write(Path(workdir) / "steps" / "01_summary" / "outputs", _test_record())
        return StoppingRunner(workdir=Path(workdir), timeout_seconds=timeout_seconds)

    llm = PatternScriptedMockLLMClient(
        [
            ("ICU-AWARE RESEARCH PLAN", [_plan()] * 4),
            ("CONSERVATIVE ICU CONCEPT-USE AUDITOR", [json.dumps({"findings": []})] * 8),
            ("INTERPRET THE RESULTS", ["The cohort summary is available."] * 8),
        ]
    )
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=llm,
        runner_factory=runner_factory,
        timeout_seconds=300.0,
        enable_literature=False,
        enable_visual_qa=False,
        enable_latex=False,
        enable_llm_concept_audit=False,
        enable_deterministic_code_fallback=False,
        enable_deterministic_runner_repair=False,
        enable_probe_step=False,
        enable_replanning=False,
    )
    result = pipeline.run(
        question="Summarize the ICU cohort.",
        cohort=pd.DataFrame({"stay_id": [1, 2, 3], "death": [0, 1, 0], "age": [40, 50, 60]}),
        cohort_name="executor_stop_test",
        database="synthetic",
        target_outcome="death",
        stop_after_step_id="01_summary",
        stop_after_analysis=True,
    )
    run_dir = Path(result.workdir)
    partial = json.loads((run_dir / "manifest_partial.json").read_text("utf-8"))
    [step] = [
        item for item in partial["per_step_records"] if item.get("step_id") == "01_summary"
    ]
    findings = [
        finding
        for finding in partial.get("findings") or []
        if (finding.get("detail") or {}).get("issue_code")
        == "deterministic_standard_executor_failed_closed"
    ]
    return run_dir, step, findings, seen_before_run


def _test_record(**overrides):
    return _record(owner=_TABLE_ONE, reason_code=_TEST_STOP, cause_code="test_cause", **overrides)


def test_a_named_stop_reaches_the_step_record_and_its_finding(ra, tmp_path, monkeypatch):
    run_dir, step, findings, _seen = _run(ra, tmp_path, monkeypatch, record=_test_record())

    assert step["status"] == "deterministic_standard_blocked"
    assert step["standard_executor_terminal_reason"] == "executor_typed_stop"
    assert step["executor_stop_reason_code"] == _TEST_STOP
    assert step["executor_stop_cause_code"] == "test_cause"
    assert [finding["detail"]["executor_stop_reason_code"] for finding in findings] == [
        _TEST_STOP
    ]
    # The record is a private work product of the attempt: it is gone before
    # the output directory is scanned for evidence.
    out_dir = run_dir / "steps" / "01_summary" / "outputs"
    assert not (out_dir / EXECUTOR_STOP_RECORD_NAME).exists()


@pytest.mark.parametrize(
    ("record", "timed_out", "rejected"),
    [
        (None, False, None),
        (_record(owner=_TABLE_ONE, reason_code=_INTERVAL), False, "record_reason_unregistered"),
        ({"owner": _TABLE_ONE}, False, "record_shape_invalid"),
        ("__test_record__", True, None),
    ],
)
def test_without_an_accepted_record_the_failure_is_what_it_always_was(
    ra, tmp_path, monkeypatch, record, timed_out, rejected
):
    if record == "__test_record__":
        record = _test_record()
    _run_dir, step, findings, _seen = _run(
        ra, tmp_path, monkeypatch, record=record, timed_out=timed_out
    )

    assert step["status"] == "deterministic_standard_blocked"
    assert step["standard_executor_terminal_reason"] == "executor_runtime_failure"
    assert "executor_stop_reason_code" not in step
    assert step.get("executor_stop_record_rejected") == rejected
    assert all("executor_stop_reason_code" not in finding["detail"] for finding in findings)


def test_a_record_left_by_an_earlier_attempt_is_gone_before_the_next(
    ra, tmp_path, monkeypatch
):
    # The step executor replaces the output directory for every attempt, so a
    # record speaks only for the attempt that wrote it.
    _run_dir, step, _findings, seen_before_run = _run(
        ra, tmp_path, monkeypatch, record=None, stale=True
    )

    assert seen_before_run == [False]
    assert step["standard_executor_terminal_reason"] == "executor_runtime_failure"
    assert "executor_stop_reason_code" not in step


# ---- the user's sentence ------------------------------------------------------


def test_every_stop_the_host_reads_has_a_sentence_for_the_user():
    # The conversation states a failed run's stop by its reason code; a code
    # without a sentence there falls back to the generic failed-check line.
    copy = (
        Path(executor_stop.__file__).resolve().parents[2]
        / "webserver/static/js/screens-guided-pi-error-text.js"
    ).read_text(encoding="utf-8")

    assert [code for code in EXECUTOR_STOP_REASONS if f"{code}: " not in copy] == []
