from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from benchmarks.figure2_canonical9.evaluator import acceptance
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS


def _write_results(root: Path) -> Path:
    scores: list[dict[str, Any]] = []
    for task_id in FIGURE2_TASK_IDS:
        run_dir = root / task_id / "aware" / f"run_{task_id}"
        run_dir.mkdir(parents=True)
        scores.append(
            {
                "item_key": task_id,
                "aware": {
                    "arm": "aware",
                    "run_id": run_dir.name,
                    "workdir": str(run_dir),
                    "figure2_evaluation_attempt": {
                        "status": "valid",
                        "task_id": task_id,
                        "run_id": run_dir.name,
                    },
                },
            }
        )
    path = root / "ehrflowbench_results.json"
    path.write_text(
        json.dumps(
            {
                "items": list(FIGURE2_TASK_IDS),
                "arms": ["aware"],
                "scores": scores,
                "pending": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _install_valid_attempt_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def parse_attempt(
        _cls: type[object],
        payload: bytes,
        *,
        strict: bool,
    ) -> SimpleNamespace:
        assert strict is True
        row = json.loads(payload)
        return SimpleNamespace(
            status=row["status"],
            task_id=row["task_id"],
            run_id=row["run_id"],
        )

    monkeypatch.setattr(
        acceptance.Figure2EvaluationAttempt,
        "model_validate_json",
        classmethod(parse_attempt),
    )
    monkeypatch.setattr(
        acceptance,
        "verify_figure2_evaluation_attempt",
        lambda _run_dir, _attempt: SimpleNamespace(
            scorecard=SimpleNamespace(tristate="gate_reportable")
        ),
    )


def _mutate(path: Path, callback: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_exact_nine_valid_aware_attempts_are_replay_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)

    report = acceptance.evaluate_figure2_paper_acceptance(path)

    assert report.status == "accepted"
    assert tuple(row.task_id for row in report.verified_tasks) == FIGURE2_TASK_IDS
    assert report.issues == ()


@pytest.mark.parametrize(
    "mutation, expected_code",
    [
        (lambda p: p["items"].reverse(), "TASK_COVERAGE_INVALID"),
        (lambda p: p["scores"].reverse(), "SCORE_ORDER_INVALID"),
        (lambda p: p["scores"].pop(), "TASK_SCORE_CARDINALITY_INVALID"),
        (lambda p: p["scores"].append(p["scores"][0]), "SCORE_ORDER_INVALID"),
        (lambda p: p.update(arms=["naive", "aware"]), "ARM_AUTHORITY_INVALID"),
        (
            lambda p: p["scores"][0]["aware"].update(arm="naive"),
            "ARM_AUTHORITY_INVALID",
        ),
        (
            lambda p: p["scores"][0]["aware"].pop("arm"),
            "ARM_AUTHORITY_INVALID",
        ),
        (
            lambda p: p["pending"].append(
                {"key": FIGURE2_TASK_IDS[0], "status": "item_exception"}
            ),
            "TASK_NOT_COMPLETED",
        ),
        (
            lambda p: p["pending"].append(
                {"key": "foreign_task", "status": "item_exception"}
            ),
            "PENDING_LEDGER_INVALID",
        ),
        (
            lambda p: p["pending"].append({"status": "item_exception"}),
            "PENDING_LEDGER_INVALID",
        ),
        (
            lambda p: p["scores"][0]["aware"]["figure2_evaluation_attempt"].update(
                status="invalid"
            ),
            "EVALUATION_ATTEMPT_INVALID",
        ),
        (
            lambda p: p["scores"][0]["aware"].update(run_id="run_wrong"),
            "RUN_IDENTITY_INVALID",
        ),
        (lambda p: p.update(totals=float("nan")), "RESULTS_DOCUMENT_INVALID"),
    ],
)
def test_coverage_identity_and_attempt_failures_are_structured_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    expected_code: str,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)
    _mutate(path, mutation)

    report = acceptance.evaluate_figure2_paper_acceptance(path)

    assert report.status == "invalid"
    assert expected_code in {issue.code for issue in report.issues}


def test_replay_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)

    def fail_replay(_run_dir: Path, _attempt: object) -> None:
        raise ValueError("authority drift")

    monkeypatch.setattr(acceptance, "verify_figure2_evaluation_attempt", fail_replay)

    report = acceptance.evaluate_figure2_paper_acceptance(path)

    assert report.status == "invalid"
    assert {issue.code for issue in report.issues} == {"EVALUATION_REPLAY_FAILED"}


def test_workdir_cannot_be_transplanted_under_another_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = payload["scores"][0]["aware"]
    transplanted = tmp_path / FIGURE2_TASK_IDS[1] / "aware" / str(first["run_id"])
    transplanted.mkdir()
    first["workdir"] = str(transplanted)
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = acceptance.evaluate_figure2_paper_acceptance(path)

    assert report.status == "invalid"
    assert "RUN_IDENTITY_INVALID" in {issue.code for issue in report.issues}


def test_duplicate_json_or_symlink_results_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"items":[],"items":[]}', encoding="utf-8")
    duplicate_report = acceptance.evaluate_figure2_paper_acceptance(duplicate)
    assert duplicate_report.status == "invalid"
    assert duplicate_report.issues[0].code == "RESULTS_DOCUMENT_INVALID"

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    link_report = acceptance.evaluate_figure2_paper_acceptance(link)
    assert link_report.status == "invalid"
    assert link_report.issues[0].code == "RESULTS_DOCUMENT_INVALID"
