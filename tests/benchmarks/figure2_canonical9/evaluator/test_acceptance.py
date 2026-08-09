from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
)
from easyicu.research_agent.providers.factory import ProviderAuthorization
from benchmarks.figure2_canonical9.evaluator import acceptance
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS


def _execution_identity(
    *,
    host_runner_authorized: bool = False,
    model: str = "frozen-model",
    input_authority_sha256: str | None = "c" * 64,
) -> dict[str, Any]:
    authorization = ProviderAuthorization.create(
        provider="openai",
        model=model,
        base_url="https://provider.example/v1",
        destination="external",
        authorization_mode="operator_env",
    )
    return ExecutionIdentity.create(
        submission_profile_name="npj_dm",
        submission_profile_version="frozen-v1",
        runner="docker",
        runner_image_digest="sha256:" + "a" * 64,
        network_policy="none",
        # A fixture object cannot self-authorize after the provider-factory
        # hardening.  Feed the sealed, non-secret authority manifest directly
        # to the identity constructor, which is the same immutable provenance
        # shape the real factory records after it has vetted a transport.
        provider_authorization={
            "schema_version": "easyicu.provider_authorization_manifest/2",
            "reasoning_effort_profile": "provider_default",
            "clients": [asdict(authorization)],
        },
        llm_seed=20260722,
        data_seed=7,
        input_authority_sha256=input_authority_sha256,
        host_runner_authorized=host_runner_authorized,
        code_version={"git_sha": "b" * 40, "git_dirty": False},
    ).model_dump(mode="json")


def _write_results(
    root: Path,
    *,
    host_runner_authorized: bool = False,
    input_authority_sha256: str | None = "c" * 64,
) -> Path:
    identity = _execution_identity(
        host_runner_authorized=host_runner_authorized,
        input_authority_sha256=input_authority_sha256,
    )
    expected_identity = ExpectedExecutionIdentity.create(
        ExecutionIdentity.model_validate(
            _execution_identity(input_authority_sha256=input_authority_sha256),
            strict=True,
        )
    )
    (root / "expected_execution_identity.json").write_text(
        json.dumps(expected_identity.model_dump(mode="json")),
        encoding="utf-8",
    )
    results_root = root / "results"
    scores: list[dict[str, Any]] = []
    for task_id in FIGURE2_TASK_IDS:
        run_dir = results_root / task_id / "aware" / f"run_{task_id}"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"execution_identity": identity}),
            encoding="utf-8",
        )
        scores.append(
            {
                "item_key": task_id,
                "aware": {
                    "arm": "aware",
                    "run_id": run_dir.name,
                    "workdir": str(run_dir),
                    "execution_identity": identity,
                    "figure2_evaluation_attempt": {
                        "status": "valid",
                        "task_id": task_id,
                        "run_id": run_dir.name,
                    },
                },
            }
        )
    path = results_root / "ehrflowbench_results.json"
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


def _expected_path(results_path: Path) -> Path:
    return results_path.parent.parent / "expected_execution_identity.json"


def _install_valid_attempt_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tristate: str = "gate_reportable",
) -> None:
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
            scorecard=SimpleNamespace(tristate=tristate)
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

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "accepted"
    assert tuple(row.task_id for row in report.verified_tasks) == FIGURE2_TASK_IDS
    assert report.issues == ()


@pytest.mark.parametrize("tristate", ["analysis_only", "diagnostic_only"])
def test_replay_verified_but_nonreportable_tasks_are_not_paper_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tristate: str,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch, tristate=tristate)

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "invalid"
    assert len(report.verified_tasks) == len(FIGURE2_TASK_IDS)
    issues = [
        issue for issue in report.issues if issue.code == "TASK_NOT_GATE_REPORTABLE"
    ]
    assert [issue.task_id for issue in issues] == list(FIGURE2_TASK_IDS)


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

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

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

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "invalid"
    assert {issue.code for issue in report.issues} == {"EVALUATION_REPLAY_FAILED"}


def test_host_runner_authorization_can_never_pass_paper_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path, host_runner_authorized=True)
    _install_valid_attempt_stubs(monkeypatch)

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "invalid"
    assert {issue.code for issue in report.issues} == {"EXECUTION_IDENTITY_INVALID"}


def test_missing_input_authority_can_never_pass_paper_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path, input_authority_sha256=None)
    _install_valid_attempt_stubs(monkeypatch)

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "invalid"
    assert {
        "EXPECTED_EXECUTION_IDENTITY_INVALID",
        "EXECUTION_IDENTITY_INVALID",
    } <= {issue.code for issue in report.issues}


def test_consistent_but_unfrozen_identity_cannot_pass_paper_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)
    alternate = _execution_identity(model="unfrozen-model")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["scores"]:
        row["aware"]["execution_identity"] = alternate
        run_dir = Path(row["aware"]["workdir"])
        (run_dir / "manifest.json").write_text(
            json.dumps({"execution_identity": alternate}),
            encoding="utf-8",
        )
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

    assert report.status == "invalid"
    assert {issue.code for issue in report.issues} == {"EXECUTION_IDENTITY_INVALID"}


def test_results_cannot_supply_their_own_expected_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)
    self_declared = path.parent / "expected_execution_identity.json"
    self_declared.write_bytes(_expected_path(path).read_bytes())

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=self_declared,
    )

    assert report.status == "invalid"
    assert "EXPECTED_EXECUTION_IDENTITY_INVALID" in {
        issue.code for issue in report.issues
    }


def test_workdir_cannot_be_transplanted_under_another_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_results(tmp_path)
    _install_valid_attempt_stubs(monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = payload["scores"][0]["aware"]
    transplanted = path.parent / FIGURE2_TASK_IDS[1] / "aware" / str(first["run_id"])
    transplanted.mkdir()
    first["workdir"] = str(transplanted)
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = acceptance.evaluate_figure2_paper_acceptance(
        path,
        expected_execution_identity_path=_expected_path(path),
    )

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
