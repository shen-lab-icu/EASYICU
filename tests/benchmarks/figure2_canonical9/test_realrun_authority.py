"""FIG2-REALRUN-AUTHORITY-P4 — pre-run freeze/authorization receipt.

One positive path (everything bound -> ``authorized``) and one fail-closed
negative per authority guarantee.  No real Provider, Docker, patient data, or
Canonical9 run is involved: frozen identities and freeze docs are synthetic
fixtures, and the only real file read is the committed input-freeze ledger
(asserted to still be blocked, i.e. NOT yet authorized for a real run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
)

from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.realrun_authority import (
    INPUT_FREEZE_SCHEMA,
    RealRunAuthorization,
    RealRunAuthorizationRequest,
    verify_realrun_authorization,
    write_realrun_authorization_receipt,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_INPUT_FREEZE = (
    _REPO_ROOT / "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json"
)
_REAL_RUBRIC = _REPO_ROOT / "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json"


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _frozen_identity(
    path: Path, *, git_dirty: bool = False, input_authority: str | None = "c" * 64
) -> ExpectedExecutionIdentity:
    identity = ExecutionIdentity.create(
        submission_profile_name="npj_dm",
        submission_profile_version="20260718",
        runner="docker",
        runner_image_digest="b" * 64,
        network_policy="none",
        llm_seed=0,
        input_authority_sha256=input_authority,
        provider_authorization={"clients": [{"authorization_mode": "operator_env"}]},
        host_runner_authorized=False,
        code_version={"git_sha": "a" * 40, "git_dirty": git_dirty},
    )
    frozen = ExpectedExecutionIdentity.create(identity)
    path.write_text(frozen.model_dump_json(), encoding="utf-8")
    return frozen


def _authorized_freeze(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": INPUT_FREEZE_SCHEMA,
                "cases": [],  # no outstanding rematerialize blockers
                "submission_profile": {
                    "ref": "npj_dm/20260718",
                    "concept_dict_sha256": "d" * 64,
                    "sofa2_dict_sha256": "e" * 64,
                },
                "manifest_ref": "figure2_canonical9/input_freeze/20260718",
            }
        ),
        encoding="utf-8",
    )


def _blocked_freeze(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": INPUT_FREEZE_SCHEMA,
                "cases": [
                    {
                        "benchmark_item_id": "e2_lactate_mortality",
                        "case_id": "e2",
                        "state": "blocked",
                        "blockers": [
                            {
                                "code": "TYPED_COHORT_AUTHORITY_MISSING",
                                "resolution": "rematerialize",
                            }
                        ],
                    }
                ],
                "submission_profile": {
                    "ref": "npj_dm/20260718",
                    "concept_dict_sha256": "d" * 64,
                    "sofa2_dict_sha256": "e" * 64,
                },
            }
        ),
        encoding="utf-8",
    )


def _rubric(path: Path) -> None:
    path.write_text(
        json.dumps({"schema_version": "rubric/v3", "criteria": []}), "utf-8"
    )


def _make_request(tmp_path: Path, **overrides) -> RealRunAuthorizationRequest:
    identity_path = tmp_path / "expected_identity.json"
    freeze_path = tmp_path / "input_freeze.json"
    rubric_path = tmp_path / "rubric.json"
    if not identity_path.exists():
        _frozen_identity(identity_path)
    if not freeze_path.exists():
        _authorized_freeze(freeze_path)
    if not rubric_path.exists():
        _rubric(rubric_path)
    base = dict(
        expected_execution_identity_path=identity_path,
        input_freeze_path=freeze_path,
        rubric_path=rubric_path,
        output_root=tmp_path / "out",
        arms=("aware",),
        requested_task_ids=FIGURE2_TASK_IDS,
        fresh_run=True,
        cross_run_memory=False,
        resume_run_id=None,
    )
    base.update(overrides)
    return RealRunAuthorizationRequest(**base)


def _codes(auth: RealRunAuthorization) -> set[str]:
    return {issue.code for issue in auth.issues}


# ---------------------------------------------------------------------------
# positive
# ---------------------------------------------------------------------------


def test_authorized_receipt_when_everything_is_bound(tmp_path) -> None:
    auth = verify_realrun_authorization(_make_request(tmp_path))
    assert auth.status == "authorized", auth.issues
    assert auth.issues == ()
    assert auth.arms == ("aware",)
    assert auth.expected_task_ids == tuple(FIGURE2_TASK_IDS)
    assert auth.requested_task_ids == tuple(FIGURE2_TASK_IDS)
    assert auth.fresh_run is True
    assert auth.cross_run_memory_disabled is True
    # every identity is durably bound as a sha
    for value in (
        auth.code_git_sha,
        auth.expected_execution_identity_sha256,
        auth.expected_execution_identity_freeze_sha256,
        auth.input_authority_sha256,
        auth.input_freeze_sha256,
        auth.input_freeze_submission_profile_ref,
        auth.rubric_sha256,
    ):
        assert value


def test_receipt_persists_and_round_trips_secret_free(tmp_path) -> None:
    auth = verify_realrun_authorization(_make_request(tmp_path))
    out = write_realrun_authorization_receipt(auth, tmp_path / "receipt.json")
    text = out.read_text(encoding="utf-8")
    reloaded = RealRunAuthorization.model_validate_json(text)
    assert reloaded.status == "authorized"
    lowered = text.lower()
    for secret_marker in ("api_key", "apikey", "secret", "token", "bearer", "password"):
        assert secret_marker not in lowered


# ---------------------------------------------------------------------------
# one fail-closed negative per authority guarantee
# ---------------------------------------------------------------------------


def test_dirty_worktree_blocks(tmp_path) -> None:
    _frozen_identity(tmp_path / "expected_identity.json", git_dirty=True)
    auth = verify_realrun_authorization(_make_request(tmp_path))
    assert auth.status == "blocked"
    assert "CODE_IDENTITY_UNCLEAN" in _codes(auth)


def test_wrong_arm_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(tmp_path, arms=("naive", "aware"))
    )
    assert auth.status == "blocked"
    assert "ARM_AUTHORITY_INVALID" in _codes(auth)


def test_implicit_default_two_arm_blocks(tmp_path) -> None:
    # The historical runner default is (naive, aware); it must never authorize.
    auth = verify_realrun_authorization(
        _make_request(tmp_path, arms=("aware", "naive"))
    )
    assert auth.status == "blocked"
    assert "ARM_AUTHORITY_INVALID" in _codes(auth)


def test_missing_input_authority_blocks(tmp_path) -> None:
    _frozen_identity(tmp_path / "expected_identity.json", input_authority=None)
    auth = verify_realrun_authorization(_make_request(tmp_path))
    assert auth.status == "blocked"
    assert "INPUT_AUTHORITY_UNBOUND" in _codes(auth)


def test_changed_input_identity_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(tmp_path, expected_input_freeze_sha256="f" * 64)
    )
    assert auth.status == "blocked"
    assert "INPUT_FREEZE_IDENTITY_CHANGED" in _codes(auth)


def test_input_not_authorized_blocks(tmp_path) -> None:
    _blocked_freeze(tmp_path / "input_freeze.json")
    auth = verify_realrun_authorization(_make_request(tmp_path))
    assert auth.status == "blocked"
    assert "INPUT_FREEZE_NOT_AUTHORIZED" in _codes(auth)


def test_missing_execution_identity_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(
            tmp_path,
            expected_execution_identity_path=tmp_path / "does_not_exist.json",
        )
    )
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_NOT_FROZEN" in _codes(auth)


def test_resume_diagnostic_run_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(
            tmp_path, fresh_run=True, resume_run_id="run_20260711T063414_7e96d3"
        )
    )
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_non_fresh_run_flag_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(_make_request(tmp_path, fresh_run=False))
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_cross_run_memory_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(_make_request(tmp_path, cross_run_memory=True))
    assert auth.status == "blocked"
    assert "CROSS_RUN_MEMORY_ENABLED" in _codes(auth)


def test_task_coverage_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(tmp_path, requested_task_ids=FIGURE2_TASK_IDS[:-1])
    )
    assert auth.status == "blocked"
    assert "TASK_COVERAGE_INVALID" in _codes(auth)


def test_rubric_mismatch_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(tmp_path, expected_rubric_sha256="a" * 64)
    )
    assert auth.status == "blocked"
    assert "RUBRIC_IDENTITY_INVALID" in _codes(auth)


def test_output_root_not_absolute_blocks(tmp_path) -> None:
    auth = verify_realrun_authorization(
        _make_request(tmp_path, output_root=Path("relative/out"))
    )
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_INVALID" in _codes(auth)


# ---------------------------------------------------------------------------
# reality anchor: the committed input freeze is still blocked (honest)
# ---------------------------------------------------------------------------


def test_committed_input_freeze_is_not_yet_authorized(tmp_path) -> None:
    # The real full6 input is not yet frozen for a real run; the receipt must
    # fail closed against the committed ledger even with a valid identity/rubric.
    if not _REAL_INPUT_FREEZE.is_file() or not _REAL_RUBRIC.is_file():
        pytest.skip("committed canonical input freeze / rubric not present")
    auth = verify_realrun_authorization(
        _make_request(
            tmp_path,
            input_freeze_path=_REAL_INPUT_FREEZE,
            rubric_path=_REAL_RUBRIC,
        )
    )
    assert auth.status == "blocked"
    assert "INPUT_FREEZE_NOT_AUTHORIZED" in _codes(auth)


def test_authorization_schema_never_reaches_paper_acceptance() -> None:
    # This receipt is pre-run only; it must not be confused with the acceptance
    # gate.  Its schema id is distinct and it carries no evaluation attempts.
    auth = RealRunAuthorization(
        schema_version="easyicu.figure2_realrun_authorization/1",
        status="blocked",
        arms=("aware",),
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        requested_task_ids=tuple(FIGURE2_TASK_IDS),
        fresh_run=True,
        cross_run_memory_disabled=True,
        output_root="/tmp/out",
        issues=(
            {
                "code": "INPUT_FREEZE_NOT_AUTHORIZED",
                "detail": "input not frozen",
                "task_id": None,
            },
        ),
    )
    assert auth.schema_version == "easyicu.figure2_realrun_authorization/1"
