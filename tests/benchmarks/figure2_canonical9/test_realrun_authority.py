"""FIG2-REALRUN-AUTHORITY-P4 (P1 closure) — operator-pinned pre-run freeze gate.

Every authority pin has a fail-closed negative.  The positive uses a SYNTHETIC
typed production input authority (a distinct contract), never a forged
``canonical_input_freeze_v1`` (which the strict loader forces to stay blocked).
The real launcher is exercised to prove a blocked authority fails closed with
zero Provider / runner / data-load activity.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
)

from benchmarks.figure2_canonical9.evaluator.input_freeze_v1 import (
    CanonicalInputFreezeError,
    canonical_input_freeze_manifest_sha256,
    load_canonical_input_freeze_manifest,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from benchmarks.figure2_canonical9.realrun_authority import (
    OPERATOR_FREEZE_DECLARATION_SCHEMA,
    OperatorFreezeDeclaration,
    ProductionInputAuthority,
    ProductionInputTask,
    RealRunAuthorization,
    RealRunAuthorizationRequest,
    load_production_input_authority,
    verify_realrun_authorization,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_V1 = _REPO_ROOT / "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json"
_REAL_RUBRIC = _REPO_ROOT / "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json"


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _production_authority(path: Path) -> ProductionInputAuthority:
    tasks = [
        ProductionInputTask(
            task_id=t, input_sha256="a" * 64, provenance_sha256="b" * 64
        )
        for t in FIGURE2_TASK_IDS
    ]
    authority = ProductionInputAuthority.build(
        submission_profile_ref="npj_dm/20260718", tasks=tasks
    )
    path.write_text(authority.model_dump_json(), encoding="utf-8")
    return authority


def _frozen_identity(
    path: Path, *, git_dirty: bool = False, input_authority: str = "c" * 64
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


def _declaration(path: Path, **overrides) -> OperatorFreezeDeclaration:
    base = dict(
        schema_version=OPERATOR_FREEZE_DECLARATION_SCHEMA,
        expected_execution_identity_sha256="0" * 64,
        input_authority_digest="0" * 64,
        input_freeze_manifest_sha256="0" * 64,
        rubric_sha256="0" * 64,
        code_commit_sha="a" * 40,
        runner="docker",
        network_policy="none",
        runner_image_digest="b" * 64,
        task_ids=tuple(FIGURE2_TASK_IDS),
        arms=("aware",),
        cross_run_memory=False,
        output_root="/tmp/canonical9_out",
        run_id="run_20260722T000000_abcdef",
    )
    base.update(overrides)
    declaration = OperatorFreezeDeclaration(**base)
    path.write_text(declaration.model_dump_json(), encoding="utf-8")
    return declaration


def _authorized_setup(tmp_path: Path):
    prod_path = tmp_path / "prod_authority.json"
    authority = _production_authority(prod_path)
    id_path = tmp_path / "identity.json"
    _frozen_identity(id_path, input_authority=authority.authority_digest)
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(json.dumps({"rubric": "v3"}), encoding="utf-8")
    decl_path = tmp_path / "declaration.json"
    _declaration(
        decl_path,
        expected_execution_identity_sha256=_sha256_file(id_path),
        input_authority_digest=authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(rubric_path),
        output_root=str(tmp_path / "out"),
    )
    request = RealRunAuthorizationRequest(
        declaration_path=decl_path,
        expected_execution_identity_path=id_path,
        input_freeze_path=_REAL_V1,
        rubric_path=rubric_path,
        output_root=tmp_path / "out",
        production_input_authority_path=prod_path,
    )
    return request, {
        "prod_path": prod_path,
        "id_path": id_path,
        "rubric_path": rubric_path,
        "decl_path": decl_path,
        "authority": authority,
    }


def _codes(auth: RealRunAuthorization) -> set[str]:
    return {issue.code for issue in auth.issues}


# ---------------------------------------------------------------------------
# positive (synthetic production authority, NOT a forged v1)
# ---------------------------------------------------------------------------


def test_authorized_with_synthetic_production_authority(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(request)
    assert auth.status == "authorized", auth.issues
    assert auth.issues == ()
    assert auth.input_authority_digest is not None


# ---------------------------------------------------------------------------
# P1.1 — strict loader reuse; v1 can never be full-9 authority
# ---------------------------------------------------------------------------


def test_strict_loader_rejects_cases_empty() -> None:
    # cases=[] is rejected by the strict typed loader (not silently authorizable).
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(
            {
                "schema_version": "easyicu.figure2_canonical_input_freeze/1",
                "manifest_ref": "figure2_canonical9/input_freeze/20260718",
                "submission_profile": {
                    "ref": "npj_dm/20260718",
                    "concept_dict_sha256": "d" * 64,
                    "sofa2_dict_sha256": "e" * 64,
                },
                "cases": [],
            },
            handle,
        )
        empty_path = Path(handle.name)
    with pytest.raises(CanonicalInputFreezeError):
        load_canonical_input_freeze_manifest(empty_path)


def test_v1_assessment_rejected_as_production_authority(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Point the production authority at the real blocked v1 assessment manifest.
    request = RealRunAuthorizationRequest(
        **{**request.__dict__, "production_input_authority_path": _REAL_V1}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_committed_v1_loads_strictly_and_is_blocked() -> None:
    if not _REAL_V1.is_file():
        pytest.skip("committed canonical input freeze not present")
    manifest = load_canonical_input_freeze_manifest(_REAL_V1)
    assert tuple(c.case_id for c in manifest.cases) == ("e2", "e3", "h2")
    assert all(c.state == "blocked" for c in manifest.cases)


def test_production_authority_wrong_task_set_rejected(tmp_path) -> None:
    path = tmp_path / "bad_prod.json"
    tasks = [
        {"task_id": t, "input_sha256": "a" * 64, "provenance_sha256": "b" * 64}
        for t in FIGURE2_TASK_IDS[:-1]  # only 8 tasks
    ]
    body = {
        "schema_version": "easyicu.figure2_production_input_authority/1",
        "submission_profile_ref": "npj_dm/20260718",
        "tasks": tasks,
    }
    from benchmarks.figure2_canonical9.realrun_authority import _canonical_json_bytes

    digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    path.write_text(json.dumps({**body, "authority_digest": digest}), encoding="utf-8")
    with pytest.raises(Exception):
        load_production_input_authority(path)


# ---------------------------------------------------------------------------
# P1.2 — operator-pinned declaration; verify actual == pin; no self-cert
# ---------------------------------------------------------------------------


def test_missing_pin_declaration_invalid(tmp_path) -> None:
    decl_path = tmp_path / "decl.json"
    # A declaration missing a required pin (input_authority_digest) must be rejected.
    decl_path.write_text(
        json.dumps(
            {
                "schema_version": OPERATOR_FREEZE_DECLARATION_SCHEMA,
                "expected_execution_identity_sha256": "0" * 64,
                "input_freeze_manifest_sha256": "0" * 64,
                "rubric_sha256": "0" * 64,
                "code_commit_sha": "a" * 40,
                "runner": "docker",
                "network_policy": "none",
                "runner_image_digest": "b" * 64,
                "task_ids": list(FIGURE2_TASK_IDS),
                "arms": ["aware"],
                "cross_run_memory": False,
                "output_root": "/tmp/out",
                "run_id": "run_x",
            }
        ),
        encoding="utf-8",
    )
    request = RealRunAuthorizationRequest(
        declaration_path=decl_path,
        expected_execution_identity_path=tmp_path / "id.json",
        input_freeze_path=_REAL_V1,
        rubric_path=_REAL_RUBRIC,
        output_root=tmp_path / "out",
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OPERATOR_DECLARATION_INVALID" in _codes(auth)


def test_swapped_rubric_after_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Swap the rubric bytes AFTER the operator pinned it -> self-certification fails.
    paths["rubric_path"].write_text(
        json.dumps({"rubric": "TAMPERED"}), encoding="utf-8"
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "RUBRIC_IDENTITY_INVALID" in _codes(auth)


def test_swapped_identity_after_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    _frozen_identity(paths["id_path"], input_authority="f" * 64)  # different bytes
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_identity_input_authority_must_equal_production_digest(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Rebuild the identity bound to a DIFFERENT input authority than production.
    _frozen_identity(paths["id_path"], input_authority="9" * 64)
    _declaration(
        paths["decl_path"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=paths["authority"].authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(request.output_root),
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_dirty_tree_identity_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    frozen = _frozen_identity(
        paths["id_path"],
        git_dirty=True,
        input_authority=paths["authority"].authority_digest,
    )
    _declaration(
        paths["decl_path"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=paths["authority"].authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(request.output_root),
    )
    assert frozen.execution_identity.paper_eligible is False
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.4 — fresh run enforced (no resume; new, empty output root)
# ---------------------------------------------------------------------------


def test_resume_run_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = RealRunAuthorizationRequest(
        **{**request.__dict__, "resume_run_id": "run_20260711T063414_7e96d3"}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_resume_from_step_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = RealRunAuthorizationRequest(
        **{**request.__dict__, "resume_from_step_id": "04_primary"}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_non_empty_output_root_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale_checkpoint.json").write_text("{}", encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_NOT_FRESH" in _codes(auth)


def test_existing_run_dir_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    out = tmp_path / "out"
    (out / "run_20260722T000000_abcdef").mkdir(parents=True)
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_NOT_FRESH" in _codes(auth)


def test_cross_run_memory_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = RealRunAuthorizationRequest(
        **{**request.__dict__, "cross_run_memory": True}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "CROSS_RUN_MEMORY_ENABLED" in _codes(auth)


# ---------------------------------------------------------------------------
# reality anchor + missing production authority
# ---------------------------------------------------------------------------


def test_no_production_authority_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = RealRunAuthorizationRequest(
        **{**request.__dict__, "production_input_authority_path": None}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_ABSENT" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.3 / P1.5 — real launcher fails closed with ZERO Provider / runner calls
# ---------------------------------------------------------------------------


def test_real_launcher_blocks_with_zero_provider_runner_calls(tmp_path, monkeypatch):
    import tools.run_research_agent_bench as bench

    calls = {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}

    def _spy(name):
        def _fn(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run when authorization is blocked")

        return _fn

    monkeypatch.setattr(bench, "_make_llm", _spy("llm"))
    monkeypatch.setattr(bench, "_run_suite", _spy("suite"))
    monkeypatch.setattr(bench, "_run_ehrflowbench_jsonl", _spy("ehrflow"))

    def _register_spy(*args, **kwargs):
        calls["register"] += 1
        raise AssertionError("case registration must not run when blocked")

    monkeypatch.setattr(bench, "_register_case_patterns", _register_spy)

    # Valid declaration + identity, but NO production authority -> blocked.
    id_path = tmp_path / "identity.json"
    authority_digest = _production_authority(tmp_path / "prod.json").authority_digest
    _frozen_identity(id_path, input_authority=authority_digest)
    decl_path = tmp_path / "declaration.json"
    _declaration(
        decl_path,
        expected_execution_identity_sha256=_sha256_file(id_path),
        input_authority_digest=authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(_REAL_RUBRIC),
        output_root=str(tmp_path / "out"),
    )

    argv = [
        "run_research_agent_bench.py",
        "--figure2-realrun-authorization",
        str(decl_path),
        "--figure2-expected-execution-identity",
        str(id_path),
        "--out-root",
        str(tmp_path / "out"),
        "--arms",
        "aware",
        "--provider",
        "openrouter",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = bench.main()
    assert rc == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_requires_identity_with_declaration(tmp_path, monkeypatch):
    import tools.run_research_agent_bench as bench

    monkeypatch.setattr(
        bench, "_register_case_patterns", lambda *a, **k: pytest.fail("ran")
    )
    decl_path = tmp_path / "declaration.json"
    _declaration(decl_path, output_root=str(tmp_path / "out"))
    argv = [
        "run_research_agent_bench.py",
        "--figure2-realrun-authorization",
        str(decl_path),
        "--out-root",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert bench.main() == 2
