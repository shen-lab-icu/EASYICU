"""FIG2-REALRUN-AUTHORITY-P4 — the authorized declaration must equal the run.

The gate no longer merely proves "a declaration is well formed"; it proves the
declaration matches the ACTUAL parsed invocation knob-for-knob AND that the real
per-task cohorts hash to the frozen production input authority.  Every binding has
a fail-closed negative.  The positive uses a SYNTHETIC typed production input
authority (a distinct contract), never a forged ``canonical_input_freeze_v1``
(which the strict loader forces to stay blocked).  The real launcher is exercised
to prove that a bypass attempt fails closed with zero Provider / runner / pipeline
activity.
"""

from __future__ import annotations

import dataclasses
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
    RealRunInvocation,
    load_production_input_authority,
    production_cohort_input_sha256,
    verify_realrun_authorization,
    verify_results_frozen_input_authority,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_V1 = _REPO_ROOT / "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json"
_REAL_RUBRIC = _REPO_ROOT / "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json"

_PROVIDER = "openrouter"
_MODEL = "gpt-5.5"
_PROFILE_REF = "npj_dm/20260718"
_COMMIT = "a" * 40
_IMAGE = "b" * 64
_CLEAN_LIVE = {"git_sha": _COMMIT, "git_dirty": False}


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cohorts(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "cohorts"
    root.mkdir(exist_ok=True)
    out: dict[str, Path] = {}
    for idx, task_id in enumerate(FIGURE2_TASK_IDS):
        path = root / f"{task_id}.parquet"
        path.write_bytes(f"cohort::{task_id}::{idx}".encode("utf-8") * 64)
        out[task_id] = path
    return out


def _write_jsonl(path: Path, cohort_paths: dict[str, Path]) -> str:
    lines = []
    for task_id in FIGURE2_TASK_IDS:
        lines.append(
            json.dumps(
                {
                    "key": task_id,
                    "cohort_path": str(cohort_paths[task_id]),
                    "question": f"canonical question {task_id}",
                    "target_outcome": "mortality",
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(path)


def _production_authority(
    path: Path, cohort_paths: dict[str, Path]
) -> ProductionInputAuthority:
    tasks = [
        ProductionInputTask(
            task_id=task_id,
            input_sha256=production_cohort_input_sha256(cohort_paths[task_id]),
            provenance_sha256=hashlib.sha256(task_id.encode("utf-8")).hexdigest(),
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    authority = ProductionInputAuthority.build(
        submission_profile_ref=_PROFILE_REF, tasks=tasks
    )
    path.write_text(authority.model_dump_json(), encoding="utf-8")
    return authority


def _frozen_identity(
    path: Path, *, git_dirty: bool = False, input_authority: str
) -> ExpectedExecutionIdentity:
    identity = ExecutionIdentity.create(
        submission_profile_name="npj_dm",
        submission_profile_version="20260718",
        runner="docker",
        runner_image_digest=_IMAGE,
        network_policy="none",
        llm_seed=0,
        input_authority_sha256=input_authority,
        provider_authorization={"clients": [{"authorization_mode": "operator_env"}]},
        host_runner_authorized=False,
        code_version={"git_sha": _COMMIT, "git_dirty": git_dirty},
    )
    frozen = ExpectedExecutionIdentity.create(identity)
    path.write_text(frozen.model_dump_json(), encoding="utf-8")
    return frozen


def _declaration(path: Path, *, jsonl_path: Path, jsonl_sha: str, **overrides):
    base = dict(
        schema_version=OPERATOR_FREEZE_DECLARATION_SCHEMA,
        expected_execution_identity_sha256="0" * 64,
        input_authority_digest="0" * 64,
        input_freeze_manifest_sha256="0" * 64,
        rubric_sha256="0" * 64,
        code_commit_sha=_COMMIT,
        runner="docker",
        network_policy="none",
        runner_image_digest=_IMAGE,
        provider=_PROVIDER,
        model=_MODEL,
        submission_profile_ref=_PROFILE_REF,
        ehrflowbench_jsonl_path=str(jsonl_path),
        ehrflowbench_jsonl_sha256=jsonl_sha,
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


def _invocation(
    *, cohort_paths: dict[str, Path], jsonl_path: Path, out_root: Path, **overrides
) -> RealRunInvocation:
    base = dict(
        arms=("aware",),
        task_ids=tuple(FIGURE2_TASK_IDS),
        task_cohort_paths=tuple(
            (task_id, str(cohort_paths[task_id])) for task_id in FIGURE2_TASK_IDS
        ),
        ehrflowbench_jsonl_path=jsonl_path,
        provider=_PROVIDER,
        model=_MODEL,
        submission_profile_enabled=True,
        submission_profile_ref=_PROFILE_REF,
        runner="docker",
        out_root=out_root,
        require_paper_acceptance=True,
        reuse_existing=False,
        repeat=1,
        force_writer_probe=False,
        development_sample_size=None,
        allow_host_runner=False,
        allow_mock_aware=False,
        resume_run_id=None,
        resume_from_step_id=None,
        cross_run_memory=False,
    )
    base.update(overrides)
    return RealRunInvocation(**base)


def _authorized_setup(tmp_path: Path):
    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    jsonl_sha = _write_jsonl(jsonl_path, cohort_paths)
    prod_path = tmp_path / "prod_authority.json"
    authority = _production_authority(prod_path, cohort_paths)
    id_path = tmp_path / "identity.json"
    _frozen_identity(id_path, input_authority=authority.authority_digest)
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(json.dumps({"rubric": "v3"}), encoding="utf-8")
    decl_path = tmp_path / "declaration.json"
    out_root = tmp_path / "out"
    _declaration(
        decl_path,
        jsonl_path=jsonl_path,
        jsonl_sha=jsonl_sha,
        expected_execution_identity_sha256=_sha256_file(id_path),
        input_authority_digest=authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(rubric_path),
        output_root=str(out_root),
    )
    invocation = _invocation(
        cohort_paths=cohort_paths, jsonl_path=jsonl_path, out_root=out_root
    )
    request = RealRunAuthorizationRequest(
        declaration_path=decl_path,
        expected_execution_identity_path=id_path,
        input_freeze_path=_REAL_V1,
        rubric_path=rubric_path,
        invocation=invocation,
        production_input_authority_path=prod_path,
        live_code_version=_CLEAN_LIVE,
    )
    return request, {
        "cohort_paths": cohort_paths,
        "jsonl_path": jsonl_path,
        "jsonl_sha": jsonl_sha,
        "prod_path": prod_path,
        "id_path": id_path,
        "rubric_path": rubric_path,
        "decl_path": decl_path,
        "out_root": out_root,
        "authority": authority,
    }


def _codes(auth: RealRunAuthorization) -> set[str]:
    return {issue.code for issue in auth.issues}


def _with_invocation(request, **overrides):
    return dataclasses.replace(
        request, invocation=dataclasses.replace(request.invocation, **overrides)
    )


# ---------------------------------------------------------------------------
# positive (synthetic production authority + matched invocation)
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


def test_strict_loader_rejects_cases_empty(tmp_path) -> None:
    empty_path = tmp_path / "empty_freeze.json"
    empty_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.figure2_canonical_input_freeze/1",
                "manifest_ref": "figure2_canonical9/input_freeze/20260718",
                "submission_profile": {
                    "ref": _PROFILE_REF,
                    "concept_dict_sha256": "d" * 64,
                    "sofa2_dict_sha256": "e" * 64,
                },
                "cases": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CanonicalInputFreezeError):
        load_canonical_input_freeze_manifest(empty_path)


def test_v1_assessment_rejected_as_production_authority(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = dataclasses.replace(request, production_input_authority_path=_REAL_V1)
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
    from benchmarks.figure2_canonical9.realrun_authority import _canonical_json_bytes

    path = tmp_path / "bad_prod.json"
    tasks = [
        {"task_id": t, "input_sha256": "a" * 64, "provenance_sha256": "b" * 64}
        for t in FIGURE2_TASK_IDS[:-1]  # only 8 tasks
    ]
    body = {
        "schema_version": "easyicu.figure2_production_input_authority/1",
        "submission_profile_ref": _PROFILE_REF,
        "tasks": tasks,
    }
    digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
    path.write_text(json.dumps({**body, "authority_digest": digest}), encoding="utf-8")
    with pytest.raises(Exception):
        load_production_input_authority(path)


# ---------------------------------------------------------------------------
# P1.2 — declaration <-> real invocation, knob for knob
# ---------------------------------------------------------------------------


def test_naive_arm_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(_with_invocation(request, arms=("naive",)))
    assert auth.status == "blocked"
    assert "INVOCATION_ARM_NOT_AWARE" in _codes(auth)


def test_default_both_arms_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, arms=("naive", "aware"))
    )
    assert auth.status == "blocked"
    assert "INVOCATION_ARM_NOT_AWARE" in _codes(auth)


def test_subset_tasks_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, task_ids=tuple(FIGURE2_TASK_IDS[:-1]))
    )
    assert auth.status == "blocked"
    assert "INVOCATION_TASKS_NOT_CANONICAL" in _codes(auth)


def test_wrong_jsonl_bytes_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Tamper the JSONL bytes AFTER the operator pinned its sha.
    paths["jsonl_path"].write_text(
        paths["jsonl_path"].read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "INVOCATION_JSONL_MISMATCH" in _codes(auth)


def test_wrong_jsonl_path_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    other = tmp_path / "other.jsonl"
    other.write_text(paths["jsonl_path"].read_text(encoding="utf-8"), encoding="utf-8")
    auth = verify_realrun_authorization(
        _with_invocation(request, ehrflowbench_jsonl_path=other)
    )
    assert auth.status == "blocked"
    assert "INVOCATION_JSONL_MISMATCH" in _codes(auth)


def test_mock_provider_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(_with_invocation(request, provider="mock"))
    assert auth.status == "blocked"
    assert "INVOCATION_PROVIDER_IS_MOCK" in _codes(auth)


def test_wrong_model_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, model="some-other-model")
    )
    assert auth.status == "blocked"
    assert "INVOCATION_MODEL_MISMATCH" in _codes(auth)


def test_profile_disabled_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, submission_profile_enabled=False)
    )
    assert auth.status == "blocked"
    assert "INVOCATION_PROFILE_DISABLED" in _codes(auth)


def test_subprocess_runner_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(_with_invocation(request, runner="subprocess"))
    assert auth.status == "blocked"
    assert "INVOCATION_RUNNER_NOT_DOCKER" in _codes(auth)


def test_host_runner_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, allow_host_runner=True)
    )
    assert auth.status == "blocked"
    assert "INVOCATION_HOST_RUNNER" in _codes(auth)


def test_wrong_output_root_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, out_root=tmp_path / "somewhere_else")
    )
    assert auth.status == "blocked"
    assert "INVOCATION_OUTPUT_ROOT_MISMATCH" in _codes(auth)


def test_repeat_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(_with_invocation(request, repeat=3))
    assert auth.status == "blocked"
    assert "UNSAFE_RUN_FLAG_REPEAT" in _codes(auth)


def test_reuse_existing_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(_with_invocation(request, reuse_existing=True))
    assert auth.status == "blocked"
    assert "UNSAFE_RUN_FLAG_REUSE" in _codes(auth)


def test_force_writer_probe_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, force_writer_probe=True)
    )
    assert auth.status == "blocked"
    assert "UNSAFE_RUN_FLAG_FORCE_WRITER" in _codes(auth)


def test_development_sample_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, development_sample_size=25)
    )
    assert auth.status == "blocked"
    assert "UNSAFE_RUN_FLAG_DEV_SAMPLE" in _codes(auth)


def test_paper_acceptance_not_required_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, require_paper_acceptance=False)
    )
    assert auth.status == "blocked"
    assert "PAPER_ACCEPTANCE_NOT_REQUIRED" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.2 — operator-pinned engine identity; verify actual == pin, no self-cert
# ---------------------------------------------------------------------------


def test_missing_pin_declaration_invalid(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Remove a required pin AFTER a valid setup -> the strict loader rejects it.
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body.pop("input_authority_digest")
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OPERATOR_DECLARATION_INVALID" in _codes(auth)


def test_swapped_rubric_after_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
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
        jsonl_path=paths["jsonl_path"],
        jsonl_sha=paths["jsonl_sha"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=paths["authority"].authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(paths["out_root"]),
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_dirty_frozen_identity_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    frozen = _frozen_identity(
        paths["id_path"],
        git_dirty=True,
        input_authority=paths["authority"].authority_digest,
    )
    _declaration(
        paths["decl_path"],
        jsonl_path=paths["jsonl_path"],
        jsonl_sha=paths["jsonl_sha"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=paths["authority"].authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(paths["out_root"]),
    )
    assert frozen.execution_identity.paper_eligible is False
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_wrong_commit_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["code_commit_sha"] = "c" * 40  # identity git_sha is "a"*40
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_wrong_image_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["runner_image_digest"] = "c" * 64
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_wrong_network_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["network_policy"] = "disabled"  # identity network is "none"
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


def test_wrong_profile_ref_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["submission_profile_ref"] = "other_profile/20260101"
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    # Both the identity profile pin and the production authority profile disagree.
    assert "EXECUTION_IDENTITY_MISMATCH" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.2 — live checkout must equal the commit pin and be clean
# ---------------------------------------------------------------------------


def test_live_dirty_tree_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = dataclasses.replace(
        request, live_code_version={"git_sha": _COMMIT, "git_dirty": True}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "LIVE_CHECKOUT_MISMATCH" in _codes(auth)


def test_live_commit_changed_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = dataclasses.replace(
        request, live_code_version={"git_sha": "d" * 40, "git_dirty": False}
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "LIVE_CHECKOUT_MISMATCH" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.3 — per-task input authority <-> runtime digest (one algorithm)
# ---------------------------------------------------------------------------


def test_cohort_replaced_after_pin_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Swap one cohort's bytes AFTER the production authority pinned its digest.
    victim = paths["cohort_paths"][FIGURE2_TASK_IDS[0]]
    victim.write_bytes(b"SWAPPED COHORT PAYLOAD" * 100)
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_production_input_algorithm_matches_launcher(tmp_path) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "c.parquet"
    cohort.write_bytes(b"rows" * 777)
    assert production_cohort_input_sha256(
        cohort
    ) == bench._benchmark_input_authority_sha256(str(cohort))


def test_bind_enforces_frozen_input_digest(tmp_path) -> None:
    import tools.run_research_agent_bench as bench

    cohort = tmp_path / "c.parquet"
    cohort.write_bytes(b"x" * 4096)
    real = bench._benchmark_input_authority_sha256(str(cohort))
    # Matching per-task override binds cleanly.
    bound = bench._bind_benchmark_execution_input(
        {"execution_input_authority_sha256": real}, cohort=str(cohort), data_seed=7
    )
    assert bound["execution_input_authority_sha256"] == real
    # A frozen override that does not match the runtime cohort fails closed mid-run.
    with pytest.raises(ValueError):
        bench._bind_benchmark_execution_input(
            {"execution_input_authority_sha256": "0" * 64},
            cohort=str(cohort),
            data_seed=7,
        )


def test_post_run_input_authority_match() -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    payload = {
        "scores": [
            {
                "item_key": t,
                "aware": {"execution_identity": {"input_authority_sha256": frozen[t]}},
            }
            for t in FIGURE2_TASK_IDS
        ]
    }
    assert verify_results_frozen_input_authority(payload, frozen) == []


def test_post_run_input_authority_mismatch_and_missing() -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    scores = []
    for t in FIGURE2_TASK_IDS[:-1]:  # last task produces no aware score
        digest = "0" * 64 if t == FIGURE2_TASK_IDS[0] else frozen[t]
        scores.append(
            {
                "item_key": t,
                "aware": {"execution_identity": {"input_authority_sha256": digest}},
            }
        )
    mismatches = dict(verify_results_frozen_input_authority({"scores": scores}, frozen))
    assert FIGURE2_TASK_IDS[0] in mismatches  # wrong digest
    assert FIGURE2_TASK_IDS[-1] in mismatches  # never produced


# ---------------------------------------------------------------------------
# P1.4 — fresh run enforced (no resume; new, empty output root)
# ---------------------------------------------------------------------------


def test_resume_run_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, resume_run_id="run_20260711T063414_7e96d3")
    )
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_resume_from_step_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, resume_from_step_id="04_primary")
    )
    assert auth.status == "blocked"
    assert "NON_FRESH_RUN" in _codes(auth)


def test_non_empty_output_root_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    paths["out_root"].mkdir()
    (paths["out_root"] / "stale_checkpoint.json").write_text("{}", encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_NOT_FRESH" in _codes(auth)


def test_existing_run_dir_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    (paths["out_root"] / "run_20260722T000000_abcdef").mkdir(parents=True)
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_NOT_FRESH" in _codes(auth)


def test_cross_run_memory_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, cross_run_memory=True)
    )
    assert auth.status == "blocked"
    assert "CROSS_RUN_MEMORY_ENABLED" in _codes(auth)


# ---------------------------------------------------------------------------
# reality anchor: no typed production authority exists today
# ---------------------------------------------------------------------------


def test_no_production_authority_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    request = dataclasses.replace(request, production_input_authority_path=None)
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_ABSENT" in _codes(auth)


# ---------------------------------------------------------------------------
# P1.1 / P1.3 / P1.5 — the REAL launcher fails closed with ZERO calls
# ---------------------------------------------------------------------------


def _spies(monkeypatch):
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
    monkeypatch.setattr(bench, "_register_case_patterns", _spy("register"))
    return bench, calls


def _launcher_files(tmp_path, *, with_identity=True):
    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    jsonl_sha = _write_jsonl(jsonl_path, cohort_paths)
    id_path = tmp_path / "identity.json"
    authority = _production_authority(tmp_path / "prod.json", cohort_paths)
    if with_identity:
        _frozen_identity(id_path, input_authority=authority.authority_digest)
    decl_path = tmp_path / "declaration.json"
    _declaration(
        decl_path,
        jsonl_path=jsonl_path,
        jsonl_sha=jsonl_sha,
        expected_execution_identity_sha256=(
            _sha256_file(id_path) if with_identity else "0" * 64
        ),
        input_authority_digest=authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(_REAL_RUBRIC),
        output_root=str(tmp_path / "out"),
    )
    return {
        "jsonl_path": jsonl_path,
        "id_path": id_path,
        "decl_path": decl_path,
        "out_root": tmp_path / "out",
    }


def test_launcher_require_acceptance_without_authorization_blocks(
    tmp_path, monkeypatch, capsys
):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--require-figure2-paper-acceptance",
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "REQUIRES" in capsys.readouterr().err


def test_launcher_canonical_jsonl_without_authorization_blocks(
    tmp_path, monkeypatch, capsys
):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_declaration_without_identity_blocks(tmp_path, monkeypatch):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path, with_identity=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(files["decl_path"]),
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_no_production_authority_blocks(tmp_path, monkeypatch, capsys):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(files["decl_path"]),
            "--figure2-expected-execution-identity",
            str(files["id_path"]),
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
            "--provider",
            _PROVIDER,
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "PRODUCTION_INPUT_AUTHORITY_ABSENT" in capsys.readouterr().out


def test_launcher_naive_arm_blocks_at_gate(tmp_path, monkeypatch, capsys):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(files["decl_path"]),
            "--figure2-expected-execution-identity",
            str(files["id_path"]),
            "--figure2-production-input-authority",
            str(tmp_path / "prod.json"),
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "naive",
            "--provider",
            _PROVIDER,
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "INVOCATION_ARM_NOT_AWARE" in capsys.readouterr().out


def test_launcher_mock_provider_blocks_at_gate(tmp_path, monkeypatch, capsys):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(files["decl_path"]),
            "--figure2-expected-execution-identity",
            str(files["id_path"]),
            "--figure2-production-input-authority",
            str(tmp_path / "prod.json"),
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
            "--provider",
            "mock",
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "INVOCATION_PROVIDER_IS_MOCK" in capsys.readouterr().out


def test_launcher_wrong_output_root_blocks_at_gate(tmp_path, monkeypatch, capsys):
    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(files["decl_path"]),
            "--figure2-expected-execution-identity",
            str(files["id_path"]),
            "--figure2-production-input-authority",
            str(tmp_path / "prod.json"),
            "--ehrflowbench-jsonl",
            str(files["jsonl_path"]),
            "--out-root",
            str(tmp_path / "elsewhere"),  # != declaration.output_root
            "--arms",
            "aware",
            "--provider",
            _PROVIDER,
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "INVOCATION_OUTPUT_ROOT_MISMATCH" in capsys.readouterr().out


def test_gate_non_canonical_run_without_declaration_proceeds() -> None:
    """A normal (non-canonical, non-paper) dev run is NOT gated."""
    import argparse

    import tools.run_research_agent_bench as bench

    args = argparse.Namespace(
        figure2_realrun_authorization=None,
        ehrflowbench_jsonl=None,
        require_figure2_paper_acceptance=False,
    )
    rc, frozen = bench._figure2_realrun_authorization_gate(args)
    assert rc is None
    assert frozen is None
