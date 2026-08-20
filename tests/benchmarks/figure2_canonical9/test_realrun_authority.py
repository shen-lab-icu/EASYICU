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
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.execution_identity import (
    ExecutionIdentity,
    ExpectedExecutionIdentity,
)
from easyicu.research_agent.authority.provider_hard_stop import (
    load_provider_hard_stop_ledger,
)
from easyicu.research_agent.know_how.registry import (
    reviewable_card_content_sha256,
)

from benchmarks.figure2_canonical9.evaluator.input_freeze_v1 import (
    CanonicalInputFreezeError,
    canonical_input_freeze_manifest_sha256,
    load_canonical_input_freeze_manifest,
)
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedCohortAuthorityRef,
)

from benchmarks.figure2_canonical9.realrun_authority import (
    OPERATOR_FREEZE_DECLARATION_SCHEMA,
    RealRunAuthorization,
    RealRunAuthorizationRequest,
    RealRunBatchBinding,
    RealRunInvocation,
    OperatorFreezeDeclaration,
    ProductionInputAuthority,
    ProductionInputTask,
    build_batch_ledger,
    build_canonical_execution_config,
    load_production_input_authority,
    production_cohort_input_sha256,
    production_provenance_sha256,
    reserve_authorized_batch_root,
    resolve_strict_jsonl_path,
    verify_batch_authorization_receipt,
    verify_realrun_authorization,
    verify_results_frozen_input_authority,
)
from benchmarks.figure2_canonical9.scientific_protocol_authority import (
    REQUIRED_SCIENTIFIC_PROTOCOLS,
    ScientificProtocolAuthority,
    ScientificProtocolTaskBinding,
)
from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    case_protocol_content_sha256,
    default_case_protocol_path,
    load_case_scientific_protocol,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_V1 = _REPO_ROOT / "benchmarks/figure2_canonical9/canonical_input_freeze_v1.json"
_REAL_RUBRIC = _REPO_ROOT / "benchmarks/figure2_canonical9/figure2_paper_rubric_v3.json"

_PROVIDER = "openrouter"
_MODEL = "gpt-5.5"
_PROFILE_REF = "npj_dm/20260718"
_COMMIT = "a" * 40
_IMAGE = "b" * 64
_BATCH_ID = "batch_20260722T000000_abcdef"
_CLEAN_LIVE = {"git_sha": _COMMIT, "git_dirty": False}


def _execution_config():
    """The canonical execution config the launcher builds from DEFAULT argv."""

    return build_canonical_execution_config(
        seed=7,
        timeout_seconds=900.0,
        standard_executor_timeout_seconds=3600.0,
    )


def _config_sha() -> str:
    return _execution_config().digest()


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cohorts(tmp_path: Path) -> dict[str, Path]:
    """A typed cohort parquet + an adjacent materialization-authority sidecar."""

    root = tmp_path / "cohorts"
    root.mkdir(exist_ok=True)
    out: dict[str, Path] = {}
    for idx, task_id in enumerate(FIGURE2_TASK_IDS):
        path = root / f"{task_id}.parquet"
        path.write_bytes(f"cohort::{task_id}::{idx}".encode("utf-8") * 64)
        sidecar = root / f"{task_id}.authority.json"
        sidecar.write_text(
            json.dumps({"task": task_id, "materialization": idx}), encoding="utf-8"
        )
        out[task_id] = path
    return out


def _cohort_ref(cohort_path: Path) -> dict:
    sidecar = cohort_path.parent / f"{cohort_path.stem}.authority.json"
    return MaterializedCohortAuthorityRef(
        file=sidecar.name,
        sha256=_sha256_file(sidecar),
        size=sidecar.stat().st_size,
    ).to_dict()


def _write_jsonl(path: Path, cohort_paths: dict[str, Path]) -> str:
    lines = []
    for task_id in FIGURE2_TASK_IDS:
        cohort = cohort_paths[task_id]
        row = {
            "key": task_id,
            "cohort_path": str(cohort),
            "cohort_authority_path": str(
                cohort.parent / f"{cohort.stem}.authority.json"
            ),
            "cohort_authority_ref": _cohort_ref(cohort),
            "question": f"canonical question {task_id}",
            "target_outcome": "mortality",
        }
        if task_id in {task for task, _card in REQUIRED_SCIENTIFIC_PROTOCOLS}:
            protocol = load_case_scientific_protocol(
                default_case_protocol_path(task_id), expected_task_id=task_id
            )
            projection = build_runtime_scientific_projection(protocol)
            row.update(
                {
                    "case_scientific_protocol_sha256": (
                        projection.protocol_content_sha256
                    ),
                    "runtime_scientific_projection": projection.model_dump(mode="json"),
                    "runtime_scientific_projection_sha256": (
                        projection.runtime_projection_sha256
                    ),
                    "expected_outputs": list(projection.agent_visible_required_outputs),
                    "semantic_guardrails": list(projection.agent_visible_guardrails),
                    "notes": projection.canonical_protocol_json,
                }
            )
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(path)


def _production_authority(
    path: Path, cohort_paths: dict[str, Path]
) -> ProductionInputAuthority:
    tasks = [
        ProductionInputTask(
            task_id=task_id,
            input_sha256=production_cohort_input_sha256(cohort_paths[task_id]),
            provenance_sha256=production_provenance_sha256(
                _cohort_ref(cohort_paths[task_id]), None
            ),
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    authority = ProductionInputAuthority.build(
        submission_profile_ref=_PROFILE_REF, tasks=tasks
    )
    path.write_text(authority.model_dump_json(), encoding="utf-8")
    return authority


def _scientific_protocol_authority(
    path: Path,
) -> ScientificProtocolAuthority:
    card_root = path.parent / "reviewed_protocols"
    card_root.mkdir(exist_ok=True)
    bindings: list[ScientificProtocolTaskBinding] = []
    for task_id, card_id in REQUIRED_SCIENTIFIC_PROTOCOLS:
        payload = json.loads(
            (
                _REPO_ROOT / "src/easyicu/data/research_know_how" / f"{card_id}.json"
            ).read_text(encoding="utf-8")
        )
        payload["review_status"] = "clinical_reviewed"
        payload["review_attestation"] = None
        reviewed_content_sha256 = reviewable_card_content_sha256(payload)
        protocol_path = card_root / f"{task_id}.protocol.json"
        protocol_path.write_bytes(default_case_protocol_path(task_id).read_bytes())
        protocol = load_case_scientific_protocol(
            protocol_path,
            expected_task_id=task_id,
        )
        protocol_content_sha256 = case_protocol_content_sha256(protocol)
        runtime_projection_sha256 = build_runtime_scientific_projection(
            protocol
        ).runtime_projection_sha256
        payload["review_attestation"] = {
            "reviewer_owner": "Synthetic clinical-and-methods test board",
            "review_date": "2026-07-26",
            "card_version": payload["version"],
            "reviewed_content_sha256": reviewed_content_sha256,
            "protocol_content_sha256": protocol_content_sha256,
            "runtime_projection_sha256": runtime_projection_sha256,
            "review_scope": ["clinical protocol", "statistical methods"],
            "literature_search_cutoff": "2026-07-25",
            "clinical_reviewed": True,
            "methods_reviewed": True,
        }
        card_path = card_root / f"{task_id}.json"
        card_path.write_text(json.dumps(payload), encoding="utf-8")
        bindings.append(
            ScientificProtocolTaskBinding(
                task_id=task_id,
                card_id=card_id,
                card_version=payload["version"],
                card_path=str(card_path),
                card_file_sha256=_sha256_file(card_path),
                reviewed_content_sha256=reviewed_content_sha256,
                protocol_path=str(protocol_path),
                protocol_file_sha256=_sha256_file(protocol_path),
                protocol_content_sha256=protocol_content_sha256,
                runtime_projection_sha256=runtime_projection_sha256,
            )
        )
    authority = ScientificProtocolAuthority.build(tasks=bindings)
    path.write_text(authority.model_dump_json(), encoding="utf-8")
    return authority


def _frozen_identity(
    path: Path,
    *,
    git_dirty: bool = False,
    input_authority: str,
    profile_name: str = "npj_dm",
    profile_version: str = "20260718",
) -> ExpectedExecutionIdentity:
    identity = ExecutionIdentity.create(
        submission_profile_name=profile_name,
        submission_profile_version=profile_version,
        runner="docker",
        runner_image_digest=_IMAGE,
        network_policy="none",
        llm_seed=0,
        input_authority_sha256=input_authority,
        provider_authorization={
            "schema_version": "easyicu.provider_authorization_manifest/2",
            "reasoning_effort_profile": "provider_default",
            "clients": [
                {
                    "provider": _PROVIDER,
                    "model": _MODEL,
                    "base_url": "https://openrouter.ai/api/v1",
                    "destination": "external",
                    "authorization_mode": "operator_env",
                    "authorization_sha256": "c" * 64,
                }
            ],
        },
        host_runner_authorized=False,
        code_version={"git_sha": _COMMIT, "git_dirty": git_dirty},
    )
    frozen = ExpectedExecutionIdentity.create(identity)
    path.write_text(frozen.model_dump_json(), encoding="utf-8")
    return frozen


def _declaration(path: Path, *, jsonl_path: Path, jsonl_sha: str, **overrides):
    protocol_path = path.parent / "scientific_protocol_authority.json"
    protocol_digest = (
        str(json.loads(protocol_path.read_text(encoding="utf-8"))["authority_digest"])
        if protocol_path.is_file()
        else "0" * 64
    )
    base = dict(
        schema_version=OPERATOR_FREEZE_DECLARATION_SCHEMA,
        expected_execution_identity_sha256="0" * 64,
        input_authority_digest="0" * 64,
        scientific_protocol_authority_digest=protocol_digest,
        input_freeze_manifest_sha256="0" * 64,
        rubric_sha256="0" * 64,
        code_commit_sha=_COMMIT,
        runner="docker",
        network_policy="none",
        runner_image_digest=_IMAGE,
        execution_config_sha256=_config_sha(),
        provider=_PROVIDER,
        model=_MODEL,
        submission_profile_ref=_PROFILE_REF,
        ehrflowbench_jsonl_path=str(jsonl_path),
        ehrflowbench_jsonl_sha256=jsonl_sha,
        task_ids=tuple(FIGURE2_TASK_IDS),
        arms=("aware",),
        cross_run_memory=False,
        output_root=f"/tmp/{_BATCH_ID}",
        batch_id=_BATCH_ID,
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
        execution_config=_execution_config(),
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
    protocol_path = tmp_path / "scientific_protocol_authority.json"
    protocol_authority = _scientific_protocol_authority(protocol_path)
    id_path = tmp_path / "identity.json"
    _frozen_identity(id_path, input_authority=authority.authority_digest)
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(json.dumps({"rubric": "v3"}), encoding="utf-8")
    decl_path = tmp_path / "declaration.json"
    out_root = tmp_path / _BATCH_ID
    _declaration(
        decl_path,
        jsonl_path=jsonl_path,
        jsonl_sha=jsonl_sha,
        expected_execution_identity_sha256=_sha256_file(id_path),
        input_authority_digest=authority.authority_digest,
        scientific_protocol_authority_digest=protocol_authority.authority_digest,
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
        scientific_protocol_authority_path=protocol_path,
        live_code_version=_CLEAN_LIVE,
    )
    return request, {
        "cohort_paths": cohort_paths,
        "jsonl_path": jsonl_path,
        "jsonl_sha": jsonl_sha,
        "prod_path": prod_path,
        "protocol_path": protocol_path,
        "protocol_authority": protocol_authority,
        "id_path": id_path,
        "rubric_path": rubric_path,
        "decl_path": decl_path,
        "out_root": out_root,
        "authority": authority,
    }


@pytest.fixture(autouse=True)
def _synthetic_materialized_authorities(monkeypatch):
    """The authority loader itself is covered in intake tests.

    These P4 tests use deliberately tiny non-Parquet payloads, so substitute only
    the expensive typed-loader result while retaining exact JSONL path/ref/hash
    checks here.  Individual negatives override this fixture to prove a rejected
    loader result blocks preflight.
    """

    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *args, **kwargs: SimpleNamespace(provenance={}),
    )
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_trajectory."
        "load_verified_materialized_trajectory_authority",
        lambda *args, **kwargs: object(),
    )


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
    assert auth.scientific_protocol_authority_digest is not None


def test_missing_scientific_protocol_blocks_before_any_cohort_read(
    tmp_path,
    monkeypatch,
) -> None:
    import benchmarks.figure2_canonical9.realrun_authority as authority_module

    request, _ = _authorized_setup(tmp_path)
    request = dataclasses.replace(
        request,
        scientific_protocol_authority_path=None,
    )

    def forbidden_cohort_read(*_args, **_kwargs):
        raise AssertionError("cohort bytes must not be read before protocol approval")

    monkeypatch.setattr(
        authority_module,
        "production_cohort_input_sha256",
        forbidden_cohort_read,
    )
    auth = verify_realrun_authorization(request)

    assert auth.status == "blocked"
    assert _codes(auth) == {"SCIENTIFIC_PROTOCOL_AUTHORITY_ABSENT"}


def test_scientific_protocol_tamper_blocks_before_production_input(
    tmp_path,
    monkeypatch,
) -> None:
    import benchmarks.figure2_canonical9.realrun_authority as authority_module

    request, paths = _authorized_setup(tmp_path)
    card_path = Path(paths["protocol_authority"].tasks[0].card_path)
    card_path.write_text(
        card_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        authority_module,
        "production_cohort_input_sha256",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production input verification must not start")
        ),
    )

    auth = verify_realrun_authorization(request)

    assert auth.status == "blocked"
    assert _codes(auth) == {"SCIENTIFIC_PROTOCOL_AUTHORITY_INVALID"}


def test_reviewed_runtime_guardrail_drift_fails_even_when_jsonl_is_repinned(
    tmp_path,
) -> None:
    """Signing the protocol also signs the exact Agent-visible projection."""

    request, paths = _authorized_setup(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["jsonl_path"].read_text(encoding="utf-8").splitlines()
    ]
    e2 = next(row for row in rows if row["key"] == "e2_lactate_mortality")
    e2["semantic_guardrails"][0] = "changed after human review"
    paths["jsonl_path"].write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    declaration = OperatorFreezeDeclaration.model_validate_json(
        paths["decl_path"].read_bytes(), strict=True
    ).model_copy(
        update={"ehrflowbench_jsonl_sha256": _sha256_file(paths["jsonl_path"])}
    )
    paths["decl_path"].write_text(declaration.model_dump_json(), encoding="utf-8")

    authorization = verify_realrun_authorization(request)

    assert authorization.status == "blocked"
    assert _codes(authorization) == {"PRODUCTION_INPUT_AUTHORITY_INVALID"}
    assert "Agent-visible scientific projection drifted" in (
        authorization.issues[0].detail
    )


def test_structural_retrofit_source_never_gains_paper_authority(
    tmp_path, monkeypatch
) -> None:
    request, _ = _authorized_setup(tmp_path)
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *args, **kwargs: SimpleNamespace(
            provenance={
                "export_authority": {"seal_kind": "retrofitted_structural_typed_export"}
            }
        ),
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


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
    paths["out_root"].mkdir(parents=True)
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OUTPUT_ROOT_NOT_FRESH" in _codes(auth)


def test_batch_root_must_equal_declared_batch_id(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["output_root"] = str(tmp_path / "not_the_batch")
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "OPERATOR_DECLARATION_INVALID" in _codes(auth)


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
    protocol_path = tmp_path / "scientific_protocol_authority.json"
    protocol_authority = _scientific_protocol_authority(protocol_path)
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
        scientific_protocol_authority_digest=protocol_authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(_REAL_RUBRIC),
        output_root=str(tmp_path / _BATCH_ID),
    )
    return {
        "jsonl_path": jsonl_path,
        "id_path": id_path,
        "decl_path": decl_path,
        "protocol_path": protocol_path,
        "out_root": tmp_path / _BATCH_ID,
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


def _development_binding_receipt(path: Path, jsonl_path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": ("easyicu.canonical9_development_binding_receipt/1"),
                "paper_authority": False,
                "output_jsonl": str(jsonl_path.resolve()),
                "output_sha256": _sha256_file(jsonl_path),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_canonical_development_diagnostic_requires_exact_nonpaper_binding(
    tmp_path,
) -> None:
    import argparse

    import tools.run_research_agent_bench as bench

    files = _launcher_files(tmp_path)
    receipt = _development_binding_receipt(
        tmp_path / "development-receipt.json",
        files["jsonl_path"],
    )
    args = argparse.Namespace(
        figure2_realrun_authorization=None,
        figure2_expected_execution_identity=None,
        figure2_production_input_authority=None,
        figure2_development_binding_receipt=str(receipt),
        development_diagnostic=True,
        ehrflowbench_jsonl=str(files["jsonl_path"]),
        require_figure2_paper_acceptance=False,
        submission_profile=False,
        runner="docker",
        arms=["aware"],
        provider="openai",
    )

    rc, binding = bench._figure2_realrun_authorization_gate(args)

    assert rc is None
    assert binding is None


def test_canonical_development_diagnostic_accepts_current_dev_profile(
    tmp_path,
) -> None:
    import argparse

    import tools.run_research_agent_bench as bench
    from easyicu.research_agent.orchestration.profiles import (
        E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19,
    )

    files = _launcher_files(tmp_path)
    receipt = _development_binding_receipt(
        tmp_path / "development-receipt.json",
        files["jsonl_path"],
    )
    args = argparse.Namespace(
        figure2_realrun_authorization=None,
        figure2_expected_execution_identity=None,
        figure2_production_input_authority=None,
        figure2_development_binding_receipt=str(receipt),
        development_diagnostic=True,
        ehrflowbench_jsonl=str(files["jsonl_path"]),
        require_figure2_paper_acceptance=False,
        submission_profile=True,
        runner="docker",
        arms=["aware"],
        provider="openai",
    )

    rc, binding = bench._figure2_realrun_authorization_gate(
        args,
        submission_profile=E1_PROGRESSIVE_PLANNER_CANARY_2026_08_19,
    )

    assert rc is None
    assert binding is None


def test_canonical_development_diagnostic_rejects_tampered_binding(
    tmp_path,
) -> None:
    import argparse

    import tools.run_research_agent_bench as bench

    files = _launcher_files(tmp_path)
    receipt = _development_binding_receipt(
        tmp_path / "development-receipt.json",
        files["jsonl_path"],
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["output_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    args = argparse.Namespace(
        figure2_realrun_authorization=None,
        figure2_expected_execution_identity=None,
        figure2_production_input_authority=None,
        figure2_development_binding_receipt=str(receipt),
        development_diagnostic=True,
        ehrflowbench_jsonl=str(files["jsonl_path"]),
        require_figure2_paper_acceptance=False,
        submission_profile=False,
        runner="docker",
        arms=["aware"],
        provider="openai",
    )

    rc, binding = bench._figure2_realrun_authorization_gate(args)

    assert rc == 2
    assert binding is None


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
            "--figure2-scientific-protocol-authority",
            str(files["protocol_path"]),
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
            "--figure2-scientific-protocol-authority",
            str(files["protocol_path"]),
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
            "--figure2-scientific-protocol-authority",
            str(files["protocol_path"]),
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
            "--figure2-scientific-protocol-authority",
            str(files["protocol_path"]),
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
    rc, binding = bench._figure2_realrun_authorization_gate(args)
    assert rc is None
    assert binding is None


# ---------------------------------------------------------------------------
# R2-1 — JSONL classification bypass is closed (strict, shared resolution)
# ---------------------------------------------------------------------------


def test_resolve_strict_jsonl_rejects_relative_and_symlink(tmp_path) -> None:
    real = tmp_path / "canonical.jsonl"
    real.write_text("{}\n", encoding="utf-8")
    assert resolve_strict_jsonl_path(real) == real.resolve()
    with pytest.raises(Exception):
        resolve_strict_jsonl_path("canonical.jsonl")  # relative
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    with pytest.raises(Exception):
        resolve_strict_jsonl_path(link)  # symlink
    with pytest.raises(Exception):
        resolve_strict_jsonl_path(tmp_path)  # directory


def _launcher_jsonl_bypass_argv(tmp_path, jsonl_arg: str) -> list[str]:
    files = _launcher_files(tmp_path)
    return [
        "run_research_agent_bench.py",
        "--figure2-realrun-authorization",
        str(files["decl_path"]),
        "--figure2-expected-execution-identity",
        str(files["id_path"]),
        "--ehrflowbench-jsonl",
        jsonl_arg,
        "--out-root",
        str(files["out_root"]),
        "--arms",
        "aware",
        "--provider",
        _PROVIDER,
    ]


def test_launcher_relative_jsonl_blocks(tmp_path, monkeypatch) -> None:
    bench, calls = _spies(monkeypatch)
    # A RELATIVE jsonl cannot be downgraded to non-canonical; it fails closed.
    monkeypatch.setattr(
        sys, "argv", _launcher_jsonl_bypass_argv(tmp_path, "relative_canonical.jsonl")
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_symlink_jsonl_blocks(tmp_path, monkeypatch) -> None:
    bench, calls = _spies(monkeypatch)
    real = tmp_path / "real_canonical.jsonl"
    real.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link_canonical.jsonl"
    link.symlink_to(real)
    monkeypatch.setattr(sys, "argv", _launcher_jsonl_bypass_argv(tmp_path, str(link)))
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_unreadable_jsonl_blocks(tmp_path, monkeypatch) -> None:
    bench, calls = _spies(monkeypatch)
    # A directory (not a regular file) cannot be strictly read -> fail closed.
    monkeypatch.setattr(
        sys, "argv", _launcher_jsonl_bypass_argv(tmp_path, str(tmp_path))
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_relative_canonical_jsonl_blocks(tmp_path, monkeypatch) -> None:
    """A real Canonical9 JSONL cannot exploit the legacy relative-path route."""

    bench, calls = _spies(monkeypatch)
    files = _launcher_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--ehrflowbench-jsonl",
            files["jsonl_path"].name,
            "--out-root",
            str(files["out_root"]),
            "--arms",
            "aware",
        ],
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}


def test_launcher_noncanonical_relative_jsonl_keeps_legacy_behavior(
    tmp_path, monkeypatch
) -> None:
    """Ordinary external fixtures remain runnable through their legacy path."""

    import tools.run_research_agent_bench as bench

    fixture = tmp_path / "external_fixture.jsonl"
    fixture.write_text(json.dumps({"key": "external_fixture"}) + "\n")
    captured: dict = {}
    monkeypatch.setattr(
        bench, "_run_ehrflowbench_jsonl", lambda **kw: captured.update(kw) or 0
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--ehrflowbench-jsonl",
            fixture.name,
            "--out-root",
            str(tmp_path / "ordinary_out"),
            "--arms",
            "naive",
        ],
    )
    assert bench.main() == 0
    assert captured["jsonl_path"] == fixture.resolve()
    assert captured["batch_binding"] is None


# ---------------------------------------------------------------------------
# R2-2 — provenance_sha256 is a real authority (typed sidecar bound)
# ---------------------------------------------------------------------------


def test_provenance_sidecar_replaced_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Swap the materialization-authority SIDECAR bytes AFTER it was pinned.
    victim = paths["cohort_paths"][FIGURE2_TASK_IDS[0]]
    sidecar = victim.parent / f"{victim.stem}.authority.json"
    sidecar.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_provenance_digest_mismatch_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Forge a production authority whose provenance digest does not match the ref.
    cohort_paths = paths["cohort_paths"]
    tasks = [
        ProductionInputTask(
            task_id=task_id,
            input_sha256=production_cohort_input_sha256(cohort_paths[task_id]),
            provenance_sha256="0" * 64,  # not the real sidecar provenance
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    forged = ProductionInputAuthority.build(
        submission_profile_ref=_PROFILE_REF, tasks=tasks
    )
    paths["prod_path"].write_text(forged.model_dump_json(), encoding="utf-8")
    # Re-pin the declaration to the forged digest so ONLY provenance is wrong.
    _declaration(
        paths["decl_path"],
        jsonl_path=paths["jsonl_path"],
        jsonl_sha=paths["jsonl_sha"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=forged.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(paths["out_root"]),
    )
    _frozen_identity(paths["id_path"], input_authority=forged.authority_digest)
    _declaration(
        paths["decl_path"],
        jsonl_path=paths["jsonl_path"],
        jsonl_sha=paths["jsonl_sha"],
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=forged.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(paths["out_root"]),
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_missing_cohort_authority_ref_blocks(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    # Rewrite the JSONL to DROP the typed authority refs, then re-pin its sha.
    cohort_paths = paths["cohort_paths"]
    lines = [
        json.dumps(
            {
                "key": task_id,
                "cohort_path": str(cohort_paths[task_id]),
                "question": f"q {task_id}",
                "target_outcome": "mortality",
            }
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    paths["jsonl_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    _declaration(
        paths["decl_path"],
        jsonl_path=paths["jsonl_path"],
        jsonl_sha=_sha256_file(paths["jsonl_path"]),
        expected_execution_identity_sha256=_sha256_file(paths["id_path"]),
        input_authority_digest=paths["authority"].authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(paths["rubric_path"]),
        output_root=str(paths["out_root"]),
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_preflight_rejects_sidecar_path_that_disagrees_with_ref(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    rows = [json.loads(line) for line in paths["jsonl_path"].read_text().splitlines()]
    alternate = tmp_path / "cohorts" / "alternate.authority.json"
    alternate.write_text("{}", encoding="utf-8")
    rows[0]["cohort_authority_path"] = str(alternate)
    paths["jsonl_path"].write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["ehrflowbench_jsonl_sha256"] = _sha256_file(paths["jsonl_path"])
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_preflight_rejects_loader_that_cannot_bind_sidecar_to_cohort(
    tmp_path, monkeypatch
) -> None:
    request, _ = _authorized_setup(tmp_path)
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *args, **kwargs: None,
    )
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


def test_preflight_rejects_unpaired_trajectory_authority(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    rows = [json.loads(line) for line in paths["jsonl_path"].read_text().splitlines()]
    rows[0]["trajectory_path"] = str(paths["cohort_paths"][FIGURE2_TASK_IDS[0]])
    paths["jsonl_path"].write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["ehrflowbench_jsonl_sha256"] = _sha256_file(paths["jsonl_path"])
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(request)
    assert auth.status == "blocked"
    assert "PRODUCTION_INPUT_AUTHORITY_INVALID" in _codes(auth)


# ---------------------------------------------------------------------------
# R2-3 — one frozen execution-config digest folds every run-semantics knob
# ---------------------------------------------------------------------------


def _config_with(**overrides):
    return build_canonical_execution_config(
        seed=7,
        timeout_seconds=900.0,
        standard_executor_timeout_seconds=3600.0,
        **overrides,
    )


def test_stop_after_step_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(
            request, execution_config=_config_with(stop_after_step_id="04_primary")
        )
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_INVALID" in _codes(auth)


def test_config_seed_mismatch_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    changed = build_canonical_execution_config(
        seed=999, timeout_seconds=900.0, standard_executor_timeout_seconds=3600.0
    )
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=changed)
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_MISMATCH" in _codes(auth)


def test_config_pubmed_mismatch_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=_config_with(enable_pubmed=True))
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_MISMATCH" in _codes(auth)


def test_config_request_timeout_mismatch_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    changed = build_canonical_execution_config(
        seed=7,
        timeout_seconds=900.0,
        standard_executor_timeout_seconds=3600.0,
        request_timeout_seconds=12.0,
    )
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=changed)
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_MISMATCH" in _codes(auth)


def test_config_reasoning_effort_profile_mismatch_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    changed = _config_with(reasoning_effort_profile="adaptive_v1")
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=changed)
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_MISMATCH" in _codes(auth)


def test_config_planner_strategy_mismatch_blocks(tmp_path) -> None:
    request, _ = _authorized_setup(tmp_path)
    changed = _config_with(planner_strategy="progressive_v2")
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=changed)
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_MISMATCH" in _codes(auth)


def test_config_mutable_case_selector_blocks_even_when_pinned(tmp_path) -> None:
    request, paths = _authorized_setup(tmp_path)
    changed = build_canonical_execution_config(
        seed=7,
        timeout_seconds=900.0,
        standard_executor_timeout_seconds=3600.0,
        case="mutable_fixture_name",
    )
    body = json.loads(paths["decl_path"].read_text(encoding="utf-8"))
    body["execution_config_sha256"] = changed.digest()
    paths["decl_path"].write_text(json.dumps(body), encoding="utf-8")
    auth = verify_realrun_authorization(
        _with_invocation(request, execution_config=changed)
    )
    assert auth.status == "blocked"
    assert "EXECUTION_CONFIG_INVALID" in _codes(auth)


def _launcher_config_argv(tmp_path, extra: list[str]) -> list[str]:
    files = _launcher_files(tmp_path)
    return [
        "run_research_agent_bench.py",
        "--figure2-realrun-authorization",
        str(files["decl_path"]),
        "--figure2-expected-execution-identity",
        str(files["id_path"]),
        "--figure2-production-input-authority",
        str(tmp_path / "prod.json"),
        "--figure2-scientific-protocol-authority",
        str(files["protocol_path"]),
        "--ehrflowbench-jsonl",
        str(files["jsonl_path"]),
        "--out-root",
        str(files["out_root"]),
        "--arms",
        "aware",
        "--provider",
        _PROVIDER,
        *extra,
    ]


def test_launcher_stop_after_step_blocks(tmp_path, monkeypatch, capsys) -> None:
    bench, calls = _spies(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", _launcher_config_argv(tmp_path, ["--stop-after-step-id", "@first"])
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "EXECUTION_CONFIG_INVALID" in capsys.readouterr().out


def test_launcher_seed_mismatch_blocks(tmp_path, monkeypatch, capsys) -> None:
    bench, calls = _spies(monkeypatch)
    monkeypatch.setattr(sys, "argv", _launcher_config_argv(tmp_path, ["--seed", "999"]))
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "EXECUTION_CONFIG_MISMATCH" in capsys.readouterr().out


def test_launcher_enable_pubmed_blocks(tmp_path, monkeypatch, capsys) -> None:
    bench, calls = _spies(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", _launcher_config_argv(tmp_path, ["--enable-pubmed"])
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "EXECUTION_CONFIG_MISMATCH" in capsys.readouterr().out


def test_launcher_request_timeout_mismatch_blocks(
    tmp_path, monkeypatch, capsys
) -> None:
    bench, calls = _spies(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", _launcher_config_argv(tmp_path, ["--request-timeout", "12"])
    )
    assert bench.main() == 2
    assert calls == {"llm": 0, "suite": 0, "ehrflow": 0, "register": 0}
    assert "EXECUTION_CONFIG_MISMATCH" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# R2-4 — batch identity: declaration pin, ledger, and the allowed end-to-end path
# ---------------------------------------------------------------------------


def test_batch_id_must_start_with_batch(tmp_path) -> None:
    with pytest.raises(Exception):
        _declaration(
            tmp_path / "bad.json",
            jsonl_path=tmp_path / "x.jsonl",
            jsonl_sha="0" * 64,
            batch_id="run_not_a_batch",  # must start with 'batch_'
            ehrflowbench_jsonl_path="/tmp/x.jsonl",
        )


def _binding(frozen, out_root: Path | None = None) -> RealRunBatchBinding:
    binding = RealRunBatchBinding(
        batch_id=_BATCH_ID,
        declaration_sha256="d" * 64,
        input_authority_digest="e" * 64,
        scientific_protocol_authority_digest="f" * 64,
        frozen_input_by_task=frozen,
    )
    if out_root is not None:
        return reserve_authorized_batch_root(
            out_root, binding, generated_at="2026-07-22T00:00:00+00:00"
        )
    return binding


def _manifest_identity(input_digest: str) -> dict:
    return ExecutionIdentity.create(
        submission_profile_name="npj_dm",
        submission_profile_version="20260718",
        runner="docker",
        runner_image_digest=_IMAGE,
        network_policy="none",
        llm_seed=0,
        input_authority_sha256=input_digest,
        provider_authorization={"clients": [{"authorization_mode": "operator_env"}]},
        code_version={"git_sha": _COMMIT, "git_dirty": False},
    ).model_dump(mode="json")


def _synthetic_scores(out_root: Path, frozen: dict[str, str]) -> list[dict]:
    scores = []
    for task_id in FIGURE2_TASK_IDS:
        run_id = f"run_{task_id}"
        workdir = out_root / task_id / "aware" / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        identity = _manifest_identity(frozen[task_id])
        (workdir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "execution_identity": identity}),
            encoding="utf-8",
        )
        scores.append(
            {
                "item_key": task_id,
                "aware": {
                    "arm": "aware",
                    "run_id": run_id,
                    "workdir": str(workdir),
                    "execution_identity": identity,
                },
            }
        )
    return scores


def test_build_batch_ledger_records_nine(tmp_path) -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)
    scores = _synthetic_scores(out_root, frozen)
    ledger = build_batch_ledger({"scores": scores}, out_root, binding)
    assert ledger["complete"] is True
    assert ledger["batch_id"] == _BATCH_ID
    assert [c["task_id"] for c in ledger["children"]] == list(FIGURE2_TASK_IDS)
    assert all(c["status"] == "recorded" for c in ledger["children"])
    assert all(c["manifest_sha256"] for c in ledger["children"])


def test_build_batch_ledger_incomplete_on_missing_child(tmp_path) -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)
    scores = _synthetic_scores(out_root, frozen)[:-1]  # drop the last task
    ledger = build_batch_ledger({"scores": scores}, out_root, binding)
    assert ledger["complete"] is False


def test_batch_root_reservation_is_atomic_and_receipt_is_bound(tmp_path) -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    out_root = tmp_path / _BATCH_ID
    initial = _binding(frozen)
    binding = reserve_authorized_batch_root(
        out_root, initial, generated_at="2026-07-22T00:00:00+00:00"
    )
    assert binding.batch_root == out_root
    assert binding.receipt_sha256
    assert verify_batch_authorization_receipt(binding).is_file()
    with pytest.raises(FileExistsError):
        reserve_authorized_batch_root(
            out_root, initial, generated_at="2026-07-22T00:00:01+00:00"
        )


def test_batch_ledger_uses_manifest_not_score_self_report(tmp_path) -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)
    scores = _synthetic_scores(out_root, frozen)
    scores[0]["aware"]["execution_identity"] = {
        "identity_sha256": "0" * 64,
        "input_authority_sha256": "1" * 64,
    }
    ledger = build_batch_ledger({"scores": scores}, out_root, binding)
    assert ledger["complete"] is False
    child = ledger["children"][0]
    assert child["status"].startswith("manifest_unreadable")
    assert (
        child["identity_sha256"]
        == _manifest_identity(frozen[FIGURE2_TASK_IDS[0]])["identity_sha256"]
    )


def test_batch_ledger_rejects_tampered_receipt(tmp_path) -> None:
    frozen = {t: hashlib.sha256(t.encode()).hexdigest() for t in FIGURE2_TASK_IDS}
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)
    receipt = out_root / "figure2_realrun_authorization_receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        build_batch_ledger({"scores": []}, out_root, binding)


def test_run_ehrflowbench_writes_receipt_and_ledger(tmp_path, monkeypatch) -> None:
    """Allowed path with a STUBBED runner/pipeline: receipt + 9-child ledger."""
    import tools.run_research_agent_bench as bench

    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    _write_jsonl(jsonl_path, cohort_paths)
    frozen = {
        t: production_cohort_input_sha256(cohort_paths[t]) for t in FIGURE2_TASK_IDS
    }
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)

    # Stub bounded cohort metadata and pipeline execution (no real Provider).
    monkeypatch.setattr(
        bench,
        "_cohort_shape_without_materialization",
        lambda path: (1, ["stay_id", "x"]),
    )
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *a, **k: object(),
    )

    def _fake_item(*, item, cohort, out_root, **kwargs):
        run_id = f"run_{item.key}"
        workdir = Path(out_root) / item.key / "aware" / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        identity = _manifest_identity(
            kwargs["pipeline_options"].get("execution_input_authority_sha256")
        )
        (workdir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "execution_identity": identity}),
            encoding="utf-8",
        )
        aware = dict(bench._skipped_arm("aware"))
        aware.update(
            {
                "status": "ok",
                "run_id": run_id,
                "workdir": str(workdir),
                "execution_identity": identity,
                "publication_ready": True,
                "manuscript_ready": True,
                "publication_artifacts_ready": True,
                "execution_paper_eligible": True,
                "paper_authorized": True,
                "execution_complete": True,
                "step_scientific_requirements_complete": True,
                "required_step_count": 1,
                "completed_step_count": 1,
                "failed_step_ids": [],
                "missing_step_ids": [],
                "n_errors": 0,
                "figure2_evaluation_attempt": {
                    "status": "valid",
                    "envelope": {
                        "scorecard": {
                            "scorecard_canonical_json": json.dumps(
                                {"tristate": "gate_reportable"}
                            )
                        }
                    },
                },
            }
        )
        return {"item_key": item.key, "aware": aware}

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", _fake_item)

    bench._run_ehrflowbench_jsonl(
        jsonl_path=jsonl_path,
        out_root=out_root,
        seed=7,
        arms=["aware"],
        provider=_PROVIDER,
        model=_MODEL,
        batch_binding=binding,
        pipeline_options=bench._benchmark_pipeline_options(
            max_total_steps=None,
            disable_replanning=False,
            max_code_repair_attempts=None,
        ),
    )

    receipt = json.loads(
        (out_root / "figure2_realrun_authorization_receipt.json").read_text()
    )
    assert receipt["batch_id"] == _BATCH_ID
    assert receipt["declaration_sha256"] == "d" * 64
    ledger = json.loads((out_root / "figure2_batch_ledger.json").read_text())
    assert ledger["complete"] is True
    assert ledger["batch_id"] == _BATCH_ID
    assert len(ledger["children"]) == 9
    progress = load_provider_hard_stop_ledger(out_root / "figure2_batch_progress.json")
    assert progress["terminal"] is True
    assert [task["status"] for task in progress["tasks"]] == ["completed"] * 9
    canary = json.loads((out_root / "figure2_canary_gate.json").read_text())
    assert canary["status"] == "passed"
    assert canary["task_id"] == FIGURE2_TASK_IDS[0]
    # Each child's frozen input authority (from the per-row binding) is recorded.
    for child in ledger["children"]:
        assert child["input_authority_sha256"] == frozen[child["task_id"]]
        assert child["manifest_sha256"]


def test_incomplete_batch_ledger_downgrades_acceptance_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No accepted terminal artifact may survive a failed batch binding."""

    import tools.run_research_agent_bench as bench
    from benchmarks.figure2_canonical9 import realrun_authority
    from benchmarks.figure2_canonical9.evaluator import acceptance

    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    _write_jsonl(jsonl_path, cohort_paths)
    frozen = {
        task_id: production_cohort_input_sha256(cohort_paths[task_id])
        for task_id in FIGURE2_TASK_IDS
    }
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)

    monkeypatch.setattr(
        bench,
        "_cohort_shape_without_materialization",
        lambda path: (1, ["stay_id", "x"]),
    )
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *a, **k: object(),
    )

    def fake_item(*, item, out_root, **kwargs):
        run_id = f"run_{item.key}"
        workdir = Path(out_root) / item.key / "aware" / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        identity = _manifest_identity(
            kwargs["pipeline_options"]["execution_input_authority_sha256"]
        )
        (workdir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "execution_identity": identity}),
            encoding="utf-8",
        )
        return {
            "item_key": item.key,
            "aware": {
                "arm": "aware",
                "status": "ok",
                "run_id": run_id,
                "workdir": str(workdir),
                "execution_identity": identity,
                "publication_ready": True,
                "manuscript_ready": True,
                "publication_artifacts_ready": True,
                "execution_paper_eligible": True,
                "paper_authorized": True,
                "execution_complete": True,
                "step_scientific_requirements_complete": True,
                "required_step_count": 1,
                "completed_step_count": 1,
                "failed_step_ids": [],
                "missing_step_ids": [],
                "n_errors": 0,
                "figure2_evaluation_attempt": {
                    "status": "valid",
                    "envelope": {
                        "scorecard": {
                            "scorecard_canonical_json": json.dumps(
                                {"tristate": "gate_reportable"}
                            )
                        }
                    },
                },
            },
        }

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", fake_item)
    monkeypatch.setattr(bench, "_aggregate", lambda _scores: {"aware": {}})
    monkeypatch.setattr(bench, "_render_markdown", lambda **_kwargs: "fixture\n")
    safety_calls: list[str] = []
    monkeypatch.setattr(
        bench,
        "_ensure_formal_figure2_safety_and_rescore",
        lambda **kwargs: safety_calls.append(str(kwargs["item"].key)),
    )
    accepted = acceptance.Figure2PaperAcceptance(
        schema_version=acceptance.FIGURE2_PAPER_ACCEPTANCE_SCHEMA,
        status="accepted",
        results_sha256="0" * 64,
        expected_execution_identity_sha256="1" * 64,
        expected_execution_identity_freeze_sha256="2" * 64,
        expected_task_ids=tuple(FIGURE2_TASK_IDS),
        observed_task_ids=tuple(FIGURE2_TASK_IDS),
        verified_tasks=tuple(
            acceptance.VerifiedFigure2Task(
                task_id=task_id,
                run_id=f"run_{task_id}",
                attempt_sha256=hashlib.sha256(task_id.encode()).hexdigest(),
                tristate="gate_reportable",
            )
            for task_id in FIGURE2_TASK_IDS
        ),
    )
    monkeypatch.setattr(
        acceptance,
        "evaluate_figure2_paper_acceptance",
        lambda *a, **k: accepted,
    )
    monkeypatch.setattr(
        realrun_authority,
        "build_batch_ledger",
        lambda *a, **k: {
            "schema_version": "fixture",
            "complete": False,
            "children": [],
        },
    )

    exit_code = bench._run_ehrflowbench_jsonl(
        jsonl_path=jsonl_path,
        out_root=out_root,
        seed=7,
        arms=["aware"],
        provider=_PROVIDER,
        model=_MODEL,
        batch_binding=binding,
        pipeline_options=bench._benchmark_pipeline_options(
            max_total_steps=None,
            disable_replanning=False,
            max_code_repair_attempts=None,
        ),
    )

    assert exit_code == 2
    assert safety_calls == list(FIGURE2_TASK_IDS)
    terminal = json.loads(
        (out_root / "figure2_paper_acceptance.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "invalid"
    assert "BATCH_LEDGER_INVALID" in {issue["code"] for issue in terminal["issues"]}


def test_formal_batch_does_not_start_e2_when_e1_canary_is_diagnostic(
    tmp_path, monkeypatch
) -> None:
    import tools.run_research_agent_bench as bench

    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    _write_jsonl(jsonl_path, cohort_paths)
    frozen = {
        task_id: production_cohort_input_sha256(cohort_paths[task_id])
        for task_id in FIGURE2_TASK_IDS
    }
    out_root = tmp_path / _BATCH_ID
    binding = _binding(frozen, out_root)
    monkeypatch.setattr(
        bench,
        "_cohort_shape_without_materialization",
        lambda path: (1, ["stay_id", "x"]),
    )
    monkeypatch.setattr(
        "easyicu.research_agent.intake.materialized_metadata."
        "load_verified_materialized_cohort_authority",
        lambda *a, **k: object(),
    )
    calls: list[str] = []

    def _diagnostic_item(*, item, **kwargs):
        calls.append(item.key)
        aware = dict(bench._skipped_arm("aware"))
        aware.update(
            {
                "status": "diagnostic_only",
                "publication_ready": False,
                "manuscript_ready": False,
                "n_errors": 1,
                "figure2_evaluation_attempt": {"status": "valid"},
            }
        )
        return {
            "item_key": item.key,
            "aware": aware,
        }

    monkeypatch.setattr(bench, "_run_one_item_from_cohort", _diagnostic_item)

    assert (
        bench._run_ehrflowbench_jsonl(
            jsonl_path=jsonl_path,
            out_root=out_root,
            seed=7,
            arms=["aware"],
            provider=_PROVIDER,
            model=_MODEL,
            batch_binding=binding,
            pipeline_options=bench._benchmark_pipeline_options(
                max_total_steps=None,
                disable_replanning=False,
                max_code_repair_attempts=None,
            ),
        )
        == 2
    )

    assert calls == [FIGURE2_TASK_IDS[0]]
    canary = json.loads((out_root / "figure2_canary_gate.json").read_text())
    assert canary["status"] == "blocked"
    payload = json.loads((out_root / "ehrflowbench_results.json").read_text())
    blocked = [
        row for row in payload["pending"] if row["status"] == "batch_canary_blocked"
    ]
    assert [row["key"] for row in blocked] == list(FIGURE2_TASK_IDS[1:])


def test_formal_canary_rejects_valid_but_analysis_only_paper_score() -> None:
    import tools.run_research_agent_bench as bench

    score = {
        "aware": {
            "publication_ready": True,
            "manuscript_ready": True,
            "publication_artifacts_ready": True,
            "execution_paper_eligible": True,
            "paper_authorized": True,
            "n_errors": 0,
            "figure2_evaluation_attempt": {
                "status": "valid",
                "envelope": {
                    "scorecard": {
                        "scorecard_canonical_json": json.dumps(
                            {"tristate": "analysis_only"}
                        )
                    }
                },
            },
        }
    }

    assert bench._figure2_canary_passed(score) is False


def test_formal_canary_requires_e1_scientific_receipt_for_closure_protocol() -> None:
    import tools.run_research_agent_bench as bench

    score = {
        "protocol_version": (
            "easyicu_evaluation_protocol_suite/v2+e1_scientific_closure/20260728-v1"
        ),
        "aware": {
            "publication_ready": True,
            "manuscript_ready": True,
            "publication_artifacts_ready": True,
            "execution_paper_eligible": True,
            "paper_authorized": True,
            "n_errors": 0,
            "figure2_evaluation_attempt": {
                "status": "valid",
                "envelope": {
                    "scorecard": {
                        "scorecard_canonical_json": json.dumps(
                            {"tristate": "gate_reportable"}
                        )
                    }
                },
            },
        },
    }

    assert bench._figure2_canary_passed(score) is False
    score["aware"]["scientific_acceptance"] = {
        "status": "accepted",
        "issues": [],
    }
    assert bench._figure2_canary_passed(score) is True


def test_formal_canary_rejects_artifact_ready_but_unauthorized_run() -> None:
    import tools.run_research_agent_bench as bench

    score = {
        "aware": {
            "publication_ready": True,
            "manuscript_ready": True,
            "publication_artifacts_ready": True,
            "execution_paper_eligible": False,
            "paper_authorized": False,
            "n_errors": 0,
            "figure2_evaluation_attempt": {
                "status": "valid",
                "envelope": {
                    "scorecard": {
                        "scorecard_canonical_json": json.dumps(
                            {"tristate": "gate_reportable"}
                        )
                    }
                },
            },
        }
    }

    assert bench._figure2_canary_passed(score) is False


def test_end_to_end_gate_authorizes_and_hands_batch_binding(
    tmp_path, monkeypatch
) -> None:
    """main() gate authorizes the allowed path and hands the runner a batch binding
    with the 9-task frozen map — proven WITHOUT a real Provider/runner."""
    import tools.run_research_agent_bench as bench

    profile_ref = bench._default_submission_profile_ref()
    profile_name, profile_version = profile_ref.split("/", 1)

    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    jsonl_sha = _write_jsonl(jsonl_path, cohort_paths)
    prod_path = tmp_path / "prod.json"
    tasks = [
        ProductionInputTask(
            task_id=task_id,
            input_sha256=production_cohort_input_sha256(cohort_paths[task_id]),
            provenance_sha256=production_provenance_sha256(
                _cohort_ref(cohort_paths[task_id]), None
            ),
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    authority = ProductionInputAuthority.build(
        submission_profile_ref=profile_ref, tasks=tasks
    )
    prod_path.write_text(authority.model_dump_json(), encoding="utf-8")
    protocol_path = tmp_path / "scientific_protocol_authority.json"
    protocol_authority = _scientific_protocol_authority(protocol_path)

    id_path = tmp_path / "identity.json"
    _frozen_identity(
        id_path,
        input_authority=authority.authority_digest,
        profile_name=profile_name,
        profile_version=profile_version,
    )
    out_root = tmp_path / _BATCH_ID
    decl_path = tmp_path / "declaration.json"
    _declaration(
        decl_path,
        jsonl_path=jsonl_path,
        jsonl_sha=jsonl_sha,
        submission_profile_ref=profile_ref,
        expected_execution_identity_sha256=_sha256_file(id_path),
        input_authority_digest=authority.authority_digest,
        scientific_protocol_authority_digest=protocol_authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(_REAL_RUBRIC),
        output_root=str(out_root),
    )

    # A clean live checkout matching the pinned commit (no real git tree needed).
    monkeypatch.setattr(
        "easyicu.research_agent.authority.runtime_artifacts.capture_code_version",
        lambda: {"git_sha": _COMMIT, "git_dirty": False},
    )
    # Capture the batch binding the gate hands the runner; never run a real batch.
    captured: dict = {}

    def _capture_ehrflow(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(bench, "_run_ehrflowbench_jsonl", _capture_ehrflow)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(decl_path),
            "--figure2-expected-execution-identity",
            str(id_path),
            "--figure2-production-input-authority",
            str(prod_path),
            "--figure2-scientific-protocol-authority",
            str(protocol_path),
            "--ehrflowbench-jsonl",
            str(jsonl_path),
            "--out-root",
            str(out_root),
            "--arms",
            "aware",
            "--provider",
            _PROVIDER,
            "--model",
            _MODEL,
            "--submission-profile",
            "--profile",
            profile_ref,
            "--runner",
            "docker",
            "--require-figure2-paper-acceptance",
        ],
    )

    assert bench.main() == 0
    binding = captured.get("batch_binding")
    assert binding is not None
    assert binding.batch_id == _BATCH_ID
    assert binding.declaration_sha256 == _sha256_file(decl_path)
    assert (
        binding.scientific_protocol_authority_digest
        == protocol_authority.authority_digest
    )
    assert set(binding.frozen_input_by_task) == set(FIGURE2_TASK_IDS)
    assert binding.frozen_input_by_task[FIGURE2_TASK_IDS[0]] == tasks[0].input_sha256
    assert binding.batch_root == out_root
    assert verify_batch_authorization_receipt(binding).is_file()


def test_gate_does_not_reopen_production_authority_after_verification(
    tmp_path, monkeypatch
) -> None:
    """The launcher consumes the immutable authorization result, not raw files."""

    import benchmarks.figure2_canonical9.realrun_authority as authority_module
    import tools.run_research_agent_bench as bench

    profile_ref = bench._default_submission_profile_ref()
    profile_name, profile_version = profile_ref.split("/", 1)
    cohort_paths = _cohorts(tmp_path)
    jsonl_path = tmp_path / "canonical.jsonl"
    jsonl_sha = _write_jsonl(jsonl_path, cohort_paths)
    production_path = tmp_path / "production.json"
    tasks = [
        ProductionInputTask(
            task_id=task_id,
            input_sha256=production_cohort_input_sha256(cohort_paths[task_id]),
            provenance_sha256=production_provenance_sha256(
                _cohort_ref(cohort_paths[task_id]), None
            ),
        )
        for task_id in FIGURE2_TASK_IDS
    ]
    production = ProductionInputAuthority.build(
        submission_profile_ref=profile_ref, tasks=tasks
    )
    production_path.write_text(production.model_dump_json(), encoding="utf-8")
    protocol_path = tmp_path / "scientific_protocol_authority.json"
    protocol_authority = _scientific_protocol_authority(protocol_path)
    identity_path = tmp_path / "identity.json"
    _frozen_identity(
        identity_path,
        input_authority=production.authority_digest,
        profile_name=profile_name,
        profile_version=profile_version,
    )
    out_root = tmp_path / _BATCH_ID
    declaration_path = tmp_path / "declaration.json"
    _declaration(
        declaration_path,
        jsonl_path=jsonl_path,
        jsonl_sha=jsonl_sha,
        submission_profile_ref=profile_ref,
        expected_execution_identity_sha256=_sha256_file(identity_path),
        input_authority_digest=production.authority_digest,
        scientific_protocol_authority_digest=protocol_authority.authority_digest,
        input_freeze_manifest_sha256=canonical_input_freeze_manifest_sha256(_REAL_V1),
        rubric_sha256=_sha256_file(_REAL_RUBRIC),
        output_root=str(out_root),
    )
    original_load = authority_module.load_production_input_authority
    calls = 0

    def _load_once(path):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("gate reopened production authority after verify")
        return original_load(path)

    monkeypatch.setattr(authority_module, "load_production_input_authority", _load_once)
    monkeypatch.setattr(
        "easyicu.research_agent.authority.runtime_artifacts.capture_code_version",
        lambda: {"git_sha": _COMMIT, "git_dirty": False},
    )
    captured: dict = {}
    monkeypatch.setattr(
        bench, "_run_ehrflowbench_jsonl", lambda **kw: captured.update(kw) or 0
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_agent_bench.py",
            "--figure2-realrun-authorization",
            str(declaration_path),
            "--figure2-expected-execution-identity",
            str(identity_path),
            "--figure2-production-input-authority",
            str(production_path),
            "--figure2-scientific-protocol-authority",
            str(protocol_path),
            "--ehrflowbench-jsonl",
            str(jsonl_path),
            "--out-root",
            str(out_root),
            "--arms",
            "aware",
            "--provider",
            _PROVIDER,
            "--model",
            _MODEL,
            "--submission-profile",
            "--profile",
            profile_ref,
            "--runner",
            "docker",
            "--require-figure2-paper-acceptance",
        ],
    )
    assert bench.main() == 0
    assert calls == 1
    assert (
        captured["batch_binding"].frozen_input_by_task
        == production.frozen_input_by_task()
    )
