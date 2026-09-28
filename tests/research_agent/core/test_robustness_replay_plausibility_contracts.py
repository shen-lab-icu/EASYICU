"""A robustness replay hands the primary script its own sealed contracts.

A standard executor's primary script carries a plausibility receipt: before it
fits anything, it checks the raw-input contracts in its resolved-input manifest
against the digest its step was sealed with.  A complete-case robustness
variant replays that exact script on a smaller cohort, so the replay manifest
must carry the contracts of the primary step and never the contracts of the
robustness step that happens to be running the replay.  Each step's contracts
have their own digest, so borrowing them made every such replay fail its
receipt, and the variant emitted no estimate.

The contracts come from the manifest sealed in the primary step's executed
authority capsule.  A resumed run keeps the record's manifest digest but drops
its mutable path, and a later attempt may overwrite the file at that path, so
a replay after a resume must still find the sealed manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.authority.step_capsule import (
    ExecutionSeal,
    StepAuthorityCapsule,
    execution_seal_identity_sha256,
    put_content_blob,
    seal_step_authority_capsule,
)
from easyicu.research_agent.authority.plausibility_receipt_code import (
    render_standard_plausibility_receipt_code,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _find_structured_primary_model_source,
    _replay_primary_model_for_complete_case,
)
from easyicu.research_agent.robustness.panel import RobustnessSpec

from tests.support.robustness_sources import write_structured_source_authority


def _contracts(maximum: float) -> dict:
    payload = {
        "contracts": {
            "age": {
                "analysis_plausibility_range": {"minimum": 0.0, "maximum": maximum},
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
        }
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**payload, "contracts_sha256": digest}


_SHA = {name: name * 64 for name in "abcdef"}


def _json_blob(run_dir: Path, payload: object):
    return put_content_blob(
        run_dir,
        payload=json.dumps(payload, sort_keys=True).encode("utf-8"),
        media_type="application/json",
    )


def _seal_primary_manifest(
    run_dir: Path,
    record: dict,
    contracts: dict,
    *,
    code: bytes | None = None,
    returncode: int = 0,
    executed: bool = True,
) -> Path:
    """Write the primary step's manifest and seal it as the step executor does."""

    step_id = record["step_id"]
    manifest = run_dir / "resolved_inputs" / f"{step_id}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"step_id": step_id, "raw_input_contracts": contracts}),
        encoding="utf-8",
    )
    script = run_dir / "steps" / step_id / "analysis.py"
    candidate_fields = {
        "step_id": step_id,
        "run_input_capsule_sha256": _SHA["a"],
        "planner_scope": _json_blob(run_dir, {"step_id": step_id}),
        "scoped_coder_context": _json_blob(run_dir, {"variables": ["exposure"]}),
        "resolved_inputs": put_content_blob(
            run_dir, payload=manifest.read_bytes(), media_type="application/json"
        ),
        "candidate_code": put_content_blob(
            run_dir,
            payload=script.read_bytes() if code is None else code,
            media_type="text/x-python",
        ),
        "typed_bindings_sha256": _SHA["b"],
        "upstream_authority_sha256": _SHA["c"],
        "candidate_origin": {
            "kind": "initial_generation",
            "authority_binding_sha256": _SHA["b"],
            "provider_category": "initial_generation",
            "provider_transport_id": "initial_generation:1",
            "logical_repair_attempt_id": None,
            "repair_ticket_sha256": None,
            "deterministic_reason_sha256": None,
        },
        "deterministic_gate_fingerprint": _SHA["d"],
        "engine_code_sha256": _SHA["e"],
        "validator_code_sha256": _SHA["f"],
        "prompt_pack_version": "2026-07-16",
        "prompt_pack_sha256": _SHA["a"],
        "concept_audit": None,
    }
    candidate = StepAuthorityCapsule.model_validate(
        {**candidate_fields, "stage": "candidate", "parent_capsule_sha256": None, "execution": None}
    )
    ref = seal_step_authority_capsule(run_dir, candidate)
    if executed:
        execution = {
            "execution_context_sha256": _SHA["d"],
            "code_sha256": candidate.candidate_code.sha256,
            "resolved_inputs_sha256": candidate.resolved_inputs.sha256,
            "returncode": returncode,
            "duration_seconds": 0.25,
            "timed_out": False,
            "outputs_safe_to_collect": returncode == 0,
            "requested_network_policy": "none",
            "effective_isolation": "macos_sandbox_exec",
            "isolation_degraded": False,
            "isolation_degradation_reason": None,
            "runtime_provenance": _json_blob(run_dir, {"python": "3.13"}),
            "stdout": put_content_blob(run_dir, payload=b"", media_type="text/plain"),
            "stderr": put_content_blob(run_dir, payload=b"", media_type="text/plain"),
            "runner_log": None,
            "outputs": (),
        }
        execution["execution_identity_sha256"] = execution_seal_identity_sha256(execution)
        ref = seal_step_authority_capsule(
            run_dir,
            StepAuthorityCapsule.model_validate(
                {
                    **candidate_fields,
                    "stage": "executed",
                    "parent_capsule_sha256": ref.capsule_sha256,
                    "execution": ExecutionSeal.model_validate(execution),
                }
            ),
        )
    record["resolved_inputs_path"] = str(manifest.relative_to(run_dir))
    record["resolved_inputs_sha256"] = candidate.resolved_inputs.sha256
    record["step_authority_capsule_ref"] = ref.model_dump(mode="json")
    return manifest


def test_the_source_carries_the_primary_steps_registered_contracts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    primary_contracts = _contracts(120.0)
    _seal_primary_manifest(run_dir, record, primary_contracts)

    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )

    assert source is not None
    assert source["raw_input_contracts"] == primary_contracts


def test_a_resumed_record_still_carries_the_sealed_contracts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    primary_contracts = _contracts(120.0)
    manifest = _seal_primary_manifest(run_dir, record, primary_contracts)
    # Resume revalidation keeps the digest and drops the mutable path; a later
    # attempt of the step may also have rewritten the file at that path.
    del record["resolved_inputs_path"]
    manifest.write_text(
        json.dumps({"raw_input_contracts": _contracts(999.0)}), encoding="utf-8"
    )

    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )

    assert source is not None
    assert source["raw_input_contracts"] == primary_contracts


def _tamper_resolved_inputs_blob(run_dir: Path, record: dict) -> None:
    digest = record["resolved_inputs_sha256"]
    blob = run_dir / ".step_authority" / "blobs" / "sha256" / digest[:2] / digest
    blob.chmod(0o600)
    blob.write_bytes(b'{"raw_input_contracts": {}}')


@pytest.mark.parametrize(
    "case",
    [
        "record_digest_differs",
        "no_capsule",
        "unexecuted_capsule",
        "other_code",
        "failed_execution",
        "tampered_blob",
    ],
)
def test_contracts_are_withheld_unless_the_executed_capsule_verifies(
    tmp_path: Path, case: str
) -> None:
    run_dir = tmp_path / "run"
    record, evidence, _script = write_structured_source_authority(run_dir)
    seal = {
        "unexecuted_capsule": {"executed": False},
        "other_code": {"code": b"print('another script')\n"},
        "failed_execution": {"returncode": 1},
    }.get(case, {})
    _seal_primary_manifest(run_dir, record, _contracts(120.0), **seal)
    if case == "record_digest_differs":
        record["resolved_inputs_sha256"] = "0" * 64
    elif case == "no_capsule":
        del record["step_authority_capsule_ref"]
    elif case == "tampered_blob":
        _tamper_resolved_inputs_blob(run_dir, record)

    source = _find_structured_primary_model_source(
        records=[record], run_dir=run_dir, evidence_records=evidence
    )

    # The source is still found; only its contracts are withheld, so a replay
    # that needs them fails closed inside the script.
    assert source is not None
    assert source["raw_input_contracts"] is None


def _receipt_bound_primary(tmp_path: Path, contracts: dict) -> Path:
    scope = FlagOnlyPlausibilityScope(
        step_id="primary",
        expected_columns=("age",),
        source_contracts_sha256=contracts["contracts_sha256"],
        authority_kind="resolved_raw_input_contracts",
    )
    script_path = tmp_path / "primary" / "analysis.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(
        "\n\n".join(
            [
                "from easyicu.research_agent.execution.runners.typed_input_binding "
                "import load_step_cohort_frame\n"
                "frame, _cohort_path = load_step_cohort_frame(\n"
                "    typed_cohort_input='artifact:analysis_cohort'\n"
                ")",
                render_standard_plausibility_receipt_code(scope, frame_name="frame"),
                """
out = Path(os.environ["STEP_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    "model_id": "primary",
    "term": "exposure",
    "term_role": "exposure",
    "source_variable": "exposure",
    "odds_ratio": 1.5,
    "ci_low": 1.1,
    "ci_high": 2.0,
    "std_error": 0.1,
}]).to_csv(out / "coefficients.csv", index=False)
summary = {
    "primary_model_id": "primary",
    "coefficient_table": "coefficients.csv",
    "plausibility_audit": plausibility_audit,
    "model_contracts": [{
        "model_id": "primary",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "exposure_role": "primary",
        "exposure_source": "exposure",
        "exposure_expression": "exposure",
        "n": len(frame),
        "event_n": int(frame["outcome"].sum()),
        "fit_status": "fitted",
        "converged": True,
        "fit_method": "registered_test_model",
    }],
}
(out / "step_summary.json").write_text(json.dumps(summary))
""".strip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return script_path


def _source(script_path: Path, contracts: dict | None) -> dict:
    return {
        "step_id": "primary",
        "script_path": script_path,
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "summary": {"input_bindings": [{"input_key": "artifact:analysis_cohort"}]},
        "primary_contract": {
            "exposure_source": "exposure",
            "exposure_expression": "exposure",
        },
        "raw_input_contracts": contracts,
    }


_SPEC = RobustnessSpec(
    spec_id="complete_case",
    axis="missing",
    description="Replay the locked complete-case membership.",
    missing_override={
        "strategy": "complete_case",
        "variables": ["exposure", "outcome", "age"],
    },
)


def _primary_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
            "age": [50.0, None, 70.0, 80.0],
        }
    )


def _run_as_robustness_step(tmp_path: Path, monkeypatch, contracts: dict) -> None:
    # The replay runs inside the robustness step, whose own manifest carries
    # the robustness step's contracts.
    current = tmp_path / "robustness_inputs.json"
    current.write_text(
        json.dumps({"step_id": "robustness_replay", "raw_input_contracts": contracts}),
        encoding="utf-8",
    )
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(current))


def test_complete_case_replay_passes_the_primary_receipt_with_its_own_contracts(
    tmp_path: Path, monkeypatch
) -> None:
    primary_contracts = _contracts(120.0)
    _run_as_robustness_step(tmp_path, monkeypatch, _contracts(110.0))
    script_path = _receipt_bound_primary(tmp_path, primary_contracts)

    replay = _replay_primary_model_for_complete_case(
        spec=_SPEC,
        source=_source(script_path, primary_contracts),
        primary_data=_primary_data(),
        out_dir=tmp_path / "robustness",
    )

    assert replay["error"] is None
    assert replay["row"].n == 3
    assert replay["index"]["input_n"] == replay["index"]["modeled_n"] == 3


@pytest.mark.parametrize("current_matches_primary", [False, True])
def test_a_replay_never_borrows_the_running_steps_contracts(
    tmp_path: Path, monkeypatch, current_matches_primary: bool
) -> None:
    primary_contracts = _contracts(120.0)
    _run_as_robustness_step(
        tmp_path,
        monkeypatch,
        primary_contracts if current_matches_primary else _contracts(110.0),
    )
    script_path = _receipt_bound_primary(tmp_path, primary_contracts)

    replay = _replay_primary_model_for_complete_case(
        spec=_SPEC,
        source=_source(script_path, None),
        primary_data=_primary_data(),
        out_dir=tmp_path / "robustness",
    )

    # Without the primary step's verified contracts the receipt cannot pass,
    # even when the running step's contracts happen to carry the same digest.
    assert replay["error"] is not None
    assert "exit code" in replay["error"]
