"""The exact host-owned scope shared by plausibility obligation consumers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.plausibility import (
    FlagOnlyPlausibilityScope,
    PlausibilityScopeError,
    compile_flag_only_plausibility_scope,
    compile_step_plausibility_authority,
)
from easyicu.research_agent.research_context.typed import (
    resolved_raw_input_contracts,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
)


def _context() -> ResearchContext:
    return ResearchContext(
        research_question="Assess exact step inputs without widening scope.",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=3,
            n_patients=3,
        ),
        variables=[
            ConceptDescriptor(
                name="global_only",
                dtype="float64",
                valid_range=[0.0, 1.0],
            ),
            ConceptDescriptor(
                name="step_ranged",
                dtype="float64",
                valid_range=[0.0, 10.0],
            ),
            ConceptDescriptor(name="step_unranged", dtype="float64"),
        ],
    )


def _step(*inputs: str) -> AnalysisStep:
    return AnalysisStep(
        step_id="03_synthetic",
        intent="test exact authority",
        method="descriptive",
        inputs=list(inputs),
    )


def _contracts(
    *,
    ranged: tuple[str, ...] = (),
    unranged: tuple[str, ...] = (),
) -> dict[str, object]:
    contracts: dict[str, object] = {}
    for column in (*ranged, *unranged):
        contract: dict[str, object] = {"column": column}
        if column in ranged:
            contract.update(
                {
                    "analysis_plausibility_range": {
                        "minimum": 0.0,
                        "maximum": 10.0,
                    },
                    "plausibility_policy": {
                        "range_policy": "flag_only",
                        "out_of_range_action": "retain_and_flag",
                    },
                }
            )
        contracts[column] = contract
    payload: dict[str, object] = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "authority_scope": (
            "host_verified_physical_representation_and_domain_constraints"
        ),
        "scientific_ownership": "Planner retains scientific decisions",
        "contracts": contracts,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["contracts_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_exact_raw_contract_scope_ignores_unrelated_global_ranges() -> None:
    scope = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("step_ranged", "step_unranged"),
        raw_input_contracts=_contracts(
            ranged=("step_ranged",),
            unranged=("step_unranged",),
        ),
    )

    assert scope.expected_columns == ("step_ranged",)
    assert "global_only" not in scope.expected_columns
    assert scope.authority_kind == "resolved_raw_input_contracts"


def test_legacy_context_materializes_exact_step_contracts() -> None:
    contracts = resolved_raw_input_contracts(
        _context(),
        ["step_ranged", "step_unranged"],
    )

    assert set(contracts["contracts"]) == {"step_ranged", "step_unranged"}
    assert contracts["contracts"]["step_ranged"][
        "analysis_plausibility_range"
    ] == {
        "minimum": 0.0,
        "maximum": 10.0,
    }
    assert (
        "analysis_plausibility_range"
        not in contracts["contracts"]["step_unranged"]
    )
    scope = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("step_ranged", "step_unranged"),
        raw_input_contracts=contracts,
    )
    assert scope.expected_columns == ("step_ranged",)
    assert scope.authority_kind == "resolved_raw_input_contracts"


def test_exact_contract_with_no_range_compiles_to_an_empty_scope() -> None:
    scope = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step(),
        raw_input_contracts=_contracts(unranged=("receipt_only_column",)),
    )

    assert scope.expected_columns == ()


def test_scope_identity_is_order_stable_and_binds_expected_columns() -> None:
    first = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("b", "a"),
        raw_input_contracts=_contracts(ranged=("b", "a")),
    )
    second = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("a", "b"),
        raw_input_contracts=_contracts(ranged=("a", "b")),
    )

    assert first.expected_columns == ("a", "b")
    assert first.source_contracts_sha256 == second.source_contracts_sha256
    assert first.scope_sha256 == second.scope_sha256
    assert first.to_dict()["scope_sha256"] == first.scope_sha256


def test_scope_cannot_be_reused_for_another_step() -> None:
    scope = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("step_ranged"),
        raw_input_contracts=_contracts(ranged=("step_ranged",)),
    )

    with pytest.raises(PlausibilityScopeError, match="different analysis step"):
        scope.require_step("04_other")


def test_compiled_step_authority_freezes_contracts_and_rejects_scope_drift() -> None:
    step = _step("step_ranged")
    contracts = _contracts(ranged=("step_ranged",))
    authority = compile_step_plausibility_authority(
        context=_context(),
        step=step,
        raw_input_contracts=contracts,
    )

    projection = authority.raw_input_contracts()
    projection["contracts"]["step_ranged"]["column"] = "tampered"
    assert authority.raw_input_contracts()["contracts"]["step_ranged"]["column"] == (
        "step_ranged"
    )

    mismatched_scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=(),
        source_contracts_sha256=authority.scope.source_contracts_sha256,
        authority_kind=authority.scope.authority_kind,
    )
    with pytest.raises(PlausibilityScopeError, match="does not match"):
        type(authority)(
            scope=mismatched_scope,
            raw_input_contracts_canonical_json=(
                authority.raw_input_contracts_canonical_json
            ),
        )


@pytest.mark.parametrize("columns", [("table:typed",), (" spaced",), ("b", "a")])
def test_scope_rejects_non_raw_or_noncanonical_columns(
    columns: tuple[str, ...],
) -> None:
    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )

    with pytest.raises(PlausibilityScopeError):
        FlagOnlyPlausibilityScope(
            step_id="03_synthetic",
            expected_columns=columns,
            source_contracts_sha256="0" * 64,
            authority_kind="test",
        )


def test_tampered_raw_contract_digest_fails_closed() -> None:
    contracts = _contracts(ranged=("step_ranged",))
    contracts["contracts"]["step_ranged"]["analysis_plausibility_range"][  # type: ignore[index]
        "maximum"
    ] = 11.0

    with pytest.raises(PlausibilityScopeError, match="digest is invalid"):
        compile_flag_only_plausibility_scope(
            context=_context(),
            step=_step("step_ranged"),
            raw_input_contracts=contracts,
        )


def test_range_without_retain_and_flag_policy_fails_closed() -> None:
    contracts = _contracts(ranged=("step_ranged",))
    contracts["contracts"]["step_ranged"].pop("plausibility_policy")  # type: ignore[index,union-attr]
    payload = dict(contracts)
    payload.pop("contracts_sha256")
    contracts["contracts_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(PlausibilityScopeError, match="without an action"):
        compile_flag_only_plausibility_scope(
            context=_context(),
            step=_step("step_ranged"),
            raw_input_contracts=contracts,
        )


def test_legacy_fallback_is_narrowed_to_exact_raw_step_inputs() -> None:
    scope = compile_flag_only_plausibility_scope(
        context=_context(),
        step=_step("step_unranged"),
        raw_input_contracts=None,
    )

    assert scope.expected_columns == ()
    assert scope.authority_kind == "legacy_step_raw_inputs"


def _sealed_resume_fixture(
    *,
    step: AnalysisStep,
    sealed_payload: dict[str, object],
    stage: str = "executed",
    candidate_code_sha256: str = "b" * 64,
    execution_code_sha256: str | None = None,
    recorded_resolved_inputs_sha256: str | None = None,
    execution_resolved_inputs_sha256: str | None = None,
) -> tuple[dict[str, object], SimpleNamespace, bytes]:
    sealed_bytes = json.dumps(sealed_payload).encode("utf-8")
    sealed_sha256 = hashlib.sha256(sealed_bytes).hexdigest()
    recorded_inputs_sha256 = recorded_resolved_inputs_sha256 or sealed_sha256
    execution_inputs_sha256 = (
        execution_resolved_inputs_sha256 or recorded_inputs_sha256
    )
    execution = (
        None
        if stage in {"candidate", "concept_audited"}
        else SimpleNamespace(
            code_sha256=execution_code_sha256 or candidate_code_sha256,
            resolved_inputs_sha256=execution_inputs_sha256,
            returncode=0,
            timed_out=False,
            outputs_safe_to_collect=True,
        )
    )
    selected = SimpleNamespace(
        capsule=SimpleNamespace(
            stage=stage,
            candidate_code=SimpleNamespace(sha256=candidate_code_sha256),
            resolved_inputs=SimpleNamespace(sha256=sealed_sha256),
            execution=execution,
        )
    )
    prior_record: dict[str, object] = {
        "status": "ok",
        "step_id": step.step_id,
        "executed_code_sha256": candidate_code_sha256,
        "resolved_inputs_sha256": recorded_inputs_sha256,
    }
    return prior_record, selected, sealed_bytes


def test_resume_scope_reads_verified_capsule_not_mutable_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.step_capsule import StepAuthorityCapsuleRef
    from easyicu.research_agent.authority import plausibility

    step = _step("step_ranged")
    capsule_ref = StepAuthorityCapsuleRef(
        step_id=step.step_id,
        capsule_sha256="a" * 64,
    )
    sealed_payload = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "planner_declared_inputs": ["step_ranged"],
        "inputs": {},
        "raw_input_contracts": _contracts(ranged=("step_ranged",)),
    }
    prior_record, selected, sealed_bytes = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
    )
    monkeypatch.setattr(
        plausibility,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        plausibility,
        "read_verified_content",
        lambda *_args, **_kwargs: sealed_bytes,
    )

    prior_record.update(
        {
            "step_authority_capsule_ref": capsule_ref.model_dump(mode="json"),
            # Observability only: this mutable projection must be ignored.
            "flag_only_plausibility_scope": {
                "expected_columns": ["global_only"]
            },
        }
    )
    scope = plausibility.compile_resumed_flag_only_plausibility_scope(
        prior_record=prior_record,
        run_dir=tmp_path,
        context=_context(),
        step=step,
    )

    assert scope.expected_columns == ("step_ranged",)
    assert (
        scope.source_contracts_sha256
        == sealed_payload["raw_input_contracts"]["contracts_sha256"]
    )


def test_resume_scope_rejects_success_without_capsule(tmp_path: Path) -> None:
    from easyicu.research_agent.authority import plausibility

    step = _step("step_ranged")
    sealed_payload = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "planner_declared_inputs": ["step_ranged"],
        "inputs": {},
        "raw_input_contracts": _contracts(ranged=("step_ranged",)),
    }
    prior_record, _, _ = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
    )

    with pytest.raises(PlausibilityScopeError, match="lacks sealed step authority"):
        plausibility.compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=tmp_path,
            context=_context(),
            step=step,
        )


def test_resume_scope_rejects_executed_capsule_without_raw_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.step_capsule import StepAuthorityCapsuleRef
    from easyicu.research_agent.authority import plausibility

    step = _step()
    capsule_ref = StepAuthorityCapsuleRef(
        step_id=step.step_id,
        capsule_sha256="a" * 64,
    )
    sealed_payload = {
        "schema_version": "easyicu.resolved_inputs/2",
        "step_id": step.step_id,
        "planner_declared_inputs": [],
        "inputs": {},
        "host_verified_cohort_execution_receipt": {"resolved_column": "step_ranged"},
    }
    prior_record, selected, sealed_bytes = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
    )
    prior_record["step_authority_capsule_ref"] = capsule_ref.model_dump(mode="json")
    monkeypatch.setattr(
        plausibility,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        plausibility,
        "read_verified_content",
        lambda *_args, **_kwargs: sealed_bytes,
    )

    with pytest.raises(PlausibilityScopeError, match="lacks raw-input contracts"):
        plausibility.compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=tmp_path,
            context=_context(),
            step=step,
        )


@pytest.mark.parametrize("stage", ["candidate", "concept_audited"])
def test_resume_scope_rejects_nonexecuted_capsule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    from easyicu.research_agent.authority.step_capsule import StepAuthorityCapsuleRef
    from easyicu.research_agent.authority import plausibility

    step = _step("step_ranged")
    capsule_ref = StepAuthorityCapsuleRef(
        step_id=step.step_id,
        capsule_sha256="a" * 64,
    )
    sealed_payload = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "planner_declared_inputs": ["step_ranged"],
        "inputs": {},
        "raw_input_contracts": _contracts(ranged=("step_ranged",)),
    }
    prior_record, selected, _ = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
        stage=stage,
    )
    prior_record["step_authority_capsule_ref"] = capsule_ref.model_dump(mode="json")
    monkeypatch.setattr(
        plausibility,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: selected,
    )

    with pytest.raises(PlausibilityScopeError, match="does not select an executed"):
        plausibility.compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=tmp_path,
            context=_context(),
            step=step,
        )


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ("code", "code does not match"),
        ("inputs", "inputs do not match"),
    ],
)
def test_resume_scope_rejects_capsule_from_another_successful_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    coordinate: str,
    message: str,
) -> None:
    from easyicu.research_agent.authority.step_capsule import StepAuthorityCapsuleRef
    from easyicu.research_agent.authority import plausibility

    step = _step("step_ranged")
    capsule_ref = StepAuthorityCapsuleRef(
        step_id=step.step_id,
        capsule_sha256="a" * 64,
    )
    sealed_payload = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "planner_declared_inputs": ["step_ranged"],
        "inputs": {},
        "raw_input_contracts": _contracts(ranged=("step_ranged",)),
    }
    fixture_kwargs: dict[str, object] = {}
    if coordinate == "code":
        fixture_kwargs["execution_code_sha256"] = "c" * 64
    else:
        fixture_kwargs["recorded_resolved_inputs_sha256"] = "d" * 64
    prior_record, selected, _ = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
        **fixture_kwargs,
    )
    prior_record["step_authority_capsule_ref"] = capsule_ref.model_dump(mode="json")
    monkeypatch.setattr(
        plausibility,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: selected,
    )

    with pytest.raises(PlausibilityScopeError, match=message):
        plausibility.compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=tmp_path,
            context=_context(),
            step=step,
        )


def test_resume_scope_fails_closed_when_sealed_resolved_inputs_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.step_capsule import StepAuthorityCapsuleRef
    from easyicu.research_agent.authority import plausibility

    step = _step("step_ranged")
    capsule_ref = StepAuthorityCapsuleRef(
        step_id=step.step_id,
        capsule_sha256="a" * 64,
    )
    sealed_payload = {
        "schema_version": "2.1",
        "step_id": step.step_id,
        "planner_declared_inputs": ["step_ranged"],
        "inputs": {},
        "raw_input_contracts": _contracts(ranged=("step_ranged",)),
    }
    prior_record, selected, _ = _sealed_resume_fixture(
        step=step,
        sealed_payload=sealed_payload,
    )
    prior_record["step_authority_capsule_ref"] = capsule_ref.model_dump(mode="json")
    monkeypatch.setattr(
        plausibility,
        "load_verified_step_authority_capsule",
        lambda *_args, **_kwargs: selected,
    )
    monkeypatch.setattr(
        plausibility,
        "read_verified_content",
        lambda *_args, **_kwargs: b"{not-json",
    )

    with pytest.raises(
        PlausibilityScopeError,
        match="sealed resolved-input authority cannot be verified",
    ):
        plausibility.compile_resumed_flag_only_plausibility_scope(
            prior_record=prior_record,
            run_dir=tmp_path,
            context=_context(),
            step=step,
        )
