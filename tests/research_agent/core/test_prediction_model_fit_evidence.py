"""V4 fit authority reaches the existing V3.1 bridge without loose artifacts."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.prediction_model_fit_evidence import (
    PredictionModelFitEvidenceEnvelope,
    PredictionModelFitEvidenceError,
    PredictionModelFitEvidenceReason,
    PredictionModelFitRuntimeAuthority,
    register_prediction_model_fit_validation_artifact,
)
from easyicu.research_agent.authority.prediction_validation_evidence import (
    prediction_validation_analysis_registration_findings,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    verified_run_evidence_path,
)
from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.contracts.prediction_validation import (
    PredictionValidationArtifactBinding,
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationRuntimeIdentity,
    PredictionValidationSpec,
)
from easyicu.research_agent.contracts.prediction_model_fit import (
    PredictionModelFitError,
    PredictionModelFitReason,
)
from easyicu.research_agent.prediction_model_fit_owner import (
    fit_binary_prediction_model,
)

from tests.research_agent.core.test_prediction_model_fit_owner import _loaded_input, _source_frame, _spec


RUN_ID = "prediction_fit_validation_v5"


def _validation_spec(**updates: object) -> PredictionValidationSpec:
    payload: dict[str, object] = {
        "unit_id_column": "record_id",
        "subject_id_column": "subject_id",
        "split_column": "split",
        "outcome_column": "outcome",
        "probability_column": "probability",
        "evaluation_split": "test",
        "analysis_unit": "subject",
        "thresholds": (0.25, 0.5, 0.75),
        "calibration_bins": 2,
    }
    payload.update(updates)
    return PredictionValidationSpec.model_validate(payload)


def _runtime_authority(
    store: EvidenceStore,
) -> tuple[PredictionModelFitRuntimeAuthority, dict[str, object]]:
    code = store.register_text(
        kind="code",
        description="Exact V5 test source snapshot.",
        text="exact-source-tree-v5\n",
        filename="source_snapshot.txt",
        evidence_id="prediction_fit_code_snapshot",
        produced_by_step="00_runtime",
        producer="prediction_model_fit",
        generation_mode="deterministic_skill",
        metadata={"run_id": RUN_ID},
        publish_aliases=False,
    )
    environment = store.register_text(
        kind="code",
        description="Exact V5 test environment lock.",
        text="easyicu==1.0.0\nscikit-learn==1.9.0\n",
        filename="environment.lock",
        evidence_id="prediction_fit_environment_lock",
        produced_by_step="00_runtime",
        producer="prediction_model_fit",
        generation_mode="deterministic_skill",
        metadata={"run_id": RUN_ID},
        publish_aliases=False,
    )
    runtime = PredictionValidationRuntimeIdentity(
        git_commit="a" * 40,
        git_dirty=False,
        source_tree_sha256=code.sha256,
        environment_sha256=environment.sha256,
        runtime_kind="local_process",
        container_image_digest=None,
        python_version="3.11.15",
        package_version="1.0.0",
    )
    runtime_record = store.register_text(
        kind="log",
        description="Exact V5 test runtime identity.",
        text=canonical_json(runtime.model_dump(mode="json"), trailing_newline=True),
        filename="runtime_identity.json",
        evidence_id="prediction_fit_runtime_receipt",
        produced_by_step="00_runtime",
        producer="prediction_model_fit",
        generation_mode="deterministic_skill",
        metadata={"run_id": RUN_ID},
        publish_aliases=False,
    )
    records = {
        "code_snapshot": code,
        "environment_lock": environment,
        "runtime_receipt": runtime_record,
    }
    bindings = tuple(
        PredictionValidationArtifactBinding(
            role=role,
            evidence_id=record.evidence_id,
            sha256=record.sha256,
            kind=record.kind,
            produced_by_step=record.produced_by_step,
        )
        for role, record in records.items()
    )
    authority = PredictionModelFitRuntimeAuthority(
        producer_run_id=RUN_ID,
        runtime=runtime,
        artifacts=bindings,
    )
    return authority, records


def _prepared(tmp_path: Path) -> dict[str, object]:
    store = EvidenceStore(tmp_path / "evidence_run")
    runtime_authority, runtime_records = _runtime_authority(store)
    source_input = _loaded_input(tmp_path / "typed_source", _source_frame())
    fit_spec = _spec()
    fit_bundle = fit_binary_prediction_model(
        source_input=source_input,
        spec=fit_spec,
    )
    return {
        "store": store,
        "runtime_authority": runtime_authority,
        "runtime_records": runtime_records,
        "source_input": source_input,
        "fit_spec": fit_spec,
        "fit_bundle": fit_bundle,
        "validation_spec": _validation_spec(),
    }


def _register(prepared: dict[str, object]):
    return register_prediction_model_fit_validation_artifact(
        evidence_store=prepared["store"],
        source_input=prepared["source_input"],
        fit_spec=prepared["fit_spec"],
        fit_bundle=prepared["fit_bundle"],
        validation_spec=prepared["validation_spec"],
        runtime_authority=prepared["runtime_authority"],
        fit_step_id="01_fit",
        validation_step_id="02_validate",
    )


def test_host_materializes_v4_into_one_v3_1_analysis_only_lineage(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)

    registration = _register(prepared)
    store = prepared["store"]

    assert len(store.records()) == 8
    assert tuple(binding.role for binding in registration.lineage.artifacts) == (
        "prediction_table",
        "cohort",
        "split_assignment",
        "model_artifact",
        "code_snapshot",
        "environment_lock",
        "runtime_receipt",
    )
    assert registration.analysis_registration.upstream_evidence_ids == tuple(
        binding.evidence_id for binding in registration.lineage.artifacts
    )
    assert registration.claim_ceiling == "analysis_only"
    assert registration.paper_authorization is False
    assert registration.planner_selection_authorized is False
    assert store.aliases() == {}
    assert store.numeric_claims() == []
    assert store.scientific_claims() == []
    assert (
        prediction_validation_analysis_registration_findings(
            evidence_store=store,
            registration=registration.analysis_registration,
        )
        == ()
    )

    model_binding = registration.lineage.binding_for("model_artifact")
    model_record = store.get(model_binding.evidence_id)
    assert model_record is not None
    model_path = verified_run_evidence_path(store.root, model_record)
    assert model_path is not None
    envelope = PredictionModelFitEvidenceEnvelope.model_validate_json(
        model_path.read_text(encoding="utf-8")
    )
    assert envelope.fit_receipt == prepared["fit_bundle"].receipt
    assert envelope.model_artifact == prepared["fit_bundle"].model_artifact
    assert envelope.source_input_receipt == prepared["source_input"].receipt
    assert (
        envelope.source_projection_sha256
        == registration.lineage.binding_for("cohort").sha256
    )
    assert envelope.split_assignment_artifact_sha256 == (
        registration.lineage.binding_for("split_assignment").sha256
    )
    assert (
        envelope.prediction_table_sha256
        == registration.lineage.binding_for("prediction_table").sha256
    )

    repeated = _register(prepared)
    assert repeated == registration
    assert len(store.records()) == 8

    reloaded_store = EvidenceStore(store.root)
    assert reloaded_store.aliases() == {}
    assert reloaded_store.numeric_claims() == []
    assert reloaded_store.scientific_claims() == []
    assert (
        prediction_validation_analysis_registration_findings(
            evidence_store=reloaded_store,
            registration=registration.analysis_registration,
        )
        == ()
    )


def test_public_bridge_accepts_no_loose_lineage_or_model_outputs() -> None:
    parameters = set(
        inspect.signature(register_prediction_model_fit_validation_artifact).parameters
    )

    assert parameters == {
        "evidence_store",
        "source_input",
        "fit_spec",
        "fit_bundle",
        "validation_spec",
        "runtime_authority",
        "fit_step_id",
        "validation_step_id",
    }
    assert not parameters & {
        "lineage",
        "prediction_table",
        "model_artifact",
        "fit_receipt",
        "validation_receipt",
        "validation_seal",
    }


def test_source_drift_fails_before_any_fit_artifact_is_registered(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    changed = _source_frame()
    changed.loc[8, "age"] += 1.0
    prepared["source_input"] = _loaded_input(tmp_path / "changed", changed)

    with pytest.raises(PredictionModelFitError) as caught:
        _register(prepared)

    assert caught.value.reason_code is PredictionModelFitReason.RECOMPUTATION_MISMATCH
    assert len(prepared["store"].records()) == 3


def test_validation_coordinates_must_match_the_fit_bundle(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    prepared["validation_spec"] = _validation_spec(
        probability_column="other_probability"
    )

    with pytest.raises(PredictionModelFitEvidenceError) as caught:
        _register(prepared)

    assert (
        caught.value.reason_code
        is PredictionModelFitEvidenceReason.VALIDATION_CONTRACT_MISMATCH
    )
    assert len(prepared["store"].records()) == 3


def test_runtime_authority_is_verified_before_fit_materialization(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    code_record = prepared["runtime_records"]["code_snapshot"]
    code_path = verified_run_evidence_path(prepared["store"].root, code_record)
    assert code_path is not None
    code_path.write_text("tampered-source\n", encoding="utf-8")

    with pytest.raises(PredictionValidationError) as caught:
        _register(prepared)

    assert caught.value.reason_code is PredictionValidationReason.LINEAGE_EVIDENCE_STALE
    assert len(prepared["store"].records()) == 3


@pytest.mark.parametrize(
    "role",
    ["prediction_table", "cohort", "split_assignment", "model_artifact"],
)
def test_later_fit_role_drift_invalidates_the_analysis(
    tmp_path: Path,
    role: str,
) -> None:
    prepared = _prepared(tmp_path)
    registration = _register(prepared)
    binding = registration.lineage.binding_for(role)
    record = prepared["store"].get(binding.evidence_id)
    assert record is not None
    artifact_path = verified_run_evidence_path(prepared["store"].root, record)
    assert artifact_path is not None
    artifact_path.write_text("{}\n", encoding="utf-8")

    findings = prediction_validation_analysis_registration_findings(
        evidence_store=prepared["store"],
        registration=registration.analysis_registration,
    )

    assert len(findings) == 1
    assert findings[0].reason_code is PredictionValidationReason.LINEAGE_EVIDENCE_STALE


def test_fit_evidence_bridge_has_no_planner_or_selection_import() -> None:
    import easyicu.research_agent.authority.prediction_model_fit_evidence as owner

    tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert not any("planning" in module for module in imported_modules)
    assert not any(
        "execution.runners.selection" in module for module in imported_modules
    )
