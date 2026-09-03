"""Persisted V5 fits are re-fitted under one captured clean runtime."""

from __future__ import annotations

import ast
import inspect
import io
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.prediction_model_fit_evidence import (
    PredictionModelFitEvidenceEnvelope,
    register_prediction_model_fit_validation_artifact,
)
from easyicu.research_agent.authority.prediction_model_fit_revalidation import (
    PredictionModelFitPersistedValidationError,
    PredictionModelFitPersistedValidationReason,
    revalidate_persisted_prediction_model_fit_validation,
)
from easyicu.research_agent.authority.prediction_model_fit_runtime import (
    PredictionModelFitCodeSnapshot,
    PredictionModelFitEnvironmentLock,
    PredictionModelFitRuntimeCaptureError,
    PredictionModelFitRuntimeCaptureReason,
    capture_prediction_model_fit_runtime_authority,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    verified_run_evidence_path,
)
from easyicu.research_agent.contracts.prediction_model_fit import (
    PredictionModelFitError,
    PredictionModelFitReason,
    prediction_model_artifact_bytes,
)
from easyicu.research_agent.prediction_model_fit_owner import (
    fit_binary_prediction_model,
    revalidate_prediction_model_fit_persisted_artifacts,
)

from tests.research_agent.core.test_prediction_model_fit_evidence import RUN_ID, _validation_spec
from tests.research_agent.core.test_prediction_model_fit_owner import _loaded_input, _source_frame, _spec


GIT_COMMIT = "c" * 40
GIT_TREE = "d" * 40


def _patch_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    commit: str = GIT_COMMIT,
    tree: str = GIT_TREE,
    status: str = "",
) -> None:
    import easyicu.research_agent.authority.prediction_model_fit_runtime as owner

    repository_root = Path(owner.__file__).resolve().parents[4]
    responses = {
        ("rev-parse", "--show-toplevel"): str(repository_root),
        ("status", "--porcelain", "--untracked-files=normal"): status,
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "HEAD^{tree}"): tree,
    }

    def fake_run_git(*arguments: str) -> str:
        return responses[arguments]

    monkeypatch.setattr(owner, "_run_git", fake_run_git)


def _prepared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    _patch_git(monkeypatch)
    store = EvidenceStore(tmp_path / "evidence_run")
    source_input = _loaded_input(tmp_path / "typed_source", _source_frame())
    fit_spec = _spec()
    fit_bundle = fit_binary_prediction_model(
        source_input=source_input,
        spec=fit_spec,
    )
    runtime_authority = capture_prediction_model_fit_runtime_authority(
        evidence_store=store,
        fit_bundle=fit_bundle,
        producer_run_id=RUN_ID,
        runtime_step_id="00_runtime",
    )
    registration = register_prediction_model_fit_validation_artifact(
        evidence_store=store,
        source_input=source_input,
        fit_spec=fit_spec,
        fit_bundle=fit_bundle,
        validation_spec=_validation_spec(),
        runtime_authority=runtime_authority,
        fit_step_id="01_fit",
        validation_step_id="02_validate",
    )
    return {
        "store": store,
        "source_input": source_input,
        "fit_spec": fit_spec,
        "fit_bundle": fit_bundle,
        "runtime_authority": runtime_authority,
        "registration": registration,
    }


def _artifact_bytes(
    store: EvidenceStore,
    registration: object,
    role: str,
) -> bytes:
    binding = registration.lineage.binding_for(role)
    record = store.get(binding.evidence_id)
    assert record is not None
    path = verified_run_evidence_path(store.root, record)
    assert path is not None
    return path.read_bytes()


def test_capture_and_reload_refit_one_closed_analysis_only_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    store = prepared["store"]
    runtime_authority = prepared["runtime_authority"]
    registration = prepared["registration"]

    receipt = revalidate_persisted_prediction_model_fit_validation(
        evidence_store=store,
        registration=registration,
    )

    assert receipt.fit_receipt_sha256 == prepared["fit_bundle"].receipt.receipt_sha256
    assert receipt.claim_ceiling == "analysis_only"
    assert receipt.paper_authorization is False
    assert receipt.planner_selection_authorized is False
    assert tuple(binding.role for binding in runtime_authority.artifacts) == (
        "code_snapshot",
        "environment_lock",
        "runtime_receipt",
    )
    code = PredictionModelFitCodeSnapshot.model_validate_json(
        _artifact_bytes(store, registration, "code_snapshot")
    )
    environment = PredictionModelFitEnvironmentLock.model_validate_json(
        _artifact_bytes(store, registration, "environment_lock")
    )
    assert code.git_commit == GIT_COMMIT
    assert code.git_tree == GIT_TREE
    assert (
        environment.package_versions == prepared["fit_bundle"].receipt.package_versions
    )
    assert store.aliases() == {}
    assert store.numeric_claims() == []
    assert store.scientific_claims() == []
    repeated_runtime = capture_prediction_model_fit_runtime_authority(
        evidence_store=store,
        fit_bundle=prepared["fit_bundle"],
        producer_run_id=RUN_ID,
        runtime_step_id="00_runtime",
    )
    assert repeated_runtime == runtime_authority
    assert len(store.records()) == 8

    reloaded = EvidenceStore(store.root)
    assert (
        revalidate_persisted_prediction_model_fit_validation(
            evidence_store=reloaded,
            registration=registration,
        )
        == receipt
    )


def test_public_runtime_and_refit_routes_accept_no_caller_identity_or_artifacts() -> (
    None
):
    capture_parameters = set(
        inspect.signature(capture_prediction_model_fit_runtime_authority).parameters
    )
    refit_parameters = set(
        inspect.signature(
            revalidate_persisted_prediction_model_fit_validation
        ).parameters
    )

    assert capture_parameters == {
        "evidence_store",
        "fit_bundle",
        "producer_run_id",
        "runtime_step_id",
    }
    assert refit_parameters == {"evidence_store", "registration"}
    assert not capture_parameters & {
        "repository_root",
        "environment_lock",
        "runtime",
        "artifacts",
        "git_commit",
        "source_tree_sha256",
        "environment_sha256",
    }
    assert not refit_parameters & {
        "source_path",
        "source_bytes",
        "lineage",
        "model_artifact",
        "prediction_table",
        "fit_receipt",
    }


def test_dirty_checkout_fails_before_runtime_evidence_is_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_git(monkeypatch, status=" M src/easyicu/research_agent/example.py")
    store = EvidenceStore(tmp_path / "evidence_run")
    source_input = _loaded_input(tmp_path / "typed_source", _source_frame())
    fit_bundle = fit_binary_prediction_model(source_input=source_input, spec=_spec())

    with pytest.raises(PredictionModelFitRuntimeCaptureError) as caught:
        capture_prediction_model_fit_runtime_authority(
            evidence_store=store,
            fit_bundle=fit_bundle,
            producer_run_id=RUN_ID,
            runtime_step_id="00_runtime",
        )

    assert (
        caught.value.reason_code
        is PredictionModelFitRuntimeCaptureReason.CHECKOUT_DIRTY
    )
    assert store.records() == []


def test_fit_environment_drift_fails_before_runtime_evidence_is_registered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent.authority.prediction_model_fit_runtime as owner

    _patch_git(monkeypatch)
    store = EvidenceStore(tmp_path / "evidence_run")
    source_input = _loaded_input(tmp_path / "typed_source", _source_frame())
    fit_bundle = fit_binary_prediction_model(source_input=source_input, spec=_spec())
    real_version = owner.importlib.metadata.version

    def changed_version(distribution: str) -> str:
        if distribution == "scikit-learn":
            return "999.0.0"
        return real_version(distribution)

    monkeypatch.setattr(owner.importlib.metadata, "version", changed_version)

    with pytest.raises(PredictionModelFitRuntimeCaptureError) as caught:
        capture_prediction_model_fit_runtime_authority(
            evidence_store=store,
            fit_bundle=fit_bundle,
            producer_run_id=RUN_ID,
            runtime_step_id="00_runtime",
        )

    assert (
        caught.value.reason_code
        is PredictionModelFitRuntimeCaptureReason.ENVIRONMENT_MISMATCH
    )
    assert store.records() == []


def test_current_checkout_drift_invalidates_persisted_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    _patch_git(monkeypatch, commit="e" * 40)

    with pytest.raises(PredictionModelFitRuntimeCaptureError) as caught:
        revalidate_persisted_prediction_model_fit_validation(
            evidence_store=prepared["store"],
            registration=prepared["registration"],
        )

    assert (
        caught.value.reason_code
        is PredictionModelFitRuntimeCaptureReason.PERSISTED_RUNTIME_MISMATCH
    )


def test_current_environment_drift_invalidates_persisted_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import easyicu.research_agent.authority.prediction_model_fit_runtime as owner

    prepared = _prepared(tmp_path, monkeypatch)
    real_version = owner.importlib.metadata.version

    def changed_version(distribution: str) -> str:
        if distribution == "numpy":
            return "999.0.0"
        return real_version(distribution)

    monkeypatch.setattr(owner.importlib.metadata, "version", changed_version)

    with pytest.raises(PredictionModelFitRuntimeCaptureError) as caught:
        revalidate_persisted_prediction_model_fit_validation(
            evidence_store=prepared["store"],
            registration=prepared["registration"],
        )

    assert (
        caught.value.reason_code
        is PredictionModelFitRuntimeCaptureReason.ENVIRONMENT_MISMATCH
    )


def test_persisted_source_is_re_fitted_not_only_digest_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    store = prepared["store"]
    registration = prepared["registration"]
    source_bytes = _artifact_bytes(store, registration, "cohort")
    prediction_bytes = _artifact_bytes(store, registration, "prediction_table")
    envelope = PredictionModelFitEvidenceEnvelope.model_validate_json(
        _artifact_bytes(store, registration, "model_artifact")
    )
    changed = pd.read_csv(io.BytesIO(source_bytes))
    changed.loc[0, "age"] = float(changed.loc[0, "age"]) + 100.0
    changed_bytes = changed.to_csv(index=False, lineterminator="\n").encode("utf-8")

    with pytest.raises(PredictionModelFitError) as caught:
        revalidate_prediction_model_fit_persisted_artifacts(
            source_projection_csv_bytes=changed_bytes,
            spec=envelope.fit_spec,
            source_receipt=envelope.source_input_receipt,
            expected_fit_receipt=envelope.fit_receipt,
            model_artifact_bytes=prediction_model_artifact_bytes(
                envelope.model_artifact
            ),
            prediction_csv_bytes=prediction_bytes,
        )

    assert caught.value.reason_code is PredictionModelFitReason.RECOMPUTATION_MISMATCH


def test_current_evidence_drift_fails_before_persisted_refit_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, monkeypatch)
    store = prepared["store"]
    registration = prepared["registration"]
    binding = registration.lineage.binding_for("cohort")
    record = store.get(binding.evidence_id)
    assert record is not None
    path = verified_run_evidence_path(store.root, record)
    assert path is not None
    path.write_text("subject_id\nchanged\n", encoding="utf-8")

    with pytest.raises(PredictionModelFitPersistedValidationError) as caught:
        revalidate_persisted_prediction_model_fit_validation(
            evidence_store=store,
            registration=registration,
        )

    assert (
        caught.value.reason_code
        is PredictionModelFitPersistedValidationReason.EVIDENCE_INVALID
    )


def test_v5_1_authority_modules_have_no_planner_or_selection_import() -> None:
    import easyicu.research_agent.authority.prediction_model_fit_revalidation as refit
    import easyicu.research_agent.authority.prediction_model_fit_runtime as runtime

    for owner in (refit, runtime):
        tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
        imported_modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("planning" in module for module in imported_modules)
        assert not any(
            "execution.runners.selection" in module for module in imported_modules
        )
