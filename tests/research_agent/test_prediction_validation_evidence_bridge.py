"""Upstream lineage and analysis-only EvidenceStore bridge contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.prediction_validation_evidence import (
    prediction_validation_analysis_registration_findings,
    register_prediction_validation_analysis_artifact,
)
from easyicu.research_agent.authority.runtime_artifacts import (
    verified_run_evidence_path,
)
from easyicu.research_agent.contracts.prediction_validation import (
    PredictionValidationAnalysisBundle,
    PredictionValidationAnalysisPolicy,
    PredictionValidationArtifactBinding,
    PredictionValidationError,
    PredictionValidationReason,
    PredictionValidationRuntimeIdentity,
    PredictionValidationSpec,
    PredictionValidationUpstreamLineage,
)
from easyicu.research_agent.execution.runners.prediction_validation_executor import (
    run_prediction_validation_csv,
    seal_prediction_validation_receipt,
)


DATA = Path(__file__).parent / "data"
SOURCE = DATA / "oracle_prediction_validation.csv"
RUN_ID = "run_prediction_validation_v3"


def _spec() -> PredictionValidationSpec:
    return PredictionValidationSpec(
        unit_id_column="row_id",
        subject_id_column="subject_id",
        split_column="split",
        outcome_column="outcome",
        probability_column="probability",
        evaluation_split="test",
        analysis_unit="subject",
        thresholds=(0.5, 0.8),
        calibration_bins=4,
    )


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _register(
    store: EvidenceStore,
    *,
    source_path: Path,
    evidence_id: str,
    kind: str,
    produced_by_step: str,
):
    return store.register_file(
        kind=kind,
        description=f"V3 upstream {evidence_id}.",
        source_path=source_path,
        evidence_id=evidence_id,
        produced_by_step=produced_by_step,
        metadata={"run_id": RUN_ID},
        publish_aliases=False,
    )


def _binding(role: str, record) -> PredictionValidationArtifactBinding:
    return PredictionValidationArtifactBinding(
        role=role,
        evidence_id=record.evidence_id,
        sha256=record.sha256,
        kind=record.kind,
        produced_by_step=record.produced_by_step,
    )


def _prepared_authority(tmp_path: Path) -> dict[str, object]:
    store = EvidenceStore(tmp_path / "run")
    inputs = tmp_path / "inputs"
    records = {
        "prediction_table": _register(
            store,
            source_path=SOURCE,
            evidence_id="prediction_table",
            kind="table",
            produced_by_step="02_predict",
        ),
        "cohort": _register(
            store,
            source_path=_write(
                inputs / "cohort.csv", b"subject_id,in_cohort\nsubject-1,1\n"
            ),
            evidence_id="prediction_cohort",
            kind="table",
            produced_by_step="00_prepare",
        ),
        "split_assignment": _register(
            store,
            source_path=_write(
                inputs / "split.csv", b"subject_id,split\nsubject-1,train\n"
            ),
            evidence_id="prediction_split",
            kind="table",
            produced_by_step="00_prepare",
        ),
        "model_artifact": _register(
            store,
            source_path=_write(inputs / "model.bin", b"sealed-model-v1"),
            evidence_id="prediction_model",
            kind="code",
            produced_by_step="01_fit",
        ),
        "code_snapshot": _register(
            store,
            source_path=_write(inputs / "source.tar", b"exact-source-tree-v1"),
            evidence_id="prediction_code_snapshot",
            kind="code",
            produced_by_step="00_runtime",
        ),
        "environment_lock": _register(
            store,
            source_path=_write(
                inputs / "requirements.lock", b"easyicu==1.0.0\npandas==2.3.3\n"
            ),
            evidence_id="prediction_environment_lock",
            kind="code",
            produced_by_step="00_runtime",
        ),
    }
    runtime = PredictionValidationRuntimeIdentity(
        git_commit="f" * 40,
        git_dirty=False,
        source_tree_sha256=records["code_snapshot"].sha256,
        environment_sha256=records["environment_lock"].sha256,
        runtime_kind="container",
        container_image_digest="sha256:" + "a" * 64,
        python_version="3.11.15",
        package_version="1.0.0",
    )
    runtime_path = inputs / "runtime_identity.json"
    runtime_path.write_text(
        json.dumps(runtime.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    records["runtime_receipt"] = _register(
        store,
        source_path=runtime_path,
        evidence_id="prediction_runtime_receipt",
        kind="log",
        produced_by_step="00_runtime",
    )
    bindings = tuple(
        _binding(role, records[role])
        for role in (
            "prediction_table",
            "cohort",
            "split_assignment",
            "model_artifact",
            "code_snapshot",
            "environment_lock",
            "runtime_receipt",
        )
    )
    lineage = PredictionValidationUpstreamLineage(
        producer_run_id=RUN_ID,
        model_identifier="binary-risk-model-v1",
        evaluation_split="test",
        split_policy="subject_disjoint",
        runtime=runtime,
        artifacts=bindings,
    )
    prediction_path = verified_run_evidence_path(
        store.root, records["prediction_table"]
    )
    assert prediction_path is not None
    spec = _spec()
    receipt = run_prediction_validation_csv(
        source_path=prediction_path,
        expected_source_sha256=records["prediction_table"].sha256,
        spec=spec,
    )
    validation_seal = seal_prediction_validation_receipt(
        source_path=prediction_path,
        expected_source_sha256=records["prediction_table"].sha256,
        spec=spec,
        candidate=receipt,
        runtime_identity=runtime,
    )
    return {
        "store": store,
        "records": records,
        "runtime": runtime,
        "lineage": lineage,
        "spec": spec,
        "receipt": receipt,
        "validation_seal": validation_seal,
        "inputs": inputs,
    }


def _register_analysis(prepared: dict[str, object]):
    return register_prediction_validation_analysis_artifact(
        evidence_store=prepared["store"],
        spec=prepared["spec"],
        receipt=prepared["receipt"],
        validation_seal=prepared["validation_seal"],
        lineage=prepared["lineage"],
        validation_step_id="03_validate",
    )


def test_bridge_registers_one_verified_analysis_only_bundle(tmp_path: Path) -> None:
    prepared = _prepared_authority(tmp_path)
    registration = _register_analysis(prepared)
    store = prepared["store"]
    record = store.get(registration.evidence_id)

    assert record is not None
    assert record.kind == "statistic"
    assert record.produced_by_step == "03_validate"
    assert record.producer == "prediction_validation"
    assert record.generation_mode == "deterministic_skill"
    assert record.metadata["claim_ceiling"] == "analysis_only"
    assert record.metadata["paper_authorization"] is False
    assert record.metadata["planner_selection_authorized"] is False
    assert record.metadata["numeric_claim_registration_authorized"] is False
    assert record.metadata["scientific_claim_registration_authorized"] is False
    assert registration.paper_authorization is False
    assert registration.aliases_published is False
    assert registration.numeric_claim_count_delta == 0
    assert registration.scientific_claim_count_delta == 0
    assert record.evidence_id not in store.aliases()
    assert record.evidence_id not in store.aliases().values()
    assert store.numeric_claims() == []
    assert store.scientific_claims() == []
    assert (
        prediction_validation_analysis_registration_findings(
            evidence_store=store,
            registration=registration,
        )
        == ()
    )

    bundle_path = verified_run_evidence_path(store.root, record)
    assert bundle_path is not None
    bundle = PredictionValidationAnalysisBundle.model_validate_json(
        bundle_path.read_text(encoding="utf-8")
    )
    assert bundle.policy.claim_ceiling == "analysis_only"
    assert bundle.policy.paper_authorization is False
    assert bundle.receipt == prepared["receipt"]
    assert bundle.lineage == prepared["lineage"]

    repeated = _register_analysis(prepared)
    assert repeated == registration
    assert len(store.records()) == 8

    reloaded_store = EvidenceStore(store.root)
    assert reloaded_store.aliases() == {}
    assert reloaded_store.numeric_claims() == []
    assert reloaded_store.scientific_claims() == []
    assert (
        prediction_validation_analysis_registration_findings(
            evidence_store=reloaded_store,
            registration=registration,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing", PredictionValidationReason.LINEAGE_EVIDENCE_MISSING),
        ("digest", PredictionValidationReason.LINEAGE_EVIDENCE_MISMATCH),
        ("stale", PredictionValidationReason.LINEAGE_EVIDENCE_STALE),
    ],
)
def test_bridge_fails_closed_on_upstream_evidence_drift(
    tmp_path: Path,
    mutation: str,
    expected_reason: PredictionValidationReason,
) -> None:
    prepared = _prepared_authority(tmp_path)
    lineage = prepared["lineage"]
    bindings = list(lineage.artifacts)
    model_index = next(
        index
        for index, binding in enumerate(bindings)
        if binding.role == "model_artifact"
    )
    model_binding = bindings[model_index]
    if mutation == "missing":
        bindings[model_index] = model_binding.model_copy(
            update={"evidence_id": "missing_prediction_model"}
        )
    elif mutation == "digest":
        bindings[model_index] = model_binding.model_copy(update={"sha256": "0" * 64})
    else:
        record = prepared["records"]["model_artifact"]
        model_path = verified_run_evidence_path(prepared["store"].root, record)
        assert model_path is not None
        model_path.write_bytes(b"tampered-model")
    changed_lineage = lineage.model_copy(update={"artifacts": tuple(bindings)})

    with pytest.raises(PredictionValidationError) as caught:
        register_prediction_validation_analysis_artifact(
            evidence_store=prepared["store"],
            spec=prepared["spec"],
            receipt=prepared["receipt"],
            validation_seal=prepared["validation_seal"],
            lineage=changed_lineage,
            validation_step_id="03_validate",
        )

    assert caught.value.reason_code is expected_reason
    assert len(prepared["store"].records()) == 7


def test_bridge_rejects_runtime_receipt_semantic_drift(tmp_path: Path) -> None:
    prepared = _prepared_authority(tmp_path)
    changed_runtime = prepared["runtime"].model_copy(
        update={"python_version": "3.12.0"}
    )
    runtime_path = prepared["inputs"] / "different_runtime.json"
    runtime_path.write_text(
        json.dumps(changed_runtime.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    different = _register(
        prepared["store"],
        source_path=runtime_path,
        evidence_id="different_runtime_receipt",
        kind="log",
        produced_by_step="00_runtime",
    )
    bindings = tuple(
        _binding("runtime_receipt", different)
        if binding.role == "runtime_receipt"
        else binding
        for binding in prepared["lineage"].artifacts
    )
    changed_lineage = prepared["lineage"].model_copy(update={"artifacts": bindings})

    with pytest.raises(PredictionValidationError) as caught:
        register_prediction_validation_analysis_artifact(
            evidence_store=prepared["store"],
            spec=prepared["spec"],
            receipt=prepared["receipt"],
            validation_seal=prepared["validation_seal"],
            lineage=changed_lineage,
            validation_step_id="03_validate",
        )

    assert (
        caught.value.reason_code is PredictionValidationReason.LINEAGE_RUNTIME_MISMATCH
    )


def test_host_seal_rejects_a_tampered_receipt(tmp_path: Path) -> None:
    prepared = _prepared_authority(tmp_path)
    invalid = prepared["receipt"].model_copy(update={"paper_authorization": True})
    prediction_path = verified_run_evidence_path(
        prepared["store"].root,
        prepared["records"]["prediction_table"],
    )
    assert prediction_path is not None

    with pytest.raises(PredictionValidationError) as caught:
        seal_prediction_validation_receipt(
            source_path=prediction_path,
            expected_source_sha256=prepared["records"]["prediction_table"].sha256,
            spec=prepared["spec"],
            candidate=invalid,
            runtime_identity=prepared["runtime"],
        )

    assert caught.value.reason_code is PredictionValidationReason.RECEIPT_SCHEMA_INVALID


def test_bundle_rejects_a_validation_seal_from_another_runtime(
    tmp_path: Path,
) -> None:
    prepared = _prepared_authority(tmp_path)
    other_runtime = prepared["runtime"].model_copy(
        update={"container_image_digest": "sha256:" + "b" * 64}
    )
    prediction_path = verified_run_evidence_path(
        prepared["store"].root,
        prepared["records"]["prediction_table"],
    )
    assert prediction_path is not None
    other_seal = seal_prediction_validation_receipt(
        source_path=prediction_path,
        expected_source_sha256=prepared["records"]["prediction_table"].sha256,
        spec=prepared["spec"],
        candidate=prepared["receipt"],
        runtime_identity=other_runtime,
    )

    with pytest.raises(PredictionValidationError) as caught:
        register_prediction_validation_analysis_artifact(
            evidence_store=prepared["store"],
            spec=prepared["spec"],
            receipt=prepared["receipt"],
            validation_seal=other_seal,
            lineage=prepared["lineage"],
            validation_step_id="03_validate",
        )

    assert (
        caught.value.reason_code is PredictionValidationReason.VALIDATION_SEAL_INVALID
    )


def test_registration_validator_detects_later_claim_or_alias_promotion(
    tmp_path: Path,
) -> None:
    prepared = _prepared_authority(tmp_path)
    registration = _register_analysis(prepared)
    store = prepared["store"]
    store.register_numeric_claim(
        value="0.6125",
        canonical=0.6125,
        evidence_id=registration.evidence_id,
        step_id="03_validate",
        source_field="result.summary.auroc",
    )
    store.publish_success_aliases(
        registration.evidence_id,
        aliases=["prediction_validation_primary"],
    )

    findings = prediction_validation_analysis_registration_findings(
        evidence_store=store,
        registration=registration,
    )

    assert len(findings) == 1
    assert (
        findings[0].reason_code
        is PredictionValidationReason.AUTHORITY_CEILING_VIOLATION
    )
    assert findings[0].detail["numeric_claim_count"] == 1
    assert findings[0].detail["published_aliases"] == [
        "prediction_validation_analysis_" + registration.bundle_sha256[:12],
        "prediction_validation_primary",
    ]


def test_analysis_policy_is_closed_and_planner_remains_unwired() -> None:
    with pytest.raises(ValidationError):
        PredictionValidationAnalysisPolicy.model_validate(
            {
                "claim_ceiling": "reportable",
                "paper_authorization": True,
                "planner_selection_authorized": True,
                "numeric_claim_registration_authorized": True,
                "scientific_claim_registration_authorized": True,
                "alias_publication_authorized": True,
            }
        )

    root = Path(__file__).resolve().parents[2]
    selection = (
        root / "src/easyicu/research_agent/execution/runners/selection.py"
    ).read_text(encoding="utf-8")
    registry = (
        root / "src/easyicu/research_agent/planning/capability_registry.py"
    ).read_text(encoding="utf-8")
    bridge = (
        root / "src/easyicu/research_agent/authority/prediction_validation_evidence.py"
    ).read_text(encoding="utf-8")

    assert "prediction_validation" not in selection
    assert 'capability_id="prediction_validation"' not in registry
    assert "execution.runners" not in bridge
    assert "register_numeric_claim" not in bridge
    assert "register_step_summary_scientific_claims" not in bridge


def test_lineage_requires_one_canonical_closed_role_order(tmp_path: Path) -> None:
    lineage = _prepared_authority(tmp_path)["lineage"]
    payload = lineage.model_dump(mode="python")
    payload["artifacts"] = tuple(reversed(payload["artifacts"]))

    with pytest.raises(ValidationError):
        PredictionValidationUpstreamLineage.model_validate(payload)
