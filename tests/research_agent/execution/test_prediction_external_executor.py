"""Direct owner tests; every cohort and binding is synthetic."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from easyicu.concept.metadata_projection import ConceptColumnRole
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.research_agent.contracts.prediction_validation import (
    ExternalValidationError, ExternalValidationSpec, FrozenLogisticModel, FrozenPredictor,
    FROZEN_MODEL_INPUT, external_cohort_input,
)
from easyicu.research_agent.execution.runners.prediction_validation_executor import run_external_validation_step
from easyicu.research_agent.execution.runners.typed_input_binding import TypedInputBindingError
from easyicu.research_agent.intake import typed_cohort_source as intake
from easyicu.research_agent.replication.metrics import (
    require_transport_capabilities, transport_concept_meaning,
)
from tests.support.typed_export import metadata_binding


def _binding(key, path, frame, run_dir):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    kind, _, product = key.partition(":")
    identity = dict(input_key=key, declared_kind=kind, product=product, sha256=digest,
                    evidence_id=product, produced_by_step="synthetic_intake")
    return {**identity, "identity_row": identity, "evidence_kind": "table",
            "relative_path": str(path.relative_to(run_dir)),
            "product_contract": {"columns": list(frame), "row_count": len(frame)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest}}


def _fixture(tmp_path, monkeypatch):
    profile = research_capability_profile("eicu")
    metadata = {name: metadata_binding(name, name, ConceptColumnRole.VALUE, database="eicu")
                for name in ("hr", "death")}
    meaning = {name: transport_concept_meaning(binding, profile) for name, binding in metadata.items()}
    model = FrozenLogisticModel(
        development_database="miiv", development_source_sha256="a" * 64,
        outcome_concept="death", outcome_meaning=meaning["death"], intercept=-1,
        predictors=(FrozenPredictor(concept="hr", meaning=meaning["hr"], coefficient=0.5,
                                    median=80, center=80, scale=10, development_mean=80,
                                    development_sd=10, development_missing_fraction=0),),
    )
    payload = model.model_dump_json()
    spec = ExternalValidationSpec(development_database="miiv", validation_databases=("eicu",),
                                  frozen_model_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                                  predictor_concepts=("hr",), outcome_concept="death")
    frame = pd.DataFrame({"subject_id": np.arange(120), "hr": np.tile(np.arange(10), 12) + 75.,
                          "death": np.tile([0, 1, 0, 0, 1, 0], 20)})
    path = tmp_path / "validation.csv"
    frame.to_csv(path, index=False)
    model_frame = pd.DataFrame({"model_json": [payload]})
    model_path = tmp_path / "frozen.csv"
    model_frame.to_csv(model_path, index=False)
    binding = _binding(external_cohort_input("eicu"), path, frame, tmp_path)
    verified = SimpleNamespace(
        authority=SimpleNamespace(cohort_sha256=binding["sha256"], cohort_columns=tuple(frame.columns)),
        sidecar=SimpleNamespace(source_database="eicu", source_database_class_prefixes=(), files=(
            SimpleNamespace(relative_path=path.name, columns=metadata),
        )),
    )
    reads = []

    def load_verified(candidate):
        reads.append(candidate.name)
        return verified

    monkeypatch.setattr(intake, "load_verified_materialized_cohort_authority", load_verified)
    manifest = {"step_id": "external", "inputs": {
        FROZEN_MODEL_INPUT: _binding(FROZEN_MODEL_INPUT, model_path, model_frame, tmp_path),
        external_cohort_input("eicu"): binding,
    }}
    return spec, manifest, verified, reads


def test_typed_runner_writes_only_aggregate_tables_figures_and_tripod_map(tmp_path, monkeypatch):
    spec, manifest, _, reads = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "outputs"
    result = run_external_validation_step(spec=spec, run_dir=tmp_path, resolved_inputs=manifest,
                                           step_id="external", out_dir=output)
    assert result["performance_rows"] == 1
    assert reads == ["validation.csv"]
    assert (output / "external_calibration_eicu.png").stat().st_size > 1000
    assert (output / "external_calibration.png").stat().st_size > 1000
    performance = pd.read_csv(output / "external_validation.csv")
    assert performance["database"].tolist() == ["eicu"]
    assert performance["result_kind"].tolist() == ["untouched_transport"]
    for path in output.glob("*.csv"):
        assert "subject_id" not in pd.read_csv(path).columns
    provenance = json.loads((output / "external_validation_provenance.json").read_text())
    assert [p["event"] for p in provenance["provenance"]] == ["model_frozen", "validation_read"]
    assert "TRIPOD" in (output / "external_validation_tripod.md").read_text()


def test_nonestimable_spline_keeps_a_typed_status_table(tmp_path, monkeypatch):
    from easyicu.research_agent.methods.prediction_validation import ValidationSource, run_external_validation
    from easyicu.research_agent.reporting.supplement_package import write_external_validation_report

    spec, _, verified, _ = _fixture(tmp_path, monkeypatch)
    payload = pd.read_csv(tmp_path / "frozen.csv").iloc[0]["model_json"].encode()
    model = FrozenLogisticModel.model_validate_json(payload)
    frame = pd.read_csv(tmp_path / "validation.csv")
    frame["hr"] = 80
    result = run_external_validation(model_bytes=payload, spec=spec, sources=(ValidationSource(
        database="eicu", columns=tuple(frame), meanings={"hr": model.predictors[0].meaning, "death": model.outcome_meaning},
        read=lambda: (frame, verified.authority.cohort_sha256),
    ),))
    output = tmp_path / "report"
    write_external_validation_report(result, output)
    table = pd.read_csv(output / "external_calibration.csv")
    assert table["method"].tolist() == ["not_estimable_insufficient_support"]
    assert table["predicted"].isna().all()


def test_frozen_model_is_verified_before_intake_can_read_validation_bytes(tmp_path, monkeypatch):
    spec, manifest, _, reads = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ExternalValidationError, match="frozen_model_digest_mismatch"):
        run_external_validation_step(spec=spec.model_copy(update={"frozen_model_sha256": "0" * 64}),
                                     run_dir=tmp_path, resolved_inputs=manifest,
                                     step_id="external", out_dir=tmp_path / "outputs")
    assert reads == []
    assert not (tmp_path / "outputs").exists()


def test_external_source_rejects_a_mismatched_consumption_digest(tmp_path, monkeypatch):
    spec, manifest, _, _ = _fixture(tmp_path, monkeypatch)
    key = external_cohort_input("eicu")
    manifest["inputs"][key]["consumption_contract"]["artifact_sha256"] = "0" * 64
    with pytest.raises(TypedInputBindingError) as excinfo:
        run_external_validation_step(
            spec=spec, run_dir=tmp_path, resolved_inputs=manifest,
            step_id="external", out_dir=tmp_path / "outputs",
        )
    assert excinfo.value.reason_code == "consumption_contract_mismatch"
    assert not (tmp_path / "outputs").exists()


def test_harmonization_selects_the_exact_meaning_among_auxiliary_columns(tmp_path, monkeypatch):
    spec, manifest, verified, _ = _fixture(tmp_path, monkeypatch)
    frame = pd.read_csv(tmp_path / "validation.csv")
    frame["hr_measured"] = True
    frame.to_csv(tmp_path / "validation.csv", index=False)
    key = external_cohort_input("eicu")
    manifest["inputs"][key] = _binding(key, tmp_path / "validation.csv", frame, tmp_path)
    verified.authority.cohort_sha256 = manifest["inputs"][key]["sha256"]
    verified.authority.cohort_columns = tuple(frame)
    verified.sidecar.files[0].columns["hr_measured"] = metadata_binding(
        "hr", "hr_measured", ConceptColumnRole.MEASUREMENT_STATUS, database="eicu",
    )
    result = run_external_validation_step(spec=spec, run_dir=tmp_path, resolved_inputs=manifest,
                                          step_id="external", out_dir=tmp_path / "outputs")
    assert result["performance_rows"] == 1


@pytest.mark.parametrize("defect,reason", [
    ("missing_authority", "validation_source_authority_missing"),
    ("wrong_digest", "validation_source_binding_invalid"),
    ("missing_predictor", "predictor_unavailable"),
    ("ambiguous_predictor", "predictor_unavailable"),
])
def test_intake_refuses_unverifiable_or_ambiguous_sources(tmp_path, monkeypatch, defect, reason):
    spec, manifest, verified, _ = _fixture(tmp_path, monkeypatch)
    if defect == "missing_authority":
        monkeypatch.setattr(intake, "load_verified_materialized_cohort_authority", lambda _: None)
    elif defect == "wrong_digest":
        verified.authority.cohort_sha256 = "0" * 64
    elif defect == "missing_predictor":
        verified.sidecar.files[0].columns.pop("hr")
    else:
        verified.sidecar.files[0].columns["other_hr"] = verified.sidecar.files[0].columns["hr"]
    with pytest.raises(ExternalValidationError, match=reason):
        run_external_validation_step(spec=spec, run_dir=tmp_path, resolved_inputs=manifest,
                                     step_id="external", out_dir=tmp_path / "outputs")


def test_database_capability_gate_does_not_accept_an_available_column(tmp_path, monkeypatch):
    spec, manifest, _, _ = _fixture(tmp_path, monkeypatch)
    payload = pd.read_csv(tmp_path / "frozen.csv").iloc[0]["model_json"]
    model = FrozenLogisticModel.model_validate_json(payload)
    profile = research_capability_profile("eicu")
    denied = profile.model_copy(update={"concepts": tuple(
        c.model_copy(update={"mapped": False}) if c.concept == "hr" else c for c in profile.concepts
    )})
    with pytest.raises(ExternalValidationError, match="predictor_unavailable"):
        require_transport_capabilities(denied, model)
    monkeypatch.setattr(intake, "research_capability_profile", lambda _: denied)
    with pytest.raises(ExternalValidationError, match="predictor_unavailable"):
        run_external_validation_step(spec=spec, run_dir=tmp_path, resolved_inputs=manifest,
                                     step_id="external", out_dir=tmp_path / "outputs")


def test_database_capability_gate_refuses_an_unavailable_outcome(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    model = FrozenLogisticModel.model_validate_json(pd.read_csv(tmp_path / "frozen.csv").iloc[0]["model_json"])
    profile = research_capability_profile("eicu")
    denied = profile.model_copy(update={"concepts": tuple(
        c.model_copy(update={"mapped": False}) if c.concept == "death" else c for c in profile.concepts
    )})
    with pytest.raises(ExternalValidationError, match="outcome_unavailable"):
        require_transport_capabilities(denied, model)


def test_changed_source_outcome_and_predictor_representation_are_disclosed():
    death = metadata_binding("death", "death", ConceptColumnRole.VALUE, database="miiv")
    age = metadata_binding("age", "age", ConceptColumnRole.VALUE, database="miiv")
    miiv, hirid = (research_capability_profile(db) for db in ("miiv", "hirid"))
    assert transport_concept_meaning(death, miiv) != transport_concept_meaning(death, hirid)
    assert transport_concept_meaning(age, miiv) != transport_concept_meaning(age, hirid)
