"""A first-stay claim needs independent source and materialization authorities."""
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.first_icu_stay import (
    FirstIcuStayPerHospitalizationBinding, derive_first_icu_stay_per_hospitalization,
)
from easyicu.research_agent.cohort.materializer import materialize_to_parquet
from easyicu.research_agent.intake.export_package import open_export_package
from easyicu.research_agent.intake.materialized_metadata import load_verified_materialized_cohort_authority
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.run_population_proof import verified_first_stay_applied
from tests.support.typed_export import typed_export


@pytest.fixture
def facts(tmp_path, monkeypatch):
    first = typed_export(tmp_path / "first", database="eicu")
    second = typed_export(tmp_path / "second", database="eicu")
    # Different packages with identical ICU coordinates.
    manifest = second / "_manifest.json"
    data = json.loads(manifest.read_text())
    data["synthetic_source_marker"] = "other"
    manifest.write_text(json.dumps(data))
    paths = [first.resolve(), second.resolve()]
    identities = [SourceIdentity("eicu", sources._source_id(str(p))) for p in paths]
    rows = [{"path": str(p), "id": ident.registration_id, "database": "eicu"}
            for p, ident in zip(paths, identities)]
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"sources": rows}))
    monkeypatch.setattr(sources, "_CONFIG_PATH", registry)
    monkeypatch.setattr(sources, "path_allowed", lambda *args, **kwargs: True)
    derived = derive_first_icu_stay_per_hospitalization(pd.DataFrame({
        "patientunitstayid": [1, 2], "patienthealthsystemstayid": [1, 1],
        "unitvisitnumber": [2, 1], "unittype": ["MICU", "MICU"],
    }))
    coordinate = tmp_path / "coordinate.parquet"
    derived.frame.to_parquet(coordinate, index=False)
    digest = hashlib.sha256(coordinate.read_bytes()).hexdigest()
    acquisitions, bindings = [], []
    for i, source in enumerate(paths):
        with open_export_package(source) as package:
            coords = {**derived.receipt,
                      "schema_version": "easyicu.first_icu_stay_per_hospitalization_runtime_authority/1",
                      "authority_ref": f"synthetic-{i}", "database": "eicu",
                      "export_manifest_file": package.manifest_path.name,
                      "export_manifest_sha256": package.manifest_sha256,
                      "identity_table_sha256": "a" * 64,
                      "coordinate_sha256": digest, "provider_visible_values": False}
        binding = FirstIcuStayPerHospitalizationBinding(coordinate, digest, coords)
        out = materialize_to_parquet(
            tmp_path / f"materialized{i}", data_path=source, database="eicu",
            static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
            **binding.materializer_kwargs(),
        )
        verified = load_verified_materialized_cohort_authority(out["parquet"])
        acquisitions.append(SimpleNamespace(blocked=False, universe_path=out["parquet"],
                            cohort_authority_ref=verified.reference,
                            cohort_authority_path=out["parquet"].parent / verified.reference.file))
        bindings.append(binding)
    scientific = SimpleNamespace(metadata_only_planning=False,
        study={"cohort": {"first_icu_stay_per_hospitalization": True}},
        foundation_profile={"first_icu_stay_per_hospitalization": bindings[0]})
    return SimpleNamespace(paths=paths, ids=identities, acquisitions=acquisitions,
                           bindings=bindings, scientific=scientific, registry=registry,
                           coordinate=coordinate)


def check(f, acquisition=0, selected=0):
    return verified_first_stay_applied(f.scientific, f.acquisitions[acquisition],
        export_path=str(f.paths[selected]), source_identity=f.ids[selected].to_dict())


def test_real_materialization_matches_all_independent_authorities(facts):
    assert check(facts)
    assert not check(facts, acquisition=1)
    assert not check(facts, selected=1)
    assert not check(facts, acquisition=1, selected=1)  # original approved rule is source A
    facts.scientific.foundation_profile["first_icu_stay_per_hospitalization"] = facts.bindings[1]
    assert check(facts, acquisition=1, selected=1)


def test_complete_replacement_does_not_replace_acquisition_ref(facts):
    original = facts.acquisitions[0]
    replacement = facts.acquisitions[1]
    original.universe_path = replacement.universe_path
    original.cohort_authority_path = original.universe_path.parent / original.cohort_authority_ref.file
    assert not check(facts)


@pytest.mark.parametrize("change", ["missing_ref", "missing_path", "bad_identity", "registry_removed", "coordinate_drift", "export_drift", "binding_drift"])
def test_missing_or_changed_authority_is_not_reported_applied(facts, change):
    if change == "missing_ref":
        facts.acquisitions[0].cohort_authority_ref = None
    elif change == "missing_path":
        facts.acquisitions[0].cohort_authority_path = None
    elif change == "bad_identity":
        facts.ids[0] = SourceIdentity("eicu", "different")
    elif change == "registry_removed":
        facts.registry.write_text(json.dumps({"sources": []}))
    elif change == "coordinate_drift":
        facts.coordinate.write_bytes(b"changed")
    elif change == "export_drift":
        with (facts.paths[0] / "labs.parquet").open("ab") as handle:
            handle.write(b"changed")
    elif change == "binding_drift":
        old = facts.bindings[0]
        coordinates = {**old.authority_coordinates, "identity_table_sha256": "b" * 64}
        facts.scientific.foundation_profile["first_icu_stay_per_hospitalization"] = FirstIcuStayPerHospitalizationBinding(old.coordinate_path, old.coordinate_sha256, coordinates)
    assert not check(facts)


def test_patient_first_real_materialization_is_bound_to_selected_export(tmp_path, monkeypatch):
    from easyicu.webserver import raw_source_authority, source_identity_authority

    monkeypatch.setattr(source_identity_authority.state_paths, "state_root", lambda: tmp_path / "state")
    monkeypatch.setattr(raw_source_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent.env")
    monkeypatch.setattr(source_identity_authority, "_DEFAULT_CONFIG_PATH", tmp_path / "absent-grouping.env")
    monkeypatch.setattr(sources, "_CONFIG_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(sources, "path_allowed", lambda *args, **kwargs: True)
    paths, identities, bindings, acquisitions = [], [], [], []
    for i in range(2):
        raw = tmp_path / f"raw{i}"
        raw.mkdir()
        pd.DataFrame({"stay_id": [1, 2], "hadm_id": [101, 102], "subject_id": [7, 7],
                      "intime": ["2026-01-01", "2026-02-01"]}).to_parquet(raw / "icustays.parquet", index=False)
        pd.DataFrame({"hadm_id": [101, 102], "dischtime": ["2026-06-01"] * 2,
                      "deathtime": [None, None], "hospital_expire_flag": [0, 0]}).to_parquet(raw / "admissions.parquet", index=False)
        source = typed_export(tmp_path / f"patient_export{i}").resolve()
        manifest = source / "_manifest.json"
        data = json.loads(manifest.read_text())
        data["data_path"] = str(raw)
        manifest.write_text(json.dumps(data))
        binding = source_identity_authority.resolve_study_first_icu_stay(export_path=source, database="miiv")
        assert binding is not None
        out = materialize_to_parquet(
            tmp_path / f"patient_materialized{i}", data_path=source, database="miiv",
            static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
            **binding.materializer_kwargs(),
        )
        assert pd.read_parquet(out["parquet"])["stay_id"].tolist() == [1]
        verified = load_verified_materialized_cohort_authority(out["parquet"])
        acquisitions.append(SimpleNamespace(blocked=False, universe_path=out["parquet"],
            cohort_authority_ref=verified.reference,
            cohort_authority_path=out["parquet"].parent / verified.reference.file))
        paths.append(source)
        identities.append(SourceIdentity("miiv", sources._source_id(str(source))))
        bindings.append(binding)
    assert bindings[0].coordinate_sha256 == bindings[1].coordinate_sha256
    (tmp_path / "registry.json").write_text(json.dumps({"sources": [
        {"path": str(path), "id": identity.registration_id, "database": "miiv"}
        for path, identity in zip(paths, identities)
    ]}))
    scientific = SimpleNamespace(metadata_only_planning=False,
        study={"cohort": {"exclude_readmissions": True}},
        foundation_profile={"first_icu_stay": bindings[0]})
    facts = SimpleNamespace(paths=paths, ids=identities, acquisitions=acquisitions,
                            scientific=scientific)
    assert check(facts)
    assert not check(facts, acquisition=1)
    assert not check(facts, selected=1)
    assert not check(facts, acquisition=1, selected=1)
    scientific.foundation_profile["first_icu_stay"] = bindings[1]
    assert check(facts, acquisition=1, selected=1)
