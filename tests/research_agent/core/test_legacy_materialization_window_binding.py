from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.cohort.materializer import _hash_df, _sha256_file
from easyicu.research_agent.intake.legacy_materialization import (
    legacy_first_icu_stay_restriction,
    load_verified_legacy_materialization_provenance,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)
from easyicu.research_agent.planning.dependence_authority import (
    repeat_units_possible,
)
from easyicu.research_agent.planning.scientific_review import (
    patient_identity_available,
)


def _write_legacy_materialization(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "age": [40.0, 50.0, 60.0],
            "marker_max": [0.0, 1.0, 2.0],
            "marker_n": [1, 2, 3],
            "marker_measured": [1, 1, 1],
            "death": [0, 0, 1],
        }
    )
    frame.to_parquet(path, index=False)
    provenance = {
        "schema_version": "easyicu.cohort_materializer/1",
        "source_mode": "export",
        "source": "/verified/export",
        "database": "synthetic",
        "cohort_window_hours": [0.0, 24.0],
        "feature_concepts": ["marker"],
        "outcome_concepts": ["death"],
        "static_concepts": ["age"],
        "cohort_definition": None,
        "n_stays_extracted": len(frame),
        "n_stays_after_inclusion_exclusion": len(frame),
        "unavailable_concepts": [],
        "event_indicator_columns_normalized": [],
        "columns": list(frame.columns),
        "cohort_sha256": _hash_df(frame.reset_index(drop=True)),
        "cohort_file_sha256": _sha256_file(path),
        "cohort_file_size": path.stat().st_size,
        "build_seconds": 0.1,
    }
    path.with_name(f"{path.stem}_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame


def test_pipeline_stages_legacy_materialization_window_for_context(
    ra, tmp_path: Path, monkeypatch
) -> None:
    from easyicu.research_agent.research_context import builder as context_builder

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "universe.parquet"
    _write_legacy_materialization(source)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    pipeline = object.__new__(ra.ResearchAgentPipeline)
    staged = pipeline._materialise_cohort(source, run_dir)

    assert staged.read_bytes() == source.read_bytes()
    assert _sha256_file(staged) == _sha256_file(source)
    staged_provenance = run_dir / "cohort_provenance.json"
    assert staged_provenance.is_file()
    assert (
        staged_provenance.read_bytes()
        == source.with_name("universe_provenance.json").read_bytes()
    )

    monkeypatch.setattr(
        context_builder,
        "_safe_get_concept_info",
        lambda name: (
            {"name": "marker", "description": "A marker."} if name == "marker" else None
        ),
    )
    context = ra.build_research_context(
        research_question="Evaluate a first-window marker against death.",
        cohort=staged,
        cohort_name="legacy_materialized",
        database="synthetic",
        target_outcome="death",
        primary_exposure="marker_max",
    )

    for column in ("marker_max", "marker_n", "marker_measured"):
        assert context.variable(column).analysis_window == "icu_admission[0,24]h"
    assert context.variable("marker_n").unit_normalization == "window_nonnull_count"
    assert "Non-null observation count" in context.variable("marker_n").description
    assert context.variable("marker_measured").unit_normalization == "window_measurement_status"
    assert "Measurement availability" in context.variable("marker_measured").description
    assert context.variable("age").analysis_window is None
    assert context.variable("death").analysis_window is None
    assert context.cohort.provenance["materialized_cohort_window_hours"] == [
        0.0,
        24.0,
    ]
    assert (
        len(context.cohort.provenance["materialized_cohort_provenance_sha256"])
        == hashlib.sha256().digest_size * 2
    )


@pytest.mark.parametrize("suffix", ["first_time", "last_time"])
def test_verified_legacy_companion_time_has_time_not_score_semantics(suffix) -> None:
    from easyicu.research_agent.research_context.builder import (
        _apply_legacy_materialization_window,
    )
    from easyicu.research_agent.research_context.representation_semantics import (
        compile_wide_representation_semantics,
    )
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

    descriptor = ConceptDescriptor(
        name=f"organ_score_{suffix}",
        dtype="float64",
        role=VariableRole.ORDINAL_SCORE,
        is_ordinal=True,
        ordinal_levels=[0, 1, 2, 3, 4],
        valid_range=(0, 4),
    )
    projected = _apply_legacy_materialization_window(
        descriptors=[descriptor],
        provenance={"cohort_window_hours": [0, 24], "feature_concepts": ["organ_score"]},
    )
    compiled = compile_wide_representation_semantics(projected)[0]

    assert compiled.role == VariableRole.TIME
    assert compiled.unit == "h"
    assert compiled.valid_range is None
    assert compiled.is_ordinal is False
    assert compiled.ordinal_levels is None
    assert compiled.analysis_window == "icu_admission[0,24]h"
    assert compiled.source_concept == "organ_score"
    assert compiled.unit_normalization == f"window_{suffix}"


def test_legacy_companion_name_alone_does_not_grant_representation_authority() -> None:
    from easyicu.research_agent.research_context.builder import (
        _apply_legacy_materialization_window,
    )
    from easyicu.research_agent.schema import ConceptDescriptor

    descriptor = ConceptDescriptor(name="unbound_first_time", dtype="float64")
    projected = _apply_legacy_materialization_window(
        descriptors=[descriptor],
        provenance={"cohort_window_hours": [0, 24], "feature_concepts": ["different"]},
    )
    assert projected == [descriptor]


def test_legacy_materialization_window_fails_closed_on_cohort_tamper(
    ra, tmp_path: Path
) -> None:
    source = tmp_path / "cohort.parquet"
    frame = _write_legacy_materialization(source)
    frame.loc[0, "marker_max"] = 3.0
    frame.to_parquet(source, index=False)

    with pytest.raises(
        MaterializedMetadataError,
        match="file binding does not match cohort",
    ):
        ra.build_research_context(
            research_question="Evaluate a marker.",
            cohort=source,
            cohort_name="tampered",
            database="synthetic",
        )


def test_unbound_legacy_window_receipt_is_not_accepted(ra, tmp_path: Path) -> None:
    source = tmp_path / "cohort.parquet"
    _write_legacy_materialization(source)
    provenance_path = tmp_path / "cohort_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("cohort_file_sha256")
    provenance.pop("cohort_file_size")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(
        MaterializedMetadataError,
        match="lacks required fields",
    ):
        ra.build_research_context(
            research_question="Evaluate a marker.",
            cohort=source,
            cohort_name="unbound_receipt",
            database="synthetic",
        )


def test_verified_composite_patient_grouping_reaches_scientific_review(
    ra, tmp_path: Path
) -> None:
    source = tmp_path / "patient_grouped.parquet"
    frame = pd.DataFrame(
        {
            "patient_stay_id": ["p10:s1", "p10:s2", "p20:s3"],
            "age": [40.0, 41.0, 70.0],
            "death": [0, 1, 0],
        }
    )
    frame.to_parquet(source, index=False)
    provenance = {
        "schema_version": "easyicu.cohort_materializer/1",
        "cohort_window_hours": [0.0, 24.0],
        "feature_concepts": [],
        "outcome_concepts": ["death"],
        "static_concepts": ["age"],
        "n_stays_after_inclusion_exclusion": len(frame),
        "columns": list(frame.columns),
        "cohort_sha256": _hash_df(frame.reset_index(drop=True)),
        "cohort_file_sha256": _sha256_file(source),
        "cohort_file_size": source.stat().st_size,
        "replacement_row_identity": {
            "mapping_file_sha256": "a" * 64,
            "output_identity_column": "patient_stay_id",
            "mapped_cohort_rows": len(frame),
            "patient_group_derivation": {
                "algorithm": "prefix_before_:s",
                "delimiter": ":s",
            },
            "authority_coordinates": {
                "schema_version": "easyicu.patient_grouping_runtime_authority/1",
                "authority_ref": "owner/bridge/v1",
                "provider_visible_values": False,
            },
        },
    }
    source.with_name("patient_grouped_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    context = ra.build_research_context(
        research_question="Estimate a patient-clustered association.",
        cohort=source,
        cohort_name="patient_grouped",
        database="synthetic",
        target_outcome="death",
    )

    assert context.cohort.id_columns == ["patient_stay_id"]
    assert context.cohort.provenance["replacement_row_identity"][
        "mapping_file_sha256"
    ] == "a" * 64
    assert patient_identity_available(context) is True


def _materialize_legacy_universe(
    tmp_path: Path, monkeypatch, *, first_stay_flags: dict | None
) -> Path:
    """Write a universe the way an untyped export package does: no typed
    authority, only the materializer's ``<stem>_provenance.json`` receipt."""

    tmp_path.mkdir()
    wide = pd.DataFrame(
        {
            "stay_id": [11, 12, 13, 14, 15],
            "age": [60.0, 61.0, 70.0, 45.0, 52.0],
            "death": [0, 1, 0, 0, 1],
        }
    )
    provenance = {
        "schema_version": "easyicu.cohort_materializer/1",
        "source_mode": "export",
        "export_authority": None,
        "database": "synthetic",
        "cohort_window_hours": [0.0, 24.0],
        "feature_concepts": [],
        "outcome_concepts": ["death"],
        "static_concepts": ["age"],
        "cohort_definition": None,
        "n_stays_extracted": len(wide),
        "n_stays_after_inclusion_exclusion": len(wide),
        "unavailable_concepts": [],
        "event_indicator_columns_normalized": [],
        "declared_positive_only_event_concepts": [],
        "host_derivations": [],
        "source_bounds_violation_policy": "reject",
        "source_bounds_exclusions": {},
        "columns": list(wide.columns),
        "cohort_sha256": _hash_df(wide),
    }
    untyped = SimpleNamespace(enabled=False, seal_existing_cohort=lambda **_kw: None)
    monkeypatch.setattr(
        materializer,
        "_materialize_cohort_with_metadata",
        lambda **_kwargs: (wide.copy(), dict(provenance), untyped),
    )
    first_stay: dict = {}
    if first_stay_flags is not None:
        coordinate = tmp_path / "first_icu_stay.parquet"
        pd.DataFrame(
            {
                "stay_id": list(first_stay_flags),
                "first_icu_stay": list(first_stay_flags.values()),
            }
        ).to_parquet(coordinate, index=False)
        digest = _sha256_file(coordinate)
        first_stay = {
            "first_icu_stay_path": coordinate.absolute(),
            "first_icu_stay_sha256": digest,
            "first_icu_stay_authority_coordinates": {
                "coordinate_sha256": digest,
                "provider_visible_values": False,
            },
        }
    paths = materializer.materialize_to_parquet(
        tmp_path / "universe",
        stem="cohort",
        feature_concepts=[],
        database="synthetic",
        data_path=str(tmp_path),
        outcome_concepts=["death"],
        static_concepts=["age"],
        **first_stay,
    )
    assert "cohort_authority" not in paths
    return paths["parquet"]


def test_first_icu_stay_restriction_in_a_legacy_receipt_rules_out_repeats(
    ra, tmp_path: Path, monkeypatch
) -> None:
    """A stay-keyed universe restricted to first ICU stays has one row per
    patient; the context must say so even when no typed authority exists."""

    flags = {11: True, 12: True, 13: False, 14: True, 15: True, 16: False}
    universe = _materialize_legacy_universe(
        tmp_path / "restricted", monkeypatch, first_stay_flags=flags
    )
    unrestricted = _materialize_legacy_universe(
        tmp_path / "unrestricted", monkeypatch, first_stay_flags=None
    )

    def context(cohort: Path):
        return ra.build_research_context(
            research_question="Describe the association of age with death.",
            cohort=cohort,
            cohort_name="stay_keyed",
            database="synthetic",
            target_outcome="death",
        )

    restricted_context = context(universe)
    digest = load_verified_legacy_materialization_provenance(universe)[
        "first_icu_stay_restriction"
    ]["coordinate_sha256"]
    assert restricted_context.cohort.n_stays == 4
    assert restricted_context.cohort.provenance["first_icu_stay_restriction"] == {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": digest,
        "stays_after": 4,
    }
    assert patient_identity_available(restricted_context) is False
    assert repeat_units_possible(restricted_context) is False

    unrestricted_context = context(unrestricted)
    assert "first_icu_stay_restriction" not in unrestricted_context.cohort.provenance
    assert repeat_units_possible(unrestricted_context) is True


def _restricted_legacy_receipt(tmp_path: Path) -> Path:
    source = tmp_path / "cohort.parquet"
    frame = _write_legacy_materialization(source)
    provenance_path = tmp_path / "cohort_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["first_icu_stay_restriction"] = {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": "c" * 64,
        "stays_before": len(frame) + 2,
        "stays_after": len(frame),
        "non_first_icu_stays_removed": 2,
        "authority_coordinates": {
            "coordinate_sha256": "c" * 64,
            "provider_visible_values": False,
        },
    }
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    return source


def test_verified_legacy_restriction_projects_only_aggregate_fields(
    tmp_path: Path,
) -> None:
    verified = load_verified_legacy_materialization_provenance(
        _restricted_legacy_receipt(tmp_path)
    )

    assert legacy_first_icu_stay_restriction(verified) == {
        "schema_version": "easyicu.first_icu_stay_restriction/1",
        "coordinate_sha256": "c" * 64,
        "stays_after": 3,
    }
    verified.pop("first_icu_stay_restriction")
    assert legacy_first_icu_stay_restriction(verified) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: "restricted",
        lambda r: {**r, "schema_version": "easyicu.first_icu_stay_restriction/0"},
        lambda r: {**r, "coordinate_sha256": "C" * 64, "authority_coordinates": {}},
        lambda r: {**r, "coordinate_sha256": "c" * 63, "authority_coordinates": {}},
        lambda r: {key: value for key, value in r.items() if key != "coordinate_sha256"},
        lambda r: {**r, "stays_before": 4, "stays_after": 2, "non_first_icu_stays_removed": 2},
        lambda r: {**r, "non_first_icu_stays_removed": 1},
        lambda r: {**r, "stays_before": 4, "non_first_icu_stays_removed": True},
        lambda r: {**r, "non_first_icu_stays_removed": 2.0},
        lambda r: {**r, "stays_before": 1, "non_first_icu_stays_removed": -2},
        lambda r: {**r, "authority_coordinates": None},
        lambda r: {**r, "authority_coordinates": {"coordinate_sha256": "d" * 64}},
        lambda r: {**r, "authority_coordinates": {"provider_visible_values": True}},
    ],
    ids=[
        "not_an_object",
        "schema",
        "coordinate_case",
        "coordinate_length",
        "coordinate_missing",
        "stays_after_not_the_rows",
        "removed_count",
        "boolean_count",
        "float_count",
        "negative_count",
        "coordinates_not_an_object",
        "coordinates_name_another_digest",
        "values_visible_to_provider",
    ],
)
def test_malformed_legacy_restriction_fails_closed(
    ra, tmp_path: Path, mutate
) -> None:
    source = _restricted_legacy_receipt(tmp_path)
    provenance_path = tmp_path / "cohort_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["first_icu_stay_restriction"] = mutate(
        provenance["first_icu_stay_restriction"]
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(MaterializedMetadataError, match="first ICU stay restriction"):
        ra.build_research_context(
            research_question="Evaluate a marker.",
            cohort=source,
            cohort_name="malformed_restriction",
            database="synthetic",
        )
