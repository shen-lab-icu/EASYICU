from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.cohort.materializer import _hash_df, _sha256_file
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
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
