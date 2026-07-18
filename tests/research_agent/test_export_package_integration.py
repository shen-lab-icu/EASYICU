"""Integration contracts for the shared EasyICU export-package intake."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent import cohort_materializer
from easyicu.research_agent.data_catalog import build_available_catalog
from easyicu.research_agent.intake import export_package as intake
from easyicu.research_agent.replication.discovery import discover_easyicu_exports


def _write_frame(path: Path, frame: pd.DataFrame, export_format: str) -> None:
    if export_format == "parquet":
        frame.to_parquet(path, index=False)
    elif export_format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_excel(path, index=False, sheet_name="EasyICU export")


def _analysis_export(
    root: Path, *, export_format: str = "parquet", database: str = "miiv"
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    extension = {"parquet": "parquet", "csv": "csv", "excel": "xlsx"}[export_format]
    frames = {
        "demographics": pd.DataFrame({"stay_id": [1, 2], "age": [50, 60]}),
        "outcomes": pd.DataFrame({"stay_id": [1, 2], "death": [0, 1]}),
        "labs": pd.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "charttime": [1.0, 2.0, 1.0],
                "lact": [1.0, 2.0, 3.0],
            }
        ),
    }
    entries = []
    selection = {}
    concept_by_module = {
        "demographics": ["age"],
        "outcomes": ["death"],
        "labs": ["lact"],
    }
    for module, frame in frames.items():
        file_name = f"{module}.{extension}"
        _write_frame(root / file_name, frame, export_format)
        concepts = concept_by_module[module]
        entries.append(
            {
                "file": file_name,
                "module": module,
                "concepts": len(concepts),
                "concept_ids": concepts,
                "rows": len(frame),
            }
        )
        selection[module] = concepts
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": database,
                "format": export_format,
                "concept_selection": {"mode": "explicit", "modules": selection},
                "files": entries,
                "feature_definitions": {"included": False},
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("export_format", ["parquet", "csv", "excel"])
def test_cohort_materializer_accepts_native_export_without_reextracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, export_format: str
) -> None:
    root = _analysis_export(tmp_path / export_format, export_format=export_format)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("native export must never fall through to load_concepts")

    import easyicu.api as easyicu_api

    monkeypatch.setattr(easyicu_api, "load_concepts", forbidden_load)
    cohort, provenance = cohort_materializer.materialize_cohort(
        data_path=root,
        feature_concepts=("lact",),
        static_concepts=("age",),
        outcome_concepts=("death",),
    )

    assert provenance["source_mode"] == "export"
    assert cohort["stay_id"].tolist() == [1, 2]
    assert cohort["age"].tolist() == [50, 60]
    assert cohort["death"].tolist() == [0, 1]
    assert cohort["lact_max"].tolist() == pytest.approx([2.0, 3.0])


def test_invalid_native_marker_never_falls_back_to_converted_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "invalid"
    root.mkdir()
    (root / intake.NATIVE_MANIFEST).write_text("{bad-json")
    calls: list[object] = []

    import easyicu.api as easyicu_api

    monkeypatch.setattr(
        easyicu_api, "load_concepts", lambda *args, **kwargs: calls.append(args)
    )
    with pytest.raises(intake.ExportPackageError):
        cohort_materializer.materialize_cohort(
            data_path=root,
            feature_concepts=(),
            static_concepts=("age",),
            outcome_concepts=("death",),
        )
    assert calls == []


def test_materializer_rejects_requested_database_mismatch(tmp_path: Path) -> None:
    root = _analysis_export(tmp_path / "eicu", database="eicu")
    with pytest.raises(intake.ExportPackageError) as exc_info:
        cohort_materializer.materialize_cohort(
            data_path=root,
            database="miiv",
            feature_concepts=("lact",),
            static_concepts=("age",),
            outcome_concepts=("death",),
        )
    assert exc_info.value.code == "export_package_database_mismatch"


def test_materializer_fail_closes_on_unprojected_database_native_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "eicu_raw_time"
    root.mkdir()
    pd.DataFrame({"patientunitstayid": [1, 2], "age": [50, 60]}).to_csv(
        root / "demographics.csv", index=False
    )
    pd.DataFrame({"patientunitstayid": [1, 2], "death": [0, 1]}).to_csv(
        root / "outcomes.csv", index=False
    )
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 2],
            "observationoffset": [60, 120, 60],
            "lact": [1.0, 4.0, 3.0],
        }
    ).to_csv(root / "labs.csv", index=False)
    entries = [
        {
            "file": "demographics.csv",
            "module": "demographics",
            "concepts": 1,
            "concept_ids": ["age"],
            "rows": 2,
        },
        {
            "file": "outcomes.csv",
            "module": "outcomes",
            "concepts": 1,
            "concept_ids": ["death"],
            "rows": 2,
        },
        {
            "file": "labs.csv",
            "module": "labs",
            "concepts": 1,
            "concept_ids": ["lact"],
            "rows": 3,
        },
    ]
    (root / intake.NATIVE_MANIFEST).write_text(
        json.dumps(
            {
                "database": "eicu",
                "format": "csv",
                "concept_selection": {
                    "mode": "explicit",
                    "modules": {
                        "demographics": ["age"],
                        "outcomes": ["death"],
                        "labs": ["lact"],
                    },
                },
                "files": entries,
                "feature_definitions": {"included": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(intake.ExportPackageError) as exc_info:
        cohort_materializer.materialize_cohort(
            data_path=root,
            database="eicu",
            feature_concepts=("lact",),
            static_concepts=("age",),
            outcome_concepts=("death",),
        )

    assert exc_info.value.code == "export_time_projection_required"


def test_schema_inspection_uses_snapshot_before_time_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _analysis_export(
        tmp_path / "schema_snapshot", export_format="csv", database="eicu"
    )
    for path in root.glob("*.csv"):
        frame = pd.read_csv(path).rename(
            columns={"stay_id": "patientunitstayid", "charttime": "observationoffset"}
        )
        frame.to_csv(path, index=False)
    labs_path = root / "labs.csv"
    original_bytes = labs_path.read_bytes()
    forged_bytes = original_bytes.replace(b"observationoffset", b"hidden_time_column")
    original_stat = labs_path.stat()
    original_reader = intake.csv.reader
    reader_calls = 0

    def mutate_live_schema_source(handle, *args, **kwargs):
        nonlocal reader_calls
        reader_calls += 1
        reader = original_reader(handle, *args, **kwargs)
        if reader_calls != 3:
            return reader

        def rows():
            labs_path.write_bytes(forged_bytes)
            try:
                yield from reader
            finally:
                labs_path.write_bytes(original_bytes)
                os.utime(
                    labs_path,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )

        return rows()

    monkeypatch.setattr(intake.csv, "reader", mutate_live_schema_source)
    with pytest.raises(intake.ExportPackageError) as exc_info:
        cohort_materializer.materialize_cohort(
            data_path=root,
            database="eicu",
            feature_concepts=("lact",),
            static_concepts=("age",),
            outcome_concepts=("death",),
        )

    assert exc_info.value.code == "export_time_projection_required"


def test_materializer_reuses_one_content_bound_package_and_records_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _analysis_export(tmp_path / "snapshot")
    original = cohort_materializer.read_exported_concept
    seen_packages: list[int] = []

    def recording_read(source, concept, **kwargs):
        assert isinstance(source, intake.ExportPackage)
        seen_packages.append(id(source))
        return original(source, concept, **kwargs)

    monkeypatch.setattr(cohort_materializer, "read_exported_concept", recording_read)
    _, provenance = cohort_materializer.materialize_cohort(
        data_path=root,
        feature_concepts=("lact",),
        static_concepts=("age",),
        outcome_concepts=("death",),
    )

    assert len(seen_packages) == 3
    assert len(set(seen_packages)) == 1
    authority = provenance["export_authority"]
    assert authority["manifest_sha256"]
    assert authority["authority_sha256"]
    assert all(item["sha256"] for item in authority["files"])


def test_materializer_rejects_package_change_between_concept_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _analysis_export(tmp_path / "changing")
    original = cohort_materializer.read_exported_concept
    calls = 0

    def mutate_after_first_read(source, concept, **kwargs):
        nonlocal calls
        frame = original(source, concept, **kwargs)
        calls += 1
        if calls == 1:
            pd.DataFrame({"stay_id": [1, 2], "death": [1, 1]}).to_parquet(
                root / "outcomes.parquet", index=False
            )
        return frame

    monkeypatch.setattr(
        cohort_materializer, "read_exported_concept", mutate_after_first_read
    )
    with pytest.raises(intake.ExportPackageError) as exc_info:
        cohort_materializer.materialize_cohort(
            data_path=root,
            feature_concepts=("lact",),
            static_concepts=("age",),
            outcome_concepts=("death",),
        )
    assert exc_info.value.code == "manifest_file_mutated"


def test_catalog_uses_manifest_inventory_and_ignores_stale_files(
    tmp_path: Path,
) -> None:
    root = _analysis_export(tmp_path / "export")
    pd.DataFrame({"stay_id": [1], "forged": [99]}).to_parquet(
        root / "stale.parquet", index=False
    )

    catalog = build_available_catalog(root)
    assert set(catalog.ids()) == {"age", "death", "lact"}
    assert "forged" not in catalog.ids()


def test_replication_discovery_accepts_native_and_legacy_packages(
    tmp_path: Path,
) -> None:
    native = _analysis_export(tmp_path / "exports" / "miiv_native", database="miiv")
    legacy = tmp_path / "exports" / "eicu_legacy"
    legacy.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "lact": [2.0]}).to_parquet(
        legacy / "labs.parquet", index=False
    )
    (legacy / intake.LEGACY_MANIFEST).write_text(
        json.dumps(
            {
                "database": "eicu",
                "export_format": "parquet",
                "modules": [{"file": "labs.parquet", "module": "labs", "rows": 1}],
                "exported_files": ["labs.parquet"],
            }
        ),
        encoding="utf-8",
    )

    found = discover_easyicu_exports(
        [tmp_path / "exports"], required_concepts=("lact",)
    )
    assert found == {"eicu": legacy, "miiv": native}


def test_case_builder_compatibility_exports_are_the_shared_intake_functions() -> None:
    import easyicu.research_agent as research_agent
    from easyicu.research_agent import easyicu_case_builder

    assert easyicu_case_builder.index_export_package is intake.index_export_package
    assert easyicu_case_builder.read_exported_concept is intake.read_exported_concept
    assert (
        easyicu_case_builder.resolve_exported_concept is intake.resolve_exported_concept
    )
    assert research_agent.index_export_package is intake.index_export_package
    assert research_agent.read_exported_concept is intake.read_exported_concept
