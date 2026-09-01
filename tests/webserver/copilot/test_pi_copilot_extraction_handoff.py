"""Contracts for compiling Copilot study setup into Data Extraction input."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.webserver.pi_copilot.extraction_handoff import (
    compile_registered_export_handoff,
    submit_study_extraction,
)
from easyicu.webserver.routes.jobs import _study_source_matches


def _write_export(
    export_path: Path,
    raw_path: Path,
    *,
    export_format: str,
    cohort: dict,
    modules: list[str],
) -> None:
    export_path.mkdir()
    (export_path / "_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu_native_export_v2",
                "database": "miiv",
                "data_path": str(raw_path),
                "format": export_format,
                "cohort_contract": cohort,
                "files": [
                    {"file": f"{module}.{export_format}", "module": module}
                    for module in modules
                ],
            }
        ),
        encoding="utf-8",
    )


def _study(export_path: Path) -> dict:
    return {
        "id": "study-a41",
        "revision": 4,
        "title": "Adult A41 cohort",
        "data_source": {
            "path": str(export_path),
            "database": "miiv",
        },
        "cohort": {
            "preset": "icd",
            "age_min": 18,
            "age_max": 100,
            "exclude_readmissions": False,
            "include_diagnoses": ["A41"],
        },
        "modules": ["demographics", "blood_gas", "outcome"],
        "time_window": {"observation_hours": 24, "anchor": "ICU admission"},
        "export_format": "CSV",
    }


def test_handoff_rejects_registered_export_with_different_extraction_contract(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    export_path = tmp_path / "export"
    _write_export(
        export_path,
        raw_path,
        export_format="parquet",
        cohort={"preset": "all_icu", "observation_window_hours": 720},
        modules=["demographics", "blood_gas", "outcome", "vitals"],
    )

    handoff = compile_registered_export_handoff(
        _study(export_path),
        {
            "id": "src-demo",
            "path": str(export_path),
            "database": "miiv",
            "ok": True,
        },
    )

    assert handoff.reusable is False
    assert handoff.mismatch_codes == (
        "registered_export_cohort_mismatch",
        "registered_export_format_mismatch",
    )
    assert handoff.source_data_path == str(raw_path.resolve())
    assert handoff.cohort["preset"] == "icd"
    assert handoff.cohort["icd_include"] == ["A41"]
    assert handoff.cohort["observation_window_hours"] == 24
    assert handoff.export_format == "csv"
    encoded = json.dumps(handoff.public_receipt())
    assert str(raw_path) not in encoded
    assert str(export_path) not in encoded


def test_handoff_reuses_only_contract_matching_registered_export(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    export_path = tmp_path / "export"
    _write_export(
        export_path,
        raw_path,
        export_format="csv",
        cohort={
            "preset": "icd",
            "age_min": 18,
            "age_max": 100,
            "exclude_readmissions": False,
            "icd_enabled": True,
            "icd_include": ["A41"],
            "observation_window_hours": 24,
        },
        modules=["demographics", "blood_gas", "outcome"],
    )

    handoff = compile_registered_export_handoff(
        _study(export_path),
        {
            "id": "src-demo",
            "path": str(export_path),
            "database": "miiv",
            "ok": True,
        },
    )

    assert handoff.reusable is True
    assert handoff.mismatch_codes == ()
    assert handoff.public_receipt()["reusable"] is True

    def unexpected_submit(_body: dict) -> dict:
        raise AssertionError("matching exports must not start a duplicate job")

    transaction = submit_study_extraction(
        study=_study(export_path),
        registered_source={
            "id": "src-demo",
            "path": str(export_path),
            "database": "miiv",
            "ok": True,
        },
        submit=unexpected_submit,
    )
    assert transaction.reused is True
    assert transaction.source_id == "src-demo"
    assert transaction.submitted is None


def test_submission_transaction_owns_the_bound_job_contract() -> None:
    study = {
        "id": "study-local",
        "revision": 7,
        "title": "Local extraction",
        "data_source": {"path": "/private/raw", "database": "miiv"},
        "modules": ["demographics", "vitals"],
        "cohort": {"preset": "adult_first"},
        "time_window": {"observation_hours": 24},
        "export_format": "parquet",
    }
    submitted_bodies: list[dict] = []

    def submit(body: dict) -> dict:
        submitted_bodies.append(dict(body))
        return {"job_id": "extract-7", "kind": "extract", "status": "running"}

    transaction = submit_study_extraction(
        study=study,
        registered_source=None,
        submit=submit,
    )

    assert transaction.reused is False
    assert transaction.submitted == {
        "job_id": "extract-7",
        "kind": "extract",
        "status": "running",
    }
    assert submitted_bodies == [
        {
            "path": "/private/raw",
            "registered_export_path": None,
            "database": "miiv",
            "modules": ["demographics", "vitals"],
            "format": "parquet",
            "merge": True,
            "cohort": {
                "preset": "adult_first",
                "observation_window_hours": 24,
            },
            "max_patients": None,
            "include_feature_definitions": True,
            "label": "Local extraction",
            "study_context_id": "study-local",
            "study_context_revision": 7,
        }
    ]


def test_job_source_gate_accepts_only_manifest_bound_raw_source(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    export_path = tmp_path / "export"
    _write_export(
        export_path,
        raw_path,
        export_format="parquet",
        cohort={"preset": "all_icu"},
        modules=["demographics"],
    )
    bound_source = {"path": str(export_path), "database": "miiv"}

    assert _study_source_matches(
        requested_path=str(raw_path),
        requested_database="miiv",
        bound_source=bound_source,
        registered_export_path=str(export_path),
    )
    assert not _study_source_matches(
        requested_path=str(raw_path),
        requested_database="miiv",
        bound_source=bound_source,
        registered_export_path=None,
    )
    other_raw = tmp_path / "other-raw"
    other_raw.mkdir()
    assert not _study_source_matches(
        requested_path=str(other_raw),
        requested_database="miiv",
        bound_source=bound_source,
        registered_export_path=str(export_path),
    )
