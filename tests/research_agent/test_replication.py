"""Deterministic cross-export replication helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.intake.export_package import ExportPackageError


def _write_synthetic_easyicu_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "age": [50, 61, 72, 64, 70, 59],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "adm": ["medical"] * 6,
            "bmi": [24, 28, 31, 22, 27, 26],
            "weight": [65, 80, 74, 68, 72, 85],
        }
    ).to_parquet(root / "demographics_adm_age_bmi_height_etc.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "death": [0, 1, 1, 1, 0, 1],
            "los_icu": [2.0, 5.0, 7.0, 4.0, 1.5, 3.0],
            "los_hosp": [5.0, 9.0, 12.0, 7.0, 3.0, 5.0],
        }
    ).to_parquet(root / "outcome_death_los_hosp_los_icu.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "charttime": [4.0, 6.0, 8.0, 10.0],
            "lact": [1.1, 3.0, 5.0, 2.5],
        }
    ).to_parquet(root / "blood_gas_be_cai_lact_methb_etc.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "charttime": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "map": [80, 82, 85, 60, 90, 70],
        }
    ).to_parquet(root / "vitals_dbp_hr_map_resp_etc.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "charttime": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "vaso_ind": [0, 0, 1, 0, 0, 0],
            "norepi_equiv": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        }
    ).to_parquet(
        root / "vasopressors_adh_rate_dobu_dur_dobu_rate_etc.parquet", index=False
    )
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6],
            "charttime": [1.0] * 6,
            "circ_event": [0, 0, 1, 1, 0, 0],
            "circ_failure": [0, 0, 1, 1, 0, 0],
        }
    ).to_parquet(root / "circulatory_circ_event_circ_failure.parquet", index=False)
    exported_files = sorted(path.name for path in root.glob("*.parquet"))
    (root / "easyicu_export_manifest.json").write_text(
        json.dumps(
            {
                "database": "miiv",
                "export_format": "parquet",
                "exported_files": exported_files,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_run_lactate_map_vaso_replication_package(ra, tmp_path: Path):
    export = _write_synthetic_easyicu_export(tmp_path / "miiv_export")
    paths = ra.run_lactate_map_vaso_replication(
        {"miiv": export, "eicu": None},
        tmp_path / "replication",
    )

    for path in paths.values():
        assert Path(path).exists()

    summary = pd.read_csv(paths["summary"])
    assert set(summary["database"]) == {"miiv", "eicu"}
    miiv = summary.set_index("database").loc["miiv"]
    assert miiv["status"] == "ok"
    assert int(miiv["n_stays"]) == 6
    assert int(miiv["lactate_measured_n"]) == 4
    assert summary.set_index("database").loc["eicu", "status"] == "pending"

    strata = pd.read_csv(paths["shock_strata"])
    assert set(strata["stratum"]) == {
        "MAP<65",
        "MAP>=65 & Lactate>2 & NoVaso",
        "MAP>=65 & Lactate>2 & Vaso",
        "MAP>=65 & Lactate<=2",
    }
    assert (
        tmp_path / "replication/cohorts/miiv/miiv_lactate_map_vaso_24h.parquet"
    ).exists()

    appendix = Path(paths["appendix"]).read_text(encoding="utf-8")
    assert "EasyICU concept-export" in appendix
    assert "does not use ad hoc SQL" in appendix


def test_discover_easyicu_exports_requires_case_concepts(ra, tmp_path: Path):
    good = _write_synthetic_easyicu_export(tmp_path / "exports" / "miiv_case")
    bad = tmp_path / "exports" / "eicu_incomplete"
    bad.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "map": [70]}).to_parquet(
        bad / "vitals_dbp_hr_map_resp_etc.parquet",
        index=False,
    )
    (bad / "easyicu_export_manifest.json").write_text(
        json.dumps(
            {
                "database": "eicu",
                "export_format": "parquet",
                "exported_files": ["vitals_dbp_hr_map_resp_etc.parquet"],
            }
        ),
        encoding="utf-8",
    )

    found = ra.discover_easyicu_exports([tmp_path / "exports"])
    assert found == {"miiv": good}


def test_replication_builder_fail_closes_on_database_native_time(tmp_path: Path):
    from easyicu.research_agent.easyicu_case_builder import (
        build_lactate_map_vaso_cohort_from_export,
    )

    export = _write_synthetic_easyicu_export(tmp_path / "eicu_raw_time")
    for path in export.glob("*.parquet"):
        frame = pd.read_parquet(path).rename(
            columns={"stay_id": "patientunitstayid", "charttime": "observationoffset"}
        )
        if "observationoffset" in frame.columns:
            frame["observationoffset"] = frame["observationoffset"] * 60
        frame.to_parquet(path, index=False)
    manifest_path = export / "easyicu_export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["database"] = "eicu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportPackageError) as exc_info:
        build_lactate_map_vaso_cohort_from_export(export)

    assert exc_info.value.code == "export_time_projection_required"


def test_explicit_replication_target_cannot_relabel_source_database(ra, tmp_path: Path):
    export = _write_synthetic_easyicu_export(tmp_path / "miiv_export")
    paths = ra.run_lactate_map_vaso_replication(
        {"eicu": export}, tmp_path / "replication"
    )

    summary = pd.read_csv(paths["summary"]).set_index("database")
    assert summary.loc["eicu", "status"] == "error"
    assert "does not match requested replication target" in str(
        summary.loc["eicu", "message"]
    )
