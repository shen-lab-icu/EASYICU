from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.webapp import research_agent as ra_page


def test_module_export_folder_builds_filtered_stay_level_cohort(tmp_path: Path) -> None:
    folder = tmp_path / "mimiciv_export"
    (folder / "sepsis3_sofa2").mkdir(parents=True)
    (folder / "outcome").mkdir()
    (folder / "vitals").mkdir()

    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "sep3_sofa2": [1, 0, 1],
    }).to_parquet(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 2, 3],
        "death": [1, 0, 0],
    }).to_parquet(folder / "outcome" / "death.parquet", index=False)
    pd.DataFrame({
        "stay_id": [1, 1, 2, 3],
        "charttime": pd.to_datetime([
            "2024-01-01 00:00",
            "2024-01-01 01:00",
            "2024-01-01 00:00",
            "2024-01-01 00:00",
        ]),
        "hr": [70, 80, 90, 100],
    }).to_parquet(folder / "vitals" / "hr.parquet", index=False)

    selected = [
        folder / "sepsis3_sofa2" / "sep3_sofa2.parquet",
        folder / "outcome" / "death.parquet",
        folder / "vitals" / "hr.parquet",
    ]
    cohort = ra_page._build_stay_level_from_module_folder(
        folder=folder,
        selected_files=selected,
        id_col="stay_id",
        filter_spec=(folder / "sepsis3_sofa2" / "sep3_sofa2.parquet", "sep3_sofa2", "nonzero / true", ""),
    )

    assert set(cohort["stay_id"]) == {1, 3}
    assert cohort.loc[cohort["stay_id"] == 1, "hr"].iloc[0] == 80
    assert set(["sep3_sofa2", "death", "hr"]) <= set(cohort.columns)


def test_infers_sepsis_filter_defaults_from_question(tmp_path: Path) -> None:
    path = tmp_path / "sep3_sofa2.parquet"
    pd.DataFrame({"stay_id": [1], "sep3_sofa2": [1]}).to_parquet(path, index=False)
    summary = ra_page._parquet_file_summary(path)

    filter_path, filter_col = ra_page._infer_filter_defaults(
        [summary],
        question="Do sepsis patients have higher hospital mortality?",
    )

    assert filter_path == path
    assert filter_col == "sep3_sofa2"
