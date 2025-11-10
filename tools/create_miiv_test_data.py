#!/usr/bin/env python3
"""Generate a 100-patient MIMIC-IV test dataset in Parquet format.

This script mirrors the logic of ``tools/archived/create_minimal_eicu.py`` but
operates on the PhysioNet MIMIC-IV (v3.1) production files. It selects a cohort
of rich ICU stays (antibiotics usage, microbiology sampling, adequate LOS) and
materialises all relevant tables into ``pyricu/test_data_miiv_100`` for local
regression tests.

Usage
-----
Run directly from the repository root:

```
python tools/create_miiv_test_data.py
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# allow importing pyricu utilities without installation
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyricu.fst_reader import read_fst  # noqa: E402  (lazy import after sys.path tweak)

NUM_TEST_PATIENTS = 3
SOURCE_PATH = Path("/home/1_publicData/icu_databases/mimiciv/3.1")
TARGET_PATH = PROJECT_ROOT / "test_data_miiv_100"


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist *df* as Parquet, creating parent directories on demand."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _maybe_concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_test_patients(num_patients: int = NUM_TEST_PATIENTS) -> Tuple[List[int], List[int]]:
    """Pick ICU stays with solid antimicrobial and microbiology coverage."""
    print("== 阶段 1: 筛选 ICU 住院记录 ==")
    icustays = read_fst(SOURCE_PATH / "icustays.fst")
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    eligible = icustays[icustays["los_hours"] > 24].copy()

    print("== 阶段 2: 抗生素使用计数 ==")
    abx_itemids = {
        225798,
        225837,
        225838,
        225840,
        225842,
        225850,
        225857,
        225860,
        225910,
        225913,
        225929,
        225931,
        225936,
        226791,
        227210,
        227214,
    }
    inputevents_path = SOURCE_PATH / "inputevents.fst"
    if inputevents_path.exists():
        inputevents = read_fst(inputevents_path)
        abx_counts = (
            inputevents[inputevents["itemid"].isin(abx_itemids)]
            .groupby("stay_id")
            .size()
            .rename("abx_count")
            .reset_index()
        )
        eligible = eligible.merge(abx_counts, on="stay_id", how="left")
        eligible["abx_count"] = eligible["abx_count"].fillna(0)
    else:
        eligible["abx_count"] = 0
        print("  ⚠️ 未找到 inputevents.fst，跳过抗生素统计")

    print("== 阶段 3: 微生物采样计数 ==")
    micro_path = SOURCE_PATH / "microbiologyevents.fst"
    if micro_path.exists():
        micro = read_fst(micro_path)
        micro_counts = micro.groupby("subject_id").size().rename("micro_count").reset_index()
        eligible = eligible.merge(micro_counts, on="subject_id", how="left")
        eligible["micro_count"] = eligible["micro_count"].fillna(0)
    else:
        eligible["micro_count"] = 0
        print("  ⚠️ 未找到 microbiologyevents.fst，跳过采样统计")

    print("== 阶段 4: 综合评分并选择 ==")
    eligible["score"] = 0
    eligible.loc[eligible["abx_count"] > 0, "score"] += 10
    eligible.loc[eligible["micro_count"] > 0, "score"] += 10
    eligible.loc[(eligible["los_hours"] >= 48) & (eligible["los_hours"] <= 168), "score"] += 5
    eligible = eligible.sort_values(["score", "los_hours"], ascending=[False, False])

    selected = eligible.head(num_patients)
    print(f"  ✅ 选中 {len(selected)} 个 stay_ids")
    return selected["stay_id"].tolist(), selected["subject_id"].tolist()


def _filter_directory(path: Path, id_col: str, ids: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for fst_file in sorted(path.glob("*.fst")):
        df = read_fst(fst_file)
        if id_col not in df.columns:
            continue
        subset = df[df[id_col].isin(ids)]
        if not subset.empty:
            frames.append(subset)
    return _maybe_concat(frames)


def extract_patient_data(stay_ids: List[int], subject_ids: List[int]) -> None:
    TARGET_PATH.mkdir(parents=True, exist_ok=True)

    print("== 保存 icustays / patients ==")
    icustays = read_fst(SOURCE_PATH / "icustays.fst")
    patients = read_fst(SOURCE_PATH / "patients.fst")
    save_parquet(icustays[icustays["stay_id"].isin(stay_ids)], TARGET_PATH / "icustays.parquet")
    save_parquet(patients[patients["subject_id"].isin(subject_ids)], TARGET_PATH / "patients.parquet")

    print("== 保存 labevents ==")
    labevents_dir = SOURCE_PATH / "labevents"
    labevents = _filter_directory(labevents_dir, "subject_id", subject_ids)
    if not labevents.empty:
        save_parquet(labevents, TARGET_PATH / "labevents.parquet")

    print("== 保存 chartevents ==")
    chartevents_dir = SOURCE_PATH / "chartevents"
    chartevents = _filter_directory(chartevents_dir, "stay_id", stay_ids)
    if not chartevents.empty:
        save_parquet(chartevents, TARGET_PATH / "chartevents.parquet")

    single_tables = [
        ("inputevents.fst", "stay_id", stay_ids),
        ("outputevents.fst", "stay_id", stay_ids),
        ("procedureevents.fst", "stay_id", stay_ids),
        ("prescriptions.fst", "subject_id", subject_ids),
        ("microbiologyevents.fst", "subject_id", subject_ids),
    ]
    for filename, id_col, id_values in single_tables:
        src = SOURCE_PATH / filename
        if not src.exists():
            print(f"  ⚠️ 缺失 {filename}, 跳过")
            continue
        df = read_fst(src)
        if id_col in df.columns:
            subset = df[df[id_col].isin(id_values)]
        else:
            print(f"  ⚠️ {filename} 中不存在列 {id_col}, 跳过过滤")
            subset = df
        if not subset.empty:
            save_parquet(subset, TARGET_PATH / f"{Path(filename).stem}.parquet")

    print("== 复制字典表 ==")
    for dictionary_file in ["d_items.fst", "d_labitems.fst"]:
        src = SOURCE_PATH / dictionary_file
        if src.exists():
            save_parquet(read_fst(src), TARGET_PATH / f"{Path(dictionary_file).stem}.parquet")
        else:
            print(f"  ⚠️ 缺失 {dictionary_file}")

    print(f"\n✅ 完成！输出目录: {TARGET_PATH}")


if __name__ == "__main__":
    stay_ids, subject_ids = select_test_patients(NUM_TEST_PATIENTS)
    extract_patient_data(stay_ids, subject_ids)
