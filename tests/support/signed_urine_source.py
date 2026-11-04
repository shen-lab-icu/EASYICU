"""MIMIC outputevents fixture with signed and duplicated urine rows."""

import numpy as np
import pandas as pd

from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource


def signed_urine_fixture_source(tmp_path, database, string_timestamps=False):
    id_col = "stay_id" if database == "miiv" else "icustay_id"
    admitted = pd.Timestamp("2180-01-01 00:15:00")
    rows = [
        (1, 5, 226559, 100.), (1, 15, 227488, 500.), (1, 25, 227489, 550.),
        (1, 65, 227488, 9000.), (1, 65, 227489, 9200.),
        (1, 125, 227488, 500.), (1, 125, 227489, 500.),
        (1, 185, 227488, 700.), (1, 185, 227489, 500.),
        (1, 245, 226559, 3500.), (1, 245, 226560, 3500.),
        (1, 305, 226559, -15.), (1, 365, 226559, np.nan),
        (1, 490, 226559, 70.), (1, 520, 226559, 80.),
        (2, 5, 226559, 40.),
    ]
    frame = pd.DataFrame(rows, columns=[id_col, "offset", "itemid", "value"])
    frame["charttime"] = admitted + pd.to_timedelta(frame.pop("offset"), unit="m")
    frame["valueuom"] = "mL"
    frame["subject_id"] = frame[id_col]
    frame["hadm_id"] = frame[id_col]
    table_dir = tmp_path / "outputevents"
    table_dir.mkdir()
    persisted = frame.copy()
    if string_timestamps:
        persisted["charttime"] = persisted.charttime.astype(str)
    persisted.to_parquet(table_dir / "part-0.parquet", index=False)
    stays = pd.DataFrame({id_col: [1, 2], "subject_id": [1, 2], "hadm_id": [1, 2],
                          "intime": [admitted]*2,
                          "outtime": [admitted+pd.Timedelta(hours=48)]*2})
    stays.to_parquet(tmp_path / "icustays.parquet", index=False)
    source = ICUDataSource(load_src_cfg(database), base_path=tmp_path, enable_cache=False)
    return source, frame, admitted, id_col
