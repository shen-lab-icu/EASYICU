from __future__ import annotations

import pandas as pd

from easyicu.io.export import write_psv


def test_write_psv_converts_datetime_index_to_patient_relative_hours(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "time": pd.to_datetime(
                [
                    "2020-01-01 03:00",
                    "2020-01-01 05:00",
                    "2020-02-01 10:00",
                    "2020-02-01 11:30",
                ]
            ),
            "heart_rate": [80, 85, 90, 95],
        }
    )

    write_psv(frame, tmp_path, "patient_id", "time")

    first = pd.read_csv(tmp_path / "p1.psv", sep="|")
    second = pd.read_csv(tmp_path / "p2.psv", sep="|")
    assert first["time"].tolist() == [0.0, 2.0]
    assert second["time"].tolist() == [0.0, 1.5]
