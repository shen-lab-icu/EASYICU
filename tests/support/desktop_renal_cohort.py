"""Synthetic renal source spanning desktop extraction batches."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def desktop_renal_cohort(tmp_path, monkeypatch):
    import psutil

    from easyicu.runtime import memory_manager
    from easyicu.webserver.hosted import deployment_kind
    from tests.support.renal_export import write_renal_source

    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setenv("EASYICU_MP_START_METHOD", "fork")
    for key in (
        "EASYICU_HOSTED", "EASYICU_ONESHOT_BUDGET_MB", "EASYICU_AUTO_CHUNK_SIZE",
        "EASYICU_DISABLE_AUTO_CHUNK", "EASYICU_FORCE_INPROCESS_BATCH",
        "EASYICU_FORCE_SUBPROCESS_BATCH",
    ):
        monkeypatch.delenv(key, raising=False)
    assert deployment_kind() == "desktop"

    ram = 16 * 1024**3
    memory = psutil.virtual_memory()._replace(total=ram, available=ram)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(memory_manager, "_read_cgroup_v2_memory_mb", lambda root: (None, None))

    # Put synthetic observations at both ends of a large logical cohort.
    source = tmp_path / "source"
    write_renal_source(source, "eicu", stays=24, hours=6, urine_rows_per_hour=1)
    id_col = "patientunitstayid"
    cohort_size = 135_003
    for path in source.glob("*.parquet"):
        frame = pd.read_parquet(path)
        if id_col in frame and not frame.empty:
            ids = frame[id_col]
            frame[id_col] = ids.where(ids <= 12, ids + cohort_size - 24)
            if path.name == "patient.parquet":
                # Every fixture stay is the first in its own hospitalization.
                frame["unitvisitnumber"] = 1
            frame.to_parquet(path, index=False)
    return source, {id_col: list(range(1, cohort_size + 1))}
