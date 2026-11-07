"""Synthetic eICU source and overlap helpers for the stay-table call and resolver-pool tests."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.concept import stay_tables
from tests.support.urine_raw_source import _source

SYNTHETIC_EICU_IDS = np.arange(1, 4)


def synthetic_eicu(root: Path):
    return _source(
        root, "eicu",
        patient=pd.DataFrame({
            "patientunitstayid": SYNTHETIC_EICU_IDS, "patienthealthsystemstayid": SYNTHETIC_EICU_IDS,
            "unitdischargeoffset": 6000, "hospitaladmitoffset": 0, "admissionweight": 70.0,
        }),
        vitalperiodic=pd.DataFrame({
            "vitalperiodicid": SYNTHETIC_EICU_IDS, "patientunitstayid": SYNTHETIC_EICU_IDS, "observationoffset": 150,
            "pasystolic": 25.0, "padiastolic": 15.0, "pamean": [30.0, 31.0, 32.0],
            "heartrate": 80.0,
        }),
    )


def gate_first_read(monkeypatch):
    """Hold each thread's first stay-table read until ``release`` is set."""
    inside: list = []
    release = threading.Event()
    seen: set = set()
    lock = threading.Lock()
    real = stay_tables.stay_table_frame

    def gated(data_source, table, stays, read, **kwargs):
        ident = threading.get_ident()
        with lock:
            first = ident not in seen
            seen.add(ident)
            if first:
                inside.append(ident)
        if first:
            release.wait(timeout=60)
        return real(data_source, table, stays, read, **kwargs)

    monkeypatch.setattr(stay_tables, "stay_table_frame", gated)
    return inside, release


def wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def run_threads(target, count):
    results: dict = {}

    def call(index):
        try:
            results[index] = target(index)
        except BaseException as error:  # noqa: BLE001 - asserted by the caller
            results[index] = error

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(count)]
    for thread in threads:
        thread.start()
    return threads, results
