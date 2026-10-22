"""Focused eICU episode-window regression contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver
from easyicu.concept.schema import ConceptSource
from easyicu.config import DataSourceConfig


def _eicu_patient_source() -> SimpleNamespace:
    """One 2-day eICU stay (discharge 2880 min) plus fallback for stay 99."""

    return SimpleNamespace(
        config=SimpleNamespace(name="eicu"),
        load_table=lambda *_args, **_kwargs: SimpleNamespace(
            data=pd.DataFrame(
                {
                    "patientunitstayid": [10],
                    "unitdischargeoffset": [2880],
                }
            )
        ),
    )


def test_eicu_offsets_outside_the_icu_episode_are_quarantined():
    """Corrupt eICU offsets must not enter producer staging."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    if hasattr(resolver, "_eicu_patient_cache"):
        delattr(resolver, "_eicu_patient_cache")
    frame = pd.DataFrame(
        {
            "patientunitstayid": [10, 10, 10, 10, 10, 10, 99, 99],
            "charttime": [
                -52_578_464.0,
                -25.0 * 60,
                -24.0 * 60,
                0.0,
                72.0 * 60,
                73.0 * 60,
                100.0 * 60,
                9_000.0 * 60,
            ],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    out = resolver._align_time_to_admission(
        frame, _eicu_patient_source(), ["patientunitstayid"], "charttime"
    )

    # Discharge 2880 min -> upper 72 h; unknown stay 99 keeps 366-day fallback.
    assert out["charttime"].tolist() == pytest.approx([-24.0, 0.0, 72.0, 100.0])
    assert out["value"].tolist() == [3.0, 4.0, 5.0, 7.0]


def test_eicu_kdigo_history_can_use_168h_without_widening_default_window():
    """The phenotype window keeps seven-day history; generic concepts do not."""

    resolver = ConceptResolver.__new__(ConceptResolver)
    if hasattr(resolver, "_eicu_patient_cache"):
        delattr(resolver, "_eicu_patient_cache")
    source = _eicu_patient_source()
    frame = pd.DataFrame(
        {
            "patientunitstayid": [10, 10, 10, 10],
            "charttime": [-169.0 * 60, -168.0 * 60, -120.0 * 60, 0.0],
            "crea": [0.8, 0.9, 1.0, 1.6],
        }
    )

    generic = resolver._align_time_to_admission(
        frame.copy(), source, ["patientunitstayid"], "charttime"
    )
    if hasattr(resolver, "_eicu_patient_cache"):
        delattr(resolver, "_eicu_patient_cache")
    kdigo = resolver._align_time_to_admission(
        frame.copy(),
        source,
        ["patientunitstayid"],
        "charttime",
        pre_admission_hours=168,
    )

    assert generic["charttime"].tolist() == pytest.approx([0.0])
    assert kdigo["charttime"].tolist() == pytest.approx([-168.0, -120.0, 0.0])


def test_eicu_los_is_unit_offset_minutes_in_days_for_each_unit_stay():
    """Long source offsets stay visible; hospital and patient spans are separate."""
    from easyicu.webserver import cohort_review, dataio

    root = Path(__file__).resolve().parents[2]
    configs = json.loads((root / "src/easyicu/data/data-sources.json").read_text())
    config = DataSourceConfig(
        **next(item for item in configs if item["name"] == "eicu")
    )
    patient = pd.DataFrame(
        {
            "patientunitstayid": [11, 12, 13],
            "patienthealthsystemstayid": [100, 100, 200],
            "hospitaladmitoffset": [-10080, -11520, -1440],
            "hospitaldischargeoffset": [2880, 1440, 730656],
            "unitdischargeoffset": [1440, 2880, 729216],
        }
    )
    source = SimpleNamespace(
        config=config,
        load_table=lambda *_args, **_kwargs: SimpleNamespace(data=patient.copy()),
    )
    result = (
        ConceptResolver.__new__(ConceptResolver)
        ._load_fun_item_los(
            "los_icu",
            ConceptSource(callback="los_callback", params={"win_type": "icustay"}),
            source,
            patient_ids=None,
            target_type="id_tbl",
        )
        .data.rename(columns={"patientunitstayid": "stay_id"})
    )
    assert result.stay_id.tolist() == [11, 12, 13]
    assert result.los_icu.tolist() == pytest.approx([1.0, 2.0, 506.4])
    # Even repeated outcome rows are reduced per unit stay, never summed.
    by_stay = dataio._stay_numeric(pd.concat([result, result]), "los_icu", "median")
    summary = cohort_review._numeric_summary(by_stay.values())
    assert summary["count"] == 3
    assert summary["median"] == 2.0
    assert summary["max"] == pytest.approx(506.4)
