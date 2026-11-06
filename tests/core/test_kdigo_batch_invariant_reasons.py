"""Renal evidence labels must not depend on which stays share a batch.

A batch in which no stay has rows for a component used to label every stay
``source_absent`` (and the eICU source-native status ``not_evaluable``), while a
whole-cohort run labels the same stays ``patient_data_absent`` (and copied one
stay's native status onto the others by forward/backward fill).  The source
receipt (:class:`KDIGOComponentSources`) now decides these labels from the
installation, and the native stamp is assigned, never filled across stays.

Synthetic data only.  Stay 1 ("A") has every component source; stay 2 ("B")
has creatinine only; stay 3 ("C") has urine but no weight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import callbacks
from easyicu.resources import load_data_sources, load_dictionary
from easyicu.scores.kdigo_aki import (
    KDIGOComponentSchemaError,
    KDIGOComponentSources,
    kdigo_component_sources,
    kdigo_stages,
)
from easyicu.table import ICUTable

DATABASES = ("mimic", "eicu", "miiv", "hirid", "aumc")
HOURS = np.arange(0.0, 31.0)


class _Source:
    """A bound data source whose resolver reports the named tables missing."""

    def __init__(self, database: str, missing: frozenset[str] = frozenset()):
        self.config = load_data_sources().get(database)
        self.base_path = None
        self._missing = missing

    def resolve_loader_from_disk(self, table: str):
        return None if table in self._missing else Path(table)


def _tables(stays: list[int], database: str) -> dict[str, ICUTable]:
    crea = pd.DataFrame(
        {
            "stay_id": np.repeat(stays, 4),
            "charttime": np.tile([0.0, 10.0, 20.0, 30.0], len(stays)),
            "crea": np.tile([1.0, 1.2, 2.5, 3.1], len(stays)),
        }
    )
    with_urine = [s for s in stays if s in (1, 3)]
    urine = pd.DataFrame(
        {
            "stay_id": np.repeat(with_urine, len(HOURS)),
            "charttime": np.tile(HOURS, len(with_urine)),
            "urine": 10.0,
        }
    )
    a = [s for s in stays if s == 1]
    weight = pd.DataFrame({"stay_id": a, "weight": [70.0] * len(a)})
    rrt = pd.DataFrame({"stay_id": a, "charttime": [20.0] * len(a), "rrt": [True] * len(a)})
    tables = {
        "kdigo_creatinine_input": ICUTable(
            crea, id_columns=["stay_id"], index_column="charttime", value_column="crea"
        ),
        "kdigo_urine_input": ICUTable(
            urine, id_columns=["stay_id"], index_column="charttime", value_column="urine"
        ),
        "weight": ICUTable(
            weight, id_columns=["stay_id"], index_column=None, value_column="weight"
        ),
        "acute_rrt_input": ICUTable(
            rrt, id_columns=["stay_id"], index_column="charttime", value_column="rrt"
        ),
    }
    if database == "miiv":
        tables["crrt_mode_input"] = ICUTable(
            rrt.copy(), id_columns=["stay_id"], index_column="charttime", value_column="rrt"
        )
    return tables


def _bundle(stays: list[int], database: str, source: _Source) -> pd.DataFrame:
    ctx = callbacks.ConceptCallbackContext(
        "aki_stage_reference", None, pd.Timedelta(hours=1), None, source, stays
    )
    return callbacks._callback_kdigo_aki(_tables(stays, database), ctx).data


def _rows(frame: pd.DataFrame, stay: int) -> pd.DataFrame:
    return frame.loc[frame["stay_id"] == stay].sort_values("charttime").reset_index(drop=True)


def _differing_cells(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    assert len(left) == len(right)
    diffs = {}
    for column in sorted(set(left.columns) | set(right.columns)):
        x = left[column] if column in left else pd.Series([None] * len(left))
        y = right[column] if column in right else pd.Series([None] * len(right))
        xs = x.astype(object).where(x.notna(), None)
        ys = y.astype(object).where(y.notna(), None)
        count = sum(1 for p, q in zip(xs, ys) if p != q)
        if count:
            diffs[column] = count
    return diffs


@pytest.mark.parametrize("database", DATABASES)
@pytest.mark.parametrize("stay", [2, 3])
def test_a_stay_is_labelled_alike_alone_and_beside_a_stay_with_every_source(database, stay):
    source = _Source(database)
    whole = _rows(_bundle([1, stay], database, source), stay)
    alone = _rows(_bundle([stay], database, source), stay)
    assert len(whole) > 0
    assert _differing_cells(whole, alone) == {}


@pytest.mark.parametrize("database", ("mimic", "hirid"))
def test_a_mapped_rrt_source_without_rows_is_patient_data_absent(database):
    # Neither database's RRT search is complete, so the reason names the gap.
    source = _Source(database)
    for stays in ([1, 2], [2]):
        b = _rows(_bundle(stays, database, source), 2)
        assert set(b["rrt_evidence_reason"]) == {"patient_data_absent"}
        assert set(b["creatinine_evidence_reason"]) <= {
            "insufficient_baseline", "criterion_negative", "criterion_positive",
        }


@pytest.mark.parametrize("stays", ([1, 2], [2]))
def test_stay_without_sources_inherits_nothing_from_its_neighbour(stays):
    # eICU: A's urine-derived native ascertainment and A's RRT/urine values
    # must not reach B, and B's native status is the profile's own stamp.
    b = _rows(_bundle(stays, "eicu", _Source("eicu")), 2)
    assert set(b["aki_source_native_status"]) == {"components_only"}
    assert b["aki_source_native_ascertainment"].isna().all()
    for column in ("uo_rt_6hr", "uo_rt_12hr", "uo_rt_24hr", "aki_stage_uo_reference"):
        if column in b:
            assert b[column].isna().all(), column
    assert b["aki_stage_rrt_reference"].fillna(0).eq(0).all()
    assert set(b["rrt_evidence_status"]) <= {"negative", "indeterminate"}


def test_urine_without_weight_keeps_its_window_coverage_in_any_batch():
    source = _Source("mimic")
    whole = _rows(_bundle([1, 3], "mimic", source), 3)
    alone = _rows(_bundle([3], "mimic", source), 3)
    assert whole["uo_rt_6hr_covered_h"].notna().any()
    assert whole["uo_rt_6hr"].isna().all()
    assert _differing_cells(whole, alone) == {}


def _component_tables(database: str, concept: str) -> set[str]:
    sources = load_dictionary().get(concept).for_data_source(load_data_sources().get(database))
    return {str(s.table) for s in sources if getattr(s, "table", None)}


def test_an_absent_source_still_reads_source_absent():
    database = next(
        db for db in DATABASES
        if not (_component_tables(db, "rrt") | _component_tables(db, "urine"))
        & _component_tables(db, "crea")
    )
    missing = frozenset(_component_tables(database, "rrt") | _component_tables(database, "urine"))
    source = _Source(database, missing)
    receipt = kdigo_component_sources(source)
    assert receipt == KDIGOComponentSources(creatinine=True, urine=False, rrt=False)
    stages = kdigo_stages(
        crea_df=_tables([2], database)["kdigo_creatinine_input"].data,
        urine_df=pd.DataFrame(columns=["stay_id", "charttime", "urine"]),
        rrt_df=pd.DataFrame(columns=["stay_id", "charttime", "rrt"]),
        time_unit="hours",
        component_sources=receipt,
    )
    assert set(stages["rrt_ascertainment_reason"]) == {"source_absent"}
    assert set(stages["uo_assessment_reason"]) == {"source_absent"}


def test_rows_for_a_source_declared_absent_fail_loudly():
    tables = _tables([1], "mimic")
    with pytest.raises(KDIGOComponentSchemaError) as raised:
        kdigo_stages(
            crea_df=tables["kdigo_creatinine_input"].data,
            rrt_df=tables["acute_rrt_input"].data,
            time_unit="hours",
            component_sources=KDIGOComponentSources(
                creatinine=True, urine=True, rrt=False
            ),
        )
    assert raised.value.reason_code == "kdigo_rrt_rows_without_declared_source"


def test_without_a_receipt_the_whole_cohort_labels_are_unchanged():
    # Direct callers that pass no receipt keep the historical per-call labels.
    stages = kdigo_stages(
        crea_df=_tables([2], "mimic")["kdigo_creatinine_input"].data,
        time_unit="hours",
    )
    assert set(stages["rrt_ascertainment_reason"]) == {"source_absent"}


def test_legacy_gcs_locf_never_carries_one_sicdb_case_into_the_next():
    # Stay 2 has no component observations; on SICdb's CaseID key the legacy
    # helper used to merge and forward-fill across cases.
    from easyicu import callbacks as legacy

    def component(name, values, ids):
        return pd.DataFrame({"CaseID": ids, "charttime": [0.0, 1.0, 1.0], name: values})

    ids = [1, 1, 2]
    gcs = legacy.gcs(
        component("egcs", [4.0, np.nan, np.nan], ids),
        component("vgcs", [5.0, np.nan, np.nan], ids),
        component("mgcs", [6.0, np.nan, np.nan], ids),
        valid_win=pd.Timedelta(hours=6),
        set_na_max=False,
    )
    assert gcs.loc[gcs["CaseID"] == 2, "gcs"].isna().all()
    assert gcs.loc[gcs["CaseID"] == 1, "gcs"].notna().any()
