"""Opt-in projections for the frozen L-COMMON thesis inputs."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from ..config import DATABASE_ID_CONFIG
from ..datasource import storage_source
from ..io.identity import require_unique_keys
from ..runtime import duckdb_spill
from ..storage_binding import in_binding_unit
from ..storage_layout import (
    parquet_relation,
    record_table_unavailable,
    refuse_broken_presence,
    resolve_table_files,
    table_decision,
)
from .catalog import THESIS_SUPPORT_MODULES
from .support_exports import select_support_cohort


@dataclass(frozen=True)
class _RawPEEP:
    source: str
    tables: tuple[str, ...]


_PEEP = {
    "aumc": _RawPEEP("aumc_numericitems", ("admissions", "numericitems")),
    "hirid": _RawPEEP("hirid_observations", ("general", "observations")),
    "miiv": _RawPEEP("miiv_chartevents", ("icustays", "chartevents")),
    "sic": _RawPEEP("sic_data_float_h.Val", ("cases", "data_float_h")),
}

# The frozen hourly builder reads native-v2 modules under an external contract.
# These names explain the missing producer in the manifest and are never
# passed to storage resolution.
_NO_RAW_PRODUCER = {
    "imv_recorded_compat": {
        **{
            db: (
                f"{db}/respiratory.parquet:mech_vent",
                f"{db}/outcome.parquet:los_icu",
                "config/hourly_core_v7.json",
            )
            for db in ("aumc", "miiv", "sic")
        },
        "hirid": ("hirid/respiratory.parquet:mech_vent",),
    },
    "rrt_time": {
        **{
            db: (
                f"{db}_hourly_core.parquet:rrt",
                f"{db}/outcome.parquet:los_icu",
                "config/hourly_core_v7.json:dynamic",
            )
            for db in ("aumc", "eicu", "hirid", "sic")
        },
        "miiv": ("miiv_rrt_time.parquet", "raw_timestamp_actual_end.parquet"),
        "mimic": (
            "mimiciii/icustays.parquet",
            "mimiciii/chartevents/*.parquet",
            "mimiciii/procedureevents_mv.parquet",
            "hc_original.rrt",
        ),
    },
}


def thesis_support_manifest(module: str, database: str, fields: Collection[str]) -> dict:
    """Report raw input provenance or the frozen producer's missing inputs."""
    if module not in THESIS_SUPPORT_MODULES:
        raise KeyError(module)
    family = database.removesuffix("_demo")
    missing = _NO_RAW_PRODUCER.get(module, {}).get(family)
    if missing is not None:
        span = module == "imv_recorded_compat" and family == "hirid"
        receipt = {
            "source": None,
            "status": "UNAVAILABLE_IN_SCOPE",
            "reason": "no_raw_producer",
            "missing_inputs": list(missing),
            "charttime_role": "first_recorded_hour" if span else "icu_hour",
            "source_grain": "stay" if span else "stay_hour",
        }
        if family == "mimic":
            receipt["detail"] = "The frozen cv_rrt producer requires raw MIMIC-III tables outside the approved #641 CareVue view."
        return receipt
    raw = _PEEP.get(family) if module == "observed_peep_set" else None
    if raw is None:
        return {"source": None, "status": "no_frozen_reducer"}
    return {
        "source": raw.source,
        "input_tables": list(raw.tables),
        "status": "projected" if "observed_peep_set" in fields else "authority_unavailable",
        "charttime_role": "icu_hour",
        "source_grain": "stay_hour",
    }


def _empty(module: str, stay: str) -> pd.DataFrame:
    fields = THESIS_SUPPORT_MODULES[module]
    return pd.DataFrame({
        stay: pd.Series(dtype="int64"),
        "charttime": pd.Series(dtype="float64"),
        **{
            name: pd.Series(dtype="boolean" if unit == "boolean" else "float64")
            for name, (_, unit) in fields.items()
        },
    })


def support_input_files(source, table, stay, ids):
    """Resolve the exact Parquet input through the native storage owner."""
    decision = table_decision(source, table)
    refuse_broken_presence(table, decision)
    if not decision.files:
        record_table_unavailable(source, table)
        return None
    return resolve_table_files(source, table, stay_column=stay, stay_values=ids)


# Clock expressions use raw Parquet units. The general concept loader can
# normalize AUMC milliseconds before subtracting the admission timestamp.
_PEEP_SPEC = {
    "aumc": ("admissionid", "itemid", "12284, 12336, 8862", "value", "(e.measuredat - s.admittedat) / 3600000.0"),
    "hirid": ("patientid", "variableid", "2600", "value",
              "epoch(CAST(e.datetime AS TIMESTAMP) - CAST(s.admissiontime AS TIMESTAMP)) / 3600.0"),
    "miiv": ("stay_id", "itemid", "220339", "valuenum",
             "epoch(CAST(e.charttime AS TIMESTAMP) - CAST(s.intime AS TIMESTAMP)) / 3600.0"),
    "sic": ("CaseID", "DataID", "2278", "Val", '(e."Offset" - s.ICUOffset) / 3600.0'),
}


def _observed_peep_set(con, family, stays, events):
    """Project frozen sources.py:28, :42–60, :91–107 and :127–145.

    Pool the specified raw setting channels in cmH2O with inclusive [0, 40]
    filtering before the per-stay/hour median. AUMC excludes 16250 and uses
    the admission clock in milliseconds. HiRID and MIIV use timestamps since
    ICU admission (seconds converted to hours). Floor to [hour, hour+1),
    including negative hours. A setting with no hour cannot join the panel.
    The later eligibility filter in events.py:34–46 applies to event pairs.
    SIC uses the unconverted Val and subtracts cases.ICUOffset in seconds
    before flooring, following frozen new_sources.py:129–151.
    """
    stay, item, ids, value, offset = _PEEP_SPEC[family]
    return con.execute(f"""
        SELECT e.{stay} AS stay_id, CAST(floor({offset}) AS INTEGER) AS charttime,
               median(e.{value}) AS observed_peep_set
        FROM {events} e
        JOIN {stays} s ON s.{stay} = e.{stay}
        SEMI JOIN selected_stays m ON m.stay_id = e.{stay}
        WHERE e.{item} IN ({ids}) AND e.{value} BETWEEN 0 AND 40
        GROUP BY 1, 2
        HAVING charttime IS NOT NULL
    """).df()


@in_binding_unit
def load_thesis_support(*, module, database, data_path, patient_ids=None) -> pd.DataFrame:
    """Rebuild from raw inputs for selected stays, with a typed empty fallback."""
    family = database.removesuffix("_demo")
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    frame = _empty(module, stay)
    frame.attrs["available_fields"] = []
    if family == "miiv" and patient_ids is None:
        raise ValueError("MIIV thesis support requires an explicit allowed stay cohort")
    raw = _PEEP.get(family) if module == "observed_peep_set" else None
    if raw is None:
        return frame

    source = storage_source(database, data_path)
    selection = select_support_cohort(database, data_path, patient_ids)
    ids = None if selection is None else selection[stay]
    inputs = [support_input_files(source, table, stay, ids) for table in raw.tables]
    if any(files is None for files in inputs):
        return frame

    frame.attrs["available_fields"] = ["observed_peep_set"]
    if ids == [] or any(not files for files in inputs):
        return frame

    with duckdb_spill.connection() as con:
        if ids is None:
            stay_table = DATABASE_ID_CONFIG[database]["table"]
            files = support_input_files(source, stay_table, stay, None)
            if files is None:
                frame.attrs["available_fields"] = []
                return frame
            con.execute(f"""
                CREATE TEMP VIEW selected_stays AS
                SELECT DISTINCT {stay} AS stay_id FROM {parquet_relation(files, ordered=False)}
            """)
        else:
            con.register("selected_stays", pd.DataFrame({"stay_id": ids}))
        rows = _observed_peep_set(
            con, family, *(parquet_relation(files, ordered=False) for files in inputs),
        ).rename(columns={"stay_id": stay})

    keys = [stay, "charttime"]
    require_unique_keys(rows, keys, table=module)
    if rows.charttime.isna().any():
        raise ValueError(f"{module}: active rows require a source ICU hour")
    result = rows.reindex(columns=frame.columns).astype(frame.dtypes).sort_values(keys, ignore_index=True)
    result.attrs.update(frame.attrs)
    return result
