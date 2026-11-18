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
class _Authority:
    source: str
    tables: tuple[str, ...]
    value: str


_IMV = {
    db: _Authority("sealed_hourly_imv", (f"{db}_hourly_core",), "imv")
    for db in ("aumc", "miiv", "sic")
}
_IMV["hirid"] = _Authority("hirid_recorded_anchor", ("hirid_vent_anchor",), "imv_recorded_end_hour")

_PEEP = {
    "aumc": _Authority("aumc_numericitems", ("admissions", "numericitems"), "observed_peep_set"),
    "hirid": _Authority("hirid_observations", ("general", "observations"), "observed_peep_set"),
    "miiv": _Authority("miiv_chartevents", ("icustays", "chartevents"), "observed_peep_set"),
}

_RRT = {
    db: _Authority("sealed_hourly_rrt", (f"{db}_hourly_core",), "rrt_active")
    for db in ("aumc", "eicu", "hirid", "sic")
}
_RRT["miiv"] = _Authority("miiv_rrt_time_authority_actual_end", ("miiv_rrt_time",), "rrt_active")
_RRT["mimic"] = _Authority("carevue_timed_chart_procedure", ("mimic3cv_rrt_hours",), "rrt_active")

_AUTHORITIES = {"imv_recorded_compat": _IMV, "observed_peep_set": _PEEP, "rrt_time": _RRT}


def thesis_support_manifest(module: str, database: str, fields: Collection[str]) -> dict:
    """Describe the frozen input and distinguish an unavailable authority."""
    family = database.removesuffix("_demo")
    authority = _AUTHORITIES[module].get(family)
    if authority is None:
        sic_peep = module == "observed_peep_set" and family == "sic"
        return {
            "source": "sic_data_float_h.Val" if sic_peep else None,
            "status": "frozen_reducer_unavailable" if sic_peep else "no_frozen_reducer",
        }
    span = authority.source == "hirid_recorded_anchor"
    return {
        "source": authority.source,
        "input_tables": list(authority.tables),
        "status": "projected" if authority.value in fields else "authority_unavailable",
        "charttime_role": "first_recorded_hour" if span else "icu_hour",
        "source_grain": "stay" if span else "stay_hour",
        **(
            {"upstream_tables": ["chartevents", "procedureevents"]}
            if authority.source == "miiv_rrt_time_authority_actual_end" else {}
        ),
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


def _imv_recorded_compat(con, family, relation):
    """Project frozen cohort.py:45–52 and :84–104.

    HiRID consumes the recorded first/last native invasive bounds from the
    flat anchor. The upstream airway/mode codes and span builder are outside
    those lines, so this projection requires the authoritative anchor. Bounds
    already use ICU-admission-relative hours. Clamp start to zero, bring end
    up to start if needed, then floor both to inclusive hour labels. The
    first hour is charttime and the last hour is imv_recorded_end_hour.

    AUMC, MIIV and SIC carry the sealed ICU-hour key as-is, selecting only
    true invasive flags. No time filter, rounding or aggregation is applied.
    Gap tolerance, longest run and episode duration are downstream.
    """
    if family == "hirid":
        return con.execute(f"""
            SELECT stay_id,
                   CAST(floor(greatest(imv_t0_h, 0)) AS INTEGER) AS charttime,
                   CAST(floor(greatest(imv_end_h, greatest(imv_t0_h, 0))) AS INTEGER)
                       AS imv_recorded_end_hour
            FROM {relation}
            SEMI JOIN selected_stays USING (stay_id)
            WHERE greatest(imv_t0_h, 0) IS NOT NULL
        """).df()
    return con.execute(f"""
        SELECT stay_id, icu_hour AS charttime, imv
        FROM {relation}
        SEMI JOIN selected_stays USING (stay_id)
        WHERE imv
    """).df()


# Clock expressions use raw Parquet units. The general concept loader can
# normalize AUMC milliseconds before subtracting the admission timestamp.
_PEEP_SPEC = {
    "aumc": ("admissionid", "itemid", "12284, 12336, 8862", "value", "(e.measuredat - s.admittedat) / 3600000.0"),
    "hirid": ("patientid", "variableid", "2600", "value",
              "epoch(CAST(e.datetime AS TIMESTAMP) - CAST(s.admissiontime AS TIMESTAMP)) / 3600.0"),
    "miiv": ("stay_id", "itemid", "220339", "valuenum",
             "epoch(CAST(e.charttime AS TIMESTAMP) - CAST(s.intime AS TIMESTAMP)) / 3600.0"),
}


def _observed_peep_set(con, family, stays, events):
    """Project frozen sources.py:28, :42–60, :91–107 and :127–145.

    Pool the specified raw setting channels in cmH2O with inclusive [0, 40]
    filtering before the per-stay/hour median. AUMC excludes 16250 and uses
    the admission clock in milliseconds. HiRID and MIIV use timestamps since
    ICU admission (seconds converted to hours). Floor to [hour, hour+1),
    including negative hours. A setting with no hour cannot join the panel.
    The later eligibility filter in events.py:34–46 applies to event pairs.
    SIC dispatches to new_sources.py via sources.py:149–151. Its reducer is
    required to establish how the original hourly Val was interpreted.
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


def _rrt_time(con, family, relation):
    """Project frozen cohort.py:55–65 and prepare_miiv_views.py:24, :28–31.

    MIIV requires the corrected actual-end authority derived from chart and
    procedure records, including its pre-ICU -24 h window. It retains true
    treatment hours and casts the supplied ICU-relative charttime to INTEGER
    with DuckDB's cast semantics. The producer rules for actual ends are
    outside the frozen view copier. CareVue consumes the preselected timed
    chart/procedure hours. The other databases select true rrt from sealed
    hourly core. Those hour keys already use the native ICU admission clock.
    No flooring, carry, aggregation or extra ICU/landmark clipping is applied.
    """
    hour, where = {
        "miiv": ("CAST(charttime AS INTEGER)", "WHERE rrt"),
        "mimic": ("icu_hour", ""),
    }.get(family, ("icu_hour", "WHERE coalesce(rrt, FALSE)"))
    return con.execute(f"""
        SELECT stay_id, {hour} AS charttime, TRUE AS rrt_active
        FROM {relation}
        SEMI JOIN selected_stays USING (stay_id)
        {where}
    """).df()


_REDUCERS = {
    "imv_recorded_compat": _imv_recorded_compat,
    "observed_peep_set": _observed_peep_set,
    "rrt_time": _rrt_time,
}


@in_binding_unit
def load_thesis_support(*, module, database, data_path, patient_ids=None) -> pd.DataFrame:
    """Load selected source stays and carry a typed schema even when unavailable.

    The lead can stage authoritative L-COMMON inputs as named Parquet tables
    under the database root. No raw proxy is substituted for an unstaged input.
    For MIIV, pass the approved stay IDs as the requested cohort.
    """
    family = database.removesuffix("_demo")
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    frame = _empty(module, stay)
    frame.attrs["available_fields"] = []
    authority = _AUTHORITIES[module].get(family)
    if authority is None:
        return frame

    if family == "miiv" and patient_ids is None:
        raise ValueError("MIIV thesis support requires an explicit allowed stay cohort")
    source = storage_source(database, data_path)
    selection = select_support_cohort(database, data_path, patient_ids)
    ids = None if selection is None else selection[stay]
    source_key = stay if module == "observed_peep_set" else "stay_id"
    inputs = [support_input_files(source, table, source_key, ids) for table in authority.tables]
    if any(files is None for files in inputs):
        return frame

    frame.attrs["available_fields"] = [authority.value]
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
        rows = _REDUCERS[module](
            con, family, *(parquet_relation(files, ordered=False) for files in inputs),
        ).rename(columns={"stay_id": stay})

    keys = [stay] if authority.source == "hirid_recorded_anchor" else [stay, "charttime"]
    require_unique_keys(rows, keys, table=module)
    if rows.charttime.isna().any():
        raise ValueError(f"{module}: active rows require a source ICU hour")
    result = rows.reindex(columns=frame.columns).astype(frame.dtypes).sort_values(keys, ignore_index=True)
    result.attrs.update(frame.attrs)
    return result
