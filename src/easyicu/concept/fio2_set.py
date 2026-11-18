"""Set FiO2 from the frozen L-COMMON settings channels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..datasource import FilterOp, FilterSpec, MissingTableSourceError, drop_mimic_chart_errors
from ..table import ICUTable
from .availability_signal import ConceptAvailabilityRecord, set_table_availability
from .carevue_view import carevue_view_has_table, is_carevue_view
from .merge_time_index import RELATIVE_HOURS
from .stay_tables import selected_stay_ids

ORIGINS = {
    "miiv": ("icustays", "intime"),
    "mimic": ("icustays", "intime"),
    "aumc": ("admissions", "admittedat"),
    "hirid": ("general", "admissiontime"),
    "sic": ("cases", "ICUOffset"),
}
_HOUR = pd.Timedelta(hours=1)


def _hourly_result(frame, stay_column):
    return ICUTable(
        data=frame, id_columns=[stay_column], index_column="charttime",
        value_column="fio2_set", time_basis=RELATIVE_HOURS, interval=_HOUR,
    )


def _empty_hourly_result(stay_column):
    return _hourly_result(pd.DataFrame({
        stay_column: pd.Series(dtype="int64"),
        "charttime": pd.Series(dtype="float64"),
        "fio2_set": pd.Series(dtype="float64"),
    }), stay_column)


def _settings_absence(database, stay_column, table, reason_code, *, missing_tables=()):
    result = _empty_hourly_result(stay_column)
    set_table_availability(result, ConceptAvailabilityRecord(
        concept="fio2_set", database=database, reason="source_unavailable", n_rows=0,
        sources_defined=(table,), missing_tables=missing_tables,
        evaluation_status="not_evaluable", reason_code=reason_code,
    ))
    return result


def _read_settings_input(data_source, table, columns, filters=None):
    try:
        frame = data_source.load_table(table, columns=columns, filters=filters).data
    except (FileNotFoundError, MissingTableSourceError):
        return None
    if frame.empty and not data_source.has_table_source(table):
        return None
    return frame


def _icu_hours(rows, source_time, origin_time, database):
    if database == "eicu":
        return pd.to_numeric(rows[source_time], errors="coerce") / 60
    if database == "sic":
        return (rows[source_time] - rows[origin_time]) / 3600
    if database == "aumc":
        # The source reader returns minute floats from integral milliseconds.
        # Recover the original precision before subtracting and flooring.
        event_ms = (rows[source_time] * 60_000).round()
        origin_ms = (rows[origin_time] * 60_000).round()
        return (event_ms - origin_ms) / 3_600_000
    delta = pd.to_datetime(rows[source_time]) - pd.to_datetime(rows[origin_time])
    return delta.dt.total_seconds() / 3600


def load_fio2_set(definition, source, data_source, patient_ids):
    """Filter raw settings on the source clock and pool one median per hour."""
    source_database = data_source.config.name
    database = "eicu" if source_database == "eicu_demo" else source_database
    stay_column = data_source.config.id_configs["icustay"].id
    selected = selected_stay_ids(patient_ids, stay_column)
    stay_filters = [] if selected is None else [FilterSpec(stay_column, FilterOp.IN, selected)]

    if database in {"mimic", "mimic_demo"}:
        if database != "mimic" or not is_carevue_view(data_source.base_path):
            return _settings_absence(database, stay_column, source.table, "carevue_view_required")
        required = (ORIGINS[database][0], source.table)
        missing = tuple(table for table in required if not carevue_view_has_table(data_source, table))
        if missing:
            return _settings_absence(
                database, stay_column, source.table, "required_source_table_missing",
                missing_tables=missing,
            )

    if selected == []:
        return _empty_hourly_result(stay_column)

    origins = None
    origin_time = None
    if database in ORIGINS:
        origin_table, origin_time = ORIGINS[database]
        columns = [stay_column, origin_time]
        if database == "mimic":
            # Verify the entire view before applying any patient-batch filter.
            origins = _read_settings_input(data_source, origin_table, [*columns, "dbsource"])
        else:
            origins = _read_settings_input(data_source, origin_table, columns, stay_filters or None)
        if origins is None:
            return _settings_absence(
                source_database, stay_column, source.table, "required_source_table_missing",
                missing_tables=(origin_table,),
            )
        if database == "mimic":
            if (
                "dbsource" not in origins
                or not origins["dbsource"].astype("string").str.strip().str.lower().eq("carevue").fillna(False).all()
            ):
                return _settings_absence(database, stay_column, source.table, "carevue_view_required")
            origins = origins[columns]

    columns = [stay_column, source.index_var, source.value_var, source.sub_var]
    filters = [FilterSpec(source.sub_var, FilterOp.IN, source.ids), *stay_filters]
    if database == "eicu":
        columns.append("respcharttypecat")
        filters.append(FilterSpec("respcharttypecat", FilterOp.EQ, "respFlowSettings"))
    if database == "mimic":
        columns.append("error")

    rows = _read_settings_input(data_source, source.table, columns, filters)
    if rows is None:
        return _settings_absence(
            source_database, stay_column, source.table, "required_source_table_missing",
            missing_tables=(source.table,),
        )
    if database == "mimic":
        rows = drop_mimic_chart_errors(rows, database, source.table)
    if origins is not None:
        rows = rows.merge(origins, on=stay_column, validate="many_to_one")

    time = _icu_hours(rows, source.index_var, origin_time, database)
    values = pd.to_numeric(rows[source.value_var], errors="coerce")
    if database in {"miiv", "mimic", "eicu"}:
        values = values.mask(values.le(1), values * 100)
    valid = values.between(definition.minimum, definition.maximum) & np.isfinite(time)
    if database == "eicu":
        # Frozen thesis L-COMMON eICU FiO2 window (r15e DIFF, accepted in #742).
        valid &= time.ge(0) & time.lt(1440)
    hourly = rows.loc[valid, [stay_column]].assign(
        charttime=np.floor(time.loc[valid]).astype("float64"),
        fio2_set=values.loc[valid].astype("float64"),
    )
    result = hourly.groupby([stay_column, "charttime"], as_index=False)["fio2_set"].median()
    return _hourly_result(result, stay_column)
