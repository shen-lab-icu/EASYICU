"""Opt-in projections for the frozen L-COMMON thesis inputs."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from hashlib import file_digest, sha256
import json
from operator import index
import os
from pathlib import Path
import re
from stat import S_ISREG

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import DATABASE_ID_CONFIG
from ..datasource import storage_source
from ..io.identity import require_unique_keys
from ..runtime import duckdb_spill
from ..storage_binding import bind_files, bound_sha256, in_binding_unit
from ..storage_layout import (
    parquet_relation,
    record_table_unavailable,
    refuse_broken_presence,
    resolve_table_files,
    table_decision,
)
from .catalog import FIRST_IN_HOSP_FROZEN, THESIS_SUPPORT_MODULES
from .metadata_projection import SourceLineage
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

_FIRST_IN_HOSP = {
    "aumc": ("admissions", "admissioncount", "admissions.admissioncount = 1"),
    "sic": ("cases", "OffsetAfterFirstAdmission", "COALESCE(cases.OffsetAfterFirstAdmission, 0) = 0"),
}
FIRST_IN_HOSP_SOURCE_KEY = b"easyicu:first_in_hosp_frozen"


def has_frozen_first_in_hosp(database: str) -> bool:
    return database.removesuffix("_demo") in _FIRST_IN_HOSP


def _first_in_hosp_manifest(family, fields, source_file):
    if family not in _FIRST_IN_HOSP:
        return {
            "source": None, "status": "UNAVAILABLE_IN_SCOPE",
            "reason": "no_frozen_predicate_in_scope", "source_grain": "stay",
            "availability_basis": "no_frozen_predicate_in_scope",
        }
    table, field, expression = _FIRST_IN_HOSP[family]
    projected = FIRST_IN_HOSP_FROZEN in fields
    receipt = {
        "source": f"{table}.{field}",
        "source_table": table,
        "source_field": field,
        "input_tables": [table],
        "expression": expression,
        "frozen_reference": "build_c565v6.py:99-107",
        "source_grain": "stay",
        "status": "projected" if projected else "authority_unavailable",
        "availability_basis": "frozen_raw_predicate" if projected else "source_unavailable",
    }
    return {**receipt, **(source_file or {})}


def first_in_hosp_column_metadata(metadata, receipt):
    """Attach the producer's resolved input to the typed column binding."""
    parameters = ("expression", "frozen_reference", "source_realpath", "source_sha256")
    lineage = SourceLineage(
        database=metadata.source_database,
        table=receipt["source_table"],
        value_variable=receipt["source_field"],
        selector_variable=None, selector_regex=None, item_ids_json=(), unit_variable=None,
        time_variable=None, duration_variable=None, source_class_name=None, target=None,
        callback=None, interval_iso8601=None,
        semantic_parameters=tuple((key, json.dumps(receipt[key])) for key in parameters),
    )
    return replace(
        metadata,
        available_databases=tuple(_FIRST_IN_HOSP),
        source_declared_for_database=True,
        availability_basis=receipt["availability_basis"],
        source_lineage=(lineage,),
    )

PINNED_SUPPORT_MODULES = frozenset({"imv_recorded_compat", "rrt_time"})
_PINNED_D0_RELEASE = "20261117_d93795ce4_default_r14"
_PINNED_D0_MANIFEST_SHA16 = "7f14448b9d9d1870"
_HOURLY_CONTRACT_PATH = Path(
    "/home/zhuhb/workspace/claude-ws/_wt/rfrunL/L-HC/frozen/hc/config/hourly_core_v7.json"
)
_HOURLY_CONTRACT_SHA256 = "e937d67be3a335f491638c173179d6ec10f222c094e934f1a5bc474f2da971ae"
_OUTCOME = ("stay_id", "los_icu")
_RESPIRATORY = ("stay_id", "charttime", "mech_vent")
_RENAL = ("stay_id", "charttime", "rrt")
# cohort.py:69-70 reads this from release parameters. The pinned hourly JSON has no cohort section.
_IMV_WITHOUT_GAP = frozenset({"aumc", "miiv", "sic"})
_PINNED_COLUMNS = {
    **{("imv_recorded_compat", db): {"respiratory": _RESPIRATORY, "outcome": _OUTCOME}
       for db in (*_IMV_WITHOUT_GAP, "hirid")},
    **{("rrt_time", db): {"renal": _RENAL, "outcome": _OUTCOME}
       for db in ("aumc", "eicu", "hirid", "sic")},
}


@dataclass(frozen=True)
class PinnedModuleInput:
    """One D0 module file with the lead's full expected digest and read projection."""

    module: str
    path: Path
    expected_sha256: str
    allowed_columns: tuple[str, ...]


class PinnedInputError(ValueError):
    """A pinned input failed closed before any rows could be returned."""

    def __init__(self, reason_code: str, module: str | None = None):
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {module}" if module else reason_code)


def _pinned_manifest(module, family, *, inputs=()):
    allowed = _PINNED_COLUMNS.get((module, family), {})
    span = module == "imv_recorded_compat" and family == "hirid"
    grain = {
        "charttime_role": "first_recorded_hour" if span else "icu_hour",
        "source_grain": "stay" if span else "stay_hour",
    }
    if module == "rrt_time" and family == "miiv":
        return {
            **grain, "source": None, "status": "UNAVAILABLE_IN_SCOPE",
            "reason": "no_exact_producer",
            "missing_inputs": ["miiv_rrt_time.parquet: treatment start and actual end through pre-ICU hour -24"],
        }
    if not allowed:
        return {"source": None, "status": "no_frozen_reducer"}
    if module == "imv_recorded_compat" and family in _IMV_WITHOUT_GAP:
        missing = ["release.params.cohort.episode_gap_tolerance_h"]
    else:
        provided = {entry.module for entry in inputs}
        missing = [f"{family}/{name}.parquet:{columns[-1]}" for name, columns in allowed.items()
                   if name not in provided]
    if missing:
        return {
            **grain, "source": None, "status": "UNAVAILABLE_IN_SCOPE",
            "reason": "no_raw_producer", "missing_inputs": missing,
        }
    return {
        **grain, "source": _PINNED_D0_RELEASE, "status": "projected",
        "source_manifest_sha16": _PINNED_D0_MANIFEST_SHA16,
        "hourly_contract_sha256": _HOURLY_CONTRACT_SHA256,
        "inputs": {entry.module: {"sha256": entry.expected_sha256, "columns": list(entry.allowed_columns)}
                   for entry in inputs},
        "frozen_reference": (
            "build_flat_tables.py:903-946; cohort.py:84-101,158" if span
            else "hourly_core.py:108-160,245-315,920-934,964-1017; cohort.py:65"
        ),
    }


def thesis_support_manifest(module: str, database: str, fields: Collection[str], *, source_file=None) -> dict:
    """Report raw input provenance or the frozen producer's missing inputs."""
    if module not in THESIS_SUPPORT_MODULES:
        raise KeyError(module)
    family = database.removesuffix("_demo")
    if module == FIRST_IN_HOSP_FROZEN:
        return _first_in_hosp_manifest(family, fields, source_file)
    if module in PINNED_SUPPORT_MODULES:
        return _pinned_manifest(module, family)
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
        **({"charttime": pd.Series(dtype="float64")} if module != FIRST_IN_HOSP_FROZEN else {}),
        **{
            name: pd.Series(dtype="boolean" if unit == "boolean" else "float64")
            for name, (_, unit) in fields.items()
        },
    })


def validate_pinned_inputs(module, database, inputs):
    """Refuse forbidden declarations without resolving or opening any path."""
    allowed = _PINNED_COLUMNS.get((module, database), {})
    seen = set()
    for entry in inputs:
        if not isinstance(entry, PinnedModuleInput):
            raise PinnedInputError("pinned_input_type_invalid")
        name = entry.module
        if name not in allowed:
            raise PinnedInputError("pinned_module_not_allowed")
        if name in seen:
            raise PinnedInputError("pinned_module_duplicate", name)
        seen.add(name)
        if (len(entry.allowed_columns) != len(allowed[name])
                or set(entry.allowed_columns) != set(allowed[name])):
            raise PinnedInputError("pinned_columns_not_allowed", name)
        path = Path(entry.path)
        if path.name != f"{name}.parquet" or path.parent.name != database:
            raise PinnedInputError("pinned_path_not_allowed", name)
        if not re.fullmatch(r"[0-9a-f]{64}", entry.expected_sha256):
            raise PinnedInputError("pinned_sha256_invalid", name)


def _file_state(handle):
    info = os.fstat(handle.fileno())
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def open_pinned_modules(module, database, inputs):
    """Verify every file before any Parquet decoder is called, then reuse its descriptor."""
    inputs = tuple(inputs)
    validate_pinned_inputs(module, database, inputs)
    with ExitStack() as stack:
        verified = {}
        for entry in inputs:
            path = Path(entry.path)
            # Reject redirection to an unrelated input, including a parent-directory symlink.
            if path.resolve() != Path(os.path.abspath(path)):
                raise PinnedInputError("pinned_path_not_allowed", entry.module)
            try:
                handle = stack.enter_context(open(path, "rb", opener=_open_regular_pinned_file))
            except FileNotFoundError:
                raise PinnedInputError("pinned_file_missing", entry.module) from None
            except OSError:
                raise PinnedInputError("pinned_file_unreadable", entry.module) from None
            before = _file_state(handle)
            if file_digest(handle, "sha256").hexdigest() != entry.expected_sha256:
                raise PinnedInputError("pinned_sha256_mismatch", entry.module)
            if _file_state(handle) != before:
                raise PinnedInputError("pinned_file_changed", entry.module)
            handle.seek(0)
            verified[entry.module] = (entry, handle, before)
        yield {name: (entry, handle) for name, (entry, handle, _) in verified.items()}
        for name, (_, handle, before) in verified.items():
            if _file_state(handle) != before:
                raise PinnedInputError("pinned_file_changed", name)


def _open_regular_pinned_file(path, flags):
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise OSError("expected regular file")
    return fd


def read_pinned_columns(opened, *, stays=None):
    """Decode the declared columns from the already verified descriptor."""
    entry, handle = opened
    handle.seek(0)
    with pq.ParquetFile(handle) as parquet:
        if not set(entry.allowed_columns).issubset(parquet.schema_arrow.names):
            raise PinnedInputError("pinned_columns_missing", entry.module)
    filters = None if stays is None else [("stay_id", "in", list(stays))]
    handle.seek(0)
    return pq.read_table(handle, columns=list(entry.allowed_columns), filters=filters).to_pandas()


def pinned_stay_ids(database, patient_ids):
    """Resolve explicit stay keys directly, without consulting a raw identity table."""
    if patient_ids is None:
        if database.removesuffix("_demo") == "miiv":
            raise ValueError("MIIV thesis support requires an explicit allowed stay cohort")
        return None
    if isinstance(patient_ids, Mapping):
        stay = DATABASE_ID_CONFIG[database]["id_col"]
        if len(patient_ids) != 1 or next(iter(patient_ids)) not in {stay, "stay_id"}:
            raise PinnedInputError("pinned_cohort_requires_stay_ids")
        patient_ids = next(iter(patient_ids.values()))
    try:
        members = set()
        for value in patient_ids:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError
            member = index(value)
            if not -(2**63) <= member < 2**63:
                raise ValueError
            members.add(member)
    except (TypeError, ValueError):
        raise PinnedInputError("pinned_cohort_requires_stay_ids") from None
    return members


def read_hourly_contract():
    """Load the exact S1338-authorized contract used by build_hourly_core.py:38-49."""
    try:
        if _HOURLY_CONTRACT_PATH.resolve() != _HOURLY_CONTRACT_PATH:
            raise PinnedInputError("frozen_hourly_contract_path_redirects")
        source = _HOURLY_CONTRACT_PATH.read_bytes()
    except OSError:
        raise PinnedInputError("frozen_hourly_contract_unreadable") from None
    if sha256(source).hexdigest() != _HOURLY_CONTRACT_SHA256:
        raise PinnedInputError("frozen_hourly_contract_sha256_mismatch")
    return json.loads(source)


def _valid_los_hours(outcome, clock):
    # hourly_core.py:108-160. The hashed config declares the clock and spine source.
    if (clock["spine_source"], clock["spine_source_unit"], clock["source_time"]) != (
        "outcome.los_icu", "days", "ICU-relative charttime in hours",
    ):
        raise PinnedInputError("frozen_hourly_clock_unsupported")
    require_unique_keys(outcome, ["stay_id"], table="outcome")
    los = pd.to_numeric(outcome.set_index("stay_id")["los_icu"], errors="coerce")
    return los[los.notna() & np.isfinite(los) & los.gt(0)].astype("float64") * 24.0


def _rrt_hours(source, los_hours, contract):
    # _prepare_dynamic checks target-bearing rows even when they lie outside the spine.
    target = contract["dynamic"]["renal"]["rrt"]
    source = source.rename(columns={"rrt": target})
    source = source.loc[source[target].notna() & source["stay_id"].isin(los_hours.index)].copy()
    time = pd.to_numeric(source["charttime"], errors="coerce")
    hour = np.floor(time)
    if not (np.isfinite(time) & np.isclose(time, hour)).all():
        raise PinnedInputError("pinned_rrt_hour_invalid", "renal")
    source["charttime"] = hour.astype("int32")
    require_unique_keys(source, ["stay_id", "charttime"], table="renal")
    clock = contract["time_contract"]
    if clock["last_hour"] != "ceil(los_icu * 24) - 1":
        raise PinnedInputError("frozen_hourly_clock_unsupported")
    last = source["stay_id"].map(np.ceil(los_hours) - 1)
    keep = source[target].astype("boolean").fillna(False) & source["charttime"].between(clock["first_hour"], last)
    return source.loc[keep, ["stay_id", "charttime"]].assign(rrt_active=True)


def _hirid_recorded_span(source, los_hours):
    # build_flat_tables.py:903-946, then cohort.py:84-101,158.
    original = source["charttime"]
    time = pd.to_numeric(original, errors="coerce")
    if (original.notna() & ~np.isfinite(time)).any():
        raise PinnedInputError("pinned_vent_hour_invalid", "respiratory")
    source = source.assign(charttime=time)
    if source["stay_id"].isna().any() or source.duplicated(["stay_id", "charttime"]).any():
        raise PinnedInputError("pinned_vent_key_invalid", "respiratory")
    invasive = source.loc[source["mech_vent"].astype("string").str.lower().eq("invasive").fillna(False)]
    if invasive["charttime"].dropna().empty:
        raise PinnedInputError("pinned_vent_anchor_no_invasive_rows", "respiratory")
    span = invasive.groupby("stay_id")["charttime"].agg(["min", "max"]).dropna()
    start = span["min"].clip(lower=0)
    span["charttime"] = np.floor(start)
    span["imv_recorded_end_hour"] = np.floor(np.maximum(start, span["max"]))
    keep = span["charttime"].lt(los_hours.reindex(span.index))
    return span.loc[keep, ["charttime", "imv_recorded_end_hour"]].reset_index().assign(imv=True)


def _load_pinned_support(frame, module, database, patient_ids, pinned_inputs):
    inputs = tuple(pinned_inputs or ())
    validate_pinned_inputs(module, database, inputs)
    stays = pinned_stay_ids(database, patient_ids)
    receipt = _pinned_manifest(module, database.removesuffix("_demo"), inputs=inputs)
    frame.attrs["thesis_support"] = receipt
    if receipt["status"] != "projected":
        return frame

    contract = read_hourly_contract()
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    with open_pinned_modules(module, database, inputs) as opened:
        if stays != set():
            outcome = read_pinned_columns(opened["outcome"], stays=stays)
            los = _valid_los_hours(outcome, contract["time_contract"])
            if los.empty:
                raise PinnedInputError("pinned_no_valid_icu_los", "outcome")
            if module == "rrt_time":
                source = read_pinned_columns(opened["renal"], stays=los.index)
                rows = _rrt_hours(source, los, contract)
            else:
                source = read_pinned_columns(opened["respiratory"], stays=stays)
                rows = _hirid_recorded_span(source, los)
            rows = rows.rename(columns={"stay_id": stay}).reindex(columns=frame.columns).astype(frame.dtypes)
            rows = rows.sort_values([stay, "charttime"], ignore_index=True)
            rows.attrs["pinned_cohort_size"] = len(los)
        else:
            rows = frame
            rows.attrs["pinned_cohort_size"] = 0
    rows.attrs.update(frame.attrs)
    rows.attrs["available_fields"] = list(THESIS_SUPPORT_MODULES[module])
    return rows


def support_input_files(source, table, stay, ids):
    """Resolve the exact Parquet input through the native storage owner."""
    decision = table_decision(source, table)
    refuse_broken_presence(table, decision)
    if not decision.files:
        record_table_unavailable(source, table)
        return None
    return resolve_table_files(source, table, stay_column=stay, stay_values=ids)


def _load_first_in_hosp(frame, database, data_path, patient_ids, stay):
    family = database.removesuffix("_demo")
    if not has_frozen_first_in_hosp(database):
        return frame
    table, _, expression = _FIRST_IN_HOSP[family]
    files = support_input_files(storage_source(database, data_path), table, stay, None)
    if not files:
        return frame
    if len(files) != 1:
        raise ValueError(f"{FIRST_IN_HOSP_FROZEN}: {table} requires a single source Parquet file")
    (file,) = files
    path = file.resolve(strict=True)
    bind_files(table, (path,), sha256=True)
    selection = select_support_cohort(database, data_path, patient_ids)
    relation = parquet_relation((path,), ordered=False)
    with duckdb_spill.connection() as con:
        cohort = ""
        if selection is not None:
            con.register("selected_stays", pd.DataFrame({stay: pd.Series(selection[stay], dtype="int64")}))
            cohort = f"SEMI JOIN selected_stays USING ({stay})"
        rows = con.execute(f"""
            SELECT {table}.{stay}, {expression} AS {FIRST_IN_HOSP_FROZEN}
            FROM {relation} AS {table}
            {cohort}
        """).df()
    bind_files(table, (path,))
    require_unique_keys(rows, [stay], table=table)
    result = rows.astype(frame.dtypes).sort_values(stay, ignore_index=True)
    result.attrs["available_fields"] = [FIRST_IN_HOSP_FROZEN]
    result.attrs["first_in_hosp_source_file"] = {
        "source_realpath": str(path),
        "source_sha256": bound_sha256(path),
    }
    return result


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
def load_thesis_support(
    *, module, database, data_path=None, patient_ids=None,
    pinned_inputs: Collection[PinnedModuleInput] | None = None,
) -> pd.DataFrame:
    """Rebuild the frozen support projection for the explicitly selected input."""
    family = database.removesuffix("_demo")
    stay = DATABASE_ID_CONFIG[database]["id_col"]
    frame = _empty(module, stay)
    frame.attrs["available_fields"] = []
    if module in PINNED_SUPPORT_MODULES:
        return _load_pinned_support(frame, module, database, patient_ids, pinned_inputs)
    if pinned_inputs is not None:
        raise PinnedInputError("pinned_module_not_allowed")
    if module == FIRST_IN_HOSP_FROZEN:
        result = _load_first_in_hosp(frame, database, data_path, patient_ids, stay)
        result.attrs["thesis_support"] = thesis_support_manifest(
            module, database, result.attrs["available_fields"],
            source_file=result.attrs.get("first_in_hosp_source_file"),
        )
        return result
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
