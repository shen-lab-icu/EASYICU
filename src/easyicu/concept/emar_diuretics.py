"""MIMIC-IV loop-diuretic doses given on the ward chart (eMAR).

``inputevents`` holds only the ICU infusion pump and nurse-charted IV doses.
Many ICU loop-diuretic doses are recorded only in the hospital eMAR: 5,829
stays have eMAR administrations in the ICU and no loop diuretic in
inputevents, 34.6% of non-oral eMAR furosemide rows have no inputevents row
within 60 minutes, and oral furosemide and torasemide have no other dose
source (2026-11-03 extraction audit round 2, D1).

A source with this callback reads ``emar`` rows of one drug and keeps the
administered events (``emar_administration.ADMINISTERED_EVENTS``).  The dose
is the sum of ``emar_detail.dose_given`` over the event's detail rows, in
``dose_given_unit``.  The route is ``emar_detail.route`` and, where that is
empty, the route of the ``prescriptions`` order (``pharmacy_id``).  A
furosemide dose whose route stays unknown is not converted (oral and IV
furosemide differ twofold); bumetanide and torasemide convert the same either
way.  ``diuretics.furosemide_equivalent`` then owns the conversion.

The same ICU dose can be charted in both eMAR and inputevents. An eMAR dose
is dropped only for an explicit IV route, the same admission and drug,
matching positive mass dose and a start within ``DEDUP_WINDOW``. The real
channels have no shared administration identifier. Each input row can remove
at most one nearest eMAR copy; oral and unknown-route doses never deduplicate.

Dictionary use::

    "table": "emar", "sub_var": "medication", "regex": "...",
    "class_name": "rgx_itm",
    "extra_vars": ["event_txt", "emar_id", "pharmacy_id"],
    "callback": "emar_furosemide_equivalent", "fe_drug": "furosemide"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .diuretics import FUROSEMIDE_EQUIVALENT_FACTORS, furosemide_equivalent
from .emar_administration import ADMINISTERED_EVENTS, EVENT_COLUMN

CALLBACK = "emar_furosemide_equivalent"
#: inputevents items of the same drug (MIMIC-IV d_items).
INPUTEVENT_ITEMS: Mapping[str, tuple] = {
    "furosemide": (221794, 228340),
    "bumetanide": (229639,),
    "torasemide": (),
}
#: Routes (upper case) that are enteral administration.
ORAL_ROUTES = frozenset(
    {"PO", "NG", "OG", "PO/NG", "PO/OG", "NG/OG", "G TUBE", "J TUBE", "PEG", "ORAL", "SL"}
)
DEDUP_WINDOW = pd.Timedelta(minutes=60)
AUDIT_ATTR = "emar_furosemide_equivalent_audit"
_REQUIRED = ("emar_id", "pharmacy_id", EVENT_COLUMN)


def classify_route(route: pd.Series) -> pd.Series:
    """``'oral'``, ``'iv'`` or missing for free-text eMAR/prescription routes."""
    text = route.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NA, index=route.index, dtype="object")
    out[text.isin(ORAL_ROUTES).fillna(False).to_numpy()] = "oral"
    iv = text.str.fullmatch(r"(?:IV(?:PB|P)?|INTRAVENOUS)(?:\s+(?:DRIP|PUSH|PIGGYBACK|INFUSION|BOLUS))?", na=False)
    out[iv.to_numpy()] = "iv"
    return out


def event_doses(details: pd.DataFrame) -> pd.DataFrame:
    """Per ``emar_id``: summed ``dose_given``, its single unit and the route."""
    child = details.loc[details["parent_field_ordinal"].notna()].copy()
    child["_dose"] = pd.to_numeric(child["dose_given"], errors="coerce")
    unit = child.groupby("emar_id")["dose_given_unit"].agg(
        lambda units: units.dropna().iloc[0] if units.dropna().nunique() == 1 else pd.NA
    )
    dose = child.groupby("emar_id")["_dose"].sum(min_count=1)
    route = details.groupby("emar_id")["route"].agg(
        lambda routes: routes.dropna().iloc[0] if routes.notna().any() else pd.NA
    )
    return pd.DataFrame({"_dose": dose, "_unit": unit}).join(route.rename("_emar_route"), how="outer")


def drop_inputevent_duplicates(
    doses: pd.DataFrame, inputs: pd.DataFrame, *, key: str, time_col: str, drug: str
) -> pd.DataFrame:
    """Match IV copies by admission, drug, mass and nearest time, without IDs.

    A sorted asof join bounds memory to the two channels' row counts instead
    of constructing all dose pairs in an admission. A nearest input can remove
    only one eMAR row, so repeated same-dose administrations are preserved.
    Mass is compared in mg rounded to six decimals after unit conversion.
    """
    if doses.empty or inputs.empty:
        return doses
    required = {key, time_col, "_route", "_dose", "_unit"}
    if not required.issubset(doses) or not {key, "itemid", "starttime", "amount", "amountuom"}.issubset(inputs):
        return doses
    factors = {"mg": 1., "g": 1000., "gm": 1000., "mcg": .001, "ug": .001, "µg": .001}
    left = doses.loc[:, [key, time_col, "_route", "_dose", "_unit"]].copy()
    left["_row"] = np.arange(len(doses))
    left["_mg"] = (pd.to_numeric(left["_dose"], errors="coerce") * left["_unit"].astype("string").str.strip().str.casefold().map(factors)).round(6)
    left["_time"] = pd.to_datetime(left[time_col], errors="coerce").astype("datetime64[ns]")
    left[key] = pd.to_numeric(left[key], errors="coerce").astype("Int64")
    left = left.loc[left["_route"].eq("iv") & left["_mg"].gt(0) & np.isfinite(left["_mg"])].dropna(subset=[key, "_time"])
    right = inputs.loc[inputs.itemid.isin(INPUTEVENT_ITEMS[drug])].copy()
    if "statusdescription" in right:
        right = right.loc[~right.statusdescription.astype("string").str.strip().str.casefold().eq("rewritten").fillna(False)]
    right["_mg"] = (pd.to_numeric(right["amount"], errors="coerce") * right["amountuom"].astype("string").str.strip().str.casefold().map(factors)).round(6)
    right["_input_time"] = pd.to_datetime(right["starttime"], errors="coerce").astype("datetime64[ns]")
    right[key] = pd.to_numeric(right[key], errors="coerce").astype("Int64")
    right = right.loc[right["_mg"].gt(0) & np.isfinite(right["_mg"])].dropna(subset=[key, "_input_time"])
    if left.empty or right.empty:
        return doses
    right["_input_row"] = np.arange(len(right))
    pairs = pd.merge_asof(
        left.sort_values("_time"),
        right[[key, "_mg", "_input_time", "_input_row"]].sort_values("_input_time"),
        left_on="_time", right_on="_input_time", by=[key, "_mg"],
        direction="nearest", tolerance=DEDUP_WINDOW,
    ).dropna(subset=["_input_row"])
    pairs["_gap"] = (pairs["_time"] - pairs["_input_time"]).abs()
    matched = pairs.sort_values(["_gap", "_row"]).drop_duplicates("_input_row")["_row"]
    return doses.iloc[~np.isin(np.arange(len(doses)), matched)]


def _lookup(data_source: Any, table: str, columns: list, column: str, values: pd.Series) -> pd.DataFrame:
    """Rows of ``table`` whose ``column`` is in ``values``, read in DuckDB.

    ``emar_detail`` (54.7M rows) and ``prescriptions`` are flat parquet tables
    without a bucket layout, and the generic table loader does not push an
    ``IN`` filter on a non-id column into the scan: it materialises the table
    first (above 4 GB resident for emar_detail).  The lookup therefore filters
    inside DuckDB against a registered key list.
    """
    from ..datasource import _get_duckdb_connection, enumerate_bucket_parquet_files

    values = values.dropna()
    if pd.api.types.is_numeric_dtype(values):
        values = values.astype("int64")  # 123.0 and 123 are one key
    keys = pd.DataFrame({"key": pd.unique(values.astype(str))})
    if keys.empty:
        return pd.DataFrame(columns=columns)
    directory = data_source.resolve_flat_parquet_directory(table)
    files = enumerate_bucket_parquet_files(directory) if directory is not None else []
    if not files:
        # The MIMIC-IV demo converts each table to one parquet file.
        resolved = data_source.resolve_loader_from_disk(table)
        if isinstance(resolved, Path) and resolved.is_file() and resolved.suffix.lower() in {".parquet", ".pq"}:
            files = [resolved]
    if not files:
        raise ValueError(f"emar_diuretic_table_unavailable:{table}")
    paths = ", ".join("'" + str(path).replace("'", "''") + "'" for path in files)
    selected = ", ".join(f't."{name}"' for name in columns)
    con = _get_duckdb_connection()
    view = f"_emar_keys_{table}"
    con.register(view, keys)
    try:
        return con.execute(
            f"SELECT {selected} FROM read_parquet([{paths}], union_by_name=true) t "
            f'JOIN {view} k ON CAST(t."{column}" AS VARCHAR) = k.key'
        ).fetchdf()
    finally:
        con.unregister(view)


def apply_emar_furosemide_equivalent(
    frame: pd.DataFrame, *, source: Any, concept_name: str, data_source: Any
) -> pd.DataFrame:
    """Source callback: administered eMAR doses, deduplicated, as mg IV furosemide."""
    from ..datasource import FilterOp, FilterSpec

    params = dict(getattr(source, "params", None) or {})
    drug = str(params.get("fe_drug") or "")
    if drug not in INPUTEVENT_ITEMS:
        raise ValueError(f"emar_diuretic_drug_unknown:{drug}")
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"emar_diuretic_columns_missing:{missing}")
    key = "hadm_id" if "hadm_id" in frame.columns else "stay_id"
    time_col = getattr(source, "index_var", None) or "charttime"
    given = frame[EVENT_COLUMN].astype("string").str.strip().isin(ADMINISTERED_EVENTS)
    frame = frame.loc[given.fillna(False).to_numpy()]
    audit = {"drug": drug, "administered_rows": int(len(frame))}
    if frame.empty or data_source is None:
        if not frame.empty:
            raise ValueError("emar_diuretic_requires_data_source")
        out = frame.assign(**{concept_name: pd.Series(dtype="float64")})
        out.attrs[AUDIT_ATTR] = audit
        return out

    details = _lookup(
        data_source, "emar_detail",
        ["emar_id", "parent_field_ordinal", "dose_given", "dose_given_unit", "route"],
        "emar_id", frame["emar_id"],
    )
    orders = _lookup(
        data_source, "prescriptions", ["pharmacy_id", "route"], "pharmacy_id", frame["pharmacy_id"],
    ).drop_duplicates("pharmacy_id").rename(columns={"route": "_order_route"})
    orders["pharmacy_id"] = pd.to_numeric(orders["pharmacy_id"], errors="coerce")
    doses = frame.join(event_doses(details), on="emar_id")
    doses["pharmacy_id"] = pd.to_numeric(doses["pharmacy_id"], errors="coerce")
    doses = doses.merge(orders, on="pharmacy_id", how="left").set_axis(frame.index)
    doses["_route"] = classify_route(doses["_emar_route"].fillna(doses["_order_route"]))
    audit["route_unknown_rows"] = int(doses["_route"].isna().sum())
    if FUROSEMIDE_EQUIVALENT_FACTORS[(drug, "iv")] != FUROSEMIDE_EQUIVALENT_FACTORS[(drug, "oral")]:
        doses = doses.loc[doses["_route"].notna()]

    items = INPUTEVENT_ITEMS[drug]
    if items and not doses.empty:
        inputs = data_source.load_table(
            "inputevents",
            columns=[key, "itemid", "starttime", "amount", "amountuom", "orderid", "statusdescription"],
            filters=[FilterSpec("itemid", FilterOp.IN, list(items)),
                     FilterSpec(key, FilterOp.IN, pd.unique(doses[key].dropna()).tolist())],
            verbose=False,
        )
        inputs = inputs.data if hasattr(inputs, "data") else inputs
        before = len(doses)
        doses = drop_inputevent_duplicates(doses, inputs, key=key, time_col=time_col, drug=drug)
        audit["inputevents_duplicates_dropped"] = int(before - len(doses))

    # The loader names the medication text after the concept; the dose is
    # ``_dose``, so that text column must not be read as the amount.
    out = furosemide_equivalent(
        doses.drop(columns=[concept_name], errors="ignore"),
        concept_name=concept_name,
        value_var="_dose",
        unit_var="_unit",
        params={"fe_drug": drug, "fe_route_var": "_route", "fe_oral_values": ["oral"]},
    )
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])
    out.attrs[AUDIT_ATTR] = audit
    return out


__all__ = [
    "AUDIT_ATTR",
    "CALLBACK",
    "DEDUP_WINDOW",
    "apply_emar_furosemide_equivalent",
    "classify_route",
    "drop_inputevent_duplicates",
    "event_doses",
]
