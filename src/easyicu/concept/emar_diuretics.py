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

The same ICU dose can be charted in both eMAR and inputevents.  An eMAR dose
is dropped only when the channels share an event identifier, IV route,
matching mass dose and a start within ``DEDUP_WINDOW``. Time proximity alone
cannot distinguish a copied administration from a second dose; ambiguous
records remain, including every oral dose.

Dictionary use::

    "table": "emar", "sub_var": "medication", "regex": "...",
    "class_name": "rgx_itm",
    "extra_vars": ["event_txt", "emar_id", "pharmacy_id"],
    "callback": "emar_furosemide_equivalent", "fe_drug": "furosemide"
"""

from __future__ import annotations

from typing import Any, Mapping

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
    out[text.str.contains("IV", na=False).to_numpy()] = "iv"
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
    doses: pd.DataFrame, inputs: pd.DataFrame, *, key: str, time_col: str
) -> pd.DataFrame:
    """Drop only verifiable copies, comparing event identity, route and mass dose."""
    if doses.empty or inputs.empty:
        return doses
    identities = [c for c in ("emar_id", "orderid", "poe_id") if c in doses and c in inputs]
    required = {"_route", "_dose", "_unit"}
    if not identities or not required.issubset(doses) or not {"amount", "amountuom"}.issubset(inputs):
        return doses
    factors = {"mg": 1., "g": 1000., "gm": 1000., "mcg": .001, "ug": .001, "µg": .001}
    left = doses.copy()
    left["_row"] = range(len(left))
    left["_mg"] = pd.to_numeric(left["_dose"], errors="coerce") * left["_unit"].astype("string").str.casefold().map(factors)
    left = left.loc[left["_route"].eq("iv")]
    right = inputs.copy()
    if "statusdescription" in right:
        right = right.loc[~right.statusdescription.astype("string").str.strip().str.casefold().eq("rewritten").fillna(False)]
    right["_input_mg"] = pd.to_numeric(right["amount"], errors="coerce") * right["amountuom"].astype("string").str.casefold().map(factors)
    duplicated = pd.Series(False, index=pd.RangeIndex(len(doses)))
    for identity in identities:
        a, b = left[identity].astype("string"), right[identity].astype("string")
        # Join on the administration identifier as well as the admission:
        # avoid a quadratic all-dose-pairs join for long ICU admissions.
        left["__event_id"], right["__event_id"] = a, b
        candidates = left.loc[a.notna() & a.str.strip().ne(""), [key, time_col, "_row", "_mg", "__event_id"]]
        evidence = right.loc[b.notna() & b.str.strip().ne(""), [key, "starttime", "_input_mg", "__event_id"]]
        pairs = candidates.merge(evidence, on=[key, "__event_id"])
        gap = (pd.to_datetime(pairs[time_col]) - pd.to_datetime(pairs["starttime"])).abs()
        same_dose = (pairs["_mg"] - pairs["_input_mg"]).abs().le(1e-6) & pairs["_mg"].gt(0)
        duplicated.loc[pairs.loc[same_dose & gap.le(DEDUP_WINDOW), "_row"]] = True
    return doses.iloc[~duplicated.to_numpy()]


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
    if FUROSEMIDE_EQUIVALENT_FACTORS[(drug, "iv")] != FUROSEMIDE_EQUIVALENT_FACTORS[(drug, "oral")]:
        audit["route_unknown_rows"] = int(doses["_route"].isna().sum())
        doses = doses.loc[doses["_route"].notna()]
    else:
        doses["_route"] = doses["_route"].fillna("iv")

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
        doses = drop_inputevent_duplicates(doses, inputs, key=key, time_col=time_col)
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
