"""C565 V2 diagnosis and past-history flags at ICU-stay grain."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..datasource import FilterOp, FilterSpec, MissingTableSourceError
from ..table import ICUTable
from .availability_signal import ConceptAvailabilityRecord, set_table_availability
from .catalog import C565_RENAL_HISTORY_CONCEPTS as FIELDS
from .fio2_set import is_carevue_extraction_root
from .stay_tables import selected_stay_ids

ESRD_ICD10 = frozenset({"N186", "Z992", "Z9115", "I120", "I1311", "I132"})
ESRD_ICD9 = frozenset({"5856", "V4511", "V4512"})
TRANSPLANT_ICD9 = frozenset({"V420", "99681"})
MIMIC_HISTORY_TABLES = ("admissions", "diagnoses_icd")
CAREVUE_WINDOW = "view_admissions_only"
CAREVUE_ERA_NOTE = "prior_window CareVue-era not guaranteed"
CAREVUE_EXTENSION = "EXTENSION beyond frozen C565."
CAREVUE_HISTORY_RECEIPT_COLUMNS = ("prior_window", "prior_window_note")
ABSENCE_REASONS = {
    "aumc": "no_icd_diagnosis_source",
    "hirid": "no_icd_diagnosis_source",
    "sic": "no prior-history diagnosis source",
    "mimic": "carevue_view_required",
    "mimic_demo": "carevue_view_required",
}


def _read(data_source, table, columns, **selections):
    filters = [
        FilterSpec(column, FilterOp.IN, ids)
        for column, ids in selections.items() if ids is not None
    ]
    return data_source.load_table(table, columns=columns, filters=filters or None).data


class _MissingHistoryTable(Exception):
    """A required history input has no readable source mapping."""


def _history(data_source, table, columns, *, optional_columns=(), **selections):
    try:
        frame = _read(data_source, table, [*columns, *optional_columns], **selections)
    except (FileNotFoundError, MissingTableSourceError) as exc:
        raise _MissingHistoryTable(table) from exc
    # The generic MIIV reader can return a zero-row default projection for
    # an absent table. Distinguish that case from a readable empty table.
    if frame.empty and not data_source.has_table_source(table):
        raise _MissingHistoryTable(table)
    present_optional = [column for column in optional_columns if column in frame.columns]
    projection = [*columns, *present_optional]
    # An empty IN selection can skip the disk read and return no columns.
    if frame.empty and set(columns).difference(frame.columns):
        if any(ids is not None and not ids for ids in selections.values()):
            return frame.reindex(columns=projection)
    # The reader can return a wider, case-normalized frame. Keep only the
    # history inputs so unrelated source columns cannot become merge suffixes.
    return frame.loc[:, projection]


def _ids(frame, column):
    return frame[column].dropna().unique().tolist()


def _code_flags(codes, version=None):
    codes = codes.str.upper().str.replace(".", "", regex=False)
    esrd_10 = codes.isin(ESRD_ICD10)
    transplant_10 = codes.eq("Z940") | codes.str.startswith("T861", na=False)
    esrd_9 = codes.isin(ESRD_ICD9)
    transplant_9 = codes.isin(TRANSPLANT_ICD9)
    if version is not None:
        version = pd.to_numeric(version, errors="coerce")
        esrd_10, transplant_10 = esrd_10 & version.eq(10), transplant_10 & version.eq(10)
        esrd_9, transplant_9 = esrd_9 & version.eq(9), transplant_9 & version.eq(9)
    return (esrd_10 | esrd_9).fillna(False), (transplant_10 | transplant_9).fillna(False)


def _mimic_flags(data_source, stay_column, stays, *, carevue_admissions=None):
    admissions = _history(
        data_source, "admissions", ["hadm_id", "subject_id", "admittime"],
        subject_id=_ids(stays, "subject_id"),
    ).dropna(subset=["hadm_id", "subject_id"]).assign(
        admittime=lambda frame: pd.to_datetime(frame["admittime"], errors="coerce"),
    )
    icd9_only = carevue_admissions is not None
    if icd9_only:
        admissions = admissions.loc[admissions["hadm_id"].isin(carevue_admissions)]
    code_column = "icd9_code" if icd9_only else "icd_code"
    code_columns = ["hadm_id", code_column]
    if not icd9_only:
        code_columns.append("icd_version")
    diagnoses = _history(
        data_source, "diagnoses_icd", code_columns,
        hadm_id=_ids(admissions, "hadm_id"),
    )
    history = diagnoses.merge(admissions, on="hadm_id", validate="many_to_one")
    version = pd.Series(9, index=history.index) if icd9_only else history["icd_version"]
    history["esrd_dialysis"], history["kidney_tx"] = _code_flags(
        history[code_column].astype("string"), version,
    )
    indexed = stays.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    # An inclusive cumulative OR is equivalent to checking the earliest
    # matching admission. This avoids a stay-by-admission Cartesian join.
    for field in FIELDS[:2]:
        first_match = history.loc[history[field]].groupby("subject_id")["admittime"].min()
        indexed[field] = indexed["subject_id"].map(first_match).le(indexed["admittime"])
    return indexed.assign(ph_dialysis=False, ph_kidney_tx=False)[[stay_column, *FIELDS[:-1]]]


def _view_root(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    if ("databases", "mimiciii") in zip(parts, parts[1:]):
        return False
    return is_carevue_extraction_root(path) or parts[-2:] == ("mimicview", "raw")


def _view_has_table(data_source, table):
    """Use the view's Parquet inputs, including declared links between view layouts."""
    root = data_source.base_path
    entries = {f"{table}.parquet", table} if table == "icustays" else {f"{table}.parquet"}
    # Check link destinations before the storage owner probes Parquet.
    candidates = set()
    for parent in (root, root / "raw"):
        for entry in entries:
            path = parent / entry
            if path.exists():
                real_parent = path.resolve().parent
                if _view_root(real_parent) or (real_parent.name == "raw" and _view_root(real_parent.parent)):
                    candidates.add(path)
    return bool(candidates) and data_source.resolve_loader_from_disk(table) in candidates


def _carevue_flags(data_source, stay_column, selected):
    stays = _history(
        data_source, "icustays", [stay_column, "hadm_id", "subject_id"], optional_columns=("dbsource",),
    )
    source = stays.get("dbsource", pd.Series(pd.NA, index=stays.index, dtype="string"))
    source = source.astype("string").str.strip().str.lower()
    carevue = source.eq("carevue").fillna(False)
    mv_present = source.isin({"metavision", "both"})
    known = carevue | mv_present
    unknown = ~known
    has_any_mv_stay = stays["hadm_id"].isin(_ids(stays.loc[mv_present], "hadm_id"))
    candidates = carevue if known.any() else unknown
    eligible = stays.loc[candidates & ~has_any_mv_stay]
    prior_admissions = _ids(eligible, "hadm_id")
    if selected is not None:
        eligible = eligible.loc[eligible[stay_column].isin(selected)]
    # A CareVue-only view can establish membership but cannot reveal
    # MetaVision siblings the view has already removed.
    era_note = CAREVUE_ERA_NOTE if unknown.any() or not mv_present.any() else None
    flags = _mimic_flags(data_source, stay_column, eligible, carevue_admissions=prior_admissions)
    return flags, era_note


def add_carevue_history_window(frame, receipt):
    """Publish the availability limitation alongside evaluated stay rows."""
    if receipt is None or receipt.prior_window is None:
        return frame
    evaluated = frame.reindex(columns=FIELDS).notna().any(axis=1)
    frame["prior_window"] = pd.Series(receipt.prior_window, index=frame.index, dtype="string").where(evaluated)
    frame["prior_window_note"] = pd.Series(
        receipt.prior_window_note, index=frame.index, dtype="string",
    ).where(evaluated)
    return frame


def _eicu_flags(data_source, stay_column, selected):
    hospital = "patienthealthsystemstayid"
    stays = _history(data_source, "patient", [stay_column, hospital], **{stay_column: selected})
    siblings = _history(data_source, "patient", [stay_column, hospital], **{hospital: _ids(stays, hospital)})
    scope = {stay_column: _ids(siblings, stay_column)}

    diagnoses = _history(data_source, "diagnosis", [stay_column, "icd9code"], **scope).reset_index(drop=True)
    codes = diagnoses["icd9code"].astype("string").str.split(",").explode().astype("string")
    codes = codes.str.strip().str.replace(r"[^0-9A-Za-z]", "", regex=True)
    esrd, transplant = _code_flags(codes)
    coded = pd.DataFrame({
        stay_column: diagnoses.loc[codes.index, stay_column].to_numpy(),
        "esrd_dialysis": esrd.to_numpy(dtype=bool),
        "kidney_tx": transplant.to_numpy(dtype=bool),
    })
    history = _history(data_source, "pasthistory", [stay_column, "pasthistorypath"], **scope)
    path = history["pasthistorypath"].astype("string").str.lower()
    past = history[[stay_column]].assign(
        ph_dialysis=path.str.contains("renal failure - hemodialysis|renal failure - peritoneal dialysis", na=False),
        ph_kidney_tx=path.str.contains("s/p renal transplant", regex=False, na=False),
    )
    by_stay = siblings.set_index(stay_column)[hospital]
    by_hospital = pd.concat([
        frame.groupby(frame[stay_column].map(by_stay))[list(fields)].any()
        for frame, fields in ((coded, FIELDS[:2]), (past, FIELDS[2:4]))
    ], axis=1)
    return stays.join(by_hospital, on=hospital).drop(columns=hospital)


def renal_history_absence(concept, data_source, missing=(), *, carevue=False):
    """Return a typed empty ID table when the history cannot be evaluated."""
    database = data_source.config.name
    reason = ABSENCE_REASONS.get(database, "no_icd_diagnosis_source")
    if missing:
        reason = "required_source_table_missing"
    stay_column = data_source.config.id_configs["icustay"].id
    result = ICUTable(
        data=pd.DataFrame({stay_column: pd.Series(dtype="int64"), concept: pd.Series(dtype="boolean")}),
        id_columns=[stay_column],
        value_column=concept,
    )
    set_table_availability(result, ConceptAvailabilityRecord(
        concept=concept, database=database, reason="source_unavailable" if missing else "unmapped",
        n_rows=0, missing_tables=missing, evaluation_status="not_evaluable", reason_code=reason,
        note=CAREVUE_EXTENSION if carevue else None,
        prior_window=CAREVUE_WINDOW if carevue else None,
    ))
    return result


def load_renal_history_field(concept, data_source, patient_ids):
    """Load the requested C565 V2 flag on its native stay identity."""
    if concept not in FIELDS:
        raise ValueError(f"Unknown C565 renal history field {concept!r}")
    database = data_source.config.name
    carevue = database == "mimic"
    if database not in {"miiv", "eicu", "eicu_demo", "mimic"}:
        return renal_history_absence(concept, data_source)
    if carevue:
        root = data_source.base_path
        if root is None or not _view_root(root) or not _view_root(root.resolve()):
            return renal_history_absence(concept, data_source)
        required = (*MIMIC_HISTORY_TABLES, "icustays")
        missing = tuple(table for table in required if not _view_has_table(data_source, table))
        if missing:
            return renal_history_absence(concept, data_source, missing=missing, carevue=True)
    stay_column = data_source.config.id_configs["icustay"].id
    selected = selected_stay_ids(patient_ids, stay_column)
    try:
        if carevue:
            flags, era_note = _carevue_flags(data_source, stay_column, selected)
        elif database == "miiv":
            stays = _history(
                data_source, "icustays", [stay_column, "hadm_id", "subject_id"],
                **{stay_column: selected},
            )
            flags = _mimic_flags(data_source, stay_column, stays)
        else:
            flags = _eicu_flags(data_source, stay_column, selected)
    except _MissingHistoryTable as exc:
        return renal_history_absence(concept, data_source, missing=(str(exc),), carevue=carevue)

    components = flags[list(FIELDS[:-1])].astype("boolean").fillna(False)
    components["renal_excl"] = components.any(axis=1).astype("boolean")
    result = ICUTable(
        data=flags[[stay_column]].assign(**{concept: components[concept]}),
        id_columns=[stay_column],
        value_column=concept,
    )
    if carevue:
        set_table_availability(result, ConceptAvailabilityRecord(
            concept=concept, database=database, reason="data_missing" if flags.empty else "mapped_present",
            n_rows=len(flags), sources_defined=("icustays", *MIMIC_HISTORY_TABLES),
            evaluation_status="evaluated", note=CAREVUE_EXTENSION,
            prior_window=CAREVUE_WINDOW, prior_window_note=era_note,
        ))
    return result
