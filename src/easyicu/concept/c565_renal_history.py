"""C565 V2 diagnosis and past-history flags at ICU-stay grain."""

from __future__ import annotations

import pandas as pd

from ..datasource import FilterOp, FilterSpec, MissingTableSourceError
from ..table import ICUTable
from .availability_signal import ConceptAvailabilityRecord, set_table_availability
from .catalog import C565_RENAL_HISTORY_CONCEPTS as FIELDS
from .stay_tables import selected_stay_ids

ESRD_ICD10 = frozenset({"N186", "Z992", "Z9115", "I120", "I1311", "I132"})
ESRD_ICD9 = frozenset({"5856", "V4511", "V4512"})
TRANSPLANT_ICD9 = frozenset({"V420", "99681"})
MIMIC_HISTORY_TABLES = ("admissions", "diagnoses_icd")
ABSENCE_REASONS = {
    "aumc": "no_icd_diagnosis_source",
    "hirid": "no_icd_diagnosis_source",
    "sic": "single_principal_diagnosis_only",
    "mimic": "carevue_diagnosis_history_not_guaranteed",
    "mimic_demo": "carevue_diagnosis_history_not_guaranteed",
}


def _read(data_source, table, columns, **selections):
    filters = [
        FilterSpec(column, FilterOp.IN, ids)
        for column, ids in selections.items() if ids is not None
    ]
    return data_source.load_table(table, columns=columns, filters=filters or None).data


class _MissingHistoryTable(Exception):
    """A required history input has no readable source mapping."""


def _history(data_source, table, columns, **selections):
    try:
        frame = _read(data_source, table, columns, **selections)
    except (FileNotFoundError, MissingTableSourceError) as exc:
        raise _MissingHistoryTable(table) from exc
    # The generic MIIV reader can return a zero-row default projection for
    # an absent table. Distinguish that case from a readable empty table.
    if frame.empty and set(columns).difference(frame.columns):
        if data_source.resolve_loader_from_disk(table) is None:
            raise _MissingHistoryTable(table)
    return frame


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


def _miiv_flags(data_source, stay_column, selected):
    stays = _history(data_source, "icustays", [stay_column, "hadm_id", "subject_id"], **{stay_column: selected})
    admissions = _history(
        data_source, "admissions", ["hadm_id", "subject_id", "admittime"],
        subject_id=_ids(stays, "subject_id"),
    ).dropna(subset=["hadm_id", "subject_id"]).assign(
        admittime=lambda frame: pd.to_datetime(frame["admittime"], errors="coerce"),
    )
    diagnoses = _history(
        data_source, "diagnoses_icd", ["hadm_id", "icd_code", "icd_version"],
        hadm_id=_ids(admissions, "hadm_id"),
    )
    history = diagnoses.merge(admissions, on="hadm_id", validate="many_to_one")
    history["esrd_dialysis"], history["kidney_tx"] = _code_flags(
        history["icd_code"].astype("string"), history["icd_version"],
    )
    indexed = stays.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="many_to_one")
    # An inclusive cumulative OR is equivalent to checking the earliest
    # matching admission. This avoids a stay-by-admission Cartesian join.
    for field in FIELDS[:2]:
        first_match = history.loc[history[field]].groupby("subject_id")["admittime"].min()
        indexed[field] = indexed["subject_id"].map(first_match).le(indexed["admittime"])
    return indexed.assign(ph_dialysis=False, ph_kidney_tx=False)[[stay_column, *FIELDS[:-1]]]


def _eicu_flags(data_source, stay_column, selected):
    hospital = "patienthealthsystemstayid"
    stays = _history(data_source, "patient", [stay_column, hospital], **{stay_column: selected})
    siblings = _history(data_source, "patient", [stay_column, hospital], **{hospital: _ids(stays, hospital)})
    scope = {stay_column: _ids(siblings, stay_column)}

    diagnoses = _history(data_source, "diagnosis", [stay_column, "icd9code"], **scope).reset_index(drop=True)
    codes = diagnoses["icd9code"].astype("string").str.split(",").explode()
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


def renal_history_absence(concept, data_source, missing=()):
    """Return a typed empty ID table when the history cannot be evaluated."""
    database = data_source.config.name
    reason = ABSENCE_REASONS.get(database, "no_icd_diagnosis_source")
    if database == "mimic":
        missing = tuple(
            table for table in MIMIC_HISTORY_TABLES
            if data_source.resolve_loader_from_disk(table) is None
        )
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
    ))
    return result


def load_renal_history_field(concept, data_source, patient_ids):
    """Load the requested C565 V2 flag on its native stay identity."""
    if concept not in FIELDS:
        raise ValueError(f"Unknown C565 renal history field {concept!r}")
    database = data_source.config.name
    if database not in {"miiv", "eicu", "eicu_demo"}:
        return renal_history_absence(concept, data_source)
    stay_column = data_source.config.id_configs["icustay"].id
    selected = selected_stay_ids(patient_ids, stay_column)
    try:
        reader = _miiv_flags if database == "miiv" else _eicu_flags
        flags = reader(data_source, stay_column, selected)
    except _MissingHistoryTable as exc:
        return renal_history_absence(concept, data_source, missing=(str(exc),))

    components = flags[list(FIELDS[:-1])].astype("boolean").fillna(False)
    components["renal_excl"] = components.any(axis=1).astype("boolean")
    return ICUTable(
        data=flags[[stay_column]].assign(**{concept: components[concept]}),
        id_columns=[stay_column],
        value_column=concept,
    )
