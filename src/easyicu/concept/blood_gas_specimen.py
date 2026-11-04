"""Arterial-specimen restriction for blood-gas concept sources.

The generic blood-gas concepts (``po2``, ``pco2``, ``ph``, ``sao2``) mean
*arterial* values: P/F, oxygenation index and the SOFA respiratory scores are
defined on PaO2.  Several databases store arterial, venous, mixed-venous and
capillary gases under the same analyte item and record the sample type in a
separate specimen record.  This module owns that specimen rule; the concept
dictionary opts a source in with the callback ``arterial_specimen_only``,
optionally wrapping the source's own value callback::

    "callback": "arterial_specimen_only"
    "callback": "arterial_specimen_only(convert_unit(...))"

Per-database evidence (see :data:`SPECIMEN_RULES`):

* MIMIC-IV: lab item 52033 "Specimen Type" with the same ``specimen_id``
  (mimic-code ``bg.sql``/``first_day_bg_art.sql``: ``specimen = 'ART.'``).
  A gas without a specimen record is dropped, as in mimic-code.
* MIMIC-III: lab item 50800 "SPECIMEN TYPE" with the same subject and
  ``charttime`` (MIMIC-III has no specimen id; mimic-code ``pivoted_bg``
  groups the panel by admission and ``charttime``).  A gas without a specimen
  record is kept, as in AUMCdb: about a quarter of 50821 rows have none, they
  are mostly arterial (median PO2 108 mmHg on the full database), and dropping
  them removes PaO2 from about 30% of stays.  mimic-code instead keeps them
  when its logistic specimen-prediction model says arterial, which is not
  reproduced here; recorded venous, mixed and conflicting samples are dropped.
* AUMCdb: freetext item 11646 "Afname (bloed)" at the same admission and
  ``measuredat``.  A gas without a specimen record is kept (most AUMC gases
  are arterial and the specimen item is not charted for every gas); every
  recorded non-arterial value, including laboratory status codes such as
  "Vervalt" or "Gestold", drops the gas.  Gases of the 2003-2008 era record
  the site in list item 7432 "Afnameplaats" instead (``Arterieel``, ``PAP``,
  ``Centrale lijn``, ``Perifeer veneus``; 11646 is on 1.65% of those gases,
  7432 on 96.4%), read at the same keys (audit round 2, G2).
* SICdb: the laboratory row's own ``LaboratoryType``; 2296 is "Arterial" in
  ``d_references``.  Rows with an empty type are dropped, except for cases
  admitted in 2013: the type was not yet recorded then, and those gases are
  arterial (60,882 PO2 rows, median 82 mmHg, against 33-41 mmHg for untyped
  rows from 2015).

A sample is arterial only when every non-empty specimen value recorded for it
is one of the database's arterial codes after trimming and upper-casing
(MIMIC-IV ``ART.``, MIMIC-III ``ART``, AUMCdb ``ART.``/``Art.``).  Conflicting
records (``ART.`` plus ``VEN.``) are not arterial.  Venous values are excluded,
not reassigned to venous concepts: the venous site is not recorded.

The same rule selects *mixed-venous* samples for ``svo2`` with the callback
``mixed_venous_specimen_only`` where a database names that specimen
(:data:`MIXED_VENOUS_TOKENS`; AUMCdb ``GEM.VEN.``/``Gem. ven.``, "gemengd
veneus", median saturation 67%).  Only recorded mixed-venous samples are
kept: a gas without a specimen record is mostly arterial.  Plain venous
samples (``Ven.``) are not ScvO2: the site (central or peripheral) is not
recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTERIAL_SPECIMEN_CALLBACK = "arterial_specimen_only"
MIXED_VENOUS_SPECIMEN_CALLBACK = "mixed_venous_specimen_only"
AUDIT_ATTR = "arterial_specimen_audit"
MIXED_VENOUS_AUDIT_ATTR = "mixed_venous_specimen_audit"


class ArterialSpecimenContractError(ValueError):
    """A source opted into the arterial rule but cannot be evaluated."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class SpecimenRecordRule:
    """Specimen type stored as a separate record joined on ``keys``."""

    table: str
    item_column: str
    item_id: int
    keys: Tuple[str, ...]
    patient_column: str
    value_column: str
    arterial_tokens: frozenset
    keep_unrecorded: bool
    #: Further specimen records at the same keys (another item or table).
    #: Their values are classified with the same token sets.
    also: Tuple["SpecimenRecordRule", ...] = ()


@dataclass(frozen=True)
class SpecimenColumnRule:
    """Specimen type stored on the measurement row itself."""

    column: str
    arterial_codes: frozenset
    #: Admission years whose untyped rows are kept: the type was not recorded
    #: yet (audit round 2, G1).  Read from ``year_table.year_column`` by case.
    untyped_kept_years: frozenset = frozenset()
    year_table: str = ""
    year_column: str = ""
    case_column: str = ""


SPECIMEN_RULES = {
    "miiv": SpecimenRecordRule(
        table="labevents", item_column="itemid", item_id=52033,
        keys=("specimen_id",), patient_column="subject_id",
        value_column="value", arterial_tokens=frozenset({"ART."}),
        keep_unrecorded=False,
    ),
    # MIMIC-III gases without a 50800 specimen record were kept, and about
    # 28% of them are venous-like; only confirmed arterial gases are kept, as
    # for MIMIC-IV (2026-11-04 audit round 4, R4-020/R4-021).
    "mimic": SpecimenRecordRule(
        table="labevents", item_column="itemid", item_id=50800,
        keys=("subject_id", "charttime"), patient_column="subject_id",
        value_column="value", arterial_tokens=frozenset({"ART"}),
        keep_unrecorded=False,
    ),
    "aumc": SpecimenRecordRule(
        table="freetextitems", item_column="itemid", item_id=11646,
        keys=("admissionid", "measuredat"), patient_column="admissionid",
        value_column="value", arterial_tokens=frozenset({"ART.", "ARTERIEEL"}),
        keep_unrecorded=True,
        also=(SpecimenRecordRule(
            table="listitems", item_column="itemid", item_id=7432,
            keys=("admissionid", "measuredat"), patient_column="admissionid",
            value_column="value", arterial_tokens=frozenset({"ARTERIEEL"}),
            keep_unrecorded=True,
        ),),
    ),
    "sic": SpecimenColumnRule(
        column="LaboratoryType", arterial_codes=frozenset({2296}),
        untyped_kept_years=frozenset({2013}), year_table="cases",
        year_column="AdmissionYear", case_column="CaseID",
    ),
}
SPECIMEN_RULES["mimic_demo"] = SPECIMEN_RULES["mimic"]

# Recorded specimen values that name a mixed-venous (pulmonary artery) sample,
# after trimming and upper-casing.  AUMCdb freetext 11646 (2026-11-03 audit 2:
# 'GEM.VEN.' 4,900 and 'Gem. ven.' 3,526 O2-saturation samples, median 0.67)
# and list item 7432 'PAP' (pulmonary artery, 2003-2008 era).
MIXED_VENOUS_TOKENS = {
    "aumc": frozenset({"GEM.VEN.", "GEM. VEN.", "GEM.VEN", "PAP"}),
}


def _parse_wrapper(expr: str, name: str) -> Tuple[bool, Optional[str]]:
    text = (expr or "").strip()
    if text == name:
        return True, None
    prefix = name + "("
    if text.startswith(prefix) and text.endswith(")"):
        inner = text[len(prefix):-1].strip()
        return True, inner or None
    return False, None


def parse_arterial_specimen_callback(expr: str) -> Tuple[bool, Optional[str]]:
    """Return ``(applies, inner_callback)`` for a dictionary callback string."""

    return _parse_wrapper(expr, ARTERIAL_SPECIMEN_CALLBACK)


def parse_mixed_venous_specimen_callback(expr: str) -> Tuple[bool, Optional[str]]:
    """``(applies, inner_callback)`` for ``mixed_venous_specimen_only[(...)]``."""

    return _parse_wrapper(expr, MIXED_VENOUS_SPECIMEN_CALLBACK)


def normalize_specimen_token(value: object) -> str:
    """Normalise one recorded specimen value (``' Art. '`` -> ``'ART.'``)."""

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def is_arterial_specimen(
    tokens: Iterable[object], arterial_tokens: frozenset
) -> Optional[bool]:
    """Classify the specimen values recorded for one sample.

    ``None`` when nothing non-empty is recorded, ``True`` when every recorded
    value is one of ``arterial_tokens``, ``False`` otherwise (non-arterial or
    conflicting).  Unknown spellings are never promoted to arterial.
    """

    recorded = {normalize_specimen_token(token) for token in tokens} - {""}
    if not recorded:
        return None
    return recorded <= arterial_tokens


def _normalized_tokens(values: pd.Series) -> pd.Series:
    return values.map(normalize_specimen_token).astype(object)


def _record_status(
    specimens: pd.DataFrame,
    rule: SpecimenRecordRule,
    accepted_tokens: Optional[frozenset] = None,
) -> pd.DataFrame:
    """One row per specimen key with ``recorded`` and ``arterial`` flags.

    ``arterial`` means every recorded value is in ``accepted_tokens``
    (default: the rule's arterial tokens).
    """

    accepted = rule.arterial_tokens if accepted_tokens is None else accepted_tokens
    tokens = _normalized_tokens(specimens[rule.value_column])
    recorded = tokens.ne("")
    frame = specimens.loc[:, list(rule.keys)].copy()
    frame["_recorded"] = recorded.to_numpy()
    frame["_non_arterial"] = (
        recorded & ~tokens.isin(accepted)
    ).to_numpy()
    grouped = frame.groupby(list(rule.keys), dropna=True, sort=False)
    status = grouped[["_recorded", "_non_arterial"]].any().reset_index()
    status["_arterial"] = status["_recorded"] & ~status["_non_arterial"]
    return status.loc[:, [*rule.keys, "_recorded", "_arterial"]]


def _load_specimen_records(frame, rule: SpecimenRecordRule, data_source) -> pd.DataFrame:
    from ..datasource import FilterOp, FilterSpec

    patients = pd.unique(frame[rule.patient_column].dropna())
    columns = list(dict.fromkeys(
        [rule.patient_column, *rule.keys, rule.item_column, rule.value_column]
    ))
    if len(patients) == 0:
        return pd.DataFrame(columns=columns)
    table = data_source.load_table(
        rule.table,
        columns=columns,
        filters=[
            FilterSpec(rule.item_column, FilterOp.IN, [rule.item_id]),
            FilterSpec(rule.patient_column, FilterOp.IN, list(patients)),
        ],
        verbose=False,
    )
    records = table.data if hasattr(table, "data") else table
    missing = [c for c in (*rule.keys, rule.item_column, rule.value_column) if c not in records.columns]
    if missing:
        raise ArterialSpecimenContractError(
            "arterial_specimen_table_columns_missing",
            f"{rule.table} lacks {missing}",
        )
    return records.loc[records[rule.item_column].eq(rule.item_id)]


def _align_key_dtypes(frame: pd.DataFrame, status: pd.DataFrame, keys) -> pd.DataFrame:
    status = status.copy()
    for key in keys:
        left = frame[key]
        if pd.api.types.is_datetime64_any_dtype(left):
            status[key] = pd.to_datetime(status[key], errors="coerce").astype(left.dtype)
        elif pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_numeric_dtype(status[key]):
            status[key] = pd.to_numeric(status[key], errors="coerce")
    return status


def _restrict_by_records(frame, rule: SpecimenRecordRule, data_source, accepted_tokens=None):
    if data_source is None:
        raise ArterialSpecimenContractError("arterial_specimen_data_source_missing")
    records = pd.concat(
        [_load_specimen_records(frame, source, data_source) for source in (rule, *rule.also)],
        ignore_index=True,
    )
    status = _record_status(records, rule, accepted_tokens)
    status = _align_key_dtypes(frame, status, rule.keys)
    keyed = frame.loc[:, list(rule.keys)].reset_index(drop=True)
    joined = keyed.merge(status, on=list(rule.keys), how="left", sort=False)
    if len(joined) != len(frame):  # status is unique per key; guard the join.
        raise ArterialSpecimenContractError("arterial_specimen_join_not_one_to_one")
    recorded = joined["_recorded"].fillna(False).astype(bool).to_numpy()
    arterial = joined["_arterial"].fillna(False).astype(bool).to_numpy()
    keep = arterial | (~recorded & rule.keep_unrecorded)
    audit = {
        "rows": int(len(frame)),
        "arterial_rows": int(arterial.sum()),
        "non_arterial_rows_dropped": int((recorded & ~arterial).sum()),
        "unrecorded_rows": int((~recorded).sum()),
        "unrecorded_rows_kept": bool(rule.keep_unrecorded),
    }
    return frame.loc[keep], audit


def _untyped_kept(frame, rule: SpecimenColumnRule, data_source) -> np.ndarray:
    """Rows whose case was admitted in a year before the type was recorded."""

    keep = np.zeros(len(frame), dtype=bool)
    if not rule.untyped_kept_years or rule.case_column not in frame.columns:
        return keep
    try:
        table = data_source.load_table(
            rule.year_table, columns=[rule.case_column, rule.year_column], verbose=False
        )
    except Exception as exc:
        raise ArterialSpecimenContractError(
            "arterial_specimen_year_unavailable",
            f"{rule.year_table}.{rule.year_column}: {type(exc).__name__}",
        ) from exc
    years = getattr(table, "data", table)
    if rule.year_column not in getattr(years, "columns", ()):
        # Without the admission year the strict rule stands: untyped rows drop.
        return keep
    by_case = years.drop_duplicates(subset=[rule.case_column], keep="last").set_index(
        rule.case_column
    )[rule.year_column]
    year = pd.to_numeric(frame[rule.case_column].map(by_case), errors="coerce")
    return year.isin(rule.untyped_kept_years).to_numpy()


def _restrict_by_column(frame, rule: SpecimenColumnRule, data_source=None):
    codes = pd.to_numeric(frame[rule.column], errors="coerce")
    arterial = codes.isin(rule.arterial_codes).to_numpy()
    untyped = codes.isna().to_numpy()
    kept_untyped = untyped & _untyped_kept(frame, rule, data_source) if untyped.any() else untyped
    audit = {
        "rows": int(len(frame)),
        "arterial_rows": int(arterial.sum()),
        "non_arterial_rows_dropped": int((codes.notna().to_numpy() & ~arterial).sum()),
        "unrecorded_rows": int(untyped.sum()),
        "unrecorded_rows_kept": int(kept_untyped.sum()),
    }
    return frame.loc[arterial | kept_untyped], audit


def restrict_to_arterial_specimens(
    frame: pd.DataFrame,
    *,
    data_source,
    concept_name: str,
) -> pd.DataFrame:
    """Keep only rows whose recorded specimen is arterial (see module doc)."""

    database = getattr(getattr(data_source, "config", None), "name", None)
    rule = SPECIMEN_RULES.get(database)
    if rule is None:
        raise ArterialSpecimenContractError(
            "arterial_specimen_rule_missing", f"database={database!r}"
        )
    required = rule.keys if isinstance(rule, SpecimenRecordRule) else (rule.column,)
    if isinstance(rule, SpecimenRecordRule):
        required = (*required, rule.patient_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ArterialSpecimenContractError(
            "arterial_specimen_keys_missing",
            f"concept={concept_name} database={database} missing={missing}; "
            "declare them in the source's extra_vars",
        )
    if frame.empty:
        return frame
    attrs = dict(frame.attrs)
    if isinstance(rule, SpecimenRecordRule):
        kept, audit = _restrict_by_records(frame, rule, data_source)
    else:
        kept, audit = _restrict_by_column(frame, rule, data_source)
    kept = kept.copy()
    kept.attrs.update(attrs)
    kept.attrs[AUDIT_ATTR] = audit
    logger.info(
        "Arterial specimen restriction concept=%s database=%s audit=%s",
        concept_name, database, audit,
    )
    return kept


def restrict_to_mixed_venous_specimens(
    frame: pd.DataFrame,
    *,
    data_source,
    concept_name: str,
) -> pd.DataFrame:
    """Keep only rows whose recorded specimen is mixed venous (see module doc).

    Unlike the arterial rule, a gas without a specimen record is dropped.
    """

    database = getattr(getattr(data_source, "config", None), "name", None)
    rule = SPECIMEN_RULES.get(database)
    tokens = MIXED_VENOUS_TOKENS.get(database)
    if not isinstance(rule, SpecimenRecordRule) or tokens is None:
        raise ArterialSpecimenContractError(
            "mixed_venous_specimen_rule_missing", f"database={database!r}"
        )
    missing = [c for c in (*rule.keys, rule.patient_column) if c not in frame.columns]
    if missing:
        raise ArterialSpecimenContractError(
            "mixed_venous_specimen_keys_missing",
            f"concept={concept_name} database={database} missing={missing}",
        )
    if frame.empty:
        return frame
    attrs = dict(frame.attrs)
    unrecorded_policy = replace(rule, keep_unrecorded=False)
    kept, audit = _restrict_by_records(frame, unrecorded_policy, data_source, tokens)
    audit = {
        "rows": audit["rows"],
        "mixed_venous_rows": audit["arterial_rows"],
        "other_specimen_rows_dropped": audit["non_arterial_rows_dropped"],
        "unrecorded_rows_dropped": audit["unrecorded_rows"],
    }
    kept = kept.copy()
    kept.attrs.update(attrs)
    kept.attrs[MIXED_VENOUS_AUDIT_ATTR] = audit
    logger.info(
        "Mixed-venous specimen restriction concept=%s database=%s audit=%s",
        concept_name, database, audit,
    )
    return kept


__all__ = [
    "ARTERIAL_SPECIMEN_CALLBACK",
    "MIXED_VENOUS_SPECIMEN_CALLBACK",
    "MIXED_VENOUS_TOKENS",
    "ArterialSpecimenContractError",
    "SPECIMEN_RULES",
    "SpecimenColumnRule",
    "SpecimenRecordRule",
    "is_arterial_specimen",
    "normalize_specimen_token",
    "parse_arterial_specimen_callback",
    "parse_mixed_venous_specimen_callback",
    "restrict_to_arterial_specimens",
    "restrict_to_mixed_venous_specimens",
]
