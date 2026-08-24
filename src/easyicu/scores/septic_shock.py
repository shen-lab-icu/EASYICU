"""Evidence-limited Sepsis-3 septic-shock phenotype.

The 2016 consensus definition requires sepsis plus vasopressor therapy needed
to maintain MAP at least 65 mmHg and serum lactate greater than 2 mmol/L
despite adequate volume resuscitation. Retrospective ICU databases do not
reliably encode the indication for a vasopressor or adequacy of resuscitation.
This module therefore returns an operational flag together with explicit
ascertainment receipts; it must not be presented as a complete bedside
diagnosis when the fluid-resuscitation receipt is ``not_observed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


PHENOTYPE_COLUMN = "septic_shock_sepsis3_2016"
EXPECTED_VASOPRESSOR_CONCEPTS = (
    "norepi_rate",
    "epi_rate",
    "dopa_rate",
    "adh_rate",
    "phn_rate",
)

_RECEIPT_COLUMNS = (
    "reason_code",
    "sepsis_ascertainment",
    "lactate_ascertainment",
    "vasopressor_ascertainment",
    "vasopressor_indication_ascertainment",
    "fluid_resuscitation_ascertainment",
    "clinical_definition_complete",
    "lactate_time",
    "lactate_value",
    "vasopressor_time",
    "vasopressor_concept",
    "vasopressor_value",
)


def _empty_result(id_cols: Sequence[str], index_col: str) -> pd.DataFrame:
    result = pd.DataFrame(columns=[*id_cols, index_col, PHENOTYPE_COLUMN, *_RECEIPT_COLUMNS])
    result[PHENOTYPE_COLUMN] = pd.Series(dtype="boolean")
    result["clinical_definition_complete"] = pd.Series(dtype="boolean")
    return result


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _positive(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _time_parameters(
    series: pd.Series,
    shock_window: pd.Timedelta,
    tolerance: pd.Timedelta,
) -> tuple[Any, Any, bool]:
    numeric = pd.api.types.is_numeric_dtype(series)
    if numeric:
        return (
            shock_window.total_seconds() / 3600.0,
            tolerance.total_seconds() / 3600.0,
            True,
        )
    return shock_window, tolerance, False


def _normalise_time(frame: pd.DataFrame, index_col: str, numeric: bool) -> pd.DataFrame:
    result = frame.copy()
    if numeric:
        result[index_col] = pd.to_numeric(result[index_col], errors="coerce")
    else:
        result[index_col] = pd.to_datetime(result[index_col], errors="coerce").dt.tz_localize(None)
    return result.dropna(subset=[index_col])


def _patient_key(row: pd.Series, id_cols: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[column] for column in id_cols)


def _group_by_patient(
    frame: pd.DataFrame,
    id_cols: Sequence[str],
) -> dict[tuple[Any, ...], pd.DataFrame]:
    if frame.empty:
        return {}
    group_key: str | list[str] = id_cols[0] if len(id_cols) == 1 else list(id_cols)
    grouped: dict[tuple[Any, ...], pd.DataFrame] = {}
    for key, patient_frame in frame.groupby(group_key, dropna=False, sort=False):
        tuple_key = key if isinstance(key, tuple) else (key,)
        grouped[tuple_key] = patient_frame
    return grouped


def septic_shock_sepsis3_2016(
    sepsis: pd.DataFrame,
    lactate: pd.DataFrame,
    vasopressors: Mapping[str, pd.DataFrame],
    *,
    id_cols: Sequence[str],
    index_col: str,
    sepsis_col: str = "sep3",
    lactate_col: str = "lact",
    shock_window: pd.Timedelta = pd.Timedelta(hours=24),
    lactate_vasopressor_tolerance: pd.Timedelta = pd.Timedelta(hours=6),
) -> pd.DataFrame:
    """Classify an evidence-limited Sepsis-3 septic-shock phenotype.

    One output row is produced for every input sepsis assessment. A positive
    operational phenotype requires:

    * a positive Sepsis-3 assessment;
    * a positive infusion rate for an eligible vasopressor from sepsis onset
      through 24 hours afterwards; and
    * lactate strictly greater than 2 mmol/L within six hours of that infusion.

    Keys present in ``vasopressors`` declare structurally available drug
    streams, even when their frames are empty. An observed positive infusion
    is sufficient for a positive result. In contrast, absence of infusion is
    a definite negative only when every expected stream is available, or when
    observed lactate is already non-elevated. Otherwise the result is nullable
    Boolean ``NA`` with a stable reason code.

    Adequate volume resuscitation and the treatment indication are not inferred
    from these inputs. The returned receipt therefore marks the complete
    bedside definition as unavailable.
    """

    if not id_cols:
        raise ValueError("id_cols must contain at least one identifier")
    _require_columns(sepsis, [*id_cols, index_col, sepsis_col], "sepsis")
    _require_columns(lactate, [*id_cols, index_col, lactate_col], "lactate")
    if shock_window <= pd.Timedelta(0):
        raise ValueError("shock_window must be positive")
    if lactate_vasopressor_tolerance < pd.Timedelta(0):
        raise ValueError("lactate_vasopressor_tolerance cannot be negative")
    if sepsis.empty:
        return _empty_result(id_cols, index_col)

    window_value, tolerance_value, numeric_time = _time_parameters(
        sepsis[index_col], shock_window, lactate_vasopressor_tolerance
    )
    sepsis_data = _normalise_time(sepsis, index_col, numeric_time)
    lactate_data = _normalise_time(lactate, index_col, numeric_time)
    lactate_data[lactate_col] = pd.to_numeric(lactate_data[lactate_col], errors="coerce")
    lactate_data = lactate_data.dropna(subset=[lactate_col])

    supported = set(vasopressors).intersection(EXPECTED_VASOPRESSOR_CONCEPTS)
    missing_streams = sorted(set(EXPECTED_VASOPRESSOR_CONCEPTS) - supported)
    pressor_parts: list[pd.DataFrame] = []
    for concept in EXPECTED_VASOPRESSOR_CONCEPTS:
        if concept not in vasopressors:
            continue
        frame = vasopressors[concept]
        _require_columns(frame, [*id_cols, index_col, concept], concept)
        part = _normalise_time(frame, index_col, numeric_time)
        part = part[[*id_cols, index_col, concept]].copy()
        part["vasopressor_value"] = pd.to_numeric(part[concept], errors="coerce")
        part["vasopressor_concept"] = concept
        pressor_parts.append(part.drop(columns=[concept]))

    if pressor_parts:
        pressor_data = pd.concat(pressor_parts, ignore_index=True)
        pressor_data = pressor_data[
            pressor_data["vasopressor_value"].notna()
            & pressor_data["vasopressor_value"].gt(0)
        ]
    else:
        pressor_data = pd.DataFrame(
            columns=[*id_cols, index_col, "vasopressor_value", "vasopressor_concept"]
        )
    lactate_by_patient = _group_by_patient(lactate_data, id_cols)
    pressor_by_patient = _group_by_patient(pressor_data, id_cols)

    rows: list[dict[str, Any]] = []
    for _, sepsis_row in sepsis_data.iterrows():
        row: dict[str, Any] = {column: sepsis_row[column] for column in id_cols}
        row[index_col] = sepsis_row[index_col]
        row.update(
            {
                PHENOTYPE_COLUMN: pd.NA,
                "reason_code": "sepsis_not_ascertained",
                "sepsis_ascertainment": "not_ascertained",
                "lactate_ascertainment": "not_assessed",
                "vasopressor_ascertainment": "not_assessed",
                "vasopressor_indication_ascertainment": "not_observed",
                "fluid_resuscitation_ascertainment": "not_observed",
                "clinical_definition_complete": False,
                "lactate_time": pd.NA,
                "lactate_value": pd.NA,
                "vasopressor_time": pd.NA,
                "vasopressor_concept": pd.NA,
                "vasopressor_value": pd.NA,
            }
        )

        sepsis_value = sepsis_row[sepsis_col]
        if pd.isna(sepsis_value):
            rows.append(row)
            continue
        row["sepsis_ascertainment"] = "observed"
        if not _positive(sepsis_value):
            row[PHENOTYPE_COLUMN] = False
            row["reason_code"] = "sepsis_not_present"
            rows.append(row)
            continue

        onset = sepsis_row[index_col]
        patient_key = _patient_key(sepsis_row, id_cols)
        patient_lactate = lactate_by_patient.get(patient_key, lactate_data.iloc[0:0])
        patient_pressor = pressor_by_patient.get(patient_key, pressor_data.iloc[0:0])
        pressor_window = patient_pressor[
            patient_pressor[index_col].ge(onset)
            & patient_pressor[index_col].le(onset + window_value)
        ].sort_values(index_col)
        possible_lactate = patient_lactate[
            patient_lactate[index_col].ge(onset - tolerance_value)
            & patient_lactate[index_col].le(onset + window_value + tolerance_value)
        ].sort_values(index_col)

        if possible_lactate.empty:
            row["lactate_ascertainment"] = "not_observed"
        else:
            row["lactate_ascertainment"] = "observed"

        if pressor_window.empty:
            if missing_streams:
                row["vasopressor_ascertainment"] = "incomplete:" + ",".join(missing_streams)
            else:
                row["vasopressor_ascertainment"] = "complete_no_positive_event"

            if not possible_lactate.empty and not possible_lactate[lactate_col].gt(2.0).any():
                row[PHENOTYPE_COLUMN] = False
                row["reason_code"] = "lactate_not_elevated"
            elif not missing_streams:
                row[PHENOTYPE_COLUMN] = False
                row["reason_code"] = "vasopressor_not_required"
            else:
                row["reason_code"] = (
                    "lactate_and_vasopressor_not_ascertained"
                    if possible_lactate.empty
                    else "vasopressor_not_ascertained"
                )
            rows.append(row)
            continue

        row["vasopressor_ascertainment"] = "positive_direct_evidence"
        first_pressor = pressor_window.iloc[0]
        row["vasopressor_time"] = first_pressor[index_col]
        row["vasopressor_concept"] = first_pressor["vasopressor_concept"]
        row["vasopressor_value"] = first_pressor["vasopressor_value"]
        if possible_lactate.empty:
            row["reason_code"] = "lactate_not_observed"
            rows.append(row)
            continue

        candidates: list[tuple[Any, pd.Series, pd.Series]] = []
        for _, pressor_row in pressor_window.iterrows():
            paired = possible_lactate[
                (possible_lactate[index_col] - pressor_row[index_col]).abs().le(tolerance_value)
                & possible_lactate[lactate_col].gt(2.0)
            ]
            for _, lactate_row in paired.iterrows():
                distance = abs(lactate_row[index_col] - pressor_row[index_col])
                candidates.append((distance, pressor_row, lactate_row))

        if candidates:
            _, matched_pressor, matched_lactate = min(
                candidates,
                key=lambda item: (item[1][index_col], item[0]),
            )
            row[PHENOTYPE_COLUMN] = True
            row["reason_code"] = "criteria_met_fluid_adequacy_unobserved"
            row["vasopressor_time"] = matched_pressor[index_col]
            row["vasopressor_concept"] = matched_pressor["vasopressor_concept"]
            row["vasopressor_value"] = matched_pressor["vasopressor_value"]
            row["lactate_time"] = matched_lactate[index_col]
            row["lactate_value"] = matched_lactate[lactate_col]
        else:
            row[PHENOTYPE_COLUMN] = False
            if possible_lactate[lactate_col].gt(2.0).any():
                row["reason_code"] = "criteria_not_temporally_aligned"
            else:
                row["reason_code"] = "lactate_not_elevated"
            nearest = possible_lactate.iloc[0]
            row["lactate_time"] = nearest[index_col]
            row["lactate_value"] = nearest[lactate_col]
        rows.append(row)

    result = pd.DataFrame(rows, columns=[*id_cols, index_col, PHENOTYPE_COLUMN, *_RECEIPT_COLUMNS])
    result[PHENOTYPE_COLUMN] = result[PHENOTYPE_COLUMN].astype("boolean")
    result["clinical_definition_complete"] = result["clinical_definition_complete"].astype("boolean")
    return result


__all__ = [
    "EXPECTED_VASOPRESSOR_CONCEPTS",
    "PHENOTYPE_COLUMN",
    "septic_shock_sepsis3_2016",
]
