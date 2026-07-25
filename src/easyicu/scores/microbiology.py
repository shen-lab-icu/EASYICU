"""Microbiology culture-positivity endpoints.

Per-ICU-stay logical flags derived from the culture tables:

* ``culture_positive`` — any positive culture (an organism was isolated)
  during the admission.
* ``bld_culture_positive`` — a positive **blood** culture (bacteraemia /
  fungaemia signal), the variant most used in sepsis work.

MIMIC-III/IV expose ``microbiologyevents`` (organism in ``org_name``,
specimen in ``spec_type_desc``; keyed by ``hadm_id`` -> mapped to ICU
stays). eICU exposes ``microlab`` (``organism`` / ``culturesite``,
already per ICU stay; ``'no growth'`` marks a negative result). SICdb,
AmsterdamUMCdb and HiRID carry no structured culture table -> empty.

Same per-database loader pattern as :mod:`easyicu.comorbidity`.
"""

from __future__ import annotations

import pandas as pd

from ..datasource import FilterOp, FilterSpec
from .comorbidity import _build_datasource, _lower_cols, _table_df

# eICU 'organism' values that denote a NEGATIVE culture, not an isolate.
_EICU_NEGATIVE = {"no growth", "no growth on culture", "", "none"}
_NO_MICRO_DATABASES = {"sic", "aumc", "hirid"}

_STAY_ID_COL = {
    "miiv": "stay_id",
    "miiv_demo": "stay_id",
    "mimic": "icustay_id",
    "mimic_demo": "icustay_id",
    "eicu": "patientunitstayid",
    "eicu_demo": "patientunitstayid",
}


def _patient_values(patient_ids):
    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return []
        patient_ids = next(iter(patient_ids.values()))
    return list(patient_ids)


def load_microbiology(
    database: str,
    data_path: object = None,
    *,
    patient_ids=None,
    max_patients: object = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load per-ICU-stay culture-positivity flags for a database."""
    db = database.lower()
    patient_values = _patient_values(patient_ids)
    if patient_values == []:
        return pd.DataFrame()
    if db in _NO_MICRO_DATABASES:
        if verbose:
            print(f"[microbiology] {database} ships no structured culture table — N/A")
        return pd.DataFrame()

    ds = _build_datasource(database, data_path)

    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        stay_col = _STAY_ID_COL[db]
        stays = _lower_cols(_table_df(ds, "icustays"))
        if patient_values is not None:
            stays = stays[stays[stay_col].isin(patient_values)].copy()
            selected_hadm_ids = stays["hadm_id"].dropna().unique().tolist()
            if not selected_hadm_ids:
                return pd.DataFrame(
                    columns=[stay_col, "culture_positive", "bld_culture_positive"]
                )
            # ``microbiologyevents`` is a large hospital table.  Filtering it
            # only after materialising the entire table made a 50-stay export
            # allocate several GB and spill to the Mac system disk.  Translate
            # the requested ICU stays to admissions first, then let the
            # datasource push the admission filter into Parquet/DuckDB.
            mb = _lower_cols(
                _table_df(
                    ds,
                    "microbiologyevents",
                    filters=[
                        FilterSpec(
                            column="hadm_id",
                            op=FilterOp.IN,
                            value=selected_hadm_ids,
                        )
                    ],
                )
            )
        else:
            mb = _lower_cols(_table_df(ds, "microbiologyevents"))
        org_col = "org_name" if "org_name" in mb.columns else "org_itemid"
        positive = mb[org_col].notna()
        if mb[org_col].dtype == object:
            positive &= mb[org_col].astype(str).str.strip().ne("")
        spec = mb.get("spec_type_desc", pd.Series("", index=mb.index)).astype(str)
        is_blood = spec.str.contains("blood", case=False, na=False)
        per_hadm = pd.DataFrame(
            {
                "hadm_id": mb["hadm_id"],
                "culture_positive": positive.values,
                "bld_culture_positive": (positive & is_blood).values,
            }
        )
        flags = per_hadm.groupby("hadm_id").any().reset_index()
        stay_col = "stay_id" if "stay_id" in stays.columns else "icustay_id"
        stays = stays[["hadm_id", stay_col]]
        out = stays.merge(flags, on="hadm_id", how="left").drop(columns="hadm_id")
        out["culture_positive"] = out["culture_positive"].fillna(False)
        out["bld_culture_positive"] = out["bld_culture_positive"].fillna(False)

    elif db in ("eicu", "eicu_demo"):
        stay_col = "patientunitstayid"
        if patient_values is not None:
            mb = _lower_cols(
                _table_df(
                    ds,
                    "microlab",
                    filters=[
                        FilterSpec(
                            column=stay_col,
                            op=FilterOp.IN,
                            value=patient_values,
                        )
                    ],
                )
            )
        else:
            mb = _lower_cols(_table_df(ds, "microlab"))
        org = mb["organism"].astype(str).str.strip().str.lower()
        positive = mb["organism"].notna() & ~org.isin(_EICU_NEGATIVE)
        site = mb.get("culturesite", pd.Series("", index=mb.index)).astype(str)
        is_blood = site.str.contains("blood", case=False, na=False)
        per_stay = pd.DataFrame(
            {
                "patientunitstayid": mb["patientunitstayid"],
                "culture_positive": positive.values,
                "bld_culture_positive": (positive & is_blood).values,
            }
        )
        flags = per_stay.groupby("patientunitstayid").any().reset_index()
        stays = _lower_cols(_table_df(ds, "patient"))[
            ["patientunitstayid"]
        ].drop_duplicates()
        if patient_values is not None:
            stays = stays[stays[stay_col].isin(patient_values)]
        out = stays.merge(flags, on="patientunitstayid", how="left")
        out["culture_positive"] = out["culture_positive"].fillna(False)
        out["bld_culture_positive"] = out["bld_culture_positive"].fillna(False)

    else:  # pragma: no cover
        return pd.DataFrame()

    for c in ("culture_positive", "bld_culture_positive"):
        out[c] = out[c].astype("boolean")

    if patient_values is not None:
        stay_col = _STAY_ID_COL.get(db)
        if stay_col and stay_col in out.columns:
            out = out[out[stay_col].isin(patient_values)]
    return out.reset_index(drop=True)


__all__ = ["load_microbiology"]
