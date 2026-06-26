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

from .comorbidity import _build_datasource, _lower_cols, _table_df

# eICU 'organism' values that denote a NEGATIVE culture, not an isolate.
_EICU_NEGATIVE = {"no growth", "no growth on culture", "", "none"}
_NO_MICRO_DATABASES = {"sic", "aumc", "hirid"}

_STAY_ID_COL = {
    "miiv": "stay_id", "miiv_demo": "stay_id",
    "mimic": "icustay_id", "mimic_demo": "icustay_id",
    "eicu": "patientunitstayid", "eicu_demo": "patientunitstayid",
}


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
    if db in _NO_MICRO_DATABASES:
        if verbose:
            print(f"[microbiology] {database} ships no structured culture table — N/A")
        return pd.DataFrame()

    ds = _build_datasource(database, data_path)

    if db in ("miiv", "miiv_demo", "mimic", "mimic_demo"):
        mb = _lower_cols(_table_df(ds, "microbiologyevents"))
        org_col = "org_name" if "org_name" in mb.columns else "org_itemid"
        positive = mb[org_col].notna()
        if mb[org_col].dtype == object:
            positive &= mb[org_col].astype(str).str.strip().ne("")
        spec = mb.get("spec_type_desc", pd.Series("", index=mb.index)).astype(str)
        is_blood = spec.str.contains("blood", case=False, na=False)
        per_hadm = pd.DataFrame({
            "hadm_id": mb["hadm_id"],
            "culture_positive": positive.values,
            "bld_culture_positive": (positive & is_blood).values,
        })
        flags = per_hadm.groupby("hadm_id").any().reset_index()
        stays = _lower_cols(_table_df(ds, "icustays"))
        stay_col = "stay_id" if "stay_id" in stays.columns else "icustay_id"
        stays = stays[["hadm_id", stay_col]]
        out = stays.merge(flags, on="hadm_id", how="left").drop(columns="hadm_id")
        out["culture_positive"] = out["culture_positive"].fillna(False)
        out["bld_culture_positive"] = out["bld_culture_positive"].fillna(False)

    elif db in ("eicu", "eicu_demo"):
        mb = _lower_cols(_table_df(ds, "microlab"))
        org = mb["organism"].astype(str).str.strip().str.lower()
        positive = mb["organism"].notna() & ~org.isin(_EICU_NEGATIVE)
        site = mb.get("culturesite", pd.Series("", index=mb.index)).astype(str)
        is_blood = site.str.contains("blood", case=False, na=False)
        per_stay = pd.DataFrame({
            "patientunitstayid": mb["patientunitstayid"],
            "culture_positive": positive.values,
            "bld_culture_positive": (positive & is_blood).values,
        })
        out = per_stay.groupby("patientunitstayid").any().reset_index()

    else:  # pragma: no cover
        return pd.DataFrame()

    for c in ("culture_positive", "bld_culture_positive"):
        out[c] = out[c].astype("boolean")

    if patient_ids:
        stay_col = _STAY_ID_COL.get(db)
        if stay_col and stay_col in out.columns:
            out = out[out[stay_col].isin(list(patient_ids))]
    return out.reset_index(drop=True)


__all__ = ["load_microbiology"]
