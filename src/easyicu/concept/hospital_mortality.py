"""Storage adapter for the declared MIMIC hospital-status clinical binding."""

from collections.abc import Mapping

import pandas as pd

from ..hospital_mortality import derive_mimic_hospital_mortality_status
from ..table import ICUTable
from .errors import ConceptExtractionUnavailable


def load_mimic_hospital_mortality(data_source, *, concept_name, patient_ids, verbose):
    database = data_source.config.name
    id_column = "icustay_id" if database in {"mimic", "mimic_demo"} else "stay_id"
    try:
        stays = data_source.load_table("icustays", verbose=verbose)
        admissions = data_source.load_table("admissions", verbose=verbose)
        stays = (
            stays.data
            if isinstance(stays, ICUTable) or hasattr(stays, "data")
            else stays
        )
        admissions = (
            admissions.data
            if isinstance(admissions, ICUTable) or hasattr(admissions, "data")
            else admissions
        )
        if patient_ids is not None:
            selectors = (
                patient_ids
                if isinstance(patient_ids, Mapping)
                else {id_column: patient_ids}
            )
            for column, ids in selectors.items():
                if column not in stays:
                    raise ValueError("hospital_status_selector_unbound")
                stays = stays.loc[stays[column].isin(list(ids))]
        canonical_stays = stays.rename(columns={id_column: "stay_id"})
        status = derive_mimic_hospital_mortality_status(canonical_stays, admissions)
        frame = status.frame.rename(
            columns={"stay_id": id_column, "hospital_death": concept_name}
        )
        # A companion is a recorded death coordinate, never the status index.
        frame[f"{concept_name}_time"] = float("nan")
        if "intime" in stays and "deathtime" in admissions:
            death_times = stays.hadm_id.map(admissions.set_index("hadm_id").deathtime)
            hours = (
                pd.to_datetime(death_times, errors="coerce")
                - pd.to_datetime(stays.intime, errors="coerce")
            ).dt.total_seconds() / 3600.0
            frame[f"{concept_name}_time"] = hours.to_numpy()
            frame.loc[~frame[concept_name].fillna(False), f"{concept_name}_time"] = (
                float("nan")
            )
        return ICUTable(frame, id_columns=[id_column], value_column=concept_name)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ConceptExtractionUnavailable(
            concept_id=concept_name,
            database=database,
            stage="hospital_status",
            detail="The admission-linked hospital status could not be established",
            cause=exc,
        ) from exc
