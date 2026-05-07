"""Cohort filtering helpers for the EasyICU Streamlit app."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from easyicu.webapp.cohort_config import DISEASE_COHORT_CONFIG, ICD_FILTER_DATABASES


def _supports_icd_filter(database: str | None) -> bool:
    """Return whether the current database supports sidebar ICD filters."""
    return str(database or "").lower() in ICD_FILTER_DATABASES


def _split_query_tokens(text: str) -> list[str]:
    """Split user ICD / keyword query into compact non-empty tokens."""
    if not text:
        return []
    cleaned = str(text).replace('，', ',').replace(';', ',').replace('；', ',').replace('\n', ',')
    raw_tokens = [tok.strip() for tok in cleaned.split(',') if tok.strip()]
    expanded_tokens: list[str] = []
    for token in raw_tokens:
        range_match = re.fullmatch(r'([A-Za-z]+)(\d+)\s*-\s*([A-Za-z]+)?(\d+)', token)
        if not range_match:
            expanded_tokens.append(token)
            continue

        prefix_start, start_num, prefix_end, end_num = range_match.groups()
        prefix_start = prefix_start.upper()
        prefix_end = (prefix_end or prefix_start).upper()
        if prefix_start != prefix_end:
            expanded_tokens.append(token)
            continue

        start_int = int(start_num)
        end_int = int(end_num)
        if end_int < start_int or end_int - start_int > 50:
            expanded_tokens.append(token)
            continue

        width = max(len(start_num), len(end_num))
        expanded_tokens.extend([f"{prefix_start}{value:0{width}d}" for value in range(start_int, end_int + 1)])

    return expanded_tokens


def _get_supported_disease_cohorts(database: str) -> list[str]:
    """Return supported disease cohort keys for the current database."""
    base = ['none', 'sepsis', 'aki', 'circ_failure', 'mech_vent', 'rrt']
    if _supports_icd_filter(database):
        base.extend(['ards', 'pneumonia', 'heart_failure', 'ami', 'stroke'])
    return base


def _match_ids_by_icd_tokens(data_path: Path, database: str, icu_df: pd.DataFrame, id_col_lower: str, tokens: list[str]) -> set:
    """Match ICU stay IDs by ICD prefixes / diagnosis keywords for DBs with diagnosis coding."""
    if not tokens or not _supports_icd_filter(database):
        return set()
    matched_ids = set()
    if database in {'miiv', 'mimic'}:
        diag_path = data_path / 'diagnoses_icd.parquet'
        if diag_path.exists() and 'hadm_id' in icu_df.columns:
            diag_df = pd.read_parquet(diag_path, columns=['hadm_id', 'icd_code'])
            codes = diag_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False)
            norm_tokens = [tok.upper().replace('.', '') for tok in tokens]
            diag_mask = pd.Series(False, index=diag_df.index)
            for token in norm_tokens:
                diag_mask |= codes.str.startswith(token)
            matched_hadm = set(diag_df.loc[diag_mask, 'hadm_id'].dropna().unique())
            matched_ids = set(icu_df.loc[icu_df['hadm_id'].isin(matched_hadm), id_col_lower].dropna().unique())
    elif database == 'eicu':
        diag_path = data_path / 'diagnosis.parquet'
        if diag_path.exists():
            diag_df = pd.read_parquet(diag_path)
            diag_df.columns = [c.lower() for c in diag_df.columns]
            if 'patientunitstayid' in diag_df.columns:
                diag_text = pd.Series('', index=diag_df.index, dtype='object')
                if 'icd9code' in diag_df.columns:
                    diag_text = diag_text.str.cat(diag_df['icd9code'].astype(str), sep=' ', na_rep='')
                if 'diagnosisstring' in diag_df.columns:
                    diag_text = diag_text.str.cat(diag_df['diagnosisstring'].astype(str), sep=' ', na_rep='')
                diag_text = diag_text.str.lower().str.replace('.', '', regex=False)
                diag_mask = pd.Series(False, index=diag_df.index)
                for token in tokens:
                    diag_mask |= diag_text.str.contains(str(token).lower().replace('.', ''), na=False)
                matched_ids = set(diag_df.loc[diag_mask, 'patientunitstayid'].dropna().unique())
    return matched_ids


def _get_positive_patient_ids_from_data(
    data: dict,
    actual_id_col: str,
    concept_priority: list[str],
) -> set:
    """Infer patient IDs with positive events/labels from loaded concept data."""
    true_tokens = {'1', 'true', 't', 'yes', 'y'}
    for concept_name in concept_priority:
        df = data.get(concept_name)
        if not isinstance(df, pd.DataFrame) or df.empty or actual_id_col not in df.columns:
            continue
        value_candidates = [concept_name] + [c for c in df.columns if c not in {actual_id_col, 'charttime', 'time', 'starttime', 'datetime', 'valueuom', 'unit'}]
        value_col = next((c for c in value_candidates if c in df.columns), None)
        if not value_col:
            continue
        vals = pd.to_numeric(df[value_col], errors='coerce')
        if vals.notna().any():
            mask = vals > 0
        else:
            str_vals = df[value_col].astype(str).str.strip().str.lower()
            mask = str_vals.isin(true_tokens)
        return set(df.loc[mask.fillna(False), actual_id_col].dropna().unique())
    return set()


def _post_filter_cohort_data(data: dict, database: str) -> dict:
    """Remove patients from loaded concept data whose cohort-critical features are None.

    After load_concepts(), certain patients may have None for features like 'death'
    or 'los_icu' because the concept extraction pipeline differs from the cohort
    filter (e.g., multi-stay death attribution). This function removes such patients
    so the exported data is consistent with the cohort criteria.

    Args:
        data: dict of {concept_name: DataFrame} loaded by load_concepts
        database: Database name for ID column detection

    Returns:
        Filtered data dict with inconsistent patients removed
    """
    cf = st.session_state.get('cohort_filter', {})
    if not cf or not data:
        return data

    # Determine ID column
    id_col_map = {
        'miiv': 'stay_id', 'eicu': 'patientunitstayid', 'aumc': 'admissionid',
        'hirid': 'patientid', 'mimic': 'icustay_id', 'sic': 'CaseID',
    }
    id_col = id_col_map.get(database, 'stay_id')

    # Detect actual ID column from data
    id_candidates = [id_col, 'stay_id', 'icustay_id', 'hadm_id',
                     'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
    actual_id_col = None
    for df in data.values():
        if isinstance(df, pd.DataFrame):
            for c in id_candidates:
                if c in df.columns:
                    actual_id_col = c
                    break
            if actual_id_col:
                break
    if not actual_id_col:
        return data

    # Collect all patient IDs across all concepts
    all_patient_ids = set()
    for df in data.values():
        if isinstance(df, pd.DataFrame) and actual_id_col in df.columns:
            all_patient_ids.update(df[actual_id_col].dropna().unique())

    if not all_patient_ids:
        return data

    # Determine which patients to exclude based on cohort filter + loaded data
    exclude_ids = set()

    # 1. Survival filter: check death column value
    #    Mock data: death=0 (survived) or death=1 (deceased) for ALL patients
    #    Real data: death column may only exist for deceased patients (NaN = survived)
    if cf.get('survived') is not None and 'death' in data:
        death_df = data['death']
        if isinstance(death_df, pd.DataFrame) and actual_id_col in death_df.columns:
            # Get value column (last column or 'death')
            val_col = 'death' if 'death' in death_df.columns else death_df.columns[-1]
            # Convert death values to numeric for comparison
            death_valid = death_df[death_df[val_col].notna()].copy()
            death_vals = pd.to_numeric(death_valid[val_col], errors='coerce')

            # Patients who died (death value == 1 or True)
            died_ids = set(death_valid.loc[death_vals == 1, actual_id_col].unique())
            # Patients who survived (death value == 0 or False, or no death record)
            survived_ids = all_patient_ids - died_ids

            if not cf['survived']:
                # Deceased filter: keep only patients who died
                exclude_ids |= survived_ids
            else:
                # Survived filter: keep only patients who survived
                exclude_ids |= died_ids

    # 2. Min LOS filter: patients must have los >= threshold
    if cf.get('los_min') is not None and 'los_icu' in data:
        los_df = data['los_icu']
        if isinstance(los_df, pd.DataFrame) and actual_id_col in los_df.columns:
            val_col = 'los_icu' if 'los_icu' in los_df.columns else los_df.columns[-1]
            los_valid = los_df[los_df[val_col].notna()].copy()
            # LOS is in days, threshold is in hours
            los_hours = pd.to_numeric(los_valid[val_col], errors='coerce') * 24
            los_ok_ids = set(los_valid.loc[los_hours >= cf['los_min'], actual_id_col].unique())
            exclude_ids |= (all_patient_ids - los_ok_ids)

    # 3. Age filter: patients must have age within range
    if (cf.get('age_min') is not None or cf.get('age_max') is not None) and 'age' in data:
        age_df = data['age']
        if isinstance(age_df, pd.DataFrame) and actual_id_col in age_df.columns:
            val_col = 'age' if 'age' in age_df.columns else age_df.columns[-1]
            age_valid = age_df[age_df[val_col].notna()].copy()
            age_vals = pd.to_numeric(age_valid[val_col], errors='coerce')
            age_mask = pd.Series(True, index=age_valid.index)
            if cf.get('age_min') is not None:
                age_mask &= (age_vals >= cf['age_min'])
            if cf.get('age_max') is not None:
                age_mask &= (age_vals <= cf['age_max'])
            age_ok_ids = set(age_valid.loc[age_mask, actual_id_col].unique())
            exclude_ids |= (all_patient_ids - age_ok_ids)

    # 4. Gender filter: patients must have matching sex
    if cf.get('gender') is not None and 'sex' in data:
        sex_df = data['sex']
        if isinstance(sex_df, pd.DataFrame) and actual_id_col in sex_df.columns:
            val_col = 'sex' if 'sex' in sex_df.columns else sex_df.columns[-1]
            sex_valid = sex_df[sex_df[val_col].notna()].copy()
            sex_vals = sex_valid[val_col].astype(str).str.strip().str.upper()
            target = cf['gender'].upper()  # 'M' or 'F'
            # Match both short ('M','F') and long ('MALE','FEMALE') formats
            if target == 'M':
                target_variants = {'M', 'MALE', 'MAN', 'MÄNNLICH'}
            else:
                target_variants = {'F', 'FEMALE', 'WOMAN', 'WEIBLICH', 'VROUW', 'W'}
            sex_ok_ids = set(sex_valid.loc[sex_vals.isin(target_variants), actual_id_col].unique())
            exclude_ids |= (all_patient_ids - sex_ok_ids)

    # 5. Disease cohort filters based on loaded clinical concepts
    disease_cohort = cf.get('disease_cohort')
    if disease_cohort and disease_cohort != 'none':
        disease_cfg = DISEASE_COHORT_CONFIG.get(disease_cohort, {})
        concept_priority = disease_cfg.get('concept_priority', [])
        if concept_priority:
            positive_ids = _get_positive_patient_ids_from_data(
                data,
                actual_id_col=actual_id_col,
                concept_priority=concept_priority,
            )
            exclude_ids |= (all_patient_ids - positive_ids)

    if not exclude_ids:
        return data

    # Remove excluded patients from all concept DataFrames
    n_excluded = len(exclude_ids)
    n_total = len(all_patient_ids)
    n_remaining = n_total - n_excluded
    print(f"[COHORT POST-FILTER] Removing {n_excluded}/{n_total} patients with inconsistent cohort feature values")

    filtered_data = {}
    for concept, df in data.items():
        if isinstance(df, pd.DataFrame) and actual_id_col in df.columns:
            filtered_data[concept] = df[~df[actual_id_col].isin(exclude_ids)].copy()
        else:
            filtered_data[concept] = df

    # 🔧 Update _cohort_stats so displayed message matches actual patient count
    cohort_stats = st.session_state.get('_cohort_stats')
    if cohort_stats:
        cohort_stats['after'] = n_remaining
        cohort_stats['excluded'] = cohort_stats['before'] - n_remaining
        # Add post-filter detail
        lang = st.session_state.get('language', 'en')
        detail_label_en = f"Data consistency check: -{n_excluded}"
        detail_label_cn = f"数据一致性检查: -{n_excluded}"
        cohort_stats.setdefault('filter_details', []).append(
            (detail_label_en, detail_label_cn, n_excluded)
        )
        st.session_state['_cohort_stats'] = cohort_stats

    return filtered_data


def _get_age_series(icu_df, database, patient_df, admission_df, id_col, subject_col):
    """Return a Series of ages aligned with icu_df index."""
    try:
        if database == 'miiv':
            # MIIV: anchor_age in patients + anchor_year; admittime in admissions
            if patient_df is not None and admission_df is not None:
                merged = icu_df[[id_col, 'hadm_id']].merge(
                    admission_df[['hadm_id', 'admittime']], on='hadm_id', how='left'
                )
                merged = merged.merge(
                    patient_df[['subject_id', 'anchor_age', 'anchor_year']],
                    left_on=icu_df[subject_col].values, right_on='subject_id', how='left'
                )
                admittime = pd.to_datetime(merged['admittime'])
                age = merged['anchor_age'] + (admittime.dt.year - merged['anchor_year'])
                return age.reindex(icu_df.index)
            return None

        elif database == 'eicu':
            # eICU: age column directly in patient table (ICU table)
            if 'age' in icu_df.columns:
                age = icu_df['age'].copy()
                # eICU stores "> 89" as string
                age = pd.to_numeric(age, errors='coerce')
                return age
            return None

        elif database == 'aumc':
            # AUMC: agegroup column (e.g. "18-39", "40-49", ...)
            if 'agegroup' in icu_df.columns:
                def parse_aumc_age(ag):
                    if pd.isna(ag):
                        return None
                    s = str(ag)
                    if '-' in s:
                        parts = s.split('-')
                        try:
                            return (int(parts[0]) + int(parts[1])) / 2
                        except ValueError:
                            return None
                    if s.startswith('80'):
                        return 85
                    try:
                        return float(s)
                    except ValueError:
                        return None
                return icu_df['agegroup'].map(parse_aumc_age)
            return None

        elif database == 'hirid':
            # HiRID: age column directly in general_table
            if 'age' in icu_df.columns:
                return pd.to_numeric(icu_df['age'], errors='coerce')
            return None

        elif database == 'mimic':
            # MIMIC-III: dob in patients, intime in icustays → age = intime.year - dob.year
            if patient_df is not None and 'dob' in patient_df.columns:
                merged = icu_df.merge(
                    patient_df[['subject_id', 'dob']], on='subject_id', how='left'
                )
                intime = pd.to_datetime(merged['intime'])
                dob = pd.to_datetime(merged['dob'])
                age = (intime - dob).dt.days / 365.25
                age = age.clip(upper=90)
                return age.reindex(icu_df.index)
            return None

        elif database == 'sic':
            # SICdb: AgeOnAdmission column
            age_col = None
            for c in icu_df.columns:
                if c.lower() == 'ageonadmission':
                    age_col = c
                    break
            if age_col:
                return pd.to_numeric(icu_df[age_col], errors='coerce')  # already in years
            return None

        return None
    except Exception as e:
        print(f"[COHORT] _get_age_series error ({database}): {e}")
        return None


def _get_first_icu_mask(icu_df, database, id_col, subject_col):
    """Return a boolean Series: True where the row is the patient's first ICU stay."""
    try:
        if database == 'miiv':
            # Earliest intime per subject_id
            if 'intime' in icu_df.columns:
                intime = pd.to_datetime(icu_df['intime'])
                first_intime = intime.groupby(icu_df[subject_col]).transform('min')
                return intime == first_intime
            return None

        elif database == 'eicu':
            # unitvisitnumber == 1
            if 'unitvisitnumber' in icu_df.columns:
                return icu_df['unitvisitnumber'] == 1
            return None

        elif database == 'aumc':
            # admissioncount == 1
            if 'admissioncount' in icu_df.columns:
                return icu_df['admissioncount'] == 1
            return None

        elif database == 'hirid':
            # HiRID: each patient has exactly one entry — all True
            return pd.Series(True, index=icu_df.index)

        elif database == 'mimic':
            # MIMIC-III: earliest intime per subject_id
            if 'intime' in icu_df.columns:
                intime = pd.to_datetime(icu_df['intime'])
                first_intime = intime.groupby(icu_df[subject_col]).transform('min')
                return intime == first_intime
            return None

        elif database == 'sic':
            # SICdb: OffsetAfterFirstAdmission == 0
            offset_col = None
            for c in icu_df.columns:
                if c.lower() == 'offsetafterfirstadmission':
                    offset_col = c
                    break
            if offset_col:
                return icu_df[offset_col] == 0
            return None

        return None
    except Exception as e:
        print(f"[COHORT] _get_first_icu_mask error ({database}): {e}")
        return None


def _get_los_hours_series(icu_df, database):
    """Return a Series of Length of Stay in hours."""
    try:
        if database == 'miiv':
            if 'los' in icu_df.columns:
                return pd.to_numeric(icu_df['los'], errors='coerce') * 24  # stored in days
            elif 'intime' in icu_df.columns and 'outtime' in icu_df.columns:
                dt = pd.to_datetime(icu_df['outtime']) - pd.to_datetime(icu_df['intime'])
                return dt.dt.total_seconds() / 3600
            return None

        elif database == 'eicu':
            # unitdischargeoffset is in minutes from admission
            if 'unitdischargeoffset' in icu_df.columns:
                return pd.to_numeric(icu_df['unitdischargeoffset'], errors='coerce') / 60
            return None

        elif database == 'aumc':
            if 'admittedat' in icu_df.columns and 'dischargedat' in icu_df.columns:
                # stored in milliseconds from some epoch
                admitted = pd.to_numeric(icu_df['admittedat'], errors='coerce')
                discharged = pd.to_numeric(icu_df['dischargedat'], errors='coerce')
                return (discharged - admitted) / 1000 / 3600  # ms -> hours
            return None

        elif database == 'hirid':
            # HiRID general_table doesn't have reliable LOS — return None to skip filter
            return None

        elif database == 'mimic':
            if 'los' in icu_df.columns:
                return pd.to_numeric(icu_df['los'], errors='coerce') * 24  # stored in days
            elif 'intime' in icu_df.columns and 'outtime' in icu_df.columns:
                dt = pd.to_datetime(icu_df['outtime']) - pd.to_datetime(icu_df['intime'])
                return dt.dt.total_seconds() / 3600
            return None

        elif database == 'sic':
            # SICdb: TimeOfStay in seconds
            tos_col = None
            for c in icu_df.columns:
                if c.lower() == 'timeofstay':
                    tos_col = c
                    break
            if tos_col:
                return pd.to_numeric(icu_df[tos_col], errors='coerce') / 3600  # seconds -> hours
            return None

        return None
    except Exception as e:
        print(f"[COHORT] _get_los_hours_series error ({database}): {e}")
        return None


def _get_sex_series(icu_df, database, patient_df, id_col, subject_col):
    """Return a Series of sex normalized to 'M'/'F'."""
    try:
        SEX_MAP_M = {'m', 'male', 'man', 'männlich', 'Man', 'Male'}
        SEX_MAP_F = {'f', 'female', 'woman', 'weiblich', 'Vrouw', 'Female'}

        def normalize_sex(s):
            if pd.isna(s):
                return None
            s_str = str(s).strip()
            if s_str.lower() in {x.lower() for x in SEX_MAP_M}:
                return 'M'
            if s_str.lower() in {x.lower() for x in SEX_MAP_F}:
                return 'F'
            return None

        if database == 'miiv':
            if patient_df is not None and 'gender' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'gender']], on=subject_col, how='left'
                )
                return merged['gender'].map(normalize_sex).reindex(icu_df.index)
            return None

        elif database == 'eicu':
            if 'gender' in icu_df.columns:
                return icu_df['gender'].map(normalize_sex)
            return None

        elif database == 'aumc':
            if 'gender' in icu_df.columns:
                return icu_df['gender'].map(normalize_sex)
            return None

        elif database == 'hirid':
            if 'sex' in icu_df.columns:
                return icu_df['sex'].map(normalize_sex)
            return None

        elif database == 'mimic':
            if patient_df is not None and 'gender' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'gender']], on=subject_col, how='left'
                )
                return merged['gender'].map(normalize_sex).reindex(icu_df.index)
            return None

        elif database == 'sic':
            sex_col = None
            for c in icu_df.columns:
                if c.lower() == 'sex':
                    sex_col = c
                    break
            if sex_col:
                def sic_sex(v):
                    if pd.isna(v):
                        return None
                    v_int = int(v) if isinstance(v, (int, float)) else None
                    # SICdb uses 735=Male, 736=Female
                    if v_int == 735 or v_int == 0 or str(v).lower() in {'m', 'male', '0'}:
                        return 'M'
                    if v_int == 736 or v_int == 1 or str(v).lower() in {'f', 'female', '1', 'w'}:
                        return 'F'
                    return normalize_sex(v)
                return icu_df[sex_col].map(sic_sex)
            return None

        return None
    except Exception as e:
        print(f"[COHORT] _get_sex_series error ({database}): {e}")
        return None


def _pick_death_stay(merged, dead_mask, id_col, deathtime_col, intime_col, outtime_col):
    """For multi-stay admissions, pick the ICU stay to which death should be attributed.

    The death concept assigns the death event (using deathtime as the index) to
    a specific ICU stay via a rolling join.  This helper replicates that logic:
      1. If deathtime falls within [intime, outtime] → that stay.
      2. Otherwise the last ICU stay whose intime ≤ deathtime.
      3. Fallback: the very last ICU stay in the admission.
    """
    dead_rows = merged[dead_mask].copy()
    if dead_rows.empty:
        return set()

    dt = pd.to_datetime(dead_rows[deathtime_col], errors='coerce')
    it = pd.to_datetime(dead_rows[intime_col], errors='coerce')
    ot = pd.to_datetime(dead_rows[outtime_col], errors='coerce')

    dead_rows = dead_rows.copy()
    dead_rows['_dt'] = dt
    dead_rows['_it'] = it
    dead_rows['_ot'] = ot
    dead_rows['_in_stay'] = (it <= dt) & (dt <= ot)

    result_ids = set()
    for hadm, grp in dead_rows.groupby('hadm_id'):
        if len(grp) == 1:
            result_ids.add(grp.iloc[0][id_col])
            continue
        # 1. deathtime within the ICU stay
        in_stay = grp[grp['_in_stay']]
        if len(in_stay) > 0:
            result_ids.add(in_stay.iloc[0][id_col])
            continue
        # 2. last stay whose intime ≤ deathtime
        before = grp[grp['_it'] <= grp['_dt']]
        if len(before) > 0:
            result_ids.add(before.sort_values('_it').iloc[-1][id_col])
            continue
        # 3. fallback: last ICU stay overall
        result_ids.add(grp.sort_values('_it').iloc[-1][id_col])
    return result_ids


def _get_death_series(icu_df, database, patient_df, admission_df, id_col, subject_col):
    """Return a boolean Series: True where patient died in hospital/ICU.

    IMPORTANT: This must match the EasyICU 'death' concept definition exactly,
    so that filtering for 'deceased' patients guarantees death=True in the output.

    Concept definitions (concept-dict.json):
      - miiv/mimic: admissions.hospital_expire_flag == 1, index_var=deathtime
      - eicu: patient.hospitaldischargestatus == 'Expired'
      - aumc: aumc_death callback → dateofdeath not null AND (dateofdeath - dischargedat) < 72h
      - hirid: hirid_death callback → discharge_status == 'dead' in general table
      - sic: no death concept defined
    """
    try:
        if database == 'miiv':
            # Concept: admissions table, hospital_expire_flag == 1, index_var = deathtime
            # Must have BOTH flag=1 AND non-null deathtime (concept needs timestamp)
            # For multi-stay admissions, death is only attributed to the ICU stay
            # where deathtime falls (matching the concept's rolling-join behavior).
            if admission_df is not None and 'hospital_expire_flag' in admission_df.columns:
                merge_cols = ['hadm_id', 'hospital_expire_flag']
                if 'deathtime' in admission_df.columns:
                    merge_cols.append('deathtime')
                merged = icu_df.merge(
                    admission_df[merge_cols].drop_duplicates('hadm_id'),
                    on='hadm_id', how='left'
                )
                dead_base = (merged['hospital_expire_flag'] == 1)
                if 'deathtime' in merged.columns:
                    dead_base = dead_base & merged['deathtime'].notna()
                # For multi-stay admissions, only attribute death to the correct stay
                if 'deathtime' in merged.columns and 'intime' in merged.columns:
                    dead_stay_ids = _pick_death_stay(merged, dead_base, id_col, 'deathtime', 'intime', 'outtime')
                    return merged[id_col].isin(dead_stay_ids).reindex(icu_df.index).fillna(False)
                return dead_base.fillna(False).reindex(icu_df.index)
            return None

        elif database == 'eicu':
            # Concept: patient.hospitaldischargestatus == 'Expired'
            # (NOT unitdischargestatus — concept uses hospitaldischargestatus)
            if 'hospitaldischargestatus' in icu_df.columns:
                return (icu_df['hospitaldischargestatus'].astype(str).str.strip() == 'Expired')
            # Fallback to unit status only if hospital status is missing
            if 'unitdischargestatus' in icu_df.columns:
                return icu_df['unitdischargestatus'].str.lower().str.contains('expire', na=False)
            return None

        elif database == 'aumc':
            # Concept: aumc_death callback → dateofdeath not null AND
            #   (dateofdeath - dischargedat) < 72 hours (in milliseconds)
            if 'dateofdeath' in icu_df.columns and 'dischargedat' in icu_df.columns:
                dateofdeath = pd.to_numeric(icu_df['dateofdeath'], errors='coerce')
                dischargedat = pd.to_numeric(icu_df['dischargedat'], errors='coerce')
                hours_72_ms = 72 * 3600 * 1000
                diff = dateofdeath - dischargedat
                return (dateofdeath.notna() & (diff < hours_72_ms)).fillna(False)
            # Fallback: dateofdeath not null
            if 'dateofdeath' in icu_df.columns:
                return icu_df['dateofdeath'].notna()
            if 'destination' in icu_df.columns:
                return icu_df['destination'].str.lower().str.contains('overleden', na=False)
            return None

        elif database == 'hirid':
            # Concept: hirid_death callback → discharge_status == 'dead' from general table
            if 'discharge_status' in icu_df.columns:
                ds = icu_df['discharge_status']
                if ds.dtype == object:
                    return ds.str.lower().str.strip() == 'dead'
                else:
                    return ds == 1
            return None

        elif database == 'mimic':
            # Concept: admissions.hospital_expire_flag == 1, index_var = deathtime
            # Same multi-stay logic as MIIV.
            if admission_df is not None and 'hospital_expire_flag' in admission_df.columns:
                if 'hadm_id' in icu_df.columns:
                    merge_cols = ['hadm_id', 'hospital_expire_flag']
                    if 'deathtime' in admission_df.columns:
                        merge_cols.append('deathtime')
                    merged = icu_df.merge(
                        admission_df[merge_cols].drop_duplicates('hadm_id'),
                        on='hadm_id', how='left'
                    )
                    dead_base = (merged['hospital_expire_flag'] == 1)
                    if 'deathtime' in merged.columns:
                        dead_base = dead_base & merged['deathtime'].notna()
                    if 'deathtime' in merged.columns and 'intime' in merged.columns:
                        dead_stay_ids = _pick_death_stay(merged, dead_base, id_col, 'deathtime', 'intime', 'outtime')
                        return merged[id_col].isin(dead_stay_ids).reindex(icu_df.index).fillna(False)
                    return dead_base.fillna(False).reindex(icu_df.index)
            # Alternative: dod in patients
            if patient_df is not None and 'dod' in patient_df.columns:
                merged = icu_df[[subject_col]].merge(
                    patient_df[[subject_col, 'dod']], on=subject_col, how='left'
                )
                return merged['dod'].notna().reindex(icu_df.index)
            return None

        elif database == 'sic':
            # No death concept defined in concept-dict.json
            # Use OffsetOfDeath > 0 as best available approximation
            death_col = None
            for c in icu_df.columns:
                if c.lower() == 'offsetofdeath':
                    death_col = c
                    break
            if death_col:
                return icu_df[death_col].notna() & (pd.to_numeric(icu_df[death_col], errors='coerce') > 0)
            return None

        return None
    except Exception as e:
        print(f"[COHORT] _get_death_series error ({database}): {e}")
        return None


def _get_sepsis_runtime_options() -> dict:
    """Read current web sepsis settings and return kwargs for load_concepts/callbacks."""
    abx_hours = st.session_state.get('sepsis_abx_win_hours', 24)
    samp_hours = st.session_state.get('sepsis_samp_win_hours', 72)
    return {
        'si_mode': st.session_state.get('sepsis_si_mode', 'auto'),
        'positive_cultures': bool(st.session_state.get('sepsis_positive_cultures', False)),
        'abx_win': f"{int(abx_hours)}h",
        'samp_win': f"{int(samp_hours)}h",
    }
