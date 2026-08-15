"""SOFA-2 score callbacks and helpers.

This module implements SOFA-2 (2025 consensus, Moreno et al. JAMA Network Open)
organ component rules and an aggregate scorer compatible with the existing easyicu
callback style.

Components implemented (0-4 each):
- Respiratory (PaO2/FiO2 or SpO2/FiO2 + advanced support/ECMO)
  * SOFA-2 ratio thresholds: ≤300/225/150/75 (vs SOFA-1: >400/300-400/200-299/100-199/<100)
  * Any ECMO → respiratory auto 4pt
  * Advanced support includes: IMV, NIV, HFNC, CPAP, BiPAP, home ventilation
  
- Hemostasis/Coagulation (platelets)
  * SOFA-2 thresholds: ≤150/100/80/50 (vs SOFA-1: >150/100-150/50-99/20-49/<20)
  * Simplified 1pt threshold: ≤150 instead of 100-150 range
  
- Liver (bilirubin)
  * SOFA-2 thresholds: ≤1.2/3.0/6.0/12.0/>12.0 mg/dL
  * Relaxed 1pt: ≤3.0 vs SOFA-1's ≤1.9 mg/dL
  
- Cardiovascular (MAP, combined norepi+epi; alt dopamine-only; mech support)
  * PRIMARY: Combined norepinephrine + epinephrine (μg/kg/min)
    - Low dose: ≤0.2 → 2pt
    - Medium dose: >0.2-0.4 → 3pt
    - High dose: >0.4 → 4pt
    - Medium + other vaso → 4pt
  * ALTERNATE (dopamine only, when norepi+epi==0):
    - ≤20 → 2pt, >20-40 → 3pt, >40 → 4pt
  * Mechanical circulatory support (VA-ECMO, IABP, LVAD, Impella) → auto 4pt
  * Dopamine downgraded from primary to backup role
  
- Brain/CNS (GCS with optional delirium treatment)
  * SOFA-2 same GCS thresholds: 15/13-14/9-12/6-8/3-5
  * NEW: Delirium treatment → 1pt even if GCS=15
  * Sedated patients: use pre-sedation GCS; if unknown → 0pt
  
- Renal (creatinine; optional urine rate; RRT = 4)
  * RRT (or meets criteria) → auto 4pt
  * Urine output standardized to mL/kg/h (vs absolute mL/day in SOFA-1):
    - <0.5 mL/kg/h (6-12h) → 1pt
    - <0.5 mL/kg/h (≥12h) → 2pt
    - <0.3 mL/kg/h (≥24h) or anuria ≥12h → 3pt
  * Creatinine: >1.2/2.0/3.5 mg/dL → 1/2/3pt

Notes:
- Inputs are pandas Series aligned on the same index. Missing values are
  handled similarly to SOFA-1 implementation (treated as normal unless a
  threshold is met by another provided variable).
- ``sofa2_score`` aggregates one observation/day-1 record only. The production
  concept callback owns 24-hour worst-value windows and post-day-1 LOCF using
  component-issued observed/available receipts.
- Drug requirements: continuous IV infusion ≥1 hour for vasopressors/inotropes
- Transient changes (<1 hour, e.g., post-suction hypoxemia) should not be scored

References:
- Moreno et al. (2025). SOFA-2 Consensus Statement. JAMA Network Open.
- JAMA Network Open (2026). Errors in Tables. doi:10.1001/jamanetworkopen.2025.60466.
- Vincent et al. (1996). Original SOFA score. Intensive Care Medicine.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .sofa2_aggregate import SOFA2_COMPONENT_NAMES, sofa2_score
from .sofa2_validation import (
    SOFA2InputError,
    normalize_fio2_input,
    validate_aligned_input,
    validate_numeric_input,
)


def _is_true(series: pd.Series) -> pd.Series:
    """Replicate R's is_true: non-NA and True."""
    return series.fillna(False).astype(bool)


def _coalesce_series(*series_list: Optional[pd.Series]) -> Optional[pd.Series]:
    """Return the first non-None series from a list of aliases."""
    for series in series_list:
        if series is not None:
            return series
    return None


def sofa2_component_evidence(
    component: str,
    *,
    index: pd.Index,
    **inputs: Optional[pd.Series],
) -> pd.DataFrame:
    """Return component-owned observation and availability receipts.

    A numeric component score is not itself evidence: every SOFA-2 scorer may
    emit zero when its physiology is missing.  ``observed`` records a genuine
    domain observation at this time point; ``available`` records whether the
    component has an evidence-backed value eligible for longitudinal LOCF.
    """

    if component not in SOFA2_COMPONENT_NAMES:
        raise ValueError(f"Unsupported SOFA-2 component evidence owner: {component}")

    def first(*names: str) -> Optional[pd.Series]:
        return next(
            (inputs[name] for name in names if inputs.get(name) is not None),
            None,
        )

    def numeric(value: Optional[pd.Series]) -> pd.Series:
        if value is None:
            return pd.Series(False, index=index, dtype=bool)
        return pd.to_numeric(pd.Series(value, index=index), errors="coerce").notna()

    def numeric_values(value: Optional[pd.Series]) -> pd.Series:
        if value is None:
            return pd.Series(np.nan, index=index, dtype=float)
        return pd.to_numeric(pd.Series(value, index=index), errors="coerce")

    def positive(value: Optional[pd.Series]) -> pd.Series:
        if value is None:
            return pd.Series(False, index=index, dtype=bool)
        return _is_true(pd.Series(value, index=index))

    observed = pd.Series(False, index=index, dtype=bool)
    available = pd.Series(False, index=index, dtype=bool)

    if component == "sofa2_resp":
        pafi = numeric(inputs.get("pafi"))
        spo2 = numeric_values(inputs.get("spo2"))
        fio2 = numeric_values(inputs.get("fio2"))
        sf_observed = spo2.notna() & fio2.notna()
        ecmo = positive(inputs.get("ecmo"))
        persistence = inputs.get("oxygenation_sustained_1h")
        transient = pd.Series(False, index=index, dtype=bool)
        if persistence is not None:
            persistence_series = pd.Series(persistence, index=index)
            transient = persistence_series.notna() & ~_is_true(persistence_series)
        observed = pafi | sf_observed | ecmo
        available = ((pafi | (sf_observed & (spo2 < 98) & (fio2 > 0))) & ~transient) | ecmo
    elif component == "sofa2_coag":
        observed = numeric(first("plt", "platelets"))
        available = observed.copy()
    elif component == "sofa2_liver":
        observed = numeric(first("bili", "bilirubin"))
        available = observed.copy()
    elif component == "sofa2_cardio":
        map_observed = numeric(inputs.get("map"))
        positive_drug = pd.Series(False, index=index, dtype=bool)
        for names in (
            ("norepi60", "norepi"),
            ("epi60", "epi"),
            ("dopa60", "dopamine60"),
            ("dobu60", "dobutamine60"),
        ):
            positive_drug |= numeric_values(first(*names)).fillna(0) > 0
        positive_support = positive(inputs.get("other_vaso")) | positive(
            inputs.get("mech_circ_support")
        )
        ecmo = positive(inputs.get("ecmo"))
        indication = pd.Series(
            inputs.get("ecmo_indication"), index=index, dtype="string"
        )
        cardiac_ecmo = ecmo & (indication == "cardiovascular")
        observed = map_observed | positive_drug | positive_support | cardiac_ecmo
        available = observed.copy()
    elif component == "sofa2_cns":
        gcs_observed = numeric(inputs.get("gcs"))
        motor_observed = numeric(inputs.get("motor_response"))
        pre_observed = numeric(first("sedated_gcs", "pre_sedation_gcs"))
        evidence = _normalize_delirium_tx_evidence(
            index,
            delirium_tx_evidence=inputs.get("delirium_tx_evidence"),
            delirium_tx_proxy=inputs.get("delirium_tx_proxy"),
            delirium_tx=inputs.get("delirium_tx"),
        )
        confirmed = evidence == "confirmed"
        sedated = positive(inputs.get("sedated"))
        sedation_blocks_current = sedated & ~pre_observed
        observed = gcs_observed | motor_observed | pre_observed | confirmed
        available = (
            pre_observed
            | ((gcs_observed | motor_observed) & ~sedation_blocks_current)
            | confirmed
        )
    elif component == "sofa2_renal":
        creatinine = numeric(first("crea", "creatinine"))
        urine_windows = pd.Series(False, index=index, dtype=bool)
        for name in ("uo_6h", "uo_12h", "uo_24h"):
            urine_windows |= numeric(inputs.get(name))
        urine_rate = numeric(inputs.get("urine_mlkgph"))
        urine_duration = numeric(inputs.get("urine_duration_h"))
        treatment = positive(inputs.get("rrt")) | positive(
            inputs.get("rrt_episode_active")
        )
        treatment &= ~positive(inputs.get("rrt_nonrenal_only"))
        qualifying_rrt = treatment | positive(inputs.get("rrt_criteria"))
        observed = creatinine | urine_windows | urine_rate | qualifying_rrt
        available = (
            creatinine | urine_windows | (urine_rate & urine_duration) | qualifying_rrt
        )

    return pd.DataFrame(
        {
            # Nullable comparisons (for example string evidence compared with
            # one category) can retain ``pd.NA``. At this owner boundary, NA
            # means that no positive observation/availability evidence was
            # established; fail closed to 0 before publishing integer receipts.
            f"{component}_observed": observed.fillna(False).astype("int8"),
            f"{component}_available": available.fillna(False).astype("int8"),
        },
        index=index,
    )


def sofa2_resp(
    pafi: Optional[pd.Series] = None,
    *,
    spo2: Optional[pd.Series] = None,
    fio2: Optional[pd.Series] = None,
    adv_resp: Optional[pd.Series] = None,
    ecmo: Optional[pd.Series] = None,
    ecmo_indication: Optional[pd.Series] = None,
    support_unavailable_or_ceiling: Optional[pd.Series] = None,
    oxygenation_sustained_1h: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 respiratory component.

    Priority of oxygenation metric: use PaO2/FiO2 if available; otherwise
    derive SpO2/FiO2 when both are present (only when SpO2 < 98%).
    
    SOFA-2 thresholds for the P/F ratio vs SOFA-1:
    ┌────────┬─────────────┬───────────────┬─────────────────────┐
    │ Score  │ SOFA-1 P/F  │ SOFA-2 P/F    │ SOFA-2 Requirements │
    ├────────┼─────────────┼───────────────┼─────────────────────┤
    │   0    │ >400        │ >300          │ None                │
    │   1    │ 300-400     │ ≤300          │ None                │
    │   2    │ 200-299     │ ≤225          │ None                │
    │   3    │ 100-199+MV  │ ≤150          │ Advanced support^   │
    │   4    │ <100+MV     │ ≤75           │ Advanced support^   │
    │        │             │ OR ECMO                                 │
    └────────┴─────────────┴───────────────┴─────────────────────┘

    ^ Advanced support: HFNC, CPAP, BiPAP, NIV, IMV, long-term home ventilation

    SpO2/FiO2 alternative thresholds (when SpO2 < 98%):
    - 0: >300  │ 1: ≤300  │ 2: ≤250  │ 3: ≤200+support  │ 4: ≤120+support or ECMO
    
    ECMO special rules:
    - Any ECMO → respiratory auto 4pt (regardless of P/F)
    - ECMO for cardiovascular indication → additionally score cardiovascular
    
    Args:
        pafi: PaO2/FiO2 ratio (unitless ratio; see the 2026 correction)
        spo2: Oxygen saturation (%)
        fio2: Fraction of inspired oxygen (0.21-1.0 or 21-100)
        adv_resp: Boolean - advanced respiratory support active
        ecmo: Boolean - ECMO in use
        ecmo_indication: String - 'respiratory' or 'cardiovascular'
        support_unavailable_or_ceiling: Boolean - advanced support was unavailable
            or precluded by a documented ceiling of treatment (Table 2 footnote h)
        oxygenation_sustained_1h: Optional Boolean persistence assessment.
            Explicit False excludes a documented transient episode shorter than
            1 hour. Missing/unknown does not suppress an otherwise valid ratio.
        
    Returns:
        Series of respiratory SOFA-2 scores (0-4)
    """
    # Determine and validate the shared component index.
    if pafi is not None:
        idx = validate_aligned_input(
            pafi, component="sofa2_resp", field="pafi"
        ).index
    elif spo2 is not None:
        idx = validate_aligned_input(
            spo2, component="sofa2_resp", field="spo2"
        ).index
    elif fio2 is not None:
        idx = validate_aligned_input(
            fio2, component="sofa2_resp", field="fio2"
        ).index
    else:
        raise ValueError("sofa2_resp requires at least one of: pafi, spo2, or fio2")
    
    # Build support/ECMO masks
    support = (
        _is_true(
            validate_aligned_input(
                adv_resp,
                component="sofa2_resp",
                field="adv_resp",
                index=idx,
            )
        )
        if adv_resp is not None
        else pd.Series(False, index=idx)
    )
    on_ecmo = (
        _is_true(
            validate_aligned_input(
                ecmo,
                component="sofa2_resp",
                field="ecmo",
                index=idx,
            )
        )
        if ecmo is not None
        else pd.Series(False, index=idx)
    )
    support_exception = (
        _is_true(
            validate_aligned_input(
                support_unavailable_or_ceiling,
                component="sofa2_resp",
                field="support_unavailable_or_ceiling",
                index=idx,
            )
        )
        if support_unavailable_or_ceiling is not None
        else pd.Series(False, index=idx)
    )
    # Table 2 footnote f excludes a *known* transient oxygenation change (for
    # example, a brief post-suction deterioration); it does not require every
    # valid P/F or S/F observation to carry a separate persistence proof. Keep
    # unknown distinct from explicit False so normal production inputs remain
    # scoreable while a documented <1 h episode can still be excluded.
    if oxygenation_sustained_1h is None:
        ratio_eligible = pd.Series(True, index=idx)
    else:
        persistence = validate_aligned_input(
            oxygenation_sustained_1h,
            component="sofa2_resp",
            field="oxygenation_sustained_1h",
            index=idx,
        )
        explicitly_transient = persistence.notna() & ~_is_true(persistence)
        ratio_eligible = ~explicitly_transient
    severe_ratio_eligible = ratio_eligible & (support | support_exception)
    
    # Initialize score
    score = pd.Series(0, index=idx, dtype=int)
    
    # Prepare P/F ratio (per-row availability check)
    pf = (
        validate_numeric_input(
            pafi,
            component="sofa2_resp",
            field="pafi",
            index=idx,
            minimum=0,
            minimum_inclusive=False,
        )
        if pafi is not None
        else pd.Series(np.nan, index=idx)
    )
    pf_available = pf.notna()
    
    # Prepare S/F ratio (only applicable when SpO2 < 98%)
    sf = pd.Series(np.nan, index=idx)
    sf_applicable = pd.Series(False, index=idx)
    
    if spo2 is not None and fio2 is not None:
        s = validate_numeric_input(
            spo2,
            component="sofa2_resp",
            field="spo2",
            index=idx,
            minimum=0,
            maximum=100,
        )
        f_adj = normalize_fio2_input(fio2, index=idx).values
    else:
        if spo2 is not None:
            validate_numeric_input(
                spo2,
                component="sofa2_resp",
                field="spo2",
                index=idx,
                minimum=0,
                maximum=100,
            )
        if fio2 is not None:
            normalize_fio2_input(fio2, index=idx)
        
    if spo2 is not None and fio2 is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            sf = s / f_adj
        
        # S/F only applicable when SpO2 < 98% (per SOFA-2 definition)
        sf_applicable = (s < 98) & sf.notna()
    
    # === Per-row scoring logic ===
    
    # Case 1: P/F available → use P/F thresholds
    pf_mask = pf_available & ratio_eligible
    score[pf_mask & (pf <= 300)] = 1
    score[pf_mask & (pf <= 225)] = 2
    # For 3/4, advanced support or ECMO is required
    score[pf_mask & (pf <= 150) & severe_ratio_eligible] = 3
    score[pf_mask & (pf <= 75) & severe_ratio_eligible] = 4
    
    # Case 2: P/F unavailable but S/F applicable → use S/F thresholds
    sf_mask = ~pf_available & sf_applicable & ratio_eligible
    score[sf_mask & (sf <= 300)] = 1
    score[sf_mask & (sf <= 250)] = 2
    score[sf_mask & (sf <= 200) & severe_ratio_eligible] = 3
    score[sf_mask & (sf <= 120) & severe_ratio_eligible] = 4
    
    # SOFA-2 footnote (i): any patient on ECMO scores 4 on the respiratory
    # component regardless of PaO2:FiO2. (ECMO for respiratory failure scores 4
    # here only; ECMO for a cardiovascular indication is additionally scored on
    # the cardiovascular component via mechanical circulatory support.) Gate on
    # `on_ecmo` (any ECMO), not on respiratory indication alone. `ecmo_indication`
    # is kept in the signature so the dictionary can still pass it for provenance.
    score[on_ecmo] = 4

    return score


def sofa2_coag(
    plt: Optional[pd.Series] = None,
    *,
    platelets: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 hemostasis/coagulation component (platelets ×10³/μL).

    SOFA-2 simplified thresholds vs SOFA-1:
    ┌────────┬──────────────┬──────────────┐
    │ Score  │ SOFA-1       │ SOFA-2       │
    ├────────┼──────────────┼──────────────┤
    │   0    │ >150         │ >150         │
    │   1    │ 100-150      │ ≤150         │ <- Simplified range
    │   2    │ 50-99        │ ≤100         │
    │   3    │ 20-49        │ ≤80          │ <- New threshold
    │   4    │ <20          │ ≤50          │ <- Raised threshold
    └────────┴──────────────┴──────────────┘
    
    Key changes:
    - 1pt threshold simplified: ≤150 instead of 100-150 range
    - 3pt threshold raised: ≤80 instead of 20-49
    - 4pt threshold raised: ≤50 instead of <20
    - Earlier detection of hemostatic dysfunction
    
    Args:
        plt: Platelet count (×10³/μL)
        
    Returns:
        Series of hemostasis SOFA-2 scores (0-4)
    """
    platelet_series = _coalesce_series(plt, platelets)
    if platelet_series is None:
        raise ValueError("sofa2_coag requires `plt` or `platelets`")
    p = validate_numeric_input(
        platelet_series,
        component="sofa2_coag",
        field="platelets",
        minimum=0,
    )
    score = pd.Series(0, index=platelet_series.index, dtype=int)
    score[p <= 150] = 1
    score[p <= 100] = 2
    score[p <= 80] = 3
    score[p <= 50] = 4
    return score

def sofa2_liver(
    bili: Optional[pd.Series] = None,
    *,
    bilirubin: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 liver component (bilirubin mg/dL).

    SOFA-2 thresholds based on consensus table:
    ┌────────┬──────────────┬──────────────┐
    │ Score  │ SOFA-1       │ SOFA-2       │
    ├────────┼──────────────┼──────────────┤
    │   0    │ <1.2         │ ≤1.20        │
    │   1    │ 1.2-1.9      │ ≤3.0         │ <- Relaxed (was 1.9)
    │   2    │ 2.0-5.9      │ ≤6.0         │
    │   3    │ 6.0-11.9     │ ≤12.0        │
    │   4    │ >12.0        │ >12.0        │
    └────────┴──────────────┴──────────────┘

    Key change:
    - 1pt threshold relaxed from ≤1.9 to ≤3.0 mg/dL
    - Reduces false positives for mild liver dysfunction
    - Better reflects clinically significant hepatic impairment
    
    Args:
        bili: Total bilirubin (mg/dL)
        
    Returns:
        Series of liver SOFA-2 scores (0-4)
        
    Note: For μmol/L units, multiply mg/dL by 17.1:
        ≤1.20 mg/dL = ≤20.6 μmol/L
        ≤3.0 mg/dL = ≤51.3 μmol/L
        ≤6.0 mg/dL = ≤102.6 μmol/L
        ≤12.0 mg/dL = ≤205 μmol/L
    """
    bilirubin_series = _coalesce_series(bili, bilirubin)
    if bilirubin_series is None:
        raise ValueError("sofa2_liver requires `bili` or `bilirubin`")
    b = validate_numeric_input(
        bilirubin_series,
        component="sofa2_liver",
        field="bilirubin",
        minimum=0,
    )
    score = pd.Series(0, index=bilirubin_series.index, dtype=int)

    # Apply thresholds according to SOFA-2 table (using upper bounds)
    # 0pt: ≤1.20, 1pt: >1.20-3.0, 2pt: >3.0-6.0, 3pt: >6.0-12.0, 4pt: >12.0
    score[b > 1.20] = 1
    score[b > 3.0] = 2
    score[b > 6.0] = 3
    score[b > 12.0] = 4
    return score

def sofa2_cardio(
    map: pd.Series,
    *,
    norepi: Optional[pd.Series] = None,
    epi: Optional[pd.Series] = None,
    norepi60: Optional[pd.Series] = None,
    epi60: Optional[pd.Series] = None,
    dopa60: Optional[pd.Series] = None,
    dobu60: Optional[pd.Series] = None,
    dopamine60: Optional[pd.Series] = None,
    dobutamine60: Optional[pd.Series] = None,
    other_vaso: Optional[pd.Series] = None,
    mech_circ_support: Optional[pd.Series] = None,
    ecmo: Optional[pd.Series] = None,
    ecmo_indication: Optional[pd.Series] = None,
    vasopressors_unavailable: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 cardiovascular component.

    Scoring based on the SOFA-2 consensus table:
    ┌────────┬────────────────────────────────────────────────────────┐
    │ Score  │ SOFA-2 Criteria                                        │
    ├────────┼────────────────────────────────────────────────────────┤
    │   0    │ MAP ≥70 mmHg, no vasopressor/inotrope                 │
    │   1    │ MAP <70 mmHg, no vasopressor/inotrope                 │
    │   2    │ Low-dose vasopressor (norepi+epi ≤0.2)               │
    │        │ OR any other vasopressor/inotrope                     │
    │   3    │ Medium-dose vasopressor (norepi+epi >0.2-0.4)        │
    │        │ OR low-dose + other vasopressor/inotrope              │
    │   4    │ High-dose vasopressor (norepi+epi >0.4)              │
    │        │ OR medium-dose + other vasopressor/inotrope           │
    │        │ OR mechanical circulatory support*                    │
    └────────┴────────────────────────────────────────────────────────┘

    *Mechanical support: VA-ECMO, IABP, LVAD, Impella, microaxial flow pump

    ALTERNATE scoring (dopamine only, when norepi+epi == 0):
    - 2pt: ≤20 μg/kg/min  │ 3pt: >20-40 μg/kg/min  │ 4pt: >40 μg/kg/min

    MAP-only fallback (when vasopressors unavailable/ceiling of care):
    - 0pt: ≥70 mmHg  │ 1pt: 60-69  │ 2pt: 50-59  │ 3pt: 40-49  │ 4pt: <40 mmHg

    Important notes:
    - Vasopressors must be continuous IV infusion ≥1 hour to count
    - Norepinephrine base equivalents (salt conversion):
      * 1 mg base = 2 mg bitartrate monohydrate
      * 1 mg base = 1.89 mg anhydrous bitartrate
      * 1 mg base = 1.22 mg hydrochloride
    - "other_vaso" includes: vasopressin, phenylephrine, dopamine (adjunct), dobutamine
    
    Args:
        map: Mean arterial pressure (mmHg)
        norepi: Norepinephrine dose (μg/kg/min) - readable alias
        epi: Epinephrine dose (μg/kg/min) - readable alias
        norepi60: Norepinephrine dose (μg/kg/min) - EasyICU runtime concept name
        epi60: Epinephrine dose (μg/kg/min) - EasyICU runtime concept name
        dopa60: Dopamine dose (μg/kg/min) - EasyICU runtime concept name
        dobu60: Dobutamine dose (μg/kg/min) - EasyICU runtime concept name
        dopamine60: Dopamine dose (μg/kg/min) - readable alias
        dobutamine60: Dobutamine dose (μg/kg/min) - readable alias
        other_vaso: Boolean - other vasoactive drugs present
        mech_circ_support: Boolean - mechanical circulatory support active
        vasopressors_unavailable: Boolean - vasopressors unavailable/precluded

    Returns:
        Series of cardiovascular SOFA-2 scores (0-4)
    """
    map_val = validate_numeric_input(
        map,
        component="sofa2_cardio",
        field="map",
        minimum=0,
    )
    idx = map_val.index
    dopamine = _coalesce_series(dopa60, dopamine60)
    dobutamine = _coalesce_series(dobu60, dobutamine60)
    norepi_series = _coalesce_series(norepi60, norepi)
    epi_series = _coalesce_series(epi60, epi)
    ne = (
        validate_numeric_input(
            norepi_series,
            component="sofa2_cardio",
            field="norepi",
            index=idx,
            minimum=0,
        )
        if norepi_series is not None
        else pd.Series(0.0, index=idx)
    )
    ep = (
        validate_numeric_input(
            epi_series,
            component="sofa2_cardio",
            field="epi",
            index=idx,
            minimum=0,
        )
        if epi_series is not None
        else pd.Series(0.0, index=idx)
    )
    da = (
        validate_numeric_input(
            dopamine,
            component="sofa2_cardio",
            field="dopamine",
            index=idx,
            minimum=0,
        )
        if dopamine is not None
        else pd.Series(0.0, index=idx)
    )
    db = (
        validate_numeric_input(
            dobutamine,
            component="sofa2_cardio",
            field="dobutamine",
            index=idx,
            minimum=0,
        )
        if dobutamine is not None
        else pd.Series(0.0, index=idx)
    )
    others = (
        _is_true(
            validate_aligned_input(
                other_vaso,
                component="sofa2_cardio",
                field="other_vaso",
                index=idx,
            )
        )
        if other_vaso is not None
        else pd.Series(False, index=idx)
    )
    mech = (
        _is_true(
            validate_aligned_input(
                mech_circ_support,
                component="sofa2_cardio",
                field="mech_circ_support",
                index=idx,
            )
        )
        if mech_circ_support is not None
        else pd.Series(False, index=idx)
    )
    vaso_unavail = (
        _is_true(
            validate_aligned_input(
                vasopressors_unavailable,
                component="sofa2_cardio",
                field="vasopressors_unavailable",
                index=idx,
            )
        )
        if vasopressors_unavailable is not None
        else pd.Series(False, index=idx)
    )

    # SOFA-2 footnote (i)+(n): ECMO used for a cardiovascular indication (VA-ECMO)
    # is a form of mechanical circulatory support and is scored 4 on the
    # cardiovascular component (in addition to the respiratory 4 handled in
    # `sofa2_resp`). Respiratory-indication (VV) ECMO is NOT scored here. Where the
    # indication is unknown, no cardiovascular point is added (conservative).
    on_ecmo = (
        _is_true(
            validate_aligned_input(
                ecmo,
                component="sofa2_cardio",
                field="ecmo",
                index=idx,
            )
        )
        if ecmo is not None
        else pd.Series(False, index=idx)
    )
    cardiac_ecmo = pd.Series(False, index=idx)
    if ecmo is not None and ecmo_indication is not None:
        indication = validate_aligned_input(
            ecmo_indication,
            component="sofa2_cardio",
            field="ecmo_indication",
            index=idx,
        )
        cardiac_ecmo = on_ecmo & (indication.astype("string") == "cardiovascular")

    # KEY SOFA-2 CHANGE: Combined norepinephrine + epinephrine
    total = ne.fillna(0) + ep.fillna(0)

    score = pd.Series(0, index=idx, dtype=int)

    # Check if any vasopressors/inotropes are being used
    any_vaso = (total > 0) | (da > 0) | (db > 0) | others
    contradiction = vaso_unavail & any_vaso
    if contradiction.any():
        raise SOFA2InputError(
            component="sofa2_cardio",
            field="vasopressors_unavailable",
            reason_code="sofa2_cardio_vasopressor_state_conflict",
            message=(
                "MAP fallback cannot be combined with an observed positive "
                "vasopressor or inotrope state"
            ),
            invalid_count=int(contradiction.sum()),
        )

    # Primary scoring: MAP when no vasopressors/inotropes
    no_vaso_mask = ~any_vaso
    score[no_vaso_mask & (map_val < 70)] = 1

    # Primary norepi+epi rule (SOFA-2 combined dosing)
    ne_ep_mask = total > 0
    score[ne_ep_mask & (total <= 0.2)] = 2  # Low dose
    score[ne_ep_mask & (total > 0.2) & (total <= 0.4)] = 3  # Medium dose
    score[ne_ep_mask & (total > 0.4)] = 4  # High dose

    # Table 2 uses "any other vasopressor or inotrope". Dopamine and
    # dobutamine are therefore adjuncts here, not only in the zero-NE/Epi path.
    adjunct = (da > 0) | (db > 0) | others
    score[(total > 0.2) & (total <= 0.4) & adjunct] = 4
    score[(total > 0) & (total <= 0.2) & adjunct] = 3

    # Any other vasopressor/inotrope (when no norepi+epi)
    no_ne_ep = total == 0
    score[no_ne_ep & adjunct] = 2

    # ALTERNATE: Dopamine-only scoring when norepi+epi == 0 (SOFA-2 backup rule)
    dopamine_only = (total == 0) & (da > 0) & (db <= 0) & (~others)
    score[dopamine_only & (da <= 20)] = 2
    score[dopamine_only & (da > 20) & (da <= 40)] = 3
    score[dopamine_only & (da > 40)] = 4

    # MAP-only fallback when vasopressors unavailable (ceiling of care)
    if vaso_unavail.any():
        score[vaso_unavail & (map_val >= 70)] = 0
        score[vaso_unavail & (map_val >= 60) & (map_val < 70)] = 1
        score[vaso_unavail & (map_val >= 50) & (map_val < 60)] = 2
        score[vaso_unavail & (map_val >= 40) & (map_val < 50)] = 3
        score[vaso_unavail & (map_val < 40)] = 4

    # Mechanical circulatory support (IABP / LVAD / Impella, SOFA-2 footnote n) and
    # cardiovascular-indication ECMO (VA-ECMO, footnotes i+n) are an automatic
    # cardiovascular 4. Apply LAST as a floor so that none of the MAP-, dose-, or
    # ceiling-of-care (footnote m) rules above can downgrade these patients.
    # Footnote (m) governs only the MAP cutoffs when vasoactive drugs are
    # unavailable; it does not override mechanical support.
    score[mech | cardiac_ecmo] = 4

    return score

DELIRIUM_TX_EVIDENCE_STATES = frozenset(
    {"confirmed", "proxy_only", "not_detected", "unavailable"}
)


def _normalize_delirium_tx_evidence(
    index: pd.Index,
    *,
    delirium_tx_evidence: Optional[pd.Series],
    delirium_tx_proxy: Optional[pd.Series],
    delirium_tx: Optional[pd.Series],
) -> pd.Series:
    """Return the explicit four-state delirium-treatment evidence contract.

    ``delirium_tx`` is the historical compatibility alias.  It is deliberately
    treated as a medication-exposure proxy, never as confirmation that the drug
    was prescribed for delirium.  ``not_detected`` means no qualifying evidence
    was detected in an assessable source; it does not mean delirium was absent.
    A negative proxy alone cannot prove that state because source and
    time-window coverage may be incomplete.
    """

    evidence = pd.Series("unavailable", index=index, dtype="string")
    if delirium_tx_evidence is not None:
        aligned = validate_aligned_input(
            delirium_tx_evidence,
            component="sofa2_cns",
            field="delirium_tx_evidence",
            index=index,
        )
        raw = aligned.astype("string")
        normalized = raw.str.strip().str.lower()
        invalid = raw.notna() & ~normalized.isin(DELIRIUM_TX_EVIDENCE_STATES)
        if invalid.any():
            raise SOFA2InputError(
                component="sofa2_cns",
                field="delirium_tx_evidence",
                reason_code="sofa2_cns_delirium_tx_evidence_invalid",
                message="Unknown delirium-treatment evidence state",
                invalid_count=int(invalid.sum()),
            )
        evidence = normalized.fillna("unavailable")

    proxy = pd.Series(False, index=index, dtype=bool)
    if delirium_tx_proxy is not None:
        proxy = _is_true(
            validate_aligned_input(
                delirium_tx_proxy,
                component="sofa2_cns",
                field="delirium_tx_proxy",
                index=index,
            )
        )
    if delirium_tx is not None:
        legacy_proxy = _is_true(
            validate_aligned_input(
                delirium_tx,
                component="sofa2_cns",
                field="delirium_tx",
                index=index,
            )
        )
        proxy = proxy | legacy_proxy

    # Only fill an otherwise unavailable state from exposure evidence.  An
    # explicit ``not_detected`` or ``confirmed`` receipt remains authoritative.
    evidence = evidence.mask((evidence == "unavailable") & proxy, "proxy_only")
    return evidence


def _sofa2_cns_assessment(
    gcs: pd.Series,
    *,
    delirium_tx_proxy: Optional[pd.Series] = None,
    delirium_tx_evidence: Optional[pd.Series] = None,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
    sedated_gcs: Optional[pd.Series] = None,
    pre_sedation_gcs: Optional[pd.Series] = None,
    sedated: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build conservative, sensitivity and ascertainment CNS outputs once.

    GCS-based scoring (same thresholds as SOFA-1):
    ┌────────┬──────────────┬──────────────────────────────────────┐
    │ Score  │ GCS          │ Motor response alternative           │
    ├────────┼──────────────┼──────────────────────────────────────┤
    │   0    │ 15           │ Thumbs-up/fist/peace sign            │
    │   1    │ 13-14        │ Localizing to pain                   │
    │   2    │ 9-12         │ Withdrawal to pain                   │
    │   3    │ 6-8          │ Flexion to pain                      │
    │   4    │ 3-5          │ Extension/no response/myoclonus      │
    └────────┴──────────────┴──────────────────────────────────────┘
    
    SOFA-2 delirium-treatment rule
    - Confirmed treatment attributable to delirium → at least 1 point.
    - Candidate medication exposure without an attributable indication does not
      change the conservative main score.
    - The explicitly named proxy sensitivity score counts ``proxy_only``.
    
    When GCS 3 domains cannot be assessed:
    - Use best motor scale domain score
    - Alternative behavioral responses acceptable (thumbs-up, etc.)
    
    Comparison with SOFA-1:
    - SOFA-1: Only GCS thresholds, no delirium consideration
    - SOFA-2: Adds delirium treatment/assessment criterion
    - SOFA-2: Motor-only alternatives formalized
    
    Args:
        gcs: Glasgow Coma Scale (3-15)
        delirium_tx_proxy: Candidate medication exposure.
        delirium_tx_evidence: One of ``confirmed``, ``proxy_only``,
            ``not_detected`` or ``unavailable``.
        delirium_tx: Deprecated compatibility alias for
            ``delirium_tx_proxy``.  It never implies confirmation.
        delirium_positive: Positive CAM-ICU metadata retained for sensitivity
            analyses; it does not score without delirium treatment
        motor_response: Motor response score when GCS cannot be fully assessed
                       (6=localizing, 5=withdrawal, 4=flexion, 3=extension, 2=no response)
        sedated_gcs/pre_sedation_gcs: Last GCS recorded before sedation
        sedated: Boolean sedation status. When true without a pre-sedation GCS,
            Table 2 footnote c assigns 0 points from GCS.

    Returns:
        DataFrame with the conservative score, proxy sensitivity score,
        ascertainment state and normalized evidence state.

    Notes:
    - Only ``confirmed`` treatment overrides GCS=15 in the main score.
    - ``proxy_only`` overrides GCS=15 only in the sensitivity score.
    - Motor alternatives allow scoring in intubated/non-verbal patients
    - When GCS 3 domains cannot be assessed, use best motor scale domain score
    """
    g = validate_numeric_input(
        gcs,
        component="sofa2_cns",
        field="gcs",
        minimum=3,
        maximum=15,
        integer=True,
    )
    pre_sedation = _coalesce_series(sedated_gcs, pre_sedation_gcs)
    sedated_mask = (
        _is_true(
            validate_aligned_input(
                sedated,
                component="sofa2_cns",
                field="sedated",
                index=g.index,
            )
        )
        if sedated is not None
        else pd.Series(False, index=g.index)
    )
    if pre_sedation is not None:
        pre = validate_numeric_input(
            pre_sedation,
            component="sofa2_cns",
            field="pre_sedation_gcs",
            index=g.index,
            minimum=3,
            maximum=15,
            integer=True,
        )
        # A recorded pre-sedation GCS is the authoritative value. This also
        # supports databases that expose the value without a separate status.
        recorded_pre = pre.notna()
        g = g.where(~recorded_pre, pre)
        sedated_mask = sedated_mask | recorded_pre

    unknown_pre_sedation = pd.Series(False, index=g.index)
    if sedated is not None:
        pre = (
            validate_numeric_input(
                pre_sedation,
                component="sofa2_cns",
                field="pre_sedation_gcs",
                index=g.index,
                minimum=3,
                maximum=15,
                integer=True,
            )
            if pre_sedation is not None
            else pd.Series(np.nan, index=g.index)
        )
        unknown_pre_sedation = sedated_mask & pre.isna()
        # Table 2 footnote c: if the pre-sedation GCS is unknown, assign 0.
        g = g.mask(unknown_pre_sedation)

    base_score = pd.Series(0, index=g.index, dtype=int)

    # Use motor response if GCS cannot be fully assessed
    if motor_response is not None:
        m = validate_numeric_input(
            motor_response,
            component="sofa2_cns",
            field="motor_response",
            index=g.index,
            minimum=1,
            maximum=6,
            integer=True,
        )
        # Map motor response to equivalent GCS scores
        # 6=obeys commands / behavioral command response (~GCS 15),
        # 5=localizing (~GCS 13-14), 4=withdrawal (~GCS 9-12),
        # 3=flexion (~GCS 6-8), 2/1=extension or no response (~GCS 3-5).
        motor_score = pd.Series(0, index=m.index, dtype=int)
        motor_score[m == 5] = 1  # Localizing to pain
        motor_score[m == 4] = 2  # Withdrawal to pain
        motor_score[m == 3] = 3  # Flexion to pain
        motor_score[m <= 2] = 4  # Extension/no response/myoclonus

        # Use motor response when GCS is missing or cannot be assessed
        gcs_available = ~g.isna()
        base_score[~gcs_available] = motor_score[~gcs_available]

    # GCS thresholds (same as SOFA-1)
    base_score[g < 15] = 1
    base_score[g < 13] = 2
    base_score[g < 9] = 3
    base_score[g < 6] = 4
    base_score[unknown_pre_sedation] = 0

    evidence = _normalize_delirium_tx_evidence(
        g.index,
        delirium_tx_evidence=delirium_tx_evidence,
        delirium_tx_proxy=delirium_tx_proxy,
        delirium_tx=delirium_tx,
    )
    confirmed = evidence == "confirmed"
    sensitivity_positive = evidence.isin({"confirmed", "proxy_only"})

    conservative = base_score.copy()
    sensitivity = base_score.copy()
    conservative[confirmed] = np.maximum(conservative[confirmed], 1)
    sensitivity[sensitivity_positive] = np.maximum(
        sensitivity[sensitivity_positive], 1
    )

    # This receipt describes only whether the delirium-treatment clause can be
    # ascertained.  A non-zero GCS score makes that clause irrelevant; it does
    # not make treatment evidence complete.
    ascertainment = pd.Series("not_score_relevant", index=g.index, dtype="string")
    zero_boundary = base_score == 0
    ascertainment[zero_boundary & (evidence == "confirmed")] = "complete"
    ascertainment[zero_boundary & (evidence == "proxy_only")] = "proxy_only"
    ascertainment[
        zero_boundary & (evidence == "not_detected")
    ] = "complete_for_proxy_source"
    ascertainment[zero_boundary & (evidence == "unavailable")] = "unavailable"

    return pd.DataFrame(
        {
            "sofa2_cns": conservative.astype(int),
            "sofa2_cns_proxy_sensitivity": sensitivity.astype(int),
            "sofa2_cns_delirium_tx_ascertainment": ascertainment,
            "delirium_tx_evidence": evidence,
        },
        index=g.index,
    )


def sofa2_cns(
    gcs: pd.Series,
    *,
    delirium_tx_proxy: Optional[pd.Series] = None,
    delirium_tx_evidence: Optional[pd.Series] = None,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
    sedated_gcs: Optional[pd.Series] = None,
    pre_sedation_gcs: Optional[pd.Series] = None,
    sedated: Optional[pd.Series] = None,
) -> pd.Series:
    """Database operationalization of SOFA-2 CNS.

    Unconfirmed delirium-treatment proxies are handled conservatively and are
    exposed separately through the sensitivity and ascertainment outputs.
    """

    return _sofa2_cns_assessment(
        gcs,
        delirium_tx_proxy=delirium_tx_proxy,
        delirium_tx_evidence=delirium_tx_evidence,
        delirium_tx=delirium_tx,
        delirium_positive=delirium_positive,
        motor_response=motor_response,
        sedated_gcs=sedated_gcs,
        pre_sedation_gcs=pre_sedation_gcs,
        sedated=sedated,
    )["sofa2_cns"]


def sofa2_cns_proxy_sensitivity(
    gcs: pd.Series,
    *,
    delirium_tx_proxy: Optional[pd.Series] = None,
    delirium_tx_evidence: Optional[pd.Series] = None,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
    sedated_gcs: Optional[pd.Series] = None,
    pre_sedation_gcs: Optional[pd.Series] = None,
    sedated: Optional[pd.Series] = None,
) -> pd.Series:
    """Sensitivity analysis that treats candidate drug exposure as positive."""

    return _sofa2_cns_assessment(
        gcs,
        delirium_tx_proxy=delirium_tx_proxy,
        delirium_tx_evidence=delirium_tx_evidence,
        delirium_tx=delirium_tx,
        delirium_positive=delirium_positive,
        motor_response=motor_response,
        sedated_gcs=sedated_gcs,
        pre_sedation_gcs=pre_sedation_gcs,
        sedated=sedated,
    )["sofa2_cns_proxy_sensitivity"]


def sofa2_cns_delirium_tx_ascertainment(
    gcs: pd.Series,
    *,
    delirium_tx_proxy: Optional[pd.Series] = None,
    delirium_tx_evidence: Optional[pd.Series] = None,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
    sedated_gcs: Optional[pd.Series] = None,
    pre_sedation_gcs: Optional[pd.Series] = None,
    sedated: Optional[pd.Series] = None,
) -> pd.Series:
    """Return ascertainment of the CNS delirium-treatment clause."""

    return _sofa2_cns_assessment(
        gcs,
        delirium_tx_proxy=delirium_tx_proxy,
        delirium_tx_evidence=delirium_tx_evidence,
        delirium_tx=delirium_tx,
        delirium_positive=delirium_positive,
        motor_response=motor_response,
        sedated_gcs=sedated_gcs,
        pre_sedation_gcs=pre_sedation_gcs,
        sedated=sedated,
    )["sofa2_cns_delirium_tx_ascertainment"]


def sofa2_cns_ascertainment(
    gcs: pd.Series,
    *,
    delirium_tx_proxy: Optional[pd.Series] = None,
    delirium_tx_evidence: Optional[pd.Series] = None,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
    sedated_gcs: Optional[pd.Series] = None,
    pre_sedation_gcs: Optional[pd.Series] = None,
    sedated: Optional[pd.Series] = None,
) -> pd.Series:
    """Deprecated alias for :func:`sofa2_cns_delirium_tx_ascertainment`."""

    return sofa2_cns_delirium_tx_ascertainment(
        gcs,
        delirium_tx_proxy=delirium_tx_proxy,
        delirium_tx_evidence=delirium_tx_evidence,
        delirium_tx=delirium_tx,
        delirium_positive=delirium_positive,
        motor_response=motor_response,
        sedated_gcs=sedated_gcs,
        pre_sedation_gcs=pre_sedation_gcs,
        sedated=sedated,
    )


def sofa2_renal(
    crea: Optional[pd.Series] = None,
    *,
    creatinine: Optional[pd.Series] = None,
    rrt: Optional[pd.Series] = None,
    rrt_criteria: Optional[pd.Series] = None,
    rrt_episode_active: Optional[pd.Series] = None,
    rrt_nonrenal_only: Optional[pd.Series] = None,
    uo_6h: Optional[pd.Series] = None,
    uo_12h: Optional[pd.Series] = None,
    uo_24h: Optional[pd.Series] = None,
    urine_mlkgph: Optional[pd.Series] = None,
    urine_duration_h: Optional[pd.Series] = None,
    potassium: Optional[pd.Series] = None,
    ph: Optional[pd.Series] = None,
    bicarb: Optional[pd.Series] = None,
    bicarbonate: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 renal component.

    MAJOR CHANGE: RRT auto-scores 4pt; urine standardized to mL/kg/h

    Scoring criteria (from SOFA-2 table):
    ┌────────┬────────────────────┬─────────────────────────────────┬─────┐
    │ Score  │ Creatinine         │ Urine output                    │ RRT │
    ├────────┼────────────────────┼─────────────────────────────────┼─────┤
    │   0    │ ≤1.20 mg/dL        │ Normal                          │ No  │
    │        │ (≤110 μmol/L)      │                                 │     │
    │   1    │ ≤2.0 mg/dL         │ OR <0.5 mL/kg/h (6-12h)        │ No  │
    │        │ (≤170 μmol/L)      │                                 │     │
    │   2    │ ≤3.50 mg/dL        │ OR <0.5 mL/kg/h (≥12h)         │ No  │
    │        │ (≤300 μmol/L)      │                                 │     │
    │   3    │ >3.50 mg/dL        │ OR <0.3 mL/kg/h (≥24h)         │ No  │
    │        │ (>300 μmol/L)      │ OR anuria ≥12h                  │     │
    │   4    │ Any                │ Any                             │ Yes │
    └────────┴────────────────────┴─────────────────────────────────┴─────┘

    RRT criteria (score 4pt - receiving or fulfils criteria for RRT):
    - Includes chronic RRT use
    - Excludes patients receiving RRT ONLY for non-renal causes
    - Meets criteria if: creatinine >1.2 AND oliguria + (K≥6.0 OR pH≤7.20 + HCO3≤12)

    Intermittent RRT:
    - Score 4pt on BOTH treatment AND non-treatment days
    - Continue until RRT permanently discontinued
    
    Comparison with SOFA-1:
    ┌──────────────────┬────────────────────┬─────────────────────┐
    │ Aspect           │ SOFA-1             │ SOFA-2              │
    ├──────────────────┼────────────────────┼─────────────────────┤
    │ Urine metric     │ mL/day (absolute)  │ mL/kg/h (body wt)   │
    │ 4pt oliguria     │ <200 mL/day        │ <0.3 mL/kg/h (24h)  │
    │ 3pt oliguria     │ <500 mL/day        │ <0.3 mL/kg/h (24h)  │
    │ RRT              │ Not scored         │ Auto 4pt            │
    │ Body weight      │ Not considered     │ Standardized        │
    └──────────────────┴────────────────────┴─────────────────────┘
    
    Args:
        crea: Serum creatinine (mg/dL) - EasyICU/runtime-style name
        creatinine: Serum creatinine (mg/dL) - readable alias
        rrt: Boolean - receiving RRT at this time
        rrt_criteria: Boolean - meets RRT criteria but not receiving it
        rrt_episode_active: Boolean state that remains true on intermittent
            non-treatment days until documented permanent termination, including
            chronic RRT
        rrt_nonrenal_only: Boolean - RRT was provided solely for a non-renal
            indication and must not trigger the 4-point RRT rule
        uo_6h: 6-hour average urine output (mL/kg/h) - EasyICU runtime concept name
        uo_12h: 12-hour average urine output (mL/kg/h) - EasyICU runtime concept name
        uo_24h: 24-hour average urine output (mL/kg/h) - EasyICU runtime concept name
        urine_mlkgph: Urine output rate (mL/kg/h) - readable fallback API
        urine_duration_h: Duration of urine measurement period (hours) - readable fallback API
        potassium: Serum potassium (mmol/L) - for RRT criteria
        ph: Arterial pH - for RRT criteria
        bicarb: Serum bicarbonate (mmol/L) - EasyICU runtime concept name
        bicarbonate: Serum bicarbonate (mmol/L) - readable alias

    Returns:
        Series of renal SOFA-2 scores (0-4)

    Notes:
    - If urine_mlkgph not available, use creatinine-only scoring
    - RRT overrides all other criteria → auto 4pt
    - For intermittent RRT: keep scoring 4pt until permanently stopped
    - Anuria defined as 0 mL for ≥12h
    - RRT criteria check: creatinine >1.2 + oliguria + (K≥6.0 OR pH≤7.20 + HCO3≤12)
    - Unit conversion: mg/dL × 88.4 = μmol/L
    """
    creatinine_series = _coalesce_series(crea, creatinine)
    if creatinine_series is None:
        raise ValueError("sofa2_renal requires `crea` or `creatinine`")
    c = validate_numeric_input(
        creatinine_series,
        component="sofa2_renal",
        field="creatinine",
        minimum=0,
    )
    idx = c.index
    score = pd.Series(0, index=idx, dtype=int)
    bicarbonate_series = _coalesce_series(bicarb, bicarbonate)

    def renal_numeric(
        value: Optional[pd.Series],
        field: str,
        *,
        minimum: float = 0,
        maximum: float | None = None,
    ) -> pd.Series:
        if value is None:
            return pd.Series(np.nan, index=idx)
        return validate_numeric_input(
            value,
            component="sofa2_renal",
            field=field,
            index=idx,
            minimum=minimum,
            maximum=maximum,
        )

    u6 = renal_numeric(uo_6h, "uo_6h")
    u12 = renal_numeric(uo_12h, "uo_12h")
    u24 = renal_numeric(uo_24h, "uo_24h")
    urine_rate = renal_numeric(urine_mlkgph, "urine_mlkgph")
    urine_duration = renal_numeric(urine_duration_h, "urine_duration_h")
    potassium_value = renal_numeric(potassium, "potassium")
    ph_value = renal_numeric(ph, "ph", maximum=14)
    bicarbonate_value = renal_numeric(bicarbonate_series, "bicarbonate")

    # Table 2 footnotes o/q: an episode remains active between intermittent
    # sessions until termination; therapy solely for a non-renal indication is
    # excluded. Raw treatment-event evidence alone cannot manufacture off-days,
    # so callers must pass the explicit episode state when they have it.
    treatment_state = pd.Series(False, index=idx)
    for field, candidate in (("rrt", rrt), ("rrt_episode_active", rrt_episode_active)):
        if candidate is not None:
            treatment_state = treatment_state | _is_true(
                validate_aligned_input(
                    candidate,
                    component="sofa2_renal",
                    field=field,
                    index=idx,
                )
            )
    criteria_state = (
        _is_true(
            validate_aligned_input(
                rrt_criteria,
                component="sofa2_renal",
                field="rrt_criteria",
                index=idx,
            )
        )
        if rrt_criteria is not None
        else pd.Series(False, index=idx)
    )
    nonrenal_only = (
        _is_true(
            validate_aligned_input(
                rrt_nonrenal_only,
                component="sofa2_renal",
                field="rrt_nonrenal_only",
                index=idx,
            )
        )
        if rrt_nonrenal_only is not None
        else pd.Series(False, index=idx)
    )
    # A non-renal-only indication excludes treatment evidence, but it cannot
    # suppress an independently documented renal RRT criterion.
    score[(treatment_state & ~nonrenal_only) | criteria_state] = 4

    # EasyICU runtime-aligned path: use windowed urine concepts when available.
    if any(series is not None for series in (uo_6h, uo_12h, uo_24h)):
        score[(c > 1.20) | ((u6 < 0.5) & ~(u12 < 0.5))] = np.maximum(score[(c > 1.20) | ((u6 < 0.5) & ~(u12 < 0.5))], 1)
        score[(c > 2.0) | (u12 < 0.5)] = np.maximum(score[(c > 2.0) | (u12 < 0.5)], 2)
        score[(c > 3.50) | (u24 < 0.3) | (u12 == 0)] = np.maximum(score[(c > 3.50) | (u24 < 0.3) | (u12 == 0)], 3)

        if (potassium is not None) and (ph is not None) and (bicarbonate_series is not None):
            base_injury = (c > 1.2) | (u6 < 0.3)
            metabolic_crisis = (potassium_value >= 6.0) | (
                (ph_value <= 7.20) & (bicarbonate_value <= 12)
            )
            score[base_injury & metabolic_crisis] = 4

        return score

    # Check if patient meets RRT criteria but not receiving RRT (e.g., ceiling of care)
    if (potassium is not None) and (ph is not None) and (bicarbonate_series is not None) and (urine_mlkgph is not None):
        # SOFA-2 footnote (p): a patient NOT on RRT scores 4 if they meet RRT
        # criteria, i.e. (creatinine >1.2 mg/dL OR oliguria <0.3 mL/kg/h for >6 h)
        # AND (K ≥6.0 mmol/L OR (pH ≤7.20 AND HCO3 ≤12 mmol/L)).
        # NOTE: in the EasyICU runtime this fallback is not reached — the dictionary
        # wires the windowed urine concepts (uo_6h/uo_12h/uo_24h) and the windowed
        # path above already implements footnote (p) with OR and the 6 h window via
        # the u6 concept. This fallback is for direct API callers; aligned here for
        # consistency.
        oliguria = (urine_rate < 0.3) & (
            urine_duration > 6
        )  # <0.3 mL/kg/h sustained for >6 h
        metabolic_crisis = (potassium_value >= 6.0) | (
            (ph_value <= 7.20) & (bicarbonate_value <= 12)
        )
        rrt_criteria = ((c > 1.2) | oliguria) & metabolic_crisis

        # Score 4pt if meets RRT criteria but not receiving RRT
        score[rrt_criteria & (score < 4)] = 4

    # Urine output criteria (body weight standardized)
    if urine_mlkgph is not None:
        # Anuria (0 mL for ≥12h) → 3pt
        anuria = (urine_rate == 0) & (urine_duration >= 12)
        score[anuria] = np.maximum(score[anuria], 3)
        
        # <0.3 mL/kg/h for ≥24h → 3pt
        score[(urine_rate < 0.3) & (urine_duration >= 24)] = np.maximum(score[(urine_rate < 0.3) & (urine_duration >= 24)], 3)
        
        # <0.5 mL/kg/h for ≥12h → 2pt
        score[(urine_rate < 0.5) & (urine_duration >= 12) & (score < 3)] = np.maximum(score[(urine_rate < 0.5) & (urine_duration >= 12) & (score < 3)], 2)
        
        # <0.5 mL/kg/h for 6-12h → 1pt
        score[(urine_rate < 0.5) & (urine_duration >= 6) & (urine_duration < 12) & (score < 2)] = np.maximum(score[(urine_rate < 0.5) & (urine_duration >= 6) & (urine_duration < 12) & (score < 2)], 1)

    # Creatinine thresholds according to SOFA-2 table
    # Note: Table shows ≤1.20/≤2.0/≤3.50/>3.50, meaning boundaries at these values
    score[c > 1.20] = np.maximum(score[c > 1.20], 1)
    score[c > 2.0] = np.maximum(score[c > 2.0], 2)
    score[c > 3.50] = np.maximum(score[c > 3.50], 3)

    return score

# Alias matching the EasyICU concept name used in the subset dictionary.
sofa2 = sofa2_score
