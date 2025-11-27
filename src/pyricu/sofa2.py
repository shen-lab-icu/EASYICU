"""SOFA-2 score callbacks and helpers.

This module implements SOFA-2 (2025 consensus, Moreno et al. JAMA Network Open)
organ component rules and an aggregate scorer compatible with the existing pyricu
callback style.

Components implemented (0-4 each):
- Respiratory (PaO2/FiO2 or SpO2/FiO2 + advanced support/ECMO)
  * SOFA-2 thresholds: ≤300/225/150/75 mmHg (vs SOFA-1: >400/300-400/200-299/100-199/<100)
  * ECMO for respiratory indication → auto 4pt
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
- 24-hour window scoring: each organ's maximum score within 24h is summed
- Missing data handling:
  * Day 1: score as 0 (assume normal)
  * After day 1: carry forward last observation (assume stability)
- Drug requirements: continuous IV infusion ≥1 hour for vasopressors/inotropes
- Transient changes (<1 hour, e.g., post-suction hypoxemia) should not be scored

References:
- Moreno et al. (2025). SOFA-2 Consensus Statement. JAMA Network Open.
- Vincent et al. (1996). Original SOFA score. Intensive Care Medicine.
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd

def _is_true(series: pd.Series) -> pd.Series:
    """Replicate R's is_true: non-NA and True."""
    return series.fillna(False).astype(bool)

def sofa2_resp(
    pafi: Optional[pd.Series] = None,
    *,
    spo2: Optional[pd.Series] = None,
    fio2: Optional[pd.Series] = None,
    adv_resp: Optional[pd.Series] = None,
    ecmo: Optional[pd.Series] = None,
    ecmo_indication: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 respiratory component.

    Priority of oxygenation metric: use PaO2/FiO2 if available; otherwise
    derive SpO2/FiO2 when both are present (only when SpO2 < 98%).
    
    SOFA-2 thresholds for P/F (mmHg) vs SOFA-1:
    ┌────────┬─────────────┬───────────────┬─────────────────────┐
    │ Score  │ SOFA-1 P/F  │ SOFA-2 P/F    │ SOFA-2 Requirements │
    ├────────┼─────────────┼───────────────┼─────────────────────┤
    │   0    │ >400        │ >300          │ None                │
    │   1    │ 300-400     │ ≤300          │ None                │
    │   2    │ 200-299     │ ≤225          │ None                │
    │   3    │ 100-199+MV  │ ≤150          │ Advanced support^   │
    │   4    │ <100+MV     │ ≤75           │ Advanced support^   │
    │        │             │ OR ECMO (respiratory indication)        │
    └────────┴─────────────┴───────────────┴─────────────────────┘

    ^ Advanced support: HFNC, CPAP, BiPAP, NIV, IMV, long-term home ventilation

    SpO2/FiO2 alternative thresholds (when SpO2 < 98%):
    - 0: >300  │ 1: ≤300  │ 2: ≤250  │ 3: ≤200+support  │ 4: ≤120+support or ECMO
    
    ECMO special rules:
    - If ECMO for respiratory indication → auto 4pt (regardless of P/F)
    - If ECMO for cardiovascular indication → score both respiratory AND cardiovascular
    
    Args:
        pafi: PaO2/FiO2 ratio (mmHg)
        spo2: Oxygen saturation (%)
        fio2: Fraction of inspired oxygen (0.21-1.0 or 21-100)
        adv_resp: Boolean - advanced respiratory support active
        ecmo: Boolean - ECMO in use
        ecmo_indication: String - 'respiratory' or 'cardiovascular'
        
    Returns:
        Series of respiratory SOFA-2 scores (0-4)
    """
    # Build support/ECMO masks
    support = _is_true(adv_resp) if adv_resp is not None else pd.Series(False, index=(pafi.index if pafi is not None else (spo2.index if spo2 is not None else fio2.index)))
    on_ecmo = _is_true(ecmo) if ecmo is not None else pd.Series(False, index=support.index)
    
    # Determine if ECMO is for respiratory indication
    ecmo_resp = pd.Series(False, index=support.index)
    if ecmo_indication is not None and ecmo is not None:
        ecmo_resp = _is_true(ecmo) & (ecmo_indication == 'respiratory')

    idx = None
    if pafi is not None:
        metric = pd.to_numeric(pafi, errors="coerce")
        idx = metric.index
        use_sf = False
    else:
        # Derive S/F if possible (only when SpO2 < 98% per SOFA-2 guidelines)
        if spo2 is not None and fio2 is not None:
            s = pd.to_numeric(spo2, errors="coerce")
            f = pd.to_numeric(fio2, errors="coerce")
            # Convert fio2 percent to fraction if looks like 21-100
            f_adj = f.copy()
            f_adj[(f_adj > 1) & (f_adj <= 100)] = f_adj[(f_adj > 1) & (f_adj <= 100)] / 100.0
            with np.errstate(invalid="ignore", divide="ignore"):
                metric = s / f_adj
            # Only use SpO2/FiO2 when SpO2 < 98%
            metric[s >= 98] = np.nan
            idx = metric.index
            use_sf = True
        else:
            raise ValueError("sofa2_resp requires either pafi or (spo2 and fio2)")

    score = pd.Series(0, index=idx, dtype=int)
    sup_or_ecmo = _is_true(support | on_ecmo)

    if not 'use_sf' in locals():
        use_sf = False

    if not use_sf:
        # P/F thresholds (SOFA-2)
        score[(metric <= 300)] = 1
        score[(metric <= 225)] = 2
        # For 3/4, advanced support or ECMO is required
        score[(metric <= 150) & sup_or_ecmo] = 3
        score[(metric <= 75) & sup_or_ecmo] = 4
    else:
        # S/F thresholds (SOFA-2 alternative)
        score[(metric <= 300)] = 1
        score[(metric <= 250)] = 2
        score[(metric <= 200) & sup_or_ecmo] = 3
        score[(metric <= 120) & sup_or_ecmo] = 4
    
    # ECMO for respiratory indication → auto 4pt
    score[ecmo_resp] = 4

    return score

def sofa2_coag(plt: pd.Series) -> pd.Series:
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
    p = pd.to_numeric(plt, errors="coerce")
    score = pd.Series(0, index=plt.index, dtype=int)
    score[p <= 150] = 1
    score[p <= 100] = 2
    score[p <= 80] = 3
    score[p <= 50] = 4
    return score

def sofa2_liver(bili: pd.Series) -> pd.Series:
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
    b = pd.to_numeric(bili, errors="coerce")
    score = pd.Series(0, index=bili.index, dtype=int)

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
    norepi60: Optional[pd.Series] = None,
    epi60: Optional[pd.Series] = None,
    dopamine60: Optional[pd.Series] = None,
    dobutamine60: Optional[pd.Series] = None,
    other_vaso: Optional[pd.Series] = None,
    mech_circ_support: Optional[pd.Series] = None,
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
        norepi60: Norepinephrine dose (μg/kg/min) - use BASE dose
        epi60: Epinephrine dose (μg/kg/min)
        dopamine60: Dopamine dose (μg/kg/min) - backup only
        dobutamine60: Dobutamine dose (μg/kg/min)
        other_vaso: Boolean - other vasoactive drugs present
        mech_circ_support: Boolean - mechanical circulatory support active
        vasopressors_unavailable: Boolean - vasopressors unavailable/precluded

    Returns:
        Series of cardiovascular SOFA-2 scores (0-4)
    """
    idx = map.index
    ne = pd.to_numeric(norepi60, errors="coerce") if norepi60 is not None else pd.Series(0.0, index=idx)
    ep = pd.to_numeric(epi60, errors="coerce") if epi60 is not None else pd.Series(0.0, index=idx)
    da = pd.to_numeric(dopamine60, errors="coerce") if dopamine60 is not None else pd.Series(0.0, index=idx)
    db = pd.to_numeric(dobutamine60, errors="coerce") if dobutamine60 is not None else pd.Series(0.0, index=idx)
    others = _is_true(other_vaso) if other_vaso is not None else pd.Series(False, index=idx)
    mech = _is_true(mech_circ_support) if mech_circ_support is not None else pd.Series(False, index=idx)
    vaso_unavail = _is_true(vasopressors_unavailable) if vasopressors_unavailable is not None else pd.Series(False, index=idx)

    # KEY SOFA-2 CHANGE: Combined norepinephrine + epinephrine
    total = ne.fillna(0) + ep.fillna(0)
    map_val = pd.to_numeric(map, errors="coerce")

    score = pd.Series(0, index=idx, dtype=int)

    # Mechanical support overrides → auto 4pt
    score[mech] = 4

    # Check if any vasopressors/inotropes are being used
    any_vaso = (total > 0) | (da > 0) | (db > 0) | others

    # Primary scoring: MAP when no vasopressors/inotropes
    no_vaso_mask = ~any_vaso
    score[no_vaso_mask & (map_val < 70)] = 1

    # Primary norepi+epi rule (SOFA-2 combined dosing)
    ne_ep_mask = total > 0
    score[ne_ep_mask & (total <= 0.2)] = 2  # Low dose
    score[ne_ep_mask & (total > 0.2) & (total <= 0.4)] = 3  # Medium dose
    score[ne_ep_mask & (total > 0.4)] = 4  # High dose

    # Escalate if medium + other vasoactive drugs → 4pt
    score[(total > 0.2) & (total <= 0.4) & others] = 4
    # Escalate if low dose + other vasoactive drugs → 3pt
    score[(total > 0) & (total <= 0.2) & others] = 3

    # Any other vasopressor/inotrope (when no norepi+epi)
    no_ne_ep = (total == 0) & (da.fillna(0) == 0)
    score[no_ne_ep & ((db > 0) | others)] = 2

    # ALTERNATE: Dopamine-only scoring when norepi+epi == 0 (SOFA-2 backup rule)
    dopamine_only = (total == 0) & (da > 0)
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

    return score

def sofa2_cns(
    gcs: pd.Series,
    *,
    delirium_tx: Optional[pd.Series] = None,
    delirium_positive: Optional[pd.Series] = None,
    motor_response: Optional[pd.Series] = None,
) -> pd.Series:
    """SOFA-2 brain/CNS component.

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
    
    NEW in SOFA-2: Delirium treatment/assessment rule
    - If receiving delirium treatment drugs → score 1pt even if GCS=15
    - If CAM-ICU positive → score 1pt even if GCS=15
    - Delirium drugs (PADIS Guidelines):
      * Haloperidol, quetiapine, olanzapine, risperidone
      * Dexmedetomidine (if used for delirium)
    - Applies to short-term OR long-term treatment
    
    When GCS 3 domains cannot be assessed:
    - Use best motor scale domain score
    - Alternative behavioral responses acceptable (thumbs-up, etc.)
    
    Comparison with SOFA-1:
    - SOFA-1: Only GCS thresholds, no delirium consideration
    - SOFA-2: Adds delirium treatment/assessment criterion
    - SOFA-2: Motor-only alternatives formalized
    
    Args:
        gcs: Glasgow Coma Scale (3-15)
        delirium_tx: Boolean - receiving delirium treatment
        delirium_positive: Boolean - positive CAM-ICU or delirium assessment
        motor_response: Motor response score when GCS cannot be fully assessed
                       (6=localizing, 5=withdrawal, 4=flexion, 3=extension, 2=no response)

    Returns:
        Series of brain/CNS SOFA-2 scores (0-4)

    Notes:
    - Delirium treatment OR positive assessment overrides GCS=15 to minimum 1pt
    - Motor alternatives allow scoring in intubated/non-verbal patients
    - When GCS 3 domains cannot be assessed, use best motor scale domain score
    """
    g = pd.to_numeric(gcs, errors="coerce")

    score = pd.Series(0, index=g.index, dtype=int)

    # Use motor response if GCS cannot be fully assessed
    if motor_response is not None:
        m = pd.to_numeric(motor_response, errors="coerce")
        # Map motor response to equivalent GCS scores
        # 6=localizing (~GCS 13-14), 5=withdrawal (~GCS 9-12), 4=flexion (~GCS 6-8),
        # 3=extension (~GCS 3-5), 2=no response (~GCS 3-5)
        motor_score = pd.Series(0, index=m.index, dtype=int)
        motor_score[m == 6] = 1  # Localizing to pain
        motor_score[m == 5] = 2  # Withdrawal to pain
        motor_score[m == 4] = 3  # Flexion to pain
        motor_score[m <= 3] = 4  # Extension/no response/myoclonus

        # Use motor response when GCS is missing or cannot be assessed
        gcs_available = ~g.isna()
        score[~gcs_available] = motor_score[~gcs_available]

    # GCS thresholds (same as SOFA-1)
    score[g < 15] = 1
    score[g < 13] = 2
    score[g < 9] = 3
    score[g < 6] = 4

    # SOFA-2 NEW: Delirium treatment rule
    # If receiving delirium treatment and GCS==15, upgrade to 1pt
    if delirium_tx is not None:
        dtx = _is_true(delirium_tx)
        mask = (g == 15) & dtx
        score[mask] = np.maximum(score[mask], 1)

    # SOFA-2 NEW: Positive delirium assessment rule
    # If CAM-ICU positive and GCS==15, upgrade to 1pt
    if delirium_positive is not None:
        dp = _is_true(delirium_positive)
        mask = (g == 15) & dp
        score[mask] = np.maximum(score[mask], 1)

    return score

def sofa2_renal(
    crea: pd.Series,
    *,
    rrt: Optional[pd.Series] = None,
    urine_mlkgph: Optional[pd.Series] = None,
    urine_duration_h: Optional[pd.Series] = None,
    potassium: Optional[pd.Series] = None,
    ph: Optional[pd.Series] = None,
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
        crea: Serum creatinine (mg/dL)
        rrt: Boolean - receiving RRT
        urine_mlkgph: Urine output rate (mL/kg/h)
        urine_duration_h: Duration of urine measurement period (hours)
        potassium: Serum potassium (mmol/L) - for RRT criteria
        ph: Arterial pH - for RRT criteria
        bicarbonate: Serum bicarbonate (mmol/L) - for RRT criteria

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
    c = pd.to_numeric(crea, errors="coerce")
    idx = c.index
    score = pd.Series(0, index=idx, dtype=int)

    # RRT = auto 4pt (SOFA-2 major addition)
    if rrt is not None:
        score[_is_true(rrt)] = 4

    # Check if patient meets RRT criteria but not receiving RRT (e.g., ceiling of care)
    if (potassium is not None) and (ph is not None) and (bicarbonate is not None) and (urine_mlkgph is not None):
        k = pd.to_numeric(potassium, errors="coerce")
        ph_val = pd.to_numeric(ph, errors="coerce")
        hco3 = pd.to_numeric(bicarbonate, errors="coerce")
        u = pd.to_numeric(urine_mlkgph, errors="coerce")

        # RRT criteria: creatinine >1.2 AND oliguria + (K≥6.0 OR pH≤7.20 + HCO3≤12)
        oliguria = u < 0.3  # <0.3 mL/kg/h
        metabolic_crisis = (k >= 6.0) | ((ph_val <= 7.20) & (hco3 <= 12))
        rrt_criteria = (c > 1.2) & oliguria & metabolic_crisis

        # Score 4pt if meets RRT criteria but not receiving RRT
        score[rrt_criteria & (score < 4)] = 4

    # Urine output criteria (body weight standardized)
    if urine_mlkgph is not None:
        u = pd.to_numeric(urine_mlkgph, errors="coerce")
        dur = pd.to_numeric(urine_duration_h, errors="coerce") if urine_duration_h is not None else pd.Series(np.nan, index=idx)
        
        # Anuria (0 mL for ≥12h) → 3pt
        anuria = (u == 0) & (dur >= 12)
        score[anuria] = np.maximum(score[anuria], 3)
        
        # <0.3 mL/kg/h for ≥24h → 3pt
        score[(u < 0.3) & (dur >= 24)] = np.maximum(score[(u < 0.3) & (dur >= 24)], 3)
        
        # <0.5 mL/kg/h for ≥12h → 2pt
        score[(u < 0.5) & (dur >= 12) & (score < 3)] = np.maximum(score[(u < 0.5) & (dur >= 12) & (score < 3)], 2)
        
        # <0.5 mL/kg/h for 6-12h → 1pt
        score[(u < 0.5) & (dur >= 6) & (dur < 12) & (score < 2)] = np.maximum(score[(u < 0.5) & (dur >= 6) & (dur < 12) & (score < 2)], 1)

    # Creatinine thresholds according to SOFA-2 table
    # Note: Table shows ≤1.20/≤2.0/≤3.50/>3.50, meaning boundaries at these values
    score[c > 1.20] = np.maximum(score[c > 1.20], 1)
    score[c > 2.0] = np.maximum(score[c > 2.0], 2)
    score[c > 3.50] = np.maximum(score[c > 3.50], 3)

    return score

def sofa2_score(data_dict: Dict[str, pd.DataFrame], *, keep_components: bool = False) -> pd.DataFrame:
    """Aggregate SOFA-2 score from component DataFrames.

    Expected component keys in data_dict:
    - sofa2_resp, sofa2_coag, sofa2_liver, sofa2_cardio, sofa2_cns, sofa2_renal

    Returns a DataFrame with a 'sofa2' column (and optional *_comp columns).
    """
    required = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]

    result = None
    for comp in required:
        if comp not in data_dict:
            raise ValueError(f"Missing required component: {comp}")
        df = data_dict[comp].copy()
        if result is None:
            result = df
        else:
            id_cols = [col for col in df.columns if col in result.columns and col != comp]
            result = pd.merge(result, df, on=id_cols, how="outer")

    result["sofa2"] = result[required].fillna(0).sum(axis=1).astype(int)

    if keep_components:
        for comp in required:
            result[f"{comp}_comp"] = result[comp]
    else:
        result = result.drop(columns=required)

    return result
