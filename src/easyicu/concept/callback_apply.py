"""Concept-source callback application (extracted 2026-05-17, Phase 2).

This module hosts ``_apply_callback`` — the dispatcher that maps a
``ConceptSource.callback`` expression string to an inline DataFrame
transform. It used to live at the tail of :mod:`easyicu.concept`
(former lines ~7676-10147, ~2,470 LOC) before Phase 2 of the
``concept.py`` split documented in CLAUDE.md.

Three-layer callback dispatch
-----------------------------
``_apply_callback`` is only one of THREE callback dispatchers in
EasyICU; it handles **source-level** data-shape transforms. The
other two live elsewhere and must not be confused with this one:

1. ``_apply_callback`` (this module) — ``transform_fun(...)``,
   ``convert_unit(...)``, ``apply_map(...)``, the per-database rate /
   duration callbacks (``aumc_rate_kg``, ``hirid_rate``, ...), and a
   handful of identity-style passthroughs (~100 distinct callback
   strings in the shipped dictionaries).
2. :data:`easyicu.concept_callbacks.CALLBACK_REGISTRY` — the
   **concept-level** registry for derived / composite scores
   (``sofa_score``, ``kdigo_aki``, ``qsofa_score``, ``sep3``, ...).
3. :meth:`easyicu.concept.ConceptResolver._load_single_concept` — the
   **single-concept loader** which special-cases ``los_callback`` and
   any callback containing ``fwd_concept(...)`` because those need
   access to other concepts at load time.

If you're adding a new callback, decide which layer it belongs to and
add it there — do NOT replicate the dispatch logic.

Public surface
--------------
The function is re-exported from :mod:`easyicu.concept` as
``_apply_callback`` so existing callers (notably ``easyicu.base`` at
former L833) keep working. When a future deprecation cycle opens, the
underscore prefix can be shed.

Tests
-----
See ``tests/test_apply_callback.py`` for the dispatch-surface tripwire
that runs every callback string in ``concept-dict.json`` /
``sofa2-dict.json`` through this function. Phase 2's safe-extraction
property relies on that tripwire to detect any branch that goes
missing during the move.
"""

from __future__ import annotations

import logging
import operator
import re
from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .schema import ConceptSource
from .errors import ConceptExtractionUnavailable
from .loader import _get_concept_bounds
from .expr_parser import (
    _apply_binary_op,
    _parse_binary_op,
    _parse_literal,
    _parse_mapping,
    _parse_r_value,
    _split_arguments,
    _strip_quotes,
)
from ..datasource import _duckdb_path, _enumerate_bucket_parquet_files

if TYPE_CHECKING:
    from ..datasource import ICUDataSource
    from . import ConceptResolver

DEBUG_MODE = False
logger = logging.getLogger(__name__)


def _duckdb_sql_path_literal(path) -> str:
    """Quote a filesystem path for a DuckDB SQL string literal."""

    normalized = str(path).replace("\\", "/")
    return "'" + normalized.replace("'", "''") + "'"


def hirid_observation_read_exprs(
    base_path,
) -> Optional[Tuple[str, str, str]]:
    """Return DuckDB read expressions for either supported HiRID layout.

    Converted HiRID data may use item buckets (``observations_bucket``) or
    the public archive's native numbered shards (``observations/N.parquet``).
    Both contain the same observation schema and can time deaths.
    """
    for directory_name in ("observations_bucket", "observations"):
        directory = base_path / directory_name
        files = _enumerate_bucket_parquet_files(directory)
        if not files:
            continue
        files_sql = (
            "["
            + ", ".join(_duckdb_sql_path_literal(path) for path in files)
            + "]"
        )
        return (
            f"read_parquet({files_sql})",
            f"read_parquet({files_sql}, union_by_name=true)",
            directory_name,
        )
    return None


def cohort_patient_ids(patient_ids) -> Optional[set]:
    """Normalize a caller's cohort selector to a set, or ``None`` for "all".

    ``patient_ids`` reaches the concept layer as a list, as a ``{id_col: ids}``
    mapping, or absent. **Only the top-level absence means "every patient".** An
    explicitly empty selector is an empty cohort, which is a different question
    with a different answer, and the package's own normalizers
    (``api.concepts._patient_filter_values``, ``scores.outcomes._patient_values``)
    already keep the two apart. Collapsing ``[]`` into "all" here made this
    helper answer for the whole database when the caller had asked about
    nobody -- the same class of mistake as the guard it was written to fix.

    ``{id_col: None}`` is refused rather than read as either one. It has no
    settled meaning: read as "all" it widens a filtered request to the whole
    database, read as "none" it empties it, and the two reference normalizers
    above accept neither -- both reach ``list(None)`` and raise. Guessing here
    would make this helper the one place in the package where that shape
    silently acquires a population.
    """

    if patient_ids is None:
        return None
    if isinstance(patient_ids, dict):
        if not patient_ids:
            return set()
        column, values = next(iter(patient_ids.items()))
        if values is None:
            raise ValueError(
                f"patient_ids={{{column!r}: None}} does not select a cohort: "
                "pass patient_ids=None for every patient, or an explicit "
                "sequence (possibly empty) for a filtered one."
            )
    else:
        values = patient_ids
    if isinstance(values, (str, bytes, int)):
        return {values}
    try:
        return set(values)
    except TypeError:
        return {values}


def deaths_within_cohort(dead_pids, patient_ids) -> set:
    """The recorded deaths that fall inside the cohort actually being asked about.

    An outcome concept must fail closed when it cannot see deaths that exist —
    but "exist" has to mean *in this cohort*. A guard written against every
    death in the source answers a different question than the caller asked:
    for a cohort of survivors it reports a failure while the correct answer,
    zero, was available. Narrowing first keeps both halves honest — a real
    zero stays a zero, and an unreadable death still raises.
    """

    cohort = cohort_patient_ids(patient_ids)
    if cohort is None:
        return set(dead_pids)
    return {pid for pid in dead_pids if pid in cohort}


def _refuse_untimed_deaths(
    *, database: str, concept_id: str, timing_ids, timed: int, untimed
) -> None:
    """Refuse a mortality that silently omits deaths it could not place in time.

    A death that cannot be timed does not come back from the query, so it is
    absent from the result and the mortality computed downstream is lower than
    the source says -- with nothing anywhere to show a number was lost. That is
    the defect this module exists to refuse, and a partial loss is the same
    defect as a total one: only the size differs.

    An earlier version warned instead, on the assumption that a patient could
    legitimately lack the observation that times their death. Measured against
    the real HiRID export that assumption is false: all 2,062 recorded deaths
    are timeable from variables 110/200, so the shortfall is 0 and this raise
    costs a correct run nothing. If a future source does carry untimed deaths,
    the answer is to widen the timing variables for that source, not to let a
    quiet undercount through.
    """

    if not untimed:
        return
    total = timed + len(untimed)
    shown = sorted(untimed)[:10]
    raise ConceptExtractionUnavailable(
        concept_id=concept_id,
        database=database,
        stage='last_observation',
        detail=(
            f'{len(untimed)} of {total} recorded deaths in this cohort have no '
            f'observation of variable(s) {sorted(timing_ids)} to time them'
            + (' (none of them could be timed)' if not timed else '')
            + f'. Omitting them would report a mortality of {timed}/{total} of '
            f'the deaths the source records. Untimed patient ids: {shown}'
            + ('...' if len(untimed) > 10 else '')
        ),
    )


def _preserve_callback_dur_var_unit(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    """Carry an unchanged duration contract across a callback projection."""

    if "dur_var" not in before.columns or "dur_var" not in after.columns:
        return after
    from ..table.duration import get_dur_var_unit, set_dur_var_unit

    previous_unit = get_dur_var_unit(before)
    if previous_unit and not get_dur_var_unit(after):
        set_dur_var_unit(after, previous_unit)
    return after


def _load_mimic_icu_outtimes(
    data_source: Optional["ICUDataSource"],
    frame: pd.DataFrame,
    id_cols: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """Load the small ICU-discharge lookup required by duration callbacks."""

    if data_source is None:
        return None
    id_col = id_cols[0] if id_cols else None
    if not id_col or id_col not in frame.columns:
        raise ValueError("MIMIC duration callback requires an ICU stay identifier")
    try:
        table = data_source.load_table(
            "icustays",
            columns=[id_col, "intime", "outtime", "los"],
            verbose=False,
        )
        bounds = table.data if hasattr(table, "data") else table
    except Exception as exc:
        raise ValueError(
            "MIMIC duration callback could not load ICU outtime for clipping"
        ) from exc
    if not isinstance(bounds, pd.DataFrame):
        bounds = pd.DataFrame(bounds)
    if id_col not in bounds.columns or "outtime" not in bounds.columns:
        raise ValueError(
            "MIMIC icustays must expose the stay identifier and outtime"
        )
    stay_ids = frame[id_col].dropna().unique()
    bounds = bounds.loc[bounds[id_col].isin(stay_ids)].copy()
    # pandas 3 preserves source datetime units (for example datetime64[s]).
    # The LOS fallback may contain sub-second rounding, so normalize both
    # columns before assigning it rather than relying on an implicit lossy cast.
    bounds["outtime"] = pd.to_datetime(
        bounds["outtime"], errors="coerce"
    ).astype("datetime64[ns]")
    if "intime" in bounds.columns:
        bounds["intime"] = pd.to_datetime(
            bounds["intime"], errors="coerce"
        ).astype("datetime64[ns]")
    missing_outtime = bounds["outtime"].isna()
    if missing_outtime.any() and {"intime", "los"}.issubset(bounds.columns):
        fallback = pd.to_datetime(
            bounds.loc[missing_outtime, "intime"], errors="coerce"
        ) + pd.to_timedelta(
            pd.to_numeric(bounds.loc[missing_outtime, "los"], errors="coerce"),
            unit="D",
        )
        bounds.loc[missing_outtime, "outtime"] = fallback
    unresolved = set(stay_ids) - set(
        bounds.loc[bounds["outtime"].notna(), id_col].unique()
    )
    # Keep unresolved stays out of the lookup.  The clipping utility drops all
    # their duration episodes fail-closed and logs the affected count; it must
    # never infer discharge from a medication event.
    if unresolved:
        bounds = bounds.loc[~bounds[id_col].isin(unresolved)].copy()
    keep_columns = [id_col]
    if "intime" in bounds.columns:
        keep_columns.append("intime")
    keep_columns.append("outtime")
    return bounds.loc[
        bounds["outtime"].notna(), keep_columns
    ].drop_duplicates()


def _parse_eicu_age(values: pd.Series) -> pd.Series:
    """Parse eICU's numeric strings and ``> 89`` sentinel onto years."""

    text = values.astype("string").str.strip()
    age = pd.to_numeric(text.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    over_89 = text.str.startswith(">").fillna(False) & age.eq(89).fillna(False)
    return age.mask(over_89, 90.0)


def _load_eicu_tidal_volume_ages(
    frame: pd.DataFrame,
    *,
    id_column: str,
    data_source: Optional["ICUDataSource"],
) -> pd.Series:
    """Return age aligned to a respiratoryCharting source frame.

    The lookup is restricted to the current extraction batch.  Failure is
    represented by missing age rather than an assumed adult age; the caller
    then quarantines unit-ambiguous values instead of guessing their scale.
    """

    if "age" in frame.columns:
        return _parse_eicu_age(frame["age"])
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    if data_source is None or id_column not in frame.columns:
        return result

    stay_ids = frame[id_column].dropna().unique().tolist()
    if not stay_ids:
        return result
    try:
        from ..datasource import FilterOp, FilterSpec

        patient = data_source.load_table(
            "patient",
            columns=[id_column, "age"],
            filters=[FilterSpec(column=id_column, op=FilterOp.IN, value=stay_ids)],
            verbose=False,
        )
        demographics = patient.data if hasattr(patient, "data") else patient
        if not isinstance(demographics, pd.DataFrame):
            demographics = pd.DataFrame(demographics)
        if id_column not in demographics.columns or "age" not in demographics.columns:
            raise KeyError(f"patient table lacks {id_column!r} or 'age'")
        demographics = demographics[[id_column, "age"]].drop_duplicates(
            subset=[id_column], keep="first"
        )
        age_lookup = pd.Series(
            _parse_eicu_age(demographics["age"]).to_numpy(),
            index=demographics[id_column],
        )
        return pd.to_numeric(frame[id_column].map(age_lookup), errors="coerce")
    except Exception as exc:
        logger.warning(
            "eICU tidal-volume age lookup failed; ambiguous values will be "
            "quarantined (stays=%d, error=%s)",
            len(stay_ids),
            exc,
        )
        return result


def _normalize_eicu_tidal_volume_frame(
    frame: pd.DataFrame,
    *,
    concept_name: str,
    value_column: Optional[str] = None,
    id_column: Optional[str] = None,
    label_column: Optional[str] = None,
    ages: Optional[pd.Series] = None,
    force_liters: bool = False,
    force_milliliters: bool = False,
) -> pd.DataFrame:
    """Normalize eICU respiratory tidal volume to mL before aggregation.

    ``respiratoryCharting`` has no unit column.  Most tidal-volume labels are
    recorded in mL, but some interfaces emit L-scale decimals under the same
    label; ``Set Vt (Drager)`` is a separately identified L-scale source.  The
    mixed-label rule therefore uses within-stay evidence first, then adult age,
    and fails closed for ambiguous paediatric/unknown-age values.  The eICU
    ``lab.TV`` source declares mL semantics but contains a sparse implausible
    low-value tail; ``force_milliliters`` preserves credible values without
    interpreting those entries as litres.  Zero is a valid ventilator setting
    and is deliberately never rescaled.
    """

    out = frame.copy()
    value_column = value_column or (
        concept_name if concept_name in out.columns else None
    )
    if value_column is None or value_column not in out.columns:
        raise ValueError(
            f"eICU tidal-volume callback requires value column for {concept_name!r}"
        )
    id_column = id_column or next(
        (
            candidate
            for candidate in ("patientunitstayid", "stay_id")
            if candidate in out.columns
        ),
        None,
    )
    label_column = label_column or next(
        (
            candidate
            for candidate in ("respchartvaluelabel", "respChartValueLabel")
            if candidate in out.columns
        ),
        None,
    )

    raw = pd.to_numeric(out[value_column], errors="coerce")
    normalized = raw.copy()
    if ages is None:
        ages = pd.Series(np.nan, index=out.index, dtype="float64")
    else:
        ages = pd.to_numeric(ages.reindex(out.index), errors="coerce")
    adult = ages.ge(18)

    if label_column and label_column in out.columns:
        labels = out[label_column].astype("string").str.strip()
    else:
        labels = pd.Series(pd.NA, index=out.index, dtype="string")
    explicit_ml = labels.eq("Vt Spontaneous (mL)").fillna(False)
    if force_milliliters:
        explicit_ml = pd.Series(True, index=out.index)
    drager = labels.eq("Set Vt (Drager)").fillna(False)
    if force_liters:
        drager = pd.Series(True, index=out.index)

    positive = raw.gt(0)
    low = positive & raw.le(2)
    if id_column and id_column in out.columns:
        has_ml_reference = raw.ge(100).groupby(out[id_column], dropna=False).transform(
            "any"
        )
    else:
        has_ml_reference = pd.Series(False, index=out.index)

    # Drager is predominantly L-scale below 2, but a tiny tail is already mL
    # (250--650 in the real source).  Convert only the evidenced L range.
    drager_convert = drager & low
    contextual_convert = low & ~drager & ~explicit_ml & has_ml_reference
    adult_pure_convert = low & ~drager & ~explicit_ml & ~has_ml_reference & adult
    convert = drager_convert | contextual_convert | adult_pure_convert
    normalized.loc[convert] = raw.loc[convert] * 1000.0

    # A low value with neither an explicit unit nor contextual/adult evidence
    # cannot safely be interpreted as L or mL.  Preserve only explicit mL.
    ambiguous_low = low & ~drager & ~explicit_ml & ~contextual_convert & ~adult_pure_convert
    normalized.loc[ambiguous_low] = np.nan

    # Values between 2 and 50 are not credible adult tidal volumes.  For an
    # unlabelled paediatric/unknown-age record they are equally unit-ambiguous,
    # while an explicitly mL-labelled paediatric record remains admissible.
    raw_mid = raw.gt(2) & raw.lt(50)
    ambiguous_mid = raw_mid & ~explicit_ml
    normalized.loc[ambiguous_mid] = np.nan
    adult_small_ml = adult & normalized.gt(0) & normalized.lt(50)
    normalized.loc[adult_small_ml] = np.nan
    unknown_small_ml = ages.isna() & explicit_ml & normalized.gt(0) & normalized.lt(50)
    normalized.loc[unknown_small_ml] = np.nan

    out[value_column] = normalized
    audit = {
        "rows": int(len(out)),
        "zero_rows_preserved": int(raw.eq(0).sum()),
        "drager_l_to_ml_rows": int(drager_convert.sum()),
        "same_stay_l_to_ml_rows": int(contextual_convert.sum()),
        "adult_pure_low_l_to_ml_rows": int(adult_pure_convert.sum()),
        "ambiguous_low_quarantined_rows": int(ambiguous_low.sum()),
        "ambiguous_mid_quarantined_rows": int(ambiguous_mid.sum()),
        "adult_small_ml_quarantined_rows": int(adult_small_ml.sum()),
        "unknown_small_ml_quarantined_rows": int(unknown_small_ml.sum()),
        "age_missing_rows": int(ages.isna().sum()),
    }
    out.attrs["eicu_tidal_volume_unit_audit"] = audit
    logger.info(
        "eICU tidal-volume normalization concept=%s audit=%s",
        concept_name,
        audit,
    )
    return out


def _apply_callback(
    frame: pd.DataFrame,
    source: ConceptSource,
    concept_name: str,
    unit_column: Optional[str] = None,
    resolver: Optional['ConceptResolver'] = None,
    patient_ids: Optional[List] = None,
    data_source: Optional['ICUDataSource'] = None,
    interval: Optional[Union[str, pd.Timedelta]] = None,
) -> pd.DataFrame:
    callback = source.callback
    if not callback:
        return frame

    expr = callback.strip()

    if expr == "identity_callback":
        return frame

    if expr in {
        "eicu_tidal_volume_mixed_scale",
        "eicu_tidal_volume_drager_l_to_ml",
        "eicu_tidal_volume_explicit_ml",
    }:
        value_column = (
            concept_name
            if concept_name in frame.columns
            else source.value_var
            if source.value_var and source.value_var in frame.columns
            else None
        )
        id_column = next(
            (
                candidate
                for candidate in ("patientunitstayid", "stay_id")
                if candidate in frame.columns
            ),
            None,
        )
        if id_column is None:
            ages = pd.Series(np.nan, index=frame.index, dtype="float64")
        else:
            ages = _load_eicu_tidal_volume_ages(
                frame,
                id_column=id_column,
                data_source=data_source,
            )
        result = _normalize_eicu_tidal_volume_frame(
            frame,
            concept_name=concept_name,
            value_column=value_column,
            id_column=id_column,
            label_column=source.sub_var,
            ages=ages,
            force_liters=expr == "eicu_tidal_volume_drager_l_to_ml",
            force_milliliters=expr == "eicu_tidal_volume_explicit_ml",
        )
        # The deprecated ConceptLoader invokes callbacks before its standard
        # value/time projection.  Preserve that compatibility route without
        # changing the main resolver, which already presents concept_name.
        if concept_name not in frame.columns and value_column in result.columns:
            result = result.rename(columns={value_column: "value"})
            if "respchartoffset" in result.columns and "time" not in result.columns:
                result = result.rename(columns={"respchartoffset": "time"})
        return result

    if expr in ("vent_mode_control", "vent_mode_seq"):
        # Harmonise a native ventilator-mode label/code onto one axis (control | seq)
        # via the per-DB map in data/vent_mode_map.json. See apply_vent_mode_frame.
        from .callbacks import apply_vent_mode_frame
        axis = "control" if expr == "vent_mode_control" else "seq"
        out_column = "vent_mode" if expr == "vent_mode_control" else "vent_breath_seq"
        val_col = concept_name if concept_name in frame.columns else (source.value_var or "value")
        db_name = None
        try:
            db_name = data_source.config.name
        except Exception:
            pass
        result = apply_vent_mode_frame(frame, val_col, db_name, axis, out_column)
        # rename the harmonised column to concept_name if the loader expects it there
        if out_column in result.columns and concept_name != out_column:
            result = result.rename(columns={out_column: concept_name})
        return result

    if expr == "aumc_death":
        # In-hospital mortality, matching ricu's aumc_death:
        #   x[, val_var := is_true(dateofdeath - dischargedat < hours(72L))]
        # i.e. died within 72h of ICU discharge. AmsterdamUMCdb has no hospital discharge
        # date, so ricu uses this 72h-of-ICU-discharge window as its in-hospital proxy
        # (it captures in-ICU deaths plus early post-ICU-discharge deaths). We keep this
        # ricu-faithful; the proxy nature is documented in Table 1's footnote.
        #   index_var = dateofdeath, value_var = dischargedat (renamed to concept_name by
        #   the loader). Both are in MINUTES as delivered by the loader.
        #
        # BUG HISTORY (fixed 2026-07-16): the threshold was hard-coded as 72h in
        # MILLISECONDS (259,200,000) while the loader delivers minutes, so the window was a
        # no-op and every patient with a registry `dateofdeath` was flagged. The declared
        # source contract is now applied deterministically in minutes. Endpoint semantics
        # must never switch based on the mortality prevalence of the requested cohort.
        df = frame.copy()
        dod_col = (source.index_var if (source.index_var and source.index_var in df.columns)
                   else ('dateofdeath' if 'dateofdeath' in df.columns else None))
        dis_col = (concept_name if concept_name in df.columns
                   else ('dischargedat' if 'dischargedat' in df.columns else None))
        if dod_col is None or dis_col is None:
            raise ValueError(
                "aumc_death requires dateofdeath and dischargedat in loader-minute units"
            )
        dod = pd.to_numeric(df[dod_col], errors='coerce')
        dis = pd.to_numeric(df[dis_col], errors='coerce')
        died = (dod.notna() & ((dod - dis) < 72 * 60)).fillna(False)
        death_values = pd.Series(index=df.index, dtype=object)
        death_values[died] = True  # survivors -> NA (ricu convention)
        df[concept_name] = death_values
        return df

    # 🔧 SICdb death callback — in-hospital mortality via HospitalDischargeType.
    if expr == "sic_death":
        df = frame.copy()
        # In-hospital mortality: HospitalDischargeType == 2028 (Deceased).
        #
        # PREVIOUS BUG (fixed 2026-07-16): death was flagged whenever OffsetOfDeath was
        # non-null. OffsetOfDeath carries registry deaths up to ~1yr of follow-up (median
        # 43 days post-admission), so this reported ~annual mortality (18.6%, == mort_365d)
        # rather than in-hospital mortality (7.8%). ricu does not define SICdb `death`;
        # HospitalDischargeType is the source's hospital discharge disposition.
        #
        # OffsetOfDeath (val_var==index_var) is renamed to concept_name before this
        # callback; reuse it as the death charttime (seconds -> hours) where present.
        offset_secs = None
        for c in ['OffsetOfDeath', 'offsetofdeath', concept_name]:
            if c in df.columns:
                offset_secs = pd.to_numeric(df[c], errors='coerce')
                break
        disp_col = None
        for c in ['HospitalDischargeType', 'hospitaldischargetype']:
            if c in df.columns:
                disp_col = c
                break
        if disp_col is None:
            raise ValueError(
                "sic_death requires HospitalDischargeType; OffsetOfDeath alone "
                "cannot identify in-hospital mortality"
            )
        disp = pd.to_numeric(df[disp_col], errors='coerce')
        died = (disp == 2028)  # 2028 = Deceased
        death_values = pd.Series(index=df.index, dtype=object)
        death_values[died] = True  # survivors/unknown -> NA (ricu convention)
        df[concept_name] = death_values
        if offset_secs is not None:
            df['charttime'] = (offset_secs / 3600.0).where(died)
        return df

    # 🔧 HiRID death callback — matches R ricu hirid_death (callback-itm.R:197)
    # R ricu flow:
    #   1. Load observations for variableid IN [110, 200]
    #   2. dt_gforce(x, "last", by=idc, vars=idx) → last observation time per patient
    #   3. load_id(env[["general"]], cols="discharge_status") → load general table
    #   4. merge with dead patients → keep only patients who died
    #   5. Set val_var = TRUE
    # 🚀 优化: variableids 110, 200 有 115M 行（高频数据），
    #    直接在 DuckDB 中 GROUP BY patientid → MAX(datetime) 避免加载全量。
    if expr == "hirid_death":
        id_col = 'patientid'
        
        # Step 1: Load general table and find dead patients
        #
        # Every failure below used to be swallowed into ``dead_pids = set()``,
        # which returns an empty frame — indistinguishable from "nobody in this
        # cohort died". A missing file, a permission change, an upstream rename
        # of ``discharge_status`` or a DuckDB error therefore reported HiRID
        # mortality as zero, and the analysis downstream ran normally on it.
        # An outcome concept must not have a silent zero as its failure mode.
        if data_source is None:
            raise ConceptExtractionUnavailable(
                concept_id=concept_name,
                database='hirid',
                stage='load_general',
                detail=(
                    'no data source was supplied, so discharge status could '
                    'not be read'
                ),
            )
        try:
            general_tbl = data_source.load_table('general', columns=[id_col, 'discharge_status'])
            general_df = general_tbl.data if hasattr(general_tbl, 'data') else general_tbl
            if not isinstance(general_df, pd.DataFrame):
                general_df = pd.DataFrame(general_df)
            if 'discharge_status' not in general_df.columns:
                raise KeyError("general table has no 'discharge_status' column")
            dead_pids = set(general_df.loc[
                general_df['discharge_status'].astype(str).str.lower() == 'dead',
                id_col
            ].unique())
        except Exception as exc:
            raise ConceptExtractionUnavailable(
                concept_id=concept_name,
                database='hirid',
                stage='load_general',
                detail=f'the general table could not be read ({exc})',
                cause=exc,
            ) from exc

        # Narrow to the cohort before deciding anything. `dead_pids` is every
        # death in the source; the caller asked about `patient_ids`. Guarding
        # on the wider set made a cohort of survivors raise, because the
        # database recorded a death somewhere else.
        cohort_dead = deaths_within_cohort(dead_pids, patient_ids)

        # Reached only after a successful read: the source was legible and
        # nobody in this cohort died. That is a real answer, so return the
        # real empty.
        if not cohort_dead:
            return frame.head(0) if hasattr(frame, 'head') else pd.DataFrame()
        
        # Step 2: 🚀 使用 DuckDB 直接聚合获取最后观测时间（避免加载 115M 行）
        last_obs = None
        aggregation_error: Optional[BaseException] = None
        if data_source is not None and hasattr(data_source, 'base_path'):
            read_exprs = hirid_observation_read_exprs(data_source.base_path)
            if read_exprs is not None:
                try:
                    import duckdb
                    conn = duckdb.connect()
                    conn.execute("SET memory_limit = '2GB'")
                    # 🚀 perf B1 + B2 (hirid_death secondary callback path):
                    # same union_by_name + inline IN list issue as the primary
                    # fast path in load_concepts. Drop union_by_name=true
                    # (same-dir bucket parquets share schema) and register
                    # dead_pids as a DuckDB view rather than a giant inline
                    # IN list. Try/except falls back to safe union_by_name.
                    (
                        _ldd_read_expr_fast,
                        _ldd_read_expr_safe,
                        _ldd_layout,
                    ) = read_exprs
                    conn.register(
                        "_ldd_dead_pids",
                        pd.DataFrame({"patientid": list(cohort_dead)}),
                    )
                    _ldd_q_tpl = """
                        SELECT obs.patientid, MAX(obs.datetime) AS datetime
                        FROM {read_expr} AS obs
                        WHERE obs.variableid IN (110, 200)
                          AND obs.patientid IN (SELECT patientid FROM _ldd_dead_pids)
                        GROUP BY obs.patientid
                    """
                    try:
                        last_obs = conn.execute(
                            _ldd_q_tpl.format(read_expr=_ldd_read_expr_fast)
                        ).fetchdf()
                    except Exception:
                        last_obs = conn.execute(
                            _ldd_q_tpl.format(read_expr=_ldd_read_expr_safe)
                        ).fetchdf()
                    finally:
                        try:
                            conn.unregister("_ldd_dead_pids")
                        except Exception:
                            pass
                    conn.close()
                except Exception as exc:
                    # Kept as a fallback, not as a result: the in-memory frame
                    # below may still be able to answer. Remembered so that if
                    # it cannot, the raise reports the real cause instead of
                    # an empty table.
                    aggregation_error = exc

        # Fallback: 使用已加载的 frame（旧行为）
        if last_obs is None or last_obs.empty:
            df = frame.copy() if hasattr(frame, 'copy') else pd.DataFrame()
            if id_col in df.columns:
                time_col = 'datetime' if 'datetime' in df.columns else ('charttime' if 'charttime' in df.columns else None)
                if time_col:
                    last_obs = df.groupby(id_col, as_index=False).agg({time_col: 'max'})
                    last_obs = last_obs[last_obs[id_col].isin(cohort_dead)]

        if last_obs is None or last_obs.empty:
            # ``cohort_dead`` is non-empty here — the general table says these
            # patients died. Returning an empty frame would report zero deaths
            # while the source we just read says otherwise, so the emptiness is
            # a failure to time the deaths, not an absence of them.
            raise ConceptExtractionUnavailable(
                concept_id=concept_name,
                database='hirid',
                stage='last_observation',
                detail=(
                    f'{len(cohort_dead)} patient(s) in this cohort are recorded '
                    'as deceased in the general table, but no last observation '
                    'time could be obtained for any of them'
                    + (
                        f' (aggregation failed: {aggregation_error})'
                        if aggregation_error is not None
                        else ''
                    )
                ),
                cause=aggregation_error,
            )

        # Some deaths timed, some not: the result under-reports mortality by
        # exactly the shortfall, and nothing downstream can see it happened.
        _refuse_untimed_deaths(
            database='hirid',
            concept_id=concept_name,
            timing_ids=(110, 200),
            timed=len(last_obs),
            untimed=cohort_dead - set(last_obs[id_col]),
        )

        # Step 3: Set death = TRUE
        result = last_obs.copy()
        result[concept_name] = True
        
        return result

    # Handle eicu_age - process eICU age data (convert '> 89' to 90)
    if re.fullmatch(r"transform_fun\(eicu_age\)", expr):
        from ..utils.callback_utils import eicu_age
        return eicu_age(frame, val_col=concept_name)

    # Handle eicu_adx - process eICU admission diagnosis to categorize as med/surg/other
    if expr == "eicu_adx":
        """
        Map eICU admitdxpath to admission type (med/surg/other).
        
        The admitdxpath contains hierarchical diagnosis path like:
        "admission diagnosis|All Diagnosis|Operative|Diagnosis|Cardiovascular|..."
        "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Genitourinary|..."
        
        Rules from R ricu (callback-itm.R eicu_adx):
        1. Split path by "|"
        2. Keep only rows where parts[1] == "All Diagnosis"
        3. If parts[4] in ["Genitourinary", "Transplant"] -> "other"
        4. Else if parts[2] == "Operative" -> "surg"
        5. Else -> "med"
        """
        frame = frame.copy()
        
        # Get the diagnosis path column
        # 🔧 FIX: 回调在重命名后调用，所以 value_var (admitdxpath) 已变为 concept_name (adm)
        # 优先使用 concept_name，然后再尝试 source.value_var
        val_col = None
        # 1. 优先使用 concept_name（重命名后的列名）
        if concept_name in frame.columns:
            val_col = concept_name
        # 2. 如果 concept_name 不存在，尝试 source.value_var
        elif source.value_var and source.value_var in frame.columns:
            val_col = source.value_var
        # 3. 最后尝试常见列名
        else:
            for col in ['admitdxpath', 'diagnosispath', 'diagnosis']:
                if col in frame.columns:
                    val_col = col
                    break
        
        if val_col is None:
            # No diagnosis column found, return empty
            frame[concept_name] = pd.Series(dtype='object')
            return frame
        
        def classify_adm_type(path):
            if pd.isna(path):
                return None  # Will be filtered out
            
            parts = str(path).split('|')
            
            # Require at least 3 segments (0, 1, 2) and check parts[1] == "All Diagnosis"
            if len(parts) < 3:
                return None
            
            if parts[1].strip() != "All Diagnosis":
                return None
            
            # Check parts[4] for Genitourinary or Transplant (if exists)
            if len(parts) > 4:
                seg4 = parts[4].strip()
                if seg4 in ["Genitourinary", "Transplant"]:
                    return 'other'
            
            # Check parts[2] for Operative
            seg2 = parts[2].strip()
            if seg2 == "Operative":
                return 'surg'
            
            # Default to med (Non-operative)
            return 'med'
        
        frame[concept_name] = frame[val_col].apply(classify_adm_type)
        
        # Filter out None values (rows that didn't match "All Diagnosis" criteria)
        frame = frame[frame[concept_name].notna()].copy()
        
        # Drop the original diagnosis path column if it's different from concept_name
        if val_col != concept_name and val_col in frame.columns:
            frame = frame.drop(columns=[val_col])
        
        return frame

    # Handle percent_as_numeric - remove '%' and convert to numeric
    if re.fullmatch(r"transform_fun\(percent_as_numeric\)", expr):
        series = frame[concept_name]

        def _scale_fractional_percent(values: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(values, errors='coerce')
            fraction_mask = numeric.gt(0) & numeric.le(1)
            if fraction_mask.any():
                numeric = numeric.copy()
                numeric.loc[fraction_mask] = numeric.loc[fraction_mask] * 100.0
            return numeric

        # 🚀 Fast path: if already numeric (DuckDB pre-processed), skip all string ops
        if pd.api.types.is_numeric_dtype(series):
            na_mask = series.isna()
            if na_mask.any():
                for fallback_col in ("value", "valuetext"):
                    if fallback_col in frame.columns and fallback_col != concept_name:
                        fallback = pd.to_numeric(frame[fallback_col], errors='coerce')
                        series = series.where(~na_mask, fallback)
                        na_mask = series.isna()
                        if not na_mask.any():
                            break
            frame.loc[:, concept_name] = _scale_fractional_percent(series)
            return frame

        # 🚀 Optimized slow path: try to_numeric first, only strip '%' on failures
        # Most string values are plain numbers ('50', '0.21') that pd.to_numeric handles directly.
        # Only actual percent strings ('50%') need the rstrip. This reduces _str_map calls from
        # 2 (strip+rstrip on ALL 12.9M rows) to at most 1 (rstrip on the small failed subset).
        na_mask = series.isna()
        if na_mask.any():
            for fallback_col in ("value", "valuetext"):
                if fallback_col in frame.columns and fallback_col != concept_name:
                    series = series.where(~na_mask, frame[fallback_col])
                    na_mask = series.isna()
                    if not na_mask.any():
                        break

        result = pd.to_numeric(series, errors='coerce')
        failed_mask = result.isna() & series.notna()
        if failed_mask.any():
            fixed = pd.to_numeric(
                series[failed_mask].astype(str).str.rstrip('%'), errors='coerce'
            )
            result = result.copy()
            result.loc[failed_mask] = fixed
        result = _scale_fractional_percent(result)
        # Cast to object first so pandas 2.x string-backed columns accept float values.
        if hasattr(frame[concept_name], 'dtype') and str(frame[concept_name].dtype) in ('string', 'str'):
            frame = frame.copy()
            frame[concept_name] = frame[concept_name].astype(object)
        frame.loc[:, concept_name] = result
        return frame

    match = re.fullmatch(r"transform_fun\(set_val\((.+)\)\)", expr, flags=re.DOTALL)
    if match:
        value = _parse_literal(match.group(1))
        frame = frame.copy()
        if concept_name in frame.columns:
            frame.drop(columns=[concept_name], inplace=True)
        dtype = "boolean" if isinstance(value, bool) else None
        result_series = pd.Series([value] * len(frame), index=frame.index, dtype=dtype)
        frame[concept_name] = result_series
        return frame

    # Handle comp_na() without arguments - check if value is not NA
    if re.fullmatch(r"transform_fun\(comp_na\(\)\)", expr):
        # 🔧 FIX: 确定要检查的列 - 优先使用 source.value_var，否则用 concept_name
        # MIMIC-III 的列名是大写的，需要智能匹配
        val_col = source.value_var if source.value_var else concept_name
        if val_col not in frame.columns:
            # 尝试大写/小写匹配
            col_map = {c.lower(): c for c in frame.columns}
            if val_col.lower() in col_map:
                val_col = col_map[val_col.lower()]
            elif concept_name.lower() in col_map:
                val_col = col_map[concept_name.lower()]
            else:
                # 最后尝试用原始列名
                for col in frame.columns:
                    if 'itemid' in col.lower() or 'org' in col.lower():
                        val_col = col
                        break
        series = frame[val_col]
        # Convert to boolean: True if not NA, False if NA
        frame.loc[:, concept_name] = series.notna().astype(float)
        return frame

    match = re.fullmatch(r"transform_fun\(comp_na\(`(.+?)`,\s*(.+)\)\)", expr, flags=re.DOTALL)
    if match:
        op_token = match.group(1)
        value = _parse_literal(match.group(2))
        op_map = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        if op_token not in op_map:
            raise NotImplementedError(
                f"Unsupported comparison operator '{op_token}' in callback '{expr}'."
            )
        # 🔧 FIX: 确定要比较的列 - 优先使用 concept_name，如果不存在则使用 source.value_var
        # 这修复了 ett_gcs 等概念在 callback 执行前 value_column 尚未重命名的情况
        compare_col = concept_name
        if compare_col not in frame.columns:
            # 尝试使用 source.value_var
            if source.value_var and source.value_var in frame.columns:
                compare_col = source.value_var
            # 或者尝试常见的值列名
            elif 'value' in frame.columns:
                compare_col = 'value'
            elif 'valuenum' in frame.columns:
                compare_col = 'valuenum'
        
        if compare_col not in frame.columns:
            # 如果仍然找不到列，返回原始 frame（不做任何处理）
            return frame
        
        series = frame[compare_col]
        if isinstance(value, (int, float)) and not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(series, errors="coerce")
        comparator = op_map[op_token]
        # 🚀 Vectorized comparison instead of .apply(lambda) — avoids N python calls
        na_mask = series.isna()
        comparison = comparator(series, value)
        comparison = comparison.where(~na_mask, False).astype("boolean")
        frame = frame.copy()
        # 如果比较的是 concept_name 列，删除它；否则保留原列
        if compare_col == concept_name:
            frame.drop(columns=[concept_name], inplace=True)
        frame[concept_name] = comparison
        return frame

    match = re.fullmatch(r"transform_fun\(binary_op\(`(.+?)`,\s*(.+)\)\)", expr, flags=re.DOTALL)
    if match:
        symbol = match.group(1)
        value = _parse_literal(match.group(2))
        frame = frame.copy()
        series = pd.to_numeric(frame[concept_name], errors="coerce")
        result = _apply_binary_op(symbol, series, value)
        frame.loc[:, concept_name] = result
        return frame

    # Handle transform_fun(floor) - apply floor function to values
    if re.fullmatch(r"transform_fun\(floor\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').apply(np.floor)
        return frame

    # Handle transform_fun(ceiling) or transform_fun(ceil) - apply ceiling function
    if re.fullmatch(r"transform_fun\(ceil(ing)?\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').apply(np.ceil)
        return frame

    # Handle transform_fun(round) - apply round function
    if re.fullmatch(r"transform_fun\(round\)", expr):
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce').round()
        return frame

    # Handle transform_fun(na_below(<threshold>)) - map sentinel values below a
    # threshold to NaN (e.g. eICU/SICdb store -1 for "score not computed"). Kept
    # generic so any native-score concept can declare its own sentinel floor.
    match = re.fullmatch(r"transform_fun\(na_below\(\s*(-?\d+(?:\.\d+)?)\s*\)\)", expr)
    if match:
        threshold = float(match.group(1))
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        if val_col in frame.columns:
            series = pd.to_numeric(frame[val_col], errors='coerce')
            frame[val_col] = series.where(series >= threshold)
        return frame

    # Handle aggregate_fun('sum', 'units') - aggregate by sum and set unit
    match = re.fullmatch(r"aggregate_fun\(['\"](\w+)['\"],\s*['\"](.+?)['\"]\)", expr)
    if match:
        agg_func = match.group(1)  # e.g., 'sum'
        new_unit = match.group(2)  # e.g., 'units'
        
        frame = frame.copy()
        val_col = concept_name if concept_name in frame.columns else (source.value_var or 'value')
        unit_col = source.unit_var
        
        # Identify ID and time columns
        id_col = None
        for cand in ['patientid', 'stay_id', 'admissionid', 'patientunitstayid', 'subject_id']:
            if cand in frame.columns:
                id_col = cand
                break
        
        time_col = None
        for cand in ['datetime', 'charttime', 'time', 'givenat']:
            if cand in frame.columns:
                time_col = cand
                break
        
        if id_col and time_col and val_col in frame.columns:
            # Convert to numeric
            frame[val_col] = pd.to_numeric(frame[val_col], errors='coerce')
            
            # Group by id and time, apply aggregation
            group_cols = [id_col, time_col]
            if agg_func == 'sum':
                result = frame.groupby(group_cols, as_index=False)[val_col].sum()
            elif agg_func == 'mean':
                result = frame.groupby(group_cols, as_index=False)[val_col].mean()
            elif agg_func == 'max':
                result = frame.groupby(group_cols, as_index=False)[val_col].max()
            elif agg_func == 'min':
                result = frame.groupby(group_cols, as_index=False)[val_col].min()
            else:
                result = frame  # Unknown aggregation, return as-is
            
            # Set unit
            if unit_col:
                result[unit_col] = new_unit
            
            return result
        
        return frame

    # 匹配 mimic_sampling (R ricu callback-itm.R)
    # mimic_sampling(x, val_var, aux_time, ...)
    # 功能：1) combine_date_time(x, aux_time, hours(12L))
    #      2) 将每个 microbiology row 标记为一次采样事件。
    #
    # ``org_itemid`` 是否缺失描述培养结果是否检出微生物，而不是标本是否
    # 已采集。把 ``!is.na(org_itemid)`` 暴露为 ``samp`` 会使 MIMIC 的阴性
    # 培养变成 False，但 eICU/AUMC 的同一概念仍为 True，破坏跨库语义。
    # 阳性结果由独立的 culture_positive 输出承担。
    if expr == "mimic_sampling":
        frame = frame.copy()
        val_var = source.value_var or concept_name
        aux_time = source.params.get("aux_time") if source.params else None
        
        # 1. combine_date_time: 如果aux_time是NA，使用index_column + 12小时
        if aux_time and aux_time in frame.columns:
            # 找到实际的index列（通常是charttime, starttime等）
            # 检查是否有明确的index_var
            index_col = source.index_var
            if not index_col:
                # 尝试从表配置中获取
                time_cols = [col for col in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[col])]
                if time_cols:
                    # 优先使用非aux_time的datetime列
                    index_col = next((col for col in time_cols if col != aux_time), time_cols[0])
            
            if index_col and index_col in frame.columns:
                # 如果aux_time是NA，使用index_col + 12小时
                mask = frame[aux_time].isna()
                if mask.any():
                    frame.loc[mask, aux_time] = pd.to_datetime(frame.loc[mask, index_col], errors='coerce') + pd.Timedelta(hours=12)
                # 更新index_column为aux_time（使用aux_time作为时间索引）
                if index_col != aux_time:
                    # 将aux_time的值复制到index_col，然后删除aux_time
                    frame[index_col] = pd.to_datetime(frame[aux_time], errors='coerce')
                    frame = frame.drop(columns=[aux_time])
        
        # 2. microbiologyevents 中每一行都来自一个已采集标本。即使
        # org_itemid 为空（阴性培养），也必须是 samp=True。
        if val_var in frame.columns:
            frame[concept_name] = True
            if val_var != concept_name:
                frame = frame.drop(columns=[val_var])
        else:
            frame[concept_name] = True
        
        return frame
    
    # 匹配 apply_map(c(...), var = 'sub_var') 或 apply_map(c(...))
    match = re.fullmatch(r"apply_map\(\s*c\((.+?)\)\s*(?:,\s*var\s*=\s*['\"](.+?)['\"])?\s*\)", expr, flags=re.DOTALL)
    if match:
        mapping = _parse_mapping(match.group(1))
        var_param = match.group(2) if match.group(2) else None
        
        frame = frame.copy()
        
        # 解析 var_param，如果是 'sub_var'，使用 source.sub_var 的实际值
        target_col = None
        if var_param:
            if var_param == 'sub_var' and source.sub_var:
                # var='sub_var' 表示映射 sub_var 列（如 itemid）
                target_col = source.sub_var
            elif var_param == 'val_col' and concept_name in frame.columns:
                # var='val_col' 表示映射值列（concept_name）
                target_col = concept_name
            elif var_param in frame.columns:
                # 直接使用 var_param 作为列名
                target_col = var_param
        
        # 如果指定了目标列且存在，映射该列；否则映射concept_name列
        if target_col and target_col in frame.columns:
            # 映射指定的列
            series = frame[target_col]
            def mapper(val):
                if pd.isna(val):
                    return val
                # 尝试直接匹配，然后尝试字符串匹配
                result = mapping.get(val, mapping.get(str(val), val))
                return result
            
            # 显式转换为 object 类型以避免 FutureWarning
            # 当映射值的类型与原列类型不兼容时（如字符串映射到 int32），需要先转换类型
            mapped_series = series.map(mapper)
            if frame[target_col].dtype != mapped_series.dtype:
                frame[target_col] = frame[target_col].astype(object)
            frame.loc[:, target_col] = mapped_series
        elif concept_name in frame.columns:
            # 默认映射concept_name列
            series = frame[concept_name]
            def mapper(val):
                if pd.isna(val):
                    return val
                return mapping.get(val, mapping.get(str(val), val))
            
            # 同样处理类型不兼容问题
            mapped_series = series.map(mapper)
            if frame[concept_name].dtype != mapped_series.dtype:
                frame[concept_name] = frame[concept_name].astype(object)
            frame.loc[:, concept_name] = mapped_series
        
        return frame

    match = re.fullmatch(r"convert_unit\((.+)\)", expr, flags=re.DOTALL)
    if match:
        arguments = _split_arguments(match.group(1))
        if not arguments:
            raise NotImplementedError(f"Callback '{callback}' is empty.")

        symbol, value = _parse_binary_op(arguments[0])
        new_unit = _strip_quotes(arguments[1]) if len(arguments) > 1 else None
        old_unit = _strip_quotes(arguments[2]) if len(arguments) > 2 else None

        frame = frame.copy()
        
        # 如果 source.unit_var 未指定，尝试自动检测单位列
        actual_unit_var = source.unit_var or unit_column
        
        # 如果仍然没有，尝试常见的单位列名
        if not actual_unit_var and 'valueuom' in frame.columns:
            actual_unit_var = 'valueuom'
        elif not actual_unit_var and 'unit' in frame.columns:
            actual_unit_var = 'unit'
        
        if actual_unit_var and actual_unit_var in frame.columns:
            unit_series = frame[actual_unit_var].fillna('').astype(str)
            if old_unit:
                case_flag = False
                try:
                    mask = unit_series.str.contains(old_unit, case=case_flag, na=False, regex=True)
                except re.error:
                    mask = unit_series.str.contains(re.escape(old_unit), case=case_flag, na=False, regex=True)
                # ⚠️ 不匹配空单位行: MIMIC-IV中单位为空时值已经正确
            else:
                # 如果old_unit为None，转换所有行（R ricu行为）
                mask = pd.Series(True, index=frame.index)
        else:
            mask = pd.Series(True, index=frame.index)

        numeric = pd.to_numeric(frame.loc[mask, concept_name], errors="coerce")
        transformed = _apply_binary_op(symbol, numeric, value)
        
        # 明确转换类型以避免 dtype 不兼容警告
        frame.loc[mask, concept_name] = transformed.astype('float64')

        # 更新单位列
        if new_unit and actual_unit_var and actual_unit_var in frame.columns:
            frame.loc[mask, actual_unit_var] = new_unit

        return frame

    match = re.fullmatch(r"combine_callbacks\((.+)\)", expr, flags=re.DOTALL)
    if match:
        frame_result = frame
        for arg in _split_arguments(match.group(1)):
            nested = arg.strip()
            if not nested:
                continue
            nested_source = replace(source, callback=nested)
            previous_frame = frame_result
            frame_result = _apply_callback(
                frame_result, nested_source, concept_name, unit_column,
                resolver=resolver, patient_ids=patient_ids, data_source=data_source,
                interval=interval,
            )
            frame_result = _preserve_callback_dur_var_unit(
                previous_frame,
                frame_result,
            )
        return frame_result
    
    # Handle dex_to_10 callback (convert different dextrose concentrations to D10 equivalent)
    # Format: dex_to_10(ids, factors) or dex_to_10(c(...), c(...)) or dex_to_10(list(...), c(...))
    match = re.fullmatch(r"dex_to_10\((.+)\)", expr, flags=re.DOTALL)
    if match:
        if frame.empty:
            return frame
        args = _split_arguments(match.group(1))
        if len(args) < 2:
            raise ValueError("dex_to_10 requires item IDs and conversion factors")

        try:
            itemids = _parse_r_value(args[0].strip())
            factors = _parse_r_value(args[1].strip())
            if not isinstance(itemids, (list, tuple)):
                itemids = [itemids]
            if not isinstance(factors, (list, tuple)):
                factors = [factors]

            sub_var = source.sub_var
            if not sub_var or sub_var not in frame.columns:
                raise ValueError(
                    "dex_to_10 requires the configured sub_var to survive "
                    "upstream aggregation"
                )

            val_col = None
            candidates = [
                source.value_var,
                concept_name,
                "dose",
                "rate",
                "amount",
                "givendose",
                "pharmavalue",
                "valuenum",
            ]
            for candidate in candidates:
                if (
                    candidate
                    and candidate in frame.columns
                    and frame[candidate].notna().any()
                ):
                    val_col = candidate
                    break
            if val_col is None:
                raise ValueError("dex_to_10 could not identify a value column")

            from ..utils.callback_utils import dex_to_10 as dex_to_10_fn

            return dex_to_10_fn(itemids, factors)(
                frame,
                sub_var=sub_var,
                val_col=val_col,
            )
        except Exception as exc:
            raise ValueError(f"dex_to_10 conversion failed: {exc}") from exc
    
    # Handle grp_mount_to_rate callback (convert grouped amounts to rates)
    # Format: grp_mount_to_rate(mins(1L), hours(1L)) or similar
    match = re.fullmatch(r"grp_mount_to_rate\((.+)\)", expr, flags=re.DOTALL)
    if match:
        args = _split_arguments(match.group(1))
        if len(args) < 2:
            raise ValueError(
                "grp_mount_to_rate requires minimum and extra durations"
            )
        try:
            min_dur_expr = args[0].strip()
            extra_dur_expr = args[1].strip()

            def _parse_duration_expr(dur_expr: str) -> pd.Timedelta:
                """Parse R duration expressions such as ``mins(1L)``."""
                duration_match = re.fullmatch(
                    r"(mins|hours|secs)\(\s*(\d+)L?\s*\)",
                    dur_expr,
                )
                if duration_match is None:
                    raise ValueError(
                        f"unsupported duration expression {dur_expr!r}"
                    )
                amount = int(duration_match.group(2))
                unit = {
                    "mins": "minutes",
                    "hours": "hours",
                    "secs": "seconds",
                }[duration_match.group(1)]
                return pd.Timedelta(**{unit: amount})

            min_dur = _parse_duration_expr(min_dur_expr)
            extra_dur = _parse_duration_expr(extra_dur_expr)

            grp_var = None
            if getattr(source, "grp_var", None):
                grp_var = source.grp_var
            elif source.params and "grp_var" in source.params:
                grp_var = source.params["grp_var"]

            val_col = source.value_var
            if not val_col:
                for candidate in [
                    "val",
                    "value",
                    "amount",
                    "dose",
                    "givendose",
                    "pharmavalue",
                ]:
                    if candidate in frame.columns:
                        val_col = candidate
                        break
            if not val_col:
                val_col = concept_name

            unit_col = source.unit_var
            if not unit_col:
                for candidate in [
                    "unit",
                    "unit_var",
                    "amountuom",
                    "doserateunit",
                    "doseunit",
                ]:
                    if candidate in frame.columns:
                        unit_col = candidate
                        break

            index_var = source.index_var
            if not index_var:
                for candidate in [
                    "datetime",
                    "givenat",
                    "starttime",
                    "charttime",
                    "time",
                ]:
                    if candidate in frame.columns:
                        index_var = candidate
                        break

            standard_id_cols = [
                "patientid",
                "stay_id",
                "admissionid",
                "patientunitstayid",
                "subject_id",
                "hadm_id",
                "icustay_id",
            ]
            id_cols = [col for col in standard_id_cols if col in frame.columns]

            from ..utils.callback_utils import grp_mount_to_rate as grp_mount_fn

            callback_fn = grp_mount_fn(
                min_dur=min_dur,
                extra_dur=extra_dur,
                grp_var=grp_var,
            )
            resolved_val_col = (
                val_col
                if val_col in frame.columns
                else concept_name
                if concept_name in frame.columns
                else "value"
            )
            return callback_fn(
                frame,
                val_col=resolved_val_col,
                unit_col=(
                    unit_col
                    if unit_col and unit_col in frame.columns
                    else "unit"
                ),
                index_var=index_var,
                id_cols=id_cols,
                sub_var=source.sub_var,
            )
        except Exception as exc:
            raise ValueError(
                f"grp_mount_to_rate conversion failed: {exc}"
            ) from exc
    
    # Handle ts_to_win_tbl callback
    match = re.fullmatch(r"ts_to_win_tbl\((.+)\)", expr, flags=re.DOTALL)
    if match:
        # Parse the duration expression (e.g., "mins(1L)")
        dur_expr = match.group(1).strip()
        # Simple parsing for common duration patterns
        if 'mins(' in dur_expr:
            mins_match = re.search(r'mins\((\d+)', dur_expr)
            if mins_match:
                duration = pd.Timedelta(minutes=int(mins_match.group(1)))
            else:
                duration = pd.Timedelta(minutes=1)  # default
        elif 'hours(' in dur_expr:
            hours_match = re.search(r'hours\((\d+)', dur_expr)
            if hours_match:
                duration = pd.Timedelta(hours=int(hours_match.group(1)))
            else:
                duration = pd.Timedelta(hours=1)  # default
        else:
            duration = pd.Timedelta(minutes=1)  # default fallback
        
        # Add duration column
        frame = frame.copy()
        
        # 🔧 FIX: 检测时间列的类型，确保dur_var与其兼容
        # 如果时间列是数值型（小时），则dur_var也应该是数值型（小时）
        # 🔧 FIX 2026-02-15: 添加 measuredat 支持 AUMC
        index_col = None
        for col in ['charttime', 'starttime', 'start', 'time', 'measuredat', 'measuredat_minutes', 'datetime']:
            if col in frame.columns:
                index_col = col
                break
        
        from ..table.duration import UNIT_MINUTES, set_dur_var_unit

        if index_col and index_col in frame.columns and pd.api.types.is_numeric_dtype(frame[index_col]):
            # 时间列是数值型（小时或分钟），dur_var用分钟数值
            # R ricu: ts_to_win_tbl(mins(1L)) → dur_var = difftime(1, units="mins")
            # 写入CSV时序列化为数值 1.0（分钟）
            frame['dur_var'] = duration.total_seconds() / 60.0  # 转换为分钟
            set_dur_var_unit(frame, UNIT_MINUTES)
        else:
            # 🔧 FIX: 始终使用数值分钟，而非 Timedelta 对象
            # 原因：后续 _align_to_admission_time 会将 datetime → 相对小时数，
            # 但 Timedelta dur_var 会被转为 int64 纳秒（而非分钟），
            # 导致 _expand_public_numeric_win_tbl_output 中 duration 被误解为
            # 60 000 000 000 小时 → 无限循环。
            # R ricu 的 dur_var 也是数值型（分钟），所以统一用分钟。
            frame['dur_var'] = duration.total_seconds() / 60.0  # 转换为分钟
            set_dur_var_unit(frame, UNIT_MINUTES)

        return frame
    
    # Handle mimic_rate_mv callback (for infusion rates)
    if expr.strip() == "mimic_rate_mv":
        from ..utils.callback_utils import mimic_rate_mv
        # Call the callback with appropriate parameters
        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        # stop_var is stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        # 🔧 FIX: mimic_rate_mv 应使用表的 'rate' 列，而不是 concept_name
        # R ricu 中 mimic_rate_mv 使用 inputevents 表的 'rate' 列作为输出值
        # 原始数据中 'rate' 是速率 (mcg/kg/min)，'amount' 是总量 (mg)
        val_col = 'rate' if 'rate' in frame.columns else concept_name
        
        # 🔧 CRITICAL FIX 2024-11-30: Get admission times for R ricu-compatible floor behavior
        # R ricu converts datetime to relative time BEFORE callbacks (in load_mihi).
        # This affects floor() behavior in expand().
        admission_times = None
        if data_source is not None:
            try:
                # Load icustays to get admission times
                icustays_result = data_source.load_table('icustays')
                # Handle ICUTable or DataFrame result
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                    
                if icustays is not None and len(icustays) > 0:
                    # Find ID column
                    id_col = None
                    for col in id_cols:
                        if col in icustays.columns:
                            id_col = col
                            break
                    if id_col is not None:
                        # Filter to patients in the current frame
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass  # Fail silently - will use fallback floor behavior
        
        result = mimic_rate_mv(
            frame,
            val_col=val_col,
            unit_col=unit_col,
            stop_var=stop_var,
            id_cols=id_cols,
            admission_times=admission_times,  # 🔧 Pass admission times for proper floor behavior
        )
        # 🔧 FIX: 将 'rate' 列重命名为 concept_name（如果不同）
        if val_col != concept_name and val_col in result.columns:
            result = result.rename(columns={val_col: concept_name})
        return result
    
    # Handle mimic_dur_inmv callback (for infusion durations)
    if expr.strip() == "mimic_dur_inmv":
        from ..utils.callback_utils import mimic_dur_inmv
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
        # stop_var and grp_var are stored in params dict
        stop_var = source.params.get('stop_var', None) if source.params else None
        grp_var = source.params.get('grp_var', None) if source.params else None
        # ``rateuom`` describes the input rate, not the derived elapsed time.
        # The concept dictionary owns the canonical output unit (hours), so do
        # not propagate a medication-rate unit onto a duration value.
        unit_col = None
        val_col = concept_name
        
        icu_stays = _load_mimic_icu_outtimes(data_source, frame, id_cols)
        status_var = source.params.get("status_var", "statusdescription")
        cancel_var = source.params.get("cancel_var")
        excluded_statuses = source.params.get("excluded_statuses")
        merge_gap_minutes = source.params.get("merge_gap_minutes", 5.0)

        return mimic_dur_inmv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            stop_var=stop_var,
            id_cols=id_cols,
            unit_col=unit_col,
            icu_stays=icu_stays,
            status_var=str(status_var),
            cancel_var=str(cancel_var) if cancel_var else None,
            excluded_statuses=excluded_statuses,
            merge_gap_minutes=merge_gap_minutes,
        )
    
    # Handle mimic_dur_incv callback (for CareVue durations)
    if expr.strip() == "mimic_dur_incv":
        from ..utils.callback_utils import mimic_dur_incv
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        # R ricu's calc_dur uses id_vars(x) which returns only the patient ID (e.g., icustay_id)
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
        # grp_var is stored in params dict
        grp_var = source.params.get('grp_var', None) if source.params else None
        # CareVue's source unit is likewise the rate unit, not a duration unit.
        unit_col = None
        val_col = concept_name
        icu_stays = _load_mimic_icu_outtimes(data_source, frame, id_cols)
        boundary_var = source.params.get("boundary_var", "stopped")
        merge_gap_hours = source.params.get("merge_gap_hours", 5.0)
        rate_var = source.params.get("rate_var", "rate")

        return mimic_dur_incv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            id_cols=id_cols,
            unit_col=unit_col,
            icu_stays=icu_stays,
            boundary_var=str(boundary_var),
            merge_gap_hours=merge_gap_hours,
            rate_var=str(rate_var),
        )
    
    # Handle mimic_rate_cv callback (for CareVue infusion rates)
    if expr.strip() == "mimic_rate_cv":
        from ..utils.callback_utils import mimic_rate_cv
        # 🔧 FIX 2025-02-10: Only use the PRIMARY patient ID column, not all "id" columns
        primary_id_candidates = ['icustay_id', 'stay_id', 'patientunitstayid', 'admissionid', 'patientid', 'CaseID']
        id_cols = None
        for cand in primary_id_candidates:
            if cand in frame.columns:
                id_cols = [cand]
                break
        # grp_var is stored in params dict
        grp_var = source.params.get('grp_var', None) if source.params else None
        unit_col = source.unit_var if hasattr(source, 'unit_var') else None
        val_col = concept_name
        
        # 🔧 FIX: Load admission_times for R ricu-compatible relative time flooring
        # R ricu converts datetime to relative difftime BEFORE callbacks (in load_difftime).
        # CareVue expand_intervals needs this to correctly floor to hour boundaries.
        admission_times = None
        if data_source is not None:
            try:
                icustays_result = data_source.load_table('icustays')
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                if icustays is not None and len(icustays) > 0:
                    id_col = id_cols[0] if id_cols else None
                    if id_col and id_col in icustays.columns:
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass
        
        return mimic_rate_cv(
            frame,
            val_col=val_col,
            grp_var=grp_var,
            unit_col=unit_col,
            id_cols=id_cols,
            admission_times=admission_times,
        )

    if expr.strip() == "vent_flag":
        from ..utils.callback_utils import vent_flag

        id_cols = [col for col in frame.columns if 'id' in col.lower()]
        index_var = source.index_var
        
        # 🔥 FIX: 如果 source.index_var 是 None，尝试从表配置获取默认 index_var
        # eICU vent_start 源有 index_var: None，但表 respiratorycare 的默认是 respcarestatusoffset
        if index_var is None and data_source is not None:
            try:
                table_cfg = data_source.config.get_table(source.table)
                if table_cfg and table_cfg.defaults:
                    index_var = table_cfg.defaults.index_var
                    if DEBUG_MODE:
                        print(f"   🔧 vent_flag: source.index_var=None，使用表默认 index_var='{index_var}'")
            except Exception:
                pass
        
        # 🔥 R ricu vent_flag: val_var 是原始列名（如 ventstartoffset），不是概念名
        # vent_flag 会将 val_var 的值作为新的时间索引，然后将 val_var 设为 TRUE
        # 🔧 FIX: 如果 value_var 已被重命名为 concept_name，使用 concept_name
        val_col = source.value_var if hasattr(source, 'value_var') and source.value_var else concept_name
        if val_col not in frame.columns and concept_name in frame.columns:
            val_col = concept_name
        return vent_flag(
            frame,
            val_col=val_col,
            index_var=index_var,
            id_cols=id_cols,
        )

    match = re.fullmatch(r"eicu_duration\(\s*gap_length\s*=\s*(.+)\)", expr, flags=re.DOTALL)
    if match:
        from ..utils.callback_utils import eicu_duration_callback

        gap_arg = match.group(1)
        # Parse interval expression directly
        gap_expr = gap_arg.strip()
        interval_match = re.fullmatch(r"([a-zA-Z]+)\((.+)\)", gap_expr)
        if interval_match:
            unit = interval_match.group(1).lower()
            value = _parse_literal(interval_match.group(2))
            if unit in {"min", "mins", "minute", "minutes"}:
                gap = pd.to_timedelta(value, unit="m")
            elif unit in {"hour", "hours"}:
                gap = pd.to_timedelta(value, unit="h")
            elif unit in {"sec", "secs", "second", "seconds"}:
                gap = pd.to_timedelta(value, unit="s")
            elif unit in {"day", "days"}:
                gap = pd.to_timedelta(value, unit="d")
            else:
                raise ValueError(f"Unsupported interval unit '{unit}' in expression '{gap_expr}'")
        else:
            raise ValueError(f"Unsupported interval expression '{gap_arg}'")
        
        callback_fn = eicu_duration_callback(gap)
        # 只使用患者级别的ID列进行分组，不要使用行级别的唯一ID（如infusiondrugid）
        # 否则每组只有一行，duration计算会变成0
        patient_id_cols = ['patientunitstayid', 'stay_id', 'icustay_id', 'hadm_id', 'admissionid', 'patientid']
        id_cols = [col for col in patient_id_cols if col in frame.columns]
        if not id_cols:
            # 回退到通用检测，但排除明显的行级别ID
            excluded_patterns = ['infusion', 'drug', 'event', 'row', 'fluid']
            id_cols = [col for col in frame.columns 
                      if 'id' in col.lower() 
                      and not any(pat in col.lower() for pat in excluded_patterns)]
        index_var = source.index_var
        return callback_fn(
            frame,
            val_col=concept_name,
            index_var=index_var,
            id_cols=id_cols,
        )

    # Handle eicu_rate_kg(ml_to_mcg = VALUE) - eICU dose rate conversion with weight
    match = re.fullmatch(r"eicu_rate_kg\(\s*ml_to_mcg\s*=\s*(.+)\)", expr, flags=re.DOTALL)
    if match:
        from ..utils.callback_utils import eicu_rate_kg_callback
        
        ml_to_mcg = float(match.group(1))
        callback_fn = eicu_rate_kg_callback(ml_to_mcg)
        
        # Get necessary variables
        val_var = source.value_var or concept_name
        sub_var = source.sub_var
        weight_var = source.params.get('weight_var', 'admissionweight') if source.params else 'admissionweight'
        
        return callback_fn(
            frame,
            val_var=val_var,
            sub_var=sub_var,
            weight_var=weight_var,
            concept_name=concept_name,
            data_source=data_source,
            patient_ids=patient_ids,
        )
        
    match = re.fullmatch(r"eicu_rate_units\((.+)\)", expr, flags=re.DOTALL)
    if match:
        from ..utils.callback_utils import eicu_rate_units_callback

        args = _split_arguments(match.group(1))
        if len(args) < 2:
            raise ValueError(f"eicu_rate_units requires two arguments, got '{expr}'")

        def _arg_to_float(text: str) -> float:
            part = text.split("=", 1)[1] if "=" in text else text
            return float(_parse_literal(part.strip()))

        ml_to_mcg = _arg_to_float(args[0])
        mcg_to_units = _arg_to_float(args[1])
        callback_fn = eicu_rate_units_callback(ml_to_mcg, mcg_to_units)

        val_var = source.value_var or concept_name
        sub_var = source.sub_var

        return callback_fn(
            frame,
            val_var=val_var,
            sub_var=sub_var,
            concept_name=concept_name,
        )

    # Handle eicu_rate_mass(target_unit = "mcg/hour") - non-kg mass-rate drugs
    match = re.fullmatch(
        r"eicu_rate_mass\(\s*target_unit\s*=\s*['\"]?([^'\"\)]+?)['\"]?\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if match:
        from ..utils.callback_utils import eicu_rate_mass_callback

        target_unit = match.group(1).strip()
        callback_fn = eicu_rate_mass_callback(target_unit)

        val_var = source.value_var or concept_name
        sub_var = source.sub_var

        return callback_fn(
            frame,
            val_var=val_var,
            sub_var=sub_var,
            concept_name=concept_name,
            data_source=data_source,
            patient_ids=patient_ids,
        )

    if expr == "aumc_rate_kg":
        from ..utils.callback_utils import aumc_rate_kg

        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        rel_weight = source.params.get("rel_weight") if source.params else None
        rate_uom = source.params.get("rate_uom") if source.params else None
        if rate_uom is None and "rateunit" in frame.columns:
            rate_uom = "rateunit"
        stop_var = source.params.get("stop_var") if source.params else None
        index_var = source.index_var
        
        # source.index_var may be None, use table default as fallback
        # For AUMC drugitems, the index_var should be 'start'
        if not index_var and source.table == 'drugitems':
            index_var = 'start'

        # 🔧 FIX: 获取体重概念并合并到 frame 中
        # R ricu 在回调中使用 add_weight(res, env, "weight") 获取体重
        # easyicu 需要在调用回调前加载 weight 概念
        # 🔧 FIX 2: Only try to get weight if frame is not empty
        if not frame.empty and 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                # 获取患者ID列
                id_cols = [c for c in frame.columns if c.lower().endswith('id') and c != 'itemid']
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    # 加载 weight 概念
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,  # 不聚合，保留原始值
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,  # 避免回调循环
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        # 确保weight列是数值型
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            # 合并到frame
                            merge_cols = [c for c in id_cols if c in weight_df.columns]
                            if merge_cols:
                                frame = frame.merge(
                                    weight_df[merge_cols + ['weight']].drop_duplicates(),
                                    on=merge_cols,
                                    how='left'
                                )
            except Exception as e:
                # 如果获取体重失败，使用默认值
                if DEBUG_MODE:
                    print(f"   ⚠️  获取体重失败: {e}")
                pass

        return aumc_rate_kg(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            rel_weight_col=rel_weight,
            rate_unit_col=rate_uom,
            index_col=index_var,
            stop_col=stop_var,
        )

    # Handle aumc_rate_mass(target_unit = "mcg/hour") — non-kg mass-rate
    match = re.fullmatch(
        r"aumc_rate_mass\(\s*target_unit\s*=\s*['\"]?([^'\"\)]+?)['\"]?\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if match:
        from ..utils.callback_utils import aumc_rate_mass

        target_unit = match.group(1).strip()

        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        rate_uom = source.params.get("rate_uom") if source.params else None
        if rate_uom is None and "rateunit" in frame.columns:
            rate_uom = "rateunit"
        stop_var = source.params.get("stop_var") if source.params else None
        index_var = source.index_var
        if not index_var and source.table == "drugitems":
            index_var = "start"

        return aumc_rate_mass(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            rate_unit_col=rate_uom,
            index_col=index_var,
            stop_col=stop_var,
            target_unit=target_unit,
        )

    # Handle hirid_duration callback - calculate infusion durations
    if expr == "hirid_duration":
        from ..utils.callback_utils import hirid_duration
        
        index_var = source.index_var or 'givenat'
        val_var = source.value_var
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        
        return hirid_duration(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            index_col=index_var,
            grp_var=grp_var,
        )

    # Handle hirid_vent callback - convert ventilation records to window table
    if expr == "hirid_vent":
        from ..utils.callback_utils import hirid_vent
        
        index_var = source.index_var or 'datetime'
        val_var = source.value_var
        
        return hirid_vent(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            index_col=index_var,
            dur_var='dur_var',
            padding_hours=4.0,
            max_gap_hours=12.0,
            expand_to_hourly=False,  # Return win_tbl format, not expanded ts_tbl
        )

    # Handle HiRID's directly recorded hourly urine-rate source.
    if expr == "hirid_urine":
        from ..utils.callback_utils import hirid_urine
        
        val_var = source.value_var or 'value'
        unit_var = source.unit_var
        
        return hirid_urine(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            interval=interval,
        )

    # Handle hirid_rate_kg callback - HiRID dose rate per kg
    if expr == "hirid_rate_kg":
        from ..utils.callback_utils import hirid_rate_kg

        val_var = source.value_var or 'givendose'
        unit_var = source.unit_var or 'doseunit'
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        index_var = source.index_var or 'givenat'
        
        # 🔧 FIX: Only try to get weight if frame is not empty
        # Avoids reading huge observations table (70M rows) when there's no data
        if not frame.empty and 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                id_col = None
                for cand in ['patientid', 'stay_id', 'admissionid']:
                    if cand in frame.columns:
                        id_col = cand
                        break
                if id_col:
                    unique_ids = frame[id_col].unique().tolist()
                    weight_per_patient = None
                    
                    # 🔧 FIX 2026-03-12: For HiRID, load raw weight from parquet
                    # and compute direct per-patient median (bypassing DuckDB hourly aggregation).
                    # R ricu: load_concepts("weight", aggregate=NULL) → median(all_raw_values)
                    # Previous easyicu: DuckDB GROUP BY (patient,hour) MEDIAN → groupby(patient).median() = "median of medians" ≠ direct median
                    db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                    if db_name == 'hirid':
                        try:
                            import duckdb
                            bucket_dir = data_source.base_path / 'observations_bucket'
                            if bucket_dir.exists():
                                conn = duckdb.connect()
                                conn.execute("SET memory_limit = '2GB'")
                                # 显式文件列表，过滤 AppleDouble
                                _wh_files = _enumerate_bucket_parquet_files(bucket_dir)
                                if _wh_files:
                                    _wh_files_sql = "[" + ", ".join(f"'{f}'" for f in _wh_files) + "]"
                                    _wh_read_expr = f"read_parquet({_wh_files_sql}, hive_partitioning=true, union_by_name=true)"
                                else:
                                    _wh_read_expr = f"read_parquet('{_duckdb_path(bucket_dir)}/**/*.parquet', hive_partitioning=true)"
                                # weight variableid = 10000400
                                pid_list = ','.join(str(int(p)) for p in unique_ids)
                                sql = f"""
                                    SELECT patientid, MEDIAN(value) as weight
                                    FROM {_wh_read_expr}
                                    WHERE variableid = 10000400 AND patientid IN ({pid_list})
                                      AND value IS NOT NULL AND value >= 1 AND value <= 500
                                    GROUP BY patientid
                                """
                                weight_per_patient = conn.execute(sql).fetchdf()
                                conn.close()
                        except Exception:
                            weight_per_patient = None
                    
                    if weight_per_patient is None or weight_per_patient.empty:
                        # Fallback: use standard loading path
                        weight_table = resolver._load_single_concept(
                            'weight',
                            data_source,
                            aggregator=False,
                            patient_ids={id_col: unique_ids},
                            verbose=False,
                            _bypass_callback=True,
                        )
                        if weight_table is not None and not weight_table.data.empty:
                            weight_df = weight_table.data
                            if 'weight' in weight_df.columns:
                                weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                                weight_per_patient = weight_df.groupby(id_col)['weight'].median().reset_index()
                    
                    if weight_per_patient is not None and not weight_per_patient.empty:
                        frame = frame.merge(weight_per_patient, on=id_col, how='left')
            except Exception as e:
                if DEBUG_MODE:
                    print(f"   ⚠️  获取体重失败: {e}")
                pass

        # 🔧 FIX: Calculate interval_minutes from concept's interval
        # R ricu uses frac = 1 / interval(x), where interval(x) is the concept's interval.
        # For dobu_rate (no interval): default 60min (1 hour)
        # For dobu60 (interval="00:01:00"): 1min → rate is 60x higher
        interval_minutes = 60.0  # default
        if interval is not None:
            if isinstance(interval, str):
                # Parse string like "00:01:00" (1 minute) or "01:00:00" (1 hour)
                try:
                    td = pd.to_timedelta(interval)
                    interval_minutes = td.total_seconds() / 60.0
                except Exception:
                    pass
            elif isinstance(interval, pd.Timedelta):
                interval_minutes = interval.total_seconds() / 60.0
        
        return hirid_rate_kg(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            grp_var=grp_var,
            index_col=index_var,
            interval_minutes=interval_minutes,
            value_min=_get_concept_bounds(concept_name, 'min'),
            value_max=_get_concept_bounds(concept_name, 'max'),
        )

    # Handle hirid_rate callback - HiRID dose rate (no weight normalization)
    if expr == "hirid_rate":
        from ..utils.callback_utils import hirid_rate

        val_var = source.value_var or 'givendose'
        unit_var = source.unit_var or 'doseunit'
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        index_var = source.index_var or 'givenat'

        return hirid_rate(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            grp_var=grp_var,
            index_col=index_var,
        )

    # Handle hirid_rate_mass(target_unit = "mcg/hour") - HiRID mass-rate (non-kg)
    match = re.fullmatch(
        r"hirid_rate_mass\(\s*target_unit\s*=\s*['\"]?([^'\"\)]+?)['\"]?\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if match:
        from ..utils.callback_utils import hirid_rate_mass

        target_unit = match.group(1).strip()

        val_var = source.value_var or 'givendose'
        unit_var = source.unit_var or 'doseunit'
        grp_var = source.params.get("grp_var") if source.params else None
        if not grp_var:
            grp_var = getattr(source, 'grp_var', None)
        if not grp_var and 'infusionid' in frame.columns:
            grp_var = 'infusionid'
        index_var = source.index_var or 'givenat'

        return hirid_rate_mass(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            unit_col=unit_var,
            grp_var=grp_var,
            index_col=index_var,
            target_unit=target_unit,
        )

    # Handle aumc_rate callback - combine unit_var and rate_var into unit/rate format
    # R: x <- x[, c(unit_var) := do_call(.SD, paste, sep = "/"), .SDcols = c(unit_var, rate_var)]
    # 🔧 FIX 2025-02-03: Also normalize rate units (min -> hr conversion)
    if expr == "aumc_rate":
        rate_var = getattr(source, 'rate_var', None)
        if not rate_var and source.params:
            rate_var = source.params.get("rate_var")
        unit_var = source.unit_var or unit_column
        val_var = source.value_var or concept_name
        
        if rate_var and rate_var in frame.columns:
            frame = frame.copy()
            # Normalize rate units: 'min' means per-minute, need to multiply by 60 to get per-hour
            # R ricu does this in aumc_rate_kg with hr_to_min, but aumc_rate needs it too for dex
            rate_lower = frame[rate_var].astype(str).str.lower().str.strip()
            
            # If rate_var is 'min' (per minute), multiply value by 60 to get per hour
            mask_min = rate_lower.isin({'min', 'minute', 'minutes', 'm'})
            if mask_min.any() and val_var in frame.columns:
                frame.loc[mask_min, val_var] = frame.loc[mask_min, val_var] * 60.0
                frame.loc[mask_min, rate_var] = 'uur'  # Now it's per hour
            
            # Combine unit and rate into "unit/rate" format
            if unit_var and unit_var in frame.columns:
                frame[unit_var] = frame[unit_var].astype(str) + "/" + frame[rate_var].astype(str)
        return frame

    match = re.fullmatch(r"aumc_rate_units\(\s*([0-9eE+\-\.]+)\s*\)", expr)
    if match:
        from ..utils.callback_utils import aumc_rate_units_callback

        factor = float(match.group(1))
        callback_fn = aumc_rate_units_callback(factor)

        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        rate_uom = source.params.get("rate_uom") if source.params else None
        if rate_uom is None and "rateunit" in frame.columns:
            rate_uom = "rateunit"
        stop_var = source.params.get("stop_var") if source.params else None

        return callback_fn(
            frame,
            val_col=val_var,
            unit_col=unit_var,
            rate_unit_col=rate_uom,
            stop_col=stop_var,
            concept_name=concept_name,
        )

    if expr == "aumc_dur":
        from ..utils.callback_utils import aumc_dur

        val_var = source.value_var or concept_name
        # stop_var and grp_var can be direct attributes on source or in source.params
        stop_var = getattr(source, 'stop_var', None)
        if not stop_var and source.params:
            stop_var = source.params.get("stop_var")
        grp_var = getattr(source, 'grp_var', None)
        if not grp_var and source.params:
            grp_var = source.params.get("grp_var")
        index_var = source.index_var
        continuous_var = (
            source.params.get("continuous_var", "iscontinuous")
            if source.params
            else "iscontinuous"
        )
        action_var = (
            source.params.get("action_var", "action")
            if source.params
            else "action"
        )
        merge_gap_minutes = (
            source.params.get("merge_gap_minutes", 5.0)
            if source.params
            else 5.0
        )

        return aumc_dur(
            frame,
            val_col=val_var,
            stop_var=stop_var,
            grp_var=grp_var,
            index_var=index_var,
            concept_name=concept_name,
            continuous_var=continuous_var,
            action_var=action_var,
            merge_gap_minutes=merge_gap_minutes,
        )

    # Handle aumc_bxs callback - negate values where direction is '-'
    # R implementation: x[get(dir_var) == "-", val_var := -1L * get(val_var)]
    if expr == "aumc_bxs":
        dir_var = getattr(source, 'dir_var', None)
        if not dir_var and source.params:
            dir_var = source.params.get("dir_var")
        if not dir_var:
            dir_var = "tag"  # default for AUMC
        
        val_var = concept_name  # Value column has already been renamed to concept_name
        
        if dir_var in frame.columns and val_var in frame.columns:
            # Negate values where direction is '-'
            mask = frame[dir_var] == '-'
            if mask.any():
                frame = frame.copy()
                frame.loc[mask, val_var] = -1 * frame.loc[mask, val_var]
        return frame

    # Handle eicu_age callback
    if expr == "transform_fun(eicu_age)":
        from ..utils.callback_utils import eicu_age
        return eicu_age(frame, val_col=concept_name)

    # Handle aumc_rass callback
    if expr == "transform_fun(aumc_rass)":
        # Apply aumc_rass transformation: extract first 2 characters as integer
        # Similar to ricu's: as.integer(substr(x, 1L, 2L))
        series = frame[concept_name].copy()
        series = series.astype(str).str[:2]
        series = pd.to_numeric(series, errors='coerce')
        frame[concept_name] = series
        return frame

    # Handle MIMIC-III mimic_age callback
    # R ricu logic:
    #   1. change_id mechanism converts dob column to (intime - dob) time difference
    #   2. mimic_age: x <- as.double(x, units = "days") / -365; ifelse(x > 90, 90, x)
    # EasyICU: need to manually join patients with icustays to get intime and calculate age
    if expr == "transform_fun(mimic_age)" or expr == "mimic_age":
        frame = frame.copy()
        val_col = source.value_var if source else 'dob'
        
        # Check if we have dob (birth date) that needs to be converted
        # Note: At this point, dob may have been renamed to concept_name (e.g., 'age')
        # So we check for either 'dob' column or concept_name column that was originally 'dob'
        has_dob = 'dob' in frame.columns
        dob_renamed_to_concept = (val_col == 'dob' and concept_name in frame.columns and 'dob' not in frame.columns)
        
        if has_dob or dob_renamed_to_concept:
            # Determine actual column name containing DOB data
            actual_dob_col = 'dob' if has_dob else concept_name
            # Need to load icustays to get intime for each patient
            if data_source is not None:
                try:
                    # For MIMIC-III: 
                    # frame's 'stay_id' column contains 'icustay_id' values (already joined/replaced)
                    # We need to merge with icustays on icustay_id to get intime
                    
                    db_name = data_source.config.name if hasattr(data_source, 'config') else 'mimic'
                    
                    # Load icustays with intime
                    icustays = data_source.load_table(
                        'icustays',
                        columns=['icustay_id', 'intime'],
                        verbose=False
                    )
                    if hasattr(icustays, 'data'):
                        icustays = icustays.data
                    
                    # Determine the ID column in frame
                    # In MIMIC-III, frame's 'stay_id' contains icustay_id values
                    if 'stay_id' in frame.columns:
                        # Rename for consistent merge
                        frame = frame.rename(columns={'stay_id': 'icustay_id'})
                        merge_col = 'icustay_id'
                    elif 'icustay_id' in frame.columns:
                        merge_col = 'icustay_id'
                    else:
                        merge_col = None
                    
                    if merge_col is not None and merge_col in icustays.columns:
                        # Merge to get intime
                        frame = frame.merge(icustays[['icustay_id', 'intime']], on=merge_col, how='left')
                        
                        if len(frame) == 0:
                            print("⚠️ [mimic_age] MERGE PRODUCED 0 ROWS!")
                            return frame
                        
                        # 2026-05-19 fix: MIMIC-III shifts dob to year 2300+
                        # for patients >=89 ("date_shift"). pandas Timestamps
                        # cap at year 2262 (datetime64[ns] window), so the
                        # naive (intime - dob) below blew up with
                        #   OverflowError: Overflow in int64 addition
                        # and lost the entire `age` concept for MIMIC-III.
                        # Parse Y/M/D from the string columns and do
                        # year-arithmetic age — never overflows, and the
                        # > 90 cap below makes the imprecise month/day shift
                        # for the obfuscated cohort moot.
                        def _ymd(series):
                            s = series.astype(str).str.extract(
                                r'^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})'
                            )
                            return (
                                pd.to_numeric(s['year'], errors='coerce'),
                                pd.to_numeric(s['month'], errors='coerce'),
                                pd.to_numeric(s['day'], errors='coerce'),
                            )

                        d_y, d_m, d_d = _ymd(frame[actual_dob_col])
                        i_y, i_m, i_d = _ymd(frame['intime'])
                        year_diff = i_y - d_y
                        before_birthday = (
                            (i_m < d_m) | ((i_m == d_m) & (i_d < d_d))
                        ).fillna(False).astype(int)
                        age_years = (year_diff - before_birthday).astype(float)
                        # Cap at 90 (R ricu: ifelse(x > 90, 90, x))
                        age_years = np.where(age_years > 90, 90, age_years)
                        frame[concept_name] = age_years
                        
                        # Use icustay_id as the final stay_id
                        if 'icustay_id' in frame.columns:
                            frame = frame.rename(columns={'icustay_id': 'stay_id'})
                        
                        # Clean up temporary columns
                        for col in ['intime', actual_dob_col, 'subject_id']:
                            if col in frame.columns and col != concept_name and col != 'stay_id':
                                frame = frame.drop(columns=[col])
                    
                except Exception as e:
                    # If loading fails, try simpler approach
                    import traceback
                    print(f"⚠️ mimic_age callback failed: {e}")
                    traceback.print_exc()
                    if concept_name in frame.columns:
                        frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                        frame.loc[frame[concept_name] > 90, concept_name] = 90
            else:
                # No data_source - just cap at 90 if age already exists
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                    frame.loc[frame[concept_name] > 90, concept_name] = 90
        elif concept_name in frame.columns:
            # Age already calculated - just cap at 90
            frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            frame.loc[frame[concept_name] > 90, concept_name] = 90
        
        return frame

    # Handle MIMIC-III mimic_abx_presc callback
    # R ricu logic: x[, c(idx, val_var) := list(get(idx) + mins(720L), TRUE)]
    if expr == "mimic_abx_presc":
        frame = frame.copy()
        index_col = source.index_var
        if not index_col:
            for candidate in ["charttime", "starttime", "startdate"]:
                if candidate in frame.columns:
                    index_col = candidate
                    break
        # Shift time forward by 720 minutes (12 hours)
        if index_col and index_col in frame.columns:
            frame[index_col] = pd.to_numeric(frame[index_col], errors='coerce') + 720
        # Set value to TRUE
        frame[concept_name] = True
        return frame

    # Handle MIMIC-III mimic_kg_rate callback
    # R ricu logic: add_weight + divide by weight + update unit
    if expr == "mimic_kg_rate":
        val_var = source.value_var or concept_name
        unit_var = source.unit_var or unit_column
        
        # Try to add weight and divide
        if 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                id_cols = [c for c in frame.columns if c.lower().endswith('id') and c != 'itemid']
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            merge_cols = [c for c in id_cols if c in weight_df.columns]
                            if merge_cols:
                                frame = frame.merge(
                                    weight_df[merge_cols + ['weight']].drop_duplicates(),
                                    on=merge_cols,
                                    how='left'
                                )
            except Exception:
                pass
        
        # Divide rate by weight
        if 'weight' in frame.columns and val_var in frame.columns:
            frame = frame.copy()
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
            frame['weight'] = pd.to_numeric(frame['weight'], errors='coerce')
            mask = frame['weight'] > 0
            frame.loc[mask, val_var] = frame.loc[mask, val_var] / frame.loc[mask, 'weight']
            # Update unit
            if unit_var and unit_var in frame.columns:
                frame[unit_var] = frame[unit_var].str.replace('mcgmin', 'mcg/kg/min', regex=False)
            frame = frame.drop(columns=['weight'], errors='ignore')
        return frame

    # Handle SICdb sic_dur callback
    # R ricu logic: calc_dur(x, val_var, index_var(x), stop_var, grp_var)
    if expr == "sic_dur":
        val_var = source.value_var or concept_name
        index_var = source.index_var
        stop_var = source.params.get("stop_var") if source.params else None
        grp_var = source.params.get("grp_var") if source.params else None
        
        if not stop_var:
            for candidate in ["OffsetDrugEnd", "stop", "endtime"]:
                if candidate in frame.columns:
                    stop_var = candidate
                    break
        
        if not index_var:
            for candidate in ["Offset", "OffsetDrugStart", "start", "charttime"]:
                if candidate in frame.columns:
                    index_var = candidate
                    break
        
        if stop_var and stop_var in frame.columns and index_var and index_var in frame.columns:
            frame = frame.copy()
            # Use standard patient-level ID columns only (not row-level id, bucket_id, PatientID etc.)
            _PATIENT_ID_COLS = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
            id_cols = [c for c in _PATIENT_ID_COLS if c in frame.columns]
            
            # Group by ID (and optionally grp_var)
            group_cols = list(id_cols)
            if grp_var and grp_var in frame.columns:
                group_cols = id_cols + [grp_var]
            
            if group_cols:
                # Calculate duration = max(stop) - min(start) per group
                agg_df = frame.groupby(group_cols).agg({
                    index_var: 'min',
                    stop_var: 'max'
                }).reset_index()
                
                # SICdb medication.Offset is in seconds → convert to hours
                # Duration = floor(max_stop/3600) - floor(min_start/3600)
                # This matches R ricu's change_interval(hours(1)) behavior
                min_start = pd.to_numeric(agg_df[index_var], errors='coerce')
                max_stop = pd.to_numeric(agg_df[stop_var], errors='coerce')
                start_hours = (min_start // 3600).astype(int)
                stop_hours = (max_stop // 3600).astype(int)
                agg_df[val_var] = stop_hours - start_hours
                agg_df[index_var] = start_hours
                
                # If grp_var was used, set index to min per patient and pick max duration
                if grp_var and grp_var in frame.columns and id_cols:
                    min_idx = agg_df.groupby(id_cols)[index_var].transform('min')
                    agg_df[index_var] = min_idx
                    agg_df = agg_df.sort_values(val_var, ascending=False).drop_duplicates(
                        subset=id_cols + [index_var], keep='first')
                
                # Keep only required columns
                result_cols = id_cols + [index_var, val_var]
                frame = agg_df[[c for c in result_cols if c in agg_df.columns]]
        
        return frame

    # Handle sic_rate_mass(target_unit = "mcg/hour") — SIC non-kg mass-rate
    # for sedatives/analgesics where AmountPerMinute is actually total dose
    match = re.fullmatch(
        r"sic_rate_mass\(\s*target_unit\s*=\s*['\"]?([^'\"\)]+?)['\"]?\s*\)",
        expr,
        flags=re.DOTALL,
    )
    if match:
        from ..utils.callback_utils import sic_rate_mass

        target_unit = match.group(1).strip()
        val_var = source.value_var or concept_name
        if val_var not in frame.columns and concept_name in frame.columns:
            val_var = concept_name
        stop_var = source.params.get("stop_var") if source.params else None
        if not stop_var:
            for candidate in ["OffsetDrugEnd", "stop", "endtime"]:
                if candidate in frame.columns:
                    stop_var = candidate
                    break
        index_var = source.index_var
        if not index_var:
            for candidate in ["Offset", "OffsetDrugStart", "start", "charttime"]:
                if candidate in frame.columns:
                    index_var = candidate
                    break

        return sic_rate_mass(
            frame,
            concept_name=concept_name,
            val_col=val_var,
            index_col=index_var,
            stop_col=stop_var,
            target_unit=target_unit,
        )

    # Handle SICdb sic_rate_kg callback
    # R ricu logic: add_weight + multiply by 10^6 / weight + expand
    if expr == "sic_rate_kg":
        val_var = source.value_var or concept_name
        # Fix: source.value_var may have been renamed to concept_name during loading
        if val_var not in frame.columns and concept_name in frame.columns:
            val_var = concept_name
        stop_var = source.params.get("stop_var") if source.params else None
        
        if not stop_var:
            for candidate in ["OffsetDrugEnd", "stop", "endtime"]:
                if candidate in frame.columns:
                    stop_var = candidate
                    break
        
        # Try to add weight
        if 'weight' not in frame.columns and resolver is not None and data_source is not None:
            try:
                _PATIENT_ID_COLS_W = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
                id_cols = [c for c in _PATIENT_ID_COLS_W if c in frame.columns]
                if id_cols:
                    unique_ids = frame[id_cols[0]].unique().tolist()
                    weight_table = resolver._load_single_concept(
                        'weight',
                        data_source,
                        aggregator=False,
                        patient_ids={id_cols[0]: unique_ids},
                        verbose=False,
                        _bypass_callback=True,
                    )
                    if weight_table is not None and not weight_table.data.empty:
                        weight_df = weight_table.data
                        if 'weight' in weight_df.columns:
                            weight_df['weight'] = pd.to_numeric(weight_df['weight'], errors='coerce')
                            # Get first weight per patient
                            weight_id_col = id_cols[0] if id_cols[0] in weight_df.columns else (
                                'CaseID' if 'CaseID' in weight_df.columns else None
                            )
                            if weight_id_col:
                                weight_agg = weight_df.groupby(weight_id_col)['weight'].first().reset_index()
                                frame = frame.merge(weight_agg, on=weight_id_col, how='left')
            except Exception:
                pass
        
        # Convert rate: multiply by 10^6 / weight (mg -> mcg/kg)
        if 'weight' in frame.columns and val_var in frame.columns:
            frame = frame.copy()
            frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
            frame['weight'] = pd.to_numeric(frame['weight'], errors='coerce')
            mask = frame['weight'] > 0
            frame.loc[mask, val_var] = frame.loc[mask, val_var] * 1e6 / frame.loc[mask, 'weight']
            frame = frame.drop(columns=['weight'], errors='ignore')
        
        # Expand time range: convert each (start, stop) interval into hourly rows
        index_var = source.index_var
        if not index_var:
            for candidate in ["Offset", "OffsetDrugStart", "start", "charttime"]:
                if candidate in frame.columns:
                    index_var = candidate
                    break
        
        if stop_var and stop_var in frame.columns and index_var and index_var in frame.columns:
            # R ricu expand(): generate hourly observations between start and stop
            # Use standard patient-level ID columns only
            _PATIENT_ID_COLS_E = ['CaseID', 'stay_id', 'icustay_id', 'patientunitstayid', 'admissionid', 'patientid']
            id_cols = [c for c in _PATIENT_ID_COLS_E if c in frame.columns]
            keep_cols = id_cols + [val_var] if val_var in frame.columns else id_cols
            
            expanded_rows = []
            for _, row in frame.iterrows():
                start_val = pd.to_numeric(row.get(index_var), errors='coerce')
                stop_val = pd.to_numeric(row.get(stop_var), errors='coerce')
                if pd.isna(start_val) or pd.isna(stop_val) or stop_val <= start_val:
                    continue
                # SICdb medication.Offset is in seconds → convert to hourly steps
                # R ricu floor(): floor to nearest hour
                start_hour = int(start_val // 3600)
                stop_hour = int(stop_val // 3600)
                for t in range(start_hour, stop_hour + 1):
                    new_row = {index_var: t}  # Output Offset in hours
                    for c in keep_cols:
                        if c in row.index:
                            new_row[c] = row[c]
                    expanded_rows.append(new_row)
            
            if expanded_rows:
                frame = pd.DataFrame(expanded_rows)
                # 🔧 FIX 2026-03-11: Do NOT hardcode median aggregation here!
                # Previously this did groupby(...).agg({val_var: 'median'}) which pre-aggregated
                # the expanded rates. This prevents vaso60 callback from getting raw per-interval
                # rates needed for MAX aggregation (dobu60/norepi60/epi60/dopa60 etc).
                # change_interval() in _load_single_concept handles aggregation correctly:
                #   - standalone dobu_rate: change_interval(median) → correct median
                #   - vaso60 sub-concept: change_interval(False) → preserves all rows → vaso60 takes max
                if val_var in frame.columns:
                    frame[val_var] = pd.to_numeric(frame[val_var], errors='coerce')
        
        return frame

    if expr.strip() == "distribute_amount":
        from ..utils.callback_utils import distribute_amount
        end_col = source.params.get("end_var") if source.params else None
        if not end_col:
            end_col = source.params.get("dur_var") if source.params else None
        if not end_col and "endtime" in frame.columns:
            end_col = "endtime"
        index_col = source.index_var
        # 🔧 FIX: 添加 starttime 作为 fallback，用于 inputevents 表的数据 (如 ins)
        if not index_col:
            for candidate in ["charttime", "starttime", "time"]:
                if candidate in frame.columns:
                    index_col = candidate
                    break
        unit_col = unit_column or source.unit_var
        if not unit_col:
            if "rateuom" in frame.columns:
                unit_col = "rateuom"
            elif "valueuom" in frame.columns:
                unit_col = "valueuom"
        if not end_col or end_col not in frame.columns:
            return frame
        if not index_col or index_col not in frame.columns:
            return frame
        
        # 🔧 FIX 2025-01: Get admission times for R ricu-compatible floor behavior
        # R ricu converts datetime to relative time BEFORE callbacks (in load_mihi).
        # This affects floor() behavior in expand().
        admission_times = None
        if data_source is not None:
            try:
                # 🔧 FIX 2026-02-09: 正确检测 ID 列
                # MIMIC-III 使用 icustay_id，需要明确指定
                db_name = data_source.config.name if hasattr(data_source, 'config') and hasattr(data_source.config, 'name') else ''
                if db_name in ['mimic', 'mimic_demo']:
                    id_cols_for_icustays = ['icustay_id']
                else:
                    id_cols_for_icustays = ['stay_id', 'icustay_id', 'hadm_id', 'admissionid', 'patientid', 'patientunitstayid']
                
                # Load icustays to get admission times
                icustays_result = data_source.load_table('icustays')
                # Handle ICUTable or DataFrame result
                if hasattr(icustays_result, 'data'):
                    icustays = icustays_result.data
                else:
                    icustays = icustays_result
                    
                if icustays is not None and len(icustays) > 0:
                    # Find ID column
                    id_col = None
                    for col in id_cols_for_icustays:
                        if col in icustays.columns:
                            id_col = col
                            break
                    if id_col is not None:
                        # Filter to patients in the current frame
                        patient_ids_in_frame = frame[id_col].unique() if id_col in frame.columns else None
                        if patient_ids_in_frame is not None:
                            admission_times = icustays[icustays[id_col].isin(patient_ids_in_frame)][[id_col, 'intime']].drop_duplicates()
            except Exception:
                pass  # Fail silently - will use fallback floor behavior
        
        return distribute_amount(
            frame,
            val_col=concept_name,
            unit_col=unit_col,
            end_col=end_col,
            index_col=index_col,
            admission_times=admission_times,  # 🔧 Pass admission times for proper floor behavior
        )

    if expr.strip() == "distribute_volume_hourly":
        from ..utils.callback_utils import (
            distribute_volume_hourly,
            normalize_volume_to_ml,
        )

        params = source.params or {}
        end_col = params.get("end_var")
        if not end_col:
            end_col = "endtime" if "endtime" in frame.columns else "stop"
        index_col = source.index_var
        if not index_col:
            index_col = next(
                (
                    candidate
                    for candidate in ("starttime", "start", "charttime")
                    if candidate in frame.columns
                ),
                None,
            )
        if not index_col or index_col not in frame.columns:
            return frame

        db_name = ""
        if data_source is not None:
            db_name = getattr(getattr(data_source, "config", None), "name", "")
        id_preferences = {
            "aumc": ("admissionid",),
            "mimic": ("icustay_id",),
            "mimic_demo": ("icustay_id",),
            "miiv": ("stay_id",),
        }.get(db_name, ("stay_id", "icustay_id", "admissionid"))
        id_col = next((column for column in id_preferences if column in frame.columns), None)
        if id_col is None:
            raise ValueError(
                f"{db_name or 'unknown database'} total-input allocation has no "
                "stay-level identifier"
            )

        alternate_value_col = params.get("alternate_value_var")
        if alternate_value_col and alternate_value_col in frame.columns:
            frame = frame.copy()
            frame[concept_name] = pd.concat(
                [
                    pd.to_numeric(frame[concept_name], errors="coerce"),
                    pd.to_numeric(frame[alternate_value_col], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1, skipna=True)

        volume_unit_col = source.unit_var or unit_column
        if volume_unit_col and volume_unit_col in frame.columns:
            frame = frame.copy()
            frame[concept_name] = normalize_volume_to_ml(
                frame[concept_name], frame[volume_unit_col]
            )

        origin_times = None
        origin_col = None
        numeric_time_unit = "hours"
        output_time_unit = "relative_hours"
        if db_name == "aumc":
            if data_source is None:
                raise ValueError("AUMC volume allocation requires admissions.admittedat")
            origin_col = "admittedat"
            origin_result = data_source.load_table(
                "admissions",
                columns=[id_col, origin_col],
                verbose=False,
            )
            origin_times = (
                origin_result.data
                if hasattr(origin_result, "data")
                else origin_result
            )
            numeric_time_unit = "minutes"
            output_time_unit = "absolute_minutes"
        elif not pd.api.types.is_numeric_dtype(frame[index_col]):
            if data_source is None:
                raise ValueError(
                    "datetime volume allocation requires icustays.intime"
                )
            origin_col = "intime"
            origin_result = data_source.load_table(
                "icustays",
                columns=[id_col, origin_col],
                verbose=False,
            )
            origin_times = (
                origin_result.data
                if hasattr(origin_result, "data")
                else origin_result
            )

        result = distribute_volume_hourly(
            frame,
            val_col=concept_name,
            end_col=end_col,
            index_col=index_col,
            id_col=id_col,
            origin_times=origin_times,
            origin_col=origin_col,
            numeric_time_unit=numeric_time_unit,
            output_time_unit=output_time_unit,
        )
        if index_col != "charttime" and index_col in result.columns:
            result = result.rename(columns={index_col: "charttime"})
        return result

    if expr.strip() == "mimv_rate":
        from ..utils.callback_utils import mimv_rate
        duration_col = None
        start_col = source.index_var
        if not start_col and "starttime" in frame.columns:
            start_col = "starttime"

        end_col = None
        if "endtime" in frame.columns:
            end_col = "endtime"
        elif source.dur_var and source.dur_var in frame.columns:
            end_col = source.dur_var

        # 首先检查是否已经有计算好的 duration 列
        possible_dur_cols = [concept_name + '_dur', 'duration', '__duration__', 'dur_var']
        for col in possible_dur_cols:
            if col in frame.columns:
                duration_col = col
                break

        if duration_col is None and end_col and end_col in frame.columns:
            duration_col = end_col
        
        if not duration_col or duration_col not in frame.columns:
            return frame
        # 🔧 FIX: amount_col 应优先使用 'amount' 列（inputevents 表的默认列）
        # R ricu mimv_rate 使用 amount 列来计算 rate = amount / duration
        # concept_name (如 'dex') 在回调执行时还不存在
        amount_col = None
        if source.params:
            alt_amount = source.params.get("amount_var")
            if alt_amount and alt_amount in frame.columns:
                amount_col = alt_amount
        if not amount_col:
            # 优先使用 'amount' 列（inputevents 表的标准列名）
            if 'amount' in frame.columns:
                amount_col = 'amount'
            elif concept_name in frame.columns:
                amount_col = concept_name
        if not amount_col or amount_col not in frame.columns:
            return frame
        unit_col = unit_column or source.unit_var
        if not unit_col:
            if "rateuom" in frame.columns:
                unit_col = "rateuom"
            elif "valueuom" in frame.columns:
                unit_col = "valueuom"
        auom_col = None
        if source.params:
            auom_col = source.params.get("auom_var")
        if not auom_col or auom_col not in frame.columns:
            if "amountuom" in frame.columns:
                auom_col = "amountuom"
            else:
                auom_col = unit_col
        
        # 🔧 FIX: mimv_rate 应使用表的默认 rate 列，而不是 concept_name
        # R ricu 中 mimv_rate 使用 val_var='rate' (来自 inputevents 表配置)
        # mimv_rate 计算 rate = amount / duration，结果写入 rate 列
        rate_col = 'rate' if 'rate' in frame.columns else concept_name
        
        return mimv_rate(
            frame,
            val_col=rate_col,
            unit_col=unit_col,
            dur_var=duration_col,
            amount_var=amount_col,
            auom_var=auom_col,
        )

    match = re.fullmatch(r"dex_to_10\((.+)\)", expr, flags=re.DOTALL)
    if match:
        from ..utils.callback_utils import dex_to_10

        args = _split_arguments(match.group(1))
        if len(args) < 2:
            return frame

        ids = _parse_r_value(args[0])
        factors = _parse_r_value(args[1])
        if not isinstance(ids, list):
            ids = [ids]
        if not isinstance(factors, list):
            factors = [factors]

        callback_fn = dex_to_10(ids, factors)
        sub_var = source.sub_var
        if not sub_var or sub_var not in frame.columns:
            return frame
        return callback_fn(
            frame,
            sub_var=sub_var,
            val_col=concept_name,
        )

    if expr.strip() == "eicu_dex_med":
        from ..utils.callback_utils import eicu_dex_med as eicu_dex_med_cb

        val_var = source.value_var or concept_name
        
        # 优先使用已计算好的duration列 (dur_is_end逻辑产生的 {concept_name}_dur)
        # 这个列包含真正的duration = stopoffset - startoffset
        dur_var = None
        duration_col = concept_name + '_dur'
        if duration_col in frame.columns:
            dur_var = duration_col
        elif "duration" in frame.columns:
            dur_var = "duration"
        else:
            # 回退到原始配置
            if source.params:
                dur_var = source.params.get("dur_var") or source.params.get("stop_var")
            if (not dur_var or dur_var not in frame.columns) and "drugstopoffset" in frame.columns:
                dur_var = "drugstopoffset"
        
        if not dur_var or dur_var not in frame.columns:
            return frame

        return eicu_dex_med_cb(
            frame,
            val_var=val_var,
            dur_var=dur_var,
            concept_name=concept_name,
        )

    if expr.strip() == "eicu_dex_inf":
        from ..utils.callback_utils import eicu_dex_inf as eicu_dex_inf_cb

        val_var = source.value_var or concept_name
        index_var = source.index_var

        return eicu_dex_inf_cb(
            frame,
            val_var=val_var,
            index_var=index_var,
        )

    # blood_cell_ratio callback - convert absolute cell counts to percentage
    # R ricu logic: 100 * value / wbc
    # Used for lymphocytes, neutrophils, etc.
    if expr.strip() == "blood_cell_ratio":
        DEBUG_CALLBACK = False  # Toggle for debugging (set to True for trace)
        if DEBUG_CALLBACK:
            print(f"  [CALLBACK DEBUG] {concept_name} blood_cell_ratio 开始")
            print(f"    frame.shape = {frame.shape}, columns = {list(frame.columns)}")
            if concept_name in frame.columns:
                print(f"    输入值: {frame[concept_name].values}")
        
        if resolver is None:
            if DEBUG_CALLBACK:
                print("    [SKIP] resolver is None")
            # Cannot convert without resolver to load WBC, return as-is
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        # Determine ID column based on database
        # AUMC uses 'admissionid', MIMIC uses 'stay_id', eICU uses 'patientunitstayid'
        # HiRID uses 'patientid', SICdb uses 'CaseID'
        id_col = None
        for possible_id in ['admissionid', 'stay_id', 'patientunitstayid', 'subject_id', 'icustay_id', 'patientid', 'CaseID']:
            if possible_id in frame.columns:
                id_col = possible_id
                break
        
        if id_col is None:
            if DEBUG_CALLBACK:
                print("    [SKIP] id_col is None")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    id_col = {id_col}")
        
        frame_patient_ids = frame[id_col].unique().tolist()
        if len(frame_patient_ids) == 0:
            if DEBUG_CALLBACK:
                print("    [SKIP] no patients")
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame
        
        if DEBUG_CALLBACK:
            print(f"    patients = {frame_patient_ids}")
        
        try:
            # Load WBC concept for the same patients
            # IMPORTANT: Use merge=False to get Dict[str, ICUTable] instead of merged DataFrame
            # IMPORTANT: Must pass data_source for resolver.load_concepts to work
            # Cache WBC across blood_cell_ratio concepts for performance
            if data_source is None:
                if DEBUG_CALLBACK:
                    print("    [SKIP] data_source is None")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            if DEBUG_CALLBACK:
                print("    加载 WBC (使用缓存)...")
            
            # Use full patient_ids for cache efficiency when available
            _wbc_pids = patient_ids if patient_ids else frame_patient_ids
            wbc_result = resolver.load_concepts(
                ['wbc'],
                data_source,
                patient_ids=_wbc_pids,
                r_compatible=False,
                merge=False,
            )
            
            if 'wbc' not in wbc_result or wbc_result['wbc'].data.empty:
                if DEBUG_CALLBACK:
                    print("    [SKIP] WBC 为空或不存在")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            wbc_df = wbc_result['wbc'].data.copy()
            if DEBUG_CALLBACK:
                print(f"    WBC loaded: {len(wbc_df)} rows, columns = {list(wbc_df.columns)}")
                print(f"    WBC样本:\n{wbc_df.head(10)}")
            
            # Find index column for merging (time column)
            index_col = source.index_var
            if not index_col:
                for possible_idx in ['measuredat', 'charttime', 'starttime', 'labresultoffset']:
                    if possible_idx in frame.columns:
                        index_col = possible_idx
                        break
            
            if DEBUG_CALLBACK:
                print(f"    index_col = {index_col}")
            
            # Prepare WBC for merge - rename value column
            wbc_val_col = wbc_result['wbc'].value_column or 'wbc'
            if DEBUG_CALLBACK:
                print(f"    wbc_val_col = {wbc_val_col}")
            if wbc_val_col != 'wbc' and wbc_val_col in wbc_df.columns:
                wbc_df = wbc_df.rename(columns={wbc_val_col: 'wbc'})
            
            # 🔧 FIX 2026-03-09: Handle time column name mismatch between raw source
            # data and WBC loaded via load_concepts (DuckDB aggregation).
            # e.g. AUMC raw source has 'measuredat' (minutes) but WBC has
            # 'measuredat_minutes' (hourly-binned minutes) from DuckDB aggregation.
            # Rename WBC's time column to match frame's time column for merge_asof.
            wbc_index_col = wbc_result['wbc'].index_column
            if (index_col and wbc_index_col and
                    index_col != wbc_index_col and
                    index_col not in wbc_df.columns and
                    wbc_index_col in wbc_df.columns):
                if DEBUG_CALLBACK:
                    print(f"    [TIME COL FIX] Renaming WBC time col: "
                          f"{wbc_index_col} -> {index_col}")
                wbc_df = wbc_df.rename(columns={wbc_index_col: index_col})
            
            # Ensure ID column exists in WBC data
            if id_col not in wbc_df.columns:
                if DEBUG_CALLBACK:
                    print(f"    [SKIP] id_col {id_col} not in wbc_df")
                if concept_name in frame.columns:
                    frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
                return frame
            
            # Ensure numeric types
            frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            wbc_df['wbc'] = pd.to_numeric(wbc_df['wbc'], errors='coerce')
            
            # Ensure matching dtypes for merge columns (fix int32 vs int64 issue)
            if id_col in frame.columns and id_col in wbc_df.columns:
                wbc_df[id_col] = wbc_df[id_col].astype(frame[id_col].dtype)
            
            # For each row in frame, find the closest WBC measurement
            # This is a time-based merge (asof merge)
            if index_col and index_col in frame.columns and index_col in wbc_df.columns:
                # CRITICAL FIX: For AUMC, frame's measuredat is in MINUTES (raw from datasource),
                # but wbc_df's measuredat is in HOURS (after load_concepts processing).
                # We need to convert frame's time to HOURS before merge.
                frame_time_max = frame[index_col].abs().max()
                wbc_time_max = wbc_df[index_col].abs().max() if not wbc_df.empty else 0
                
                # Create copies to avoid modifying original
                frame_work = frame.copy()
                wbc_work = wbc_df.copy()
                
                # CRITICAL: Filter WBC to frame patients BEFORE time unit detection.
                # Otherwise, long-stay patients not in the frame can push wbc_time_max
                # past the 1000-hour threshold, breaking the minutes-vs-hours heuristic.
                _unique_pids = frame_work[id_col].unique()
                wbc_work = wbc_work[wbc_work[id_col].isin(set(_unique_pids))].copy()
                
                # Recalculate time maxes after filtering
                frame_time_max = frame_work[index_col].abs().max()
                wbc_time_max = wbc_work[index_col].abs().max() if not wbc_work.empty else 0
                
                # Improved time unit detection:
                # 1. Large absolute threshold (>1000) clearly indicates minutes
                # 2. Relative comparison: if frame_time >> wbc_time (e.g., 5x+), convert
                # 3. For AUMC with measuredat, frame comes from raw table (minutes) while
                #    wbc comes from load_concepts (hours)
                need_frame_to_hours = False
                need_wbc_to_hours = False
                
                if frame_time_max > 1000 and wbc_time_max < 1000 and wbc_time_max > 0:
                    # Clear case: frame is in minutes (>1000), wbc is in hours
                    need_frame_to_hours = True
                elif frame_time_max < 1000 and wbc_time_max > 1000:
                    # Opposite: wbc is in minutes, frame is in hours
                    need_wbc_to_hours = True
                elif frame_time_max > 0 and wbc_time_max > 0:
                    # Both are < 1000, but may still have different units
                    # If ratio is significantly different (5x+), assume different units
                    ratio = frame_time_max / wbc_time_max if wbc_time_max > 0 else 0
                    if ratio > 5:
                        # frame is much larger, likely in minutes vs hours
                        need_frame_to_hours = True
                        if DEBUG_CALLBACK:
                            print("    [TIME FIX] 基于比率检测时间单位不匹配:")
                            print(f"      ratio = {ratio:.2f}")
                    elif ratio < 0.2 and ratio > 0:
                        # wbc is much larger
                        need_wbc_to_hours = True
                
                if need_frame_to_hours:
                    if DEBUG_CALLBACK:
                        print("    [TIME FIX] 检测到时间单位不匹配:")
                        print(f"      frame max time: {frame_time_max} (分钟)")
                        print(f"      wbc max time: {wbc_time_max} (小时)")
                        print("      -> 将 frame 时间从分钟转换为小时")
                    frame_work[index_col] = frame_work[index_col] / 60.0
                elif need_wbc_to_hours:
                    if DEBUG_CALLBACK:
                        print("    [TIME FIX] 检测到时间单位不匹配（反向）:")
                        print(f"      frame max time: {frame_time_max}")
                        print(f"      wbc max time: {wbc_time_max}")
                        print("      -> 将 wbc 时间从分钟转换为小时")
                    wbc_work[index_col] = wbc_work[index_col] / 60.0
                
                # Ensure matching dtypes for index column
                wbc_work[index_col] = wbc_work[index_col].astype(frame_work[index_col].dtype)
                
                # CRITICAL: merge_asof requires the 'on' column to be sorted globally.
                # With multiple patients, their time ranges may overlap.
                # Solution: Add per-patient offset to make times globally monotonic,
                # then do a single merge_asof call instead of per-patient loops.
                _max_time = max(
                    frame_work[index_col].abs().max() if len(frame_work) > 0 else 0,
                    wbc_work[index_col].abs().max() if len(wbc_work) > 0 else 0,
                ) + 1000  # generous padding

                # Build offset map: each patient gets a non-overlapping time range
                _pid_offset = {pid: i * _max_time * 2 for i, pid in enumerate(_unique_pids)}

                # Add offset to make global time monotonic
                frame_work['_gtime'] = frame_work[id_col].map(_pid_offset) + frame_work[index_col]
                wbc_work['_gtime'] = wbc_work[id_col].map(_pid_offset) + wbc_work[index_col]

                # Sort by global time for merge_asof
                frame_work = frame_work.sort_values('_gtime')
                wbc_work = wbc_work.sort_values('_gtime')

                try:
                    frame_merged = pd.merge_asof(
                        frame_work,
                        wbc_work[[id_col, '_gtime', 'wbc']],
                        on='_gtime',
                        by=id_col,
                        direction='nearest',
                    )
                except Exception:
                    # Fallback: per-patient merge_asof
                    merged_parts = []
                    for patient_id in _unique_pids:
                        fp = frame_work[frame_work[id_col] == patient_id].sort_values(index_col)
                        wp = wbc_work[wbc_work[id_col] == patient_id].sort_values(index_col)
                        if wp.empty:
                            merged_parts.append(fp)
                        else:
                            try:
                                mp = pd.merge_asof(fp, wp[[id_col, index_col, 'wbc']],
                                                   on=index_col, by=id_col, direction='nearest')
                                merged_parts.append(mp)
                            except Exception:
                                merged_parts.append(fp)
                    frame_merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else frame_work.copy()

                # Clean up temp column
                frame_merged = frame_merged.drop(columns=['_gtime'], errors='ignore')
                
                if DEBUG_CALLBACK:
                    print(f"    Frame before merge:\n{frame_work[[id_col, index_col, concept_name]]}")
                    print(f"    After merge_asof:\n{frame_merged[[id_col, index_col, concept_name] + (['wbc'] if 'wbc' in frame_merged.columns else [])]}")
                
                # Calculate ratio: 100 * value / wbc
                if 'wbc' in frame_merged.columns:
                    valid_mask = (frame_merged['wbc'].notna()) & (frame_merged['wbc'] != 0)
                    if DEBUG_CALLBACK:
                        print(f"    valid_mask: {valid_mask.values}, sum={valid_mask.sum()}")
                    frame_merged.loc[valid_mask, concept_name] = (
                        100 * frame_merged.loc[valid_mask, concept_name] / 
                        frame_merged.loc[valid_mask, 'wbc']
                    )
                    if DEBUG_CALLBACK:
                        print(f"    计算后值: {frame_merged[concept_name].values}")
                    # Set unit to %
                    if unit_column and unit_column in frame_merged.columns:
                        frame_merged.loc[valid_mask, unit_column] = '%'
                    # Drop WBC column
                    frame_merged = frame_merged.drop(columns=['wbc'])
                else:
                    if DEBUG_CALLBACK:
                        print("    [WARNING] 'wbc' not in frame_merged.columns!")
                
                # CRITICAL: Convert time back to original format (minutes) for AUMC
                # The subsequent processing will apply the minutes->hours conversion again
                if need_frame_to_hours:
                    # We converted frame from minutes to hours, now convert back
                    frame_merged[index_col] = frame_merged[index_col] * 60.0
                    if DEBUG_CALLBACK:
                        print("    [TIME RESTORE] 将时间从小时转换回分钟")
                
                if DEBUG_CALLBACK:
                    print(f"    返回 frame_merged, shape={frame_merged.shape}")
                return frame_merged
            else:
                if DEBUG_CALLBACK:
                    print("    [FALLBACK] index_col 不在两个 frame 中, 使用平均 WBC")
                # No index column, use simple merge on ID (average WBC per patient)
                wbc_grouped = wbc_df.groupby(id_col)['wbc'].mean().reset_index()
                frame = frame.merge(wbc_grouped, on=id_col, how='left')
                
                valid_mask = (frame['wbc'].notna()) & (frame['wbc'] != 0)
                frame.loc[valid_mask, concept_name] = (
                    100 * frame.loc[valid_mask, concept_name] / 
                    frame.loc[valid_mask, 'wbc']
                )
                if unit_column and unit_column in frame.columns:
                    frame.loc[valid_mask, unit_column] = '%'
                frame = frame.drop(columns=['wbc'])
                
                return frame
                
        except Exception as e:
            if DEBUG_CALLBACK:
                print(f"    [EXCEPTION] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            # On error, return frame as-is with numeric conversion
            if concept_name in frame.columns:
                frame[concept_name] = pd.to_numeric(frame[concept_name], errors='coerce')
            return frame

    raise NotImplementedError(
        f"Callback '{callback}' is not yet supported."
    )

__all__ = ["_apply_callback", "_normalize_eicu_tidal_volume_frame"]
