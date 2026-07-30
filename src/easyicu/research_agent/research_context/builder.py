"""Build a :class:`ResearchContext` from an EasyICU cohort dataframe.

The builder is the bridge between the *data* world (a parquet file
or DataFrame produced by ``easyicu.load_concepts`` / ``filter_patients``)
and the *agent* world (a structured ``ResearchContext`` that drives
prompts and validators).

The builder is intentionally tolerant: it works on a plain DataFrame
even if EasyICU is not installed, falling back to dtype-only
classification. When EasyICU *is* installed, it enriches each
column with description, source databases and category from the
concept dictionary.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..icu_rules import (
    ICU_RULES,
    VariableKind,
    aggregation_rule_for,
    classify_variable,
    default_time_windows,
)
from ..schema import (
    AggregationRule,
    CohortDescriptor,
    ConceptDescriptor,
    EndpointSpec,
    MissingnessProfile,
    ResearchContext,
    TimeWindow,
    UserPreferences,
    VariableRole,
)
from ..intake.materialized_metadata import (
    MaterializedMetadataError,
    VerifiedMaterializedCohortAuthority,
    load_verified_materialized_cohort_authority,
    read_verified_materialized_cohort_table,
)
from ..intake.materialized_trajectory import (
    MaterializedTrajectoryError,
    StagedTrajectoryBinding,
    VerifiedMaterializedTrajectoryAuthority,
    load_verified_materialized_trajectory_authority,
)
from ..cohort.artifact_facts import observed_domain_for_series
from ..intake.legacy_materialization import (
    load_verified_legacy_materialization_provenance,
)
from .typed import (
    canonical_column_binding,
    ResearchContextAuthority,
    ResearchContextV2,
    descriptor_physical_updates,
    materialized_research_inputs_from_authority,
    project_research_context_variables,
)
from ..concept_availability import normalize_database_name
from .cohort_granularity import resolve_cohort_granularity
from .observation_semantics import compile_observation_semantics
from .representation_semantics import compile_wide_representation_semantics
from .temporal_semantics import (
    ConceptValidationLayer,
    ICUEpisodeResolver,
    TemporalAlignmentEngine,
)
from ..trajectory.contract import infer_fixed_window_trajectory_metadata

# Archived callers and the public test surface import this historical private
# name.  Keep it as an object-identity alias while the canonical implementation
# lives in the interpretation-free cohort-artifact leaf.
_observed_domain = observed_domain_for_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_get_concept_info(name: str) -> Optional[Dict[str, Any]]:
    """Best-effort fetch of EasyICU concept metadata. Returns None if unavailable."""
    try:
        from easyicu import get_concept_info  # type: ignore
    except Exception:
        return None
    try:
        return get_concept_info(name)
    except Exception:
        return None


_WIDE_COMPANION_SUFFIXES: tuple[str, ...] = (
    "_first_time",
    "_last_time",
    "_measured",
    "_median",
    "_first",
    "_last",
    "_mean",
    "_max",
    "_min",
    "_sum",
    "_n",
)


def _concept_info_for_wide_column(
    name: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve exact or mechanically derived wide columns to a base concept.

    Resolution is metadata-only and conservative: the exact column is tried
    first; one known EasyICU wide-export companion suffix is stripped only when
    exact lookup fails.  No fuzzy token or clinical-name matching occurs.
    """

    exact = _safe_get_concept_info(name)
    if exact is not None:
        return exact, name
    lowered = str(name or "").strip().lower()
    for suffix in _WIDE_COMPANION_SUFFIXES:
        if not lowered.endswith(suffix) or len(lowered) <= len(suffix):
            continue
        base = lowered[: -len(suffix)]
        info = _safe_get_concept_info(base)
        if info is not None:
            return info, base
        # Composite loaders expose canonical output names that intentionally
        # differ from their source concept id.  Keep that ownership in the
        # upstream concept catalog and project it into ResearchContext instead
        # of duplicating output aliases in the research-agent engine.
        try:
            from easyicu.concept_output_sources import (
                COMPOSITE_CONCEPT_OUTPUT_SOURCES,
            )
        except Exception:
            composite_source = None
        else:
            composite_source = COMPOSITE_CONCEPT_OUTPUT_SOURCES.get(base)
        if composite_source:
            info = _safe_get_concept_info(str(composite_source))
            if info is not None:
                return info, str(composite_source)
        break
    return None, None


def _missingness_severity(fraction: float) -> str:
    """Operational severity label for missingness.

    Unlike MCAR/MAR/MNAR, this label intentionally avoids claiming a
    mechanism from the fraction missing alone. It only communicates
    how disruptive the observed missingness burden is likely to be.
    """
    if fraction == 0.0:
        return "low"
    if fraction < 0.05:
        return "low"
    if fraction < 0.30:
        return "medium"
    return "high"


def _profile_missingness(series: pd.Series) -> MissingnessProfile:
    n_total = int(len(series))
    n_missing = int(series.isna().sum())
    fraction = (n_missing / n_total) if n_total > 0 else 0.0
    return MissingnessProfile(
        fraction_missing=fraction,
        n_missing=n_missing,
        n_total=n_total,
        missingness_severity=_missingness_severity(fraction),  # type: ignore[arg-type]
        missingness_test="not_run",
    )


def _compute_missingness_test_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """Best-effort global Little's MCAR test over a small numeric panel."""
    numeric = df.select_dtypes(include=["number", "bool"]).replace(
        [np.inf, -np.inf], np.nan
    )
    if numeric.empty:
        return {"name": "not_run", "p_value": None, "note": "no_numeric_variables"}
    numeric = numeric.loc[:, numeric.isna().any()]
    if numeric.shape[1] < 2:
        return {
            "name": "not_run",
            "p_value": None,
            "note": "fewer_than_two_incomplete_numeric_variables",
        }
    cols = [
        col
        for col in numeric.columns
        if numeric[col].notna().sum() >= max(10, int(len(numeric) * 0.2))
    ]
    if len(cols) < 2:
        return {
            "name": "not_run",
            "p_value": None,
            "note": "insufficient_complete_support",
        }
    panel = numeric[cols[: min(len(cols), 8)]]
    # Little's test is invariant to an affine rescaling of each variable.
    # Standardising here prevents a nearly singular or mixed-scale ICU panel
    # from making the EM covariance iteration overflow.  Columns with no
    # observed variance carry no information for this screen and are omitted.
    means = panel.mean(skipna=True)
    scales = panel.std(skipna=True, ddof=0)
    stable_columns = [
        column
        for column in panel.columns
        if np.isfinite(means[column])
        and np.isfinite(scales[column])
        and float(scales[column]) > np.finfo(float).eps
    ]
    panel = (panel[stable_columns] - means[stable_columns]) / scales[stable_columns]
    if panel.shape[1] < 2:
        return {
            "name": "not_run",
            "p_value": None,
            "note": "fewer_than_two_nonconstant_incomplete_numeric_variables",
        }
    complete = panel.dropna()
    if len(complete) < max(10, panel.shape[1] + 2):
        return {
            "name": "not_run",
            "p_value": None,
            "note": "too_few_complete_cases_for_mcar_screen",
        }
    try:
        from scipy.stats import chi2  # type: ignore
    except Exception:
        return {"name": "not_run", "p_value": None, "note": "scipy_unavailable"}

    try:
        mu, cov = _estimate_mvn_with_em(panel.to_numpy(dtype=float))
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return {
            "name": "not_run",
            "p_value": None,
            "note": "numerically_unstable_mcar_screen",
        }
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        return {
            "name": "not_run",
            "p_value": None,
            "note": "nonfinite_mcar_estimate",
        }

    pattern_df = panel.isna().astype(int)
    patterns = pattern_df.astype(str).agg("".join, axis=1)
    stat = 0.0
    dof = 0
    for pattern, idx in patterns.groupby(patterns).groups.items():
        observed = [i for i, marker in enumerate(pattern) if marker == "0"]
        if not observed:
            continue
        sub = panel.iloc[list(idx), observed].dropna()
        if sub.empty:
            continue
        mu_obs = mu[observed]
        cov_obs = np.asarray(cov)[np.ix_(observed, observed)]
        try:
            inv = np.linalg.pinv(cov_obs)
        except Exception:
            continue
        diff = sub.mean().to_numpy(dtype=float) - mu_obs
        stat += float(len(sub) * (diff.T @ inv @ diff))
        dof += len(observed)
    dof = max(dof - panel.shape[1], 1)
    p_value = float(chi2.sf(stat, dof))
    return {
        "name": "little_mcar_em",
        "p_value": p_value,
        "note": f"panel={panel.shape[1]}vars/{len(panel)}rows complete_cases={len(complete)}",
    }


def _estimate_mvn_with_em(
    data: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate MVN mean/covariance under missingness via a simple EM loop."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("data must be a 2D array")
    n, p = x.shape
    mu = np.nanmean(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    filled = np.where(np.isnan(x), mu, x)
    cov = np.cov(filled, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(p) * 1e-6

    # Missingness does not change across EM iterations. Grouping rows once lets
    # each iteration evaluate the same conditional-moment formulas in matrix
    # form, rather than allocating one expectation and p×p outer product per
    # ICU stay. With at most eight MCAR-screen variables there are at most 256
    # groups, and real source-status exports typically have only a handful.
    missing_patterns, pattern_index = np.unique(
        np.isnan(x), axis=0, return_inverse=True
    )
    pattern_groups = [
        (
            np.flatnonzero(~missing),
            np.flatnonzero(missing),
            x[pattern_index == index],
        )
        for index, missing in enumerate(missing_patterns)
    ]

    for _ in range(max_iter):
        expected_sum = np.zeros(p, dtype=float)
        second_sum = np.zeros((p, p), dtype=float)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for obs, mis, rows in pattern_groups:
                group_n = len(rows)
                if len(mis) == 0:
                    expected = rows.astype(float, copy=False)
                    expected_sum += expected.sum(axis=0)
                    second_sum += expected.T @ expected
                elif len(obs) == 0:
                    expected_sum += group_n * mu
                    second_sum += group_n * (cov + np.outer(mu, mu))
                else:
                    sigma_oo = cov[np.ix_(obs, obs)] + np.eye(len(obs)) * 1e-8
                    sigma_mo = cov[np.ix_(mis, obs)]
                    sigma_om = cov[np.ix_(obs, mis)]
                    sigma_mm = cov[np.ix_(mis, mis)]
                    inv_oo = np.linalg.pinv(sigma_oo)
                    conditional_weights = sigma_mo @ inv_oo
                    cond_cov = sigma_mm - conditional_weights @ sigma_om

                    expected = rows.copy().astype(float)
                    expected[:, mis] = (
                        mu[mis] + (rows[:, obs] - mu[obs]) @ conditional_weights.T
                    )
                    expected_sum += expected.sum(axis=0)
                    second_sum += expected.T @ expected
                    second_sum[np.ix_(mis, mis)] += group_n * cond_cov

            mu_new = expected_sum / n
            cov_new = second_sum / n - np.outer(mu_new, mu_new)
        cov_new = (cov_new + cov_new.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(cov_new)
        covariance_floor = 1e-8
        cov_new = (eigenvectors * np.maximum(eigenvalues, covariance_floor)) @ (
            eigenvectors.T
        )
        cov_new += np.eye(p) * 1e-6

        if np.max(np.abs(mu_new - mu)) < tol and np.max(np.abs(cov_new - cov)) < tol:
            mu, cov = mu_new, cov_new
            break
        mu, cov = mu_new, cov_new
    return mu, cov


def _allowed_aggregations(
    role: VariableRole, kind: VariableKind
) -> List[AggregationRule]:
    return aggregation_rule_for(role, kind)


def _apply_materialized_column_metadata(
    *,
    descriptors: Sequence[ConceptDescriptor],
    verified: VerifiedMaterializedCohortAuthority,
) -> List[ConceptDescriptor]:
    """Overlay only verified physical facts onto inferred descriptors.

    Physical column roles are deliberately not converted into analysis roles:
    a source ``value`` can still be a covariate, exposure, outcome, or audit
    input depending on the study.  The host owns lineage/unit/range facts;
    Planner/Coder retain those scientific assignments.
    """

    file_binding = verified.sidecar.files[0]
    derivations = {
        item.output_column: item for item in verified.authority.output_derivations
    }
    projected: List[ConceptDescriptor] = []
    for descriptor in descriptors:
        binding = file_binding.columns.get(descriptor.name)
        if binding is None:
            projected.append(descriptor)
            continue
        derivation = derivations.get(descriptor.name)
        source_files = sorted(
            {source.file for source in derivation.sources}
            if derivation is not None
            else set()
        )
        canonical = canonical_column_binding(descriptor.name, binding)
        # ConceptDescriptor only supports a closed two-sided interval.  When
        # the typed sidecar publishes a one-sided range, leave this legacy view
        # empty instead of displaying a conflicting ICU fallback; V2 exposes
        # the exact one-sided authority in materialized_inputs.
        projected.append(
            descriptor.model_copy(
                update={
                    **descriptor_physical_updates(canonical),
                    # A physical allowed-value set does not establish an
                    # ordered scientific scale. Preserve the descriptor's
                    # independently inferred semantics; the V2 typed facts
                    # expose allowed values without promoting nominal data.
                    "is_ordinal": descriptor.is_ordinal,
                    "ordinal_levels": descriptor.ordinal_levels,
                    "source_files": source_files,
                }
            )
        )
    return projected


def _apply_legacy_materialization_window(
    *,
    descriptors: Sequence[ConceptDescriptor],
    provenance: Dict[str, Any],
) -> List[ConceptDescriptor]:
    """Project one verified legacy materialization window onto wide outputs.

    The host-owned receipt says which concepts were summarized and over which
    ICU-admission-relative window.  Only mechanically named wide summary and
    companion columns inherit it; bare/static/outcome columns do not.  Existing
    concept-catalog or typed-authority windows always take precedence.
    """

    window = provenance["cohort_window_hours"]
    start, end = float(window[0]), float(window[1])
    window_label = f"icu_admission[{start:g},{end:g}]h"
    feature_concepts = {
        str(value).strip().lower() for value in provenance["feature_concepts"]
    }
    projected: List[ConceptDescriptor] = []
    for descriptor in descriptors:
        if descriptor.analysis_window:
            projected.append(descriptor)
            continue
        lowered = descriptor.name.strip().lower()
        base: Optional[str] = None
        for suffix in _WIDE_COMPANION_SUFFIXES:
            if lowered.endswith(suffix) and len(lowered) > len(suffix):
                base = lowered[: -len(suffix)]
                break
        if base is None or base not in feature_concepts:
            projected.append(descriptor)
            continue
        projected.append(
            descriptor.model_copy(update={"analysis_window": window_label})
        )
    return projected


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_research_context(
    *,
    research_question: str,
    cohort: Union[pd.DataFrame, str, Path],
    cohort_name: str,
    database: str,
    inclusion_criteria: Optional[Sequence[str]] = None,
    exclusion_criteria: Optional[Sequence[str]] = None,
    target_outcome: Optional[str] = None,
    endpoint: Optional[EndpointSpec] = None,
    primary_exposure: Optional[str] = None,
    cross_database_validation: Optional[Sequence[str]] = None,
    id_columns: Optional[Sequence[str]] = None,
    time_columns: Optional[Sequence[str]] = None,
    outcome_columns: Optional[Sequence[str]] = None,
    concept_descriptions: Optional[Dict[str, str]] = None,
    time_windows: Optional[Sequence[TimeWindow]] = None,
    user_preferences: Optional[Union[UserPreferences, Dict[str, Any]]] = None,
    notes: Optional[str] = None,
    trajectory_binding: Optional[StagedTrajectoryBinding] = None,
) -> ResearchContextAuthority:
    """Build a :class:`ResearchContext` from a cohort dataframe.

    Parameters
    ----------
    research_question
        Plain-language research question; becomes the spine of the
        agent prompts and the manuscript scaffold.
    cohort
        DataFrame or path to a parquet file containing one row per
        analysis unit (typically per ICU stay) with all variables
        already aggregated to that unit. The cohort is *not*
        re-aggregated by the pipeline.
    cohort_name, database
        Human-friendly tags. ``database`` should match an EasyICU
        source key (``miiv``, ``eicu``, ``aumc``, ``hirid``, ``sic``)
        when applicable so the cross-database validator can compare.
    inclusion_criteria, exclusion_criteria
        Free-text criteria that already shaped the cohort. The
        ``CohortAuditor`` checks these against generated code's
        further filters.
    target_outcome
        Name of the primary outcome column. Used by validators to
        confirm the analysis actually predicts this column and not a
        proxy.
    cross_database_validation
        Other databases to replicate this analysis on. The pipeline
        records these targets in every single-database run. The
        higher-level ``ResearchAgentPipeline.replicate(...)`` helper
        can execute the same question across multiple cohorts and
        summarise which databases were promised vs. actually run.
    """
    # --- normalise cohort input
    verified_cohort: Optional[VerifiedMaterializedCohortAuthority] = None
    verified_trajectory: Optional[VerifiedMaterializedTrajectoryAuthority] = None
    legacy_materialization_provenance: Optional[Dict[str, Any]] = None
    if isinstance(cohort, (str, Path)):
        cohort_path_obj = Path(cohort).resolve()
        cohort_path = str(cohort_path_obj)
        verified_cohort = load_verified_materialized_cohort_authority(cohort_path_obj)
        if verified_cohort is not None:
            normalized_database = normalize_database_name(database)
            if verified_cohort.sidecar.source_database != normalized_database:
                raise MaterializedMetadataError(
                    "ResearchContext database does not match materialized authority"
                )
            database = normalized_database
            if trajectory_binding is not None and trajectory_binding.authority_ref:
                verified_trajectory = load_verified_materialized_trajectory_authority(
                    trajectory_binding.path,
                    expected_authority=trajectory_binding.authority_ref,
                    expected_universe_authority=verified_cohort.reference,
                )
                if verified_trajectory is None:  # pragma: no cover - expected ref
                    raise MaterializedTrajectoryError(
                        "typed trajectory authority is missing"
                    )
        elif trajectory_binding is not None and trajectory_binding.authority_ref:
            raise MaterializedTrajectoryError(
                "typed trajectory cannot bind an untyped ResearchContext cohort"
            )
        if verified_cohort is not None:
            df = read_verified_materialized_cohort_table(
                cohort_path_obj,
                verified=verified_cohort,
            ).to_pandas()
        else:
            df = pd.read_parquet(cohort_path)
            legacy_materialization_provenance = (
                load_verified_legacy_materialization_provenance(
                    cohort_path_obj,
                    cohort=df,
                )
            )
    else:
        cohort_path = None
        df = cohort

    if not isinstance(df, pd.DataFrame):
        raise TypeError("cohort must be a pandas DataFrame or a parquet path")

    # --- cohort descriptor
    id_columns = list(id_columns) if id_columns else _guess_id_columns(df)
    time_columns = list(time_columns) if time_columns else _guess_time_columns(df)
    outcome_columns = (
        list(outcome_columns) if outcome_columns else _guess_outcome_columns(df)
    )
    cohort_path_str = (
        str(Path(cohort).resolve()) if isinstance(cohort, (str, Path)) else None
    )
    episode = ICUEpisodeResolver().resolve(
        df=df,
        database=database,
        id_columns=id_columns,
        time_columns=time_columns,
        outcome_columns=outcome_columns,
        target_outcome=target_outcome,
        cohort_path=cohort_path_str,
    )

    granularity = resolve_cohort_granularity(
        frame=df,
        id_columns=episode.id_columns,
    )
    n_stays = int(len(df))

    cohort_desc = CohortDescriptor(
        cohort_name=cohort_name,
        database=database,
        n_patients=granularity.n_patients,
        n_stays=n_stays,
        inclusion_criteria=list(inclusion_criteria or []),
        exclusion_criteria=list(exclusion_criteria or []),
        id_columns=episode.id_columns,
        time_columns=episode.time_columns,
        outcome_columns=episode.outcome_columns,
        provenance={
            **episode.provenance,
            **granularity.provenance(),
            "inclusion_criteria": list(inclusion_criteria or []),
            "exclusion_criteria": list(exclusion_criteria or []),
            **(
                {
                    "materialized_cohort_window_hours": list(
                        legacy_materialization_provenance["cohort_window_hours"]
                    ),
                    "materialized_cohort_provenance_sha256": (
                        legacy_materialization_provenance["provenance_sha256"]
                    ),
                }
                if legacy_materialization_provenance is not None
                else {}
            ),
        },
    )

    # --- per-column descriptors
    descriptors: List[ConceptDescriptor] = []
    user_descriptions = dict(concept_descriptions or {})
    missingness_test_meta = _compute_missingness_test_metadata(df)
    for col in df.columns:
        descriptors.append(
            _describe_column(
                df=df,
                col=col,
                user_descriptions=user_descriptions,
                id_columns=episode.id_columns,
                time_columns=episode.time_columns,
                outcome_columns=episode.outcome_columns,
                missingness_test_meta=missingness_test_meta,
            )
        )
    _enrich_target_outcome_descriptor(
        descriptors=descriptors,
        research_question=research_question,
        target_outcome=target_outcome,
    )
    if verified_cohort is not None:
        descriptors = _apply_materialized_column_metadata(
            descriptors=descriptors,
            verified=verified_cohort,
        )
    elif legacy_materialization_provenance is not None:
        descriptors = _apply_legacy_materialization_window(
            descriptors=descriptors,
            provenance=legacy_materialization_provenance,
        )
    descriptors = compile_wide_representation_semantics(descriptors)
    descriptors = compile_observation_semantics(
        frame=df,
        descriptors=descriptors,
    )

    prefs_obj = (
        user_preferences
        if isinstance(user_preferences, UserPreferences)
        else (
            UserPreferences.model_validate(user_preferences)
            if user_preferences
            else None
        )
    )

    # --- time windows + deterministic temporal semantics
    inferred_windows, temporal_constraints = TemporalAlignmentEngine().infer(
        research_question=research_question,
        timing_and_design=(prefs_obj.timing_and_design if prefs_obj else None),
        explicit_windows=time_windows,
    )
    windows = (
        list(time_windows)
        if time_windows
        else inferred_windows or default_time_windows()
    )

    base_context = ResearchContext(
        research_question=research_question,
        cohort=cohort_desc,
        variables=descriptors,
        time_windows=windows,
        temporal_constraints=temporal_constraints,
        target_outcome=target_outcome,
        # Passed through exactly as declared. This builder knows the column
        # names, the dtypes and the order they arrived in -- which is precisely
        # why it must not consult any of them to fill this in. Every one of the
        # four defects this type exists for came from a layer that had those
        # signals and used them.
        endpoint=endpoint,
        primary_exposure=primary_exposure,
        cross_database_validation=list(cross_database_validation or []),
        cohort_parquet=cohort_path,
        user_preferences=prefs_obj,
        notes=notes,
    )
    if verified_cohort is None:
        return base_context
    return ResearchContextV2.model_validate(
        {
            **base_context.model_dump(mode="python"),
            "schema_version": "easyicu.research_context/2",
            "materialized_inputs": materialized_research_inputs_from_authority(
                cohort=verified_cohort,
                trajectory=verified_trajectory,
            ).model_dump(mode="python"),
        }
    )


# ---------------------------------------------------------------------------
# Column-level reasoning
# ---------------------------------------------------------------------------


def _describe_column(
    *,
    df: pd.DataFrame,
    col: str,
    user_descriptions: Dict[str, str],
    id_columns: Sequence[str],
    time_columns: Sequence[str],
    outcome_columns: Sequence[str],
    missingness_test_meta: Dict[str, Any],
) -> ConceptDescriptor:
    series = df[col]
    sample = series.dropna().head(50).tolist() if len(series) else []
    hint = classify_variable(col, str(series.dtype), sample)

    # role fix-ups: respect user-declared id/time/outcome
    role = hint.role
    if col in id_columns:
        role = VariableRole.ID
    elif col in time_columns:
        role = VariableRole.TIME
    elif col in outcome_columns:
        role = VariableRole.OUTCOME

    # description: prefer user, fall back to EasyICU concept dict
    description = user_descriptions.get(col)
    source_concept = None
    source_databases: List[str] = []
    concept_validation = ConceptValidationLayer()
    source_tables: List[str] = []
    item_ids: List[str] = []
    unit_normalization: Optional[str] = None
    analysis_window: Optional[str] = None
    temporal_resolution: Optional[str] = None
    clinical_caveats: List[str] = []
    missingness_semantics: Optional[str] = None
    info, resolved_concept = _concept_info_for_wide_column(col)
    if info is not None:
        if description is None:
            description = info.get("description") or None
        meta = concept_validation.validate_descriptor_payload(
            source_info=info,
            column_name=col,
        )
        source_concept = str(info.get("name") or resolved_concept or col)
        srcs = info.get("sources") or info.get("source_databases") or []
        if isinstance(srcs, dict):
            source_databases = sorted(map(str, srcs.keys()))
        else:
            source_databases = [str(s) for s in srcs]
        source_tables = meta.get("source_tables") or []
        item_ids = meta.get("item_ids") or []
        unit_normalization = meta.get("unit_normalization")
        raw_analysis_window = meta.get("analysis_window") or info.get("analysis_window")
        analysis_window = (
            str(raw_analysis_window).strip()
            if isinstance(raw_analysis_window, str) and raw_analysis_window.strip()
            else None
        )
        temporal_resolution = meta.get("temporal_resolution")
        clinical_caveats = meta.get("clinical_caveats") or []
        missingness_semantics = meta.get("missingness_semantics")

    allowed = _allowed_aggregations(role, hint.kind)
    miss = _profile_missingness(series)
    if miss.fraction_missing > 0 and missingness_test_meta.get("name") != "not_run":
        miss.missingness_test = str(missingness_test_meta.get("name"))
        miss.missingness_test_p_value = missingness_test_meta.get("p_value")
        note = missingness_test_meta.get("note")
        if note:
            miss.notes = str(note)
    fixed_window_trajectory = infer_fixed_window_trajectory_metadata(
        column_name=col,
        values=series,
        source_scale=hint.kind.value,
    )
    if fixed_window_trajectory is not None and temporal_resolution is None:
        temporal_resolution = (
            f"fixed {fixed_window_trajectory.window_width_hours:g}-hour windows "
            "on a relative time axis"
        )

    return ConceptDescriptor(
        name=col,
        description=description,
        role=role,
        dtype=str(series.dtype),
        unit=hint.unit,
        valid_range=list(hint.valid_range) if hint.valid_range else None,
        observed_domain=observed_domain_for_series(series),
        allowed_aggregations=allowed,
        aggregation_default=hint.aggregation_default,
        is_ordinal=hint.is_ordinal,
        ordinal_levels=list(hint.ordinal_levels) if hint.ordinal_levels else None,
        source_concept=source_concept,
        source_databases=source_databases,
        source_tables=source_tables,
        item_ids=item_ids,
        unit_normalization=unit_normalization,
        analysis_window=analysis_window,
        temporal_resolution=temporal_resolution,
        fixed_window_trajectory=fixed_window_trajectory,
        pitfalls=list(hint.pitfalls),
        clinical_caveats=clinical_caveats or list(hint.pitfalls),
        missingness_semantics=missingness_semantics,
        missingness=miss,
    )


def _guess_id_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        c
        for c in df.columns
        if c.lower()
        in {
            "patient_id",
            "icustay_id",
            "hadm_id",
            "stay_id",
            "subject_id",
            "patientunitstayid",
            "uniquepid",
            "admissionid",
        }
    ]
    return candidates[:3]


def _guess_time_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        s = df[c]
        if "datetime" in str(s.dtype).lower() or "timestamp" in str(s.dtype).lower():
            out.append(c)
        elif c.lower() in {
            "intime",
            "outtime",
            "admittime",
            "dischtime",
            "deathtime",
            "charttime",
        }:
            out.append(c)
    return out


def _guess_outcome_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if cl in {
            "death",
            "death_icu",
            "death_hosp",
            "mortality",
            "los_icu",
            "los_hosp",
            "readmission",
            "readmit_30d",
        }:
            out.append(c)
        elif cl.startswith("outcome_"):
            out.append(c)
    return out


def _infer_outcome_semantics(
    *,
    research_question: str,
    outcome_name: Optional[str],
) -> Dict[str, str]:
    question = (research_question or "").lower()
    outcome = (outcome_name or "").lower()
    if any(
        term in question
        for term in ("survival", "time-to-event", "time to event", "cox", "hazard")
    ) or outcome in {
        "survival_time",
        "time_to_event",
        "event_time",
        "followup_time",
        "follow_up_time",
    }:
        return {
            "label": "time-to-event endpoint",
            "description": (
                "Outcome component for a time-to-event analysis; keep the event "
                "indicator, follow-up time, censoring rule and time zero explicit. "
                "Use a follow-up/event product only when its typed authority and "
                "declared semantics bind both fields to the analysis time origin. "
                "The shared data-foundation layer does not invent a censoring rule "
                "or derive an estimand-specific follow-up column from convenient "
                "source fields."
            ),
            "source_concept": "time_to_event_endpoint",
            "substitution_note": (
                "Do not substitute a binary event-rate, logistic model target, "
                "or unrelated follow-up horizon for this time-to-event endpoint."
            ),
        }
    if "icu mortality" in question or outcome in {
        "death_icu",
        "icu_death",
        "icu_mortality",
    }:
        return {
            "label": "ICU mortality",
            "description": "Binary outcome flag operationalizing ICU mortality for this analysis.",
            "source_concept": "icu_mortality",
            "substitution_note": (
                "Do not silently substitute ICU, hospital, 28-day, or 30-day mortality "
                f"for one another when using '{outcome_name}'."
            ),
        }
    if (
        "in-hospital mortality" in question
        or "hospital mortality" in question
        or outcome in {"death_hosp", "hospital_death", "hospital_mortality"}
    ):
        return {
            "label": "hospital mortality",
            "description": "Binary outcome flag operationalizing hospital mortality for this analysis.",
            "source_concept": "hospital_mortality",
            "substitution_note": (
                "Do not silently substitute ICU, hospital, 28-day, or 30-day mortality "
                f"for one another when using '{outcome_name}'."
            ),
        }
    if "28-day mortality" in question or outcome in {"death_28d", "mortality_28d"}:
        return {
            "label": "28-day mortality",
            "description": "Binary outcome flag operationalizing 28-day mortality for this analysis.",
            "source_concept": "mortality_28d",
            "substitution_note": (
                "Do not silently substitute ICU, hospital, 28-day, or 30-day mortality "
                f"for one another when using '{outcome_name}'."
            ),
        }
    if "30-day mortality" in question or outcome in {"death_30d", "mortality_30d"}:
        return {
            "label": "30-day mortality",
            "description": "Binary outcome flag operationalizing 30-day mortality for this analysis.",
            "source_concept": "mortality_30d",
            "substitution_note": (
                "Do not silently substitute ICU, hospital, 28-day, or 30-day mortality "
                f"for one another when using '{outcome_name}'."
            ),
        }
    if outcome in {"death", "mortality"}:
        return {
            "label": "all-cause mortality",
            "description": "Binary mortality outcome flag; confirm whether this refers to ICU, hospital, or fixed-horizon mortality before interpretation.",
            "source_concept": "mortality_unspecified",
            "substitution_note": (
                "Do not silently substitute ICU, hospital, fixed-horizon, or "
                "analysis-specific mortality definitions without an explicit protocol."
            ),
        }
    if any(
        term in question for term in ("length of stay", "los", "duration of stay")
    ) or outcome in {
        "los",
        "los_icu",
        "icu_los",
        "los_hosp",
        "hospital_los",
        "length_of_stay",
    }:
        return {
            "label": "length-of-stay outcome",
            "description": (
                "Continuous or count-like length-of-stay outcome; summarize and model "
                "it with methods appropriate for skewed non-binary endpoints."
            ),
            "source_concept": "length_of_stay",
            "substitution_note": (
                "Do not convert length of stay into a mortality/event-rate endpoint "
                "or silently binarize it without an explicit protocol."
            ),
        }
    if "readmission" in question or outcome in {
        "readmission",
        "readmit_30d",
        "readmission_30d",
    }:
        return {
            "label": "readmission outcome",
            "description": (
                "Readmission endpoint declared for this analysis; keep the horizon "
                "and event definition explicit before interpreting rates or models."
            ),
            "source_concept": "readmission",
            "substitution_note": (
                "Do not substitute mortality, length of stay, or another event "
                "definition for this declared readmission outcome."
            ),
        }
    if outcome:
        return {
            "label": "declared primary outcome",
            "description": (
                "Primary outcome column declared by the caller for this analysis; "
                "use the endpoint definition from the run context or case protocol."
            ),
            "source_concept": "declared_primary_outcome",
            "substitution_note": (
                "Do not replace this declared outcome with another endpoint, time "
                "horizon, or transformed target unless the plan explicitly says so."
            ),
        }
    return {}


def _enrich_target_outcome_descriptor(
    *,
    descriptors: Sequence[ConceptDescriptor],
    research_question: str,
    target_outcome: Optional[str],
) -> None:
    if not target_outcome:
        return
    semantics = _infer_outcome_semantics(
        research_question=research_question,
        outcome_name=target_outcome,
    )
    if not semantics:
        return
    for descriptor in descriptors:
        if descriptor.name != target_outcome:
            continue
        if descriptor.role != VariableRole.OUTCOME:
            descriptor.role = VariableRole.OUTCOME
        if (
            descriptor.description is None
            or semantics["source_concept"] != "declared_primary_outcome"
        ):
            descriptor.description = semantics["description"]
        if (
            descriptor.source_concept is None
            or semantics["source_concept"] != "declared_primary_outcome"
        ):
            descriptor.source_concept = semantics["source_concept"]
        explicit_note = (
            f"For this analysis, '{target_outcome}' is explicitly treated as "
            f"{semantics['label']} because that is what the research question asks for."
        )
        if explicit_note not in descriptor.clinical_caveats:
            descriptor.clinical_caveats.append(explicit_note)
        harmonization_note = semantics.get(
            "substitution_note",
            (
                "Do not replace this declared outcome with another endpoint, time "
                "horizon, or transformed target unless the plan explicitly says so."
            ),
        )
        if harmonization_note not in descriptor.cross_database_notes:
            descriptor.cross_database_notes.append(harmonization_note)
        break


# ---------------------------------------------------------------------------
# Naive context builder (T1.4 — ablation arm)
# ---------------------------------------------------------------------------


def build_naive_research_context(
    *,
    research_question: str,
    cohort: Union[pd.DataFrame, str, Path],
    cohort_name: str,
    database: str,
    inclusion_criteria: Optional[Sequence[str]] = None,
    exclusion_criteria: Optional[Sequence[str]] = None,
    target_outcome: Optional[str] = None,
    primary_exposure: Optional[str] = None,
    cross_database_validation: Optional[Sequence[str]] = None,
    id_columns: Optional[Sequence[str]] = None,
    time_columns: Optional[Sequence[str]] = None,
    outcome_columns: Optional[Sequence[str]] = None,
    concept_descriptions: Optional[Dict[str, str]] = None,
    time_windows: Optional[Sequence[TimeWindow]] = None,
    user_preferences: Optional[Union[UserPreferences, Dict[str, Any]]] = None,
    notes: Optional[str] = None,
) -> ResearchContext:
    """Hero-ablation "naive" builder.

    Emits the *minimum* viable context: every column gets only its
    name, dtype, and a single allowed aggregation ``ANY``. No
    ICU-specific role inference, no pitfalls, no missingness profile
    hints, no ordinal flags. This approximates what a generic
    analysis agent would synthesise from a CSV.

    The returned :class:`ResearchContext` is structurally identical to
    the ICU-aware one — same schema, same fields — so downstream code
    is unchanged. Only the *informational content* of the context is
    stripped, which is the variable T1.4 measures.
    """
    if isinstance(cohort, (str, Path)):
        cohort_path = str(Path(cohort).resolve())
        df = pd.read_parquet(cohort_path)
    else:
        cohort_path = None
        df = cohort
    if not isinstance(df, pd.DataFrame):
        raise TypeError("cohort must be a pandas DataFrame or a parquet path")

    id_cols = list(id_columns) if id_columns else _guess_id_columns(df)
    time_cols = list(time_columns) if time_columns else _guess_time_columns(df)
    out_cols = list(outcome_columns) if outcome_columns else _guess_outcome_columns(df)
    if (
        target_outcome
        and target_outcome in df.columns
        and target_outcome not in out_cols
    ):
        out_cols.append(target_outcome)

    granularity = resolve_cohort_granularity(frame=df, id_columns=id_cols)
    n_stays = int(len(df))
    cohort_desc = CohortDescriptor(
        cohort_name=cohort_name,
        database=database,
        n_patients=granularity.n_patients,
        n_stays=n_stays,
        inclusion_criteria=list(inclusion_criteria or []),
        exclusion_criteria=list(exclusion_criteria or []),
        id_columns=id_cols,
        time_columns=time_cols,
        outcome_columns=out_cols,
        provenance=granularity.provenance(),
    )

    descriptors: List[ConceptDescriptor] = []
    for col in df.columns:
        # role = OTHER for everything except declared id / time / outcome.
        if col in id_cols:
            role = VariableRole.ID
        elif col in time_cols:
            role = VariableRole.TIME
        elif col in out_cols:
            role = VariableRole.OUTCOME
        else:
            role = VariableRole.OTHER
        descriptors.append(
            ConceptDescriptor(
                name=col,
                description=None,
                role=role,
                dtype=str(df[col].dtype),
                unit=None,
                valid_range=None,
                allowed_aggregations=[],
                aggregation_default=None,
                is_ordinal=False,
                ordinal_levels=None,
                source_concept=None,
                source_databases=[],
                pitfalls=[],
                missingness=None,
            )
        )
    _enrich_target_outcome_descriptor(
        descriptors=descriptors,
        research_question=research_question,
        target_outcome=target_outcome,
    )

    # Default windows are also stripped — a naive agent does not know
    # about "first_24h"; keep an empty list unless the caller passed one.
    windows = list(time_windows) if time_windows else []
    return ResearchContext(
        research_question=research_question,
        cohort=cohort_desc,
        variables=descriptors,
        time_windows=windows,
        target_outcome=target_outcome,
        primary_exposure=primary_exposure,
        cross_database_validation=list(cross_database_validation or []),
        cohort_parquet=cohort_path,
        user_preferences=None,
        notes=notes,
    )


def retrieve_context_variables(
    context: ResearchContext,
    *,
    query: str,
    top_k: int = 40,
) -> List[ConceptDescriptor]:
    """Return the most relevant concept descriptors for a question.

    O6 — long-context guard. This is deliberately dependency-free: it
    uses lexical overlap across variable name, description, role and
    pitfall text, with small boosts for target outcomes and explicitly
    question-mentioned variables. If a future install has an embedding
    index, it can replace this scorer without changing the pipeline
    contract.
    """
    if top_k <= 0 or top_k >= len(context.variables):
        return list(context.variables)
    q_tokens = _tokens(query or context.research_question)
    scored: List[Tuple[float, int, ConceptDescriptor]] = []
    for i, v in enumerate(context.variables):
        haystack = " ".join(
            [
                v.name,
                v.description or "",
                v.role.value,
                v.dtype,
                " ".join(v.pitfalls),
                v.missingness_semantics or "",
                " ".join(v.forbidden_transformations),
                " ".join(v.cross_database_notes),
            ]
        )
        v_tokens = _tokens(haystack)
        overlap = len(q_tokens & v_tokens)
        score = float(overlap)
        name_norm = re.sub(r"[^a-z0-9]+", "", v.name.lower())
        q_norm = re.sub(r"[^a-z0-9]+", "", (query or context.research_question).lower())
        if name_norm and name_norm in q_norm:
            score += 4.0
        if context.target_outcome and v.name == context.target_outcome:
            score += 3.0
        if context.primary_exposure and v.name == context.primary_exposure:
            score += 3.0
        if v.role in {
            VariableRole.OUTCOME,
            VariableRole.COMPOSITE_SCORE,
            VariableRole.ORDINAL_SCORE,
        }:
            score += 1.0
        if v.pitfalls:
            score += 0.5
        scored.append((score, -i, v))
    ranked = sorted(scored, key=lambda t: (t[0], t[1]), reverse=True)
    selected = [v for score, _, v in ranked[:top_k] if score > 0]
    if not selected:
        selected = [v for _, _, v in ranked[:top_k]]

    # Always preserve declared id/time/outcome columns even if the
    # natural-language query did not mention them.
    required = set(
        context.cohort.id_columns
        + context.cohort.time_columns
        + context.cohort.outcome_columns
    )
    if context.target_outcome:
        required.add(context.target_outcome)
    if context.primary_exposure:
        required.add(context.primary_exposure)
    by_name = {v.name: v for v in context.variables}
    selected_names = {v.name for v in selected}
    for name in required:
        if name in by_name and name not in selected_names:
            selected.append(by_name[name])
            selected_names.add(name)
    return selected


def build_retrieved_research_context(
    context: ResearchContext,
    *,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
) -> ResearchContext:
    """Return a prompt-sized context with only top-K variables.

    The full :class:`ResearchContext` should still be used by validators
    and manifest writing. This helper is for agent prompts only.
    """
    if top_k is None or top_k <= 0 or top_k >= len(context.variables):
        return context
    selected = retrieve_context_variables(
        context,
        query=query or context.research_question,
        top_k=top_k,
    )
    selected_names = ", ".join(v.name for v in selected)
    retrieval_note = (
        f"Context retrieval active: showing {len(selected)}/"
        f"{len(context.variables)} variables selected for this question. "
        f"Selected variables: {selected_names}."
    )
    notes = f"{context.notes}\n\n{retrieval_note}" if context.notes else retrieval_note
    projected = project_research_context_variables(context, selected)
    return projected.model_copy(update={"notes": notes})


def _tokens(text: str) -> set:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) >= 2}


__all__ = [
    "build_research_context",
    "build_naive_research_context",
    "retrieve_context_variables",
    "build_retrieved_research_context",
]
