"""Case-specific EasyICU research-context packages.

The generic :mod:`easyicu.research_agent.research_context.builder` builder can infer a
lot from a cohort dataframe. Manuscript-grade agent work needs one more
layer: a case-level contract that says exactly which EasyICU concepts
produced each derived variable, what operations are unsafe, and what
cross-database caveats must remain visible to the LLM and validators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd

from .research_context.builder import build_naive_research_context, build_research_context
from .schema import AggregationRule, ConceptDescriptor, ResearchContext, VariableRole


PathLike = Union[str, Path]

LACTATE_MAP_VASO_QUESTION = (
    "Among ICU stays, does early lactate identify hospital mortality risk "
    "beyond apparent MAP adequacy and vasopressor exposure?"
)


def _meta(
    *,
    role: VariableRole,
    derived_from: Sequence[str],
    source_key: Optional[str] = None,
    analysis_window: Optional[str] = "first_24h",
    missingness_semantics: Optional[str] = None,
    forbidden_transformations: Optional[Sequence[str]] = None,
    cross_database_notes: Optional[Sequence[str]] = None,
    pitfalls: Optional[Sequence[str]] = None,
    aggregation_default: Optional[AggregationRule] = None,
) -> Dict[str, object]:
    return {
        "role": role,
        "derived_from": list(derived_from),
        "source_key": source_key,
        "analysis_window": analysis_window,
        "missingness_semantics": missingness_semantics,
        "forbidden_transformations": list(forbidden_transformations or []),
        "cross_database_notes": list(cross_database_notes or []),
        "pitfalls": list(pitfalls or []),
        "aggregation_default": aggregation_default,
    }


LACTATE_MAP_VASO_VARIABLE_METADATA: Mapping[str, Mapping[str, object]] = {
    "death": _meta(
        role=VariableRole.OUTCOME,
        derived_from=("death",),
        source_key="death",
        analysis_window=None,
        missingness_semantics="Death is a complete outcome flag in the EasyICU outcome concept export.",
        forbidden_transformations=("Do not substitute ICU mortality, 28-day mortality, or LOS for death.",),
        cross_database_notes=("Confirm the target remains hospital mortality in every database.",),
    ),
    "lactate_max_24h": _meta(
        role=VariableRole.LAB,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics=(
            "Lactate measurement is clinically triggered in ICU data; unmeasured lactate "
            "must be modelled with an explicit measurement indicator."
        ),
        forbidden_transformations=(
            "Do not fill missing lactate with 0 or interpret missing lactate as normal.",
            "Do not use mean lactate as the primary summary for the right-skewed distribution.",
        ),
        cross_database_notes=(
            "Lactate measurement frequency differs sharply across MIMIC-IV, eICU and HiRID.",
            "Use the same first-24h window and preserve the lactate_measured_24h flag.",
        ),
        pitfalls=(
            "High lactate missingness should trigger a missingness audit before modelling.",
            "A lactate effect estimate without a measurement-status term is not review-ready.",
        ),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "lactate_median_24h": _meta(
        role=VariableRole.LAB,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics="Same measurement-triggered semantics as lactate_max_24h.",
        forbidden_transformations=("Do not treat missing lactate as 0.",),
        cross_database_notes=("Keep the first-24h aggregation window identical across databases.",),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "lactate_first_24h": _meta(
        role=VariableRole.LAB,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics="First measured lactate exists only when lactate was ordered.",
        forbidden_transformations=("Do not compare first lactate without reporting measurement rate.",),
        cross_database_notes=("Timestamp resolution and ordering can differ by source database.",),
        aggregation_default=AggregationRule.FIRST_VALUE,
    ),
    "lactate_measured_24h": _meta(
        role=VariableRole.META,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics="Primary indicator for lactate ascertainment; preserve in every model.",
        forbidden_transformations=("Do not drop this flag when lactate is a predictor.",),
        cross_database_notes=("Measurement rate is itself a replication endpoint.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "hyperlactatemia_24h": _meta(
        role=VariableRole.LAB,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics="Undefined when lactate is unmeasured; use only among measured stays or with an explicit missing flag.",
        forbidden_transformations=("Do not coerce unmeasured lactate to the negative class.",),
        cross_database_notes=("The >2 mmol/L threshold must remain unchanged across databases.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "lactate_gt4_24h": _meta(
        role=VariableRole.LAB,
        derived_from=("lact",),
        source_key="lactate_max_24h",
        missingness_semantics="Undefined when lactate is unmeasured; preserve the measurement flag.",
        forbidden_transformations=("Do not coerce unmeasured lactate to the negative class.",),
        cross_database_notes=("The >4 mmol/L threshold must remain unchanged across databases.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "map_min_24h": _meta(
        role=VariableRole.VITAL,
        derived_from=("map",),
        source_key="map_min_24h",
        missingness_semantics="MAP is usually densely observed; missing MAP should trigger source extraction review.",
        forbidden_transformations=("Do not replace minimum MAP with mean MAP for the shock-discordance stratum.",),
        cross_database_notes=("MAP derivation may differ between invasive, non-invasive and charted sources.",),
        pitfalls=("MAP >=65 is apparent adequacy, not proof of adequate tissue perfusion.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "map_median_24h": _meta(
        role=VariableRole.VITAL,
        derived_from=("map",),
        source_key="map_min_24h",
        missingness_semantics="Use as a supportive summary; the primary stratum uses minimum MAP.",
        forbidden_transformations=("Do not substitute median MAP for minimum MAP in the pre-specified stratum.",),
        cross_database_notes=("MAP observation density can differ across sources.",),
        aggregation_default=AggregationRule.MEDIAN_ONLY,
    ),
    "vaso_any_24h": _meta(
        role=VariableRole.INTERVENTION,
        derived_from=("vaso_ind",),
        source_key="vaso_any_24h",
        missingness_semantics=(
            "A missing vasopressor concept export should be reported as unavailable, not treated as no exposure."
        ),
        forbidden_transformations=(
            "Do not use vasopressor exposure as a causal treatment effect without a causal design.",
            "Do not silently convert unavailable vasopressor data to no vasopressor exposure.",
        ),
        cross_database_notes=(
            "Medication tables and infusion encodings differ substantially across ICU databases.",
            "An empty vasopressor-positive stratum is a replication caveat, not a negative result.",
        ),
        pitfalls=("Vasopressor exposure is confounded by indication.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
    "norepi_equiv_max_24h": _meta(
        role=VariableRole.INTERVENTION,
        derived_from=("norepi_equiv",),
        source_key="vaso_any_24h",
        missingness_semantics="Norepinephrine-equivalent dose can be unavailable even when binary vaso exposure exists.",
        forbidden_transformations=("Do not compare doses across databases until units and mappings are harmonised.",),
        cross_database_notes=("Dose harmonisation is database-specific and should remain validator-visible.",),
        aggregation_default=AggregationRule.MAX_LAST,
    ),
}


def _load_json(value: Union[Mapping[str, object], PathLike]) -> Dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _load_cohort(value: Union[pd.DataFrame, PathLike]) -> tuple[pd.DataFrame, Optional[str]]:
    if isinstance(value, pd.DataFrame):
        return value, None
    path = Path(value)
    return pd.read_parquet(path), str(path.resolve())


def _unique(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def _update_descriptor(
    descriptor: ConceptDescriptor,
    *,
    metadata: Mapping[str, object],
    concept_sources: Mapping[str, object],
    database: str,
) -> ConceptDescriptor:
    derived = [str(v) for v in metadata.get("derived_from", [])]
    source_key = str(metadata.get("source_key") or descriptor.name)
    source_file = concept_sources.get(source_key)
    source_files = [str(source_file)] if source_file else []
    pitfalls = _unique([*descriptor.pitfalls, *[str(v) for v in metadata.get("pitfalls", [])]])
    cross_database_notes = _unique([str(v) for v in metadata.get("cross_database_notes", [])])
    forbidden = _unique([str(v) for v in metadata.get("forbidden_transformations", [])])
    aggregation_default = metadata.get("aggregation_default") or descriptor.aggregation_default

    return descriptor.model_copy(update={
        "role": metadata.get("role") or descriptor.role,
        "source_concept": derived[0] if derived else descriptor.source_concept,
        "derived_from_concepts": derived,
        "source_files": source_files,
        "analysis_window": metadata.get("analysis_window"),
        "source_databases": _unique([*descriptor.source_databases, database]),
        "pitfalls": pitfalls,
        "missingness_semantics": metadata.get("missingness_semantics"),
        "forbidden_transformations": forbidden,
        "cross_database_notes": cross_database_notes,
        "aggregation_default": aggregation_default,
    })


def build_lactate_map_vaso_research_context(
    *,
    cohort: Union[pd.DataFrame, PathLike],
    source_manifest: Union[Mapping[str, object], PathLike],
    database: str = "miiv",
    cohort_name: str = "lactate_map_vaso_24h",
    cross_database_validation: Sequence[str] = ("eicu", "hirid"),
) -> ResearchContext:
    """Build the formal EasyICU context contract for the shock case."""
    df, cohort_path = _load_cohort(cohort)
    manifest = _load_json(source_manifest)
    concept_sources = manifest.get("concept_sources", {})
    if not isinstance(concept_sources, Mapping):
        concept_sources = {}

    ctx = build_research_context(
        research_question=LACTATE_MAP_VASO_QUESTION,
        cohort=df,
        cohort_name=cohort_name,
        database=database,
        inclusion_criteria=[
            "One row per ICU stay from an EasyICU concept export.",
            "Lactate, MAP and vasopressor variables are derived within the first 24 hours after ICU admission.",
            "Unmeasured lactate stays are retained to keep ascertainment visible.",
        ],
        exclusion_criteria=[],
        target_outcome="death",
        cross_database_validation=cross_database_validation,
        id_columns=["stay_id"] if "stay_id" in df.columns else None,
        outcome_columns=[c for c in ["death", "los_icu", "los_hosp"] if c in df.columns],
        notes=(
            "Case-specific EasyICU context contract. The LLM may plan, code and write around "
            "this context, but source files, time windows, missingness semantics and forbidden "
            "transformations are deterministic metadata."
        ),
    )
    variables = []
    for descriptor in ctx.variables:
        metadata = LACTATE_MAP_VASO_VARIABLE_METADATA.get(descriptor.name)
        if metadata is not None:
            variables.append(_update_descriptor(
                descriptor,
                metadata=metadata,
                concept_sources=concept_sources,
                database=database,
            ))
        else:
            variables.append(descriptor)
    return ctx.model_copy(update={
        "variables": variables,
        "cohort_parquet": cohort_path,
    })


def context_information_summary(context: ResearchContext, *, label: str) -> Dict[str, object]:
    """Compact metrics for the EasyICU-context vs generic-context ablation."""
    variables = context.variables
    return {
        "context": label,
        "variables": len(variables),
        "variables_with_units": sum(1 for v in variables if v.unit),
        "variables_with_source_files": sum(1 for v in variables if v.source_files),
        "variables_with_derived_concepts": sum(1 for v in variables if v.derived_from_concepts),
        "variables_with_missingness_profiles": sum(1 for v in variables if v.missingness is not None),
        "variables_with_missingness_semantics": sum(1 for v in variables if v.missingness_semantics),
        "variables_with_pitfalls": sum(1 for v in variables if v.pitfalls),
        "variables_with_forbidden_transformations": sum(1 for v in variables if v.forbidden_transformations),
        "time_windows": len(context.time_windows),
        "cross_database_targets": len(context.cross_database_validation),
    }


def build_lactate_map_vaso_context_ablation_table(
    *,
    cohort: Union[pd.DataFrame, PathLike],
    source_manifest: Union[Mapping[str, object], PathLike],
    database: str = "miiv",
) -> pd.DataFrame:
    """Return a two-row table comparing generic and EasyICU-aware contexts."""
    aware = build_lactate_map_vaso_research_context(
        cohort=cohort,
        source_manifest=source_manifest,
        database=database,
    )
    df, _ = _load_cohort(cohort)
    naive = build_naive_research_context(
        research_question=LACTATE_MAP_VASO_QUESTION,
        cohort=df,
        cohort_name=aware.cohort.cohort_name,
        database=database,
        target_outcome="death",
        cross_database_validation=aware.cross_database_validation,
        id_columns=aware.cohort.id_columns,
        outcome_columns=aware.cohort.outcome_columns,
    )
    return pd.DataFrame([
        context_information_summary(naive, label="generic_csv_context"),
        context_information_summary(aware, label="easyicu_icu_context"),
    ])


def write_research_context(context: ResearchContext, path: PathLike) -> Path:
    """Write a research context JSON with stable indentation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(context.model_dump_json(indent=2), encoding="utf-8")
    return out


__all__ = [
    "LACTATE_MAP_VASO_QUESTION",
    "LACTATE_MAP_VASO_VARIABLE_METADATA",
    "build_lactate_map_vaso_research_context",
    "context_information_summary",
    "build_lactate_map_vaso_context_ablation_table",
    "write_research_context",
]
