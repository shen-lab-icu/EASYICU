"""Display-only concept metadata from cached packaged declarations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Mapping

from easyicu.concept import catalog as cc
from easyicu.concept import loader
from easyicu.concept_output_sources import COMPOSITE_LOADER_SUPPORT
from easyicu.outcome_availability import OUTCOME_CONCEPT_SUPPORTED_DATABASES
from easyicu.resources import package_path

_HELPER_NAMES = re.compile(
    r"(?:_(?:ascertainment|receipt|proxy_sensitivity|evidence_status|evidence_reason)$"
    r"|^aki_reference_.+|^creat_(?:baseline_|low_past_).+"
    r"|^aki_source_native_(?:profile|status|authority|reliability_grade|fidelity|reason|time_scale|uses_future)$"
    r"|^kidney_observation_window_coverage$"
    r"|_offset$|^raw_.+_(?:time|timestamp)$)"
)
_HELPER_ROLES = frozenset(
    {
        "internal", "bookkeeping", "helper", "ascertainment", "receipt",
        "observed_flag", "available_flag", "reference_helper", "baseline_helper",
        "proxy_sensitivity", "raw_offset", "raw_time",
    }
)


#: One dictionary entry per clinical concept: its other definitions and
#: component criteria fold under it.  Display only -- every column is still
#: extracted, exported and selectable by its own name.
CONCEPT_VARIANTS: Mapping[str, tuple[str, ...]] = {
    "aki_stage_reference": (
        "aki_reference", "aki_severe_reference",
        "aki_stage_creat_reference", "aki_stage_uo_reference",
        "aki_stage_rrt_reference", "aki_stage_reference_smoothed_6h",
        "aki_stage_source_native", "aki_source_native",
        "aki_severe_source_native", "aki_stage_creat_source_native",
        "aki_stage_uo_source_native", "aki_stage_crrt_source_native",
        "aki_stage_source_native_smoothed",
    ),
}
_VARIANT_PRIMARY = {
    variant: primary for primary, variants in CONCEPT_VARIANTS.items() for variant in variants
}


def variant_primary(concept_id: str) -> str | None:
    """The entry a folded variant belongs to, or None for an entry itself."""
    return _VARIANT_PRIMARY.get(concept_id)


def concept_audience(concept_id: str, definition: Mapping[str, Any]) -> str:
    """Return a display audience without changing extraction or selection.

    Explicit audience, deprecation/alias/helper markers, category/role and
    receipt-state levels or callback type take precedence. For older declarations,
    recognize only helper callback/identifier conventions. Observed/available
    suffixes denote flags only when no clinical measurement unit is declared.
    A derived score, Boolean outcome, clinical timestamp or public-reference
    AKI phenotype is not internal merely because it is derived or a reference.
    """
    if definition.get("audience") in {"clinical", "internal"}:
        return str(definition["audience"])
    if (
        definition.get("deprecated")
        or definition.get("alias_of")
        or definition.get("derived_helper")
        or definition.get("clinical_status") == "deprecated"
    ):
        return "internal"
    if any(
        definition.get(field) in _HELPER_ROLES
        for field in ("category", "role", "callback_type")
    ):
        return "internal"
    if set(definition.get("levels") or []) == {
        "confirmed", "proxy_only", "not_detected", "unavailable",
    }:
        return "internal"
    callback = str(definition.get("callback") or "")
    if callback.endswith("_alias"):
        return "internal"
    if callback in {"kdigo_creatinine_input", "kdigo_urine_input", "acute_rrt_input"}:
        return "internal"
    if _HELPER_NAMES.search(callback) or _HELPER_NAMES.search(concept_id):
        return "internal"
    unit = definition.get("unit") or ""
    units = unit if isinstance(unit, list) else [unit]
    if concept_id.endswith(("_observed", "_available")) and all(
        value in {"", "boolean", "bool", "state"} for value in units
    ):
        return "internal"
    return "clinical"


@lru_cache(maxsize=1)
def catalog_definitions() -> dict[str, Any]:
    """Cache base/overlay facts once; catalog requests do not reopen assets."""
    definitions = dict(loader.load_concept_dict_cached())
    with package_path("sofa2-dict.json") as path:
        overlay = json.loads(path.read_text(encoding="utf-8"))
    for key, patch in overlay.items():
        base = definitions.get(key, {})
        definitions[key] = {
            **base, **patch,
            "sources": {**base.get("sources", {}), **patch.get("sources", {})},
        }
    return definitions


def concept_databases(
    concept_id: str,
    definitions: Mapping[str, Any],
    visiting: frozenset[str] = frozenset(),
) -> set[str]:
    """Project declared mapping/derived-input support, never observed coverage.

    Partial derived inputs count as support, as in concept availability. Loader
    outputs inherit their owner, including its required-input and no-source
    constraints; fixed-horizon outcomes retain their dedicated authority.
    """
    if concept_id in visiting:
        return set()
    visiting = visiting | {concept_id}
    if concept_id in OUTCOME_CONCEPT_SUPPORTED_DATABASES:
        return set(OUTCOME_CONCEPT_SUPPORTED_DATABASES[concept_id])
    definition = definitions.get(concept_id)
    if definition is None:
        source = cc.COMPOSITE_CONCEPT_OUTPUT_SOURCES.get(concept_id)
        if source in definitions:
            return concept_databases(source, definitions, visiting)
        support = COMPOSITE_LOADER_SUPPORT.get(source)
        if support is None:
            return set()
        databases = {
            key
            for database in cc.SUPPORTED_DB_KEYS
            if database not in support.no_source_databases
            for key in (database, database + "_demo")
        }
        for required in support.required_concepts:
            databases &= concept_databases(required, definitions, visiting)
        return databases
    databases = {
        key for key, mappings in definition.get("sources", {}).items() if mappings
    }
    for field in ("concepts", "depends_on", "alias_of"):
        dependencies = definition.get(field) or []
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        for dependency in dependencies:
            databases |= concept_databases(dependency, definitions, visiting)
    return databases
