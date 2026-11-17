"""Data Extraction's explicit minute module catalogue and export policy."""

from collections.abc import Mapping

from easyicu.concept.minute_resolution import (
    SUBHOUR_MODULE_INTERVALS,
    SubhourSelectionError,
    select_subhour_concepts,
    subhour_module_coverage,
)
from easyicu.webserver.hosted import is_hosted

HOSTED_SUBHOUR_CODE = "subhour_unavailable_when_hosted"
HOSTED_SUBHOUR_REASON = "Patient-level minute exports are unavailable on hosted instances."

_LABELS = {
    "subhour_signals": ("Bedside signals", "床旁信号"),
    "vent_native": ("Ventilator mechanics", "呼吸力学"),
    "haemo_native": ("Hemodynamics", "血流动力学"),
}


def extraction_module_options(database: str) -> dict:
    coverage = subhour_module_coverage(database)
    return {
        "database": database,
        "minute_resolution": {
            "unavailable_reason": HOSTED_SUBHOUR_CODE if is_hosted() else None,
            "modules": [
                {
                    "id": module,
                    "label": _LABELS[module],
                    "default_selected": False,
                    "interval_minutes": grid.total_seconds() / 60,
                    "signals": coverage[module],
                    "unavailable_reason": None if coverage[module] else "no_whitelisted_signals",
                }
                for module, grid in SUBHOUR_MODULE_INTERVALS.items()
            ],
        },
    }


def minute_export_modules(database, modules, concepts) -> dict[str, list[str]]:
    """Validate explicit minute cells before starting an export job."""
    selected = [
        module for module in (modules or ())
        if isinstance(module, str) and module in SUBHOUR_MODULE_INTERVALS
    ]
    projection = concepts if isinstance(concepts, Mapping) else {}
    concept_minute = SUBHOUR_MODULE_INTERVALS.keys() & projection.keys()
    if not selected and not concept_minute:
        return {}
    if is_hosted():
        raise SubhourSelectionError(HOSTED_SUBHOUR_CODE, HOSTED_SUBHOUR_REASON)
    if concept_minute.difference(selected):
        raise SubhourSelectionError(
            "subhour_signal_selection_invalid",
            "Signal projections require explicitly selected minute-resolution modules.",
        )

    coverage = subhour_module_coverage(database)
    for module in selected:
        select_subhour_concepts(module, database, coverage[module], projection.get(module))
    return {module: coverage[module] for module in selected}
