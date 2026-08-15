"""Shared support for deterministic figures rebuilt from prior step outputs.

The parent-selection rule is an evidence boundary: a split figure may read only
its direct parent, while a legacy terminal overview may inspect prior steps.
Label helpers preserve the historical renderer vocabulary without putting this
policy back into the pipeline entry module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def figure_parent_candidate_step_dirs(
    *,
    steps_dir: Path,
    current_step_id: str,
) -> tuple[list[Path], bool]:
    """Return direct parent only for split figures, else legacy prior steps."""

    parent_step_id = str(current_step_id or "").removesuffix("_figure")
    direct_parent = Path(steps_dir) / parent_step_id
    is_split = parent_step_id != str(current_step_id or "")
    if is_split and direct_parent.is_dir():
        return [direct_parent], True
    return (
        [
            step_dir
            for step_dir in sorted(Path(steps_dir).iterdir())
            if step_dir.is_dir() and step_dir.name != current_step_id
        ],
        False,
    )


def publication_label(value: Any) -> str:
    """Return the stable reader-facing fallback used by legacy renderers."""

    token = str(value or "").strip()
    mapping = {
        "sepsis3": "Sepsis-3",
        "sep3_sofa2_max": "Experimental SOFA-2 Sepsis-3 phenotype",
        "age": "Age",
        "age_filled": "Age",
        "age_per_10y": "Age, per 10 years",
        "sex_m": "Male sex",
        "sex_male": "Male sex",
        "male": "Male sex",
        "hr_first": "Heart rate",
        "hr_first_filled": "Heart rate",
        "hr_max_per_10bpm": "Maximum heart rate, per 10 bpm",
        "map_first": "Mean arterial pressure",
        "map_first_filled": "Mean arterial pressure",
        "map_min": "Minimum mean arterial pressure",
        "resp_max_per_5": "Maximum respiratory rate, per 5/min",
        "temp_max_c": "Maximum temperature, per 1 deg C",
        "lactate": "Lactate",
        "lact": "Lactate",
        "lact_max_mmol_l": "Maximum lactate, per 1 mmol/L",
        "lact_measured": "Lactate measured",
        "bun_max_per_10": "Maximum BUN, per 10 units",
        "wbc_max_per_10": "Maximum WBC, per 10 units",
        "sofa2": "SOFA-2",
        "death": "In-hospital mortality",
        "alt_adult_no_los_all_vitals_sepsis3_derivable": "No LOS threshold",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable": ">=3 of 4 vitals",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable": "No temperature",
        "alt_adult_los2_all_vitals_sepsis3_derivable": "ICU LOS >=2 d",
    }
    lower = token.lower()
    if lower in mapping:
        return mapping[lower]
    cleaned = lower
    for suffix in ("_filled", "_first", "_measured"):
        cleaned = cleaned.removesuffix(suffix)
    return cleaned.replace("_", " ").strip().title() or token


def short_figure_label(value: Any, *, limit: int = 38) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "..."


__all__ = [
    "figure_parent_candidate_step_dirs",
    "publication_label",
    "short_figure_label",
]
