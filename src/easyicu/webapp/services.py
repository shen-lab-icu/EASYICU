"""Shared non-UI services for EasyICU web pages."""

from __future__ import annotations


COLUMN_NORMALIZATION_MAP = {
    "kdigo_aki_aki": "aki",
    "kdigo_aki_aki_stage": "aki_stage",
    "kdigo_aki_aki_stage_creat": "aki_stage_creat",
    "kdigo_aki_aki_stage_uo": "aki_stage_uo",
    "kdigo_aki_crea": "crea",
    "kdigo_aki_creat_low_past_48hr": "creat_low_past_48hr",
    "kdigo_aki_creat_low_past_7day": "creat_low_past_7day",
    "kdigo_aki_rrt": "rrt",
    "kdigo_aki_uo_rt_6hr": "uo_rt_6hr",
    "kdigo_aki_uo_rt_12hr": "uo_rt_12hr",
    "kdigo_aki_uo_rt_24hr": "uo_rt_24hr",
    "kdigo_creat_aki_stage_creat": "aki_stage_creat",
    "kdigo_creat_crea": "crea",
    "kdigo_creat_creat_low_past_48hr": "creat_low_past_48hr",
    "kdigo_creat_creat_low_past_7day": "creat_low_past_7day",
    "kdigo_uo_aki_stage_uo": "aki_stage_uo",
    "kdigo_uo_uo_rt_6hr": "uo_rt_6hr",
    "kdigo_uo_uo_rt_12hr": "uo_rt_12hr",
    "kdigo_uo_uo_rt_24hr": "uo_rt_24hr",
}

NORMALIZED_TO_ORIGINAL_MAP: dict[str, list[str]] = {}
for original_name, normalized_name in COLUMN_NORMALIZATION_MAP.items():
    NORMALIZED_TO_ORIGINAL_MAP.setdefault(normalized_name, []).append(original_name)


def normalize_column_name(col_name: str) -> str:
    """Normalize expanded concept columns to their canonical webapp name."""
    return COLUMN_NORMALIZATION_MAP.get(col_name, col_name)


def count_unique_columns(column_names: list[str]) -> int:
    """Count normalized unique data columns."""
    return len({normalize_column_name(col) for col in column_names})


def map_column_to_concept(col_name: str) -> str:
    """Backward-compatible alias for column normalization."""
    return normalize_column_name(col_name)


def count_unique_concepts(column_names: list[str]) -> int:
    """Backward-compatible alias for normalized unique column counts."""
    return count_unique_columns(column_names)


def get_unique_concepts(column_names: list[str]) -> set[str]:
    """Return normalized unique concept names."""
    return {normalize_column_name(col) for col in column_names}
