"""Lazy, single-feature Patient Review payloads.

The caller verifies the registered source and pseudonymous entity token. This
owner reads only ``stay_id`` + one feature + one source time column and returns
at most twelve observations without exposing the source identifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from easyicu.concept import catalog as concept_catalog
from easyicu.webserver import dataio
from easyicu.webserver.patient_drilldown.coverage import TIME_COLUMNS
from easyicu.webserver.entity_ids import (
    canonicalize_entity_frame,
    resolve_entity_id_column,
)


MAX_FEATURE_POINTS = 12


class FeatureDetailError(Exception):
    """Stable feature-detail boundary failure."""

    def __init__(self, detail: Dict[str, Any]):
        super().__init__(str(detail.get("error") or "patient_feature_error"))
        self.detail = detail


def load_feature_detail(
    *,
    export_path: Path,
    description: Mapping[str, Any],
    entity_id: str,
    entity_ref: str,
    entity_ordinal: int,
    feature: str,
) -> Dict[str, Any]:
    """Read one catalog feature for one already-verified entity."""

    feature_id = str(feature or "").strip().lower()
    if feature_id not in concept_catalog.CONCEPT_DICTIONARY:
        raise FeatureDetailError(
            {"error": "unknown_patient_feature", "feature": feature_id}
        )
    module = _feature_module(feature_id)
    item = next(
        (
            dict(row)
            for row in description.get("files") or []
            if row.get("module") == module and feature_id in (row.get("columns") or [])
        ),
        None,
    )
    if not item:
        return _unavailable_payload(
            description,
            entity_ref,
            entity_ordinal,
            feature_id,
            module,
            "feature_not_materialized",
        )

    columns = [str(value) for value in item.get("columns") or []]
    entity_column = resolve_entity_id_column(columns)
    if not entity_column:
        raise FeatureDetailError(
            {
                "error": "patient_feature_read_failed",
                "reason_code": "feature_entity_identifier_unavailable",
                "feature": feature_id,
                "module": module,
            }
        )
    time_column = next((name for name in TIME_COLUMNS if name in columns), None)
    selected = [entity_column]
    if time_column:
        selected.append(time_column)
    selected.append(feature_id)
    try:
        frame = dataio.read_export_projection(
            export_path / str(item.get("file") or ""),
            columns=selected,
            stay_ids={entity_id},
            entity_column=entity_column,
        )
    except Exception as exc:
        raise FeatureDetailError(
            {
                "error": "patient_feature_read_failed",
                "reason_code": "feature_projection_failed",
                "feature": feature_id,
                "module": module,
            }
        ) from exc
    frame = canonicalize_entity_frame(frame, entity_column)

    if frame is None or getattr(frame, "empty", True) or feature_id not in frame:
        return _unavailable_payload(
            description,
            entity_ref,
            entity_ordinal,
            feature_id,
            module,
            "selected_entity_has_no_rows",
        )

    observed = frame.dropna(subset=[feature_id]).copy()
    if observed.empty:
        return _unavailable_payload(
            description,
            entity_ref,
            entity_ordinal,
            feature_id,
            module,
            "selected_entity_has_no_observation",
        )
    if time_column and time_column in observed:
        observed = observed.sort_values(time_column)

    numeric_rows: list[tuple[Any, float]] = []
    for _, row in observed.iterrows():
        number = _finite_number(row.get(feature_id))
        if number is None:
            continue
        numeric_rows.append(
            (
                _json_cell(row.get(time_column)) if time_column else None,
                number,
            )
        )
    metadata = concept_catalog.CONCEPT_DICTIONARY.get(feature_id) or ()
    base = {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": {
            "database": description.get("database"),
            "generated": description.get("generated"),
        },
        "entity": {
            "ref": entity_ref,
            "label": f"Entity {entity_ordinal}",
            "ordinal": entity_ordinal,
        },
        "feature": {
            "feature": feature_id,
            "module": module,
            "name": str(metadata[0] if metadata else feature_id),
            "name_zh": str(
                metadata[1]
                if len(metadata) > 1 and metadata[1]
                else (metadata[0] if metadata else feature_id)
            ),
            "unit": str(metadata[2] if len(metadata) > 2 else ""),
            "time_column": time_column,
        },
        "privacy": {
            "direct_identifiers_returned": False,
            "raw_source_rows_returned": False,
            "max_points": MAX_FEATURE_POINTS,
            "payload_scope": "one_verified_entity_one_feature_bounded_projection",
        },
    }

    if len(numeric_rows) >= 2 and time_column:
        indices = _bounded_indices(len(numeric_rows))
        full_values = [row[1] for row in numeric_rows]
        sampled = [numeric_rows[index] for index in indices]
        base.update(
            {
                "status": "numeric_trajectory",
                "reason_code": None,
                "signal": {
                    "feature": feature_id,
                    "key": feature_id,
                    "name": base["feature"]["name"],
                    "unit": base["feature"]["unit"],
                    "times": [row[0] for row in sampled],
                    "values": [row[1] for row in sampled],
                    "time_axis": _time_axis_payload(
                        str(time_column), [row[0] for row in sampled]
                    ),
                    "point_count": len(numeric_rows),
                    "current": full_values[-1],
                    "min": round(min(full_values), 3),
                    "max": round(max(full_values), 3),
                    "mean": round(sum(full_values) / len(full_values), 3),
                    "thresholds": [],
                    "bounded": True,
                    "max_points": MAX_FEATURE_POINTS,
                },
            }
        )
        return base

    values = [_json_cell(value) for value in observed[feature_id].tolist()]
    unique_values: list[Any] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
        if len(unique_values) >= 12:
            break
    numeric_value = numeric_rows[-1][1] if numeric_rows else None
    base.update(
        {
            "status": (
                "observed_numeric_static"
                if numeric_value is not None
                else "observed_categorical"
            ),
            "reason_code": "fewer_than_two_timed_numeric_observations",
            "observation": {
                "current": numeric_value if numeric_value is not None else values[-1],
                "observed_values": unique_values,
                "observation_count": len(values),
            },
        }
    )
    return base


def _feature_module(feature: str) -> str:
    for module, concept_ids in concept_catalog.CONCEPT_GROUPS_INTERNAL.items():
        if feature in concept_ids:
            return module
    raise FeatureDetailError(
        {"error": "patient_feature_module_unavailable", "feature": feature}
    )


def _unavailable_payload(
    description: Mapping[str, Any],
    entity_ref: str,
    entity_ordinal: int,
    feature: str,
    module: str,
    reason_code: str,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": "real",
        "demo": False,
        "source": {
            "database": description.get("database"),
            "generated": description.get("generated"),
        },
        "entity": {
            "ref": entity_ref,
            "label": f"Entity {entity_ordinal}",
            "ordinal": entity_ordinal,
        },
        "feature": {"feature": feature, "module": module},
        "status": "unavailable",
        "reason_code": reason_code,
        "signal": None,
        "privacy": {
            "direct_identifiers_returned": False,
            "raw_source_rows_returned": False,
            "max_points": MAX_FEATURE_POINTS,
            "payload_scope": "one_verified_entity_one_feature_no_observation",
        },
    }


def _bounded_indices(point_count: int) -> list[int]:
    if point_count <= MAX_FEATURE_POINTS:
        return list(range(point_count))
    last = point_count - 1
    intervals = MAX_FEATURE_POINTS - 1
    return [(index * last) // intervals for index in range(MAX_FEATURE_POINTS)]


def _finite_number(value: Any) -> float | None:
    try:
        import math

        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _json_cell(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _time_axis_payload(time_column: str, times: list[Any]) -> Dict[str, str]:
    lowered = time_column.strip().lower()
    numeric = bool(times) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in times
    )
    if lowered == "hour" or (lowered == "charttime" and numeric):
        return {
            "kind": "relative_hours",
            "label_en": "ICU hour",
            "label_zh": "ICU 入科后小时",
            "unit": "hour",
            "source_column": lowered,
        }
    if lowered in {"measuredat_minutes", "observationoffset"}:
        return {
            "kind": "relative_minutes",
            "label_en": "ICU minute",
            "label_zh": "ICU 入科后分钟",
            "unit": "minute",
            "source_column": lowered,
        }
    if numeric:
        return {
            "kind": "recorded_offset",
            "label_en": "Recorded offset",
            "label_zh": "源记录偏移",
            "unit": "",
            "source_column": lowered,
        }
    return {
        "kind": "datetime",
        "label_en": "Recorded time",
        "label_zh": "源记录时间",
        "unit": "",
        "source_column": lowered,
    }


__all__ = [
    "FeatureDetailError",
    "MAX_FEATURE_POINTS",
    "load_feature_detail",
]
