"""Bounded local analysis artifacts for native Agent run outputs.

The Agent Outputs tab must not invent Table 1, missingness, ROC, or
calibration cards. This module builds those artifacts from the active EasyICU
export only, using aggregate payloads that avoid row-level identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from easyicu.webserver import dataio

OUTPUT_ARTIFACT_NAMES = [
    "table1_summary.json",
    "missingness_audit.json",
    "roc_curve.json",
    "calibration_curve.json",
]

_ENTITY_COLUMN = "stay_id"
_HIDDEN_COLUMNS = {
    "stay_id",
    "subject_id",
    "hadm_id",
    "icustay_id",
    "patientunitstayid",
    "patientid",
    "charttime",
    "time",
    "timestamp",
    "datetime",
}
_PREFERRED_PREDICTORS = [
    ("sofa2_score", "sofa2"),
    ("sepsis3_sofa2", "sep3_sofa2"),
    ("vitals", "shock_index"),
    ("vitals", "map"),
    ("vitals", "hr"),
    ("vitals", "spo2"),
    ("outcome", "los_icu"),
]


def build_agent_output_artifacts(
    *,
    export_path: str,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return safe local output artifacts keyed by artifact filename."""
    if _is_metadata_only_summary(summary):
        return _metadata_only_output_artifacts(
            source=source,
            summary=summary,
            cohort=cohort,
            quality=quality,
        )

    context = _load_context(export_path, source)
    frames = context["frames"]
    entity_ids = context["entity_ids"]
    death_by_entity = _bool_by_entity(
        frames.get("outcome"), "death", missing_false=True
    )
    feature_values = _feature_values_by_entity(frames, entity_ids)
    predictor = _select_predictor(feature_values, death_by_entity)

    return {
        "table1_summary.json": _table1_payload(
            source=source,
            summary=summary,
            cohort=cohort,
            frames=frames,
            entity_ids=entity_ids,
            death_by_entity=death_by_entity,
        ),
        "missingness_audit.json": _missingness_payload(
            source=source,
            summary=summary,
            quality=quality,
            frames=frames,
            entity_ids=entity_ids,
        ),
        "roc_curve.json": _roc_payload(
            source=source,
            summary=summary,
            predictor=predictor,
            death_by_entity=death_by_entity,
        ),
        "calibration_curve.json": _calibration_payload(
            source=source,
            summary=summary,
            predictor=predictor,
            death_by_entity=death_by_entity,
        ),
    }


def _is_metadata_only_summary(summary: Dict[str, Any]) -> bool:
    return str(summary.get("snapshot_basis") or "") == "registry_metadata"


def _metadata_only_output_artifacts(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    quality: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {
        "table1_summary.json": _metadata_table1_payload(
            source=source,
            summary=summary,
            cohort=cohort,
        ),
        "missingness_audit.json": _metadata_missingness_payload(
            source=source,
            summary=summary,
            quality=quality,
        ),
        "roc_curve.json": _metadata_metric_payload("roc_curve", source, summary),
        "calibration_curve.json": _metadata_metric_payload(
            "calibration_curve", source, summary
        ),
    }


def _metadata_table1_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
) -> Dict[str, Any]:
    denominator = _metadata_denominator(summary)
    variables = [
        {
            "type": "metadata_only",
            "module": item["module"],
            "feature": item["feature"],
            "label": _label(item["feature"]),
            "status": "available_in_manifest",
            "values": {},
        }
        for item in _metadata_features(source)[:120]
    ]
    return {
        "kind": "table1_summary",
        "status": "metadata_only",
        "source": _source_summary(source),
        "denominator": denominator,
        "outcome": {
            "event": "death",
            "available": False,
            "event_count": None,
            "non_event_count": None,
            "basis": "metadata_only_preflight",
        },
        "groups": [
            {"id": "overall", "label": "Overall", "entities": denominator},
            {"id": "survived", "label": "Survived", "entities": None},
            {"id": "deceased", "label": "Deceased", "entities": None},
        ],
        "variables": variables,
        "cohort_snapshot": {
            "summary": summary,
            "cohort": cohort,
        },
        "privacy": _privacy_scope(),
    }


def _metadata_missingness_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    quality: List[Dict[str, Any]],
) -> Dict[str, Any]:
    denominator = _metadata_denominator(summary)
    rows = []
    for item in _metadata_features(source):
        rows.append(
            {
                "module": item["module"],
                "feature": item["feature"],
                "label": _label(item["feature"]),
                "entities_observed": None,
                "denominator": denominator,
                "coverage_pct": None,
                "missing_pct": None,
                "records_non_missing": None,
                "declared_module_rows": item.get("rows"),
                "coverage_basis": "manifest_file_inventory",
                "status": "metadata_only",
            }
        )
    return {
        "kind": "missingness_audit",
        "status": "metadata_only",
        "source": _source_summary(source),
        "denominator": denominator,
        "feature_count": len(rows),
        "module_quality": quality,
        "rows": rows[:250],
        "summary": {
            "features_with_full_coverage": None,
            "features_below_80_pct": None,
            "modules": summary.get("modules"),
            "basis": "metadata_only_preflight",
        },
        "privacy": _privacy_scope(),
    }


def _metadata_metric_payload(
    kind: str,
    source: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "status": "not_available",
        "source": _source_summary(source),
        "summary": summary,
        "reason": (
            "requires a bounded sampled cohort with observed outcomes and "
            "predictors; skipped for large-export metadata preflight"
        ),
        "predictor": None,
        "privacy": _privacy_scope(),
    }


def _metadata_denominator(summary: Dict[str, Any]) -> Optional[int]:
    try:
        return int(summary.get("stays"))
    except (TypeError, ValueError):
        return None


def _metadata_features(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    files = source.get("files") if isinstance(source, dict) else []
    for meta in files or []:
        if not isinstance(meta, dict):
            continue
        module = str(meta.get("module") or "").strip()
        rows = meta.get("rows")
        for column in meta.get("columns") or []:
            feature = str(column or "").strip()
            if not module or not feature or _is_hidden_column(feature):
                continue
            features.append({"module": module, "feature": feature, "rows": rows})
    features.sort(key=lambda row: (row["module"], row["feature"]))
    return features


def _load_context(export_path: str, source: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(export_path).expanduser()
    files = [f for f in source.get("files", []) if isinstance(f, dict)]
    frames: Dict[str, Any] = {}
    for meta in files:
        module = str(meta.get("module") or "")
        file_name = str(meta.get("file") or "")
        if not module or not file_name:
            continue
        path = root / file_name
        if not path.exists() or not path.is_file():
            continue
        try:
            frame = dataio._read_export_frame(path)  # local exported module table
        except Exception:
            continue
        if (
            frame is None
            or getattr(frame, "empty", True)
            or _ENTITY_COLUMN not in frame.columns
        ):
            continue
        frame = frame.copy()
        frame[_ENTITY_COLUMN] = frame[_ENTITY_COLUMN].map(dataio._norm_id)
        frames[module] = frame

    demo = frames.get("demographics")
    entity_ids: List[str] = []
    if demo is not None and _ENTITY_COLUMN in demo.columns:
        entity_ids = [
            sid
            for sid in demo[_ENTITY_COLUMN]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(500)
            .tolist()
            if sid
        ]
    if not entity_ids:
        for frame in frames.values():
            if _ENTITY_COLUMN in frame.columns:
                entity_ids = [
                    sid
                    for sid in frame[_ENTITY_COLUMN]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(500)
                    .tolist()
                    if sid
                ]
                if entity_ids:
                    break
    entity_set = set(entity_ids)
    if entity_set:
        frames = {
            module: frame[frame[_ENTITY_COLUMN].astype(str).isin(entity_set)].copy()
            for module, frame in frames.items()
        }
    return {"frames": frames, "entity_ids": entity_ids}


def _table1_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    cohort: Dict[str, Any],
    frames: Dict[str, Any],
    entity_ids: List[str],
    death_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    groups = _group_entity_ids(entity_ids, death_by_entity)
    variables = [
        _numeric_table_row("Age", "demographics", "age", frames, groups),
        _categorical_bool_row(
            "Female sex",
            "demographics",
            "sex",
            frames,
            groups,
            true_values={"f", "female"},
        ),
        _numeric_table_row(
            "SOFA-2 score", "sofa2_score", "sofa2", frames, groups, aggregate="max"
        ),
        _numeric_table_row("ICU length of stay", "outcome", "los_icu", frames, groups),
        _numeric_table_row(
            "Hospital length of stay", "outcome", "los_hosp", frames, groups
        ),
        _categorical_bool_row(
            "Sepsis-3 (SOFA-2 based)", "sepsis3_sofa2", "sep3_sofa2", frames, groups
        ),
    ]
    variables = [row for row in variables if row is not None]
    return {
        "kind": "table1_summary",
        "status": "ok" if variables and entity_ids else "not_available",
        "source": _source_summary(source),
        "denominator": len(entity_ids),
        "outcome": {
            "event": "death",
            "available": bool(death_by_entity),
            "event_count": sum(1 for v in death_by_entity.values() if v),
            "non_event_count": sum(1 for v in death_by_entity.values() if v is False),
        },
        "groups": [
            {"id": "overall", "label": "Overall", "entities": len(groups["overall"])},
            {
                "id": "survived",
                "label": "Survived",
                "entities": len(groups["survived"]),
            },
            {
                "id": "deceased",
                "label": "Deceased",
                "entities": len(groups["deceased"]),
            },
        ],
        "variables": variables,
        "cohort_snapshot": {
            "summary": summary,
            "cohort": cohort,
        },
        "privacy": _privacy_scope(),
    }


def _missingness_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    quality: List[Dict[str, Any]],
    frames: Dict[str, Any],
    entity_ids: List[str],
) -> Dict[str, Any]:
    denominator = len(entity_ids)
    rows: List[Dict[str, Any]] = []
    for module, frame in sorted(frames.items()):
        if frame is None or getattr(frame, "empty", True):
            continue
        for col in frame.columns:
            if _is_hidden_column(str(col)):
                continue
            present = _present_entity_count(frame, str(col))
            non_missing = _non_missing_record_count(frame, str(col))
            coverage = round(present / denominator * 100, 1) if denominator else None
            rows.append(
                {
                    "module": module,
                    "feature": str(col),
                    "label": _label(str(col)),
                    "entities_observed": present,
                    "denominator": denominator,
                    "coverage_pct": coverage,
                    "missing_pct": (
                        round(100 - coverage, 1) if coverage is not None else None
                    ),
                    "records_non_missing": non_missing,
                    "coverage_basis": "entity_non_missing_presence",
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["module"]),
            -(row.get("coverage_pct") or 0),
            str(row["feature"]),
        )
    )
    return {
        "kind": "missingness_audit",
        "status": "ok" if rows else "not_available",
        "source": _source_summary(source),
        "denominator": denominator,
        "feature_count": len(rows),
        "module_quality": quality,
        "rows": rows,
        "summary": {
            "features_with_full_coverage": sum(
                1 for row in rows if row.get("coverage_pct") == 100.0
            ),
            "features_below_80_pct": sum(
                1
                for row in rows
                if isinstance(row.get("coverage_pct"), (int, float))
                and row["coverage_pct"] < 80
            ),
            "modules": summary.get("modules"),
        },
        "privacy": _privacy_scope(),
    }


def _roc_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    predictor: Optional[Dict[str, Any]],
    death_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    pairs = _prediction_pairs(predictor, death_by_entity) if predictor else []
    if not _can_score(pairs, min_pairs=2):
        return _unavailable_metric_payload(
            "roc_curve",
            source,
            summary,
            "requires an outcome column with both classes and a numeric predictor observed in the cohort",
            predictor,
        )
    points, auc = _roc_points(pairs)
    return {
        "kind": "roc_curve",
        "status": "ok",
        "source": _source_summary(source),
        "outcome": "death",
        "predictor": _predictor_summary(predictor),
        "n_entities": len(pairs),
        "auc": auc,
        "points": points,
        "interpretation_scope": "exploratory_univariate_preexperiment_not_reportable",
        "privacy": _privacy_scope(),
    }


def _calibration_payload(
    *,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    predictor: Optional[Dict[str, Any]],
    death_by_entity: Dict[str, bool],
) -> Dict[str, Any]:
    pairs = _prediction_pairs(predictor, death_by_entity) if predictor else []
    if not _can_score(pairs, min_pairs=5):
        return _unavailable_metric_payload(
            "calibration_curve",
            source,
            summary,
            "requires at least 5 entities, both outcome classes, and a numeric predictor for bounded bins",
            predictor,
        )
    probabilities = _logistic_probabilities(
        [score for score, _ in pairs], [event for _, event in pairs]
    )
    bins = _calibration_bins(probabilities, [event for _, event in pairs])
    return {
        "kind": "calibration_curve",
        "status": "ok" if bins else "not_available",
        "source": _source_summary(source),
        "outcome": "death",
        "predictor": _predictor_summary(predictor),
        "n_entities": len(pairs),
        "model": "single_predictor_logistic_preexperiment",
        "bins": bins,
        "interpretation_scope": "exploratory_univariate_preexperiment_not_reportable",
        "privacy": _privacy_scope(),
    }


def _group_entity_ids(
    entity_ids: List[str], death_by_entity: Dict[str, bool]
) -> Dict[str, List[str]]:
    overall = list(entity_ids)
    return {
        "overall": overall,
        "survived": [sid for sid in overall if death_by_entity.get(sid) is False],
        "deceased": [sid for sid in overall if death_by_entity.get(sid) is True],
    }


def _numeric_table_row(
    label: str,
    module: str,
    feature: str,
    frames: Dict[str, Any],
    groups: Dict[str, List[str]],
    *,
    aggregate: str = "median",
) -> Optional[Dict[str, Any]]:
    values = _numeric_by_entity(frames.get(module), feature, aggregate=aggregate)
    if not values:
        return None
    return {
        "type": "numeric",
        "label": label,
        "module": module,
        "feature": feature,
        "values": {
            group: _numeric_summary([values[sid] for sid in ids if sid in values])
            for group, ids in groups.items()
        },
    }


def _categorical_bool_row(
    label: str,
    module: str,
    feature: str,
    frames: Dict[str, Any],
    groups: Dict[str, List[str]],
    *,
    true_values: Optional[set[str]] = None,
) -> Optional[Dict[str, Any]]:
    values = _bool_by_entity(frames.get(module), feature, true_values=true_values)
    if not values:
        return None
    return {
        "type": "binary",
        "label": label,
        "module": module,
        "feature": feature,
        "values": {
            group: _binary_summary([values[sid] for sid in ids if sid in values])
            for group, ids in groups.items()
        },
    }


def _feature_values_by_entity(
    frames: Dict[str, Any], entity_ids: List[str]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    entity_set = set(entity_ids)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for module, frame in frames.items():
        if frame is None or getattr(frame, "empty", True):
            continue
        for col in frame.columns:
            col = str(col)
            if _is_hidden_column(col):
                continue
            values = _numeric_by_entity(frame, col)
            if not values:
                continue
            values = {sid: val for sid, val in values.items() if sid in entity_set}
            if values:
                out[(module, col)] = values
    return out


def _select_predictor(
    feature_values: Dict[Tuple[str, str], Dict[str, float]],
    death_by_entity: Dict[str, bool],
) -> Optional[Dict[str, Any]]:
    candidates = _PREFERRED_PREDICTORS + sorted(feature_values.keys())
    seen: set[Tuple[str, str]] = set()
    best: Optional[Dict[str, Any]] = None
    for module, feature in candidates:
        key = (module, feature)
        if key in seen:
            continue
        seen.add(key)
        values = feature_values.get(key)
        if not values:
            continue
        pairs = _prediction_pairs({"values": values}, death_by_entity)
        if not _can_score(pairs, min_pairs=2):
            continue
        _, auc = _roc_points(pairs)
        candidate = {
            "module": module,
            "feature": feature,
            "label": _label(feature),
            "values": values,
            "n_entities": len(pairs),
            "auc": auc,
        }
        if key in _PREFERRED_PREDICTORS:
            return candidate
        if best is None or len(pairs) > int(best.get("n_entities") or 0):
            best = candidate
    return best


def _prediction_pairs(
    predictor: Optional[Dict[str, Any]], death_by_entity: Dict[str, bool]
) -> List[Tuple[float, int]]:
    if not predictor:
        return []
    values = predictor.get("values") or {}
    pairs: List[Tuple[float, int]] = []
    for sid, score in values.items():
        event = death_by_entity.get(str(sid))
        if event is None:
            continue
        try:
            pairs.append((float(score), 1 if event else 0))
        except (TypeError, ValueError):
            continue
    return pairs


def _can_score(pairs: List[Tuple[float, int]], *, min_pairs: int) -> bool:
    if len(pairs) < min_pairs:
        return False
    labels = {event for _, event in pairs}
    return labels == {0, 1}


def _roc_points(
    pairs: List[Tuple[float, int]],
) -> Tuple[List[Dict[str, float]], Optional[float]]:
    total_pos = sum(event for _, event in pairs)
    total_neg = len(pairs) - total_pos
    if not total_pos or not total_neg:
        return [], None
    ordered = sorted(pairs, key=lambda row: row[0], reverse=True)
    tp = 0
    fp = 0
    points = [{"threshold": None, "fpr": 0.0, "tpr": 0.0}]
    last_score: Optional[float] = None
    for score, event in ordered:
        if last_score is not None and score != last_score:
            points.append(
                {
                    "threshold": round(float(last_score), 6),
                    "fpr": round(fp / total_neg, 6),
                    "tpr": round(tp / total_pos, 6),
                }
            )
        if event:
            tp += 1
        else:
            fp += 1
        last_score = score
    points.append(
        {
            "threshold": round(float(last_score if last_score is not None else 0), 6),
            "fpr": round(fp / total_neg, 6),
            "tpr": round(tp / total_pos, 6),
        }
    )
    auc = 0.0
    for left, right in zip(points, points[1:]):
        auc += (right["fpr"] - left["fpr"]) * (right["tpr"] + left["tpr"]) / 2
    return points, round(max(0.0, min(1.0, auc)), 4)


def _logistic_probabilities(scores: List[float], labels: List[int]) -> List[float]:
    import math

    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / max(len(scores), 1)
    scale = math.sqrt(variance) or 1.0
    xs = [(x - mean) / scale for x in scores]
    intercept = 0.0
    coef = 0.0
    lr = 0.08
    for _ in range(300):
        grad_i = 0.0
        grad_c = 0.0
        for x, y in zip(xs, labels):
            p = 1.0 / (1.0 + math.exp(max(-35.0, min(35.0, -(intercept + coef * x)))))
            grad_i += p - y
            grad_c += (p - y) * x
        intercept -= lr * grad_i / len(xs)
        coef -= lr * grad_c / len(xs)
    return [
        round(1.0 / (1.0 + math.exp(max(-35.0, min(35.0, -(intercept + coef * x))))), 6)
        for x in xs
    ]


def _calibration_bins(
    probabilities: List[float], labels: List[int]
) -> List[Dict[str, Any]]:
    if not probabilities or len(probabilities) != len(labels):
        return []
    paired = sorted(zip(probabilities, labels), key=lambda row: row[0])
    bin_count = min(5, max(2, len(paired) // 5))
    bins: List[Dict[str, Any]] = []
    for i in range(bin_count):
        start = round(i * len(paired) / bin_count)
        end = round((i + 1) * len(paired) / bin_count)
        chunk = paired[start:end]
        if not chunk:
            continue
        probs = [p for p, _ in chunk]
        events = [y for _, y in chunk]
        bins.append(
            {
                "bin": i + 1,
                "entities": len(chunk),
                "predicted_mean": round(sum(probs) / len(probs), 4),
                "observed_event_rate": round(sum(events) / len(events), 4),
                "probability_min": round(min(probs), 4),
                "probability_max": round(max(probs), 4),
            }
        )
    return bins


def _numeric_by_entity(
    frame: Any, feature: str, *, aggregate: str = "median"
) -> Dict[str, float]:
    if (
        frame is None
        or getattr(frame, "empty", True)
        or _ENTITY_COLUMN not in frame.columns
        or feature not in frame.columns
    ):
        return {}
    import pandas as pd

    tmp = frame[[_ENTITY_COLUMN, feature]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp = tmp.dropna(subset=[feature])
    if tmp.empty:
        return {}
    grouped = tmp.groupby(_ENTITY_COLUMN)[feature]
    values = grouped.max() if aggregate == "max" else grouped.median()
    return {str(k): round(float(v), 6) for k, v in values.items()}


def _bool_by_entity(
    frame: Any,
    feature: str,
    *,
    true_values: Optional[set[str]] = None,
    missing_false: bool = False,
) -> Dict[str, bool]:
    if (
        frame is None
        or getattr(frame, "empty", True)
        or _ENTITY_COLUMN not in frame.columns
        or feature not in frame.columns
    ):
        return {}
    out: Dict[str, bool] = {}
    accepted = true_values or {"1", "true", "t", "yes", "y", "positive", "present"}
    for sid, vals in frame.groupby(_ENTITY_COLUMN)[feature]:
        flags = []
        for value in vals:
            text = "" if value is None else str(value).strip().lower()
            if text in accepted:
                flags.append(True)
            elif text in {
                "0",
                "false",
                "f",
                "m",
                "male",
                "no",
                "n",
                "negative",
                "absent",
            }:
                flags.append(False)
            elif missing_false and text in {"", "nan", "none", "null"}:
                flags.append(False)
            else:
                try:
                    numeric = float(text)
                except ValueError:
                    continue
                if numeric == 1:
                    flags.append(True)
                elif numeric == 0:
                    flags.append(False)
        if flags:
            out[str(sid)] = any(flags)
    return out


def _present_entity_count(frame: Any, feature: str) -> int:
    if frame is None or getattr(frame, "empty", True) or feature not in frame.columns:
        return 0
    mask = _present_mask(frame[feature])
    if not mask.any():
        return 0
    return int(frame.loc[mask, _ENTITY_COLUMN].dropna().astype(str).nunique())


def _non_missing_record_count(frame: Any, feature: str) -> int:
    if frame is None or getattr(frame, "empty", True) or feature not in frame.columns:
        return 0
    return int(_present_mask(frame[feature]).sum())


def _present_mask(values: Any) -> Any:
    mask = values.notna()
    try:
        text = values.astype(str).str.strip()
        mask = mask & text.ne("") & text.str.lower().ne("nan")
    except Exception:
        pass
    return mask


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {"entities": 0, "mean": None, "median": None, "q1": None, "q3": None}
    return {
        "entities": len(xs),
        "mean": round(sum(xs) / len(xs), 3),
        "median": _quantile(xs, 0.5),
        "q1": _quantile(xs, 0.25),
        "q3": _quantile(xs, 0.75),
        "min": round(min(xs), 3),
        "max": round(max(xs), 3),
    }


def _binary_summary(values: List[bool]) -> Dict[str, Any]:
    if not values:
        return {"entities": 0, "positive": 0, "positive_pct": None}
    positives = sum(1 for v in values if v)
    return {
        "entities": len(values),
        "positive": positives,
        "positive_pct": round(positives / len(values) * 100, 1),
    }


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return round(xs[0], 3)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return round(xs[lo] * (1 - frac) + xs[hi] * frac, 3)


def _is_hidden_column(column: str) -> bool:
    lower = column.strip().lower()
    return lower in _HIDDEN_COLUMNS or lower.endswith("_id") or lower.endswith("id")


def _label(feature: str) -> str:
    return feature.replace("_", " ").strip().title()


def _source_summary(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path": source.get("path"),
        "label": source.get("label"),
        "database": source.get("database"),
        "generated": source.get("generated"),
    }


def _predictor_summary(predictor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not predictor:
        return None
    return {
        "module": predictor.get("module"),
        "feature": predictor.get("feature"),
        "label": predictor.get("label"),
        "entities": predictor.get("n_entities"),
    }


def _unavailable_metric_payload(
    kind: str,
    source: Dict[str, Any],
    summary: Dict[str, Any],
    reason: str,
    predictor: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "status": "not_available",
        "source": _source_summary(source),
        "summary": summary,
        "reason": reason,
        "predictor": _predictor_summary(predictor),
        "privacy": _privacy_scope(),
    }


def _privacy_scope() -> Dict[str, Any]:
    return {
        "aggregate_only": True,
        "raw_entity_rows_persisted": False,
        "direct_identifiers_persisted": False,
    }
