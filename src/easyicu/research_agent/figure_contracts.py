"""Shared readers and classifiers for figure-contract artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


EXPORT_SUFFIXES = ("png", "svg", "pdf", "tiff", "tif")


def relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def figure_contract_paths(run_dir: Path) -> List[Path]:
    paths = [
        *run_dir.glob("publication_figures/*.figure_contract.json"),
        *run_dir.glob("steps/*/outputs/*.figure_contract.json"),
    ]
    seen: set[str] = set()
    unique: List[Path] = []
    for path in sorted(paths):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def figure_contract_tier(path: Path, run_dir: Path) -> str:
    try:
        path.resolve().relative_to((run_dir / "publication_figures").resolve())
        return "primary_publication"
    except ValueError:
        pass
    try:
        path.resolve().relative_to((run_dir / "steps").resolve())
        return "supporting_step"
    except ValueError:
        return "other"


def relative_contract_paths(paths: Sequence[Path], run_dir: Path) -> List[str]:
    return sorted(relative_to_run(path, run_dir) for path in paths)


def figure_contract_export_paths(contract_path: Path) -> Dict[str, Path]:
    name = contract_path.name
    if name.endswith(".figure_contract.json"):
        stem = name[: -len(".figure_contract.json")]
    else:
        stem = contract_path.with_suffix("").name
    exports: Dict[str, Path] = {"contract": contract_path}
    for suffix in EXPORT_SUFFIXES:
        path = contract_path.with_name(f"{stem}.{suffix}")
        if path.exists():
            exports[suffix] = path
    return exports


def read_figure_contract(contract_path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def figure_contract_label(contract_path: Path) -> str:
    raw = read_figure_contract(contract_path)
    for key in ("title", "figure_id"):
        value = str(raw.get(key) or "").strip()
        if value:
            return value.replace("_", " ")
    name = contract_path.name
    if name.endswith(".figure_contract.json"):
        name = name[: -len(".figure_contract.json")]
    return name.replace("_", " ")


def figure_contract_text(raw: Mapping[str, Any]) -> str:
    parts: List[str] = [
        str(raw.get("figure_id") or ""),
        str(raw.get("title") or ""),
        str(raw.get("core_claim") or ""),
        str(raw.get("statistics_note") or ""),
    ]
    panels = raw.get("panels")
    if isinstance(panels, list):
        for panel in panels:
            if not isinstance(panel, Mapping):
                continue
            parts.extend(
                [
                    str(panel.get("panel_id") or ""),
                    str(panel.get("title") or ""),
                    str(panel.get("role") or ""),
                    str(panel.get("claim") or ""),
                    str(panel.get("review_risk") or ""),
                ]
            )
    return "\n".join(part for part in parts if part)


def panel_text(panel: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(panel.get("panel_id") or ""),
            str(panel.get("title") or ""),
            str(panel.get("role") or ""),
            str(panel.get("claim") or ""),
            str(panel.get("review_risk") or ""),
            json.dumps(panel.get("metadata") or {}, ensure_ascii=False, default=str),
        ]
    ).lower()


def panel_chart_type(panel: Mapping[str, Any]) -> str:
    metadata = panel.get("metadata") if isinstance(panel.get("metadata"), Mapping) else {}
    explicit = str(
        panel.get("chart_type")
        or panel.get("visual_form")
        or metadata.get("chart_type")
        or metadata.get("visual_form")
        or ""
    ).strip().lower()
    if explicit:
        return explicit.replace(" ", "_")
    text = panel_text(panel)
    if any(token in text for token in ("heatmap", "matrix", "jaccard", "overlap")):
        return "heatmap"
    if any(
        token in text
        for token in (
            "forest",
            "odds ratio",
            "odds-ratio",
            "risk ratio",
            "risk-ratio",
            "hazard ratio",
            "hazard-ratio",
            "ratio-scale",
        )
    ):
        return "forest"
    if any(
        token in text
        for token in ("risk difference", "prevalence", "event rate", "absolute risk")
    ):
        return "dot_interval"
    if any(
        token in text
        for token in ("missingness", "availability", "denominator", "included", "count")
    ):
        return "bar"
    if any(token in text for token in ("flow", "attrition", "eligibility")):
        return "flow"
    if any(
        token in text
        for token in ("distribution", "density", "histogram", "violin", "ridge")
    ):
        return "distribution"
    if any(token in text for token in ("calibration", "roc", "curve")):
        return "curve"
    return "unspecified"


def figure_contract_panel_summaries(raw: Mapping[str, Any]) -> List[Dict[str, str]]:
    panels = raw.get("panels")
    if not isinstance(panels, list):
        return []
    summaries: List[Dict[str, str]] = []
    for panel in panels:
        if not isinstance(panel, Mapping):
            continue
        summaries.append(
            {
                "panel_id": str(panel.get("panel_id") or ""),
                "title": str(panel.get("title") or ""),
                "role": str(panel.get("role") or "").strip().lower(),
                "chart_type": panel_chart_type(panel),
            }
        )
    return summaries
