"""Shared readers and classifiers for figure-contract artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .authority.runtime_artifacts import current_successful_step_records


EXPORT_SUFFIXES = ("png", "svg", "pdf", "tiff", "tif")


def relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def figure_contract_paths(
    run_dir: Path,
    *,
    per_step_records: Sequence[Mapping[str, Any]] | None = None,
) -> List[Path]:
    supporting_paths = list(run_dir.glob("steps/*/outputs/*.figure_contract.json"))
    if per_step_records is not None:
        current_records = current_successful_step_records(per_step_records)
        declared_contracts: Dict[str, set[str] | None] = {}
        for record in current_records:
            step_id = str(record.get("step_id") or "").strip()
            if not step_id:
                continue
            summary = record.get("step_summary")
            if isinstance(summary, Mapping) and "contract_files" in summary:
                raw_files = summary.get("contract_files")
                declared_contracts[step_id] = {
                    Path(str(name)).name
                    for name in (raw_files if isinstance(raw_files, list) else [])
                    if str(name).strip()
                }
            else:
                # Compatibility for successful legacy records that predate the
                # explicit contract_files field. Modern records with the field
                # present (including an empty list) remain fail-closed.
                declared_contracts[step_id] = None
        supporting_paths = [
            path
            for path in supporting_paths
            if path.parents[1].name in declared_contracts
            and (
                declared_contracts[path.parents[1].name] is None
                or path.name in declared_contracts[path.parents[1].name]
            )
        ]
    paths = [
        *run_dir.glob("publication_figures/*.figure_contract.json"),
        *supporting_paths,
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
    # Newline-joined so multiword tokens cannot accidentally span two fields;
    # per-field whitespace is collapsed so double spaces cannot break them.
    parts = [
        str(panel.get("panel_id") or ""),
        str(panel.get("title") or ""),
        str(panel.get("role") or ""),
        str(panel.get("claim") or ""),
        str(panel.get("review_risk") or ""),
        json.dumps(panel.get("metadata") or {}, ensure_ascii=False, default=str),
    ]
    return "\n".join(re.sub(r"\s+", " ", part.strip().lower()) for part in parts)


def panel_chart_type(panel: Mapping[str, Any]) -> str:
    # Single source of truth for panel chart-family classification. Both the
    # display-suite gate and the article figure-strategy audit call this;
    # keeping one classifier prevents the same panel being reported with two
    # different chart types in sibling audit artifacts.
    metadata = panel.get("metadata") if isinstance(panel.get("metadata"), Mapping) else {}
    explicit = str(
        panel.get("chart_type")
        or panel.get("visual_form")
        or metadata.get("chart_type")
        or metadata.get("visual_form")
        or ""
    ).strip().lower()
    if explicit:
        return "_".join(explicit.split())
    text = panel_text(panel)
    if any(
        token in text
        for token in ("calibration", "roc", "curve", "kaplan", "cumulative incidence")
    ):
        return "curve"
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
        for token in ("distribution", "density", "histogram", "violin", "ridge")
    ):
        return "distribution"
    if any(
        token in text
        for token in ("flow", "attrition", "eligibility", "protocol", "schematic")
    ):
        return "flow"
    if any(
        token in text
        for token in ("missingness", "availability", "denominator", "included", "count")
    ):
        return "bar"
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
