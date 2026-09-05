"""Shared readers and classifiers for figure-contract artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..authority.runtime_artifacts import current_successful_step_records
from ..contracts.frozen_payload import freeze_payload, thaw_payload

EXPORT_SUFFIXES = ("png", "svg", "pdf", "tiff", "tif")

ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION = "easyicu.article_display_policy/1"
ARTICLE_DISPLAY_POLICY_VALIDATOR = "article_display_policy"

DisplayPlacement = Literal["main", "supplementary"]
DisplayPurpose = Literal["scientific_result", "diagnostic", "context", "audit"]

ARTICLE_DISPLAY_ROLE_UNSUPPORTED = "ARTICLE_DISPLAY_ROLE_UNSUPPORTED"
ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN = (
    "ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN"
)
ARTICLE_DISPLAY_PURPOSE_CONFLICT = "ARTICLE_DISPLAY_PURPOSE_CONFLICT"

_SCIENTIFIC_RESULT_ROLES = frozenset(
    {
        "relationship",
        "absolute_risk",
        "validation",
        "descriptive_result",
        "primary_estimand",
        "robustness",
        "heterogeneity",
        "model_performance",
        "calibration",
        "clinical_utility",
        "explainability",
        "temporal_absolute_risk",
        "survival_effect",
        "phenotype_structure",
        "phenotype_profile",
        "downstream_characterization",
        "causal_contrast",
        "distribution",
        "transportability",
    }
)
_DIAGNOSTIC_ROLES = frozenset(
    {
        "deviation",
        "diagnostics",
        "balance_positivity",
        "stability",
        "cluster_selection",
    }
)
_CONTEXT_ROLES = frozenset(
    {
        "overview",
        "mechanism",
        "workflow",
        "cohort_accounting",
        "baseline_context",
        "validation_design",
        "causal_protocol",
    }
)
_AUDIT_ROLES = frozenset(
    {
        "audit",
        "data_quality",
        "measurement_missingness",
        "measurement_process",
        "supplementary_provenance",
    }
)


class ArticleDisplayPolicyError(ValueError):
    """Owner-attributable display-policy failure with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.validator = ARTICLE_DISPLAY_POLICY_VALIDATOR


class ArticleDisplayPolicyRequest(BaseModel):
    """Typed inputs used to decide one panel or table's article role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    article_role: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    requested_placement: DisplayPlacement | None = None
    analysis_type: str = ""
    scientific_status: str = "analysis_only"
    central_to_question: bool = False
    interpretation_critical: bool = False
    terminal_diagnostic: bool = False


class ArticleDisplayDecision(BaseModel):
    """Immutable receipt compiled by the display-policy owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION
    placement: DisplayPlacement
    display_purpose: DisplayPurpose
    reason_code: str


def _base_display_purpose(article_role: str) -> DisplayPurpose:
    if article_role in _SCIENTIFIC_RESULT_ROLES:
        return "scientific_result"
    if article_role in _DIAGNOSTIC_ROLES:
        return "diagnostic"
    if article_role in _CONTEXT_ROLES:
        return "context"
    if article_role in _AUDIT_ROLES:
        return "audit"
    raise ArticleDisplayPolicyError(
        ARTICLE_DISPLAY_ROLE_UNSUPPORTED,
        f"Unsupported typed article role: {article_role!r}",
    )


def decide_article_display(
    request: ArticleDisplayPolicyRequest,
) -> ArticleDisplayDecision:
    """Compile role, placement, and status without case or title inference."""

    role = request.article_role
    purpose = _base_display_purpose(role)
    failed_closed = request.scientific_status in {"failed_closed", "blocked"}

    if failed_closed:
        if request.terminal_diagnostic:
            return ArticleDisplayDecision(
                placement=(
                    "main"
                    if request.central_to_question or request.interpretation_critical
                    else "supplementary"
                ),
                display_purpose="diagnostic",
                reason_code="TERMINAL_DIAGNOSTIC_DISPLAY",
            )
        if purpose == "scientific_result":
            raise ArticleDisplayPolicyError(
                ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN,
                "A failed-closed or blocked analysis cannot emit a scientific-result display.",
            )
        return ArticleDisplayDecision(
            placement="supplementary",
            display_purpose=purpose,
            reason_code="FAILED_CLOSED_SUPPORTING_DISPLAY",
        )

    if role in _AUDIT_ROLES:
        if request.central_to_question or request.analysis_type == "data_quality_audit":
            return ArticleDisplayDecision(
                placement=request.requested_placement or "main",
                display_purpose="scientific_result",
                reason_code="MEASUREMENT_PROCESS_IS_RESEARCH_RESULT",
            )
        if request.interpretation_critical:
            return ArticleDisplayDecision(
                placement="main",
                display_purpose="diagnostic",
                reason_code="INTERPRETATION_CRITICAL_DATA_DIAGNOSTIC",
            )
        return ArticleDisplayDecision(
            placement="supplementary",
            display_purpose="audit",
            reason_code="ROUTINE_DATA_AUDIT_SUPPLEMENTARY",
        )

    default_placement: DisplayPlacement = (
        "supplementary" if purpose == "diagnostic" else "main"
    )
    placement = request.requested_placement or default_placement
    return ArticleDisplayDecision(
        placement=placement,
        display_purpose=purpose,
        reason_code={
            "scientific_result": "SCIENTIFIC_RESULT_DISPLAY",
            "diagnostic": "DIAGNOSTIC_DISPLAY",
            "context": "READER_CONTEXT_DISPLAY",
            "audit": "AUDIT_DISPLAY",
        }[purpose],
    )


def relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _current_contract_files(
    per_step_records: Sequence[Mapping[str, Any]] | None,
) -> dict[str, set[str] | None] | None:
    if per_step_records is None:
        return None
    declared: dict[str, set[str] | None] = {}
    for record in current_successful_step_records(per_step_records):
        step_id = str(record.get("step_id") or "").strip()
        if not step_id:
            continue
        summary = record.get("step_summary")
        if isinstance(summary, Mapping) and "contract_files" in summary:
            raw_files = summary.get("contract_files")
            declared[step_id] = {
                Path(str(name)).name
                for name in (raw_files if isinstance(raw_files, list) else [])
                if str(name).strip()
            }
        else:
            # Legacy successful records can predate explicit contract_files;
            # a modern explicit empty list still selects no step contracts.
            declared[step_id] = None
    return declared


def figure_contract_paths(
    run_dir: Path,
    *,
    per_step_records: Sequence[Mapping[str, Any]] | None = None,
    include_publication_figures: bool = True,
) -> List[Path]:
    supporting_paths = list(run_dir.glob("steps/*/outputs/*.figure_contract.json"))
    declared_contracts = _current_contract_files(per_step_records)
    if declared_contracts is not None:
        supporting_paths = [
            path
            for path in supporting_paths
            if path.parents[1].name in declared_contracts
            and (
                declared_contracts[path.parents[1].name] is None
                or path.name in declared_contracts[path.parents[1].name]
            )
        ]
    paths = list(supporting_paths)
    if include_publication_figures:
        paths = [
            *run_dir.glob("publication_figures/*.figure_contract.json"),
            *paths,
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


@dataclass(frozen=True, slots=True)
class FigureContractReadError:
    path: Path
    reason_code: str
    detail: str

    def message(self, run_dir: Path) -> str:
        return (
            f"{self.reason_code}: {relative_to_run(self.path, run_dir)} ({self.detail})"
        )


@dataclass(frozen=True, slots=True)
class FigureContractSnapshot:
    """One observed file, not scientific validation or publication authority."""

    path: Path
    tier: str
    sha256: str
    _payload: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return thaw_payload(self._payload)


def _selection_identity(
    per_step_records: Sequence[Mapping[str, Any]] | None,
) -> tuple | None:
    declared = _current_contract_files(per_step_records)
    if declared is None:
        return None
    return tuple(
        (step_id, tuple(sorted(names)) if names is not None else None)
        for step_id, names in sorted(declared.items())
    )


@dataclass(frozen=True, slots=True)
class FigureContractInventory:
    """An immutable read of current contracts shared by one reporting pass.

    Selection remains owned by the current-step artifact rules. Consumers keep
    their scientific policies; this value only owns reading, read errors, and
    detached projections. It must be reloaded for a later reporting pass.
    """

    run_dir: Path
    paths: tuple[Path, ...]
    snapshots: tuple[FigureContractSnapshot, ...]
    errors: tuple[FigureContractReadError, ...]
    _selection: tuple | None

    @classmethod
    def load(
        cls,
        run_dir: Path,
        *,
        per_step_records: Sequence[Mapping[str, Any]] | None = None,
        current: FigureContractInventory | None = None,
    ) -> FigureContractInventory:
        root = run_dir.resolve()
        selection = _selection_identity(per_step_records)
        if current is not None:
            if current.run_dir != root or current._selection != selection:
                raise ValueError("figure_contract_inventory_scope_mismatch")
            return current
        paths = tuple(figure_contract_paths(root, per_step_records=per_step_records))
        snapshots: list[FigureContractSnapshot] = []
        errors: list[FigureContractReadError] = []
        for path in paths:
            if not path.resolve().is_relative_to(root):
                errors.append(
                    FigureContractReadError(
                        path,
                        "figure_contract_outside_run",
                        "contract resolves outside the run directory",
                    )
                )
                continue
            try:
                content = path.read_bytes()
            except OSError as exc:
                errors.append(
                    FigureContractReadError(
                        path, "figure_contract_unreadable", str(exc)
                    )
                )
                continue
            try:
                raw = json.loads(content)
            except (ValueError, UnicodeError) as exc:
                errors.append(
                    FigureContractReadError(
                        path, "figure_contract_invalid_json", str(exc)
                    )
                )
                continue
            try:
                if not isinstance(raw, dict):
                    raise ValueError("contract must be a JSON object")
                panels = raw.get("panels", [])
                if not isinstance(panels, list) or any(
                    not isinstance(panel, dict) for panel in panels
                ):
                    raise ValueError("panels must be a list of JSON objects")
                payload = freeze_payload(raw)
            except ValueError as exc:
                errors.append(
                    FigureContractReadError(
                        path, "figure_contract_invalid_shape", str(exc)
                    )
                )
                continue
            snapshots.append(
                FigureContractSnapshot(
                    path,
                    figure_contract_tier(path, root),
                    sha256(content).hexdigest(),
                    payload,
                )
            )
        return cls(root, paths, tuple(snapshots), tuple(errors), selection)

    def error_messages(self) -> list[str]:
        return [error.message(self.run_dir) for error in self.errors]

    def panel_projections(self) -> list[dict[str, Any]]:
        panels = []
        for snapshot in self.snapshots:
            raw = snapshot.to_payload()
            for panel in raw.get("panels", []):
                panels.append(
                    {
                        **panel,
                        "_contract_path": str(snapshot.path),
                        "_figure_id": str(raw.get("figure_id") or snapshot.path.stem),
                        "_primary_publication_contract": snapshot.tier
                        == "primary_publication",
                    }
                )
        return panels

    def texts(self, *, allowed_step_ids: set[str] | None = None) -> tuple[str, ...]:
        return tuple(
            figure_contract_text(snapshot.to_payload())
            for snapshot in self.snapshots
            if allowed_step_ids is None
            or (
                snapshot.tier == "supporting_step"
                and snapshot.path.parents[1].name in allowed_step_ids
            )
        )


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
    metadata = (
        panel.get("metadata") if isinstance(panel.get("metadata"), Mapping) else {}
    )
    explicit = (
        str(
            panel.get("chart_type")
            or panel.get("visual_form")
            or metadata.get("chart_type")
            or metadata.get("visual_form")
            or ""
        )
        .strip()
        .lower()
    )
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
