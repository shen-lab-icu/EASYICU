"""Publication figure contracts and export helpers.

This module gives the research agent a small, auditable figure-making
surface inspired by the MIT-licensed ``nature-figure`` skill from
Yuan1z0825/nature-skills. We do not vendor that project; instead we
encode the parts that matter for EasyICU:

* every figure starts from a claim and a panel-level evidence chain;
* each panel must carry a distinct role in the argument;
* SVG is the primary export so text remains editable for journals;
* PDF/PNG/TIFF exports are produced from the same matplotlib figure;
* output QA checks for missing files, tiny files and non-editable SVG.

The plotting code itself remains ordinary matplotlib. The contract is
what prevents agent-generated figures from becoming decorative
dashboards with no reviewable scientific logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import ValidationFinding


FigureArchetype = Literal[
    "quantitative_grid",
    "schematic_led_composite",
    "image_plate_plus_quant",
    "asymmetric_mixed_modality",
]

PanelRole = Literal[
    "overview",
    "deviation",
    "relationship",
    "validation",
    "robustness",
    "mechanism",
    "audit",
    "workflow",
]


class PanelSpec(BaseModel):
    """A single panel's job in a publication figure."""

    model_config = ConfigDict(extra="forbid")

    panel_id: str = Field(..., description="Stable panel label, e.g. 'a' or 'b'.")
    title: str = Field(..., description="Short panel title.")
    role: PanelRole = Field(..., description="Panel's unique role in the evidence chain.")
    claim: str = Field(..., description="One sentence this panel supports.")
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="EvidenceStore ids or aliases used to draw this panel.",
    )
    review_risk: Optional[str] = Field(
        default=None,
        description="What a reviewer might challenge about this panel.",
    )

    @field_validator("panel_id")
    @classmethod
    def _panel_id_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("panel_id must be non-empty")
        return value


class FigureContract(BaseModel):
    """Claim-first specification for a manuscript figure."""

    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(..., description="Stable figure id, e.g. 'Figure2'.")
    core_claim: str = Field(..., description="The one-sentence claim the figure defends.")
    archetype: FigureArchetype = "asymmetric_mixed_modality"
    panels: List[PanelSpec] = Field(default_factory=list)
    export_formats: List[Literal["svg", "pdf", "png", "tiff"]] = Field(
        default_factory=lambda: ["svg", "pdf", "png"],
    )
    width_mm: float = Field(default=183.0, gt=0)
    height_mm: float = Field(default=120.0, gt=0)
    source_data: List[str] = Field(default_factory=list)
    statistics_note: Optional[str] = None
    image_integrity_note: Optional[str] = None

    @model_validator(mode="after")
    def _validate_panels(self) -> "FigureContract":
        panel_ids = [p.panel_id for p in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("panel_id values must be unique within a figure")
        if len(self.export_formats) != len(set(self.export_formats)):
            raise ValueError("export_formats must not contain duplicates")
        return self

    def evidence_chain(self) -> Dict[str, List[str]]:
        """Return panel_id -> evidence ids for provenance rendering."""
        return {p.panel_id: list(p.evidence_ids) for p in self.panels}


PALETTE_CLINICAL: Dict[str, str] = {
    "baseline": "#272727",
    "blue": "#0F4D92",
    "blue_soft": "#B4C0E4",
    "teal": "#42949E",
    "orange": "#E28E2C",
    "red": "#B64342",
    "red_soft": "#F6CFCB",
    "neutral": "#8F8F8F",
    "neutral_light": "#D8D8D8",
    "band": "#F3F0EA",
}


# Okabe-Ito colourblind-safe palette (T1.8). Used by the codex-grade
# SOFA2AuditSkill's mock figures so manuscript-ready output matches
# what an external Codex-style agent emits when asked for publication
# figures. Order is stable — generators index into it by panel.
PALETTE_OKABE_ITO: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#6F6F6F",  # neutral grey
    "#F0E442",  # yellow
]


def apply_codex_publication_style(
    *,
    font_size: float = 8.0,
) -> List[str]:
    """Apply the codex-grade publication rcParams and return Okabe-Ito.

    Mirrors the rcParams the standalone codex-led independent analysis
    used (sans-serif Arial 8pt, fonttype 42 for editable PDF/PS,
    no top/right spines, savefig DPI 300). Returns the 8-colour
    Okabe-Ito palette so callers can index into it directly.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1.0,
        "xtick.labelsize": max(font_size - 1.0, 6.0),
        "ytick.labelsize": max(font_size - 1.0, 6.0),
        "legend.fontsize": max(font_size - 1.0, 6.0),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return list(PALETTE_OKABE_ITO)


def make_figure_contract(
    *,
    figure_id: str,
    core_claim: str,
    panels: Sequence[PanelSpec | Mapping[str, object]],
    archetype: FigureArchetype = "asymmetric_mixed_modality",
    export_formats: Sequence[Literal["svg", "pdf", "png", "tiff"]] = ("svg", "pdf", "png"),
    width_mm: float = 183.0,
    height_mm: float = 120.0,
    source_data: Optional[Sequence[str]] = None,
    statistics_note: Optional[str] = None,
    image_integrity_note: Optional[str] = None,
) -> FigureContract:
    """Build and validate a claim-first publication figure contract."""
    parsed_panels = [
        p if isinstance(p, PanelSpec) else PanelSpec.model_validate(dict(p))
        for p in panels
    ]
    return FigureContract(
        figure_id=figure_id,
        core_claim=core_claim,
        archetype=archetype,
        panels=parsed_panels,
        export_formats=list(export_formats),
        width_mm=width_mm,
        height_mm=height_mm,
        source_data=list(source_data or []),
        statistics_note=statistics_note,
        image_integrity_note=image_integrity_note,
    )


def audit_figure_contract(contract: FigureContract) -> List[ValidationFinding]:
    """Check whether a contract is strong enough for manuscript use."""
    findings: List[ValidationFinding] = []
    if not contract.core_claim.strip():
        findings.append(ValidationFinding(
            validator="publication_figure_contract",
            severity="error",
            message=f"{contract.figure_id} has no core claim.",
        ))
    if len(contract.panels) == 0:
        findings.append(ValidationFinding(
            validator="publication_figure_contract",
            severity="error",
            message=f"{contract.figure_id} has no panels.",
        ))
        return findings
    roles = [p.role for p in contract.panels]
    duplicate_roles = sorted({r for r in roles if roles.count(r) > 1})
    if duplicate_roles:
        findings.append(ValidationFinding(
            validator="publication_figure_contract",
            severity="warning",
            message=(
                f"{contract.figure_id} repeats panel role(s): "
                + ", ".join(duplicate_roles)
                + ". Confirm the panels are not redundant."
            ),
            detail={"duplicate_roles": duplicate_roles},
        ))
    no_evidence = [p.panel_id for p in contract.panels if not p.evidence_ids]
    if no_evidence:
        findings.append(ValidationFinding(
            validator="publication_figure_contract",
            severity="warning",
            message=(
                f"{contract.figure_id} has panel(s) without evidence ids: "
                + ", ".join(no_evidence)
            ),
            detail={"panel_ids": no_evidence},
        ))
    if "svg" not in contract.export_formats:
        findings.append(ValidationFinding(
            validator="publication_figure_contract",
            severity="warning",
            message=f"{contract.figure_id} does not request SVG as a primary editable export.",
        ))
    return findings


def apply_publication_style(
    *,
    font_size: float = 7.5,
    axes_linewidth: float = 0.8,
    palette: Optional[Mapping[str, str]] = None,
) -> Mapping[str, str]:
    """Apply Nature-style matplotlib rcParams and return the palette.

    Imports matplotlib lazily so the research-agent package remains cheap
    to import in non-plotting contexts.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": font_size,
        "axes.labelsize": font_size + 0.5,
        "axes.titlesize": font_size + 1.0,
        "xtick.labelsize": max(font_size - 0.5, 5.5),
        "ytick.labelsize": max(font_size - 0.5, 5.5),
        "axes.linewidth": axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return dict(palette or PALETTE_CLINICAL)


def add_panel_label(
    ax: object,
    label: str,
    *,
    x: float = -0.08,
    y: float = 1.08,
    fontsize: float = 11.0,
    color: str = PALETTE_CLINICAL["baseline"],
) -> None:
    """Add a compact Nature-style panel label to a matplotlib Axes."""
    ax.text(  # type: ignore[attr-defined]
        x, y, label,
        transform=ax.transAxes,  # type: ignore[attr-defined]
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


def save_publication_figure(
    fig: object,
    output_stem: str | Path,
    *,
    contract: Optional[FigureContract] = None,
    formats: Optional[Sequence[Literal["svg", "pdf", "png", "tiff"]]] = None,
    dpi: int = 600,
) -> Dict[str, Path]:
    """Save a matplotlib figure in journal-friendly formats.

    Parameters
    ----------
    fig:
        Matplotlib Figure instance.
    output_stem:
        Path without suffix, or a path whose suffix will be ignored.
    contract:
        Optional :class:`FigureContract`. If supplied and ``formats`` is
        omitted, ``contract.export_formats`` decides the outputs.
    formats:
        Explicit format override.
    dpi:
        Raster DPI for PNG/TIFF.
    """
    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    requested = list(formats or (contract.export_formats if contract else ["svg", "pdf", "png"]))
    saved: Dict[str, Path] = {}
    for fmt in requested:
        path = stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)  # type: ignore[attr-defined]
        saved[fmt] = path
    if contract is not None:
        contract_path = stem.with_suffix(".figure_contract.json")
        contract_path.write_text(
            contract.model_dump_json(indent=2),
            encoding="utf-8",
        )
        saved["contract"] = contract_path
    return saved


def audit_publication_exports(
    paths: Mapping[str, Path] | Iterable[Path],
    *,
    min_bytes: int = 1024,
    require_svg_text: bool = True,
) -> List[ValidationFinding]:
    """Audit exported figure files for basic journal-readiness."""
    figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif"}
    if isinstance(paths, Mapping):
        path_list = [
            Path(p) for p in paths.values()
            if Path(p).suffix.lower() in figure_suffixes
        ]
    else:
        path_list = [
            Path(p) for p in paths
            if Path(p).suffix.lower() in figure_suffixes
        ]

    findings: List[ValidationFinding] = []
    for path in path_list:
        if not path.exists():
            findings.append(ValidationFinding(
                validator="publication_figure_export",
                severity="error",
                message=f"Figure export missing: {path}",
            ))
            continue
        size = path.stat().st_size
        if size < min_bytes:
            findings.append(ValidationFinding(
                validator="publication_figure_export",
                severity="warning",
                message=f"Figure export '{path.name}' is suspiciously small ({size} bytes).",
                detail={"path": str(path), "bytes": size},
            ))
        if require_svg_text and path.suffix.lower() == ".svg":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "<text" not in text:
                findings.append(ValidationFinding(
                    validator="publication_figure_export",
                    severity="warning",
                    message=(
                        f"SVG export '{path.name}' does not contain editable <text> nodes. "
                        "Set matplotlib rcParams['svg.fonttype'] = 'none'."
                    ),
                    detail={"path": str(path)},
                ))
    return findings


__all__ = [
    "FigureArchetype",
    "PanelRole",
    "PanelSpec",
    "FigureContract",
    "PALETTE_CLINICAL",
    "make_figure_contract",
    "audit_figure_contract",
    "apply_publication_style",
    "add_panel_label",
    "save_publication_figure",
    "audit_publication_exports",
]
