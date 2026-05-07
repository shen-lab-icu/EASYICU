"""Publication figure contracts and export helpers.

This module is EasyICU's default Nature-compatible figure path for the
research agent. It exposes a small, auditable figure-making surface
with the pieces that matter for EasyICU:

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
import zipfile
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .schema import ValidationFinding


FigureArchetype = Literal[
    "quantitative_grid",
    "schematic_led_composite",
    "image_plate_plus_quant",
    "asymmetric_mixed_modality",
]

ExportFormat = Literal["svg", "pdf", "png", "tiff", "pptx"]

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
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: ["svg", "pdf", "png", "tiff"],
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


def _normalise_statistics_note(
    note: Optional[str | Sequence[str]],
) -> Optional[str]:
    if note is None:
        return None
    if isinstance(note, str):
        text = note.strip()
        return text or None
    lines = [str(item).strip() for item in note if str(item).strip()]
    return "\n".join(lines) if lines else None


def _infer_panel_role(panel: Mapping[str, Any], index: int) -> PanelRole:
    title = str(panel.get("title", "")).lower()
    claim = str(panel.get("claim", "")).lower()
    joined = f"{title} {claim}"
    if any(token in joined for token in ("missing", "audit", "quality", "ascertainment")):
        return "audit"
    if any(token in joined for token in ("validation", "replication", "feasibility", "protocol")):
        return "validation"
    if any(token in joined for token in ("robust", "sensitivity")):
        return "robustness"
    if any(token in joined for token in ("association", "relationship", "odds ratio", "mortality by")):
        return "relationship"
    if any(token in joined for token in ("workflow", "pipeline")):
        return "workflow"
    default_roles: List[PanelRole] = [
        "overview",
        "relationship",
        "audit",
        "validation",
        "robustness",
        "mechanism",
    ]
    return default_roles[min(index, len(default_roles) - 1)]


def _canonical_panel_role(value: object, index: int, panel: Mapping[str, Any]) -> PanelRole:
    if isinstance(value, str):
        role = value.strip()
        if role in {
            "overview",
            "deviation",
            "relationship",
            "validation",
            "robustness",
            "mechanism",
            "audit",
            "workflow",
        }:
            return role  # type: ignore[return-value]
        aliases = {
            "cohort_anchor": "overview",
            "ordinal_severity_gradient": "relationship",
            "distributional_context": "validation",
            "association_forest": "robustness",
            "quality_audit": "audit",
        }
        if role in aliases:
            return aliases[role]  # type: ignore[return-value]
    return _infer_panel_role(panel, index)


def _normalise_panels(
    panels: Sequence[PanelSpec | Mapping[str, object]],
) -> List[PanelSpec]:
    parsed: List[PanelSpec] = []
    for idx, panel in enumerate(panels):
        if isinstance(panel, PanelSpec):
            parsed.append(panel)
            continue
        raw = dict(panel)
        if "panel_id" not in raw and "panel" in raw:
            raw["panel_id"] = raw.pop("panel")
        if "evidence_ids" not in raw and "source_evidence" in raw:
            source = raw.pop("source_evidence")
            if isinstance(source, (list, tuple)):
                raw["evidence_ids"] = [str(item) for item in source]
            elif source in (None, ""):
                raw["evidence_ids"] = []
            else:
                raw["evidence_ids"] = [str(source)]
        raw["role"] = _canonical_panel_role(raw.get("role"), idx, raw)
        parsed.append(PanelSpec.model_validate(raw))
    return parsed


def make_figure_contract(
    payload: Optional[Mapping[str, Any]] = None,
    *,
    figure_id: Optional[str] = None,
    core_claim: Optional[str] = None,
    panels: Optional[Sequence[PanelSpec | Mapping[str, object]]] = None,
    archetype: FigureArchetype = "asymmetric_mixed_modality",
    export_formats: Sequence[ExportFormat] = ("svg", "pdf", "png", "tiff"),
    width_mm: float = 183.0,
    height_mm: float = 120.0,
    source_data: Optional[Sequence[str]] = None,
    statistics_note: Optional[str | Sequence[str]] = None,
    image_integrity_note: Optional[str] = None,
    title: Optional[str] = None,
    claim: Optional[str] = None,
    source_evidence: Optional[Sequence[str] | Mapping[str, str]] = None,
    statistical_notes: Optional[str | Sequence[str]] = None,
    target_outcome: Optional[str] = None,
    cohort: Optional[str] = None,
) -> FigureContract:
    """Build and validate a claim-first publication figure contract.

    Accepts both the native EasyICU schema and a small compatibility
    layer for agent-generated payloads that use aliases such as
    ``claim``/``core_claim`` or ``source_evidence``/``source_data``.
    Extra manuscript metadata such as ``title``, ``target_outcome`` and
    ``cohort`` is ignored here on purpose; it belongs in step summaries,
    not in the strict figure contract model.
    """
    merged: Dict[str, Any] = dict(payload or {})
    explicit = {
        "figure_id": figure_id,
        "core_claim": core_claim,
        "panels": panels,
        "archetype": archetype,
        "export_formats": export_formats,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "source_data": source_data,
        "statistics_note": statistics_note,
        "image_integrity_note": image_integrity_note,
        "title": title,
        "claim": claim,
        "source_evidence": source_evidence,
        "statistical_notes": statistical_notes,
        "target_outcome": target_outcome,
        "cohort": cohort,
    }
    for key, value in explicit.items():
        if value is not None:
            merged[key] = value

    figure_id_value = str(merged.get("figure_id") or merged.get("title") or "").strip()
    core_claim_value = str(merged.get("core_claim") or merged.get("claim") or "").strip()
    if not figure_id_value:
        raise ValueError("figure_id is required")
    if not core_claim_value:
        raise ValueError("core_claim is required")
    raw_panels = merged.get("panels")
    if not raw_panels:
        raise ValueError("panels are required")

    if merged.get("source_data") is not None:
        raw_source_data = list(merged["source_data"])
        source_data_value = []
        for item in raw_source_data:
            if isinstance(item, Mapping):
                path = item.get("path")
                evidence_id = item.get("evidence_id")
                if path not in (None, ""):
                    source_data_value.append(str(path))
                elif evidence_id not in (None, ""):
                    source_data_value.append(str(evidence_id))
            else:
                source_data_value.append(str(item))
    elif isinstance(merged.get("source_evidence"), Mapping):
        source_data_value = [str(v) for v in merged["source_evidence"].values()]
    elif merged.get("source_evidence") is not None:
        source_data_value = [str(v) for v in merged["source_evidence"]]
    else:
        source_data_value = []

    stats_note_value = _normalise_statistics_note(
        merged.get("statistics_note", merged.get("statistical_notes"))
    )
    parsed_panels = _normalise_panels(raw_panels)
    return FigureContract(
        figure_id=figure_id_value,
        core_claim=core_claim_value,
        archetype=merged.get("archetype", archetype),
        panels=parsed_panels,
        export_formats=list(merged.get("export_formats", export_formats)),
        width_mm=float(merged.get("width_mm", width_mm)),
        height_mm=float(merged.get("height_mm", height_mm)),
        source_data=source_data_value,
        statistics_note=stats_note_value,
        image_integrity_note=merged.get("image_integrity_note", image_integrity_note),
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
    x: float = -0.03,
    y: float = 1.02,
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
    output_stem: str | Path | FigureContract | Mapping[str, Any],
    *legacy_args: object,
    contract: Optional[FigureContract] = None,
    figure_contract: Optional[FigureContract] = None,
    output_dir: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    stem: Optional[str] = None,
    filename: Optional[str] = None,
    basename: Optional[str] = None,
    formats: Optional[Sequence[ExportFormat]] = None,
    dpi: int = 300,
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
        Raster DPI for PNG exports. TIFF exports are capped to a
        journal-reasonable 300 DPI and compressed with LZW so agent
        reruns do not silently produce 100MB+ artefacts.
    """
    stem_path: Path
    resolved_contract = figure_contract or contract

    if output_dir is not None or out_dir is not None or stem is not None or filename is not None or basename is not None:
        if isinstance(output_stem, (FigureContract, Mapping)):
            resolved_contract = (
                output_stem
                if isinstance(output_stem, FigureContract)
                else FigureContract.model_validate(output_stem)
            )
            directory = Path(output_dir or out_dir or ".")
            stem_name = stem or filename or basename or (
                resolved_contract.figure_id if isinstance(resolved_contract, FigureContract) else "publication_figure"
            )
            stem_path = directory / stem_name
        else:
            directory = Path(output_dir or out_dir or Path(output_stem).parent)
            stem_name = stem or filename or basename or Path(output_stem).stem
            stem_path = directory / stem_name
    elif isinstance(output_stem, (FigureContract, Mapping)):
        resolved_contract = (
            output_stem
            if isinstance(output_stem, FigureContract)
            else FigureContract.model_validate(output_stem)
        )
        if not legacy_args:
            raise TypeError("legacy save_publication_figure requires an output directory or output stem")
        directory = Path(legacy_args[0])
        if len(legacy_args) >= 2:
            stem_name = str(legacy_args[1])
        elif isinstance(resolved_contract, FigureContract):
            stem_name = resolved_contract.figure_id
        else:
            stem_name = "publication_figure"
        stem_path = directory / stem_name
    else:
        stem_path = Path(output_stem)

    if stem_path.suffix:
        stem_path = stem_path.with_suffix("")
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    requested = list(formats or (resolved_contract.export_formats if resolved_contract else ["svg", "pdf", "png"]))
    saved: Dict[str, Path] = {}
    for fmt in requested:
        path = stem_path.with_suffix(f".{fmt}")
        if fmt == "pptx":
            png_path = saved.get("png")
            cleanup_png = False
            if png_path is None:
                png_path = stem_path.with_suffix(".pptx_source.png")
                fig.savefig(png_path, dpi=dpi, bbox_inches="tight")  # type: ignore[attr-defined]
                cleanup_png = True
            _write_single_image_pptx(
                image_path=png_path,
                pptx_path=path,
                title=resolved_contract.figure_id if resolved_contract else stem_path.name,
            )
            if cleanup_png:
                try:
                    png_path.unlink()
                except Exception:
                    pass
            saved[fmt] = path
            continue
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if fmt == "png":
            kwargs["dpi"] = dpi
        elif fmt == "tiff":
            kwargs["dpi"] = min(int(dpi), 300)
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)  # type: ignore[attr-defined]
        saved[fmt] = path
    if resolved_contract is not None:
        contract_path = stem_path.with_suffix(".figure_contract.json")
        contract_path.write_text(
            resolved_contract.model_dump_json(indent=2),
            encoding="utf-8",
        )
        saved["contract"] = contract_path
    return saved


def _write_single_image_pptx(*, image_path: Path, pptx_path: Path, title: str) -> None:
    """Write a minimal one-slide PPTX with ``image_path`` centred.

    This avoids adding ``python-pptx`` as a hard dependency while still
    giving clinical collaborators an editable PowerPoint container.
    """
    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes = image_path.read_bytes()
    slide_w = 12192000
    slide_h = 6858000
    margin = 457200
    img_w = slide_w - margin * 2
    img_h = slide_h - margin * 2
    image_name = "image1.png"
    with zipfile.ZipFile(pptx_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _pptx_content_types())
        z.writestr("_rels/.rels", _pptx_root_rels())
        z.writestr("ppt/presentation.xml", _pptx_presentation(slide_w, slide_h))
        z.writestr("ppt/_rels/presentation.xml.rels", _pptx_presentation_rels())
        z.writestr("ppt/slides/slide1.xml", _pptx_slide(title, margin, margin, img_w, img_h))
        z.writestr("ppt/slides/_rels/slide1.xml.rels", _pptx_slide_rels(image_name))
        z.writestr(f"ppt/media/{image_name}", image_bytes)


def _pptx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"""


def _pptx_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""


def _pptx_presentation(slide_w: int, slide_h: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
  <p:sldSz cx="{slide_w}" cy="{slide_h}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def _pptx_presentation_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"""


def _pptx_slide(title: str, x: int, y: int, cx: int, cy: int) -> str:
    safe_title = _xml_escape(title)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="2" name="{safe_title}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def _pptx_slide_rels(image_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>
</Relationships>"""


def _xml_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def audit_publication_exports(
    paths: Mapping[str, Path] | Iterable[Path] | str | Path | None = None,
    *,
    output_dir: Optional[str | Path] = None,
    stem: Optional[str] = None,
    min_bytes: int = 1024,
    require_svg_text: bool = True,
) -> List[ValidationFinding]:
    """Audit exported figure files for basic journal-readiness."""
    figure_suffixes = {".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"}
    if paths is None and output_dir is not None:
        paths = output_dir
    if isinstance(paths, (str, Path)):
        base = Path(paths)
        if stem is not None:
            stem_path = base / stem
            path_list = [
                stem_path.with_suffix(suffix)
                for suffix in [".svg", ".pdf", ".png", ".tiff", ".tif", ".pptx"]
                if stem_path.with_suffix(suffix).exists()
            ]
        else:
            path_list = [base]
    elif isinstance(paths, Mapping):
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
            else:
                from .visual_qa import audit_svg_text_layout

                findings.extend(audit_svg_text_layout(
                    path,
                    validator="publication_figure_export",
                ))
    return findings


__all__ = [
    "FigureArchetype",
    "ExportFormat",
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
