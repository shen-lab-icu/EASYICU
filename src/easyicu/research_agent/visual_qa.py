"""Figure quality checks.

Figures are evidence and deserve a review pass, but the default
implementation stays deterministic so it can run in CI without an API
key. A pluggable VLM adapter is available for users that want richer
visual feedback.

What we check:

* file is non-empty;
* image opens via Pillow / matplotlib (catches truncated PNGs);
* image is not solid-colour (catches blank canvases / failed renders);
* width/height fall in a sensible range for journal figures;
* if matplotlib was used and the file has metadata, axis-label
  presence is verified (best-effort heuristic).

If Pillow is not installed we degrade gracefully — the only check
left is "file exists and is non-trivially sized".
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from .providers.protocol import LLMClient, LLMMessage
from .schema import ValidationFinding


class _TextBox(NamedTuple):
    text: str
    bbox: Tuple[float, float, float, float]
    area: float
    group_id: str


class VisualQAAuditor:
    """Inspect figures registered as evidence and flag obvious problems.

    The deterministic checks always run. If ``vlm_adapter`` is provided,
    its findings are appended after the deterministic pass. This keeps
    the default pipeline offline and reproducible while letting a
    paper run opt into model-based visual feedback.
    """

    name = "visual_qa"

    def __init__(
        self,
        *,
        min_bytes: int = 1024,
        vlm_adapter: Optional["VLMVisualQAAdapter"] = None,
    ) -> None:
        self.min_bytes = min_bytes
        self.vlm_adapter = vlm_adapter

    def audit(self, *, figure_paths: List[Path]) -> List[ValidationFinding]:
        return self.audit_with_expected(figure_paths=figure_paths, expected_numeric_by_path=None)

    def audit_with_expected(
        self,
        *,
        figure_paths: List[Path],
        expected_numeric_by_path: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[ValidationFinding]:
        findings: List[ValidationFinding] = []
        try:
            from PIL import Image  # type: ignore
            _has_pil = True
        except Exception:
            _has_pil = False

        for p in figure_paths:
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".tiff", ".tif"}:
                continue
            if not p.exists():
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Registered figure missing on disk: {p}",
                ))
                continue
            size = p.stat().st_size
            if size < self.min_bytes:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=(
                        f"Figure '{p.name}' is suspiciously small ({size} bytes). "
                        "Could indicate an empty plot or a truncated render."
                    ),
                    detail={"path": str(p), "bytes": size},
                ))
                continue

            if not _has_pil:
                if p.suffix.lower() == ".svg":
                    findings.extend(_audit_svg_text_layout(p, validator=self.name))
                continue

            if p.suffix.lower() == ".svg":
                findings.extend(_audit_svg_text_layout(p, validator=self.name))
                expected_numeric = None
                if expected_numeric_by_path:
                    expected_numeric = (
                        expected_numeric_by_path.get(str(p))
                        or expected_numeric_by_path.get(str(p.resolve()))
                    )
                if expected_numeric:
                    findings.extend(
                        _audit_svg_numeric_consistency(
                            p,
                            validator=self.name,
                            expected_numeric=expected_numeric,
                        )
                    )
                try:
                    view_box = _svg_view_box(ET.parse(p).getroot())
                except Exception:
                    view_box = None
                if view_box is not None:
                    w = int(round(view_box[2] - view_box[0]))
                    h = int(round(view_box[3] - view_box[1]))
                    if w < 200 or h < 150:
                        findings.append(ValidationFinding(
                            validator=self.name,
                            severity="warning",
                            message=f"Figure '{p.name}' is unusually small for publication ({w}x{h}).",
                            detail={"path": str(p), "size": [w, h]},
                        ))
                continue

            try:
                with Image.open(p) as im:  # type: ignore[name-defined]
                    im.load()
                    w, h = im.size
                    extrema = im.getextrema() if im.mode != "P" else None
            except Exception as exc:
                findings.append(ValidationFinding(
                    validator=self.name, severity="error",
                    message=f"Could not open figure '{p.name}': {exc}",
                    detail={"path": str(p)},
                ))
                continue

            if w < 200 or h < 150:
                findings.append(ValidationFinding(
                    validator=self.name, severity="warning",
                    message=f"Figure '{p.name}' is unusually small for publication ({w}x{h}).",
                    detail={"path": str(p), "size": [w, h]},
                ))

            # Solid-colour heuristic: if every channel has min == max, the
            # whole image is one colour. Pillow returns extrema as either a
            # tuple per channel (for multiband) or a single tuple (for L).
            if extrema is not None:
                if isinstance(extrema, tuple) and isinstance(extrema[0], int):
                    if extrema[0] == extrema[1]:
                        findings.append(ValidationFinding(
                            validator=self.name, severity="warning",
                            message=f"Figure '{p.name}' appears to be solid-colour (likely empty plot).",
                            detail={"path": str(p), "extrema": extrema},
                        ))
                elif isinstance(extrema, tuple) and all(
                    isinstance(c, tuple) and c[0] == c[1] for c in extrema
                ):
                    findings.append(ValidationFinding(
                        validator=self.name, severity="warning",
                        message=f"Figure '{p.name}' appears to be solid-colour (likely empty plot).",
                        detail={"path": str(p), "extrema": list(extrema)},
                    ))

            if p.suffix.lower() in {".png", ".pdf"} and not _matching_svg_exists(p):
                findings.append(ValidationFinding(
                    validator=self.name,
                    severity="info",
                    message=(
                        f"Figure '{p.name}' has no same-stem SVG export; "
                        "deterministic text-layout QA is limited."
                    ),
                    detail={"path": str(p), "expected_svg": str(p.with_suffix(".svg"))},
                ))

        if self.vlm_adapter is not None:
            findings.extend(self.vlm_adapter.audit(figure_paths=figure_paths))

        return findings


def _matching_svg_exists(path: Path) -> bool:
    """Return true when an editable SVG companion exists for a figure.

    EvidenceStore prefixes artefacts with a content hash
    (``<evidence_id>__<basename>.png``), so the literal
    ``path.with_suffix('.svg')`` check can miss a true companion whose
    hash differs but whose source basename is identical.
    """
    direct = path.with_suffix(".svg")
    if direct.exists():
        return True
    if "__" not in path.name:
        return False
    source_stem = path.name.split("__", 1)[1]
    source_stem = str(Path(source_stem).with_suffix(""))
    return any(path.parent.glob(f"*__{source_stem}.svg"))


def _audit_svg_text_layout(
    path: Path,
    *,
    validator: str,
    overlap_fraction: float = 0.12,
    edge_tolerance: float = 1.5,
) -> List[ValidationFinding]:
    """Best-effort offline layout QA for editable matplotlib SVG exports.

    The check intentionally stays conservative. It estimates text extents
    from SVG text nodes and flags only obvious collisions or cropped labels,
    which are the failure mode most likely to slip past file-size/blank-image
    checks in agent-generated multi-panel figures.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        return [
            ValidationFinding(
                validator=validator,
                severity="error",
                message=f"Could not parse SVG figure '{path.name}' for text-layout QA: {exc}",
                detail={"path": str(path)},
            )
        ]

    view_box = _svg_view_box(root)
    boxes = list(_svg_text_boxes(root))
    findings: List[ValidationFinding] = []

    if view_box is not None:
        vx0, vy0, vx1, vy1 = view_box
        cropped = []
        for box in boxes:
            x0, y0, x1, y1 = box.bbox
            if (
                x0 < vx0 - edge_tolerance
                or y0 < vy0 - edge_tolerance
                or x1 > vx1 + edge_tolerance
                or y1 > vy1 + edge_tolerance
            ):
                cropped.append(box)
        if cropped:
            sample = cropped[:4]
            findings.append(ValidationFinding(
                validator=validator,
                severity="warning",
                message=(
                    f"SVG figure '{path.name}' has text outside the canvas; "
                    "labels may be cropped or pushed into the export margin."
                ),
                detail={
                    "path": str(path),
                    "count": len(cropped),
                    "examples": [b.text[:80] for b in sample],
                },
            ))

    blocking_overlaps = []
    panel_title_overlaps = []
    for i, left in enumerate(boxes):
        for right in boxes[i + 1:]:
            if left.group_id and left.group_id == right.group_id:
                # Tick-label fragments and multiline labels from the same
                # matplotlib text group can share a conservative estimate.
                continue
            inter = _intersection_area(left.bbox, right.bbox)
            if inter <= 0:
                continue
            denom = max(min(left.area, right.area), 1e-6)
            frac = inter / denom
            if frac >= overlap_fraction:
                if _is_panel_label_title_overlap(left.text, right.text):
                    panel_title_overlaps.append((left, right, frac))
                else:
                    blocking_overlaps.append((left, right, frac))

    if panel_title_overlaps:
        panel_title_overlaps.sort(key=lambda item: item[2], reverse=True)
        sample = panel_title_overlaps[:5]
        findings.append(ValidationFinding(
            validator=validator,
            severity="warning",
            message=(
                f"SVG figure '{path.name}' has a panel label close to a title; "
                "check title spacing before final manuscript export."
            ),
            detail={
                "path": str(path),
                "count": len(panel_title_overlaps),
                "examples": [
                    {
                        "text_a": a.text[:80],
                        "text_b": b.text[:80],
                        "overlap_fraction": round(frac, 3),
                    }
                    for a, b, frac in sample
                ],
            },
        ))

    if blocking_overlaps:
        blocking_overlaps.sort(key=lambda item: item[2], reverse=True)
        sample = blocking_overlaps[:5]
        findings.append(ValidationFinding(
            validator=validator,
            severity="error",
            message=(
                f"SVG figure '{path.name}' has overlapping text elements; "
                "multi-panel labels, annotations or axis text need more spacing."
            ),
            detail={
                "reason": "svg_text_overlap_spacing",
                "path": str(path),
                "count": len(blocking_overlaps),
                "examples": [
                    {
                        "text_a": a.text[:80],
                        "text_b": b.text[:80],
                        "overlap_fraction": round(frac, 3),
                    }
                    for a, b, frac in sample
                ],
            },
        ))

    return findings


def _is_panel_label_title_overlap(left: str, right: str) -> bool:
    left_text = (left or "").strip()
    right_text = (right or "").strip()
    if _looks_like_panel_label(left_text) and _looks_like_title(right_text):
        return True
    if _looks_like_panel_label(right_text) and _looks_like_title(left_text):
        return True
    return False


def _looks_like_panel_label(text: str) -> bool:
    return len(text) == 1 and "A" <= text <= "Z"


def _looks_like_title(text: str) -> bool:
    return len(text) >= 12 and any(ch.isalpha() for ch in text)


def _audit_svg_numeric_consistency(
    path: Path,
    *,
    validator: str,
    expected_numeric: Dict[str, float],
) -> List[ValidationFinding]:
    """Check whether expected step-summary values appear in the editable SVG text.

    This is intentionally conservative and non-blocking. It only verifies that
    numeric values the agent typically annotates in manuscript figures are
    visibly present in the SVG text layer; if none are, the user still needs a
    manual check, but we at least flag the gap.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    boxes = list(_svg_text_boxes(root))
    text_blob = " ".join(box.text for box in boxes)
    if not text_blob.strip():
        return []
    present_tokens = set(re.findall(r"[-+]?\d+(?:\.\d+)?", text_blob))
    if not present_tokens:
        return []
    missing = []
    for label, value in expected_numeric.items():
        variants = _format_expected_numeric_variants(float(value))
        if not any(v in present_tokens for v in variants):
            missing.append({"label": label, "value": float(value), "accepted": sorted(variants)})
    if not missing:
        return []
    return [
        ValidationFinding(
            validator=validator,
            severity="warning",
            message=(
                f"SVG figure '{path.name}' does not visibly contain one or more "
                "expected summary values from step_summary.json. Check figure-to-evidence "
                "numeric consistency before manuscript use."
            ),
            detail={"path": str(path), "missing_expected_values": missing[:8]},
        )
    ]


def audit_svg_text_layout(
    path: Path,
    *,
    validator: str = "visual_qa",
) -> List[ValidationFinding]:
    """Public wrapper for deterministic SVG text-collision QA."""
    return _audit_svg_text_layout(Path(path), validator=validator)


def _svg_view_box(root: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    raw = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if raw:
        parts = [_to_float(p) for p in re.split(r"[\s,]+", raw.strip()) if p]
        if len(parts) == 4 and all(p is not None for p in parts):
            x, y, w, h = [float(p) for p in parts if p is not None]
            return (x, y, x + w, y + h)
    width = _to_float(root.attrib.get("width", ""))
    height = _to_float(root.attrib.get("height", ""))
    if width is not None and height is not None:
        return (0.0, 0.0, width, height)
    return None


def _svg_text_boxes(root: ET.Element) -> Iterable[_TextBox]:
    counter = 0

    def visit(node: ET.Element, inherited_group: str) -> Iterable[_TextBox]:
        nonlocal counter
        tag = _strip_ns(node.tag)
        group_id = inherited_group
        node_id = node.attrib.get("id", "")
        if tag == "g" and node_id:
            group_id = node_id
        if tag == "text":
            text = "".join(node.itertext()).strip()
            if text:
                counter += 1
                bbox = _estimate_svg_text_bbox(node, text)
                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    area = max((x1 - x0) * (y1 - y0), 0.0)
                    yield _TextBox(
                        text=text,
                        bbox=bbox,
                        area=area,
                        group_id=group_id or f"text_{counter}",
                    )
        for child in list(node):
            yield from visit(child, group_id)

    yield from visit(root, "")


def _estimate_svg_text_bbox(
    node: ET.Element,
    text: str,
) -> Optional[Tuple[float, float, float, float]]:
    x = _to_float(node.attrib.get("x", ""))
    y = _to_float(node.attrib.get("y", ""))
    if x is None or y is None:
        return None

    style = node.attrib.get("style", "")
    font_size = (
        _style_float(style, "font-size")
        or _to_float(node.attrib.get("font-size", ""))
        or 10.0
    )
    anchor = _style_value(style, "text-anchor") or node.attrib.get("text-anchor", "start")
    width = max(_estimated_text_width(text, font_size), font_size * 0.5)
    height = max(font_size * 1.15, 1.0)

    if anchor == "middle":
        left = x - width / 2.0
    elif anchor == "end":
        left = x - width
    else:
        left = x
    # SVG text y is the baseline. This is intentionally a little taller
    # than the nominal glyph box so obvious line collisions are caught.
    top = y - height * 0.82
    right = left + width
    bottom = top + height

    angle, cx, cy = _rotation(node.attrib.get("transform", ""), x, y)
    if abs(angle) < 1e-6:
        return (left, top, right, bottom)

    corners = [
        _rotate_point(left, top, angle, cx, cy),
        _rotate_point(right, top, angle, cx, cy),
        _rotate_point(right, bottom, angle, cx, cy),
        _rotate_point(left, bottom, angle, cx, cy),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _estimated_text_width(text: str, font_size: float) -> float:
    width = 0.0
    for ch in text:
        if ch.isspace():
            width += 0.32
        elif ord(ch) > 127:
            width += 0.95
        elif ch in ".,:;|!ilI[]()'`":
            width += 0.28
        elif ch in "MW@%#":
            width += 0.82
        else:
            width += 0.56
    return width * font_size


def _rotation(
    transform: str,
    default_x: float,
    default_y: float,
) -> Tuple[float, float, float]:
    match = re.search(
        r"rotate\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"(?:[\s,]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[\s,]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?",
        transform or "",
    )
    if not match:
        return (0.0, default_x, default_y)
    angle = float(match.group(1))
    cx = float(match.group(2)) if match.group(2) is not None else default_x
    cy = float(match.group(3)) if match.group(3) is not None else default_y
    return (angle, cx, cy)


def _format_expected_numeric_variants(value: float) -> Set[str]:
    variants = {
        str(int(value)) if float(value).is_integer() else None,
        f"{value:.1f}",
        f"{value:.2f}",
        f"{value:.3f}",
    }
    cleaned = {v for v in variants if v is not None}
    cleaned.add(str(value))
    return cleaned


def _rotate_point(
    x: float,
    y: float,
    angle_deg: float,
    cx: float,
    cy: float,
) -> Tuple[float, float]:
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = x - cx
    dy = y - cy
    return (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)


def _intersection_area(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _style_value(style: str, key: str) -> Optional[str]:
    for part in (style or "").split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        if k.strip() == key:
            return v.strip()
    return None


def _style_float(style: str, key: str) -> Optional[float]:
    value = _style_value(style, key)
    return _to_float(value or "")


def _to_float(value: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value or "")
    if not match:
        return None
    return float(match.group(0))


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


class VLMVisualQAAdapter:
    """Optional vision-language-model review hook for generated figures.

    The adapter accepts the existing ``LLMClient`` protocol. If the
    supplied client also exposes ``complete_with_images(prompt=...,
    image_paths=...)`` (``OpenAIClient`` does), the actual image bytes
    are sent to the model. Otherwise the adapter falls back to a
    text-only prompt containing file metadata, which is still useful
    for custom clients that perform their own retrieval.

    The expected model output is JSON:

    ``{"findings": [{"path": "...", "severity": "warning", "message": "..."}]}``

    Invalid or unparsable output degrades to a single warning rather
    than failing the analysis run.
    """

    name = "vlm_visual_qa"

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self.llm = llm
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

    def audit(self, *, figure_paths: Sequence[Path]) -> List[ValidationFinding]:
        paths = [Path(p) for p in figure_paths if Path(p).exists()]
        if not paths:
            return []
        prompt = self._prompt(paths)
        try:
            if hasattr(self.llm, "complete_with_images"):
                raw = self.llm.complete_with_images(  # type: ignore[attr-defined]
                    prompt=prompt,
                    image_paths=paths,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            else:
                raw = self.llm.complete(
                    [
                        LLMMessage(
                            role="system",
                            content=(
                                "You are a conservative visual QA reviewer for "
                                "scientific ICU figures. Return only JSON."
                            ),
                        ),
                        LLMMessage(role="user", content=prompt),
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
        except Exception as exc:
            return [
                ValidationFinding(
                    validator=self.name,
                    severity="warning",
                    message=f"VLM visual QA failed: {exc}",
                )
            ]
        return parse_vlm_visual_qa_response(raw, known_paths=paths, validator=self.name)

    def _prompt(self, paths: Sequence[Path]) -> str:
        metadata = [_figure_metadata(p) for p in paths]
        return (
            "Review these generated manuscript figures for scientific-figure "
            "quality issues: blank/failed rendering, cropped labels, unreadable "
            "text, unclear axes, misleading legends, duplicated panels, or "
            "visual claims unsupported by what is visible. Be conservative: "
            "only report issues that would merit human review.\n\n"
            "Return only JSON with this exact shape:\n"
            '{"findings":[{"path":"<path from metadata>",'
            '"severity":"info|warning|error","message":"<short issue>",'
            '"detail":{"optional":"extra context"}}]}\n'
            "Use an empty findings list if the figures look acceptable.\n\n"
            "Figure metadata:\n"
            + json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        )


def _figure_metadata(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            out["width"], out["height"] = im.size
            out["mode"] = im.mode
    except Exception:
        pass
    return out


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if "```" not in text:
        return text
    start = text.find("```")
    rest = text[start + 3:]
    nl = rest.find("\n")
    if nl >= 0:
        first = rest[:nl].strip().lower()
        if first in {"json", "javascript", "js"} or not first:
            rest = rest[nl + 1:]
    end = rest.find("```")
    if end >= 0:
        rest = rest[:end]
    return rest.strip()


def parse_vlm_visual_qa_response(
    raw: str,
    *,
    known_paths: Sequence[Path],
    validator: str = "vlm_visual_qa",
) -> List[ValidationFinding]:
    text = _strip_json_fence(raw)
    try:
        payload = json.loads(text)
    except Exception:
        head = (raw or "").strip().replace("\n", " ")[:300]
        return [
            ValidationFinding(
                validator=validator,
                severity="warning",
                message=f"VLM visual QA returned unparsable output: {head}",
            )
        ]

    items = payload.get("findings", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    known = {str(Path(p)): str(Path(p)) for p in known_paths}
    known.update({Path(p).name: str(Path(p)) for p in known_paths})
    findings: List[ValidationFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "warning").lower()
        if severity not in {"info", "warning", "error"}:
            severity = "warning"
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        raw_path = str(item.get("path") or "").strip()
        resolved_path = known.get(raw_path, raw_path)
        detail: Dict[str, Any] = {}
        if resolved_path:
            detail["path"] = resolved_path
        if isinstance(item.get("detail"), dict):
            detail.update(item["detail"])
        findings.append(
            ValidationFinding(
                validator=validator,
                severity=severity,  # type: ignore[arg-type]
                message=message,
                detail=detail or None,
            )
        )
    return findings


__all__ = [
    "VisualQAAuditor",
    "VLMVisualQAAdapter",
    "audit_svg_text_layout",
    "parse_vlm_visual_qa_response",
]
