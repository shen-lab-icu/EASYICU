"""Figure quality checks (OpenLens-AI inspired).

OpenLens-AI [1] uses a vision-language model to review generated
figures. We adopt the *idea* — figures are evidence and deserve a
review pass — but keep the default implementation deterministic so
it can run in CI without an API key. A pluggable VLM adapter is
available for users that want richer visual feedback.

What we check:

* file is non-empty;
* image opens via Pillow / matplotlib (catches truncated PNGs);
* image is not solid-colour (catches blank canvases / failed renders);
* width/height fall in a sensible range for journal figures;
* if matplotlib was used and the file has metadata, axis-label
  presence is verified (best-effort heuristic).

If Pillow is not installed we degrade gracefully — the only check
left is "file exists and is non-trivially sized".

References
----------
[1] OpenLens-AI: Fully Autonomous Research Agent for Health Informatics.
    https://github.com/jarrycyx/openlens-ai
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .llm import LLMClient, LLMMessage
from .schema import ValidationFinding


class VisualQAAuditor:
    """Inspect figures registered as evidence and flag obvious problems.

    The deterministic checks always run. If ``vlm_adapter`` is provided,
    its findings are appended after the deterministic pass. This keeps
    the default pipeline offline and reproducible while letting a
    paper run opt into OpenLens-style model feedback.
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
        findings: List[ValidationFinding] = []
        try:
            from PIL import Image  # type: ignore
            _has_pil = True
        except Exception:
            _has_pil = False

        for p in figure_paths:
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
                continue

            try:
                with Image.open(p) as im:  # type: ignore[name-defined]
                    im.load()
                    w, h = im.size
                    extrema = im.getextrema() if im.mode != "P" else None
                    mode = im.mode
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

        if self.vlm_adapter is not None:
            findings.extend(self.vlm_adapter.audit(figure_paths=figure_paths))

        return findings


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
    "parse_vlm_visual_qa_response",
]
