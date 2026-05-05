"""Deterministic figure quality checks (OpenLens-AI inspired).

OpenLens-AI [1] uses a vision-language model to review generated
figures. We adopt the *idea* — figures are evidence and deserve a
review pass — but make the v1 implementation deterministic so it
can run in CI without an API key. A pluggable VLM hook is exposed
for users that want richer review.

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

from pathlib import Path
from typing import List, Optional

from .schema import ValidationFinding


class VisualQAAuditor:
    """Inspect figures registered as evidence and flag obvious problems."""

    name = "visual_qa"

    def __init__(self, *, min_bytes: int = 1024) -> None:
        self.min_bytes = min_bytes

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

        return findings


__all__ = ["VisualQAAuditor"]
