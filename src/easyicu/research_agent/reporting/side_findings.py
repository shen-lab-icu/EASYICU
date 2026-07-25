"""[Layer 4: Evidence & Provenance] Side-finding archive helpers.

Side findings are appendix-only observations produced during execution.
They are deliberately outside the main manuscript evidence namespace: they are
archived for transparency, excluded from writer digests, and blocked if their
text leaks into the manuscript.

Non-goals:
- Do not route side findings back into the writer.
- Do not promote side findings into primary analyses without replanning.
- Do not register side-finding numbers as NumericClaim entries.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


SIDE_FINDINGS_FILENAME = "side_findings.md"


@dataclass(frozen=True)
class SideFinding:
    finding_id: str
    step_id: str
    title: str
    description: str
    n: int | None = None
    related_concept: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SideFinding":
        n_raw = data.get("n")
        try:
            n = int(n_raw) if n_raw is not None else None
        except (TypeError, ValueError):
            n = None
        concept = data.get("related_concept")
        return cls(
            finding_id=str(data.get("finding_id") or "").strip(),
            step_id=str(data.get("step_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            n=n,
            related_concept=str(concept).strip() if concept else None,
        )


def collect_side_findings(per_step_records: Sequence[Dict[str, Any]]) -> List[SideFinding]:
    findings: List[SideFinding] = []
    for record in per_step_records:
        step_id = str(record.get("step_id") or "")
        candidates = record.get("side_findings")
        if candidates is None and isinstance(record.get("step_summary"), dict):
            candidates = record["step_summary"].get("side_findings")
        if not isinstance(candidates, list):
            continue
        for idx, raw in enumerate(candidates):
            if not isinstance(raw, dict):
                continue
            payload = dict(raw)
            payload.setdefault("step_id", step_id)
            payload.setdefault("finding_id", f"{step_id}_side_{idx + 1}")
            finding = SideFinding.from_dict(payload)
            if finding.title or finding.description:
                findings.append(finding)
    return findings


def render_side_findings_md(findings: Sequence[SideFinding]) -> str:
    lines = [
        "# Side Findings (Appendix, advisory only)",
        "",
        "These observations were recorded during execution but are NOT part of the",
        "primary or robustness analyses, and are NOT cited in the manuscript.",
        "",
    ]
    if not findings:
        lines.append("No side findings recorded.")
        return "\n".join(lines) + "\n"
    for finding in findings:
        lines.append(f"## {finding.finding_id} (step={finding.step_id})")
        lines.append("")
        if finding.title:
            lines.append(f"**{finding.title}**")
            lines.append("")
        if finding.description:
            lines.append(finding.description.strip())
            lines.append("")
        meta = []
        if finding.n is not None:
            meta.append(f"n={finding.n}")
        if finding.related_concept:
            meta.append(f"related_concept={finding.related_concept}")
        if meta:
            lines.append("_" + "; ".join(meta) + "_")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_side_findings(
    *,
    run_dir: Path,
    findings: Sequence[SideFinding],
    evidence: Any,
    prompt_pack_version: str | None,
) -> tuple[Path, str]:
    path = run_dir / SIDE_FINDINGS_FILENAME
    text = render_side_findings_md(findings)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence.register_file(
        kind="log",
        description="Appendix-only side findings excluded from the manuscript writer digest.",
        source_path=path,
        evidence_id="side_findings",
        aliases=["side_findings"],
        producer="pipeline",
        generation_mode="system",
        prompt_pack_version=prompt_pack_version,
        on_sha_change="new_id",
    )
    return path, digest


def side_finding_leaks(
    manuscript: str,
    findings: Sequence[SideFinding],
) -> List[SideFinding]:
    haystack = manuscript or ""
    leaks: List[SideFinding] = []
    for finding in findings:
        snippets = _leak_snippets(finding)
        if any(re.search(re.escape(snippet), haystack, flags=re.IGNORECASE) for snippet in snippets):
            leaks.append(finding)
    return leaks


def annotate_side_finding_leaks(
    manuscript: str,
    leaks: Sequence[SideFinding],
) -> str:
    out = manuscript
    for finding in leaks:
        marker = f"<!-- SIDE_FINDING_LEAK:{finding.finding_id} -->"
        if marker not in out:
            out = out.rstrip() + f"\n{marker}\n"
    return out


def _leak_snippets(finding: SideFinding) -> List[str]:
    snippets: List[str] = []
    for text in (finding.title, finding.description):
        normal = " ".join((text or "").split())
        if len(normal) >= 40:
            snippets.append(normal[:120])
        elif normal:
            snippets.append(normal)
    return snippets


__all__ = [
    "SIDE_FINDINGS_FILENAME",
    "SideFinding",
    "annotate_side_finding_leaks",
    "collect_side_findings",
    "render_side_findings_md",
    "side_finding_leaks",
    "write_side_findings",
]
