"""Multiple-testing correction (O22).

Scans every per-step artefact produced by the research-agent run for
reported p-values, applies a single Benjamini–Hochberg FDR correction
across the whole family, and writes a ``multiple_testing_report.csv``
+ ``multiple_testing_report.md`` that the manuscript scaffold can cite.

Design:

* **One family per run.** The default policy is to treat every
  hypothesis test the agent executed in a run as members of a single
  family. This is the most conservative default (highest adjusted
  p-values), and it matches the reviewer reading of "how many tests
  did this paper do, and did you correct for them?".
* **Deterministic input extraction.** We scan registered evidence of
  kind ``table`` and ``statistic`` for columns whose name contains
  ``p_value`` / ``pvalue`` / ``p_val`` / ``pval`` (case-insensitive)
  plus a ``p=`` pattern in CSV cell strings, then pool every finite
  ``p ∈ [0, 1]`` into a flat vector. Non-finite / out-of-range values
  are dropped and surfaced as info findings.
* **Pure stdlib.** The BH implementation is ~15 lines of numpy; no
  statsmodels dependency. When ``numpy`` is unavailable the function
  still runs via a pure-Python fallback.
* **No rewriting of original artefacts.** We only *add* a report. The
  original per-step tables are untouched, so existing provenance and
  validators are unaffected. The report registers as evidence with a
  stable id ``multiple_testing_report`` so manuscript scaffolds can
  cite it via ``{evidence:multiple_testing_report}``.

The module does not mutate evidence records in place. It returns a
:class:`MultipleTestingReport` dataclass and lets the pipeline decide
where to write it.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Any CSV column containing one of these substrings is treated as a
# p-value column. Kept narrow on purpose — "p" alone matches too much.
_PVALUE_COLUMN_PATTERNS = (
    "p_value",
    "pvalue",
    "p_val",
    "pval",
    "p-value",
)

# For cells stored as strings that embed ``p=0.031`` or ``p<0.001`` we
# also scan the value text. The regex is liberal but capped at 12
# characters to avoid surprising matches.
_INLINE_P_RE = re.compile(r"\bp\s*[=<>]\s*([0-9.eE\-]{1,12})")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PValueRecord:
    """A single p-value observed in the run, with full provenance."""

    p_value: float
    evidence_id: str
    artefact_path: str
    source: str  # "column" or "inline"
    column: Optional[str] = None
    row_index: Optional[int] = None
    label: Optional[str] = None  # first non-numeric column value on the row, if any

    def to_json(self) -> Dict[str, Any]:
        return {
            "p_value": self.p_value,
            "evidence_id": self.evidence_id,
            "artefact_path": self.artefact_path,
            "source": self.source,
            "column": self.column,
            "row_index": self.row_index,
            "label": self.label,
        }


@dataclass
class MultipleTestingReport:
    """Result of a run-wide BH correction."""

    records: List[PValueRecord] = field(default_factory=list)
    bh_adjusted: List[float] = field(default_factory=list)
    bonferroni_adjusted: List[float] = field(default_factory=list)
    alpha: float = 0.05
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    @property
    def n_tests(self) -> int:
        return len(self.records)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "n_tests": 0,
                "alpha": self.alpha,
                "n_significant_raw": 0,
                "n_significant_bh": 0,
                "n_significant_bonferroni": 0,
                "min_p_raw": None,
                "min_p_bh": None,
                "notes": list(self.notes),
            }
        return {
            "n_tests": self.n_tests,
            "alpha": self.alpha,
            "n_significant_raw": sum(1 for p in (r.p_value for r in self.records) if p <= self.alpha),
            "n_significant_bh": sum(1 for q in self.bh_adjusted if q <= self.alpha),
            "n_significant_bonferroni": sum(
                1 for q in self.bonferroni_adjusted if q <= self.alpha
            ),
            "min_p_raw": min(r.p_value for r in self.records),
            "min_p_bh": min(self.bh_adjusted) if self.bh_adjusted else None,
            "notes": list(self.notes),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "evidence_id",
                    "artefact_path",
                    "column",
                    "row_index",
                    "label",
                    "source",
                    "p_raw",
                    "p_bh",
                    "p_bonferroni",
                    "significant_at_alpha_bh",
                ]
            )
            for rec, pbh, pbon in zip(
                self.records, self.bh_adjusted, self.bonferroni_adjusted
            ):
                writer.writerow(
                    [
                        rec.evidence_id,
                        rec.artefact_path,
                        rec.column or "",
                        rec.row_index if rec.row_index is not None else "",
                        rec.label or "",
                        rec.source,
                        rec.p_value,
                        pbh,
                        pbon,
                        "yes" if pbh <= self.alpha else "no",
                    ]
                )
        return path

    def write_markdown(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        lines = [
            "# Multiple-testing correction (O22)",
            "",
            f"**Family-wise policy:** single run-wide family; alpha = {self.alpha:.3f}",
            f"**Total tests observed:** {s['n_tests']}",
            f"**Significant (raw p ≤ {self.alpha:.3f}):** {s['n_significant_raw']}",
            f"**Significant after BH (FDR):** {s['n_significant_bh']}",
            f"**Significant after Bonferroni:** {s['n_significant_bonferroni']}",
        ]
        if s["min_p_raw"] is not None:
            lines.append(f"**Min raw p:** {s['min_p_raw']:.3g}")
        if s["min_p_bh"] is not None:
            lines.append(f"**Min BH-adjusted p:** {s['min_p_bh']:.3g}")
        if self.notes:
            lines += ["", "## Notes", ""]
            for note in self.notes:
                lines.append(f"- {note}")
        if not self.records:
            lines += [
                "",
                "No p-values were registered in this run. If the analysis plan",
                "deliberately avoided hypothesis testing, document that in",
                "Methods. Otherwise the agent may have reported effect sizes",
                "without p-values; consider adding them for auditability.",
            ]
        else:
            lines += [
                "",
                "## Test-level detail",
                "",
                "| evidence | column | row | label | p_raw | p_BH | p_Bonferroni |",
                "|---|---|---|---|---|---|---|",
            ]
            for rec, pbh, pbon in zip(
                self.records, self.bh_adjusted, self.bonferroni_adjusted
            ):
                lines.append(
                    "| {evid} | {col} | {row} | {lab} | {p:.3g} | {pbh:.3g} | {pbon:.3g} |".format(
                        evid=rec.evidence_id,
                        col=rec.column or "",
                        row=rec.row_index if rec.row_index is not None else "",
                        lab=(rec.label or "")[:40],
                        p=rec.p_value,
                        pbh=pbh,
                        pbon=pbon,
                    )
                )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# BH / Bonferroni
# ---------------------------------------------------------------------------


def _benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """Return BH-adjusted p-values in the *original* input order.

    Implements the standard ``q_i = min_{k >= i} (n * p_{(k)} / k)``
    form (monotonic from the top down). Pure Python so the module has
    no numpy dependency; for hundreds of p-values this is well under a
    millisecond.
    """
    n = len(pvals)
    if n == 0:
        return []
    # indexed_sorted: list of (orig_index, p) sorted by p ascending
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted: List[Tuple[int, float]] = []
    running_min = 1.0
    # Walk from largest p to smallest so we can apply the monotone cap.
    for rank_from_top, (orig_idx, p) in enumerate(reversed(indexed)):
        # rank in 1..n where 1 is the smallest p
        rank = n - rank_from_top
        q = min(1.0, p * n / rank)
        running_min = min(running_min, q)
        adjusted.append((orig_idx, running_min))
    # Reorder to original input positions
    out = [0.0] * n
    for orig_idx, q in adjusted:
        out[orig_idx] = q
    return out


def _bonferroni(pvals: Sequence[float]) -> List[float]:
    n = len(pvals)
    if n == 0:
        return []
    return [min(1.0, p * n) for p in pvals]


# ---------------------------------------------------------------------------
# P-value extraction
# ---------------------------------------------------------------------------


def _is_pvalue_column(name: str) -> bool:
    name_l = str(name).lower()
    return any(pat in name_l for pat in _PVALUE_COLUMN_PATTERNS)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or not math.isfinite(v):
        return None
    if v < 0 or v > 1:
        return None
    return v


def _first_label_in_row(row: Dict[str, Any], pvalue_column: str) -> Optional[str]:
    """Return the first string cell value other than the p-value column."""
    for col, val in row.items():
        if col == pvalue_column:
            continue
        if isinstance(val, str) and val.strip():
            # Only treat as label if not parseable as a plain number.
            try:
                float(val)
            except (TypeError, ValueError):
                return val.strip()
    return None


def _extract_pvalues_from_csv(
    *, csv_path: Path, evidence_id: str, artefact_path: str
) -> Tuple[List[PValueRecord], List[str]]:
    """Read a CSV file and collect every p-value we can find."""
    records: List[PValueRecord] = []
    notes: List[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return records, notes
            pvalue_columns = [c for c in reader.fieldnames if _is_pvalue_column(c)]
            for row_idx, row in enumerate(reader):
                for col in pvalue_columns:
                    p = _safe_float(row.get(col))
                    if p is None:
                        continue
                    records.append(
                        PValueRecord(
                            p_value=p,
                            evidence_id=evidence_id,
                            artefact_path=artefact_path,
                            source="column",
                            column=col,
                            row_index=row_idx,
                            label=_first_label_in_row(row, col),
                        )
                    )
                # Inline ``p=...`` scan for narrative cells.
                for col, raw_val in row.items():
                    if col in pvalue_columns:
                        continue
                    if not isinstance(raw_val, str):
                        continue
                    for match in _INLINE_P_RE.finditer(raw_val):
                        p = _safe_float(match.group(1))
                        if p is None:
                            continue
                        records.append(
                            PValueRecord(
                                p_value=p,
                                evidence_id=evidence_id,
                                artefact_path=artefact_path,
                                source="inline",
                                column=col,
                                row_index=row_idx,
                                label=raw_val.strip()[:60],
                            )
                        )
    except FileNotFoundError:
        notes.append(f"{artefact_path}: file not found; skipped.")
    except Exception as exc:
        notes.append(f"{artefact_path}: unreadable ({type(exc).__name__}: {exc}); skipped.")
    return records, notes


def _extract_pvalues_from_json(
    *, json_path: Path, evidence_id: str, artefact_path: str
) -> Tuple[List[PValueRecord], List[str]]:
    records: List[PValueRecord] = []
    notes: List[str] = []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        notes.append(f"{artefact_path}: file not found; skipped.")
        return records, notes
    except Exception as exc:
        notes.append(
            f"{artefact_path}: unreadable json ({type(exc).__name__}: {exc}); skipped."
        )
        return records, notes

    def _walk(node: Any, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_path = f"{path}.{key}" if path else str(key)
                if _is_pvalue_column(key):
                    p = _safe_float(value)
                    if p is not None:
                        records.append(
                            PValueRecord(
                                p_value=p,
                                evidence_id=evidence_id,
                                artefact_path=artefact_path,
                                source="column",
                                column=key_path,
                            )
                        )
                _walk(value, key_path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]")

    _walk(payload)
    return records, notes


def _iter_pvalue_sources(
    *, evidence_records: Iterable[Any], run_dir: Path
) -> Iterable[Tuple[Path, str, str]]:
    """Yield ``(path_on_disk, evidence_id, relative_path)`` triples to scan."""
    for rec in evidence_records:
        kind = getattr(rec, "kind", None)
        if kind not in {"table", "statistic"}:
            continue
        rel_path = getattr(rec, "relative_path", None)
        if not rel_path:
            continue
        # Evidence relative paths are stored relative to the evidence
        # directory (``evidence/``), which lives inside the run dir.
        candidates = [
            run_dir / "evidence" / rel_path,
            run_dir / rel_path,
        ]
        abs_path = next((c for c in candidates if c.exists()), None)
        if abs_path is None:
            continue
        yield abs_path, rec.evidence_id, str(rel_path)


def build_multiple_testing_report(
    *,
    evidence_records: Iterable[Any],
    run_dir: Path,
    alpha: float = 0.05,
) -> MultipleTestingReport:
    """Collect every p-value in the run and BH/Bonferroni-adjust them.

    ``evidence_records`` is the current ``EvidenceStore.records()``
    output. ``run_dir`` is the pipeline's per-run directory; evidence
    relative paths resolve underneath ``<run_dir>/evidence/``.
    """
    records: List[PValueRecord] = []
    notes: List[str] = []
    for abs_path, ev_id, rel_path in _iter_pvalue_sources(
        evidence_records=evidence_records, run_dir=run_dir
    ):
        suffix = abs_path.suffix.lower()
        if suffix == ".csv":
            recs, ns = _extract_pvalues_from_csv(
                csv_path=abs_path, evidence_id=ev_id, artefact_path=rel_path
            )
        elif suffix == ".json":
            recs, ns = _extract_pvalues_from_json(
                json_path=abs_path, evidence_id=ev_id, artefact_path=rel_path
            )
        else:
            continue
        records.extend(recs)
        notes.extend(ns)

    pvals = [r.p_value for r in records]
    bh = _benjamini_hochberg(pvals)
    bon = _bonferroni(pvals)
    if not records:
        notes.append(
            "No p-values were observed in the registered evidence. "
            "Either the plan avoided hypothesis testing, or p-values were "
            "reported only in free-form prose (not recommended for audit)."
        )
    return MultipleTestingReport(
        records=records,
        bh_adjusted=bh,
        bonferroni_adjusted=bon,
        alpha=float(alpha),
        notes=notes,
    )


__all__ = [
    "MultipleTestingReport",
    "PValueRecord",
    "build_multiple_testing_report",
]
