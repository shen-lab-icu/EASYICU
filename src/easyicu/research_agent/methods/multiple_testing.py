"""Family-aware multiple-testing correction (O22).

The extractor reads registered table/statistic evidence, admits only exact
raw-p field names, and applies Benjamini–Hochberg and Bonferroni corrections
*within declared hypothesis families*. A simple legacy table without family
metadata may form one clearly labelled source-local family. Ambiguous
coefficient dumps and heterogeneous untyped tables are omitted rather than
silently turned into a scientific hypothesis family.

The module never rewrites source artefacts or mutates evidence records. It
only produces a :class:`MultipleTestingReport` for the pipeline to persist.
"""

from __future__ import annotations

import csv
import json
import math
import mmap
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

from ..authority.runtime_artifacts import verified_run_evidence_path


# Exact, normalised names for an *unadjusted* p-value.  Substring matching is
# intentionally forbidden: ``group_value`` contains ``p_val`` and
# ``p_value_bounded`` is a reporting flag, not a p-value.
_PVALUE_COLUMN_NAMES = frozenset(
    {
        "p",
        "p_val",
        "p_value",
        "p_raw",
        "pval",
        "pvalue",
        "primary_p_value",
        "raw_p",
        "raw_p_value",
        "unadjusted_p",
        "unadjusted_p_value",
    }
)

_FAMILY_COLUMN_NAMES = (
    "hypothesis_family_id",
    "hypothesis_family",
    "multiplicity_family_id",
    "multiplicity_family",
    "family_id",
)

_EXPLICIT_FAMILY_SCOPE_COLUMN_NAMES = (
    "hypothesis_family_scope",
    "family_scope",
)

_COEFFICIENT_TERM_COLUMNS = (
    "term",
    "variable",
    "predictor",
    "parameter",
    "feature",
    "covariate",
)

_COEFFICIENT_MARKERS = frozenset(
    {
        "coefficient",
        "estimate",
        "hazard_ratio",
        "log_estimate",
        "odds_ratio",
        "standard_error",
        "std_error",
        "se",
        "ci_low",
        "ci_high",
    }
)

_NON_HYPOTHESIS_TERM_ROLES = frozenset(
    {
        "adjuster",
        "adjustment",
        "adjustment_covariate",
        "availability",
        "baseline_covariate",
        "confounder",
        "covariate",
        "intercept",
        "model_intercept",
        "nuisance",
        "offset",
        "random_effect",
    }
)

_NON_HYPOTHESIS_ANALYSIS_ROLES = frozenset(
    {
        "audit",
        "data_quality",
        "diagnostic",
        "quality_control",
        "sensitivity",
        "sensitivity_analysis",
    }
)

_IDENTITY_COLUMNS = (
    "hypothesis_id",
    "test_id",
    "contrast_id",
    "comparison_id",
    "estimand_id",
    "candidate_id",
)

_IDENTITY_CONTEXT_COLUMNS = (
    "outcome",
    "endpoint",
    "model_id",
    "analysis_id",
    "analysis_set",
    "spec_id",
    "specification_id",
    "subgroup",
    "population",
    "cohort",
    "time_window",
    "term",
    "contrast",
    "comparison",
    "estimate_type",
    "alternative",
)

# For cells stored as strings that embed ``p=0.031`` or ``p<0.001`` we
# also scan the value text. The regex is liberal but capped at 12
# characters to avoid surprising matches.
_INLINE_P_RE = re.compile(r"\bp\s*[=<>]\s*([0-9.eE\-]{1,12})")
_ASCII_WORD_BYTES = frozenset(
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
)
_ASCII_WHITESPACE_BYTES = frozenset(b" \t\n\r\v\f")
_INLINE_P_OPERATOR_BYTES = frozenset(b"=<>")
_INLINE_P_NUMBER_BYTES = frozenset(b"0123456789.eE-")


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
    family_id: Optional[str] = None
    family_source: Optional[str] = None  # "declared" or "source-local"
    hypothesis_key: Optional[str] = None
    model_id: Optional[str] = None
    outcome: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "p_value": self.p_value,
            "evidence_id": self.evidence_id,
            "artefact_path": self.artefact_path,
            "source": self.source,
            "column": self.column,
            "row_index": self.row_index,
            "label": self.label,
            "family_id": self.family_id,
            "family_source": self.family_source,
            "hypothesis_key": self.hypothesis_key,
            "model_id": self.model_id,
            "outcome": self.outcome,
        }


@dataclass
class MultipleTestingReport:
    """Result of family-scoped BH and Bonferroni corrections."""

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

    @property
    def family_ids(self) -> List[str]:
        return sorted({str(record.family_id) for record in self.records if record.family_id})

    @property
    def n_families(self) -> int:
        return len(self.family_ids)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "n_tests": 0,
                "n_families": 0,
                "family_sizes": {},
                "alpha": self.alpha,
                "n_significant_raw": 0,
                "n_significant_bh": 0,
                "n_significant_bonferroni": 0,
                "min_p_raw": None,
                "min_p_bh": None,
                "notes": list(self.notes),
            }
        family_sizes = {
            family_id: sum(1 for record in self.records if record.family_id == family_id)
            for family_id in self.family_ids
        }
        return {
            "n_tests": self.n_tests,
            "n_families": self.n_families,
            "family_sizes": family_sizes,
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
                    "hypothesis_family",
                    "family_source",
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
                        rec.family_id or "",
                        rec.family_source or "",
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
            (
                "**Multiplicity policy:** BH and Bonferroni corrections were "
                "computed independently within declared or source-local "
                f"hypothesis families; alpha = {self.alpha:.3f}"
            ),
            f"**Hypothesis families represented:** {s['n_families']}",
            f"**Total tests observed:** {s['n_tests']}",
            f"**Significant (raw p ≤ {self.alpha:.3f}):** {s['n_significant_raw']}",
            f"**Significant after BH (FDR):** {s['n_significant_bh']}",
            f"**Significant after Bonferroni:** {s['n_significant_bonferroni']}",
        ]
        if s["min_p_raw"] is not None:
            lines.append(f"**Min raw p:** {s['min_p_raw']:.3g}")
        if s["min_p_bh"] is not None:
            lines.append(f"**Min BH-adjusted p:** {s['min_p_bh']:.3g}")
        if s["family_sizes"]:
            lines += ["", "## Family summary", "", "| family | n tests |", "|---|---:|"]
            for family_id, family_size in s["family_sizes"].items():
                lines.append(f"| {family_id} | {family_size} |")
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
                "| family | evidence | column | row | label | p_raw | p_BH | p_Bonferroni |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for rec, pbh, pbon in zip(
                self.records, self.bh_adjusted, self.bonferroni_adjusted
            ):
                lines.append(
                    "| {family} | {evid} | {col} | {row} | {lab} | {p:.3g} | {pbh:.3g} | {pbon:.3g} |".format(
                        family=rec.family_id or "",
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
    return _normalise_name(name) in _PVALUE_COLUMN_NAMES


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _normalised_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_normalise_name(key): value for key, value in row.items()}


def _first_value(row: Dict[str, Any], names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = _clean_string(row.get(name))
        if value is not None:
            return value
    return None


def _declared_family(row: Dict[str, Any]) -> Optional[str]:
    value = _first_value(row, _FAMILY_COLUMN_NAMES)
    if value is None:
        return None
    family_id = f"declared:{_normalise_name(value)}"
    # An explicit family id is the scientific authority.  Model/specification
    # columns describe where a result came from; they must not silently split a
    # prespecified family that intentionally spans models.  Callers that reuse a
    # local family label may opt into disambiguation with an explicit scope.
    scope = _first_value(row, _EXPLICIT_FAMILY_SCOPE_COLUMN_NAMES)
    if scope is not None:
        family_id += f"::scope:{_normalise_name(scope)}"
    return family_id


def _is_structured_coefficient(row_or_columns: Dict[str, Any] | Sequence[str]) -> bool:
    if isinstance(row_or_columns, dict):
        names = set(row_or_columns)
    else:
        names = {_normalise_name(name) for name in row_or_columns}
    if not names.intersection(_COEFFICIENT_TERM_COLUMNS):
        return False
    return bool(
        names & _COEFFICIENT_MARKERS
        or names & {"analysis_role", "model_id", "source_variable", "term_role"}
    )


def _row_is_nonhypothesis(row: Dict[str, Any]) -> bool:
    term_role = _normalise_name(row.get("term_role"))
    if term_role in _NON_HYPOTHESIS_TERM_ROLES:
        return True
    analysis_role = _normalise_name(row.get("analysis_role"))
    if analysis_role in _NON_HYPOTHESIS_ANALYSIS_ROLES:
        return True
    for role_column in ("record_role", "record_type", "result_role", "row_role"):
        if _normalise_name(row.get(role_column)) in _NON_HYPOTHESIS_ANALYSIS_ROLES:
            return True
    coefficient_term = _first_value(row, _COEFFICIENT_TERM_COLUMNS)
    return _normalise_name(coefficient_term) in {
        "const",
        "constant",
        "intercept",
        "model_intercept",
    }


def _hypothesis_identity(row: Dict[str, Any]) -> Optional[str]:
    primary_name: Optional[str] = None
    primary_value: Optional[str] = None
    for name in _IDENTITY_COLUMNS:
        value = _clean_string(row.get(name))
        if value is not None:
            primary_name, primary_value = name, value
            break
    if primary_value is None:
        for name in (
            *_COEFFICIENT_TERM_COLUMNS,
            "test_name",
            "comparison",
            "contrast",
        ):
            value = _clean_string(row.get(name))
            if value is not None:
                primary_name, primary_value = name, value
                break
    if primary_name is None or primary_value is None:
        return None
    parts = [f"{primary_name}={_normalise_name(primary_value)}"]
    for name in _IDENTITY_CONTEXT_COLUMNS:
        if name == primary_name:
            continue
        value = _clean_string(row.get(name))
        if value is not None:
            parts.append(f"{name}={_normalise_name(value)}")
    return "|".join(parts)


def _label_from_row(row: Dict[str, Any], pvalue_column: str) -> Optional[str]:
    normalised = _normalised_row(row)
    preferred = _first_value(
        normalised,
        (
            "hypothesis_id",
            "test_id",
            "contrast_id",
            "comparison",
            "term",
            "variable",
            "predictor",
            "parameter",
            "feature",
            "covariate",
            "outcome",
            "endpoint",
            "label",
            "name",
        ),
    )
    if preferred is not None:
        return preferred
    return _first_label_in_row(row, pvalue_column)


def _record_from_value(
    *,
    p_value: float,
    evidence_id: str,
    artefact_path: str,
    source: str,
    column: str,
    metadata: Dict[str, Any],
    label: Optional[str],
    row_index: Optional[int] = None,
) -> PValueRecord:
    family_id = _declared_family(metadata)
    return PValueRecord(
        p_value=p_value,
        evidence_id=evidence_id,
        artefact_path=artefact_path,
        source=source,
        column=column,
        row_index=row_index,
        label=label,
        family_id=family_id,
        family_source="declared" if family_id else None,
        hypothesis_key=_hypothesis_identity(metadata),
        model_id=_first_value(metadata, ("model_id", "analysis_id")),
        outcome=_first_value(metadata, ("outcome", "endpoint", "target_outcome")),
    )


def _assign_source_local_family(
    records: List[PValueRecord], *, artefact_path: str, notes: List[str]
) -> List[PValueRecord]:
    untyped = [record for record in records if record.family_id is None]
    if not untyped:
        return records
    models = {record.model_id for record in untyped if record.model_id}
    outcomes = {record.outcome for record in untyped if record.outcome}
    if len(models) > 1 or len(outcomes) > 1:
        notes.append(
            f"{artefact_path}: omitted {len(untyped)} untyped p-values spanning "
            "multiple models or outcomes; declare hypothesis_family_id explicitly."
        )
        return [record for record in records if record.family_id is not None]
    source_family = f"source-local:{artefact_path}"
    for record in untyped:
        record.family_id = source_family
        record.family_source = "source-local"
    notes.append(
        f"{artefact_path}: treated {len(untyped)} legacy untyped p-value(s) as "
        "one source-local family."
    )
    return records


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


def _mapped_file_may_contain_inline_p(fh: TextIO) -> bool:
    """Cheaply detect whether a CSV may contain an inline ``p=...`` cell.

    The exact p-value columns are handled from the header before this helper is
    called.  For the legacy arbitrary-column inline syntax, a read-only mapping
    lets the platform's native byte search reject large numeric files without
    constructing a Python dictionary for every row.  ASCII candidates mirror
    the required prefix of :data:`_INLINE_P_RE`.  Bytes whose Unicode boundary
    or whitespace role is ambiguous conservatively return ``True`` so the
    existing text/CSV parser remains the authority; false positives cost time
    but cannot change extracted results.
    """
    try:
        payload = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    except ValueError:
        # ``mmap`` rejects empty files, but ValueError is not guaranteed to
        # mean empty on every platform/filesystem.  Only return a definitive
        # negative when fstat proves zero bytes; otherwise preserve semantics
        # by falling back to the row parser.
        try:
            return os.fstat(fh.fileno()).st_size != 0
        except (AttributeError, OSError, ValueError):
            return True
    except (AttributeError, OSError):
        # Some virtual filesystems do not support mapping.  Falling back to the
        # existing row parser preserves extraction semantics on those systems.
        return True

    with payload:
        size = len(payload)
        position = payload.find(b"p")
        while position >= 0:
            if position:
                previous = payload[position - 1]
                if previous >= 128:
                    return True
                if previous in _ASCII_WORD_BYTES:
                    position = payload.find(b"p", position + 1)
                    continue

            cursor = position + 1
            while cursor < size and payload[cursor] in _ASCII_WHITESPACE_BYTES:
                cursor += 1
            if cursor < size and payload[cursor] >= 128:
                return True
            if cursor >= size or payload[cursor] not in _INLINE_P_OPERATOR_BYTES:
                position = payload.find(b"p", position + 1)
                continue

            cursor += 1
            while cursor < size and payload[cursor] in _ASCII_WHITESPACE_BYTES:
                cursor += 1
            if cursor < size and payload[cursor] >= 128:
                return True
            if cursor < size and payload[cursor] in _INLINE_P_NUMBER_BYTES:
                return True
            position = payload.find(b"p", position + 1)
    return False


def _extract_pvalues_from_csv(
    *, csv_path: Path, evidence_id: str, artefact_path: str
) -> Tuple[List[PValueRecord], List[str]]:
    """Read raw p-values from a CSV while preserving family metadata."""
    records: List[PValueRecord] = []
    notes: List[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return records, notes
            pvalue_columns = [c for c in reader.fieldnames if _is_pvalue_column(c)]
            coefficient_table = _is_structured_coefficient(reader.fieldnames)
            if not pvalue_columns:
                # Structured coefficient tables deliberately disable inline
                # narrative extraction.  With no raw-p column there is
                # therefore nothing this source can contribute.
                if coefficient_table:
                    return records, notes
                # Preserve arbitrary-column inline ``p=...`` support without
                # paying DictReader + per-row normalisation cost for large
                # numeric artefacts that contain no such token.
                if not _mapped_file_may_contain_inline_p(fh):
                    return records, notes
                fh.seek(0)
                reader = csv.DictReader(fh)
            omitted_untyped_coefficients = 0
            for row_idx, row in enumerate(reader):
                metadata = _normalised_row(row)
                if _row_is_nonhypothesis(metadata):
                    continue
                family_id = _declared_family(metadata)
                for col in pvalue_columns:
                    p = _safe_float(row.get(col))
                    if p is None:
                        continue
                    if coefficient_table and family_id is None:
                        omitted_untyped_coefficients += 1
                        continue
                    records.append(
                        _record_from_value(
                            p_value=p,
                            evidence_id=evidence_id,
                            artefact_path=artefact_path,
                            source="column",
                            column=col,
                            row_index=row_idx,
                            label=_label_from_row(row, col),
                            metadata=metadata,
                        )
                    )
                # Inline ``p=...`` scan for narrative cells.
                if pvalue_columns or coefficient_table:
                    continue
                for col, raw_val in row.items():
                    if not isinstance(raw_val, str):
                        continue
                    for match in _INLINE_P_RE.finditer(raw_val):
                        p = _safe_float(match.group(1))
                        if p is None:
                            continue
                        records.append(
                            _record_from_value(
                                p_value=p,
                                evidence_id=evidence_id,
                                artefact_path=artefact_path,
                                source="inline",
                                column=col,
                                row_index=row_idx,
                                label=raw_val.strip()[:60],
                                metadata=metadata,
                            )
                        )
            if omitted_untyped_coefficients:
                notes.append(
                    f"{artefact_path}: omitted {omitted_untyped_coefficients} p-value(s) "
                    "from a structured coefficient table without declared family metadata."
                )
    except FileNotFoundError:
        notes.append(f"{artefact_path}: file not found; skipped.")
    except Exception as exc:
        notes.append(f"{artefact_path}: unreadable ({type(exc).__name__}: {exc}); skipped.")
    return _assign_source_local_family(
        records, artefact_path=artefact_path, notes=notes
    ), notes


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

    omitted_untyped_coefficients = 0

    def _walk(node: Any, path: str = "", inherited: Optional[Dict[str, Any]] = None) -> None:
        nonlocal omitted_untyped_coefficients
        if isinstance(node, dict):
            metadata = dict(inherited or {})
            metadata.update(
                {
                    _normalise_name(key): value
                    for key, value in node.items()
                    if not isinstance(value, (dict, list))
                }
            )
            for key, value in node.items():
                key_path = f"{path}.{key}" if path else str(key)
                if _is_pvalue_column(key):
                    p = _safe_float(value)
                    if p is not None:
                        if _row_is_nonhypothesis(metadata):
                            continue
                        if _is_structured_coefficient(metadata) and not _declared_family(
                            metadata
                        ):
                            omitted_untyped_coefficients += 1
                            continue
                        records.append(
                            _record_from_value(
                                p_value=p,
                                evidence_id=evidence_id,
                                artefact_path=artefact_path,
                                source="column",
                                column=key_path,
                                label=_first_value(
                                    metadata,
                                    ("test_id", "term", "outcome", "endpoint", "label"),
                                ),
                                metadata=metadata,
                            )
                        )
                _walk(value, key_path, metadata)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _walk(item, f"{path}[{i}]", inherited)

    _walk(payload)
    if omitted_untyped_coefficients:
        notes.append(
            f"{artefact_path}: omitted {omitted_untyped_coefficients} p-value(s) "
            "from structured coefficient objects without declared family metadata."
        )
    return _assign_source_local_family(
        records, artefact_path=artefact_path, notes=notes
    ), notes


def _iter_pvalue_sources(
    *,
    evidence_records: Iterable[Any],
    run_dir: Path,
    active_evidence_ids: Optional[Iterable[str]] = None,
) -> Iterable[Tuple[Path, str, str]]:
    """Yield ``(path_on_disk, evidence_id, relative_path)`` triples to scan."""
    active_ids = (
        None
        if active_evidence_ids is None
        else {str(evidence_id) for evidence_id in active_evidence_ids}
    )
    for rec in evidence_records:
        kind = getattr(rec, "kind", None)
        if kind not in {"table", "statistic"}:
            continue
        evidence_id = str(getattr(rec, "evidence_id", "") or "")
        produced_by_step = getattr(rec, "produced_by_step", None)
        if (
            active_ids is not None
            and produced_by_step
            and evidence_id not in active_ids
        ):
            # Resumed runs retain immutable evidence copies from the superseded
            # step execution.  Only records referenced by the current step
            # checkpoint belong in the current multiplicity denominator.
            continue
        if _normalise_name(evidence_id).startswith("multiple_testing_report"):
            # A resumed run may already have an O22 report registered. Never
            # treat the correction report's own p_raw column as new hypotheses.
            continue
        rel_path = getattr(rec, "relative_path", None)
        if not rel_path:
            continue
        abs_path = verified_run_evidence_path(run_dir, rec)
        if abs_path is None:
            continue
        yield abs_path, evidence_id, str(rel_path)


def _declared_families_compatible(left: PValueRecord, right: PValueRecord) -> bool:
    if left.family_source != "declared" or right.family_source != "declared":
        return True
    return left.family_id == right.family_id


def _deduplicate_records(records: Sequence[PValueRecord]) -> Tuple[List[PValueRecord], int]:
    """Collapse exact semantic duplicates without merging conflicting families."""
    deduplicated: List[PValueRecord] = []
    semantic_buckets: Dict[Tuple[str, str], List[int]] = {}
    exact_seen: Dict[Tuple[Any, ...], int] = {}
    duplicate_count = 0

    for record in records:
        if record.hypothesis_key:
            semantic_key = (record.hypothesis_key, record.p_value.hex())
            matched_index: Optional[int] = None
            for candidate_index in semantic_buckets.get(semantic_key, []):
                if _declared_families_compatible(
                    deduplicated[candidate_index], record
                ):
                    matched_index = candidate_index
                    break
            if matched_index is not None:
                duplicate_count += 1
                if (
                    deduplicated[matched_index].family_source != "declared"
                    and record.family_source == "declared"
                ):
                    deduplicated[matched_index] = record
                continue
            semantic_buckets.setdefault(semantic_key, []).append(len(deduplicated))
            deduplicated.append(record)
            continue

        exact_key = (
            record.artefact_path,
            record.column,
            record.row_index,
            record.label,
            record.p_value.hex(),
        )
        if exact_key in exact_seen:
            duplicate_count += 1
            continue
        exact_seen[exact_key] = len(deduplicated)
        deduplicated.append(record)

    return deduplicated, duplicate_count


def _adjust_within_families(
    records: Sequence[PValueRecord],
) -> Tuple[List[float], List[float]]:
    bh_adjusted = [0.0] * len(records)
    bonferroni_adjusted = [0.0] * len(records)
    family_indices: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        if record.family_id is None:
            continue
        family_indices.setdefault(record.family_id, []).append(index)
    for indices in family_indices.values():
        p_values = [records[index].p_value for index in indices]
        family_bh = _benjamini_hochberg(p_values)
        family_bonferroni = _bonferroni(p_values)
        for index, adjusted_bh, adjusted_bonferroni in zip(
            indices, family_bh, family_bonferroni
        ):
            bh_adjusted[index] = adjusted_bh
            bonferroni_adjusted[index] = adjusted_bonferroni
    return bh_adjusted, bonferroni_adjusted


def build_multiple_testing_report(
    *,
    evidence_records: Iterable[Any],
    run_dir: Path,
    alpha: float = 0.05,
    active_evidence_ids: Optional[Iterable[str]] = None,
) -> MultipleTestingReport:
    """Collect auditable p-values and correct each hypothesis family.

    ``evidence_records`` is the current ``EvidenceStore.records()``
    output. ``run_dir`` is the pipeline's per-run directory; evidence
    relative paths resolve underneath ``<run_dir>/evidence/``.
    """
    records: List[PValueRecord] = []
    notes: List[str] = []
    for abs_path, ev_id, rel_path in _iter_pvalue_sources(
        evidence_records=evidence_records,
        run_dir=run_dir,
        active_evidence_ids=active_evidence_ids,
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

    records, duplicate_count = _deduplicate_records(records)
    if duplicate_count:
        notes.append(
            f"Collapsed {duplicate_count} duplicate p-value representation(s) "
            "using matching hypothesis identity, family metadata, and raw value."
        )
    bh, bon = _adjust_within_families(records)
    if not records:
        notes.append(
            "No p-values with a defensible hypothesis-family scope were "
            "observed in the registered evidence. Either the plan avoided hypothesis "
            "testing, or family metadata was absent from structured results."
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
