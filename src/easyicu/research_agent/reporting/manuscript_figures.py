"""Reader projection of registered figure exports and their exact contracts.

The caller supplies current evidence membership. This owner verifies bytes,
selects one export, and projects an existing legend; it computes no statistics
and never discovers manuscript authority by walking mutable step directories.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal, InvalidOperation
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from ..authority.evidence_store import evidence_artifact_basename_stem
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..figures.contracts import (
    ArticleDisplayPolicyError, ArticleDisplayPolicyRequest, decide_article_display,
)
from ..figures.publication import FigureContract
from ..schema import ValidationFinding


class ManuscriptFigureProjectionError(ValueError):
    """A reader figure cannot be bound to one unchanged source contract."""


@dataclass(frozen=True)
class ManuscriptFigure:
    evidence_id: str
    relative_path: str
    caption: str
    placement: str
    figure_sha256: str
    contract_evidence_id: str | None = None
    contract_sha256: str | None = None


@dataclass(frozen=True)
class ManuscriptFigures:
    figures: tuple[ManuscriptFigure, ...]
    omitted_evidence_ids: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]
    context_notes: tuple[dict[str, str], ...] = ()

    def as_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "easyicu.manuscript_figure_projection/1",
            "figures": [asdict(figure) for figure in self.figures],
            "omitted_evidence_ids": list(self.omitted_evidence_ids),
            "findings": [finding.model_dump(mode="json") for finding in self.findings],
            "context_notes": list(self.context_notes),
        }


def _owner(record: Any) -> tuple[Any, ...]:
    return tuple(getattr(record, key, None) for key in (
        "producer", "generation_mode", "produced_by_step",
    ))


def _stem(record: Any) -> str:
    return evidence_artifact_basename_stem(Path(record.relative_path), record.evidence_id)


def select_figure_exports(records: Sequence[Any], *, prefer_png: bool = False) -> tuple[list[Any], tuple[str, ...]]:
    """Choose PDF then PNG within one owner, never across different steps."""
    priority = {".pdf": int(prefer_png), ".png": int(not prefer_png)}
    selected: dict[tuple[Any, ...], tuple[int, int, Any]] = {}
    unsupported: dict[tuple[Any, ...], str] = {}
    for index, record in enumerate(records):
        if record.kind != "figure" or not record.relative_path:
            continue
        key = (*_owner(record), _stem(record))
        suffix = Path(record.relative_path).suffix.lower()
        if suffix not in priority:
            unsupported.setdefault(key, record.evidence_id)
            continue
        candidate = (priority[suffix], index, record)
        if key not in selected or candidate[:2] < selected[key][:2]:
            selected[key] = candidate
    return (
        [row[2] for row in sorted(selected.values(), key=lambda row: row[1])],
        tuple(identifier for key, identifier in unsupported.items() if key not in selected),
    )


def _contract_record(figure: Any, contracts: Sequence[Any]) -> Any | None:
    metadata = getattr(figure, "metadata", None) or {}
    linked = metadata.get("contract_evidence_id") or metadata.get("figure_contract")
    if linked:
        candidates = [record for record in contracts if record.evidence_id == linked]
        if len(candidates) != 1 or linked not in (getattr(figure, "inputs", None) or []):
            raise ManuscriptFigureProjectionError("Figure's explicit contract link is unavailable")
        if _owner(candidates[0]) != _owner(figure):
            raise ManuscriptFigureProjectionError("Figure and contract have different owners")
        return candidates[0]
    candidates = [record for record in contracts
                  if _owner(record) == _owner(figure)
                  and _stem(record).removesuffix(".figure_contract") == _stem(figure)]
    if len(candidates) > 1:
        raise ManuscriptFigureProjectionError("Figure contract membership is ambiguous")
    return candidates[0] if candidates else None


def _read_contract(record: Any, root: Path) -> FigureContract:
    path = verified_run_evidence_path(root, record)
    if path is None:
        raise ManuscriptFigureProjectionError("Figure contract bytes could not be verified")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ManuscriptFigureProjectionError("Figure contract could not be verified again") from exc
    if sha256(payload).hexdigest() != record.sha256:
        raise ManuscriptFigureProjectionError("Figure contract changed after it was verified")
    try:
        return FigureContract.model_validate(json.loads(payload))
    except (ValueError, TypeError) as exc:
        raise ManuscriptFigureProjectionError("Registered figure contract is invalid") from exc


def _placement(contract: FigureContract) -> str:
    placements = []
    for panel in contract.panels:
        placement = panel.metadata.get("placement")
        if placement is None:
            # Legacy contracts use the same typed policy owner as the producer;
            # never classify scientific content from a filename or figure title.
            try:
                placement = decide_article_display(ArticleDisplayPolicyRequest(
                    article_role=panel.role,
                )).placement
            except ArticleDisplayPolicyError as exc:
                raise ManuscriptFigureProjectionError(str(exc)) from exc
        if placement not in {"main", "supplementary"}:
            raise ManuscriptFigureProjectionError("Figure contract has an invalid placement")
        placements.append(placement)
    # A composite cannot be split without redrawing it. A main panel keeps that
    # whole display in main; a routine all-audit display stays supplementary.
    return "main" if "main" in placements else "supplementary"


def _is_primary(record: Any) -> bool:
    return _owner(record) == ("publication_figure_skill", "deterministic_figure_skill", None)


def _denominator_note(figure, contract, records, root):
    """A verified single accounting count is prose, not a result figure.

    Never infer this from the image, its title, or a study identifier. A mixed
    panel or a full attrition ledger still retains its figure representation.
    """
    if contract is None or len(contract.panels) != 1:
        return None
    panel = contract.panels[0]
    if panel.role != "cohort_accounting" or panel.metadata.get("accounting_completeness") != "analysis_denominator_only":
        return None
    names = panel.metadata.get("source_data") or []
    candidates = [r for r in records if r.kind == "table" and _owner(r) == _owner(figure)
                  and _stem(r) + Path(r.relative_path).suffix in names]
    if len(candidates) != 1:
        raise ManuscriptFigureProjectionError("Single-denominator display has no unique registered source")
    source = candidates[0]
    path = verified_run_evidence_path(root, source)
    if path is None or path.suffix.lower() != ".csv":
        raise ManuscriptFigureProjectionError("Single-denominator source changed")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0]["accounting_completeness"] != "analysis_denominator_only":
            raise ValueError("Expected one accounting row")
        count = Decimal(rows[0]["n_remaining"])
        if not count.is_finite() or count < 0 or count != count.to_integral_value():
            raise ValueError("Invalid denominator")
    except (OSError, UnicodeError, csv.Error, KeyError, ValueError, InvalidOperation) as exc:
        raise ManuscriptFigureProjectionError("Single-denominator source is invalid") from exc
    return {
        "text": f"The recorded analysis denominator is {int(count):,} records. "
                "The source ledger contains only one stage; earlier eligibility stages "
                "and exclusions cannot be reconstructed from this ledger.",
        "source_evidence_id": source.evidence_id,
        "source_sha256": source.sha256,
        "figure_evidence_id": figure.evidence_id,
        "reason_code": "SINGLE_DENOMINATOR_AS_TEXT",
    }


def build_manuscript_figures(
    *, evidence_records: Sequence[Any], run_dir: Path, prefer_png: bool = False,
) -> ManuscriptFigures:
    """Project captions and placement from current digest-verified contracts."""
    versions: dict[tuple[Any, ...], set[tuple[str, Any]]] = {}
    for record in evidence_records:
        if record.kind == "figure" and Path(record.relative_path).suffix.lower() in {".pdf", ".png"}:
            metadata = getattr(record, "metadata", None) or {}
            key = (*_owner(record), _stem(record), Path(record.relative_path).suffix.lower())
            versions.setdefault(key, set()).add((
                record.sha256, metadata.get("contract_evidence_id") or metadata.get("figure_contract"),
            ))
    if any(len(choices) > 1 for choices in versions.values()):
        raise ManuscriptFigureProjectionError("Current figure export versions are ambiguous")
    selected, omitted = select_figure_exports(evidence_records, prefer_png=prefer_png)
    contracts = [record for record in evidence_records
                 if record.kind == "log" and record.relative_path.endswith(".figure_contract.json")]
    resolved: list[tuple[Any, Any | None, FigureContract | None]] = []
    canonical_paths: dict[str, str] = {}
    for figure in selected:
        image_path = verified_run_evidence_path(run_dir, figure)
        if image_path is None:
            raise ManuscriptFigureProjectionError("Figure image bytes could not be verified")
        canonical_paths[figure.evidence_id] = image_path.relative_to(run_dir.resolve()).as_posix()
        record = _contract_record(figure, contracts)
        resolved.append((figure, record, _read_contract(record, run_dir) if record else None))

    # Promotion copies bytes and explicitly records the source export. Shared
    # roles alone do not prove duplication: they may describe different results.
    promoted = [figure for figure, record, _contract in resolved
                if record is not None and _is_primary(figure)]
    findings: list[ValidationFinding] = []
    figures: list[ManuscriptFigure] = []
    context_notes: list[dict[str, str]] = []
    for figure, record, contract in sorted(resolved, key=lambda row: not _is_primary(row[0])):
        if not _is_primary(figure) and any(
            figure.evidence_id in parent.inputs and figure.sha256 == parent.sha256
            for parent in promoted
        ):
            continue
        note = _denominator_note(figure, contract, evidence_records, run_dir)
        if note is not None:
            context_notes.append(note)
            continue
        caption = contract.reader_caption if contract is not None else None
        if not caption:
            reason = ("MANUSCRIPT_FIGURE_CAPTION_MISSING" if contract
                      else "MANUSCRIPT_FIGURE_CONTRACT_MISSING")
            findings.append(ValidationFinding(
                validator="manuscript_figure_projection", severity="error",
                message="A reader figure has no source-bound explanatory legend.",
                evidence_ids=[figure.evidence_id], detail={"reason_code": reason},
            ))
            caption = "Source-bound explanatory legend unavailable; figure requires review."
        figures.append(ManuscriptFigure(
            evidence_id=figure.evidence_id, relative_path=canonical_paths[figure.evidence_id],
            caption=caption, placement=_placement(contract) if contract else "supplementary",
            figure_sha256=figure.sha256,
            contract_evidence_id=record.evidence_id if record else None,
            contract_sha256=record.sha256 if record else None,
        ))
    return ManuscriptFigures(tuple(figures), omitted, tuple(findings), tuple(context_notes))


def manuscript_figure_receipt_is_current(*, run_dir: Path, evidence_records: Sequence[Any]) -> bool:
    """Supersede earlier legend errors only with an exact current clean receipt."""
    receipts = [record for record in evidence_records
                if record.kind == "log" and _owner(record) == ("pipeline", "system", None)
                and record.metadata.get("artifact_role") == "manuscript_figure_projection"]
    if not receipts:
        return False
    try:
        projection = build_manuscript_figures(evidence_records=evidence_records, run_dir=run_dir)
        if projection.findings or projection.omitted_evidence_ids:
            return False
        for record in receipts:
            path = verified_run_evidence_path(run_dir, record)
            if path is not None:
                payload = path.read_bytes()
                if (sha256(payload).hexdigest() == record.sha256
                        and json.loads(payload) == projection.as_receipt()):
                    return True
    except (OSError, ValueError, TypeError):
        return False
    return False


def register_manuscript_figure_projection(evidence: Any, projection: ManuscriptFigures) -> Any:
    return evidence.register_json(
        kind="log", description="Source-bound reader figure legends and placements.",
        payload=projection.as_receipt(),
        filename="manuscript_figure_projection.json", evidence_id="manuscript_figure_projection",
        inputs=[identifier for figure in projection.figures
                for identifier in (figure.evidence_id, figure.contract_evidence_id) if identifier],
        producer="pipeline", generation_mode="system", on_sha_change="new_id",
        metadata={"artifact_role": "manuscript_figure_projection"},
    )
