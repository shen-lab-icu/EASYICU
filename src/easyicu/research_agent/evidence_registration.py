"""Success-only evidence promotion for the execute control plane.

``EvidenceStore`` remains the sole durable evidence and alias authority.  This
module is only the workflow seam through which an eligible sealed historical
bundle is promoted.  It deliberately does not inspect clinical/statistical
findings, choose aliases, write checkpoints, or retain a second notion of
``current``.

The caller must first finish validation and seal the result bundle.  Promotion
then checks the caller-supplied attempt index before delegating to the store's
validated batch-publication API.  Cross-file physical atomicity remains a
separate EvidenceStore concern; current authority still depends on the terminal
step checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple


class _EvidencePromotionStore(Protocol):
    """Narrow EvidenceStore surface required for success promotion."""

    def records(self) -> Sequence[Any]: ...

    def aliases(self) -> Mapping[str, str]: ...

    def publish_step_success_aliases(
        self,
        bindings: Mapping[str, Sequence[str]],
        *,
        step_id: str,
        suppressed_basename_evidence_ids: Sequence[str] = (),
    ) -> Dict[str, Dict[str, str]]: ...


@dataclass(frozen=True)
class EvidencePromotionResult:
    """Mechanical result of one successful current-authority promotion."""

    published_aliases: Dict[str, Dict[str, str]]
    retained_cross_step_aliases: Dict[str, str]
    suppressed_basename_evidence_ids: Set[str]


def _record_field(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def filter_success_alias_bindings(
    bindings: Mapping[str, Sequence[str]],
    *,
    existing_aliases: Mapping[str, str],
    owners_by_evidence_id: Mapping[str, str],
    step_id: str,
    records_by_evidence_id: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Set[str]]:
    """Keep cross-step product aliases on their existing authority.

    A figure step may legitimately repeat a parent analysis role such as
    ``primary_association``. Same-step retries may replace their own aliases,
    but a child must not steal the parent's semantic authority merely because
    both products mention that role.
    """

    filtered: Dict[str, List[str]] = {}
    retained: Dict[str, str] = {}
    for evidence_id, aliases in bindings.items():
        accepted: List[str] = []
        for alias in aliases:
            alias = str(alias).strip()
            if not alias:
                continue
            existing_id = str(existing_aliases.get(alias) or "").strip()
            existing_owner = str(owners_by_evidence_id.get(existing_id) or "").strip()
            if existing_id and existing_id != evidence_id and existing_owner != step_id:
                retained[alias] = existing_id
                continue
            accepted.append(alias)
        filtered[str(evidence_id)] = list(dict.fromkeys(accepted))
    records = records_by_evidence_id or {}

    def _record_source_name(evidence_id: str) -> str:
        record = records.get(evidence_id)
        relative_path = str(_record_field(record, "relative_path") or "")
        name = Path(relative_path).name
        prefix = f"{evidence_id}__"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def _is_product_authority(evidence_id: str) -> bool:
        record = records.get(evidence_id)
        kind = str(_record_field(record, "kind") or "").lower()
        source_name = _record_source_name(evidence_id)
        if kind in {"table", "figure"}:
            return True
        return kind == "statistic" and source_name != "step_summary.json"

    implicit_basename_aliases = {
        evidence_id: Path(_record_source_name(evidence_id)).stem
        for evidence_id in filtered
        if _record_source_name(evidence_id)
    }
    alias_claimants: Dict[str, List[str]] = {}
    for evidence_id, aliases in filtered.items():
        for alias in aliases:
            alias_claimants.setdefault(alias, []).append(evidence_id)
        basename_alias = implicit_basename_aliases.get(evidence_id)
        if basename_alias:
            alias_claimants.setdefault(basename_alias, []).append(evidence_id)
    suppressed_basename_evidence_ids: Set[str] = set()
    for alias, claimants in alias_claimants.items():
        unique_claimants = list(dict.fromkeys(claimants))
        if len(unique_claimants) <= 1:
            continue
        product_claimants = [
            evidence_id
            for evidence_id in unique_claimants
            if _is_product_authority(evidence_id)
        ]
        selected_product: Optional[str] = None
        if len(product_claimants) == 1:
            selected_product = product_claimants[0]
        elif len(product_claimants) > 1:
            product_sources = {
                evidence_id: _record_source_name(evidence_id)
                for evidence_id in product_claimants
            }
            figure_claimants = [
                evidence_id
                for evidence_id in product_claimants
                if str(_record_field(records.get(evidence_id), "kind") or "").lower()
                == "figure"
            ]
            stems = {
                Path(product_sources[evidence_id]).stem
                for evidence_id in figure_claimants
            }
            # PNG/SVG/PDF exports with one stem are formats of the same logical
            # figure, not competing scientific products. Prefer the editable
            # vector authority deterministically. Distinct real products keep
            # their duplicate claims so EvidenceStore fails closed.
            if len(figure_claimants) == len(product_claimants) and len(stems) == 1:
                format_rank = {
                    ".svg": 0,
                    ".pdf": 1,
                    ".png": 2,
                    ".tiff": 3,
                    ".tif": 3,
                }
                ranked = sorted(
                    product_claimants,
                    key=lambda evidence_id: (
                        format_rank.get(
                            Path(product_sources[evidence_id]).suffix.lower(), 99
                        ),
                        evidence_id,
                    ),
                )
                best_rank = format_rank.get(
                    Path(product_sources[ranked[0]]).suffix.lower(), 99
                )
                if (
                    sum(
                        format_rank.get(
                            Path(product_sources[evidence_id]).suffix.lower(), 99
                        )
                        == best_rank
                        for evidence_id in ranked
                    )
                    == 1
                ):
                    selected_product = ranked[0]
        if selected_product is None:
            continue
        for evidence_id in unique_claimants:
            if evidence_id != selected_product:
                filtered[evidence_id] = [
                    candidate
                    for candidate in filtered[evidence_id]
                    if candidate != alias
                ]
                if implicit_basename_aliases.get(evidence_id) == alias:
                    suppressed_basename_evidence_ids.add(evidence_id)
    return filtered, retained, suppressed_basename_evidence_ids


class EvidenceRegistrar:
    """Promote one validated attempt through the sole EvidenceStore."""

    __slots__ = ("_evidence_store",)

    def __init__(self, evidence_store: _EvidencePromotionStore) -> None:
        self._evidence_store = evidence_store

    def promote_validated_step(
        self,
        *,
        step_id: str,
        pending_aliases: Mapping[str, Sequence[str]],
        allowed_evidence_ids: Sequence[str],
    ) -> EvidencePromotionResult:
        """Publish aliases only for evidence indexed by this exact attempt.

        Gate/seal evaluation remains with the caller.  Accepting only the
        explicit attempt index keeps this mechanical registrar from scanning
        for, or inventing, another notion of current authority.
        """

        step_id = str(step_id or "").strip()
        if not step_id:
            raise ValueError("success promotion requires a non-empty step_id")
        if isinstance(allowed_evidence_ids, (str, bytes)):
            raise ValueError(
                f"step {step_id} has no attempt-bound evidence index for promotion"
            )
        attempt_evidence_ids = {
            str(evidence_id).strip()
            for evidence_id in allowed_evidence_ids
            if str(evidence_id).strip()
        }
        pending_evidence_ids = {
            str(evidence_id).strip()
            for evidence_id in pending_aliases
            if str(evidence_id).strip()
        }
        unbound_evidence_ids = pending_evidence_ids.difference(attempt_evidence_ids)
        if unbound_evidence_ids:
            raise ValueError(
                "success promotion contains evidence outside the current attempt: "
                f"{sorted(unbound_evidence_ids)}"
            )

        current_records = list(self._evidence_store.records())
        records_by_evidence_id = {
            str(_record_field(record, "evidence_id") or ""): record
            for record in current_records
        }
        success_alias_bindings, retained, suppressed = filter_success_alias_bindings(
            pending_aliases,
            existing_aliases=self._evidence_store.aliases(),
            owners_by_evidence_id={
                evidence_id: str(
                    _record_field(record, "produced_by_step") or ""
                ).strip()
                for evidence_id, record in records_by_evidence_id.items()
            },
            step_id=step_id,
            records_by_evidence_id=records_by_evidence_id,
        )
        published = self._evidence_store.publish_step_success_aliases(
            success_alias_bindings,
            step_id=step_id,
            suppressed_basename_evidence_ids=suppressed,
        )
        return EvidencePromotionResult(
            published_aliases=published,
            retained_cross_step_aliases=retained,
            suppressed_basename_evidence_ids=suppressed,
        )


__all__ = [
    "EvidencePromotionResult",
    "EvidenceRegistrar",
    "filter_success_alias_bindings",
]
