"""Pure metric-comparison rules shared by replication and audits."""

from __future__ import annotations

import json
from easyicu.concept.metadata_sidecar import ColumnMetadataBinding
from easyicu.databases.research_capabilities import ResearchCapabilityProfile
from ..contracts.prediction_validation import ExternalValidationError, FrozenLogisticModel

from types import MappingProxyType
from typing import Any, Mapping, Optional

#: The structured EasyICU metric each paper-claim metric is compared with.  A
#: ratio is compared only on its own scale: a paper's risk ratio with a
#: reported risk ratio (``primary_rr``), never with an odds ratio.
CLAIM_METRIC_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "or": "primary_or",
        "hr": "primary_or",
        "rr": "primary_rr",
        "auroc": "auroc",
        "auc": "auroc",
        "brier_score": "brier_score",
        "p_value": "primary_pvalue",
        "p": "primary_pvalue",
        "n": "n_stays",
    }
)


def claim_metric_key(metric: Optional[str]) -> Optional[str]:
    """The EasyICU metric a paper claim's metric is compared with, if any."""

    return None if metric is None else CLAIM_METRIC_KEYS.get(metric.lower())


def compare_metric_values(
    *,
    metric: Optional[str],
    paper_value: Optional[float],
    paper_direction: Optional[str],
    easyicu_value: Any,
) -> tuple[str, str]:
    """Classify paper vs EasyICU alignment for one claim."""

    if paper_value is None:
        if paper_direction and easyicu_value is not None:
            easy_dir = "positive" if float(easyicu_value) > 1 else "negative"
            if paper_direction == easy_dir:
                return (
                    "directionally_aligned",
                    "Direction matched but no exact paper scalar was available.",
                )
        return (
            "not_comparable",
            "Paper claim did not expose a comparable numeric value.",
        )
    if easyicu_value is None:
        return (
            "not_comparable",
            "EasyICU run did not emit a comparable structured metric.",
        )
    try:
        easy = float(easyicu_value)
    except Exception:
        return (
            "not_comparable",
            "EasyICU metric could not be interpreted numerically.",
        )

    metric_name = (metric or "").lower()
    if metric_name in {"p_value", "p"}:
        paper_sig = paper_value < 0.05
        easy_sig = easy < 0.05
        if paper_sig == easy_sig:
            return "directionally_aligned", "Significance state matched."
        return "not_aligned", "Significance state did not match."

    if metric_name in {"or", "hr", "rr"}:
        if (paper_value > 1 and easy > 1) or (paper_value < 1 and easy < 1):
            delta = abs(paper_value - easy) / max(abs(paper_value), 1e-6)
            if delta <= 0.25:
                return "aligned", "Effect direction and magnitude were close."
            return (
                "directionally_aligned",
                "Effect direction matched but magnitude differed.",
            )
        return "not_aligned", "Effect direction did not match."

    if metric_name in {"auroc", "auc", "brier_score", "outcome_rate", "n"}:
        delta = abs(paper_value - easy)
        tolerance = (
            0.03
            if metric_name in {"auroc", "auc", "brier_score", "outcome_rate"}
            else max(5.0, 0.05 * max(abs(paper_value), 1.0))
        )
        if delta <= tolerance:
            return "aligned", "Numeric value was within tolerance."
        return (
            "not_aligned",
            "Numeric value differed beyond the comparison tolerance.",
        )

    delta = abs(paper_value - easy)
    if delta <= max(0.05, 0.2 * max(abs(paper_value), 1.0)):
        return "aligned", "Generic numeric tolerance matched."
    return "not_aligned", "Generic numeric tolerance did not match."


__all__ = ["CLAIM_METRIC_KEYS", "claim_metric_key", "compare_metric_values"]


def require_transport_capabilities(profile: ResearchCapabilityProfile, model: FrozenLogisticModel) -> None:
    """A physical column cannot override an absent/degraded dictionary recipe."""
    if not profile.source_registered:
        raise ExternalValidationError("validation_database_unavailable", database=profile.database)
    missing = [p.concept for p in model.predictors if not profile.concept(p.concept).mapped]
    if missing:
        raise ExternalValidationError("predictor_unavailable", database=profile.database, concepts=missing)
    if not profile.concept(model.outcome_concept).mapped:
        raise ExternalValidationError("outcome_unavailable", database=profile.database, concept=model.outcome_concept)


def transport_concept_meaning(binding: ColumnMetadataBinding, profile: ResearchCapabilityProfile) -> str:
    """Compare meaning, units, aggregation, window and representation, not names.

    Source-table and local column names deliberately do not enter a meaning:
    harmonized concepts can come from different tables in different databases.
    """
    metadata = binding.metadata
    concept = metadata.source_concept
    extra = None
    if concept == "death":
        extra = profile.mortality.meaning if profile.mortality else "unavailable"
    elif concept in {"age", "weight"}:
        representation = getattr(profile, concept)
        extra = {
            "kind": representation.kind, "top_code": representation.top_code,
            "bin_edges": representation.bin_edges, "bin_values": representation.bin_values,
            "exact_through": representation.exact_through,
        }
    return json.dumps({
        "concept": concept, "role": metadata.role.value, "unit": metadata.canonical_unit,
        "aggregation": metadata.aggregation,
        "window": binding.derivation_window.to_dict() if binding.derivation_window else None,
        "representation_transform": binding.representation_transform, "source_meaning": extra,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
