"""Pure metric-comparison rules shared by replication and audits."""

from __future__ import annotations

from typing import Any, Optional


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


__all__ = ["compare_metric_values"]
