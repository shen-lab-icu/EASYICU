"""[Layer 5: Evaluation And Submission Scaffold] Tier-2 jury rubric.

Encodes the 4-dimension x 4-level (0-3) scale from
``02_npj_Digital_Medicine/tier_evaluation_protocol_20260527.md``. The
rubric is data, not behaviour: changing anchors should not require touching
jury orchestration, and rubric versions can be tracked alongside submission
profiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

DimensionId = Literal[
    "plan_completeness",
    "evidence_binding",
    "missingness_handling",
    "overclaim_avoidance",
]


@dataclass(frozen=True)
class DimensionAnchors:
    dimension_id: DimensionId
    label: str
    anchors: Dict[int, str]


@dataclass(frozen=True)
class JuryRubric:
    version: str
    dimensions: List[DimensionAnchors]

    @property
    def dimension_ids(self) -> List[str]:
        return [dimension.dimension_id for dimension in self.dimensions]


NPJ_DM_RUBRIC_V1 = JuryRubric(
    version="npj_dm_rubric/20260527",
    dimensions=[
        DimensionAnchors(
            dimension_id="plan_completeness",
            label="Plan completeness",
            anchors={
                0: "No coherent plan; missing primary analysis",
                1: "Plan present but skips a key sensitivity or QC step",
                2: "Plan covers primary + at least one sensitivity, but motivation is thin",
                3: "Plan covers primary, sensitivity, and QC steps with explicit rationale tied to the research question",
            },
        ),
        DimensionAnchors(
            dimension_id="evidence_binding",
            label="Evidence binding",
            anchors={
                0: "Bound result sentences absent or unhash-able",
                1: "Some bound; UNTRACED markers remain on headline numbers",
                2: "Most numbers bound; only ancillary descriptors UNTRACED",
                3: "All headline numbers (OR/HR/AUROC/CI/p, primary n) carry evidence ids with no UNTRACED markers in the primary results paragraph",
            },
        ),
        DimensionAnchors(
            dimension_id="missingness_handling",
            label="Missingness handling",
            anchors={
                0: "Missingness not mentioned",
                1: "Missingness mentioned but not quantified",
                2: "Missingness quantified (e.g., complete-case n vs source n) but consequences unstated",
                3: "Missingness quantified, sensitivity analysis performed, and Discussion explicitly limits inference",
            },
        ),
        DimensionAnchors(
            dimension_id="overclaim_avoidance",
            label="Over-claim avoidance",
            anchors={
                0: "Headline claim exceeds evidence (e.g., causal language, \"validated\")",
                1: "Mild over-extension (broad generalisation without external validation)",
                2: "Cautious tone but some unsupported scope expansion in Discussion",
                3: "Conclusion stays within the run's evidence; explicit external-validation caveat present",
            },
        ),
    ],
)

RUBRIC_REGISTRY = {
    NPJ_DM_RUBRIC_V1.version: NPJ_DM_RUBRIC_V1,
}


def get_rubric(version: str) -> JuryRubric:
    """Return a registered Tier-2 rubric by version."""

    try:
        return RUBRIC_REGISTRY[version]
    except KeyError as exc:
        known = ", ".join(sorted(RUBRIC_REGISTRY))
        raise ValueError(f"unknown Tier-2 rubric {version!r}; known: {known}") from exc


__all__ = [
    "DimensionAnchors",
    "DimensionId",
    "JuryRubric",
    "NPJ_DM_RUBRIC_V1",
    "RUBRIC_REGISTRY",
    "get_rubric",
]
