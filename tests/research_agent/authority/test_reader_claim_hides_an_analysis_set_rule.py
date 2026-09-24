"""A reader never sees the engineering name of the analysis set behind a claim."""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.scientific_claims import (
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)

# Study coordinates unrelated to any benchmark item, on both effect scales.
_STUDIES = [
    ("lactate_max", "icu_readmission", "odds_ratio", 1.84, (1.21, 2.79)),
    ("norepinephrine_dose", "sofa_day3", "coefficient", -0.42, (-0.9, 0.06)),
    ("aki_stage", "death", "odds_ratio", 1.2000296869591074, (0.5142, 2.8004)),
]


def _association_claim(
    analysis_set: str,
    *,
    exposure: str = "lactate_max",
    outcome: str = "icu_readmission",
    effect_scale: str = "odds_ratio",
    estimate: float = 1.84,
    interval: tuple[float, float] = (1.21, 2.79),
):
    drafts = derive_scientific_claim_drafts(
        {
            "interpretation_class": "adjusted_association",
            "exposure": exposure,
            "outcome": outcome,
            "effect_scale": effect_scale,
            "primary_estimate": estimate,
            "primary_estimate_interval": list(interval),
            "analysis_set": analysis_set,
            "analysis_role": "primary",
            "adjustment_covariates": ["age", "sex"],
        }
    )
    return bind_scientific_claim_drafts(
        [drafts[0].model_dump(mode="json")],
        step_id="adjusted_association",
        evidence_id="adjusted_association_summary",
    )[0]


@pytest.mark.parametrize("include_estimate", [True, False])
@pytest.mark.parametrize(
    ("exposure", "outcome", "effect_scale", "estimate", "interval"), _STUDIES,
    ids=[study[0] for study in _STUDIES],
)
def test_b_source_aware_set_reads_as_the_analysis_set(
    include_estimate: bool,
    exposure: str,
    outcome: str,
    effect_scale: str,
    estimate: float,
    interval: tuple[float, float],
) -> None:
    claim = _association_claim(
        "source_aware", exposure=exposure, outcome=outcome,
        effect_scale=effect_scale, estimate=estimate, interval=interval,
    )

    reader = claim.render_reader_text(include_estimate=include_estimate)

    assert f"{outcome.replace('_', ' ')} in the analysis set (" in reader
    assert "source aware" not in reader.casefold()
    # The machine authority the Writer's claim tokens are checked against is unchanged.
    assert "the source aware analysis set" in claim.render_text()


def test_b_a_population_readers_recognise_keeps_its_name() -> None:
    reader = _association_claim("complete_case").render_reader_text()

    assert "icu readmission in the complete case analysis set" in reader
