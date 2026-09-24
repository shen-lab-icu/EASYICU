"""A family template's claim boundary is written about the study's own exposure.

The sealed survival template once told every study that it could not prove "a
causal ventilation effect" -- the exposure of the development question it was
built beside.  The boundary must name whatever exposure the request carries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.research_agent.planning.family_spec import survival_template


def _request(exposure: str) -> SimpleNamespace:
    return SimpleNamespace(
        sealed_suite=SimpleNamespace(
            landmark_hours=24.0,
            endpoint_horizon_days=28.0,
            adjustment_columns=("age", "sex"),
            source_columns=("event_time", "event", exposure, "age", "sex"),
        ),
        primary_exposure=exposure,
        outcome="death",
        cluster_unit="patient",
        research_question="Is an early exposure associated with 28-day survival?",
        comparison_literature_keys=[],
        allowed_literature_citation_keys=["strobe_2007", "record_2015"],
        identity_column="stay_id",
    )


@pytest.mark.parametrize(
    ("exposure", "label"),
    [("vaso_ind", "vasopressor use"), ("rrt", "kidney replacement therapy")],
)
def test_the_survival_claim_boundary_names_the_requested_exposure(
    exposure: str, label: str
) -> None:
    spec = SimpleNamespace(labels={exposure: label})

    selection = survival_template._design_selection(
        _request(exposure), spec, method_keys=["strobe_2007"]
    )
    selected = next(
        candidate
        for candidate in selection.candidates
        if candidate.disposition == "selected"
    )

    assert selected.cannot_prove.startswith(f"No causal effect of {label},")
    assert "ventilation" not in selected.cannot_prove
