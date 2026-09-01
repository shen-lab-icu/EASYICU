"""The figure-source repair must name the shape that makes nothing verifiable.

MEASURED (m3 sepsis subphenotype, 10 of 11 steps, every figure format rendered):
step 09 failed `contract_failed` on

    Figure source-data table 'cluster_profile_visualisation_source_data.csv' is
    not a traceable subset of the declared upstream table(s); source rows joined
    to phenotype_profiles.csv on source_row_index, but these source-data value
    columns were not verified against any row-aligned upstream value vector:
    ['raw_value', 'value']

The bundle is one long table over three panels:

    plot_panel,plot_key,source_row_index,source_table,cluster,variable,level,
    value,raw_value,count,denominator,percent
    continuous_profile,hr_mean|0,0,...phenotype_profiles.csv,0,hr_mean,,
    -0.5,82.91999816894531,,,

while the parent carries `cluster, summary_type, variable, level, n, ..., median,
q1, q3, mean, sd, count`. `raw_value` is the parent's `mean` on a continuous
panel and its `percent` on a categorical one, so no single upstream vector
matches it -- the column that would have verified on its own was made
unverifiable by being stacked. `value` is a standardized rendering of it and has
no upstream counterpart at all.

The route's existing guidance covers a different case (one CSV per PARENT) and
tells the coder to remove derived fields that are "unplotted" -- which the
standardized column is not. Together the two left this shape with no compliant
form.

The validator itself already measures the shape: 12 of 361 recorded source-data
tables carry duplicate keys and 6 of the 8 with a known step status failed.
"""

from __future__ import annotations

from easyicu.research_agent.repairs.reasons import (
    RepairRoute,
    _derived_repair_routes,
)


def _guidance_for_figure_source_trace() -> str:
    """The guidance text the coder actually receives for this route.

    Built by calling the real function, not by reading its source: the source
    is a run of adjacent string literals, so every phrase that spans two of
    them is broken by a quote pair and a substring assertion over it silently
    tests nothing.
    """

    from easyicu.research_agent.agents.core import _repair_specialization
    from easyicu.research_agent.repairs.reasons import RepairPromptAuthority

    authority = RepairPromptAuthority.create(
        typed_ticket=[{"validator": "figure_source_data"}]
    )
    text = _repair_specialization(
        context=None, repair_authority=authority, code=""
    )
    marker = "DIAGNOSED FIGURE SOURCE-DATA TRACE REPAIR"
    assert marker in text, "the route produced no guidance"
    start = text.index(marker)
    return " ".join(text[start:].split())


def test_the_route_fires_on_the_validator_that_reported_it() -> None:
    """Reachability first: guidance on a route that never fires is dead text."""

    # A real ticket entry, keyed the way the validator reports it.
    routes = _derived_repair_routes([{"validator": "figure_source_data"}])
    assert RepairRoute.FIGURE_SOURCE_TRACE.value in routes


def test_the_stacked_panel_shape_is_named() -> None:
    text = _guidance_for_figure_source_trace()
    assert "ONE PANEL PER FILE" in text
    # The mechanism, not just the instruction: the coder has to understand why
    # a column that would verify alone stops verifying when stacked.
    assert "alternates between" in text
    assert "no single upstream vector matches it" in text


def test_the_previously_verifiable_column_loss_is_stated() -> None:
    """The non-obvious part: stacking loses columns that were fine.

    m3's `raw_value` was the parent's own value. Nothing was wrong with it
    except its neighbours.
    """

    text = _guidance_for_figure_source_trace()
    assert "including the ones that would have verified" in text


def test_a_plotted_rescaling_is_given_a_compliant_form() -> None:
    """The gap that made the shape uncompliable.

    The route already says to drop derived fields that are UNPLOTTED. A
    standardized value that is plotted had no instruction at all, so the coder
    could neither keep it nor remove it.
    """

    text = _guidance_for_figure_source_trace()
    assert "standardized" in text
    assert "export the upstream raw value" in text
    # And it must not contradict the existing rule it sits beside.
    assert "unplotted derived numeric/boolean audit fields" in text


def test_the_per_parent_rule_is_still_there() -> None:
    """The new case is additive; the original one still has to be stated."""

    text = _guidance_for_figure_source_trace()
    assert "separate exact/subset CSV" in text
    assert "never collapse unrelated parents into generic" in text
