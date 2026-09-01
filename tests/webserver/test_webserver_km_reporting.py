"""A Kaplan-Meier curve has to ship the numbers needed to read it.

Two facts were computed and then dropped on the floor. ``number_at_risk`` was
in the payload but never rendered, so a reader could not tell whether a flat
tail was a finding or three remaining patients. Censoring times were not in the
payload at all — ``_km_points`` only emits a point where an event occurred — so
a flat stretch caused by attrition looked identical to one caused by survival.

Both are reporting requirements, not chart polish, so they are locked here.
"""

from __future__ import annotations

from pathlib import Path
import re

from easyicu.webserver import cohort_review


STATIC = (
    Path(__file__).resolve().parents[2] / "src" / "easyicu" / "webserver" / "static"
)


def _asset(*parts: str) -> str:
    return STATIC.joinpath(*parts).read_text(encoding="utf-8")


# (time, event) — events at 3 and 7; censoring at 2, 5, 5 and 9.
RECORDS = [(3.0, True), (7.0, True), (2.0, False), (5.0, False), (5.0, False), (9.0, False)]


def test_group_payload_carries_censoring_marks() -> None:
    payload = cohort_review._km_group_payload("Sepsis", RECORDS)

    assert payload["events"] == 2
    assert payload["censored"] == 4
    marks = payload["censor_marks"]
    # One tick per distinct censoring time, the standard KM convention.
    assert [mark["time"] for mark in marks] == [2, 5, 9]


def test_censor_marks_sit_on_the_step_the_subject_left_from() -> None:
    points = cohort_review._km_points(RECORDS)
    marks = cohort_review._censor_marks(RECORDS, points)

    survival_at = {mark["time"]: mark["survival"] for mark in marks}
    # t=2 precedes the first event (t=3), so survival is still 100%.
    assert survival_at[2] == 100.0
    # t=5 sits between the events at t=3 and t=7.
    assert survival_at[5] == 80.0
    # t=9 follows the last event.
    assert survival_at[9] == 40.0
    assert all(0.0 <= mark["survival"] <= 100.0 for mark in marks)


def test_censor_marks_handle_empty_and_event_only_cohorts() -> None:
    assert cohort_review._censor_marks([], []) == []

    events_only = [(1.0, True), (2.0, True)]
    points = cohort_review._km_points(events_only)
    assert cohort_review._censor_marks(events_only, points) == []


def test_censor_marks_are_bounded_for_large_cohorts() -> None:
    records = [(float(i) / 10.0, False) for i in range(1, 900)]
    points = cohort_review._km_points(records)
    marks = cohort_review._censor_marks(records, points)

    assert 0 < len(marks) <= 40
    times = [mark["time"] for mark in marks]
    assert times == sorted(times)


def test_chart_owner_renders_censor_ticks_and_the_risk_table() -> None:
    charts = _asset("js", "screens-viz-cohort-charts.js")

    assert "km-censor:" in charts
    assert "group.censorMarks" in charts
    assert "function riskTableHtml" in charts
    # A real table, not a second chart grid: selectable and screen-readable.
    assert 'class="km-risk-table"' in charts
    assert 'scope="row"' in charts and 'scope="col"' in charts
    # Censor ticks must not duplicate every group in the axis tooltip.
    assert "startsWith('km-censor:')" in charts


def test_risk_table_alignment_shares_one_grid_definition() -> None:
    """Columns land on their axis ticks only if both read the same insets."""

    charts = _asset("js", "screens-viz-cohort-charts.js")
    css = _asset("css", "cohort-charts.css")

    assert "const SURVIVAL_GRID = {" in charts
    assert "chartCore.grid(survivalGrid(spec))" in charts
    assert "--km-inset-left:${survivalGrid(spec).left}px" in charts
    assert "--km-inset-left" in css
    # An explicit axis max is what makes the percentage positions meaningful.
    assert "max: survivalHorizon(spec)" in charts


def test_dropping_a_crowded_column_cannot_shift_the_counts() -> None:
    """The gap filter drops columns; values stay indexed by the backend grid.

    On a 28-day axis the clinical grid puts day 1 only 3.6% along, so its
    three-digit count collided with day 0's into "214201". Filtering by
    position fixes that, but re-indexing `row.values` by the *filtered*
    position would silently print the wrong number under the wrong day.
    """

    charts = _asset("js", "screens-viz-cohort-charts.js")

    assert "function spacedRiskTimes" in charts
    # Columns must carry their original index, not their filtered position.
    assert "times.map((time, index) => ({ time, index }))" in charts
    assert "${times.map(({ time, index }) =>" in charts
    assert "const value = (row.values || [])[index];" in charts


def test_risk_table_does_not_reserve_a_scrollbar_gutter() -> None:
    """`overflow-x:auto` computes overflow-y to auto and steals 15px.

    That gutter came out of the content box, so every column rendered left of
    the axis tick it labelled. Columns are positioned as a fraction of the
    axis and row headers ellipsise, so the strip cannot overflow.
    """

    # Strip comments first — the rule's own comment names the value it avoids.
    css = re.sub(r"/\*.*?\*/", "", _asset("css", "cohort-charts.css"), flags=re.S)
    km_block = css[css.index(".km-risk{") :]
    km_block = km_block[: km_block.index("}")]

    assert "overflow:hidden" in km_block
    assert "overflow-x:auto" not in km_block


def test_risk_table_widens_the_gutter_for_group_labels() -> None:
    """58px fits "100%", not "Sepsis-3 positive"; labels ran into the counts."""

    charts = _asset("js", "screens-viz-cohort-charts.js")

    assert "const SURVIVAL_RISK_GUTTER = 124;" in charts
    assert "function survivalGrid(spec)" in charts
    # Both the plot and the strip must read the widened value, or they detach.
    assert "chartCore.grid(survivalGrid(spec))" in charts
    assert "--km-inset-left:${survivalGrid(spec).left}px" in charts


def test_route_passes_the_reporting_fields_through() -> None:
    viz = _asset("js", "screens-viz.js")

    assert "censorMarks: group.censor_marks" in viz
    assert "horizon: curve.display_horizon_days" in viz
    assert "curve.number_at_risk" in viz
    # Risk-row labels are translated like group labels, or the fallback table
    # cannot match a row to its group.
    assert "label: cohortText(row.label)" in viz


def test_fail_closed_fallback_states_the_same_facts() -> None:
    charts = _asset("js", "screens-viz-cohort-charts.js")
    fallback = charts[charts.index("function survivalFallback") :]
    fallback = fallback[: fallback.index("function survivalSlot")]

    assert "spec.censoredLabel" in fallback
    assert "spec.atRiskLabel" in fallback
    assert "group.censored" in fallback


# The palette guards moved to test_static_frontend_ownership.py: they are an
# ownership rule, not a KM reporting rule, and the version here enumerated the
# copies it knew about instead of scanning for them.


def test_risk_table_declares_roles_its_css_would_otherwise_strip() -> None:
    """Positioning the columns needs display:block, which drops table roles.

    CSS `display` overrides the implicit ARIA role of a table element, so a
    <table> laid out this way is announced as a stack of anonymous text and
    the scope attributes go inert. The explicit roles restore the grid.
    """

    charts = _asset("js", "screens-viz-cohort-charts.js")
    css = _asset("css", "cohort-charts.css")

    assert 'role="table"' in charts
    assert 'role="rowgroup"' in charts
    assert charts.count('role="row"') >= 2
    assert 'role="columnheader"' in charts
    assert 'role="rowheader"' in charts
    assert 'role="cell"' in charts
    # The CSS that makes the roles necessary.
    assert "display:block" in css
