"""An event whose time is unknown must stop the step, not switch arms.

MEASURED on the never-passing canonical tasks: three runs, three different
silent recodings of the same unreconciled event/time pair.

* "For death=1 rows with missing death_time, the outcome construction silently
  converts death_time > 24h to 0, treating an unavailable post-landmark time as
  a negative outcome." -- ``NaN > 24`` is ``False``, so a death became a
  survivor with no error and no count.
* "Outcome and censoring times are not reconciled before risk-set construction.
  Deaths with death=1 but missing death_time, or survivors with missing los_hosp,
  receive neither event nor censoring and are silently removed through
  duration/event missingness."
* "Deaths with missing or unusable death_time are silently treated as
  non-events and censored."

And the same absent rule produced its mirror in the opposite direction:

* "complete_case requires death_time for death-negative rows, although
  death_time is structurally not applicable when death=0; this silently excludes
  non-events from the survival denominator."

Those four are not four judgement calls. An event whose time is unknown cannot be
placed on the follow-up axis under any protocol, and recoding it to "no event"
moves a death into the survivor arm without changing any number's appearance.
An absent event time on a row whose flag says *no event* is the expected shape,
not a missing value to exclude on.

``EndpointSpec.absence_semantics`` declares what an absent ROW means. It has never
had anything to say about a present row whose event time is missing -- which is
why every run answered that question again, differently.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.method_kernels import CURATED_METHOD_KERNELS
from easyicu.research_agent.methods.survival_inputs import (
    SurvivalInputError,
    event_time_reconciliation_receipt,
)

_RUNTIME = ("pandas", "numpy", "statsmodels", "lifelines", "scipy", "sklearn")


def _receipt(frame: pd.DataFrame, **overrides: object) -> dict:
    kwargs: dict = {
        "event_column": "death",
        "time_column": "death_time",
        "event_levels": [0, 1],
        "censored_level": 0,
    }
    kwargs.update(overrides)
    return event_time_reconciliation_receipt(frame, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The defect, in the shape it was recorded in.
# --------------------------------------------------------------------------


def test_an_event_with_no_usable_time_fails_closed() -> None:
    frame = pd.DataFrame({"death": [1, 0, 1], "death_time": [10.0, np.nan, np.nan]})
    with pytest.raises(SurvivalInputError) as raised:
        _receipt(frame)
    assert raised.value.audit["event_without_placeable_time_n"] == 1


def test_the_refusal_names_both_ways_the_row_disappears() -> None:
    """The two recodings measured, so neither reads as the safe option.

    One run moved the row into the survivor arm; another dropped it out of both
    arms. A message naming only one of them invites the other.
    """

    frame = pd.DataFrame({"death": [1], "death_time": [np.nan]})
    with pytest.raises(SurvivalInputError) as raised:
        _receipt(frame)
    message = str(raised.value)
    assert "survivor arm" in message
    assert "duration/event missingness" in message
    # And the two protocol-legal exits, so the step is not simply stuck.
    assert "Exclude them explicitly" in message
    assert "censor them at a declared last-contact time" in message


def test_a_non_finite_event_time_is_not_usable_either() -> None:
    """`inf` passes a notna() check and then poisons every comparison."""

    frame = pd.DataFrame({"death": [1], "death_time": [np.inf]})
    with pytest.raises(SurvivalInputError):
        _receipt(frame)


def test_a_string_event_time_is_judged_the_host_way() -> None:
    """Reuses the host's own numeric conversion, not `pd.to_numeric`.

    "Unusable" has to mean the same thing here as it does one module away, or a
    time column arriving as text is unusable to one boundary and fine to the
    other.
    """

    usable = pd.DataFrame({"death": [1, 0], "death_time": ["10.5", None]})
    assert _receipt(usable)["status"] == "reconciled"

    unusable = pd.DataFrame({"death": [1, 0], "death_time": ["not a time", None]})
    with pytest.raises(SurvivalInputError):
        _receipt(unusable)


# --------------------------------------------------------------------------
# The mirror: the expected shape must not be reported as a defect.
# --------------------------------------------------------------------------


def test_a_censored_row_with_no_event_time_is_the_expected_shape() -> None:
    """`death_time` is structurally N/A when `death=0`.

    Treating its absence as a missing value is what excluded non-events from a
    survival denominator -- the same pair, mishandled in the other direction.
    """

    frame = pd.DataFrame({"death": [0, 0, 1], "death_time": [np.nan, np.nan, 12.0]})
    receipt = _receipt(frame)
    assert receipt["status"] == "reconciled"
    assert receipt["censored_n"] == 2
    assert receipt["event_n"] == 1


def test_the_count_of_censored_rows_without_a_time_is_still_reported() -> None:
    """Not a defect, but the caller has to be able to see its size.

    A complete-case filter on the event-time column would remove exactly these
    rows, and the count is what makes that visible before it happens.
    """

    frame = pd.DataFrame({"death": [0, 0, 1], "death_time": [np.nan, np.nan, 12.0]})
    assert _receipt(frame)["censored_without_time_n"] == 2


# --------------------------------------------------------------------------
# The closed level set, which is the endpoint's declaration.
# --------------------------------------------------------------------------


def test_an_undeclared_event_code_stops_the_step() -> None:
    frame = pd.DataFrame({"death": [0, 1, 2], "death_time": [np.nan, 1.0, 3.0]})
    with pytest.raises(SurvivalInputError, match="outside the declared closed set"):
        _receipt(frame)


def test_a_competing_risks_level_set_is_accepted() -> None:
    """More than one event code is a declaration, not an error.

    A competing-risks design declares one code per event; refusing a third level
    would make the kernel unusable for exactly the design that needs it most.
    """

    frame = pd.DataFrame(
        {"outcome": [0, 1, 2], "event_time": [np.nan, 4.0, 9.0]}
    )
    receipt = event_time_reconciliation_receipt(
        frame,
        event_column="outcome",
        time_column="event_time",
        event_levels=[0, 1, 2],
        censored_level=0,
    )
    assert receipt["event_n"] == 2
    assert receipt["censored_n"] == 1


def test_the_censored_level_must_be_one_of_the_declared_levels() -> None:
    frame = pd.DataFrame({"death": [0, 1], "death_time": [np.nan, 1.0]})
    with pytest.raises(SurvivalInputError, match="not one of the declared"):
        _receipt(frame, censored_level=9)


def test_a_single_level_event_column_is_refused() -> None:
    """One level cannot distinguish an event from its absence."""

    frame = pd.DataFrame({"death": [1], "death_time": [1.0]})
    with pytest.raises(SurvivalInputError, match="at least two declared levels"):
        _receipt(frame, event_levels=[1])


# --------------------------------------------------------------------------
# The receipt cannot change the population it reports on.
# --------------------------------------------------------------------------


def test_the_receipt_carries_no_values_mask_or_frame() -> None:
    """The property that makes this safe to call unconditionally.

    A helper that returned a filtered frame would be making the protocol
    decision it exists to surface.
    """

    frame = pd.DataFrame({"death": [0, 1], "death_time": [np.nan, 5.0]})
    receipt = _receipt(frame)
    assert receipt["role"] == "audit_only"
    for value in receipt.values():
        assert not isinstance(value, (pd.Series, pd.DataFrame, np.ndarray))


def test_missing_or_identical_columns_are_typed_failures() -> None:
    frame = pd.DataFrame({"death": [1], "death_time": [1.0]})
    with pytest.raises(SurvivalInputError, match="missing"):
        _receipt(frame, time_column="not_a_column")
    with pytest.raises(SurvivalInputError, match="two distinct columns"):
        _receipt(frame, time_column="death")


def test_the_error_is_catchable_as_the_existing_host_input_error() -> None:
    """So an existing `except DescriptiveInputError` keeps working.

    A second unrelated exception type is a second entry the runtime failure
    classifier would need, and one nobody would remember to add.
    """

    from easyicu.research_agent.methods.descriptive_inputs import DescriptiveInputError

    assert issubclass(SurvivalInputError, DescriptiveInputError)


# --------------------------------------------------------------------------
# Reachability: a kernel the Coder is never offered is dead code.
# --------------------------------------------------------------------------


def _software_slots(*, family: str, intent: str, method: str) -> list[str]:
    """The software resources actually selected, by the host's own builder."""

    from easyicu.research_agent.resources.coder import build_coder_resource_bundle

    bundle = build_coder_resource_bundle(
        step_id="s",
        profile_ref="test/profile@1",
        analysis_family=family,
        step_role="primary",
        question="Is the exposure associated with 28-day mortality?",
        intent=intent,
        method=method,
        planner_inputs=("cohort",),
        expected_outputs=("table:result",),
        resolved_input_bindings={},
        runtime_import_names=_RUNTIME,
    )
    return [
        name.split(".")[-1]
        for name in sorted(
            set(re.findall(r'"import_name":\s*"([^"]+)"', bundle.prompt_projection))
        )
    ]


def test_the_kernel_is_declared_with_the_families_that_measured_the_defect() -> None:
    kernel = next(k for k in CURATED_METHOD_KERNELS if k.module == "survival_inputs")
    # h1 stamps `survival`; h2 stamps `causal_inference`; `time_to_event` is the
    # registry's other name for the first. All three produced findings.
    assert set(kernel.families) == {"time_to_event", "survival", "causal_inference"}


def test_the_risk_set_step_is_offered_the_reconciliation_kernel() -> None:
    """The step where the recoding happened."""

    slots = _software_slots(
        family="survival",
        intent=(
            "Describe the landmark risk set, follow-up eligibility, event counts "
            "and censoring."
        ),
        method="survival_risk_set_accounting",
    )
    assert "survival_inputs" in slots, slots


def test_the_causal_landmark_step_is_offered_it_too() -> None:
    """MEASURED: `causal_inference` was offered ZERO software resources before.

    h2's outcome construction is where `NaN > 24` became a survivor, and the
    family had no kernel of any kind to reach for.
    """

    slots = _software_slots(
        family="causal_inference",
        intent=(
            "Construct the landmark outcome and estimate the average treatment "
            "effect with IPTW."
        ),
        method="iptw_landmark_ate",
    )
    assert "survival_inputs" in slots, slots


def test_adding_it_did_not_cost_the_cox_step_its_ph_kernel() -> None:
    """Only three software slots exist, so a fourth candidate can displace one.

    MEASURED, and it corrects a wrong reading of my own: a query naming BOTH
    jobs at once ("build the landmark risk set ... and fit a Cox model") does
    rank ph_schoenfeld out on the `survival` family. Real per-step intents do
    not -- the risk-set step gets this kernel and the Cox step keeps
    ph_schoenfeld, which is the only route by which `lifelines` reaches a Cox
    step's prompt at all.

    The existing guard for that (`test_a_survival_step_is_offered_the_ph_kernel_
    and_the_library_it_wraps`) queries only the `time_to_event` family, so it
    stayed green either way. This pins the family h1 actually stamps.
    """

    slots = _software_slots(
        family="survival",
        intent=(
            "Fit the primary Cox proportional hazards model and test the "
            "proportional hazards assumption."
        ),
        method="cox_proportional_hazards",
    )
    assert "ph_schoenfeld" in slots, slots
