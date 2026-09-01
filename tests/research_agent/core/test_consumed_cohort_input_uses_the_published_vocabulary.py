"""The owner reads one key and the host attests to another.

``sole_typed_cohort_input`` is where "which keys name the closed cohort
product" is decided.  Every deterministic owner's ``owns_step`` routes through
it, and the rendered entrypoint passes the key it returns.

Four selection branches then spelled the rule out again by hand -- match
``cohort:`` or exactly ``artifact:analysis_cohort`` -- to build the
``consumed_input_keys`` the host uses to stamp its input-binding receipts.
That copy is blind to ``table:analysis_cohort`` and ``dataset:analysis_cohort``.

What follows from one missing key is the whole 2026-08-01 E1 failure:

* no entry in ``consumed_input_keys`` -> the host stamps no receipt;
* ``step_summary_integrity`` reports the step did not account for an input the
  host itself resolved;
* the host dispatches a CONTRACT REPAIR against its own rendered code;
* the model, trying to produce the missing receipt, inserts
  ``consumption_contract: 'all_rows'`` into the spec payload;
* ``ExposureOutcomeDistributionSpec`` forbids extra fields, so the step dies on
  a ``ValidationError`` for a defect no model authored.

The executed script was byte-identical to the host's own render except for that
inserted key and one flipped flag.

Measured through the real selector over every recorded plan (838 owned steps):
ownership is unchanged and identical, 30 steps gain the receipt they were owed
(12 ``table:analysis_cohort``, 18 ``dataset:analysis_cohort``), none loses one,
and owned steps that could not produce a receipt at all fall from 68 to 38.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.selection import (
    _consumed_typed_cohort_inputs,
    select_standard_executor,
)
from easyicu.research_agent.execution.runners.typed_input_binding import (
    sole_typed_cohort_input,
)
from easyicu.research_agent.schema import AnalysisStep


def _distribution_step(cohort_key: str) -> AnalysisStep:
    """A step the exposure/outcome distribution owner claims."""

    return AnalysisStep.model_validate(
        {
            "step_id": "05_distribution",
            "planned_analysis_role": "auxiliary",
            "intent": "Describe the exposure by outcome.",
            "inputs": [cohort_key, "exposure", "death"],
            "expected_outputs": ["table:exposure_outcome_distribution"],
            "method": "descriptive",
            # Every field the schema requires, copied from the spec the
            # 2026-08-01 E1 plan actually declared.
            "exposure_outcome_distribution_spec": {
                "schema_version": "easyicu.exposure_outcome_distribution/2",
                "exposure": "exposure",
                "exposure_levels": [0, 1],
                "outcome": "death",
                "outcome_levels": [0, 1],
                "outcome_positive_value": 1,
                "level_match_policy": "exact_typed",
                "denominator_policy": "all_declared_rows",
                "missing_exposure_policy": "fail_closed",
                "missing_outcome_policy": "fail_closed",
                "undeclared_outcome_policy": "fail_closed",
                "interval_method": "wilson",
                "confidence_level": 0.95,
            },
        }
    )


@pytest.mark.parametrize(
    "cohort_key",
    [
        "table:analysis_cohort",
        "dataset:analysis_cohort",
        "artifact:analysis_cohort",
        "cohort:analysis_set",
    ],
)
def test_every_spelling_the_owner_reads_is_also_attested(cohort_key: str) -> None:
    """The property that was false for two of these four spellings.

    Whatever key the owner will actually read must be the key the host says it
    consumed; otherwise the host cannot stamp a receipt for it and then blames
    the step for the gap.
    """

    step = _distribution_step(cohort_key)
    assert sole_typed_cohort_input(step) == cohort_key
    assert _consumed_typed_cohort_inputs(step) == (cohort_key,)


def test_the_key_the_host_attests_is_the_key_the_owner_was_given() -> None:
    """End to end through the real selector, on the spelling that died.

    The rendered entrypoint and the consumed-key declaration must name the same
    input -- that equality is the whole contract.
    """

    step = _distribution_step("table:analysis_cohort")
    selection = select_standard_executor(
        step, plan=None, plausibility_scope=None, resolved_bindings={}, trace=[]
    )
    assert selection is not None
    assert selection.consumed_input_keys == ("table:analysis_cohort",)
    assert "table:analysis_cohort" in selection.code


def test_a_step_with_no_typed_cohort_input_attests_to_nothing() -> None:
    step = _distribution_step("exposure")
    # 'exposure' is not typed, so there is no typed cohort input at all.
    assert _consumed_typed_cohort_inputs(step) == ()


def test_two_typed_cohort_inputs_attest_to_nothing() -> None:
    """Ambiguity must not resolve itself by picking one.

    The published reader answers "" for more than one typed input, and the
    owners decline such a step; a hand-written prefix match would have picked
    whichever came first and attested to a frame the owner never read.
    """

    step = AnalysisStep.model_validate(
        {
            "step_id": "05_distribution",
            "planned_analysis_role": "auxiliary",
            "intent": "Describe the exposure by outcome.",
            "inputs": ["table:analysis_cohort", "cohort:other", "death"],
            "expected_outputs": ["table:exposure_outcome_distribution"],
            "method": "descriptive",
        }
    )
    assert sole_typed_cohort_input(step) == ""
    assert _consumed_typed_cohort_inputs(step) == ()


def test_no_branch_spells_the_vocabulary_out_again() -> None:
    """The rule that keeps the copies from coming back.

    Four branches carried the same hand-written match.  A fifth would silently
    reintroduce the same failure, so the shape itself is what is banned.
    """

    from easyicu.research_agent.execution.runners import selection

    tree = ast.parse(Path(selection.__file__).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        rendered = ast.unparse(node)
        if "startswith('cohort:')" in rendered or 'startswith("cohort:")' in rendered:
            offenders.append(rendered[:80])
    assert not offenders, f"a branch matches the cohort vocabulary by hand: {offenders}"


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_no_owned_step_reads_a_cohort_the_host_does_not_attest_to() -> None:
    """Real plans: the two readers must agree on every step an owner claims.

    This is the invariant, stated over real data rather than over a fixture.
    Before the fix, 30 owned steps read a cohort key the host never listed.
    """

    mismatched = []
    owned = 0
    for path in sorted(_CORPUS.glob("batch_*/*/aware/run_*/analysis_plan*.json")):
        try:
            plan = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for raw in plan.get("steps") or []:
            try:
                step = AnalysisStep.model_validate(raw)
                selection = select_standard_executor(
                    step,
                    plan=None,
                    plausibility_scope=None,
                    resolved_bindings={},
                    trace=[],
                )
            except Exception:
                continue
            if selection is None:
                continue
            owned += 1
            read = sole_typed_cohort_input(step)
            if read and read not in set(selection.consumed_input_keys or ()):
                mismatched.append((step.step_id, selection.analysis_kind, read))

    if not owned:
        pytest.skip("no recorded plan step is claimed by a deterministic owner")
    assert not mismatched, (
        f"{len(mismatched)} owned steps read a cohort key the host does not "
        f"attest to, e.g. {mismatched[:3]}"
    )
