"""A product promised as a statistic and as something else cannot be executed.

Measured 2026-07-30 over every recorded plan (6,571 steps with typed
``expected_outputs``): 369 promise one bare name under two kinds, and 359 of
those are ``robustness_summary`` as both ``table:`` and ``statistic:``.  Running
the gate over the 675 real plan files flags 359 of them and nothing else.

That shape is unexecutable rather than merely untidy.  A ``statistic:`` product
is one finite number in a JSON sidecar; every typed declaration the host owns
names its products *without* the kind, so no declaration can say which of the
two promises it backs -- and the schema forbids the only declaration that would
try, because ``product_id`` values must be unique.  The deterministic robustness
owner can produce five of the six products the real step promises and therefore
claims none of them.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List

from easyicu.research_agent.gates.product_promise import (
    product_promise_plan_findings,
    product_promise_replan_directive,
)

_PHASE = (
    Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/execution/phase.py"
)

_REAL_STEP_OUTPUTS = [
    "statistic:primary_or",
    "statistic:complete_case_n",
    "table:robustness_summary",
    "log:missingness_strategy_notes",
    "table:robustness_matrix",
    "statistic:robustness_summary",
]


class _ReplaySpec:
    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def output_for(self, product_id: str) -> str | None:
        return self.mapping.get(product_id)


class _Step:
    def __init__(
        self,
        step_id: str,
        outputs: List[str],
        *,
        replay_outputs: dict[str, str] | None = None,
    ) -> None:
        self.step_id = step_id
        self.expected_outputs = outputs
        self.robustness_replay_spec = (
            _ReplaySpec(replay_outputs) if replay_outputs is not None else None
        )


class _Plan:
    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)


def _details(*steps: Any) -> List[dict]:
    return [
        finding.detail or {}
        for finding in product_promise_plan_findings(plan=_Plan(*steps))
    ]


def test_the_real_recorded_step_is_flagged_with_its_product_and_kinds() -> None:
    """``07_missingness_robustness_replay`` as the Planner really wrote it."""

    details = _details(_Step("07_missingness_robustness_replay", _REAL_STEP_OUTPUTS))

    assert len(details) == 1
    assert details[0]["reason"] == "product_promised_as_statistic_and_more"
    assert details[0]["step_id"] == "07_missingness_robustness_replay"
    assert details[0]["product"] == "robustness_summary"
    assert sorted(details[0]["kinds"]) == ["statistic", "table"]


def test_replay_owner_tells_the_planner_exactly_which_kind_to_keep() -> None:
    """The failed Web E1 replan guessed statistic; the owner writes a table."""

    findings = product_promise_plan_findings(
        plan=_Plan(
            _Step(
                "06_robustness_replay",
                ["table:robustness_summary", "statistic:robustness_summary"],
                replay_outputs={"robustness_summary": "robustness_summary"},
            )
        )
    )

    assert len(findings) == 1
    assert findings[0].detail["required_kind"] == "table"
    assert findings[0].detail["required_promise"] == "table:robustness_summary"
    assert "Keep exactly 'table:robustness_summary'" in findings[0].message


def test_wrong_single_kind_is_repaired_even_after_collision_was_removed() -> None:
    """A replan that removes the correct kind must not reach execution."""

    wrong = product_promise_plan_findings(
        plan=_Plan(
            _Step(
                "06_robustness_replay",
                ["statistic:robustness_summary"],
                replay_outputs={"robustness_summary": "robustness_summary"},
            )
        )
    )
    correct = product_promise_plan_findings(
        plan=_Plan(
            _Step(
                "06_robustness_replay",
                ["table:robustness_summary"],
                replay_outputs={"robustness_summary": "robustness_summary"},
            )
        )
    )

    assert [finding.detail["reason"] for finding in wrong] == [
        "product_kind_conflicts_with_owner"
    ]
    assert wrong[0].detail["required_promise"] == "table:robustness_summary"
    assert correct == []


def test_an_artifact_and_table_pair_is_deliberately_left_alone() -> None:
    """The 10 recorded non-statistic collisions.

    A dataset offered as both an artifact and a table may be one thing in two
    forms; nothing has shown it wrong, and blocking it would cost a healthy
    step a replan.  Only the statistic pair is provably two artifacts.
    """

    assert (
        _details(
            _Step("09_clusters", ["artifact:cluster_assignments", "table:cluster_assignments"])
        )
        == []
    )


def test_a_step_promising_each_product_once_is_not_flagged() -> None:
    """The ordinary case must stay silent, or the gate is just noise."""

    assert (
        _details(
            _Step(
                "06_primary",
                ["table:adjusted_association_estimates", "statistic:primary_or"],
            )
        )
        == []
    )


def test_the_same_token_twice_is_not_a_two_kind_promise() -> None:
    """A repeated identical promise is one kind, however many times listed.

    Deliberately repeats the ``statistic:`` token: counting occurrences rather
    than distinct kinds would flag this, and it is the one shape where that
    mistake produces a false positive rather than a harmless miss.
    """

    assert _details(_Step("x", ["statistic:robustness_summary"] * 3)) == []


def test_a_token_with_an_empty_kind_is_ignored_rather_than_flagged() -> None:
    """Plans are model output; a malformed entry must not become a finding.

    ``':robustness_summary'`` parses to an empty kind for a real product name.
    Admitting it would pair that empty kind with the genuine ``statistic:``
    promise and report a collision the step does not have.
    """

    details = _details(
        _Step(
            "x",
            ["statistic:robustness_summary", ":robustness_summary", "", "no_kind"],
        )
    )

    assert details == []


def test_every_offending_step_in_a_plan_is_reported_not_just_the_first() -> None:
    """A plan repairs in one round only if it is told about every step."""

    details = _details(
        _Step("07", ["table:robustness_summary", "statistic:robustness_summary"]),
        _Step("08", ["figure:robustness", "statistic:robustness"]),
    )

    assert [item["step_id"] for item in details] == ["07", "08"]


def test_the_finding_tells_the_planner_to_delete_a_promise_not_declare_a_field() -> None:
    """The lesson from the ownership gate's own scar.

    That gate's wording machinery says "does not declare X ... declare the
    missing field(s)".  Here nothing is absent: one promise too many is
    present, and the repair is removal.  Sending the Planner the other sentence
    would be false about the step and wrong about the fix.
    """

    findings = product_promise_plan_findings(
        plan=_Plan(_Step("07", ["table:robustness_summary", "statistic:robustness_summary"]))
    )
    message = findings[0].message

    assert "robustness_summary" in message
    assert "statistic:robustness_summary" in message
    assert "table:robustness_summary" in message
    assert "delete the other promise" in message
    assert "does not declare" not in message
    assert "Declare the missing field" not in message


def test_the_finding_does_not_forbid_the_repair_it_demands() -> None:
    """A demand that forbids itself is unsatisfiable, so nothing is a defensible reading."""

    message = product_promise_plan_findings(
        plan=_Plan(_Step("07", ["table:robustness_summary", "statistic:robustness_summary"]))
    )[0].message

    assert "delete the other promise" in message
    assert "do not delete" not in message.lower()
    # The repair is a promise edit, so the prohibitions must not cover it.
    assert "Do not rename the product" in message
    assert "do not change any scientific" in message


def test_it_blocks_rather_than_advises() -> None:
    """The step cannot be claimed by anyone; a warning would just be ignored."""

    findings = product_promise_plan_findings(
        plan=_Plan(_Step("07", ["table:robustness_summary", "statistic:robustness_summary"]))
    )

    assert [f.severity for f in findings] == ["error"]
    assert [f.validator for f in findings] == ["plan_product_promise"]


def test_the_directive_is_absent_when_there_is_nothing_to_repair() -> None:
    """A directive with no findings would send the Planner to fix nothing."""

    assert product_promise_replan_directive([]) is None
    assert (
        product_promise_replan_directive(
            product_promise_plan_findings(plan=_Plan(_Step("ok", ["table:x"])))
        )
        is None
    )


def test_the_directive_carries_the_same_instruction_as_the_finding() -> None:
    """They travel to the Planner together; drift between them is the bug."""

    findings = product_promise_plan_findings(
        plan=_Plan(_Step("07", ["table:robustness_summary", "statistic:robustness_summary"]))
    )
    directive = product_promise_replan_directive(findings)

    assert directive is not None
    assert "keep the product under the one kind" in directive
    assert "delete the other promise" in directive
    assert "robustness_summary" in directive
    assert "declare the missing" not in directive.lower()


def test_the_directive_ignores_another_gate_s_findings() -> None:
    """Each gate owns its own wording; mixing them is how a directive contradicts itself."""

    from easyicu.research_agent.schema import ValidationFinding

    foreign = ValidationFinding(
        validator="plan_owner_declaration",
        severity="error",
        message="Step 06 does not declare 'model_requirements[0].covariates'.",
        detail={"reason": "owner_declaration_incomplete"},
    )

    assert product_promise_replan_directive([foreign]) is None


def _called_names(source: str) -> list[str]:
    tree = ast.parse(source)
    return [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]


def test_the_gate_runs_in_the_plan_time_preflight() -> None:
    """A gate nobody calls is the defect this whole file exists to close.

    ``_module_level_unbound_names`` sat in a test file for a month and checked
    nobody's real code; this asserts the wiring rather than the helper.
    """

    source = _PHASE.read_text(encoding="utf-8")
    called = _called_names(source)

    # Once to generate the bounded replan directive, then again to verify the
    # model-authored revision before any executor is selected.
    assert called.count("product_promise_plan_findings") == 2
    assert called.count("product_promise_replan_directive") == 1


def test_the_findings_both_force_a_replan_and_reach_the_directive() -> None:
    """Producing findings nobody acts on is the same as not checking."""

    tree = ast.parse(_PHASE.read_text(encoding="utf-8"))
    reads = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == "product_promise_preflight"
    ]

    # One read feeds the directive, one feeds the force= disjunction.
    assert len(reads) >= 2
