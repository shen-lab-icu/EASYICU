"""An owner must say *why* it declined, and the two whys must stay apart.

Before :class:`OwnershipVerdict` every ``*_owns_step`` predicate returned
``bool``, so "this is not my contract" and "this IS my contract but the
Planner left one of its own fields blank" were the same answer, and the host
took the same action for both: hand the step to the stochastic coder and say
nothing.  Measured over 553 recorded real steps, that silently sent 26 primary
adjusted-association models -- the paper's headline estimate -- down the coder
path when a deterministic owner was one declared adjustment set away.

These tests lock the distinction at three places: the type refuses to be
constructed ambiguously, the association owner classifies each real clause
correctly, and the selector's trace carries the gap so the ledger and the
pre-registration gate can read it without re-deriving ownership.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.ownership_verdict import OwnershipVerdict
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
    ADJUSTED_ASSOCIATION_OUTPUT,
    adjusted_association_executor_owns_step,
    adjusted_association_executor_verdict,
)
from easyicu.research_agent.schema import AnalysisStep

from .test_adjusted_association_executor import _model_terms, _real_step_payload


# ---------------------------------------------------------------------------
# 1. The type refuses to be built in a way that erases the distinction
# ---------------------------------------------------------------------------


def test_incomplete_declaration_must_name_at_least_one_field():
    """An unnamed gap is a bool wearing a new type."""

    with pytest.raises(ValueError, match="at least one missing field"):
        OwnershipVerdict.incomplete_declaration(
            "kind", missing=(), reason="something is missing"
        )


def test_a_missing_field_name_may_not_be_blank():
    with pytest.raises(ValueError, match="must name a field"):
        OwnershipVerdict.incomplete_declaration(
            "kind", missing=("  ",), reason="something is missing"
        )


def test_every_verdict_must_carry_a_reason():
    for build in (
        lambda: OwnershipVerdict.claim("kind", reason=""),
        lambda: OwnershipVerdict.wrong_shape("kind", reason="   "),
    ):
        with pytest.raises(ValueError, match="must carry a reason"):
            build()


def test_a_claim_cannot_also_report_a_gap():
    with pytest.raises(ValueError, match="cannot also report missing"):
        OwnershipVerdict(
            analysis_kind="kind",
            claimed=True,
            reason="claimed",
            missing_declarations=("covariates",),
        )


def test_the_verdict_is_not_usable_as_a_bool():
    """``if verdict:`` would silently collapse exactly what this type separates.

    A truthiness protocol would let every ``if owner_owns_step(step):`` keep
    compiling while reading a wrong-shape decline and a declaration gap the
    same way -- the failure being replaced, reintroduced by convenience.
    """

    assert "__bool__" not in vars(OwnershipVerdict)
    gap = OwnershipVerdict.incomplete_declaration(
        "kind", missing=("covariates",), reason="no adjustment set"
    )
    wrong = OwnershipVerdict.wrong_shape("kind", reason="not my contract")
    # Both are declines; only one is actionable, and ``claimed`` is how a
    # caller asks. ``declaration_is_incomplete`` is how the gate asks.
    assert gap.claimed is False and wrong.claimed is False
    assert gap.declaration_is_incomplete is True
    assert wrong.declaration_is_incomplete is False


# ---------------------------------------------------------------------------
# 2. The association owner classifies each real clause correctly
# ---------------------------------------------------------------------------


#: The recorded fresh19 step is itself the measured case: it declares the
#: primary model and **no** adjustment set. Every test that wants a complete
#: declaration has to add one, which is the finding, not a fixture quirk.
_COVARIATES = ("age", "sex", "charlson_max")


def _step(*, covariates=_COVARIATES, **overrides) -> AnalysisStep:
    payload = json.loads(json.dumps(_real_step_payload()))
    if covariates is not None:
        payload["model_requirements"][0]["covariates"] = list(covariates)
        payload["model_requirements"][0]["model_terms"] = _model_terms(covariates)
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def test_the_recorded_step_declares_no_adjustment_set():
    """Anchor the measurement in the artifact, not in a hand-built fixture."""

    recorded = _real_step_payload()
    assert recorded["model_requirements"][0].get("covariates") is None


def test_a_completely_declared_step_is_claimed():
    verdict = adjusted_association_executor_verdict(_step())
    assert verdict.claimed is True
    assert verdict.analysis_kind == ADJUSTED_ASSOCIATION_ANALYSIS_KIND
    assert verdict.missing_declarations == ()


def test_a_declared_model_without_an_adjustment_set_is_a_declaration_gap():
    """The measured 26-step case: one model, no covariates.

    This is the verdict the host must act on. Reconstructing an adjustment set
    from ``step.inputs`` would be inference, so the owner is right to decline
    -- and wrong to let the host silently substitute the coder for it.
    """

    verdict = adjusted_association_executor_verdict(_step(covariates=None))

    assert verdict.claimed is False
    assert verdict.declaration_is_incomplete is True
    assert verdict.missing_declarations == ("model_requirements[0].covariates",)


def test_a_multi_model_step_is_a_wrong_shape_not_a_declaration_gap():
    """The measured 33-step case, and it must NOT be reported as a gap.

    28 recorded steps declare two model requirements and 5 declare three;
    none declares zero. More declaring is not what would let this owner claim
    them -- ``bind_primary_output`` binds a one-row table -- so calling it a
    missing declaration would send the Planner to fix something that is not
    broken. It is task #105's question of whether an owner's claim may depend
    on Planner bundling at all.
    """

    payload = json.loads(json.dumps(_real_step_payload()))
    payload["model_requirements"] = [
        payload["model_requirements"][0],
        {
            **payload["model_requirements"][0],
            "requirement_id": "second_model",
            "analysis_role": "secondary",
        },
    ]
    verdict = adjusted_association_executor_verdict(
        AnalysisStep.model_validate(payload)
    )

    assert verdict.claimed is False
    assert verdict.declaration_is_incomplete is False
    assert "2 model requirements" in verdict.reason


def test_a_foreign_method_is_a_wrong_shape():
    """``model_requirements`` must be dropped with the method, and that is a fact.

    ``AnalysisStep`` already refuses to carry model requirements on a step
    whose method is not this owner's, so a foreign-method step reaching this
    predicate never has them. The clause is still reachable -- any step of
    another family is consulted here before its own owner -- and must decline
    as a wrong shape, not as something the Planner forgot to declare.
    """

    verdict = adjusted_association_executor_verdict(
        _step(covariates=None, method="visualization", model_requirements=[])
    )
    assert verdict.claimed is False
    assert verdict.declaration_is_incomplete is False


def test_a_bundled_product_step_is_a_wrong_shape():
    """The measured 5-step case: this product plus another in one step."""

    verdict = adjusted_association_executor_verdict(
        _step(
            expected_outputs=[ADJUSTED_ASSOCIATION_OUTPUT, "figure:primary_association"]
        )
    )
    assert verdict.claimed is False
    assert verdict.declaration_is_incomplete is False
    assert "2 expected output(s)" in verdict.reason


# ---------------------------------------------------------------------------
# 3. One predicate, not two
# ---------------------------------------------------------------------------


def test_the_bool_wrapper_delegates_instead_of_re_testing_the_clauses():
    """Two copies of one ownership rule drifting apart is the recurring defect.

    Asserting agreement on a handful of shapes would not catch a second copy
    that agrees today; this reads the wrapper's body and requires it to be a
    call to the verdict, so a future clause added to only one of them cannot
    exist.
    """

    tree = ast.parse(inspect.getsource(adjusted_association_executor_owns_step))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls == {"adjusted_association_executor_verdict"}, (
        "the bool wrapper must delegate to the verdict, not re-test clauses: "
        f"it calls {sorted(calls)}"
    )
    # ... and it must return that call's ``claimed``, not a fresh judgement.
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Attribute)
    assert returns[0].value.attr == "claimed"


# ---------------------------------------------------------------------------
# 4. The gap survives the boundary the ledger and the gate read
# ---------------------------------------------------------------------------


def test_the_selector_trace_carries_the_declaration_gap():
    """Reporting reads the decider's own trace; it must not re-run predicates.

    ``StandardExecutorCandidate`` already existed for that reason. Without the
    gap on it, every consumer would have to re-derive ownership to learn what
    the owner was waiting for -- a second registry, which is what the trace
    exists to prevent.
    """

    from easyicu.research_agent.execution.runners.selection import (
        select_standard_executor,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    payload = json.loads(json.dumps(_real_step_payload()))
    payload["model_requirements"][0].pop("covariates", None)
    step = AnalysisStep.model_validate(payload)
    plan = AnalysisPlan.model_validate(
        {
            "research_question": "Is the exposure associated with the outcome?",
            "steps": [payload],
            "rationale": "declaration-gap trace regression",
        }
    )

    trace: list = []
    assert select_standard_executor(step, plan=plan, trace=trace) is None

    gaps = {
        candidate.analysis_kind: tuple(candidate.missing_declarations)
        for candidate in trace
        if candidate.missing_declarations
    }
    assert gaps == {
        ADJUSTED_ASSOCIATION_ANALYSIS_KIND: ("model_requirements[0].covariates",)
    }
    gap_candidate = next(
        candidate for candidate in trace if candidate.missing_declarations
    )
    assert gap_candidate.decline_reason, "a gap without a reason is undiagnosable"


def test_the_ledger_reports_the_gap_bucket_separately(capsys):
    """The ledger's third number is the roadmap; assert the printed report.

    Grepping the tool's source would pass on a string that is never printed,
    and fail on one that is printed but wrapped across two source lines. Run
    the reporter and read what a human actually sees.
    """

    import sys

    sys.path.insert(0, str(Path("tools").resolve()))
    from measure_executor_ownership import (  # noqa: PLC0415
        OwnershipLedger,
        StepOwnership,
        _report,
    )

    ledger = OwnershipLedger(readable_plans=1)
    ledger.steps = [
        StepOwnership(
            key="a",
            step_id="01_claimed",
            method="grouped_table_one",
            declared_products=("table:table_one",),
            upper_owner="grouped_table_one",
            lower_owner="grouped_table_one",
            upper_trace=(),
        ),
        StepOwnership(
            key="b",
            step_id="02_gap",
            method="adjusted_association_models",
            declared_products=(ADJUSTED_ASSOCIATION_OUTPUT,),
            upper_owner=None,
            lower_owner=None,
            upper_trace=(),
            declaration_gaps=(
                (
                    ADJUSTED_ASSOCIATION_ANALYSIS_KIND,
                    ("model_requirements[0].covariates",),
                ),
            ),
        ),
        StepOwnership(
            key="c",
            step_id="03_unowned",
            method="visualization",
            declared_products=("figure:whatever",),
            upper_owner=None,
            lower_owner=None,
            upper_trace=(),
        ),
    ]

    _report(ledger, top=10)
    printed = capsys.readouterr().out

    assert "ONE DECLARATION AWAY    :     1" in printed
    assert "model_requirements[0].covariates" in printed
    # The gap must not be folded into either edge: 1 claimed, 1 gap, 1 neither.
    assert "claimed, upper bound (no receipt owed):     1" in printed
    # And it must be labelled a lower bound, because only owners already
    # converted to a typed verdict can contribute to it.
    assert "lower bound" in printed
