"""The host publishes a cohort spelling the development plane cannot read.

canary20 ran on a sealed 1,000-row development sample.  Step 06's typed input
was ``cohort:bench_e1_sepsis3_prevalence_mortality`` -- the plan's own locked
cohort name, which the host's *own* published vocabulary
(``closed_cohort_product_vocabulary``) offers the Planner as
``cohort:<exact cohort.name>``.  The typed-input plane asked
``reserved_primary_cohort_product``, which knows only ``analysis_cohort`` and
``cohort:analysis_set``, so it skipped the development projection and bound the
full 94,458-row cohort while the run-level plane mounted and reported the
1,000-row sample.

The primary model was then fitted on 94,425 rows / 9,461 events and judged
against a contract expecting 1,000 / 102.  The resulting
``model_denominator_or_event_mismatch`` is what spent the step's two repairs on
a host-owned scaffold and killed it -- the repair was the last domino, not the
cause.

Measured over 819 recorded real plans / 8,051 steps: 3,995 typed primary-cohort
inputs, of which 3,959 used a reserved spelling and **36 used the plan's own
cohort name**.  None was unrecognisable by both readers, so recognition goes
from 3,959/3,995 to 3,995/3,995 and no new fail-closed path is reachable.

This is the second time this exact hole has been patched one spelling at a
time: ``cohort:analysis_set`` was added when the same split was found before.
So the load-bearing test here is not any single spelling -- it is
:func:`test_every_published_spelling_reaches_the_sample`, which asserts that
what the host *offers* the Planner and what this plane can *read* are the same
set.  A third spelling added to the directive fails there instead of in a run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.development_projection import (
    resolve_development_input_projection,
)
from easyicu.research_agent.authority.typed_binding import (
    TypedBindingResolver,
    _resolved_typed_input_binding,
    _resume_typed_input_bindings,
)
from easyicu.research_agent.authority.plan_scope import (
    _serializable_plan_scientific_scope_signature,
)
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.contracts.declared_product import (
    _primary_analysis_cohort_product_matches_plan,
    locked_primary_cohort_product,
    reserved_primary_cohort_product,
)
from easyicu.research_agent.execution.development_sample import (
    DEVELOPMENT_COHORT_EVIDENCE_ID,
)
from easyicu.research_agent.execution.runners.typed_input_binding import (
    CLOSED_COHORT_PRODUCT_KIND,
    closed_cohort_product_vocabulary,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, EvidenceRef

from tests.research_agent.core.test_development_typed_binding_projection import (
    _Lock,
    _arrange_projection,
    _register_full_universe_producer_output,
)

LOCKED = "bench_e1_sepsis3_prevalence_mortality"


# --- the reader: one question, one answer ------------------------------------


def test_the_locked_name_is_the_same_population():
    assert locked_primary_cohort_product(
        f"{CLOSED_COHORT_PRODUCT_KIND}:{LOCKED}", locked_cohort_name=LOCKED
    ) == ("dataset", LOCKED)


def test_the_shipped_reader_could_not_see_it():
    """The defect itself, kept as a regression rather than described."""

    assert reserved_primary_cohort_product(f"cohort:{LOCKED}") is None


@pytest.mark.parametrize(
    ("raw", "locked"),
    [
        (f"cohort:{LOCKED}", None),
        (f"cohort:{LOCKED}", ""),
        (f"cohort:{LOCKED}", "   "),
        ("cohort:some_other_population", LOCKED),
        (f"artifact:{LOCKED}", LOCKED),
        (f"table:{LOCKED}", LOCKED),
        ("table:model_design_matrix", LOCKED),
        ("", LOCKED),
        (None, LOCKED),
    ],
    ids=[
        "no-locked-name-known",
        "blank-locked-name",
        "whitespace-locked-name",
        "a-different-cohort",
        "locked-name-outside-the-cohort-namespace",
        "locked-name-as-a-plain-table",
        "not-a-cohort-product",
        "empty",
        "none",
    ],
)
def test_it_does_not_widen_beyond_the_locked_population(raw, locked):
    """A reader that decides which population a step sees must not guess.

    The name-outside-``cohort:`` cases matter most: ``artifact:<locked name>``
    is a step artifact that happens to share the cohort's name, not a claim on
    the locked population, and treating it as one would substitute the sample
    for a table the sample is not a subset of.
    """

    assert locked_primary_cohort_product(raw, locked_cohort_name=locked) is None


def test_the_reserved_spellings_still_answer_without_a_locked_name():
    for raw in ("artifact:analysis_cohort", "cohort:analysis_set"):
        assert locked_primary_cohort_product(raw, locked_cohort_name=None) is not None


@pytest.mark.parametrize(
    "raw",
    [
        f"cohort:{LOCKED}",
        "cohort:analysis_set",
        "artifact:analysis_cohort",
        "dataset:analysis_cohort",
        "table:analysis_cohort",
        "cohort:some_other_population",
        "table:model_design_matrix",
    ],
)
def test_the_plan_aware_caller_shares_this_one_implementation(raw):
    """Not "both are correct" -- the same object, so they cannot drift apart.

    A copy of this rule in the plan-aware reader is exactly how the two planes
    disagreed in the first place.
    """

    class _Plan:
        cohort = type("C", (), {"name": LOCKED})()

    assert _primary_analysis_cohort_product_matches_plan(
        raw, plan=_Plan
    ) == locked_primary_cohort_product(raw, locked_cohort_name=LOCKED)


# --- the plane: the sample is what the step actually reads --------------------


def _binding(tmp_path: Path, input_name: str, *, locked_cohort_name):
    store, _parent, sample, _plan, records = _arrange_projection(tmp_path)
    producer_output = _register_full_universe_producer_output(
        tmp_path,
        store,
        records,
        frame=pd.read_parquet(tmp_path / "cohort_analysis.parquet"),
        evidence_id="step01_locked_named_cohort",
    )
    binding = _resolved_typed_input_binding(
        input_name=input_name,
        evidence_ref=EvidenceRef(evidence_id=producer_output.evidence_id),
        evidence_records=store.records(),
        run_dir=tmp_path,
        producer_step_records=records,
        authoritative_cohort_path=sample.cohort_path,
        development_sample=sample,
        locked_cohort_name=locked_cohort_name,
    )
    return binding, sample, producer_output


def test_the_locked_name_binds_the_sample(tmp_path: Path) -> None:
    binding, sample, producer_output = _binding(
        tmp_path, f"cohort:{LOCKED}", locked_cohort_name=LOCKED
    )

    assert binding is not None
    assert binding["evidence_id"] == DEVELOPMENT_COHORT_EVIDENCE_ID
    assert binding["sha256"] == sample.sample_sha256
    assert binding["product_contract"]["row_count"] == sample.selected_rows
    projection = binding["execution_projection"]
    assert projection["paper_authority"] is False
    # The declared parent is still the producer's full output: the sample
    # replaces the physical bytes, never the lineage.
    assert projection["declared_parent_input"]["evidence_id"] == (
        producer_output.evidence_id
    )
    assert projection["locked_parent_cohort_sha256"] == sample.parent_sha256


def test_without_the_locked_name_it_binds_the_full_cohort(tmp_path: Path) -> None:
    """canary20's behaviour exactly, so a revert is visible here.

    This is the silent half of the defect: no error, no finding -- the step
    simply reads 94x the rows the run says it is running on.
    """

    binding, sample, producer_output = _binding(
        tmp_path, f"cohort:{LOCKED}", locked_cohort_name=None
    )

    assert binding is not None
    assert binding["evidence_id"] == producer_output.evidence_id
    assert binding["sha256"] != sample.sample_sha256
    assert binding["product_contract"]["row_count"] > sample.selected_rows
    assert "execution_projection" not in binding or not binding["execution_projection"]


def test_a_cohort_named_something_else_is_not_substituted(tmp_path: Path) -> None:
    """Fail-open in the other direction would hand a step the wrong population."""

    binding, sample, producer_output = _binding(
        tmp_path, "cohort:a_different_population", locked_cohort_name=LOCKED
    )

    assert binding is not None
    assert binding["evidence_id"] == producer_output.evidence_id
    assert binding["sha256"] != sample.sample_sha256


# --- through the two real entry points, not just the helper ------------------
#
# The helper tests above all passed while BOTH production call sites were
# mutated to drop ``locked_cohort_name`` -- the fix lived only in the tests.
# These two drive the functions the pipeline actually calls.


def _locked_plan() -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Bind the locked population by its own name.",
        cohort=CohortDefinition(name=LOCKED, locked_at="2026-07-31T00:00:00Z"),
        steps=[
            AnalysisStep(
                step_id="01_cohort",
                intent="Lock the full analysis cohort.",
                expected_outputs=[f"cohort:{LOCKED}"],
            ),
            AnalysisStep(
                step_id="02_model",
                intent="Fit the primary model on the locked population.",
                inputs=[f"cohort:{LOCKED}"],
            ),
        ],
    )


def _arrange_locked_named_producer(tmp_path: Path):
    store, _parent, sample, _plan, records = _arrange_projection(tmp_path)
    plan = _locked_plan()
    # Registered under the product name so lineage resolution finds it: this
    # test is about which population the projection picks, not about how the
    # producer artifact is located.
    producer_output = _register_full_universe_producer_output(
        tmp_path,
        store,
        records,
        frame=pd.read_parquet(tmp_path / "cohort_analysis.parquet"),
        evidence_id=LOCKED,
    )
    records[0]["analysis_request"] = {"step": plan.steps[0].model_dump(mode="json")}
    records[0]["plan_scientific_signature"] = (
        _serializable_plan_scientific_scope_signature(plan)
    )
    return store, sample, plan, records, producer_output


def test_the_resume_path_binds_the_sample(tmp_path: Path) -> None:
    store, sample, plan, records, _producer = _arrange_locked_named_producer(tmp_path)

    bindings, _evidence_ids = _resume_typed_input_bindings(
        step=plan.steps[1],
        plan=plan,
        evidence_records=store.records(),
        trusted_step_records=records,
        run_dir=tmp_path,
        cohort_path=sample.cohort_path,
        development_sample=sample,
    )

    binding = bindings[f"cohort:{LOCKED}"]
    assert binding["evidence_id"] == DEVELOPMENT_COHORT_EVIDENCE_ID
    assert binding["sha256"] == sample.sample_sha256
    assert binding["product_contract"]["row_count"] == sample.selected_rows


def test_the_resolver_path_binds_the_sample(tmp_path: Path) -> None:
    store, sample, plan, records, _producer = _arrange_locked_named_producer(tmp_path)

    resolver = TypedBindingResolver(
        evidence_store=store,
        per_step_records=records,
        records_lock=_Lock(),
        run_dir=tmp_path,
        authoritative_cohort_path=sample.cohort_path,
        development_sample=sample,
    )
    _refs, _evidence_ids, bindings = resolver.resolve_names(
        [f"cohort:{LOCKED}"],
        plan=plan,
        consumer_step=plan.steps[1],
    )

    binding = bindings[f"cohort:{LOCKED}"]
    assert binding["evidence_id"] == DEVELOPMENT_COHORT_EVIDENCE_ID
    assert binding["sha256"] == sample.sample_sha256
    assert binding["product_contract"]["row_count"] == sample.selected_rows


# --- and the two surfaces stay in agreement ----------------------------------


def test_every_published_spelling_reaches_the_sample(tmp_path: Path) -> None:
    """What the host offers the Planner is what this plane can read.

    ``closed_cohort_product_vocabulary`` is rendered verbatim into the planner
    directive.  Every spelling in it is therefore a legal declaration, and a
    legal declaration that this plane cannot recognise silently runs the step
    on a different population than the run reports.  Adding a spelling to the
    directive without teaching this reader fails here.
    """

    store, _parent, sample, _plan, records = _arrange_projection(tmp_path)
    producer_output = _register_full_universe_producer_output(
        tmp_path,
        store,
        records,
        frame=pd.read_parquet(tmp_path / "cohort_analysis.parquet"),
        evidence_id="step01_published_vocabulary",
    )

    published = [
        spelling.replace("<exact cohort.name>", LOCKED)
        for spelling in closed_cohort_product_vocabulary()
    ]
    assert f"cohort:{LOCKED}" in published, (
        "the spelling canary20 used must still be one the host publishes; "
        "if it is not, this test is measuring the wrong vocabulary"
    )

    unreadable = []
    for spelling in published:
        binding = _resolved_typed_input_binding(
            input_name=spelling,
            evidence_ref=EvidenceRef(evidence_id=producer_output.evidence_id),
            evidence_records=store.records(),
            run_dir=tmp_path,
            producer_step_records=records,
            authoritative_cohort_path=sample.cohort_path,
            development_sample=sample,
            locked_cohort_name=LOCKED,
        )
        if binding is None or binding["evidence_id"] != DEVELOPMENT_COHORT_EVIDENCE_ID:
            unreadable.append(spelling)

    assert not unreadable, (
        "the host publishes these spellings to the Planner but the typed-input "
        f"plane binds the full cohort for them: {unreadable}"
    )
