from __future__ import annotations

import pytest

import ast
import importlib
import inspect
from pathlib import Path

from easyicu.research_agent.cohort import schema as cohort_schema
from easyicu.research_agent import schema
from easyicu.research_agent.planning import cohort_contract, robustness_contract


def _imports(module) -> set[str]:
    tree = ast.parse(Path(inspect.getsourcefile(module)).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_robustness_contract_runtime_exports_are_identical() -> None:
    runtime_panel = importlib.import_module("easyicu.research_agent.robustness.panel")
    for name in (
        "MIN_AXIS_COUNTS",
        "RobustnessPlanError",
        "RobustnessSpec",
        "validate_robustness_specs",
    ):
        assert getattr(runtime_panel, name) is getattr(robustness_contract, name)


def test_schema_uses_the_pure_robustness_contract() -> None:
    assert schema.RobustnessPlanError is robustness_contract.RobustnessPlanError
    assert schema.RobustnessSpec is robustness_contract.RobustnessSpec
    # The validator schema actually calls: the STRUCTURAL one, and only it.
    #
    # This assertion used to demand the planner-scoped validator here, which is
    # the arrangement that was tried and measured: constructing an
    # ``AnalysisPlan`` is not the same act as accepting one from the Planner --
    # the same constructor loads a recorded plan from disk, re-reads a lock on
    # resume, and builds the framework's own case-neutral placeholders. Asking
    # all of those what the *Planner must declare* stopped 190 of 409 recorded
    # plan documents from parsing, and a resume of any of them would have
    # failed at load.
    assert (
        schema.validate_robustness_specs
        is robustness_contract.validate_robustness_specs
    )
    # The negative half is the one that protects those 190: schema must not
    # even import the planner-scoped rule, or the revert can quietly return.
    assert not hasattr(schema, "validate_planner_robustness_specs")
    assert not hasattr(schema, "_validate_robustness_specs_locally")


def test_the_planner_rule_still_runs_where_planner_output_is_accepted() -> None:
    """Moving the rule off the constructor must not lose it.

    The property the old assertion was reaching for -- "the wiring can revert
    and only a real run would notice" -- is real; it was just pointed at the
    wrong object. The Planner-output rule belongs at the single point where
    Planner output is accepted and the Planner can still answer for it.
    """

    # ``agents/core.py`` is now a compatibility facade; ``PlannerAgent`` and
    # its ``_parse`` acceptance point live in the ``agents.planner`` owner
    # module, so the rule is pinned where it actually runs.
    planner = importlib.import_module("easyicu.research_agent.agents.planner")
    assert (
        planner.validate_planner_robustness_specs
        is robustness_contract.validate_planner_robustness_specs
    )
    # Imported is not called: pin the call site too.
    tree = ast.parse(Path(inspect.getsourcefile(planner)).read_text(encoding="utf-8"))
    parse_bodies = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_parse"
    ]
    assert parse_bodies, "PlannerAgent._parse is where Planner output is accepted"
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "validate_planner_robustness_specs"
        for body in parse_bodies
        for call in ast.walk(body)
        if isinstance(call, ast.Call)
    ), "the Planner-output rule is imported but never applied"


def test_robustness_contract_has_no_runtime_or_evidence_dependency() -> None:
    imported = _imports(robustness_contract)
    forbidden = {
        "easyicu.research_agent.authority.evidence_store",
        "easyicu.research_agent.authority.lock_contract",
        "easyicu.research_agent.robustness.panel",
        "easyicu.research_agent.authority.runtime_artifacts",
    }
    assert not imported & forbidden
    assert all(not name.endswith("pipeline") for name in imported)


def test_robustness_spec_round_trip_preserves_payload() -> None:
    spec = robustness_contract.RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Complete-case sensitivity",
        missing_override={"strategy": "complete_case"},
    )
    restored = robustness_contract.RobustnessSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_cohort_contract_legacy_exports_are_identical() -> None:
    for name in (
        "ALLOWED_CTAS_AGGREGATIONS",
        "Aggregation",
        "CohortDefinition",
        "CohortSchemaError",
        "ConceptPredicate",
        "PatternRegistry",
        "PredicateOp",
        "TimeAnchor",
        "TimeWindow",
        "UNIVERSAL_ANCHORS",
        "clear_cohort_concept_ids",
        "coerce_cohort_definition",
        "cohort_definition_sha",
        "concept_id_exists",
        "default_pattern_registry",
        "ensure_cohort_definition",
        "expand_named_cohort",
        "known_concept_ids",
        "register_cohort_concept_ids",
        "register_pattern",
        "register_patterns_from_file",
        "reset_pattern_registry",
        "validate_cohort_definition",
        "validate_concept_predicate",
    ):
        assert getattr(cohort_schema, name) is getattr(cohort_contract, name)


def test_schema_uses_the_pure_cohort_contract() -> None:
    assert schema.CohortDefinition is cohort_contract.CohortDefinition
    assert schema.CohortSchemaError is cohort_contract.CohortSchemaError


def test_cohort_contract_has_no_runtime_or_evidence_dependency() -> None:
    imported = _imports(cohort_contract)
    forbidden_suffixes = (
        "cohort_schema",
        "evidence",
        "lock_authority",
        "pipeline",
        "runtime_artifacts",
    )
    assert not any(name.endswith(forbidden_suffixes) for name in imported)


def test_cohort_contract_owns_one_process_registry() -> None:
    assert (
        cohort_schema.default_pattern_registry()
        is cohort_contract.default_pattern_registry()
    )
    cohort_contract.clear_cohort_concept_ids()
    try:
        cohort_schema.register_cohort_concept_ids(["materialized_signal"])
        assert cohort_contract.concept_id_exists("materialized_signal")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_cohort_concept_id_scope_restores_the_exact_prior_registry() -> None:
    """A caller asking a hypothetical must not answer it for everyone else.

    ``clear_cohort_concept_ids`` empties the registry wholesale, so a scoped
    question ("would this plan validate if these columns existed?") could only
    be asked by destroying a registration its owner still needs.  The scope
    must restore what was there, including ids registered before it ran.
    """

    cohort_contract.clear_cohort_concept_ids()
    try:
        cohort_contract.register_cohort_concept_ids(["already_owned"])
        with cohort_schema.cohort_concept_id_scope(["hypothetical"]):
            assert cohort_contract.concept_id_exists("hypothetical")
            assert cohort_contract.concept_id_exists("already_owned")
        assert not cohort_contract.concept_id_exists("hypothetical")
        assert cohort_contract.concept_id_exists("already_owned")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_cohort_concept_id_scope_restores_even_when_the_block_raises() -> None:
    cohort_contract.clear_cohort_concept_ids()
    try:
        with pytest.raises(ValueError):
            with cohort_schema.cohort_concept_id_scope(["hypothetical"]):
                raise ValueError("validation failed inside the scope")
        assert not cohort_contract.concept_id_exists("hypothetical")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_ensure_cohort_definition_revalidates_against_run_context_without_leak() -> (
    None
):
    from easyicu.research_agent.schema import (
        AnalysisPlan,
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
    )

    concept_id = "run_specific_materialized_column"
    with cohort_contract.cohort_concept_id_scope([concept_id]):
        definition = cohort_contract.CohortDefinition(
            name="sealed",
            selection_mode="predicate_filtered",
            inclusion=(
                cohort_contract.ConceptPredicate(
                    concept_id=concept_id,
                    time_window=cohort_contract.TimeWindow("icu_admit", 0, 24),
                    aggregation="max",
                    op=">=",
                    value=1,
                ),
            ),
        )
    plan = AnalysisPlan.model_construct(
        research_question="Use the sealed run-specific column.",
        steps=[],
        cohort=definition,
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="sealed",
            database="synthetic",
            n_stays=2,
            id_columns=["stay_id"],
        ),
        variables=[ConceptDescriptor(name=concept_id, dtype="int64")],
    )

    with pytest.raises(cohort_contract.CohortSchemaError, match="unknown concept_id"):
        cohort_contract.ensure_cohort_definition(plan)

    revised = cohort_contract.ensure_cohort_definition(plan, context=context)

    assert revised.cohort == definition
    assert not cohort_contract.concept_id_exists(concept_id)


def test_cohort_contract_resolves_packaged_concept_dictionary() -> None:
    assert cohort_contract._CONCEPT_DICT_PATH.is_file()
    assert cohort_contract.known_concept_ids()


def test_cohort_contract_round_trip_and_digest_match_legacy_path() -> None:
    definition = cohort_contract.CohortDefinition(
        name="adult",
        inclusion=(
            cohort_contract.ConceptPredicate(
                concept_id="age",
                time_window=cohort_contract.TimeWindow("icu_admit", 0, 24),
                aggregation="max",
                op=">=",
                value=18,
            ),
        ),
    )
    restored = cohort_contract.CohortDefinition.from_dict(definition.to_dict())
    assert restored == definition
    assert cohort_schema.cohort_definition_sha(definition) == (
        cohort_contract.cohort_definition_sha(definition)
    )


def test_cohort_concept_id_scope_survives_interleaved_threads() -> None:
    """Two overlapping scopes must not restore each other's snapshots.

    Unsynchronised, the interleaving A-enter / B-enter / A-exit / B-exit has B
    snapshot a set that already contains A's ids and then reinstate them on
    exit -- so A's hypothetical leaks into the process permanently, and every
    later validation silently accepts a column no run registered.  The scope is
    therefore mutually exclusive; this asserts the outcome, not the mechanism.
    """

    import threading

    cohort_contract.clear_cohort_concept_ids()
    barrier = threading.Barrier(2)
    seen: dict[str, set[str]] = {}
    errors: list[BaseException] = []

    def worker(name: str, own: str, foreign: str) -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(60):
                with cohort_schema.cohort_concept_id_scope([own]):
                    assert cohort_contract.concept_id_exists(own)
                    # Mutual exclusion is what makes this assertion safe: the
                    # other thread's id must never be visible inside our scope.
                    if cohort_contract.concept_id_exists(foreign):
                        seen.setdefault(name, set()).add(foreign)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("a", "col_a", "col_b")),
        threading.Thread(target=worker, args=("b", "col_b", "col_a")),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "scope deadlocked"
        assert not errors, errors
        assert seen == {}, f"a scope observed another thread's ids: {seen}"
        # The decisive assertion: nothing survives the last scope's exit.
        assert not cohort_contract.concept_id_exists("col_a")
        assert not cohort_contract.concept_id_exists("col_b")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_cohort_concept_id_scope_nests_within_one_thread() -> None:
    """The lock must be re-entrant, or a nested scope would deadlock."""

    cohort_contract.clear_cohort_concept_ids()
    try:
        with cohort_schema.cohort_concept_id_scope(["outer"]):
            with cohort_schema.cohort_concept_id_scope(["inner"]):
                assert cohort_contract.concept_id_exists("outer")
                assert cohort_contract.concept_id_exists("inner")
            assert cohort_contract.concept_id_exists("outer")
            assert not cohort_contract.concept_id_exists("inner")
        assert not cohort_contract.concept_id_exists("outer")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_a_bare_reader_never_observes_a_scoped_concept_id() -> None:
    """Guarding the writers left the readers outside.

    A thread that never enters a scope still calls ``concept_id_exists``.  With
    the read unguarded it could observe another thread's *hypothetical* id --
    so "would this plan validate if these columns existed?" asked in one place
    silently became "yes, it exists" somewhere else.

    The window has to be held open deliberately.  A first version of this test
    let two threads race and passed with the read lock removed: the scope
    enters and exits in microseconds, so the reader simply never sampled inside
    it, and the test asserted nothing.  Here the scope stays open for a fixed
    interval, which the reader is released into -- so an unguarded read sees the
    id every time, and a guarded one blocks until the scope has restored the
    registry and then correctly sees nothing.
    """

    import threading
    import time

    cohort_contract.clear_cohort_concept_ids()
    inside = threading.Event()
    observed: list[str] = []
    errors: list[BaseException] = []

    def scoped() -> None:
        try:
            with cohort_schema.cohort_concept_id_scope(["scoped_only"]):
                assert cohort_contract.concept_id_exists("scoped_only")
                inside.set()
                # Held open on purpose. Never waits on the reader: with the
                # lock working the reader is blocked on us, so waiting for it
                # would deadlock rather than fail.
                time.sleep(0.5)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
            inside.set()

    def bare_reader() -> None:
        try:
            assert inside.wait(timeout=10)
            if cohort_contract.concept_id_exists("scoped_only"):
                observed.append("scoped_only")
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=scoped), threading.Thread(target=bare_reader)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "reader or scope deadlocked"
        assert not errors, errors
        assert observed == [], (
            "a thread outside the scope observed a temporary concept id: " f"{observed}"
        )
        assert not cohort_contract.concept_id_exists("scoped_only")
    finally:
        cohort_contract.clear_cohort_concept_ids()


def test_the_batch_runner_scopes_its_cohort_columns_rather_than_registering_them():
    """Nine cases share one process, so a permanent registration accumulates.

    ``_run_one_arm`` runs once per benchmark case. It used to call
    ``register_cohort_concept_ids``, which never un-registers, so case N's
    planner could name a column only case N-1 had materialised, validate
    against the leaked registry, and fail at execution -- after its provider
    calls were paid for. This asserts the structure that prevents it rather
    than the symptom, because the symptom only shows up in a paid batch.
    """

    import ast
    from pathlib import Path

    source = Path("tools/run_research_agent_bench.py").read_text()
    tree = ast.parse(source)

    arm = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_one_arm"
    )
    scoped = [
        item
        for node in ast.walk(arm)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and getattr(item.context_expr.func, "id", None) == "cohort_concept_id_scope"
    ]
    assert scoped, "_run_one_arm must scope its cohort columns"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "register_cohort_concept_ids" not in called, (
        "the batch runner must not register cohort columns permanently; "
        "use cohort_concept_id_scope so each case validates against its own cohort"
    )
