"""The runner computed everything, then labelled it in a vocabulary its own
host rejects.

``result_envelope`` requires every ``output_files`` key to be a ``kind:name``
identity and drops anything else as ``invalid_product_identity``.  A dropped
product is absent from the canonical envelope while still present in the
summary, so the bounded-metric shadow reports it as a declared product that is
missing and fails the step closed -- after the science has been computed and
written to disk.

The deterministic robustness runner registered bare names
(``"robustness_matrix": "robustness_matrix.csv"``).  On the 2026-08-01 E1 run
its step produced 17 real result files -- the robustness matrix, the
coefficients, the primary effect, the specification grid -- exited 0, spent no
provider call, and was discarded because all 17 registrations were rejected.

Measured over every recorded run: 490 product registrations use a valid
identity and 191 do not.  All 191 came from this one runner; every other
producer was already correct.

Note what this check is NOT.  It compares the runner's own registration against
what normalization accepted -- the Planner's ``expected_outputs`` are not
involved.  Real plans disagree with each other about ``robustness_summary``
(declared ``table`` 243 times and ``statistic`` 239 times), and that
disagreement is real but has no bearing here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.result_envelope import _PRODUCT_RE
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    ROBUSTNESS_REPLAY_OUTPUT_KINDS,
    _ROBUSTNESS_PRODUCT_KINDS,
    canonical_robustness_output_files,
)

# Exactly what the 2026-08-01 E1 step registered and lost.
E1_PRODUCT_FILES = {
    "coefficients": "coefficients.csv",
    "cohort_definition_overlap_attrition": "cohort_definition_overlap_attrition.csv",
    "cohort_overlap_and_attrition": "cohort_overlap_and_attrition.csv",
    "complete_case_n": "complete_case_n.json",
    "membership_change_summary": "membership_change_summary.csv",
    "missingness_strategy_notes": "missingness_strategy_notes.txt",
    "missingness_strategy_notes_json": "missingness_strategy_notes.json",
    "model_replay_index": "model_replay_index.json",
    "model_summaries": "model_summaries.csv",
    "outcome_label_executability": "outcome_label_executability.csv",
    "primary_or": "primary_or.json",
    "robustness_matrix": "robustness_matrix.csv",
    "robustness_summary": "robustness_summary.csv",
    "robustness_variant_coefficients": "robustness_variant_coefficients.csv",
    "sensitivity_comparison": "sensitivity_comparison.csv",
    "sensitivity_specification_grid": "sensitivity_specification_grid.csv",
    "sensitivity_specification_matrix": "sensitivity_specification_matrix.csv",
}


def test_every_product_the_real_step_registered_now_survives_normalization() -> None:
    """The property that was false for all 17 of them."""

    canonical = canonical_robustness_output_files(E1_PRODUCT_FILES)
    assert len(canonical) == len(E1_PRODUCT_FILES)
    unreadable = [pid for pid in canonical if not _PRODUCT_RE.fullmatch(pid)]
    assert not unreadable, f"still rejected by the envelope: {unreadable}"


def test_the_files_are_untouched() -> None:
    """Only the identity changes; the artifact each product points at does not."""

    canonical = canonical_robustness_output_files(E1_PRODUCT_FILES)
    assert sorted(canonical.values()) == sorted(E1_PRODUCT_FILES.values())
    for product_id, filename in canonical.items():
        assert product_id.split(":", 1)[1] in E1_PRODUCT_FILES
        assert E1_PRODUCT_FILES[product_id.split(":", 1)[1]] == filename


def test_a_product_with_no_declared_kind_is_refused() -> None:
    """Fail closed rather than repeat the silent drop.

    A new artifact must say what it is; guessing a kind here would put the
    runner right back to registering something the envelope may reject.
    """

    with pytest.raises(ValueError) as excinfo:
        canonical_robustness_output_files({"a_brand_new_product": "new.csv"})
    assert "a_brand_new_product" in str(excinfo.value)


def test_no_kind_escapes_what_this_runner_declares_it_can_emit() -> None:
    """The two declarations must not drift apart.

    ``ROBUSTNESS_REPLAY_OUTPUT_KINDS`` is what the emittability check and the
    declaration-gap verdict already read. A kind here that is not there would
    mean the runner registers something it also says it cannot produce.
    """

    stray = sorted(
        set(_ROBUSTNESS_PRODUCT_KINDS.values()) - ROBUSTNESS_REPLAY_OUTPUT_KINDS
    )
    assert not stray, f"registers a kind it does not declare it can emit: {stray}"


def test_empty_registration_is_empty_not_an_error() -> None:
    assert canonical_robustness_output_files({}) == {}


def test_the_summary_registers_the_canonical_map_and_not_the_bare_one() -> None:
    """The wiring, not just the helper.

    A first pass tested ``canonical_robustness_output_files`` alone, and
    mutation showed that pointing the summary's ``output_files`` back at the
    bare-name dict left every test green -- the whole defect restored by one
    word.  ``aliases`` deliberately keeps the bare names, so it is not enough
    that the canonical map merely exists somewhere in the function.
    """

    import ast
    import inspect

    from easyicu.research_agent.execution.runners import deterministic_robustness

    tree = ast.parse(inspect.getsource(deterministic_robustness))

    canonical_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "canonical_robustness_output_files"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert canonical_names, "nothing is built from the canonical identity map"

    registrations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "output_files":
                registrations.append(ast.unparse(value))
    assert registrations, "no summary registers output_files"
    for rendered in registrations:
        assert rendered in canonical_names, (
            "the summary registers something other than the canonical identity "
            f"map: {rendered!r}"
        )


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")

#: Names that reached ``aliases`` from a writer that no longer exists.
#:
#: Between 55b901b (2026-08-01) and its removal, the promised-product patch
#: also wrote the *Planner's* product id, bare, into ``aliases`` -- a key whose
#: whole job is to be this runner's own internal-stem map.  One recorded run
#: caught that window:
#: ``batch_20260802_luna_miiv_FULL_d5baff6_nine1/m1_hepatobiliary_missingness``
#: registered ``robustness_grid -> sensitivity_specification_grid.csv``.
#:
#: Recorded bytes do not change when the code is fixed, so without this the
#: assertion below could never be green again -- a permanently red guard tells
#: you nothing.  Naming the one known artifact keeps it a guard: any OTHER
#: undeclared name still fails, including a recurrence of this one under a
#: different spelling.  Delete this set when the corpus is regenerated.
_PRE_FIX_ALIAS_POLLUTION = frozenset({"robustness_grid"})


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_every_product_this_runner_ever_registered_has_a_declared_kind() -> None:
    """Real bytes: the kind map must cover what the runner actually emits.

    Population corrected 2026-08-01.  It used to be every recorded robustness
    summary's ``output_files``, which is wider than the claim: a
    ``robustness_sensitivity`` step is not necessarily written by THIS runner,
    and the mapping only ever receives this runner's own ``product_files``.
    Scanning all of them made the test fail on ``robustness_grid``, a
    Planner-named product registered on a different path entirely -- a name the
    mapping has never been handed and, on this evidence, cannot be.

    ``aliases`` is the bare ``product_files`` map and only this runner writes
    it, so it is both the right population marker and the exact set the mapping
    is called with.  Measured over the corpus: 19 summaries carry it, holding
    17 distinct names -- exactly the 17 the kind map declares.

    The wider question of whether a Planner-named product can reach a consumer
    that does not know it is real and open; it is not this mapping's, and
    asserting it here only produced a false accusation against working code.
    """

    seen: set[str] = set()
    for path in sorted(
        _CORPUS.glob("batch_*/*/aware/run_*/steps/*/outputs/step_summary.json")
    ):
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(summary, dict):
            continue
        if summary.get("analysis_family") != "robustness_sensitivity":
            continue
        registered = summary.get("aliases")
        if not isinstance(registered, dict):
            continue
        seen.update(str(product_id) for product_id in registered)

    if not seen:
        pytest.skip("no recorded run was written by this runner's summary path")
    missing = sorted(seen - set(_ROBUSTNESS_PRODUCT_KINDS) - _PRE_FIX_ALIAS_POLLUTION)
    assert not missing, (
        f"this runner has emitted {len(missing)} product(s) with no declared "
        f"kind, so they would be refused: {missing}"
    )
