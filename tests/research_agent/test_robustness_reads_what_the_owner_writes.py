"""Two host-owned contracts the host's own code could never satisfy.

The deterministic robustness replay is the most repeated host-owned failure in
the recorded corpus: it is claimed by a host owner, spends its whole repair
budget, and dies.  No model is involved, so nothing the Coder does can fix it.
Measured across every recorded run, this code had **never once** passed the
contract gate; the single step that did was a 528-line Coder rewrite that
hand-built the receipt block the host refused to stamp.

Two independent causes, both of the same shape -- one side of the host writes
something, another side of the host reads something else:

**The analysis definition.**  The equivalence proof required
``summary["analysis_definition"]``.  The host's own primary-model owner writes
``exposure`` / ``outcome`` / ``covariates`` as flat top-level keys and has never
written that nested one.  A repository-wide search for the name returned two
hits: the reader, and a test fixture added in the same commit.  Over 358
recorded step summaries it appeared exactly once, in a Coder summary.  So the
proof was unreachable in production from the day it was written, and the test
that covered it was reading a shape only the test produced.

**The input-binding receipt.**  ``step_summary_integrity`` requires one receipt
per host-resolved typed input.  The host has a function for exactly this, whose
docstring says a renderer asked to manufacture its own receipt would be
attesting to its own input -- but its call site was gated on
``deterministic_standard_executor_used``, and the robustness and sensitivity
paths set ``deterministic_fallback_used`` instead.  Host-authored code could
not satisfy a rule about host-resolved inputs.

The load-bearing tests here are the two that read a **real** failing summary
from a real run rather than a shape written for the occasion -- that is the
property the original tests lacked.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.research_agent.contracts.result_envelope import (
    MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY,
    MODEL_SUMMARY_COVARIATE_KEYS,
    MODEL_SUMMARY_EXPOSURE_KEYS,
    MODEL_SUMMARY_OUTCOME_KEYS,
    model_summary_analysis_definition,
)

# The shape the host's own deterministic primary owner really writes, copied
# from adjusted_association_executor.py's summary dict.
OWNER_SUMMARY = {
    "status": "ok",
    "analysis_family": "association",
    "requirement_id": "primary_association",
    "exposure": "severity_score",
    "outcome": "death",
    "covariates": ["age", "sex", "comorbidity"],
    "adjustment_covariates": ["age", "sex", "comorbidity"],
    "n_total": 1000,
    "n_events": 102,
}


# --- the reader: what the owner writes is what the proof reads ---------------


def test_the_owner_shape_is_read():
    """The property that was false: the host could not read its own summary."""

    assert model_summary_analysis_definition(OWNER_SUMMARY) == {
        "exposure": "severity_score",
        "outcome": "death",
        "covariates": ["age", "sex", "comorbidity"],
    }


def test_the_nested_spelling_still_reads():
    """The only spelling the old reader knew must keep working."""

    assert model_summary_analysis_definition(
        {
            MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY: {
                "exposure": "x",
                "outcome": "y",
                "covariates": ["a"],
            }
        }
    ) == {"exposure": "x", "outcome": "y", "covariates": ["a"]}


def test_the_nested_spelling_wins_and_is_not_merged():
    """A summary declaring both is read by the nested one, not by a blend.

    Merging would let a stale flat key contribute a covariate the nested
    declaration deliberately dropped.
    """

    assert model_summary_analysis_definition(
        {
            MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY: {
                "exposure": "nested",
                "outcome": "y",
                "covariates": ["a"],
            },
            "exposure": "flat",
            "outcome": "y",
            "covariates": ["a", "b"],
        }
    ) == {"exposure": "nested", "outcome": "y", "covariates": ["a"]}


def test_an_unadjusted_model_states_an_empty_set_not_nothing():
    """``[]`` is a real answer -- an unadjusted primary says exactly that."""

    assert model_summary_analysis_definition(
        {"exposure": "x", "outcome": "y", "covariates": []}
    ) == {"exposure": "x", "outcome": "y", "covariates": []}


def test_two_covariate_spellings_that_disagree_are_refused():
    """The one case where answering is worse than refusing.

    Whichever were picked, the summary itself says the other is also true, and
    the proof would silently be taken over a set the model may not have used.
    """

    assert (
        model_summary_analysis_definition(
            {
                "exposure": "x",
                "outcome": "y",
                "covariates": ["a"],
                "adjustment_covariates": ["b"],
            }
        )
        is None
    )


def test_two_covariate_spellings_that_agree_are_accepted():
    assert model_summary_analysis_definition(
        {
            "exposure": "x",
            "outcome": "y",
            "covariates": ["a"],
            "adjustment_covariates": ["a"],
        }
    ) == {"exposure": "x", "outcome": "y", "covariates": ["a"]}


@pytest.mark.parametrize(
    "summary",
    [
        {},
        {"exposure": "x", "covariates": []},
        {"outcome": "y", "covariates": []},
        {"exposure": "x", "outcome": "y"},
        {"exposure": "  ", "outcome": "y", "covariates": []},
        {"exposure": "x", "outcome": "  ", "covariates": []},
        {"exposure": "x", "outcome": "y", "covariates": "age"},
        {"exposure": "x", "outcome": "y", "covariates": ["age", 7]},
        {"exposure": "x", "outcome": "y", "covariates": ["age", "  "]},
        {MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY: "not-a-mapping"},
        {MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY: {"exposure": "x", "outcome": "y"}},
        "not-a-summary",
        None,
    ],
    ids=[
        "empty",
        "no-outcome",
        "no-exposure",
        "no-covariate-key-at-all",
        "blank-exposure",
        "blank-outcome",
        "a-string-is-not-a-list",
        "a-number-is-not-a-column",
        "a-blank-column-name",
        "nested-but-unnavigable",
        "nested-without-covariates",
        "not-a-mapping",
        "none",
    ],
)
def test_an_incomplete_declaration_is_refused_not_guessed(summary):
    """Partial or malformed is a refusal, never a fall-through.

    The bare-string and unnavigable-nested cases are the ones that separate
    this reader from a hand-rolled ``summary.get(...)``: both would survive a
    local re-implementation and neither states an analysis.
    """

    assert model_summary_analysis_definition(summary) is None


# --- and the refusal names the keys, so it is repairable ---------------------


def test_the_refusal_names_every_spelling_it_looked_for():
    from easyicu.research_agent.execution.runners import deterministic_robustness

    source = Path(deterministic_robustness.__file__).read_text()
    marker = "primary analysis definition is unavailable for equivalence proof"
    assert marker in source
    passage = source[source.index(marker) : source.index(marker) + 700]

    # Rendered from the constants, so publication cannot drift from enforcement.
    for name in (
        "MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY",
        "MODEL_SUMMARY_EXPOSURE_KEYS",
        "MODEL_SUMMARY_OUTCOME_KEYS",
        "MODEL_SUMMARY_COVARIATE_KEYS",
    ):
        assert name in passage, f"the refusal must render {name}, not restate it"


def test_the_proof_delegates_rather_than_re_reading_the_key():
    """One reader on both sides, so neither can accept what the other rejects."""

    from easyicu.research_agent.execution.runners import deterministic_robustness

    source = Path(deterministic_robustness.__file__).read_text()
    assert "model_summary_analysis_definition(" in source
    # The old hand-rolled read must be gone, not merely shadowed.
    assert '.get("analysis_definition")' not in source


# --- the receipt gate: host-authored code can satisfy the host's own rule ----


def _receipt_gate_flags() -> set[str]:
    """The worker-progress flags the receipt-stamping branch actually tests.

    Read from the AST rather than by string search: a flag mentioned in a
    comment, or in the ``consumed_input_keys`` expression below the branch,
    is not the same as a flag the branch is gated on.
    """

    from easyicu.research_agent.execution import phase

    tree = ast.parse(Path(phase.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = ast.unparse(node.body[0]) if node.body else ""
        if "_write_host_input_binding_receipts" not in body:
            continue
        test = ast.unparse(node.test)
        return {
            name
            for name in (
                "deterministic_standard_executor_used",
                "deterministic_fallback_used",
            )
            if f"worker_progress.{name}" in test
        }
    raise AssertionError("no branch guarding _write_host_input_binding_receipts")


def test_deterministic_fallback_code_also_gets_its_receipts_stamped():
    """The defect: the robustness path sets the flag the gate did not test.

    Host-authored deterministic code cannot be asked to manufacture a receipt
    about inputs the host resolved -- that is the stamping function's own
    stated reason for existing -- so the gate must cover every path that runs
    host-authored code, not only the registered standard executors.
    """

    assert _receipt_gate_flags() == {
        "deterministic_standard_executor_used",
        "deterministic_fallback_used",
    }


def test_the_robustness_path_really_sets_that_flag():
    """Anchors the test above on the producing side, not on a name I chose."""

    from easyicu.research_agent.execution import phase

    source = Path(phase.__file__).read_text()
    marker = "_deterministic_robustness_sensitivity_code"
    body = source[source.index(f"def {marker}") :][:1500]
    assert "worker_progress.deterministic_fallback_used = True" in body
    assert "worker_progress.deterministic_standard_executor_used" not in body


# --- the load-bearing pair: a real failing summary from a real run -----------

_REAL_RUNS = (
    sorted(
        Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs").glob(
            "batch_*/*/aware/run_*/steps/*/outputs/step_summary.json"
        )
    )
    if Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs").exists()
    else []
)


def _real_primary_summaries() -> list[dict]:
    found = []
    for path in _REAL_RUNS:
        try:
            summary = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(summary, dict) and summary.get("model_contracts"):
            found.append(summary)
    return found


@pytest.mark.skipif(not _REAL_RUNS, reason="recorded runs are not on this machine")
def test_a_real_primary_summary_is_readable():
    """The property the old test could not have: real bytes, not a fixture.

    The original test wrote ``analysis_definition`` into its own fixture and
    then asserted the reader could read it -- so it passed for the whole time
    production never produced that key.
    """

    summaries = _real_primary_summaries()
    if not summaries:
        pytest.skip("no recorded summary carries model_contracts")

    readable = [
        s for s in summaries if model_summary_analysis_definition(s) is not None
    ]
    assert readable, (
        "no recorded primary summary states an analysis the proof can read; "
        "that is exactly the condition this fix exists to remove"
    )


@pytest.mark.skipif(not _REAL_RUNS, reason="recorded runs are not on this machine")
def test_the_nested_key_is_effectively_absent_from_real_runs():
    """Why the flat spelling had to be published rather than demanded.

    If this ever stops holding -- if producers start writing the nested key --
    that is a real change worth noticing, not a silent drift.
    """

    summaries = _real_primary_summaries()
    if not summaries:
        pytest.skip("no recorded summary carries model_contracts")
    nested = sum(
        1
        for s in summaries
        if isinstance(s.get(MODEL_SUMMARY_ANALYSIS_DEFINITION_KEY), dict)
    )
    flat = sum(
        1
        for s in summaries
        if any(k in s for k in MODEL_SUMMARY_EXPOSURE_KEYS)
        and any(k in s for k in MODEL_SUMMARY_OUTCOME_KEYS)
        and any(k in s for k in MODEL_SUMMARY_COVARIATE_KEYS)
    )
    assert flat > nested, (
        f"the flat spelling is what producers write ({flat} vs {nested} nested); "
        "if that inverts, revisit which spelling the proof should prefer"
    )
