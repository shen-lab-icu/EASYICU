"""The robustness replay must register the label the plan promised.

The replanner directive publishes this contract in as many words: "Name the
step and its products whatever your reader should see; the ``output`` field is
what the execution layer reads."  ``product_id`` is a label, ``output`` is the
claim about which of the runner's products backs it.

The execution layer read ``output`` nowhere.  It registered its own internal
file stems, so a plan that took the directive at its word promised a product
that was written to disk and registered under a name nobody had asked for.

Measured 2026-08-01 over the 32 distinct recorded steps carrying a replay spec:
28 are emittable, 22 happen to label every product exactly as the stem is
spelled, and 6 do not --

    table:robustness_grid      -> specification_grid -> sensitivity_specification_grid  x4
    table:specification_grid   -> specification_grid -> sensitivity_specification_grid  x1
    statistic:primary_effect   -> primary_effect     -> primary_or                      x1

Each of the 6 raises ``declared_product_missing`` on a file sitting in its own
output directory.  canary32's E1 is one of them: the deterministic runner wrote
a complete ``status: ok`` replay (OR 1.566, 95% CI 1.025-2.395, n=1000, both
locked specs converged), the host then declared a contract violation over the
one unregistered label, spent two LLM contract repairs on its own correct
output, and the rewrite died with ``Typed input file not found``.  That killed
the replay, its figure, the robustness figure and the missingness figure.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap

import pytest

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    ROBUSTNESS_REPLAY_OUTPUT_FILES,
    ROBUSTNESS_REPLAY_OUTPUT_KINDS,
    declared_robustness_product_registrations,
    robustness_replay_spec_is_emittable,
    robustness_sensitivity_preflight_scaffold,
)
from easyicu.research_agent.schema import AnalysisStep


def _spec(*pairs: tuple[str, str]) -> dict:
    return {
        "schema_version": "easyicu.robustness_replay/1",
        "products": [
            {"product_id": product_id, "output": output} for product_id, output in pairs
        ],
    }


def _step(outputs, spec=None) -> AnalysisStep:
    return AnalysisStep(
        step_id="09_robustness_replay",
        planned_analysis_role="sensitivity",
        intent="Replay the locked robustness grid without changing the estimand.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=list(outputs),
        method="robustness_sensitivity",
        robustness_replay_spec=spec,
    )


# ---------------------------------------------------------------------------
# The three measured shapes
# ---------------------------------------------------------------------------


def test_the_label_canary32_promised_resolves_to_the_file_that_backs_it():
    """canary32 E1: one unregistered label cost four steps."""

    step = _step(
        ["table:robustness_matrix", "table:specification_grid"],
        _spec(
            ("robustness_matrix", "robustness_matrix"),
            ("specification_grid", "specification_grid"),
        ),
    )

    assert declared_robustness_product_registrations(step) == {
        "table:robustness_matrix": "robustness_matrix.csv",
        "table:specification_grid": "sensitivity_specification_grid.csv",
    }


def test_the_label_four_recorded_plans_chose_resolves_to_the_same_file():
    """``robustness_grid`` is the most common spelling in the corpus (x4).

    It is the whole point of the ``product_id``/``output`` split: the reader
    sees the plan's word, the execution layer reads the claim.
    """

    step = _step(
        ["table:robustness_grid"],
        _spec(("robustness_grid", "specification_grid")),
    )

    assert declared_robustness_product_registrations(step) == {
        "table:robustness_grid": "sensitivity_specification_grid.csv"
    }


def test_a_statistic_label_resolves_through_the_same_table():
    step = _step(
        ["statistic:primary_effect"],
        _spec(("primary_effect", "primary_effect")),
    )

    assert declared_robustness_product_registrations(step) == {
        "statistic:primary_effect": "primary_effect.json"
    }


# ---------------------------------------------------------------------------
# What must NOT be registered
# ---------------------------------------------------------------------------


def test_a_step_with_no_declaration_registers_nothing():
    """22 of the 28 emittable steps need nothing from this, and a step that
    declares no spec at all has only the runner's internal stems.  Registering
    a label nobody declared would invent a contract."""

    assert (
        declared_robustness_product_registrations(_step(["table:robustness_matrix"]))
        == {}
    )
    assert declared_robustness_product_registrations(None) == {}


def test_a_promised_product_the_spec_does_not_back_is_not_registered():
    """The spec is the claim.  A product it is silent about has no backing."""

    step = _step(
        ["table:robustness_matrix", "table:something_else_entirely"],
        _spec(("robustness_matrix", "robustness_matrix")),
    )

    assert declared_robustness_product_registrations(step) == {
        "table:robustness_matrix": "robustness_matrix.csv"
    }


def test_a_silent_spec_is_not_answered_by_the_label_looking_like_an_output():
    """The distinguishing case, and the whole disease in one step.

    ``execution/phase.py::_robustness_sensitivity_preflight_supported`` has a
    second arm: a step whose ``method`` head is on the allowlist reaches this
    runner *without* an emittable spec.  So a spec that backs only some of the
    promised products really does arrive here.

    For a product it is silent about, guessing from the label -- "the word
    happens to be one of my own outputs, so it must mean that one" -- is
    exactly the inference this contract exists to remove.  ``robustness_summary``
    is a published output name AND a plausible reader-facing label, so the two
    readings are indistinguishable without the guard.
    """

    assert "robustness_summary" in ROBUSTNESS_REPLAY_OUTPUT_FILES
    step = _step(
        ["table:robustness_matrix", "table:robustness_summary"],
        _spec(("robustness_matrix", "robustness_matrix")),
    )

    assert declared_robustness_product_registrations(step) == {
        "table:robustness_matrix": "robustness_matrix.csv"
    }


def test_a_kind_this_replay_never_writes_is_not_registered():
    """``figure`` is deliberately outside ``ROBUSTNESS_REPLAY_OUTPUT_KINDS``.

    A figure label registered against a csv would hand the figure lineage a
    table and call it a rendered panel.
    """

    assert "figure" not in ROBUSTNESS_REPLAY_OUTPUT_KINDS
    step = _step(
        ["figure:robustness_grid"],
        _spec(("robustness_grid", "specification_grid")),
    )

    assert declared_robustness_product_registrations(step) == {}


def test_an_output_this_runner_does_not_emit_is_not_registered():
    """Belt to ``robustness_replay_spec_is_emittable``'s braces.

    That predicate already refuses the step, so this can only be reached by a
    caller that skipped it -- and the wrong answer there is a registration
    pointing at no file at all, which is worse than the missing product it
    would be replacing.
    """

    step = _step(
        ["table:robustness_matrix"],
        _spec(("robustness_matrix", "robustness_matrix")),
    )
    object.__setattr__(
        step.robustness_replay_spec.products[0], "output", "not_a_product"
    )

    assert robustness_replay_spec_is_emittable(step) is False
    assert declared_robustness_product_registrations(step) == {}


def test_every_filename_comes_from_the_runners_own_published_table():
    """A second copy of this map is how the contract and the runner drift.

    Sweeps every published output rather than the three that bite today, so a
    newly published product cannot be added without a file behind it.
    """

    for output in sorted(ROBUSTNESS_REPLAY_OUTPUT_FILES):
        step = _step(
            ["table:reader_chosen_label"], _spec(("reader_chosen_label", output))
        )
        assert declared_robustness_product_registrations(step) == {
            "table:reader_chosen_label": ROBUSTNESS_REPLAY_OUTPUT_FILES[output]
        }


# ---------------------------------------------------------------------------
# The host epilogue -- the half that makes it cost something
# ---------------------------------------------------------------------------


def _run_epilogue(step: AnalysisStep, out_dir: pathlib.Path) -> dict:
    epilogue = robustness_sensitivity_preflight_scaffold(step).epilogue
    previous = os.environ.get("STEP_OUT_DIR")
    os.environ["STEP_OUT_DIR"] = str(out_dir)
    try:
        exec(  # noqa: S102 - executing the host's own generated region is the test
            compile(epilogue, "<epilogue>", "exec"),
            {"Path": pathlib.Path, "json": json, "os": os},
        )
    finally:
        if previous is None:
            os.environ.pop("STEP_OUT_DIR", None)
        else:
            os.environ["STEP_OUT_DIR"] = previous
    return json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))


def _summary(out_dir: pathlib.Path, **summary) -> None:
    (out_dir / "step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )


def test_the_epilogue_registers_the_promised_identity(tmp_path):
    """The exact canary32 shape: file written, label absent, step killed."""

    (tmp_path / "sensitivity_specification_grid.csv").write_text(
        "spec_id,axis\nprimary,primary\n", encoding="utf-8"
    )
    _summary(
        tmp_path,
        status="ok",
        output_files={
            "table:sensitivity_specification_grid": "sensitivity_specification_grid.csv"
        },
        aliases={
            "sensitivity_specification_grid": "sensitivity_specification_grid.csv"
        },
    )
    step = _step(
        ["table:specification_grid"],
        _spec(("specification_grid", "specification_grid")),
    )

    summary = _run_epilogue(step, tmp_path)

    assert (
        summary["output_files"]["table:specification_grid"]
        == "sensitivity_specification_grid.csv"
    )
    # The runner's own stem keeps its registration: a step that declares no
    # spec still has nothing else to be found by.
    assert "table:sensitivity_specification_grid" in summary["output_files"]
    # ...and `aliases` keeps holding ONLY those stems. It is the runner's own
    # internal-stem map -- the exact set `canonical_robustness_output_files` is
    # handed, and the marker that says "this runner wrote this summary". The
    # epilogue used to copy the plan's bare name in here too, which gave one key
    # two meanings: a real run recorded `robustness_grid` there, a name the kind
    # map does not declare and would raise on. The promised identity belongs in
    # `output_files`, which is what the envelope reads, and nowhere else.
    assert (
        summary["aliases"]["sensitivity_specification_grid"]
        == "sensitivity_specification_grid.csv"
    )
    assert "specification_grid" not in summary["aliases"]


def test_the_epilogue_will_not_register_a_file_that_was_not_written(tmp_path):
    """A promised label over an absent file is a false receipt.

    The blocked path is real: canary32's E3 replay wrote no matrix at all
    because the primary model emitted three coefficients where the robustness
    matrix requires one, and reported ``status: blocked``.  Registering its
    declared products anyway would turn a correctly blocked step into one
    claiming products nothing backs.
    """

    _summary(tmp_path, status="blocked", output_files={}, aliases={})
    step = _step(
        ["table:specification_grid"],
        _spec(("specification_grid", "specification_grid")),
    )

    summary = _run_epilogue(step, tmp_path)

    assert summary["output_files"] == {}
    assert summary["aliases"] == {}


def test_the_epilogue_never_overwrites_a_registration_the_runner_made(tmp_path):
    """The runner's own answer wins; this only adds the label it lacked."""

    (tmp_path / "robustness_matrix.csv").write_text("a\n1\n", encoding="utf-8")
    _summary(
        tmp_path,
        output_files={"table:robustness_matrix": "already_here.csv"},
        aliases={"robustness_matrix": "already_here.csv"},
    )
    step = _step(
        ["table:robustness_matrix"], _spec(("robustness_matrix", "robustness_matrix"))
    )

    summary = _run_epilogue(step, tmp_path)

    assert summary["output_files"]["table:robustness_matrix"] == "already_here.csv"


def test_the_epilogue_is_absent_when_there_is_nothing_to_persist():
    """No scope and no declaration means nothing in this script is host property.

    ``test_a_scaffold_without_host_regions_is_all_body`` asserts the same
    thing from the boundary's side; this asserts the new arm did not quietly
    give every step an epilogue.
    """

    assert robustness_sensitivity_preflight_scaffold().epilogue == ""
    assert (
        robustness_sensitivity_preflight_scaffold(
            _step(["table:robustness_matrix"])
        ).epilogue
        == ""
    )


def test_the_declaration_arm_adds_exactly_one_canonical_write(tmp_path):
    """Two read-modify-write blocks would be two canonical writes.

    The static obligation gate reasons about the ``step_summary.json`` write
    that stays visible in the generated source; a second one is a second thing
    for it to prove.
    """

    step = _step(
        ["table:specification_grid"],
        _spec(("specification_grid", "specification_grid")),
    )
    epilogue = robustness_sensitivity_preflight_scaffold(step).epilogue

    assert epilogue.count("summary_path.write_text(") == 1
    assert epilogue.count('json.loads(summary_path.read_text(encoding="utf-8"))') == 1


def test_the_generated_epilogue_is_valid_python_for_every_published_output():
    """A repr rendered into source is source; it has to parse."""

    for output in sorted(ROBUSTNESS_REPLAY_OUTPUT_FILES):
        step = _step(["table:reader_label"], _spec(("reader_label", output)))
        compile(
            robustness_sensitivity_preflight_scaffold(step).epilogue,
            "<epilogue>",
            "exec",
        )


# ---------------------------------------------------------------------------
# Boundaries this fix deliberately does not cross
# ---------------------------------------------------------------------------


def test_no_case_specific_literal_reaches_the_registration():
    """Case-neutral: the mapping is the plan's, not the benchmark's."""

    source = pathlib.Path(
        declared_robustness_product_registrations.__globals__["__file__"]
    ).read_text(encoding="utf-8")
    start = source.index("def declared_robustness_product_registrations")
    end = source.index("def robustness_replay_declaration_verdict")
    body = source[start:end]
    for token in ("sep3", "kdigo", "aki_stage", "lactate", "mimic", "e1_", "e3_"):
        assert token not in body.casefold()


def test_the_promised_label_never_changes_which_science_ran(tmp_path):
    """Only the registration moves; the numbers are the runner's.

    A relabelling that could alter a reported estimate would be the worst
    possible version of this fix.
    """

    (tmp_path / "primary_or.json").write_text('{"primary_or": 1.57}', encoding="utf-8")
    (tmp_path / "primary_effect.json").write_text(
        '{"statistic": "primary_effect", "value": 1.57}',
        encoding="utf-8",
    )
    _summary(
        tmp_path,
        status="ok",
        primary_or=1.566375890701969,
        complete_case_n=1000,
        output_files={"statistic:primary_or": "primary_or.json"},
        aliases={"primary_or": "primary_or.json"},
    )
    step = _step(
        ["statistic:headline_effect"], _spec(("headline_effect", "primary_effect"))
    )

    summary = _run_epilogue(step, tmp_path)

    assert summary["primary_or"] == 1.566375890701969
    assert summary["complete_case_n"] == 1000
    assert summary["status"] == "ok"
    assert summary["output_files"]["statistic:headline_effect"] == "headline_effect.json"
    headline_payload = json.loads(
        (tmp_path / "headline_effect.json").read_text(encoding="utf-8")
    )
    assert headline_payload == {
        "statistic": "headline_effect",
        "value": 1.57,
    }


@pytest.mark.parametrize(
    "declared",
    [
        "table:robustness_matrix",
        "statistic:complete_case_n",
        "log:missingness_strategy_notes",
    ],
    ids=["table", "statistic", "log"],
)
def test_every_kind_this_replay_writes_can_carry_a_promised_label(declared):
    kind, _, product_id = declared.partition(":")
    step = _step([declared], _spec((product_id, product_id)))

    assert declared_robustness_product_registrations(step) == {
        declared: ROBUSTNESS_REPLAY_OUTPUT_FILES[product_id]
    }
    assert kind in ROBUSTNESS_REPLAY_OUTPUT_KINDS


def test_the_epilogue_text_is_not_rebuilt_from_a_second_template():
    """The rendered dict must be the function's answer, verbatim."""

    step = _step(
        ["table:robustness_grid", "statistic:complete_case_n"],
        _spec(
            ("robustness_grid", "specification_grid"),
            ("complete_case_n", "complete_case_n"),
        ),
    )
    rendered = robustness_sensitivity_preflight_scaffold(step).epilogue

    assert repr(declared_robustness_product_registrations(step)) in rendered


def test_the_indentation_of_the_rendered_block_is_module_level():
    """A repr dropped in at the wrong depth is a syntax error at run time."""

    step = _step(
        ["table:robustness_grid"], _spec(("robustness_grid", "specification_grid"))
    )
    epilogue = robustness_sensitivity_preflight_scaffold(step).epilogue

    declaration = next(
        line
        for line in epilogue.splitlines()
        if line.startswith("declared_product_files")
    )
    assert declaration == textwrap.dedent(declaration)
