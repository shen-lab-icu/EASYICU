"""The host writes ``<stem>.figure_contract.json``; the Coder must not.

MEASURED 2026-07-30 over every recorded generated script on this machine: 96
distinct figure scripts call ``make_figure_contract``, **all 96** call
``save_publication_figure`` -- which writes the contract itself from the typed
model -- and **15 of the 96 (1 in 6)** ALSO hand-write the same file first.

That redundant write is not merely wasteful, it is fatal.  ``FigureContract`` is
a pydantic ``BaseModel``.  A generated script that reaches for
``json.dump(contract, handle, default=json_default)`` opens the file (creating
it at 0 bytes), then ``json.dump`` hands the model to ``default=``; a converter
whose fallthrough returns an unrecognised value unchanged gives the encoder the
same object back, which raises ``ValueError: Circular reference detected``.  The
step dies with an empty contract, before ``save_publication_figure`` -- the call
that would have written it correctly -- ever runs.

That is exactly what killed ``08_robustness_sensitivity_figure`` in fresh28
(run_20260730T012358_c58197): ``step_summary.error == "Circular reference
detected"``, a 0-byte ``robustness_plot.figure_contract.json``, and two
downstream host findings (``figure_contract_quality`` could not parse it,
``figure_source_data`` reported ``unsafe_declared_figure_path`` because the PNG
was never written).  One defect, three symptoms, one dead step.

The fix is a deletion: the guidance now says the host writes the file, so the
Coder has no reason to.  These tests lock the two halves that must travel
together -- the call, and the prohibition -- because guidance that names the
call without forbidding the hand-write is what 15 scripts already read.
"""

from __future__ import annotations

import inspect
import re

from easyicu.research_agent.figures import publication
from easyicu.research_agent.research_context import prompt_scope


def _render_only_guidance() -> str:
    return prompt_scope._COMPACT_RENDER_ONLY_GUIDANCE


# ---------------------------------------------------------------------------
# The host really does write it -- the premise of the whole prohibition
# ---------------------------------------------------------------------------


def test_save_publication_figure_writes_the_contract_itself():
    """If this ever stops being true, the guidance below becomes a lie."""

    source = inspect.getsource(publication.save_publication_figure)
    assert ".figure_contract.json" in source
    assert "model_dump_json" in source


def test_the_contract_is_a_pydantic_model_not_a_plain_dict():
    """The reason a hand-rolled ``json.dump`` cannot be trusted here."""

    from pydantic import BaseModel

    assert issubclass(publication.FigureContract, BaseModel)


# ---------------------------------------------------------------------------
# The two halves of the guidance, which must travel together
# ---------------------------------------------------------------------------


def test_guidance_tells_the_coder_to_call_the_host_writer():
    guidance = _render_only_guidance()
    assert (
        "save_publication_figure(fig=fig, out_dir=out_dir, stem=stem, contract=contract)"
        in guidance
    )


def test_guidance_forbids_writing_the_contract_by_hand():
    """The half that was missing.

    Anchored on the prohibition, not on the mere presence of the filename --
    the old guidance also mentioned ``.figure_contract.json`` and 15 scripts
    still hand-wrote it.
    """

    guidance = _render_only_guidance()
    assert re.search(r"never write that file yourself", guidance, re.IGNORECASE)


def test_the_call_and_the_prohibition_are_in_the_same_bullet():
    """Scope must match the defect span.

    A Coder that reads the call bullet and stops has to see the prohibition
    there.  Splitting them across bullets is how the instruction gets lost --
    so this asserts on one bullet, not on the whole block.
    """

    bullets = [b for b in _render_only_guidance().split("\n- ") if b.strip()]
    owning = [b for b in bullets if "save_publication_figure(fig=fig" in b]
    assert len(owning) == 1, "exactly one bullet should own the writer call"
    assert re.search(r"never write that file yourself", owning[0], re.IGNORECASE)


def test_the_guidance_names_the_failure_it_is_preventing():
    """A prohibition without its reason is the first thing an LLM overrides."""

    guidance = _render_only_guidance()
    assert "Circular reference detected" in guidance
    assert "pydantic" in guidance.lower()


# ---------------------------------------------------------------------------
# Negative control
# ---------------------------------------------------------------------------


def test_the_step_summary_bullet_no_longer_reads_as_write_the_file():
    """The original wording is what 15 scripts acted on.

    ``Save the generated `.figure_contract.json` ... in `step_summary.json```
    bundles "save the contract file" and "record it in the summary"; the
    measured reading was the first one.  The bullet must now name the host as
    the writer.
    """

    guidance = _render_only_guidance()
    summary_bullets = [
        b
        for b in guidance.split("\n- ")
        if "step_summary.json" in b and ".figure_contract.json" in b
    ]
    assert summary_bullets, "the step_summary bullet should still mention the contract"
    for bullet in summary_bullets:
        assert not re.search(r"Save the generated `\.figure_contract\.json`", bullet)
        assert "host-written" in bullet
