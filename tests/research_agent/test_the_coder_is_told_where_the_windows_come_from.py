"""The Coder built a wide representation for a long-bound run.

verify38, h3 ``02_build_fixed_anchor_trajectory_representation``: the plan
passed validation (0 plan_contract errors) and the step EXECUTED -- then died on
its own guard, three times, deterministic repair declining each:

    ValueError: No declared analysis-cohort columns encode fixed windows
    within hours 0-72.

The generated script referenced ``COHORT_PARQUET`` four times and
``TRAJECTORY_PARQUET`` ZERO times. It defined ``parse_window_from_column`` and
regex-scanned cohort column names for ``_h<start>_<end>``, found none -- a
long-bound run has none by construction -- and raised.

The Planner guide was corrected for exactly this misconception in 30c9ac8. The
Coder had the same one, one layer down: ``coder.txt`` frames the trajectory as
"OPTIONAL ... when the wide cohort's per-stay summaries cannot express what the
step needs", which reads as enrichment for onset/timing questions, not as the
source of a trajectory representation.

The instruction is keyed to the DECLARED PRODUCT -- a step whose
expected_outputs contain ``manifest:trajectory_window_manifest`` -- so it names
no task, disease, or database.
"""

from __future__ import annotations

import pathlib

_CODER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/providers/prompts/v1/coder.txt"
)


def _text() -> str:
    return _CODER.read_text(encoding="utf-8")


def test_the_coder_is_told_the_windows_come_from_the_trajectory():
    text = _text()

    assert "manifest:trajectory_window_manifest" in text
    # It must say where they come FROM...
    assert "the fixed windows come FROM this table" in text
    # ...and forbid the exact thing verify38's script did.
    assert "do not fail when none exist" in text


def test_the_instruction_is_keyed_to_the_declared_product_not_a_task():
    """Case neutrality: it triggers on what the step promises, not on h3."""

    text = _text()
    start = text.index("manifest:trajectory_window_manifest")
    paragraph = text[start - 200 : start + 700].lower()

    for banned in ("h3", "sofa2", "sepsis", "kdigo", "mimic", "miiv", "lactate"):
        assert banned not in paragraph, banned


def test_the_optional_framing_still_covers_the_other_uses():
    """Onset / incident / landmark uses are unchanged; this only adds a case."""

    text = _text()
    assert "OPTIONAL trajectory" in text
    assert "threshold-crossing onset" in text
    assert "MANDATORY when a timing/onset/incident question is being gated" in text
