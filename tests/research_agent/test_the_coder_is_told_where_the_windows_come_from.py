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


def test_the_coder_is_given_the_exact_schema_key_names():
    """verify39: one key name cost the whole step, after it had already run.

    h3 03_construct_fixed_anchor_trajectories executed, read the long
    trajectory, built hourly windows and wrote a rich schema -- and was refused
    with a single issue: ``observation_columns is missing``. It had written
    ``ordered_observation_columns``.

    The required key set is stated in the PLANNER guide (plan_contract.py,
    "id_column, observation_family, ordered observation_columns, ...") and
    appeared ZERO times in every prompt the Coder sees -- and the Coder is the
    one that writes the file. The planner phrasing "ordered observation_columns"
    is itself the likely source of the near-miss.
    """

    text = _text()

    assert "manifest:trajectory_representation_schema" in text
    for key in (
        "id_column",
        "observation_family",
        "observation_columns",
        "profile_columns",
        "representation_columns",
        "frozen_population_n",
        "representation_sha256",
        "trailing_na_policy",
    ):
        # BACKTICKED, not substring. A plain `key in text` passes when the list
        # says `ordered_observation_columns` -- the very near-miss this exists
        # to prevent -- because the required name is a substring of the wrong
        # one. That mutation survived the first version of this test, which is
        # the third time this branch has been bitten by substring-vs-name.
        assert f"`{key}`" in text, key
    # And it warns about the exact near-miss that was observed.
    assert "ordered_observation_columns" in text


def test_the_disclosed_keys_are_the_ones_the_host_actually_requires():
    """A prompt that lists the wrong keys is worse than one that lists none."""

    import inspect

    from easyicu.research_agent.trajectory import plan_contract

    source = inspect.getsource(plan_contract)
    start = source.index("trajectory_representation_schema_incomplete")
    block = source[max(0, start - 1400) : start]
    required = {
        key
        for key in (
            "id_column",
            "observation_family",
            "observation_columns",
            "profile_columns",
            "representation_columns",
            "frozen_population_n",
            "representation_sha256",
        )
        if f'"{key}"' in block
    }
    assert len(required) >= 6, required

    text = _text()
    for key in required:
        assert f"`{key}`" in text, key


def test_the_coder_is_told_the_concept_vocabulary_is_in_the_table():
    """verify41: the step blocked ITSELF waiting for an authorization.

    h3 02_build_fixed_anchor_trajectory_representation ran, wrote a schema with
    every required key correctly named (c2a1919 working), and then emitted

        observation_columns=[], profile_columns=[], representation_columns=[]

    with its own stated reason:

        block_reason: "Planner-authorized trajectory concept mapping is
        incomplete for: sofa2_resp, sofa2_coag, sofa2_liver, lactate"
        planner_authorized_concept_mapping_count: 0

    Three of those four names are literally present in the bound table's
    `concept` column (`sofa2_resp`, `sofa2_coag`, `sofa2_liver`; the fourth is
    spelled `lact`). The agent was refusing to read the vocabulary it had been
    handed, waiting on an authorization nothing produces.
    """

    text = _text()

    assert "DISTINCT VALUES OF ITS" in text
    assert "`concept` COLUMN" in text
    # And it must say that the missing authorization is not a reason to stop.
    assert "not a\n  reason to block" in text or "not a reason to block" in text
    assert "instead of emitting an empty representation" in text
