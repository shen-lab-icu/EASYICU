"""The ambient trajectory must be named in the record the script verifies.

MEASURED (h3, run_20260805T065731_9c5d79, step
``02_build_fixed_anchor_representation``): the step declared
``manifest:trajectory_window_manifest``, so its Coder prompt carried the
MANDATORY paragraph saying the fixed windows come from ``TRAJECTORY_PARQUET``.
The generated script wrote a correct loader for that table -- right columns,
right env var -- and then threw the result away::

    # This step has only the typed analysis-cohort input.  Use the explicitly
    # registered fixed-window columns and do not process the undeclared,
    # potentially very large trajectory table.
    trajectory = pd.DataFrame()

    trajectory["charttime_num"] = finite_numeric_series(trajectory["charttime"])

It died on ``KeyError: 'charttime'`` against the empty frame it had just
substituted.  The same script validates the cohort against ``product_contract``
and ``row_count`` read from ``resolved_inputs`` -- so it trusts that record --
and ``resolved_inputs`` named the cohort and nothing else.  The prose lost to
the record.

These tests lock the record naming the table, and lock the two ways the fix
could be quietly undone: dropping the entry, or emitting it for runs that have
no trajectory at all.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.typed_binding import (
    HOST_AUTHORIZED_AMBIENT_INPUTS_SCHEMA_VERSION,
    _write_resolved_inputs_manifest,
    host_authorized_ambient_trajectory_entry,
)

PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/providers/prompts/v1/coder.txt"
)
PHASE = (
    Path(__file__).resolve().parents[2]
    / "src/easyicu/research_agent/execution/phase.py"
)


class _BoundTrajectory:
    """The fields the host verified for the measured run."""

    trajectory_file = "cohort_trajectory.parquet"
    trajectory_sha256 = (
        "5fda5eb4bd9b1a76a28a1ff7eb281d4bd2837214150c771124c37b07bdd67250"
    )
    trajectory_columns = ["stay_id", "charttime", "concept", "value_num", "value_str"]
    identity_column = "stay_id"
    time_column = "charttime"
    concept_column = "concept"
    numeric_value_column = "value_num"
    text_value_column = "value_str"
    materialized_concepts = [
        "sofa2",
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
        "lact",
    ]
    trajectory_rows = 19067154
    time_unit = "h"
    time_origin = "icu_admission"
    window = {"origin": "icu_admission", "unit": "h", "start_hours": 0.0}


def test_the_entry_carries_the_roles_the_script_had_to_guess() -> None:
    entry = host_authorized_ambient_trajectory_entry(_BoundTrajectory())
    assert entry is not None
    # The column the measured script died on must be readable as a ROLE, not
    # inferred from a name it happened to hardcode.
    assert entry["time_column"] == "charttime"
    assert entry["identity_column"] == "stay_id"
    assert entry["concept_column"] == "concept"
    assert entry["numeric_value_column"] == "value_num"
    assert entry["relative_path"] == "cohort_trajectory.parquet"
    assert entry["sha256"] == _BoundTrajectory.trajectory_sha256


def test_the_entry_carries_the_exact_vocabulary() -> None:
    entry = host_authorized_ambient_trajectory_entry(_BoundTrajectory())
    assert entry is not None
    # The measured script queried "lactate" and "sofa"; the table holds
    # "lact" and "sofa2".  The record must publish what is actually there.
    assert entry["concepts"] == list(_BoundTrajectory.materialized_concepts)
    assert "lact" in entry["concepts"]
    assert "lactate" not in entry["concepts"]


def test_the_entry_states_that_reading_it_is_authorized() -> None:
    entry = host_authorized_ambient_trajectory_entry(_BoundTrajectory())
    assert entry is not None
    authorization = entry["authorization"].lower()
    # The premise of the measured refusal, contradicted in the record itself.
    assert "not an undeclared file" in authorization
    assert "authorized" in authorization


def test_a_run_without_a_trajectory_gets_no_entry() -> None:
    assert host_authorized_ambient_trajectory_entry(None) is None


def test_a_partially_bound_trajectory_is_refused_rather_than_guessed() -> None:
    class _MissingRole(_BoundTrajectory):
        time_column = ""

    assert host_authorized_ambient_trajectory_entry(_MissingRole()) is None

    class _RoleNotInTable(_BoundTrajectory):
        time_column = "measured_at"

    with pytest.raises(ValueError, match="role columns"):
        host_authorized_ambient_trajectory_entry(_RoleNotInTable())


def test_the_manifest_records_the_entry_under_a_versioned_key(
    tmp_path: Path,
) -> None:
    (tmp_path / "cohort_trajectory.parquet").write_bytes(b"parquet")
    path = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="02_build_fixed_anchor_representation",
        planner_declared_inputs=[],
        bindings={},
        host_authorized_ambient_trajectory=host_authorized_ambient_trajectory_entry(
            _BoundTrajectory()
        ),
    )
    payload = json.loads(path.read_text())
    ambient = payload["host_authorized_ambient_inputs"]
    assert ambient["schema_version"] == HOST_AUTHORIZED_AMBIENT_INPUTS_SCHEMA_VERSION
    assert ambient["trajectory"]["time_column"] == "charttime"
    # The record must not smuggle it into the declared-input surface: the plan
    # contract refuses a plan that lists this table, so the two must not fight.
    assert payload["planner_declared_inputs"] == []
    assert payload["inputs"] == {}


def test_a_manifest_without_a_trajectory_is_unchanged(tmp_path: Path) -> None:
    kwargs = dict(
        run_dir=tmp_path,
        step_id="02_build_fixed_anchor_representation",
        planner_declared_inputs=[],
        bindings={},
    )
    path = _write_resolved_inputs_manifest(**kwargs)
    without = path.read_text()
    path = _write_resolved_inputs_manifest(
        **kwargs, host_authorized_ambient_trajectory=None
    )
    # Byte-identical: a wide-column run and a non-trajectory run must not
    # notice this change at all.
    assert path.read_text() == without
    assert "host_authorized_ambient_inputs" not in without


def test_a_trajectory_outside_the_run_directory_is_refused(tmp_path: Path) -> None:
    class _Escaping(_BoundTrajectory):
        trajectory_file = "../elsewhere.parquet"

    (tmp_path.parent / "elsewhere.parquet").write_bytes(b"parquet")
    with pytest.raises(ValueError, match="contained by run_dir"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="02_build_fixed_anchor_representation",
            planner_declared_inputs=[],
            bindings={},
            host_authorized_ambient_trajectory=(
                host_authorized_ambient_trajectory_entry(_Escaping())
            ),
        )


def test_the_write_point_reads_the_bound_trajectory() -> None:
    """The builder must be called where the manifest is written.

    A version of this fix that adds the builder and never calls it would pass
    every test above.
    """

    tree = ast.parse(PHASE.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_write_resolved_inputs_manifest"
    ]
    assert calls, "the resolved-inputs manifest write point disappeared"
    assert any(
        any(
            keyword.arg == "host_authorized_ambient_trajectory"
            and "host_authorized_ambient_trajectory_entry" in ast.dump(keyword.value)
            for keyword in call.keywords
        )
        for call in calls
    ), "the manifest is written without consulting the bound trajectory"


def test_the_coder_is_pointed_at_the_record_not_at_prose() -> None:
    # Whitespace-insensitive: this prompt is hard-wrapped, so a sentence moves
    # across lines whenever a word ahead of it changes.  An assertion that
    # matches raw text tests the wrap, not the instruction.
    flat = " ".join(PROMPT.read_text().lower().split())
    assert "host_authorized_ambient_inputs.trajectory" in flat
    # The absence from `inputs` is the exact premise the measured script used
    # to refuse the read; the prompt must name that absence as intentional.
    assert "not a reason to refuse to read it" in flat
    assert "substitute an empty frame" in flat
    assert "select concepts by exact string from that list" in flat
