"""The half that authorizes columns and the half that checks them must agree.

``research_context.typed.raw_contract_inputs_for_step`` authorizes a raw column
for every predicate coordinate the host's mask actually read, and says why:
"For a predicate the host narrowed to an event-time window that is two columns,
not one, so the event-time column is authorized on the same footing as
``resolved_column``: the Coder is asked to reproduce that predicate's counts,
and it cannot do so from a column it has no contract for."

``authority.typed_binding._write_resolved_inputs_manifest`` then re-derived the
authorized set to check the contracts it is handed, and read ``resolved_column``
alone.  The producer was widened; the checker was not.

canary33 paid for it with a whole run.  The Planner wrote a cohort whose
early-death exclusion was narrowed to hours 0-24, so the receipt named
``resolved_column="death"`` and ``event_time_column="death_time"``.  The
producer authorized ``{age, death, death_time}``, the checker authorized
``{age, death}``, and step 01 raised

    ValueError: raw input contracts must exactly match Planner-declared or
    host-receipt raw inputs

before any analysis ran: 0 of 12 steps, and every later finding in that
manifest is downstream of a first step that never happened.  canary32 survived
the same code only because its plan declared no cohort predicates at all, so
both sides were empty and the disagreement had nothing to disagree about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.run_input import canonical_sha256
from easyicu.research_agent.authority.typed_binding import (
    _write_resolved_inputs_manifest,
)
from easyicu.research_agent.contracts.cohort_receipt import (
    COHORT_RECEIPT_COLUMN_FIELDS,
    cohort_receipt_authorized_columns,
)
from easyicu.research_agent.research_context.typed import (
    raw_contract_inputs_for_step,
)

_SRC = Path(__file__).resolve().parents[3] / "src" / "easyicu" / "research_agent"


def _receipt(flow) -> dict:
    """A valid host cohort execution receipt carrying ``flow``."""

    return {
        "schema_version": "easyicu.primary_cohort_execution_prompt/1",
        "cohort_definition_sha256": "c" * 64,
        "raw_universe": {"rows": 94458, "sha256": "a" * 64},
        "authoritative_analysis_cohort": {
            "rows": list(flow)[-1]["n_remaining"],
            "sha256": "b" * 64,
            "identity_column": "stay_id",
            "row_identity_sha256": "d" * 64,
            "authority_sha256": "e" * 64,
        },
        "ordered_predicate_flow": list(flow),
    }


#: canary33's own predicate flow, field for field.
_WINDOWED_FLOW = [
    {
        "step_order": 0,
        "predicate_kind": "universe",
        "resolved_column": None,
        "event_time_column": None,
        "n_before": 94458,
        "n_excluded": 0,
        "n_remaining": 94458,
    },
    {
        "step_order": 1,
        "predicate_kind": "inclusion",
        "resolved_column": "age",
        "event_time_column": None,
        "op": ">=",
        "n_before": 94458,
        "n_excluded": 0,
        "n_remaining": 94458,
    },
    {
        "step_order": 2,
        "predicate_kind": "exclusion",
        "resolved_column": "death",
        "event_time_column": "death_time",
        "event_time_start_hours": 0.0,
        "event_time_end_hours": 24.0,
        "op": "==",
        "n_before": 94458,
        "n_excluded": 2060,
        "n_remaining": 92398,
    },
]


def _contracts_for(names) -> dict:
    """A well-formed contracts payload over exactly ``names``.

    The names come from the real producer; only the envelope is built here, so
    this test never restates the rule it is checking.
    """

    payload = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "contracts": {name: {"dtype": "float64"} for name in names},
    }
    payload["contracts_sha256"] = canonical_sha256(payload)
    return payload


def _write(tmp_path: Path, *, declared, receipt) -> Path:
    names = raw_contract_inputs_for_step(
        planner_declared_inputs=declared,
        primary_cohort_execution_receipt=receipt,
    )
    return _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="01_define_analysis_cohort",
        planner_declared_inputs=list(declared),
        bindings={},
        context_path=None,
        raw_input_contracts=_contracts_for(names),
        host_verified_cohort_execution_receipt=receipt,
    )


# ---------------------------------------------------------------------------
# The run canary33 lost
# ---------------------------------------------------------------------------


def test_a_time_windowed_predicate_does_not_kill_the_first_step(tmp_path):
    """Producer output goes straight into the checker and is accepted."""

    manifest = _write(
        tmp_path, declared=["age", "death"], receipt=_receipt(_WINDOWED_FLOW)
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["raw_input_contracts"]["contracts"]) == {
        "age",
        "death",
        "death_time",
    }


def test_the_event_time_column_is_what_the_two_halves_disagreed_about(tmp_path):
    """Named explicitly, so a future narrowing of either half is visible."""

    authorized = cohort_receipt_authorized_columns(_WINDOWED_FLOW)

    assert authorized == {"age", "death", "death_time"}
    assert "death_time" not in {row.get("resolved_column") for row in _WINDOWED_FLOW}


def test_the_shape_canary32_had_still_works(tmp_path):
    """A plan with no cohort predicates: the case that hid this for weeks."""

    flow = [
        {
            "step_order": 0,
            "predicate_kind": "universe",
            "n_before": 94458,
            "n_excluded": 0,
            "n_remaining": 94458,
        }
    ]
    manifest = _write(tmp_path, declared=[], receipt=_receipt(flow))

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["raw_input_contracts"]["contracts"] == {}


# ---------------------------------------------------------------------------
# The check still fails closed
# ---------------------------------------------------------------------------


def test_a_column_neither_the_plan_nor_the_receipt_names_is_still_refused(tmp_path):
    """Widening the authorized set must not turn the check off.

    ``lact_max`` is in no predicate and in no declared input, so a contract for
    it is a column the step was never authorized to read.
    """

    receipt = _receipt(_WINDOWED_FLOW)
    names = raw_contract_inputs_for_step(
        planner_declared_inputs=["age", "death"],
        primary_cohort_execution_receipt=receipt,
    )

    with pytest.raises(ValueError, match="raw input contracts must exactly match"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="01_define_analysis_cohort",
            planner_declared_inputs=["age", "death"],
            bindings={},
            context_path=None,
            raw_input_contracts=_contracts_for(list(names) + ["lact_max"]),
            host_verified_cohort_execution_receipt=receipt,
        )


def test_a_contract_missing_an_authorized_column_is_still_refused(tmp_path):
    """The check is equality, not containment, in both directions."""

    receipt = _receipt(_WINDOWED_FLOW)

    with pytest.raises(ValueError, match="raw input contracts must exactly match"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="01_define_analysis_cohort",
            planner_declared_inputs=["age", "death"],
            bindings={},
            context_path=None,
            raw_input_contracts=_contracts_for(["age", "death"]),
            host_verified_cohort_execution_receipt=receipt,
        )


# ---------------------------------------------------------------------------
# One declaration, not two
# ---------------------------------------------------------------------------


def test_neither_half_keeps_its_own_list_of_the_receipt_fields():
    """The drift is only fixed if the second copy is gone.

    Both files may name a field in prose or in a comment; what neither may do
    is enumerate the set again in code, because that is exactly what let one
    side be widened alone.
    """

    fields = [field for field, _reason in COHORT_RECEIPT_COLUMN_FIELDS]
    assert fields == ["resolved_column", "event_time_column"]

    for relative in (
        "authority/typed_binding.py",
        "research_context/typed.py",
    ):
        source = (_SRC / relative).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for field in fields:
            assert (
                code.count(f'"{field}"') + code.count(f"'{field}'") <= 1
            ), f"{relative} enumerates {field} again instead of reading the declaration"


def test_every_declared_field_authorizes_its_column():
    """Sweeps the declaration, so adding a field cannot skip the producer."""

    for field, _reason in COHORT_RECEIPT_COLUMN_FIELDS:
        flow = [
            {
                "step_order": 1,
                "predicate_kind": "inclusion",
                field: "some_column",
                "n_before": 94458,
                "n_excluded": 0,
                "n_remaining": 94458,
            }
        ]

        assert cohort_receipt_authorized_columns(flow) == {"some_column"}
        assert "some_column" in raw_contract_inputs_for_step(
            planner_declared_inputs=[],
            primary_cohort_execution_receipt=_receipt(flow),
        )


def test_a_blank_or_non_string_column_names_nothing():
    """Each caller validates with its own exception; this only reads."""

    assert cohort_receipt_authorized_columns([{"resolved_column": "   "}]) == set()
    assert cohort_receipt_authorized_columns([{"resolved_column": 7}]) == set()
    assert cohort_receipt_authorized_columns([None, "not a row"]) == set()
    assert cohort_receipt_authorized_columns(None) == set()
