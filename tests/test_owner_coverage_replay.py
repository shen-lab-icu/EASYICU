"""The pre-run coverage replay must never over-state ownership.

This tool exists to answer "will this plan actually get deterministic
executors?" before a run is paid for.  Its only real failure mode is
optimism: a report that counts an owner the run will decline green-lights
exactly the run that then falls through to the LLM coder, and it does so with
the authority of having been checked.

Three concrete over-statements are locked here, each one observed:

* An earlier version dropped ``robustness_specs`` when the plan failed
  validation and scored what was left -- changing the plan's semantics and
  then reporting a precise number for it.
* It reported ``coder`` for steps whose owner needs the host's
  ``resolved_bindings``.  Offline those decline by construction, which is not
  evidence about what the run will do.
* A receipt-gated owner counted as owned.  An earlier scan had no scope
  object, could not see that gate, and called the E1 Step 02 shape owned; the
  real run recorded ``declined_receipt_required``.  That particular owner now
  renders its own receipt and really is selected, so the conditional case here
  is carried by a renderer that still genuinely cannot emit one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.owner_coverage_replay import (
    CODER,
    CONDITIONAL_RECEIPT,
    OWNED,
    UNKNOWN_RUNTIME_BINDING,
    PlanNotScannable,
    SelectionContextSnapshot,
    load_plan,
    main,
    replay_owner_coverage,
)


def _plan_payload(*steps: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    payload = {
        "research_question": "Estimate an association in an ICU cohort.",
        "robustness_specs": [],
        "steps": list(steps),
    }
    payload.update(overrides)
    return payload


def _receipt_gated_step() -> dict[str, Any]:
    """A renderer that a plausibility receipt would take away.

    It reads two parent tables rather than the ranged raw columns, so it has
    nothing to compare against the sealed bounds and the selector declines it
    when the step owes a receipt.  ``descriptive_cohort_summary`` was the
    original instance; it now renders the receipt itself and is selected, so
    the conditional shape moved here.
    """

    return {
        "step_id": "04_prevalence_mortality_figure",
        "intent": "Render the sealed prevalence and mortality tables.",
        "method": "visualization",
        "planned_analysis_role": "auxiliary",
        "inputs": ["table:cohort_summary", "table:outcome_incidence"],
        "expected_outputs": ["figure:prevalence_mortality"],
        "input_consumption_contracts": [
            {"input_key": "table:cohort_summary", "mode": "all_rows"},
            {"input_key": "table:outcome_incidence", "mode": "all_rows"},
        ],
    }


def _receipt_free_cohort_summary_step() -> dict[str, Any]:
    """The E1 Step 02 shape, which a receipt obligation no longer disowns."""

    return {
        "step_id": "02_cohort_definition_summary",
        "intent": "Summarise the analysis cohort.",
        "method": "descriptive_cohort_summary",
        "planned_analysis_role": "auxiliary",
        "inputs": ["artifact:analysis_cohort", "sofa2_liver_max"],
        "expected_outputs": ["table:cohort_summary"],
    }


def _audit_step() -> dict[str, Any]:
    """Owned outright: the audit contract owes no plausibility receipt."""

    return {
        "step_id": "05_missingness_measurement_process_audit",
        "intent": "Audit measurement process.",
        "method": "data_quality_audit",
        "planned_analysis_role": "auxiliary",
        "inputs": ["artifact:analysis_cohort", "sep3_sofa2_measured"],
        "expected_outputs": [
            "table:missingness_measurement_audit",
            "table:measurement_process_audit",
        ],
    }


def _unowned_step() -> dict[str, Any]:
    return {
        "step_id": "09_bespoke_analysis",
        "intent": "Something outside every closed contract.",
        "method": "bespoke_method",
        "inputs": [],
        "expected_outputs": ["table:bespoke_product"],
    }


def _downstream_figure_step() -> dict[str, Any]:
    """Reads a product another step promises: its schema is a runtime fact."""

    return {
        "step_id": "06_downstream_render",
        "intent": "Render a product produced upstream.",
        "method": "visualization",
        "planned_analysis_role": "auxiliary",
        "inputs": ["table:bespoke_product"],
        "expected_outputs": ["figure:bespoke_render"],
    }


def _write(
    tmp_path: Path, payload: dict[str, Any], name: str = "analysis_plan.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# A plan that did not validate produces no coverage at all.
# --------------------------------------------------------------------------


def test_a_plan_that_does_not_validate_is_refused_not_repaired(
    tmp_path: Path,
) -> None:
    """The prior version dropped robustness_specs and scored the remainder.

    That is a different plan.  Reporting ownership for it, precise to the
    step, invites the operator to launch on an answer about something else.
    """

    payload = _plan_payload(
        _audit_step(),
        robustness_specs=[{"spec_id": "rs1", "kind": "not_a_kind", "description": "x"}],
    )
    plan_path = _write(tmp_path, payload)

    with pytest.raises(PlanNotScannable) as caught:
        replay_owner_coverage(plan_path)

    assert caught.value.reason_code == "invalid_plan"


def test_the_refused_plan_prints_no_coverage_number(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _plan_payload(
        _audit_step(),
        robustness_specs=[{"spec_id": "rs1", "kind": "not_a_kind", "description": "x"}],
    )
    plan_path = _write(tmp_path, payload)

    assert main([str(plan_path)]) == 2

    captured = capsys.readouterr()
    assert "invalid_plan" in captured.err
    # Not one ownership claim may survive onto either stream.
    assert "owned" not in captured.out
    assert captured.out.strip() == ""


def test_robustness_specs_are_never_dropped_to_make_a_plan_scan(
    tmp_path: Path,
) -> None:
    """Lock the specific coercion that caused this: no silent field removal."""

    payload = _plan_payload(
        _audit_step(),
        robustness_specs=[{"spec_id": "rs1", "kind": "not_a_kind", "description": "x"}],
    )
    plan_path = _write(tmp_path, payload)

    with pytest.raises(PlanNotScannable):
        load_plan(plan_path)


def test_a_plan_that_is_not_json_is_refused_with_its_own_reason(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "analysis_plan.json"
    plan_path.write_text("{ not json", encoding="utf-8")

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(plan_path)

    assert caught.value.reason_code == "plan_not_json"


# --------------------------------------------------------------------------
# Unknown is not coder.
# --------------------------------------------------------------------------


def test_a_step_awaiting_a_parent_binding_is_unknown_not_coder(
    tmp_path: Path,
) -> None:
    """Offline these decline by construction; that is not a prediction."""

    plan_path = _write(
        tmp_path, _plan_payload(_unowned_step(), _downstream_figure_step())
    )

    rows = {row.step_id: row for row in replay_owner_coverage(plan_path)}

    downstream = rows["06_downstream_render"]
    assert downstream.verdict == UNKNOWN_RUNTIME_BINDING
    assert "table:bespoke_product" in downstream.note
    # The producing step reads nothing upstream, so its coder verdict stands.
    assert rows["09_bespoke_analysis"].verdict == CODER


def test_a_step_reading_only_the_cohort_is_a_supportable_coder_verdict(
    tmp_path: Path,
) -> None:
    """``artifact:`` inputs are host knowledge, not a parent's runtime schema.

    Treating them as unknown would make almost every step unknown and the
    report useless.
    """

    step = dict(_unowned_step())
    step["inputs"] = ["artifact:analysis_cohort", "age"]
    plan_path = _write(tmp_path, _plan_payload(step))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == CODER


def test_supplying_the_parent_binding_resolves_the_unknown(
    tmp_path: Path,
) -> None:
    """The snapshot is the way to sharpen an answer, not a wider matcher."""

    plan_path = _write(
        tmp_path, _plan_payload(_unowned_step(), _downstream_figure_step())
    )
    plan = load_plan(plan_path)
    snapshot = SelectionContextSnapshot(
        plan=plan,
        resolved_bindings={"06_downstream_render": {}},
    )

    rows = {row.step_id: row for row in replay_owner_coverage(snapshot=snapshot)}

    # With bindings in hand the selector's answer is definite: no owner claims
    # this contract, so the coder verdict is now supported rather than guessed.
    assert rows["06_downstream_render"].verdict == CODER


# --------------------------------------------------------------------------
# A receipt-gated owner is unknown, never owned.
# --------------------------------------------------------------------------


def test_a_receipt_gated_owner_is_reported_conditional_not_owned(
    tmp_path: Path,
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == CONDITIONAL_RECEIPT
    assert row.analysis_kind == "prevalence_mortality_figure"
    assert "receipt" in row.note


def test_conditional_steps_are_not_counted_as_covered(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The headline number is what an operator acts on."""

    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))

    assert main([str(plan_path)]) == 0

    out = capsys.readouterr().out
    assert "owned outright          : 0/1" in out
    assert "unknown (receipt)       : 1/1" in out


def test_a_real_obligation_replaces_the_probe_with_a_definite_answer(
    tmp_path: Path,
) -> None:
    """A known empty obligation means the owner really does survive."""

    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )

    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))
    plan = load_plan(plan_path)
    snapshot = SelectionContextSnapshot(
        plan=plan,
        plausibility_scopes={
            "04_prevalence_mortality_figure": FlagOnlyPlausibilityScope(
                step_id="04_prevalence_mortality_figure",
                expected_columns=(),
                source_contracts_sha256="0" * 64,
                authority_kind="resolved_raw_input_contracts",
            )
        },
    )

    (row,) = replay_owner_coverage(snapshot=snapshot)

    assert row.verdict == OWNED
    assert row.analysis_kind == "prevalence_mortality_figure"


def test_a_real_nonempty_obligation_settles_the_step_against_the_owner(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.authority.plausibility import (
        FlagOnlyPlausibilityScope,
    )

    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))
    plan = load_plan(plan_path)
    snapshot = SelectionContextSnapshot(
        plan=plan,
        plausibility_scopes={
            "04_prevalence_mortality_figure": FlagOnlyPlausibilityScope(
                step_id="04_prevalence_mortality_figure",
                expected_columns=("sofa2_liver_max",),
                source_contracts_sha256="0" * 64,
                authority_kind="resolved_raw_input_contracts",
            )
        },
    )

    (row,) = replay_owner_coverage(snapshot=snapshot)

    assert row.verdict == CODER


# --------------------------------------------------------------------------
# Coverage is a description; only a declared protocol makes it a gate.
# --------------------------------------------------------------------------


def test_without_a_declared_protocol_the_report_is_advisory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An open-ended scientific step may legitimately go to the Coder."""

    plan_path = _write(tmp_path, _plan_payload(_unowned_step()))

    assert main([str(plan_path)]) == 0

    out = capsys.readouterr().out
    assert "Advisory only" in out
    assert "not a gate" in out


def test_a_protocol_required_step_without_an_owner_fails_the_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_unowned_step(), _audit_step()))

    exit_code = main([str(plan_path), "--require-deterministic", "09_bespoke_analysis"])

    assert exit_code == 1
    assert "gate failed" in capsys.readouterr().err


def test_a_protocol_required_step_that_is_owned_passes_the_gate(
    tmp_path: Path,
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_unowned_step(), _audit_step()))

    exit_code = main(
        [
            str(plan_path),
            "--require-deterministic",
            "05_missingness_measurement_process_audit",
        ]
    )

    assert exit_code == 0


def test_a_conditional_step_cannot_satisfy_the_gate(
    tmp_path: Path,
) -> None:
    """Unknown must not be spent as if it were owned."""

    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step()))

    exit_code = main(
        [str(plan_path), "--require-deterministic", "04_prevalence_mortality_figure"]
    )

    assert exit_code == 1


def test_requiring_a_step_the_plan_does_not_contain_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A protocol naming a step that vanished must not silently pass."""

    plan_path = _write(tmp_path, _plan_payload(_audit_step()))

    exit_code = main([str(plan_path), "--require-deterministic", "99_renamed_away"])

    assert exit_code == 2
    assert "required_step_absent" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Ordinary reporting.
# --------------------------------------------------------------------------


def test_the_e1_step_02_shape_is_owned_even_when_it_owes_a_receipt(
    tmp_path: Path,
) -> None:
    """The step whose decline this whole tool was built after.

    Reported as ``declined_receipt_required`` by the real run, then reported as
    owned by a scan that could not see the gate.  It is now genuinely owned:
    the executor renders the receipt itself.
    """

    plan_path = _write(tmp_path, _plan_payload(_receipt_free_cohort_summary_step()))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == OWNED
    assert row.analysis_kind == "descriptive_cohort_summary"


def test_an_unconditionally_owned_step_is_reported_owned(tmp_path: Path) -> None:
    plan_path = _write(tmp_path, _plan_payload(_audit_step()))

    (row,) = replay_owner_coverage(plan_path)

    assert row.verdict == OWNED
    assert row.analysis_kind == "declared_missingness_audit_products"


def test_a_raising_step_degrades_to_coder_rather_than_killing_the_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan that dies on one step tells the operator nothing about the rest."""

    def _explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("selector exploded")

    monkeypatch.setattr(
        "easyicu.research_agent.execution.runners.selection."
        "select_standard_executor",
        _explode,
    )
    plan_path = _write(tmp_path, _plan_payload(_unowned_step(), _receipt_gated_step()))

    rows = replay_owner_coverage(plan_path)

    assert len(rows) == 2
    assert all(row.verdict == CODER for row in rows)
    assert all("RuntimeError" in row.note for row in rows)


def test_json_output_carries_the_tally_and_the_declared_protocol(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = _write(tmp_path, _plan_payload(_receipt_gated_step(), _audit_step()))

    assert (
        main(
            [
                str(plan_path),
                "--json",
                "--require-deterministic",
                "05_missingness_measurement_process_audit",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert [row["verdict"] for row in payload["steps"]] == [
        CONDITIONAL_RECEIPT,
        OWNED,
    ]
    assert payload["tally"][OWNED] == 1
    assert payload["tally"][CONDITIONAL_RECEIPT] == 1
    assert payload["deterministic_required_step_ids"] == [
        "05_missingness_measurement_process_audit"
    ]
