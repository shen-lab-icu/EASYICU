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
    load_run_context,
    main,
    receipt_obligations_from_run,
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


# --------------------------------------------------------------------------
# "I cannot validate this" is not "the pipeline would reject this".
#
# The E1 plan names ``icu_readmission`` in a robustness spec's cohort
# override.  It is not a packaged dictionary concept; it is a column of the
# materialised cohort, which the bench registers with
# ``register_cohort_concept_ids`` before planning.  The pipeline accepted that
# plan and ran it.  Reporting it as ``invalid_plan`` -- printed as "a plan the
# pipeline would reject" -- was a false statement about a run that happened,
# and it is the permissive-direction failure inverted: over-strict, but still
# an answer the tool did not hold.
# --------------------------------------------------------------------------


_MATERIALISED_COLUMN = "icu_readmission"


def _plan_naming_a_materialised_column() -> dict[str, Any]:
    return _plan_payload(
        _audit_step(),
        robustness_specs=[
            {
                "spec_id": "rs_readmission",
                "description": "Exclude readmissions.",
                "axis": "cohort",
                "cohort_override": {
                    "name": "no_readmission",
                    "selection_mode": "predicate_filtered",
                    "inclusion": [
                        {
                            "concept_id": _MATERIALISED_COLUMN,
                            "op": "==",
                            "value": 0,
                            "aggregation": "any",
                            "time_window": {
                                "anchor": "icu_admit",
                                "start_offset_hours": 0.0,
                                "end_offset_hours": 24.0,
                            },
                        }
                    ],
                    "exclusion": [],
                },
            }
        ],
    )


def _raw_contracts(*columns: str) -> dict[str, Any]:
    """The exact resolved-contract envelope the host seals, digest included."""

    import hashlib

    payload: dict[str, Any] = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "authority_scope": (
            "host_verified_physical_representation_and_domain_constraints"
        ),
        "scientific_ownership": "Planner retains scientific decisions",
        "contracts": {
            column: {
                "column": column,
                "physical_role": "value",
                "availability_basis": "direct_source",
                "analysis_plausibility_range": {"minimum": 0.0, "maximum": 10.0},
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
            for column in columns
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["contracts_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _write_run_dir(
    tmp_path: Path,
    *,
    plan_payload: dict[str, Any] | None = None,
    cohort_columns: tuple[str, ...] = (_MATERIALISED_COLUMN, "age", "death"),
    cohort_sha256: str = "c" * 64,
    authority_cohort_sha256: str | None = None,
    tamper_authority: bool = False,
    tamper_plan: bool = False,
    tamper_bindings_for: str | None = None,
    drop_binding_digest_for: str | None = None,
    duplicate_record_for: str | None = None,
    stray_binding_file: bool = False,
    evidence_disagrees: bool = False,
    resolved_inputs: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """A run directory shaped like the ones the pipeline actually seals.

    Every artefact the loader consumes is reached through a digest the manifest
    records, so the fixture has to write those digests too -- which is exactly
    what makes the tampering cases below expressible.
    """

    import hashlib

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)

    authority = {
        "cohort_sha256": authority_cohort_sha256 or cohort_sha256,
        "cohort_columns": list(cohort_columns),
    }
    body = json.dumps(authority, indent=2).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    name = f"cohort_authority.sha256-{digest}.json"
    if tamper_authority:
        body += b"\n"
    (run_dir / name).write_bytes(body)
    (run_dir / "run_input_capsule.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.run_input_capsule/2",
                "cohort_sha256": cohort_sha256,
                "materialized_cohort_authority_ref": {
                    "file": name,
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    plan_rel = "evidence/analysis_plan_revision_3__analysis_plan_revision_3.json"
    plan_body = json.dumps(
        plan_payload
        if plan_payload is not None
        else _plan_naming_a_materialised_column()
    ).encode("utf-8")
    plan_digest = hashlib.sha256(plan_body).hexdigest()
    if tamper_plan:
        plan_body += b"\n"
    (run_dir / plan_rel).write_bytes(plan_body)

    binding_dir = run_dir / "resolved_inputs"
    binding_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for step_id, payload in (resolved_inputs or {}).items():
        capsule = json.dumps({"step_id": step_id, **payload}).encode("utf-8")
        capsule_digest = hashlib.sha256(capsule).hexdigest()
        if tamper_bindings_for == step_id:
            capsule += b"\n"
        (binding_dir / f"{step_id}.json").write_bytes(capsule)
        records.append(
            {
                "step_id": step_id,
                "resolved_inputs_path": f"resolved_inputs/{step_id}.json",
                "resolved_inputs_sha256": (
                    None if drop_binding_digest_for == step_id else capsule_digest
                ),
            }
        )
    if duplicate_record_for is not None:
        records.append(
            {
                "step_id": duplicate_record_for,
                "resolved_inputs_path": f"resolved_inputs/{duplicate_record_for}.json",
                "resolved_inputs_sha256": "0" * 64,
            }
        )
    if stray_binding_file:
        (binding_dir / "99_left_over_attempt.json").write_text("{}", encoding="utf-8")

    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "plan_path": plan_rel,
                "current_plan_authority": {
                    "schema_version": "easyicu.current_plan_authority/1",
                    "evidence_id": "analysis_plan_revision_3",
                    "relative_path": plan_rel,
                    "sha256": plan_digest,
                    "revision": 3,
                },
                "evidence": [
                    {
                        "evidence_id": "analysis_plan_revision_3",
                        "relative_path": plan_rel,
                        "sha256": ("f" * 64) if evidence_disagrees else plan_digest,
                    }
                ],
                "per_step_records": records,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _authority_plan(run_dir: Path) -> Path:
    return run_dir / "evidence/analysis_plan_revision_3__analysis_plan_revision_3.json"


def test_an_id_a_run_registers_is_missing_context_not_an_invalid_plan(
    tmp_path: Path,
) -> None:
    plan_path = _write(tmp_path, _plan_naming_a_materialised_column())

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(plan_path)

    assert caught.value.reason_code == "missing_validation_context"
    assert _MATERIALISED_COLUMN in str(caught.value)
    # The false claim that produced this fix must not survive anywhere in the
    # message: this plan is one the pipeline accepted.
    assert "would reject" not in str(caught.value)


def test_the_run_registry_makes_the_same_plan_validate(tmp_path: Path) -> None:
    plan_path = _write(tmp_path, _plan_naming_a_materialised_column())
    run_dir = _write_run_dir(tmp_path)

    plan = load_plan(plan_path, run_context=load_run_context(run_dir))

    assert [str(step.step_id) for step in plan.steps] == [
        "05_missingness_measurement_process_audit"
    ]
    assert len(plan.robustness_specs) == 1


def test_registering_the_run_registry_does_not_leak_out_of_the_call(
    tmp_path: Path,
) -> None:
    """The next question must get the same answer as the first one."""

    plan_path = _write(tmp_path, _plan_naming_a_materialised_column())
    load_plan(plan_path, run_context=load_run_context(_write_run_dir(tmp_path)))

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(plan_path)
    assert caught.value.reason_code == "missing_validation_context"


def test_a_column_no_run_context_backs_is_still_missing_context(
    tmp_path: Path,
) -> None:
    """A context that does not cover the id proves nothing about the plan.

    It may be the wrong run's context.  Answering ``invalid_plan`` here would
    let a mismatched pair of inputs condemn a plan that is fine.
    """

    plan_path = _write(tmp_path, _plan_naming_a_materialised_column())
    run_dir = _write_run_dir(tmp_path, cohort_columns=("age", "death"))

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(plan_path, run_context=load_run_context(run_dir))

    assert caught.value.reason_code == "missing_validation_context"
    assert "would reject" not in str(caught.value)


def test_a_plan_broken_for_any_other_reason_is_still_invalid(
    tmp_path: Path,
) -> None:
    """The new reason code must not become a blanket excuse."""

    payload = _plan_payload(
        _audit_step(),
        robustness_specs=[{"spec_id": "rs1", "kind": "not_a_kind", "description": "x"}],
    )
    run_dir = _write_run_dir(tmp_path, plan_payload=payload)

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(_authority_plan(run_dir), run_context=load_run_context(run_dir))

    assert caught.value.reason_code == "invalid_plan"


# --------------------------------------------------------------------------
# The registry is reached by digest, not by directory listing.
# --------------------------------------------------------------------------


def test_an_authority_that_does_not_match_its_declared_digest_is_refused(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_dir(tmp_path, tamper_authority=True)

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert caught.value.reason_code == "missing_validation_context"
    assert "digest" in str(caught.value)


def test_an_authority_for_a_different_cohort_is_refused(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path, cohort_sha256="a" * 64, authority_cohort_sha256="b" * 64
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert caught.value.reason_code == "missing_validation_context"
    assert "different cohort" in str(caught.value)


def test_a_run_without_a_capsule_reference_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    (run_dir / "run_input_capsule.json").write_text(
        json.dumps({"cohort_sha256": "d" * 64}), encoding="utf-8"
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert caught.value.reason_code == "missing_validation_context"


# --------------------------------------------------------------------------
# Recorded bindings turn an unknown into a definite verdict.
# --------------------------------------------------------------------------


def test_recorded_bindings_settle_a_step_the_offline_scan_could_not(
    tmp_path: Path,
) -> None:
    payload = _plan_payload(_unowned_step(), _downstream_figure_step())
    run_dir = _write_run_dir(
        tmp_path,
        plan_payload=payload,
        resolved_inputs={
            "06_downstream_render": {
                "inputs": {"table:bespoke_product": {"product": "bespoke_product"}}
            }
        },
    )
    plan_path = _authority_plan(run_dir)
    context = load_run_context(run_dir)

    without = replay_owner_coverage(plan_path)
    with_run = replay_owner_coverage(
        snapshot=SelectionContextSnapshot(
            plan=load_plan(plan_path, run_context=context),
            resolved_bindings=context.resolved_bindings,
        )
    )

    assert without[1].verdict == UNKNOWN_RUNTIME_BINDING
    assert with_run[1].verdict == CODER


def test_a_step_the_run_never_reached_stays_unknown(tmp_path: Path) -> None:
    """Absent bindings are absent, not an answer."""

    payload = _plan_payload(_unowned_step(), _downstream_figure_step())
    run_dir = _write_run_dir(tmp_path, plan_payload=payload)
    plan_path = _authority_plan(run_dir)
    context = load_run_context(run_dir)

    rows = replay_owner_coverage(
        snapshot=SelectionContextSnapshot(
            plan=load_plan(plan_path, run_context=context),
            resolved_bindings=context.resolved_bindings,
        )
    )

    assert rows[1].verdict == UNKNOWN_RUNTIME_BINDING


def test_obligations_are_only_compiled_for_steps_the_run_recorded(
    tmp_path: Path,
) -> None:
    """No recorded contracts must not read as "this step owes no receipt"."""

    payload = _plan_payload(_audit_step(), _receipt_gated_step())
    run_dir = _write_run_dir(
        tmp_path,
        plan_payload=payload,
        resolved_inputs={
            "05_missingness_measurement_process_audit": {
                "inputs": {},
                "raw_input_contracts": _raw_contracts("sep3_sofa2_measured"),
            }
        },
    )
    context = load_run_context(run_dir)
    plan = load_plan(_authority_plan(run_dir), run_context=context)

    scopes = receipt_obligations_from_run(plan, run_context=context)

    assert set(scopes) == {"05_missingness_measurement_process_audit"}


# --------------------------------------------------------------------------
# Every binding is read at the digest the manifest recorded for it.
#
# The first version of the run-context loader scanned ``resolved_inputs/*.json``
# off the directory.  That accepts a capsule the run never sealed -- a leftover
# from an earlier attempt, or an edited one -- and every verdict downstream
# silently inherits it.  Only the cohort authority was actually digest-bound,
# so "all digest-bound" was not yet true.
# --------------------------------------------------------------------------


def _one_binding() -> dict[str, dict[str, Any]]:
    return {
        "06_downstream_render": {
            "inputs": {"table:bespoke_product": {"product": "bespoke_product"}}
        }
    }


def test_a_binding_capsule_that_does_not_match_its_digest_is_refused(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        resolved_inputs=_one_binding(),
        tamper_bindings_for="06_downstream_render",
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert caught.value.reason_code == "missing_validation_context"
    assert "declared digest" in str(caught.value)


def test_a_path_without_a_digest_is_refused(tmp_path: Path) -> None:
    """Half a receipt proves nothing; it must not read as a whole one."""

    run_dir = _write_run_dir(
        tmp_path,
        resolved_inputs=_one_binding(),
        drop_binding_digest_for="06_downstream_render",
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert "half a binding receipt" in str(caught.value)


def test_two_manifest_records_for_one_step_are_refused(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path,
        resolved_inputs=_one_binding(),
        duplicate_record_for="06_downstream_render",
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert "more than once" in str(caught.value)


def test_a_binding_capsule_no_manifest_record_claims_is_refused(
    tmp_path: Path,
) -> None:
    """The directory-scan defect, pinned from the other side."""

    run_dir = _write_run_dir(
        tmp_path, resolved_inputs=_one_binding(), stray_binding_file=True
    )

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert "99_left_over_attempt.json" in str(caught.value)


def test_a_capsule_filed_under_another_steps_id_is_refused(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, resolved_inputs=_one_binding())
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["per_step_records"][0]["step_id"] = "07_a_different_step"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert "filed it under" in str(caught.value)


def test_a_binding_path_outside_the_run_is_refused(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, resolved_inputs=_one_binding())
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["per_step_records"][0]["resolved_inputs_path"] = "../elsewhere.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(run_dir)

    assert "escapes the run directory" in str(caught.value)


# --------------------------------------------------------------------------
# The scanned plan must be the revision the run treated as authoritative.
# --------------------------------------------------------------------------


def test_a_tampered_authority_plan_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(_write_run_dir(tmp_path, tamper_plan=True))

    assert caught.value.reason_code == "missing_validation_context"
    assert "current_plan_authority" in str(caught.value)


def test_an_evidence_record_that_disagrees_with_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    """Two records of which plan is authoritative must agree, or neither counts."""

    with pytest.raises(PlanNotScannable) as caught:
        load_run_context(_write_run_dir(tmp_path, evidence_disagrees=True))

    assert "disagrees with current_plan_authority" in str(caught.value)


def test_scanning_a_plan_that_is_not_the_authority_is_refused(
    tmp_path: Path,
) -> None:
    """A file name check would have passed this; only the digest catches it.

    The revision the run executed and the copy sitting at the run root can be
    different bytes -- in the real E1 run they are.  Pairing one run's context
    with another revision produces verdicts for neither.
    """

    run_dir = _write_run_dir(tmp_path)
    other = _write(tmp_path, _plan_payload(_audit_step()), name="analysis_plan.json")

    with pytest.raises(PlanNotScannable) as caught:
        load_plan(other, run_context=load_run_context(run_dir))

    assert caught.value.reason_code == "plan_not_authority"


def test_run_dir_alone_scans_the_authority_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The run knows which revision it ran; the operator should not have to."""

    run_dir = _write_run_dir(
        tmp_path,
        plan_payload=_plan_payload(_audit_step()),
        resolved_inputs=_one_binding(),
    )

    assert main(["--run-dir", str(run_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["step_id"] for row in payload["steps"]] == [
        "05_missingness_measurement_process_audit"
    ]
