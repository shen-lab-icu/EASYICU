from __future__ import annotations

import json
import re
from pathlib import Path

from easyicu.research_agent import code_repair
from easyicu.research_agent.repair_registry import (
    InvariantStatus,
    RepairClass,
    RepairLedger,
    RepairObservedState,
    assert_registry_invariants,
    evaluate_invariants,
    make_repair_provenance,
    repair_metadata_for,
)


def test_repair_registry_invariants_hold() -> None:
    assert_registry_invariants()


def test_literal_code_repair_ids_are_classified() -> None:
    source = Path(code_repair.__file__).read_text(encoding="utf-8")
    repair_ids = set(re.findall(r"repair_name\s*=\s*['\"]([^'\"]+)['\"]", source))
    assert repair_ids
    unclassified = [
        repair_id
        for repair_id in sorted(repair_ids)
        if repair_metadata_for(repair_id).classification_source.startswith("fallback:")
    ]
    assert unclassified == []


def test_dynamic_repair_id_patterns_are_classified() -> None:
    assert (
        repair_metadata_for("strip_fake_easyicu_import_easyicu_foo_v1").repair_class
        is RepairClass.SYNTACTIC
    )
    assert (
        repair_metadata_for("undefined_helper_stub_to_json_serializable_v1").repair_class
        is RepairClass.STRUCTURAL
    )


def test_retired_case_specific_repair_ids_are_not_registered() -> None:
    retired = [
        "generic_v15_table_one_fallback_v1",
        "age_stratified_mortality_dependency_free_v1",
        "norepinephrine_dose_response_dependency_free_v1",
        "robustness_complete_case_or_fallback_v1",
        "shock_primary_assoc_sklearn_v1",
    ]

    for repair_id in retired:
        metadata = repair_metadata_for(repair_id)
        assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
        assert metadata.classification_source == "fallback:unknown_method_substitution"
        assert metadata.requires_disclosure is True


def test_nonconvergence_fallback_is_method_substitution() -> None:
    metadata = repair_metadata_for("validation_nonconvergence_fallback_v1")
    assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
    assert metadata.introduces_numbers is True
    assert metadata.requires_disclosure is True


def test_contract_fill_requires_selection_rule() -> None:
    metadata = repair_metadata_for("categorical_primary_association_selection_v1")
    assert metadata.repair_class is RepairClass.CONTRACT_FILL
    assert metadata.selection_rule_required is True


def test_repair_ledger_writes_provenance_json(tmp_path: Path) -> None:
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    provenance = ledger.append_application(
        repair_id="statsmodels_endog_exog_index_align_v1",
        step_id="04_model",
        trigger={"error_type": "ValueError"},
        transformation="Aligned endog/exog indices.",
        before_text="before",
        after_text="after",
    )
    assert provenance.repair_class == RepairClass.STRUCTURAL.value

    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    assert payload["schema_version"] == "easyicu.repair_ledger/1"
    assert payload["repairs"][0]["repair_id"] == "statsmodels_endog_exog_index_align_v1"
    assert payload["repairs"][0]["before_hash"].startswith("sha256:")
    assert payload["repairs"][0]["after_hash"].startswith("sha256:")


def test_make_repair_provenance_conservatively_classifies_unknown() -> None:
    provenance = make_repair_provenance(
        repair_id="future_unreviewed_repair_v1",
        step_id="01_step",
    )
    assert provenance.repair_class == RepairClass.METHOD_SUBSTITUTION.value
    assert provenance.classification_source == "fallback:unknown_method_substitution"
    assert provenance.requires_disclosure is True


# --- P1: runtime invariant verification -----------------------------------


def test_structural_invariant_unverified_without_state_is_not_a_pass() -> None:
    """The P0 honesty hole: a STRUCTURAL repair must not report a pass when no
    state was supplied to check ``row_set_unchanged`` / ``n_unchanged``."""

    provenance = make_repair_provenance(
        repair_id="statsmodels_endog_exog_index_align_v1",
        step_id="04_model",
    )
    assert provenance.invariant_status == InvariantStatus.UNVERIFIED.value
    assert provenance.invariants_passed is None


def test_structural_n_unchanged_verified_pass_and_fail() -> None:
    metadata = repair_metadata_for("statsmodels_endog_exog_index_align_v1")

    ok = evaluate_invariants(
        metadata,
        before_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
        after_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
    )
    assert ok.status == InvariantStatus.VERIFIED_PASS.value
    assert ok.passed is True
    assert ok.failures == ()

    dropped = evaluate_invariants(
        metadata,
        before_state=RepairObservedState(row_count=500, id_values=(1, 2, 3)),
        after_state=RepairObservedState(row_count=498, id_values=(1, 2)),
    )
    assert dropped.status == InvariantStatus.VERIFIED_FAIL.value
    assert dropped.passed is False
    assert "n_unchanged" in dropped.failures
    assert "row_set_unchanged" in dropped.failures


def test_syntactic_repair_has_no_invariants_and_passes_vacuously() -> None:
    metadata = repair_metadata_for("missing_os_import_v1")
    evaluation = evaluate_invariants(metadata)
    assert evaluation.status == InvariantStatus.VERIFIED_PASS.value
    assert evaluation.passed is True


def test_contract_fill_without_selection_rule_fails() -> None:
    metadata = repair_metadata_for("categorical_primary_association_selection_v1")

    missing = evaluate_invariants(metadata, selection_rule=None)
    assert missing.status == InvariantStatus.VERIFIED_FAIL.value
    assert "deterministic_selection_rule" in missing.failures

    # A present rule clears that invariant, but the others remain unobservable
    # at this layer, so the overall result is honestly UNVERIFIED, not a pass.
    with_rule = evaluate_invariants(
        metadata, selection_rule="first_finite_adjusted_or"
    )
    assert with_rule.status == InvariantStatus.UNVERIFIED.value


def test_ledger_records_invariant_status(tmp_path: Path) -> None:
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    ledger.append_application(
        repair_id="statsmodels_endog_exog_index_align_v1",
        step_id="04_model",
        before_state=RepairObservedState(row_count=500),
        after_state=RepairObservedState(row_count=480),
    )
    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    record = payload["repairs"][0]
    assert record["invariant_status"] == InvariantStatus.VERIFIED_FAIL.value
    assert "n_unchanged" in record["invariant_failures"]


# --- P1.5: step-summary salvage is inside the provenance net ----------------


def test_summary_salvage_repairs_are_classified() -> None:
    assert (
        repair_metadata_for("summary_salvage_stdout_json_v1").repair_class
        is RepairClass.STRUCTURAL
    )
    assert (
        repair_metadata_for("summary_salvage_named_json_v1").repair_class
        is RepairClass.STRUCTURAL
    )
    minimal = repair_metadata_for("summary_salvage_minimal_contract_v1")
    assert minimal.repair_class is RepairClass.CONTRACT_FILL
    assert minimal.selection_rule_required is True
    # None of the salvage ids may fall through to the conservative
    # unknown -> METHOD_SUBSTITUTION classification.
    for repair_id in (
        "summary_salvage_stdout_json_v1",
        "summary_salvage_named_json_v1",
        "summary_salvage_minimal_contract_v1",
    ):
        assert not repair_metadata_for(repair_id).classification_source.startswith(
            "fallback:"
        )


def test_summary_salvage_minimal_contract_records_selection_rule() -> None:
    provenance = make_repair_provenance(
        repair_id="summary_salvage_minimal_contract_v1",
        step_id="03_model",
        selection_rule="first non-const association row; mean of perf rows",
    )
    assert provenance.repair_class == RepairClass.CONTRACT_FILL.value
    assert provenance.selection_rule
    # A CONTRACT_FILL whose other invariants are unobservable here is honestly
    # UNVERIFIED, never a fabricated pass.
    assert provenance.invariant_status == InvariantStatus.UNVERIFIED.value


# --- P1.5 integration: real salvage -> classified outcome -> ledger entry ---


def test_salvage_step_summary_records_stdout_salvage_end_to_end(tmp_path: Path) -> None:
    from easyicu.research_agent.runner import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.summary_repair import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    run_result = RunResult(
        step_id="01_assoc",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout='noise\n{"primary_or": 1.4, "sample_size": 500}\n',
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(
        step_id="01_assoc", intent="assoc", expected_outputs=["statistic:primary_or"]
    )

    outcome = salvage_step_summary(run_result, step=step)
    assert outcome is not None
    assert outcome.repair_id == "summary_salvage_stdout_json_v1"
    assert outcome.reset_artefacts is True
    assert (
        repair_metadata_for(outcome.repair_id).repair_class is RepairClass.STRUCTURAL
    )
    # Salvage actually wrote the summary the registration step will read.
    assert (out_dir / "step_summary.json").exists()

    # Feeding the outcome to the ledger (what the execute-phase closure does)
    # produces a real provenance entry — closing the wiring gap.
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    ledger.append_application(
        repair_id=outcome.repair_id,
        step_id=step.step_id,
        trigger={"source": "summary_salvage", "reason": outcome.trigger_reason},
        transformation=outcome.transformation,
        selection_rule=outcome.selection_rule,
    )
    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    assert payload["repairs"][0]["repair_id"] == "summary_salvage_stdout_json_v1"
    assert payload["repairs"][0]["repair_class"] == RepairClass.STRUCTURAL.value


def test_salvage_step_summary_reports_minimal_contract(tmp_path: Path) -> None:
    from easyicu.research_agent.runner import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.summary_repair import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text("{}", encoding="utf-8")
    (out_dir / "table_one.csv").write_text(
        "variable,median\nage,65.0\nsofa2,7.0\n", encoding="utf-8"
    )
    run_result = RunResult(
        step_id="01_table_one",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(
        step_id="01_table_one", intent="t1", expected_outputs=["table:table_one"]
    )

    outcome = salvage_step_summary(run_result, step=step)
    assert outcome is not None
    assert outcome.repair_id == "summary_salvage_minimal_contract_v1"
    assert (
        repair_metadata_for(outcome.repair_id).repair_class is RepairClass.CONTRACT_FILL
    )
    assert outcome.selection_rule  # CONTRACT_FILL must record how it selected
    assert outcome.reset_artefacts is False


def test_salvage_step_summary_returns_none_when_summary_present(tmp_path: Path) -> None:
    from easyicu.research_agent.runner import RunResult
    from easyicu.research_agent.schema import AnalysisStep
    from easyicu.research_agent.summary_repair import salvage_step_summary

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "step_summary.json").write_text('{"primary_or": 1.2}', encoding="utf-8")
    run_result = RunResult(
        step_id="01_assoc",
        script_path=tmp_path / "analysis.py",
        cwd=tmp_path,
        out_dir=out_dir,
        stdout="",
        stderr="",
        returncode=0,
        duration_seconds=0.1,
    )
    step = AnalysisStep(step_id="01_assoc", intent="assoc", expected_outputs=[])
    assert salvage_step_summary(run_result, step=step) is None
