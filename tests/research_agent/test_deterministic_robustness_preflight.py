from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.authority.plausibility import (
    FlagOnlyPlausibilityScope,
)
from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    cohort_definition_sha,
)
from easyicu.research_agent.execution.runners import (
    deterministic_robustness as robustness_module,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_sensitivity_preflight_code,
)
from easyicu.research_agent.gates.plausibility_obligation import (
    flag_only_plausibility_obligation_findings,
)
from easyicu.research_agent.gates.plausibility_receipt import (
    plausibility_audit_receipt_findings,
)
from easyicu.research_agent.robustness.panel import (
    RobustnessSpec,
    robustness_specs_sha,
)
from easyicu.research_agent.schema import AnalysisStep


def _predicate(concept_id: str, op: str, value: object) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id=concept_id,
        time_window=TimeWindow(
            anchor="icu_admit",
            start_offset_hours=0.0,
            end_offset_hours=720.0,
        ),
        aggregation="first",
        op=op,
        value=value,
    )


def _specs() -> list[RobustnessSpec]:
    return [
        RobustnessSpec(
            spec_id="adult_any_los",
            axis="cohort",
            description="Adults without a length-of-stay threshold.",
            cohort_override=CohortDefinition(
                name="adult_any_los",
                inclusion=[_predicate("age", ">=", 18)],
            ),
        ),
        RobustnessSpec(
            spec_id="adult_los_half_day",
            axis="cohort",
            description="Adults with at least half a day of ICU stay.",
            cohort_override=CohortDefinition(
                name="adult_los_half_day",
                inclusion=[
                    _predicate("age", ">=", 18),
                    _predicate("los_icu", ">=", 0.5),
                ],
            ),
        ),
        RobustnessSpec(
            spec_id="older_adults",
            axis="cohort",
            description="Restrict to an alternative adult age threshold.",
            cohort_override=CohortDefinition(
                name="older_adults",
                inclusion=[
                    _predicate("age", ">=", 40),
                ],
            ),
        ),
        RobustnessSpec(
            spec_id="missing_complete_case",
            axis="missing",
            description="Complete-case analysis.",
            missing_override={"strategy": "complete_case"},
        ),
        RobustnessSpec(
            spec_id="missing_median",
            axis="missing",
            description="Median imputation.",
            missing_override={"strategy": "median_imputation"},
        ),
        RobustnessSpec(
            spec_id="outcome_first",
            axis="outcome",
            description="First recorded value of the supplied outcome label.",
            outcome_override={
                "concept_id": "outcome",
                "aggregation": "first",
            },
        ),
        RobustnessSpec(
            spec_id="outcome_any",
            axis="outcome",
            description="Any recorded event in the requested window.",
            outcome_override={
                "concept_id": "outcome",
                "aggregation": "any",
            },
        ),
    ]


def _prepare_run(tmp_path: Path, *, include_primary: bool) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "08_robustness" / "outputs"
    out_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n = 360
    exposure = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-(-0.5 + 0.7 * exposure)))
    outcome = (rng.random(n) < probability).astype(int)
    universe = pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "age": rng.integers(18, 90, size=n),
            "los_icu": rng.uniform(0.1, 3.0, size=n),
            "exposure": exposure,
            "outcome": outcome,
        }
    )
    universe.loc[::13, "exposure"] = np.nan
    primary_mask = (universe["age"] >= 18) & (universe["los_icu"] >= 1)
    cohort = universe.loc[primary_mask].copy()
    universe_path = tmp_path / "universe.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    universe.to_parquet(universe_path, index=False)
    cohort.to_parquet(cohort_path, index=False)

    context = {
        "research_question": "Does the declared exposure predict the outcome?",
        "primary_exposure": "exposure",
        "target_outcome": "outcome",
        "variables": [],
    }
    (run_dir / "research_context.json").write_text(json.dumps(context))

    specs = _specs()
    lock = {
        "schema_version": "easyicu.robustness_specs/1",
        "locked_at": "2026-07-10T00:00:00+00:00",
        "spec_sha256": robustness_specs_sha(specs),
        "specs": [spec.to_dict() for spec in specs],
    }
    (run_dir / "robustness_specs_locked.json").write_text(json.dumps(lock))

    primary_cohort = CohortDefinition(
        name="adult_los_one_day",
        inclusion=[
            _predicate("age", ">=", 18),
            _predicate("los_icu", ">=", 1),
        ],
    )
    cohort_lock = {
        "schema_version": "easyicu.cohort_definition/1",
        "locked_at": "2026-07-10T00:00:00+00:00",
        "cohort_sha256": cohort_definition_sha(primary_cohort),
        "cohort": primary_cohort.to_dict(),
    }
    (run_dir / "cohort_locked.json").write_text(json.dumps(cohort_lock))

    records = []
    if include_primary:
        records.append(
            {
                "step_id": "07_primary_model",
                "status": "ok",
                "planned_analysis_role": "primary",
                "analysis_request": {
                    "step": {
                        "step_id": "07_primary_model",
                        "planned_analysis_role": "primary",
                    }
                },
                "step_summary_evidence_id": "stat_primary",
                "step_summary": {
                    "primary_predictor": "exposure",
                    "primary_or": 1.8,
                    "primary_ci_low": 1.3,
                    "primary_ci_high": 2.5,
                    "n_total": int(len(cohort)),
                },
            }
        )
    manifest = {
        "run_id": "test_run",
        "checkpoint_sequence": 1,
        "per_step_records": records,
    }
    (run_dir / "manifest_partial.json").write_text(json.dumps(manifest))
    return out_dir, cohort_path, universe_path


def _run_generated(
    monkeypatch,
    *,
    out_dir: Path,
    cohort_path: Path,
    universe_path: Path,
    tamper_snapshot: bool = False,
) -> None:
    from easyicu.research_agent.execution.runner import (
        _capture_run_artifact_authority_snapshot,
    )

    run_dir = out_dir.parents[2]
    snapshot_path, snapshot_sha256, authority_error = (
        _capture_run_artifact_authority_snapshot(
            workdir=run_dir,
            step_dir=out_dir.parent,
        )
    )
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("EASYICU_UNIVERSE_PARQUET", str(universe_path))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv(
        "EASYICU_MANIFEST_PARTIAL", str(run_dir / "manifest_partial.json")
    )
    if snapshot_path is not None and snapshot_sha256:
        if tamper_snapshot:
            snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")
        monkeypatch.setenv(
            "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT",
            str(snapshot_path),
        )
        monkeypatch.setenv(
            "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256",
            snapshot_sha256,
        )
        monkeypatch.delenv("EASYICU_RUN_ARTIFACT_AUTHORITY_ERROR", raising=False)
    else:
        monkeypatch.delenv("EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT", raising=False)
        monkeypatch.delenv(
            "EASYICU_RUN_ARTIFACT_AUTHORITY_SNAPSHOT_SHA256", raising=False
        )
        monkeypatch.setenv(
            "EASYICU_RUN_ARTIFACT_AUTHORITY_ERROR",
            authority_error or "authority unavailable",
        )
    code = robustness_sensitivity_preflight_code()
    exec(compile(code, "<robustness-preflight>", "exec"), {})


def test_preflight_binds_plausibility_receipt_to_exact_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    step = AnalysisStep(
        step_id="06_robustness_sensitivity",
        planned_analysis_role="sensitivity",
        intent="Replay the locked sensitivity analysis.",
        inputs=["artifact:analysis_cohort", "age"],
        expected_outputs=["table:robustness_matrix"],
        method="robustness_sensitivity",
    )
    raw_contracts: dict[str, object] = {
        "schema_version": "easyicu.resolved_raw_input_contracts/1",
        "authority_scope": (
            "host_verified_physical_representation_and_domain_constraints"
        ),
        "scientific_ownership": "Planner retains scientific decisions",
        "contracts": {
            "age": {
                "column": "age",
                "analysis_plausibility_range": {
                    "minimum": 0.0,
                    "maximum": 100.0,
                },
                "plausibility_policy": {
                    "range_policy": "flag_only",
                    "out_of_range_action": "retain_and_flag",
                },
            }
        },
    }
    contracts_sha256 = hashlib.sha256(
        json.dumps(
            raw_contracts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    raw_contracts["contracts_sha256"] = contracts_sha256
    scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=("age",),
        source_contracts_sha256=contracts_sha256,
        authority_kind="resolved_raw_input_contracts",
    )

    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / step.step_id / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    universe_path = run_dir / "universe.parquet"
    pd.DataFrame({"age": [20.0, 50.0]}).to_parquet(cohort_path, index=False)
    pd.DataFrame({"age": [-1.0, 50.0, 101.0]}).to_parquet(
        universe_path,
        index=False,
    )
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(
        json.dumps(
            {
                "step_id": step.step_id,
                "raw_input_contracts": raw_contracts,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("EASYICU_UNIVERSE_PARQUET", str(universe_path))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))

    def _write_robustness_summary() -> None:
        (out_dir / "step_summary.json").write_text(
            json.dumps({"status": "ok"}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        robustness_module,
        "_run_robustness_preflight_from_env",
        _write_robustness_summary,
    )
    code = robustness_sensitivity_preflight_code(
        step,
        plausibility_scope=scope,
    )
    assert (
        flag_only_plausibility_obligation_findings(
            None,
            script_text=code,
            step=step,
            scope=scope,
        )
        == []
    )

    exec(compile(code, "<robustness-preflight>", "exec"), {})

    summary = json.loads((out_dir / "step_summary.json").read_text("utf-8"))
    receipt = summary["plausibility_audit"]["age"]
    assert set(summary["plausibility_audit"]) == {"age"}
    assert receipt["policy"] == "retain_and_flag"
    assert receipt["below_minimum_n"] == 1
    assert receipt["above_maximum_n"] == 1
    assert receipt["out_of_range_n"] == 2
    # The denominator is what separates "2 of many were out of range" from a
    # column that was never observed and reports 0 for both.
    assert receipt["compared_n"] >= receipt["out_of_range_n"]
    assert receipt["compared_n"] > 0
    assert (
        plausibility_audit_receipt_findings(
            step_summary=summary,
            step=step,
            script_text=code,
            scope=scope,
        )
        == []
    )


def test_preflight_scope_requires_its_exact_step() -> None:
    step = AnalysisStep(
        step_id="06_robustness_sensitivity",
        intent="Replay robustness.",
        method="robustness_sensitivity",
    )
    scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=("age",),
        source_contracts_sha256="0" * 64,
        authority_kind="resolved_raw_input_contracts",
    )

    with pytest.raises(ValueError, match="requires an exact analysis step"):
        robustness_sensitivity_preflight_code(plausibility_scope=scope)


def test_preflight_emits_renderer_contract_and_nonindependent_scalar_outcomes(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    matrix = pd.read_csv(out_dir / "robustness_matrix.csv")
    assert {
        "spec_id",
        "effect_scale",
        "point_estimate",
        "ci_low",
        "ci_high",
        "modeled_analytic_n",
        "axis",
        "converged",
        "membership_n",
        "outcome_executable",
        "independent_variant",
        "notes",
    } <= set(matrix.columns)
    primary = matrix.loc[matrix["spec_id"] == "primary"].iloc[0]
    assert primary["point_estimate"] == 1.8
    assert primary["ci_low"] == 1.3
    assert primary["ci_high"] == 2.5

    outcome_rows = matrix[matrix["axis"] == "outcome"]
    assert len(outcome_rows) == 2
    assert not outcome_rows["independent_variant"].astype(bool).any()
    assert not outcome_rows["converged"].astype(bool).any()
    assert outcome_rows["point_estimate"].isna().all()
    assert outcome_rows["notes"].str.contains("not independently executable").all()

    outcome_audit = pd.read_csv(out_dir / "outcome_label_executability.csv")
    assert not outcome_audit["event_timing_available"].astype(bool).any()
    assert not outcome_audit["independent_variant"].astype(bool).any()

    membership = pd.read_csv(out_dir / "membership_change_summary.csv")
    cohort_rows = membership[membership["axis"] == "cohort"]
    assert set(cohort_rows["membership_source"]) == {"universe"}
    assert cohort_rows["membership_executable"].astype(bool).all()
    assert cohort_rows["variant_membership_n"].notna().all()
    assert (
        cohort_rows["overlap_n"]
        == cohort_rows["primary_membership_n"] - cohort_rows["outflow_n"]
    ).all()
    assert (
        cohort_rows["overlap_n"]
        == cohort_rows["variant_membership_n"] - cohort_rows["inflow_n"]
    ).all()

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "did not emit verifiable estimates" in summary["blocking_reason"]
    assert summary["aliases"]["sensitivity_comparison"] == "sensitivity_comparison.csv"
    assert (
        summary["aliases"]["cohort_overlap_and_attrition"]
        == "cohort_overlap_and_attrition.csv"
    )
    assert (
        summary["aliases"]["sensitivity_specification_grid"]
        == "sensitivity_specification_grid.csv"
    )
    assert (
        summary["aliases"]["cohort_definition_overlap_attrition"]
        == "cohort_definition_overlap_attrition.csv"
    )
    assert (
        summary["aliases"]["sensitivity_specification_matrix"]
        == "sensitivity_specification_matrix.csv"
    )
    assert summary["aliases"]["primary_or"] == "primary_or.json"
    assert summary["aliases"]["complete_case_n"] == "complete_case_n.json"
    assert summary["robustness_panel"]["rows"] == summary["robustness_rows"]
    # A locked sensitivity label is not enough to let the auxiliary runner
    # invent the primary estimator, covariates, or refit policy.  Without an
    # explicit estimator_adapter (or exact registered model replay), retain
    # the validated primary row and leave the variant unreported.
    assert summary["complete_case_n"] is None
    assert any(
        "generic deterministic robustness refitting is disabled" in warning
        for warning in summary["warnings"]
    )
    assert (out_dir / "cohort_overlap_and_attrition.csv").exists()
    assert (out_dir / "sensitivity_specification_grid.csv").exists()
    assert (out_dir / "missingness_strategy_notes.txt").exists()
    assert (out_dir / "sensitivity_comparison.csv").read_bytes() == (
        out_dir / "robustness_matrix.csv"
    ).read_bytes()
    assert (out_dir / "cohort_definition_overlap_attrition.csv").read_bytes() == (
        out_dir / "membership_change_summary.csv"
    ).read_bytes()
    assert (out_dir / "sensitivity_specification_matrix.csv").read_bytes() == (
        out_dir / "sensitivity_specification_grid.csv"
    ).read_bytes()
    assert json.loads((out_dir / "primary_or.json").read_text())["value"] == 1.8
    assert json.loads((out_dir / "complete_case_n.json").read_text())["value"] is None


def test_preflight_fails_closed_without_completed_primary_estimate(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=False)

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "completed primary estimate" in summary["blocking_reason"]
    matrix = pd.read_csv(out_dir / "robustness_matrix.csv")
    assert list(matrix.columns[:8]) == [
        "spec_id",
        "effect_scale",
        "point_estimate",
        "ci_low",
        "ci_high",
        "modeled_analytic_n",
        "axis",
        "converged",
    ]
    assert not matrix["converged"].astype(bool).any()


def test_preflight_does_not_fall_back_when_newest_checkpoint_is_corrupt(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)
    run_dir = out_dir.parents[2]
    (run_dir / "manifest_partial.json").replace(run_dir / "manifest.json")
    (run_dir / "manifest_partial.json").write_text("{corrupt", encoding="utf-8")

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "refusing to fall back" in summary["blocking_reason"]
    assert json.loads((out_dir / "primary_or.json").read_text())["value"] is None


def test_preflight_latest_failed_primary_supersedes_older_success(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)
    manifest_path = out_dir.parents[2] / "manifest_partial.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_sequence"] = 2
    manifest["per_step_records"].append(
        {
            "step_id": "07_primary_model",
            "status": "contract_failed",
            "step_summary": {"status": "rejected"},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "completed primary estimate" in summary["blocking_reason"]
    assert json.loads((out_dir / "primary_or.json").read_text())["value"] is None


def test_preflight_rejects_tampered_host_authority_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
        tamper_snapshot=True,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "snapshot digest mismatch" in summary["blocking_reason"]
    assert json.loads((out_dir / "primary_or.json").read_text())["value"] is None


def test_preflight_fails_closed_without_locked_specs(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)
    (out_dir.parents[2] / "robustness_specs_locked.json").unlink()

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "Locked robustness specifications unavailable" in summary["blocking_reason"]


def test_preflight_blocks_locked_variants_for_non_or_primary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)
    manifest_path = out_dir.parents[2] / "manifest_partial.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary = manifest["per_step_records"][0]["step_summary"]
    primary.pop("primary_or")
    primary.pop("primary_ci_low")
    primary.pop("primary_ci_high")
    primary.update(
        {
            "hazard_ratio": 1.4,
            "hazard_ratio_ci_low": 1.1,
            "hazard_ratio_ci_high": 1.8,
            "n_analysis": primary["n_total"],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "primary effect scale HR" in summary["blocking_reason"]


def test_preflight_blocks_unsupported_locked_missing_strategy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=True)
    run_dir = out_dir.parents[2]
    lock_path = run_dir / "robustness_specs_locked.json"
    specs = _specs()
    median_index = next(
        index for index, spec in enumerate(specs) if spec.spec_id == "missing_median"
    )
    specs[median_index] = RobustnessSpec(
        spec_id="missing_unimplemented",
        axis="missing",
        description="A locked strategy not implemented by the runner.",
        missing_override={"strategy": "multiple_imputation"},
    )
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "easyicu.robustness_specs/1",
                "locked_at": "2026-07-10T00:00:00+00:00",
                "spec_sha256": robustness_specs_sha(specs),
                "specs": [spec.to_dict() for spec in specs],
            }
        ),
        encoding="utf-8",
    )

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "blocked"
    assert "missing_unimplemented" in summary["blocking_reason"]
    assert (
        "not executable under the registered analysis contract"
        in summary["blocking_reason"]
    )


def test_structured_preflight_replays_exact_primary_code_and_emits_spec_by_model_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(tmp_path, include_primary=False)
    run_dir = out_dir.parents[2]
    source_step = run_dir / "steps" / "07_primary_model"
    source_outputs = source_step / "outputs"
    source_outputs.mkdir(parents=True)
    source_script = source_step / "analysis.py"
    source_script.write_text(
        textwrap.dedent(
            """
            import json, math, os
            from pathlib import Path
            import pandas as pd

            data = pd.read_parquet(os.environ["COHORT_PARQUET"])
            out = Path(os.environ["STEP_OUT_DIR"])
            out.mkdir(parents=True, exist_ok=True)
            n_full = int(len(data))
            event_full = int(data["outcome"].sum())
            complete = data[["exposure", "outcome"]].dropna()
            n_cc = int(len(complete))
            event_cc = int(complete["outcome"].sum())
            primary_or = 1.5 + n_full / 10000.0
            cc_or = 1.2 + n_cc / 10000.0

            def contract(model_id, source, expression, exposure_role,
                         analysis_role, analysis_set, n, event_n):
                return {
                    "model_id": model_id,
                    "exposure_source": source,
                    "exposure_expression": expression,
                    "exposure_role": exposure_role,
                    "analysis_role": analysis_role,
                    "analysis_set": analysis_set,
                    "baseline_missing_policy": (
                        "explicit_missing_category" if analysis_set == "source_aware"
                        else "drop_missing_baseline"
                    ),
                    "n": n,
                    "event_n": event_n,
                    "fit_status": "fitted",
                    "converged": True,
                    "separation_detected": False,
                    "penalized": False,
                    "fit_method": "registered_test_model",
                }

            contracts = [
                contract("primary_full", "exposure", "log1p(exposure)",
                         "primary", "primary", "source_aware", n_full, event_full),
                contract("secondary_full", "secondary", "secondary",
                         "secondary", "secondary", "source_aware", n_full, event_full),
                contract("primary_cc", "exposure", "log1p(exposure)",
                         "primary", "sensitivity", "complete_case", n_cc, event_cc),
                contract("secondary_cc", "secondary", "secondary",
                         "secondary", "sensitivity", "complete_case", n_cc, event_cc),
            ]
            coefficient_specs = [
                ("primary_full", "exposure_log1p", "exposure", primary_or,
                 "primary", "source_aware"),
                ("secondary_full", "secondary_level", "secondary", 1.1,
                 "secondary", "source_aware"),
                ("primary_cc", "exposure_log1p", "exposure", cc_or,
                 "sensitivity", "complete_case"),
                ("secondary_cc", "secondary_level", "secondary", 1.05,
                 "sensitivity", "complete_case"),
            ]
            rows = []
            for model_id, term, source, odds_ratio, role, analysis_set in coefficient_specs:
                rows.append({
                    "model_id": model_id,
                    "term": term,
                    "term_role": "exposure",
                    "source_variable": source,
                    "estimate": math.log(odds_ratio),
                    "odds_ratio": odds_ratio,
                    "ci_low": odds_ratio - 0.1,
                    "ci_high": odds_ratio + 0.1,
                    "std_error": 0.02,
                    "analysis_role": role,
                    "analysis_set": analysis_set,
                })
            pd.DataFrame(rows).to_csv(out / "coefficients.csv", index=False)
            pd.DataFrame(contracts).to_csv(out / "model_summaries.csv", index=False)
            summary = {
                "status": "ok",
                "primary_model_id": "primary_full",
                "primary_exposure": "exposure",
                "primary_or": primary_or,
                "primary_ci_low": primary_or - 0.1,
                "primary_ci_high": primary_or + 0.1,
                "primary_model_n": n_full,
                "model_contracts": contracts,
                # Named the way a producer names it. This fixture used to
                # publish the companion only under `output_files.coefficients`
                # -- a key the reader never consulted -- and passed anyway,
                # because the reader assumed a fixed `coefficients.csv` when no
                # companion was declared. No recorded run has ever written a
                # file by that name, so the assumption resolved to nothing in
                # production while this test stayed green.
                "coefficient_table": "coefficients.csv",
                "output_files": {
                    "coefficients": "coefficients.csv",
                    "model_summaries": "model_summaries.csv",
                },
            }
            (out / "step_summary.json").write_text(json.dumps(summary))
            """
        ),
        encoding="utf-8",
    )
    base_env = os.environ.copy()
    base_env["COHORT_PARQUET"] = str(cohort_path)
    base_env["STEP_OUT_DIR"] = str(source_outputs)
    subprocess.run(
        [sys.executable, str(source_script)],
        env=base_env,
        check=True,
    )
    source_summary = json.loads((source_outputs / "step_summary.json").read_text())
    script_sha = hashlib.sha256(source_script.read_bytes()).hexdigest()
    coefficient_sha = hashlib.sha256(
        (source_outputs / "coefficients.csv").read_bytes()
    ).hexdigest()
    summary_sha = hashlib.sha256(
        (source_outputs / "step_summary.json").read_bytes()
    ).hexdigest()
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir()
    code_evidence_path = evidence_dir / "code_primary_analysis__analysis.py"
    coefficient_evidence_path = (
        evidence_dir / "table_primary_coefficients__coefficients.csv"
    )
    summary_evidence_path = evidence_dir / "structured_primary__step_summary.json"
    code_evidence_path.write_bytes(source_script.read_bytes())
    coefficient_evidence_path.write_bytes(
        (source_outputs / "coefficients.csv").read_bytes()
    )
    summary_evidence_path.write_bytes(
        (source_outputs / "step_summary.json").read_bytes()
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "run_id": "structured_replay",
                "checkpoint_sequence": 1,
                "per_step_records": [
                    {
                        "step_id": "07_primary_model",
                        "status": "ok",
                        "planned_analysis_role": "primary",
                        "analysis_request": {
                            "step": {
                                "step_id": "07_primary_model",
                                "planned_analysis_role": "primary",
                            }
                        },
                        "executed_code_sha256": script_sha,
                        "evidence_ids": [
                            "code_primary_analysis",
                            "table_primary_coefficients",
                            "structured_primary",
                        ],
                        "step_summary_evidence_id": "structured_primary",
                        "step_summary": source_summary,
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "code_primary_analysis",
                        "kind": "code",
                        "relative_path": str(code_evidence_path.relative_to(run_dir)),
                        "sha256": script_sha,
                        "produced_by_step": "07_primary_model",
                    },
                    {
                        "evidence_id": "table_primary_coefficients",
                        "kind": "table",
                        "relative_path": str(
                            coefficient_evidence_path.relative_to(run_dir)
                        ),
                        "sha256": coefficient_sha,
                        "produced_by_step": "07_primary_model",
                        "script_evidence_id": "code_primary_analysis",
                    },
                    {
                        "evidence_id": "structured_primary",
                        "kind": "statistic",
                        "relative_path": str(
                            summary_evidence_path.relative_to(run_dir)
                        ),
                        "sha256": summary_sha,
                        "produced_by_step": "07_primary_model",
                        "script_evidence_id": "code_primary_analysis",
                    },
                ],
            }
        )
    )
    specs = [spec for spec in _specs() if spec.spec_id != "missing_median"]
    specs.append(
        RobustnessSpec(
            spec_id="missing_source_aware",
            axis="missing",
            description="Reuse the registered source-aware model.",
            missing_override={"strategy": "source_aware_categories_no_imputation"},
        )
    )
    (run_dir / "robustness_specs_locked.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.robustness_specs/1",
                "locked_at": "2026-07-10T00:00:00+00:00",
                "spec_sha256": robustness_specs_sha(specs),
                "specs": [spec.to_dict() for spec in specs],
            }
        )
    )

    _run_generated(
        monkeypatch,
        out_dir=out_dir,
        cohort_path=cohort_path,
        universe_path=universe_path,
    )

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["primary_model_replay"]["mode"] == (
        "exact_registered_primary_model_code"
    )
    assert summary["model_contracts"][0]["exposure_expression"] == ("log1p(exposure)")
    assert summary["complete_case_n"] == source_summary["model_contracts"][2]["n"]
    contracts = summary["robustness_model_contracts"]
    assert len(contracts) == 16
    assert {(contract["spec_id"], contract["model_id"]) for contract in contracts} == {
        (spec_id, model_id)
        for spec_id in (
            "adult_any_los",
            "adult_los_half_day",
            "older_adults",
        )
        for model_id in (
            "primary_full",
            "secondary_full",
            "primary_cc",
            "secondary_cc",
        )
    } | {
        (spec_id, model_id)
        for spec_id in ("missing_complete_case", "missing_source_aware")
        for model_id in (
            ("primary_cc", "secondary_cc")
            if spec_id == "missing_complete_case"
            else ("primary_full", "secondary_full")
        )
    }
    matrix = pd.read_csv(out_dir / "robustness_matrix.csv")
    assert {
        "model_id",
        "source_model_id",
        "exposure_expression",
        "model_contract_n",
        "event_n",
        "coefficient_source_table",
        "coefficient_term",
        "model_contract_source",
        "source_script_sha256",
        "estimability_status",
    } <= set(matrix.columns)
    cohort_rows = matrix[matrix["axis"] == "cohort"]
    assert cohort_rows["converged"].astype(bool).all()
    assert (
        cohort_rows["modeled_analytic_n"].astype(int)
        == cohort_rows["membership_n"].astype(int)
    ).all()
    assert cohort_rows["model_id"].eq("primary_full").all()
    assert cohort_rows["coefficient_term"].eq("exposure_log1p").all()
    assert (
        cohort_rows["model_contract_n"].astype(int)
        == cohort_rows["modeled_analytic_n"].astype(int)
    ).all()
    assert cohort_rows["event_n"].notna().all()
    missing_rows = matrix[matrix["axis"] == "missing"].set_index("spec_id")
    assert missing_rows.loc["missing_complete_case", "model_id"] == "primary_cc"
    assert missing_rows.loc["missing_source_aware", "model_id"] == "primary_full"
    outcome_rows = matrix[matrix["axis"] == "outcome"]
    assert outcome_rows["modeled_analytic_n"].isna().all()
    assert outcome_rows["event_n"].isna().all()
    assert outcome_rows["estimability_status"].eq("not_independent").all()
    coefficients = pd.read_csv(out_dir / "robustness_variant_coefficients.csv")
    assert coefficients[["spec_id", "model_id"]].drop_duplicates().shape[0] == 16
    replay_index = json.loads((out_dir / "model_replay_index.json").read_text())
    assert (
        replay_index["source_script_sha256"]
        == summary["primary_model_replay"]["source_script_sha256"]
    )
    assert all(item["status"] == "ok" for item in replay_index["variants"])


def _write_structured_source_authority(
    run_dir: Path,
    *,
    coefficient_filename: str = "coefficients.csv",
    diagnostic_reference: object | None = None,
    include_primary_model_id: bool = True,
    include_headline: bool = True,
    model_n: int = 100,
    analysis_covariates: list[str] | None = None,
):
    step_id = "01_primary_model"
    step_dir = run_dir / "steps" / step_id
    outputs_dir = step_dir / "outputs"
    evidence_dir = run_dir / "evidence"
    outputs_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    script_path = step_dir / "analysis.py"
    coefficient_path = outputs_dir / coefficient_filename
    script_path.write_text("print('registered primary model')\n", encoding="utf-8")
    coefficient_path.write_text(
        "model_id,term,term_role,source_variable,odds_ratio,ci_low,ci_high,std_error\n"
        "primary,exposure,exposure,exposure,1.4,1.1,1.8,0.1\n",
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "primary_exposure": "exposure",
        "primary_model_n": model_n,
        "model_contracts": [
            {
                "model_id": "primary",
                "analysis_role": "primary",
                "exposure_role": "primary",
                "exposure_source": "exposure",
                "exposure_expression": "exposure",
                "analysis_set": "source_aware",
                "n": model_n,
                "event_n": min(20, model_n),
                "fit_status": "fitted",
                "converged": True,
                "fit_method": "registered_test_model",
            }
        ],
    }
    if analysis_covariates is not None:
        summary["analysis_definition"] = {
            "exposure": "exposure",
            "outcome": "outcome",
            "covariates": list(analysis_covariates),
        }
    if include_headline:
        summary.update(
            {
                "primary_or": 1.4,
                "primary_ci_low": 1.1,
                "primary_ci_high": 1.8,
            }
        )
    if include_primary_model_id:
        summary["primary_model_id"] = "primary"
    if diagnostic_reference is not None:
        summary["diagnostic_companions"] = {
            "coefficients": diagnostic_reference,
        }
    else:
        summary["diagnostic_companions"] = {
            "coefficients": coefficient_filename,
        }
    # A summary that names its companion badly is refused outright, so this
    # sibling never rescues the malformed case above -- it is here so that the
    # ordinary fixture declares the file the way a producer does, instead of
    # relying on a default filename no run has ever written.
    summary["coefficient_table"] = coefficient_filename
    summary_path = outputs_dir / "step_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    coefficient_sha = hashlib.sha256(coefficient_path.read_bytes()).hexdigest()
    summary_sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    code_copy = evidence_dir / "code_primary__analysis.py"
    coefficient_copy = evidence_dir / f"table_coefficients__{coefficient_filename}"
    summary_copy = evidence_dir / "stat_primary__step_summary.json"
    code_copy.write_bytes(script_path.read_bytes())
    coefficient_copy.write_bytes(coefficient_path.read_bytes())
    summary_copy.write_bytes(summary_path.read_bytes())
    evidence = [
        {
            "evidence_id": "code_primary",
            "kind": "code",
            "relative_path": str(code_copy.relative_to(run_dir)),
            "sha256": script_sha,
            "produced_by_step": step_id,
        },
        {
            "evidence_id": "table_coefficients",
            "kind": "table",
            "relative_path": str(coefficient_copy.relative_to(run_dir)),
            "sha256": coefficient_sha,
            "produced_by_step": step_id,
            "script_evidence_id": "code_primary",
        },
        {
            "evidence_id": "stat_primary",
            "kind": "statistic",
            "relative_path": str(summary_copy.relative_to(run_dir)),
            "sha256": summary_sha,
            "produced_by_step": step_id,
            "script_evidence_id": "code_primary",
        },
    ]
    record = {
        "step_id": step_id,
        "status": "ok",
        "planned_analysis_role": "primary",
        "analysis_request": {
            "step": {
                "step_id": step_id,
                "planned_analysis_role": "primary",
            }
        },
        "executed_code_sha256": script_sha,
        "evidence_ids": ["code_primary", "table_coefficients", "stat_primary"],
        "step_summary_evidence_id": "stat_primary",
        "step_summary": summary,
    }
    return record, evidence, script_path


def test_structured_source_uses_latest_step_record_and_registered_code_sha(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, script_path = _write_structured_source_authority(run_dir)

    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is not None
    )

    wrong_kind_evidence = [dict(item) for item in evidence]
    wrong_kind_evidence[1]["kind"] = "log"
    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=wrong_kind_evidence,
        )
        is None
    )

    failed_retry = {
        "step_id": record["step_id"],
        "status": "contract_failed",
        "step_summary": {"status": "rejected"},
    }
    assert (
        _find_structured_primary_model_source(
            records=[record, failed_retry],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )

    script_path.write_text("print('mutated after execution')\n", encoding="utf-8")
    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )


def test_structured_source_uses_sealed_diagnostic_coefficient_companion(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(
        run_dir,
        coefficient_filename="adjusted_association_coefficients.csv",
        include_primary_model_id=False,
    )

    source = _find_structured_primary_model_source(
        records=[record],
        run_dir=run_dir,
        evidence_records=evidence,
    )

    assert source is not None
    assert source["coefficient_path"].name == "adjusted_association_coefficients.csv"
    assert source["primary_contract"]["model_id"] == "primary"


def test_structured_primary_allows_table_only_headline_authority(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
        _structured_primary_effect_payload,
    )
    from easyicu.research_agent.robustness.primary_effect import (
        _extract_primary_effect_payload_from_records,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(
        run_dir,
        coefficient_filename="adjusted_association_coefficients.csv",
        include_primary_model_id=False,
        include_headline=False,
    )
    source = _find_structured_primary_model_source(
        records=[record],
        run_dir=run_dir,
        evidence_records=evidence,
    )
    assert source is not None
    reported = _extract_primary_effect_payload_from_records(
        [record],
        preferred_predictor="exposure",
    )

    payload, errors = _structured_primary_effect_payload(
        source=source,
        reported_payload=reported,
        preferred_predictor="exposure",
    )

    assert errors == []
    assert payload is not None
    assert payload["primary_or"] == 1.4
    assert payload["sample_size"] == 100


def test_complete_case_equivalence_requires_identical_locked_membership(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
        _verified_complete_case_equivalence,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(
        run_dir,
        coefficient_filename="adjusted_association_coefficients.csv",
        include_primary_model_id=False,
        include_headline=False,
        model_n=4,
        analysis_covariates=["age"],
    )
    source = _find_structured_primary_model_source(
        records=[record],
        run_dir=run_dir,
        evidence_records=evidence,
    )
    assert source is not None
    spec = RobustnessSpec(
        spec_id="complete_case",
        axis="missing",
        description="Use complete cases for every primary model input.",
        missing_override={
            "strategy": "complete_case",
            "variables": ["exposure", "outcome", "age"],
        },
    )
    complete = pd.DataFrame(
        {
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
            "age": [50.0, 60.0, 70.0, 80.0],
        }
    )

    row, coefficient_rows, contract, error = _verified_complete_case_equivalence(
        spec=spec,
        source=source,
        primary_data=complete,
    )

    assert error is None
    assert row.converged is True
    assert row.n == 4
    assert contract is not None
    assert contract["analysis_set"] == "complete_case"
    assert contract["source_analysis_set"] == "source_aware"
    assert contract["replay_mode"] == "verified_complete_case_equivalence"
    assert coefficient_rows[0]["analysis_set"] == "complete_case"

    incomplete = complete.copy()
    incomplete.loc[0, "age"] = None
    blocked, _rows, _contract, error = _verified_complete_case_equivalence(
        spec=spec,
        source=source,
        primary_data=incomplete,
    )
    assert blocked.converged is False
    assert "complete_case_n=3, model_n=4" in str(error)


@pytest.mark.parametrize(
    "diagnostic_reference",
    [
        "../coefficients.csv",
        "/tmp/coefficients.csv",
        "coefficients.json",
        17,
    ],
)
def test_structured_source_rejects_malformed_diagnostic_companion_path(
    tmp_path: Path,
    diagnostic_reference: object,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(
        run_dir,
        diagnostic_reference=diagnostic_reference,
    )

    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )


def test_structured_source_rejects_symlinked_analysis_script(tmp_path: Path) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, script_path = _write_structured_source_authority(run_dir)
    registered_copy = run_dir / evidence[0]["relative_path"]
    script_path.unlink()
    script_path.symlink_to(registered_copy)

    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )


def test_structured_source_rejects_sensitivity_role_self_claim(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(run_dir)
    record["planned_analysis_role"] = "sensitivity"
    record["analysis_request"]["step"]["planned_analysis_role"] = "sensitivity"

    assert (
        _find_structured_primary_model_source(
            records=[record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )


def test_structured_source_rejects_duplicate_verified_primary_records(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(run_dir)
    decoy = {
        "step_id": "02_other_primary",
        "status": "ok",
        "planned_analysis_role": "primary",
        "analysis_request": {
            "step": {
                "step_id": "02_other_primary",
                "planned_analysis_role": "primary",
            }
        },
        "step_summary": {},
    }

    assert (
        _find_structured_primary_model_source(
            records=[decoy, record],
            run_dir=run_dir,
            evidence_records=evidence,
        )
        is None
    )


def test_structured_primary_headline_blocks_manifest_scalar_forgery(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _find_structured_primary_model_source,
        _structured_primary_effect_payload,
    )
    from easyicu.research_agent.robustness.primary_effect import (
        _extract_primary_effect_payload_from_records,
    )

    run_dir = tmp_path / "run"
    record, evidence, _script_path = _write_structured_source_authority(run_dir)
    source = _find_structured_primary_model_source(
        records=[record],
        run_dir=run_dir,
        evidence_records=evidence,
    )
    assert source is not None

    forged_record = json.loads(json.dumps(record))
    forged_record["step_summary"].update(
        {
            "primary_or": 9.9,
            "primary_ci_low": 9.8,
            "primary_ci_high": 10.0,
        }
    )
    forged_record["step_summary_evidence_id"] = "structured_primary"
    reported = _extract_primary_effect_payload_from_records(
        [forged_record],
        preferred_predictor="exposure",
    )

    authoritative, errors = _structured_primary_effect_payload(
        source=source,
        reported_payload=reported,
        preferred_predictor="exposure",
    )

    assert errors
    assert any("Current manifest primary_or disagrees" in item for item in errors)
    assert any("evidence id" in item for item in errors)
    assert authoritative is not None
    assert authoritative["primary_or"] == pytest.approx(1.4)
    assert authoritative["evidence_id"] == "table_coefficients"


def test_exact_replay_does_not_advertise_unimplemented_variants() -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _missing_strategy_audit,
        _outcome_executability_audit,
        _unexecutable_locked_spec_ids,
    )

    median = RobustnessSpec(
        spec_id="missing_median",
        axis="missing",
        description="Median imputation.",
        missing_override={"strategy": "median_imputation"},
    )
    missing_audit = _missing_strategy_audit(
        [median],
        structured_source_aware_available=True,
    )
    assert missing_audit[0]["strategy_executable"] is False

    decoy = RobustnessSpec(
        spec_id="missing_source_aware_decoy",
        axis="missing",
        description="Unsupported look-alike strategy.",
        missing_override={"strategy": "not_source_aware_categories"},
    )
    decoy_audit = _missing_strategy_audit(
        [decoy],
        structured_source_aware_available=True,
    )
    assert decoy_audit[0]["strategy_executable"] is False

    alternate_outcome = RobustnessSpec(
        spec_id="alternate_outcome",
        axis="outcome",
        description="Use the locked alternate endpoint.",
        outcome_override={"column": "alternate_outcome"},
    )
    outcome_audit = _outcome_executability_audit(
        specs=[alternate_outcome],
        data=pd.DataFrame(
            {
                "stay_id": [1, 2, 3],
                "primary_outcome": [0, 1, 0],
                "alternate_outcome": [1, 0, 1],
            }
        ),
        primary_outcome="primary_outcome",
        exact_primary_replay_available=True,
    )
    assert outcome_audit[0]["independent_variant"] is True
    assert outcome_audit[0]["outcome_executable"] is False
    assert _unexecutable_locked_spec_ids(
        specs=[median, alternate_outcome],
        membership_rows=[],
        missing_rows=missing_audit,
        outcome_rows=outcome_audit,
    ) == ["missing_median", "alternate_outcome"]


def test_same_scalar_outcome_is_disclosed_without_becoming_blocking() -> None:
    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _outcome_executability_audit,
        _unexecutable_locked_spec_ids,
    )

    duplicate = RobustnessSpec(
        spec_id="same_scalar",
        axis="outcome",
        description="Repeat the same scalar endpoint.",
        outcome_override={"column": "outcome"},
    )
    audits = _outcome_executability_audit(
        specs=[duplicate],
        data=pd.DataFrame({"stay_id": [1, 2], "outcome": [0, 1]}),
        primary_outcome="outcome",
        exact_primary_replay_available=True,
    )

    assert audits[0]["outcome_executable"] is True
    assert audits[0]["independent_variant"] is False
    assert (
        _unexecutable_locked_spec_ids(
            specs=[duplicate],
            membership_rows=[],
            missing_rows=[],
            outcome_rows=audits,
        )
        == []
    )


def test_exact_replay_blocks_script_that_ignores_locked_cohort_membership(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _replay_primary_model_for_cohort,
    )

    source_dir = tmp_path / "source_step"
    source_outputs = source_dir / "outputs"
    source_outputs.mkdir(parents=True)
    source_script = source_dir / "analysis.py"
    source_script.write_text(
        textwrap.dedent(
            """
            import json, os
            from pathlib import Path
            import pandas as pd

            # Regression decoy: ignore COHORT_PARQUET and repeat the original
            # four-row result regardless of the locked replay membership.
            out = Path(os.environ["STEP_OUT_DIR"])
            out.mkdir(parents=True, exist_ok=True)
            contract = {
                "model_id": "primary",
                "exposure_source": "exposure",
                "exposure_expression": "exposure",
                "exposure_role": "primary",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
                "n": 4,
                "event_n": 2,
                "fit_status": "fitted",
                "converged": True,
                "fit_method": "constant_decoy",
            }
            pd.DataFrame([{
                "model_id": "primary",
                "term": "exposure",
                "term_role": "exposure",
                "source_variable": "exposure",
                "odds_ratio": 1.5,
                "ci_low": 1.1,
                "ci_high": 2.0,
            }]).to_csv(out / "coefficients.csv", index=False)
            (out / "step_summary.json").write_text(json.dumps({
                "primary_model_id": "primary",
                # The decoy is wrong about ONE thing -- it ignores the locked
                # membership -- so it must be well formed everywhere else, or
                # the replay refuses it for the wrong reason and this test
                # stops testing membership.
                "coefficient_table": "coefficients.csv",
                "model_contracts": [contract],
            }))
            """
        ),
        encoding="utf-8",
    )
    primary_contract = {
        "model_id": "primary",
        "exposure_source": "exposure",
        "exposure_expression": "exposure",
        "exposure_role": "primary",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "n": 4,
        "event_n": 2,
        "fit_status": "fitted",
        "converged": True,
        "fit_method": "constant_decoy",
    }
    source = {
        "primary_contract": primary_contract,
        "script_path": source_script,
        "step_id": "01_primary_model",
        "script_sha256": hashlib.sha256(source_script.read_bytes()).hexdigest(),
        "outputs_dir": source_outputs,
    }
    spec = RobustnessSpec(
        spec_id="older_adults",
        axis="cohort",
        description="Restrict the replay to two older adults.",
        cohort_override=CohortDefinition(
            name="older_adults",
            inclusion=[_predicate("age", ">=", 50)],
        ),
    )
    data = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [20, 30, 60, 70],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
        }
    )

    replay = _replay_primary_model_for_cohort(
        spec=spec,
        source=source,
        data=data,
        context=SimpleNamespace(),
        out_dir=tmp_path / "robustness_outputs",
    )

    assert replay["index"]["input_n"] == 2
    assert replay["index"]["status"] == "blocked"
    assert replay["row"].converged is False
    assert "model_n=4, input_n=2" in replay["error"]


def test_exact_replay_uses_declared_coefficient_companion_without_primary_id(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from easyicu.research_agent.execution.runners.deterministic_robustness import (
        _replay_primary_model_for_cohort,
    )

    source_dir = tmp_path / "source_step"
    source_outputs = source_dir / "outputs"
    source_outputs.mkdir(parents=True)
    source_script = source_dir / "analysis.py"
    source_script.write_text(
        textwrap.dedent(
            """
            import json, os
            from pathlib import Path
            import pandas as pd

            frame = pd.read_parquet(os.environ["COHORT_PARQUET"])
            out = Path(os.environ["STEP_OUT_DIR"])
            out.mkdir(parents=True, exist_ok=True)
            contract = {
                "model_id": "primary",
                "exposure_source": "exposure",
                "exposure_expression": "exposure",
                "exposure_role": "primary",
                "analysis_role": "primary",
                "analysis_set": "source_aware",
                "n": int(len(frame)),
                "event_n": int(frame["outcome"].sum()),
                "fit_status": "fitted",
                "converged": True,
                "fit_method": "registered_test_model",
            }
            coefficient_name = "adjusted_association_coefficients.csv"
            pd.DataFrame([{
                "model_id": "primary",
                "term": "exposure",
                "term_role": "exposure",
                "source_variable": "exposure",
                "odds_ratio": 1.5,
                "ci_low": 1.1,
                "ci_high": 2.0,
            }]).to_csv(out / coefficient_name, index=False)
            (out / "step_summary.json").write_text(json.dumps({
                "model_contracts": [contract],
                "diagnostic_companions": {"coefficients": coefficient_name},
            }))
            """
        ),
        encoding="utf-8",
    )
    primary_contract = {
        "model_id": "primary",
        "exposure_source": "exposure",
        "exposure_expression": "exposure",
        "exposure_role": "primary",
        "analysis_role": "primary",
        "analysis_set": "source_aware",
        "n": 4,
        "event_n": 2,
        "fit_status": "fitted",
        "converged": True,
        "fit_method": "registered_test_model",
    }
    source = {
        "primary_contract": primary_contract,
        "script_path": source_script,
        "step_id": "01_primary_model",
        "script_sha256": hashlib.sha256(source_script.read_bytes()).hexdigest(),
        "outputs_dir": source_outputs,
    }
    spec = RobustnessSpec(
        spec_id="older_adults",
        axis="cohort",
        description="Restrict the replay to two older adults.",
        cohort_override=CohortDefinition(
            name="older_adults",
            inclusion=[_predicate("age", ">=", 50)],
        ),
    )
    data = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "age": [20, 30, 60, 70],
            "exposure": [0.0, 1.0, 0.0, 1.0],
            "outcome": [0, 1, 0, 1],
        }
    )

    replay = _replay_primary_model_for_cohort(
        spec=spec,
        source=source,
        data=data,
        context=SimpleNamespace(),
        out_dir=tmp_path / "robustness_outputs",
    )

    assert replay["index"]["status"] == "ok"
    assert replay["index"]["input_n"] == 2
    assert replay["row"].converged is True
    assert replay["row"].n == 2
    assert replay["row"].point_estimate == 1.5
    assert replay["contracts"][0]["model_id"] == "primary"
