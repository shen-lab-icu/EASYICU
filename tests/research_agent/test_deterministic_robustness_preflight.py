from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

from easyicu.research_agent.cohort_schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
    cohort_definition_sha,
)
from easyicu.research_agent.deterministic_robustness import (
    robustness_sensitivity_preflight_code,
)
from easyicu.research_agent.robustness_panel import (
    RobustnessSpec,
    robustness_specs_sha,
)


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
    manifest = {"run_id": "test_run", "per_step_records": records}
    (run_dir / "manifest_partial.json").write_text(json.dumps(manifest))
    return out_dir, cohort_path, universe_path


def _run_generated(
    monkeypatch,
    *,
    out_dir: Path,
    cohort_path: Path,
    universe_path: Path,
) -> None:
    run_dir = out_dir.parents[2]
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("COHORT_PARQUET", str(cohort_path))
    monkeypatch.setenv("EASYICU_UNIVERSE_PARQUET", str(universe_path))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv(
        "EASYICU_MANIFEST_PARTIAL", str(run_dir / "manifest_partial.json")
    )
    code = robustness_sensitivity_preflight_code()
    exec(compile(code, "<robustness-preflight>", "exec"), {})


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

    summary = json.loads((out_dir / "step_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["aliases"]["sensitivity_comparison"] == "robustness_matrix.csv"
    assert (
        summary["aliases"]["cohort_overlap_and_attrition"]
        == "cohort_overlap_and_attrition.csv"
    )
    assert (
        summary["aliases"]["sensitivity_specification_grid"]
        == "sensitivity_specification_grid.csv"
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


def test_structured_preflight_replays_exact_primary_code_and_emits_spec_by_model_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    out_dir, cohort_path, universe_path = _prepare_run(
        tmp_path, include_primary=False
    )
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
    (run_dir / "manifest_partial.json").write_text(
        json.dumps(
            {
                "run_id": "structured_replay",
                "per_step_records": [
                    {
                        "step_id": "07_primary_model",
                        "status": "ok",
                        "step_summary_evidence_id": "structured_primary",
                        "step_summary": source_summary,
                    }
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
            missing_override={
                "strategy": "source_aware_categories_no_imputation"
            },
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
    assert summary["model_contracts"][0]["exposure_expression"] == (
        "log1p(exposure)"
    )
    assert summary["complete_case_n"] == source_summary["model_contracts"][2]["n"]
    contracts = summary["robustness_model_contracts"]
    assert len(contracts) == 16
    assert {
        (contract["spec_id"], contract["model_id"])
        for contract in contracts
    } == {
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
    assert replay_index["source_script_sha256"] == summary["primary_model_replay"][
        "source_script_sha256"
    ]
    assert all(item["status"] == "ok" for item in replay_index["variants"])
