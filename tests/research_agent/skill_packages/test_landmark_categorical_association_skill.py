"""Contract tests for the fixed-landmark categorical association skill.

The skill is a packaged reference workflow: four entry functions, verification
tokens, a single-row ``key_metrics.csv`` and an export consistency gate.  These
tests pin that contract on the synthetic example so the package can be smoke
tested in seconds and so a regression in any owner kernel it composes surfaces
here as well.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    run_adjusted_association_from_env,
)
from easyicu.research_agent.skill_packages.landmark_categorical_association import (
    ANALYSIS_TOKEN,
    EXPORT_TOKEN,
    LOAD_TOKEN,
    PLOTS_TOKEN,
    UNDECLARED_PROVENANCE,
    AnalysisContractError,
    CohortContractError,
    CovariateSpec,
    ExportConsistencyError,
    LandmarkCategoricalSpec,
    caveat_sentences,
    example_spec,
    example_truth,
    export_all,
    generate_all_plots,
    load_cohort,
    make_example_cohort,
    run_all,
    run_analysis,
)
from easyicu.research_agent.skill_packages.landmark_categorical_association.scripts.run_all import main

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

EXPECTED_TABLES = {
    "cohort_flow",
    "exposure_level_counts",
    "measurement_audit",
    "table_one",
    "absolute_risk",
    "adjusted_association_estimates",
    "adjusted_association_coefficients",
    "adjusted_trend",
    "ordinal_trend_tests",
    "secondary_outcome_summary",
    "association_sensitivity_grid",
    "functional_form_sensitivity",
    "robustness_summary",
}


@pytest.fixture(scope="module")
def example_run(tmp_path_factory: pytest.TempPathFactory):
    out_dir = tmp_path_factory.mktemp("skill_example")
    frame = make_example_cohort(3000)
    return run_all(frame, example_spec(), out_dir, verbose=False), out_dir


def _text_hashes(out_dir: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out_dir.iterdir())
        if path.suffix in {".csv", ".json", ".md"} and path.name != "manifest.json"
    }


def test_four_steps_print_their_verification_tokens(tmp_path: Path, capsys) -> None:
    frame = make_example_cohort(1500)
    spec = example_spec()
    cohort = load_cohort(frame, spec)
    result = run_analysis(cohort, work_dir=tmp_path)
    figures = generate_all_plots(result, tmp_path)
    export_all(result, tmp_path, figures=figures)
    captured = capsys.readouterr().out
    for token in (LOAD_TOKEN, ANALYSIS_TOKEN, PLOTS_TOKEN, EXPORT_TOKEN):
        assert token in captured
    assert "Consistency check: PASSED" in captured


def test_example_run_exports_every_table_key_metrics_and_manifest(example_run) -> None:
    run, out_dir = example_run
    names = {path.name for path in out_dir.iterdir()}
    for table in EXPECTED_TABLES:
        assert f"{table}.csv" in names
    for required in ("key_metrics.csv", "caveat_flags.json", "spec.json", "report.md", "manifest.json", "figures.json"):
        assert required in names
    assert run.receipt.checks_passed == run.receipt.checks_total >= 12

    key_metrics = pd.read_csv(out_dir / "key_metrics.csv")
    assert len(key_metrics) == 1
    row = key_metrics.iloc[0]
    assert int(row["n_landmark"]) == run.cohort.n_landmark
    assert int(row["n_exposure_known"]) + int(row["n_exposure_unknown"]) == run.cohort.n_landmark
    assert int(row["n_fit"]) <= int(row["n_exposure_known"])
    assert row["primary_contrast_level"] == 3 or str(row["primary_contrast_level"]) == "3"
    assert math.isclose(float(row["primary_or"]), float(run.result.primary_row()["estimate"]))

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_provenance"] == example_truth()["provenance"]
    assert manifest["consistency_checks_passed"] == manifest["consistency_checks_total"]
    for name, digest in _text_hashes(out_dir).items():
        assert manifest["files"][name]["sha256"] == digest


def test_primary_estimate_matches_the_host_kernel_and_recovers_the_design(example_run, tmp_path: Path) -> None:
    run, _out_dir = example_run
    spec = run.result.spec
    primary = run.result.primary_row()
    truth = example_truth()["true_or_stage3_vs_0"]
    assert float(primary["ci_low"]) < truth < float(primary["ci_high"])
    estimates = run.result.adjusted_association_estimates.set_index("exposure_level")["estimate"]
    assert float(estimates.loc[3] if 3 in estimates.index else estimates.loc["3"]) > float(
        estimates.loc[1] if 1 in estimates.index else estimates.loc["1"]
    )
    # The same kernel, called directly on the same rows, must give the same number.
    known = run.cohort.known()
    direct = run_adjusted_association_from_env(
        requirement_id="direct",
        exposure=spec.exposure,
        outcome=spec.outcome,
        covariates=spec.covariate_names(),
        model_terms=[term.model_dump(mode="json") for term in spec.model_terms()],
        estimator_kind="logistic",
        analysis_set="source_aware",
        analysis_role="primary",
        method_family="statsmodels_logit_mle",
        primary_contrast_level=spec.primary_contrast_level,
        dependence=spec.dependence,
        frame=known,
        cohort_path=None,
        emit_step_summary=False,
        output_dir=tmp_path / "direct",
    )
    assert math.isclose(float(direct["primary_estimate"]), float(primary["estimate"]), rel_tol=1e-12)
    assert direct["variance_estimator"] == "cluster_robust"
    assert run.result.key_metrics["cluster_count"] == direct["cluster_count"]


def test_unknown_exposure_is_described_but_never_modelled(example_run) -> None:
    run, _out_dir = example_run
    spec = run.result.spec
    risk = run.result.absolute_risk.set_index("level")
    assert int(risk.loc[spec.unknown_level_label, "n"]) == run.cohort.n_exposure_unknown > 0
    assert bool(risk.loc[spec.unknown_level_label, "in_primary_model"]) is False
    assert int(risk.loc[risk["in_primary_model"].astype(bool), "n"].sum()) == run.cohort.n_exposure_known
    assert int(run.result.primary_summary["n_total"]) <= run.cohort.n_exposure_known
    flow = run.result.cohort_flow
    split = flow.loc[flow["action"].eq("split_not_exclude")].iloc[0]
    assert int(split["n_excluded"]) == 0
    audit = run.result.measurement_audit
    primary_audit = audit.loc[audit["column"].eq(spec.exposure)].iloc[0]
    assert int(primary_audit["n_unknown"]) == run.cohort.n_exposure_unknown
    assert run.result.caveat_flags["unknown_exposure_share_high"] is True
    sentences = caveat_sentences(spec, run.result.caveat_flags)
    assert any("non-evaluable exposure" in sentence for sentence in sentences)


def test_sensitivity_grid_and_functional_form_cover_every_declared_axis(example_run) -> None:
    run, _out_dir = example_run
    spec = run.result.spec
    grid = run.result.association_sensitivity_grid
    assert grid["variant_id"].tolist()[0] == "primary"
    assert set(grid["variant_id"]) == {
        "primary",
        *(f"alternate_exposure__{column}" for column in spec.alternate_exposures),
        "first_stay_only",
    }
    assert grid["fit_status"].eq("fitted").all()
    assert grid["direction_consistent_with_primary"].astype(bool).all()
    functional = run.result.functional_form_sensitivity
    assert functional["covariate"].tolist() == spec.functional_form_covariates
    assert functional["fit_status"].eq("fitted").all()
    assert functional["variance_estimator"].eq("cluster_robust").all()
    trend = run.result.ordinal_trend_tests
    assert trend.loc[trend["in_holm_family"].astype(bool), "p_value_holm"].notna().all()
    assert int(trend["holm_family_size"].iloc[0]) == 2
    adjusted_trend = run.result.adjusted_trend.iloc[0]
    assert adjusted_trend["fit_status"] == "fitted" and float(adjusted_trend["estimate"]) > 1.0


def test_report_quotes_only_exported_numbers(example_run) -> None:
    run, out_dir = example_run
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    metrics = run.result.key_metrics
    assert f"{metrics['primary_or']:.2f}" in report
    assert f"{metrics['n_fit']:,}" in report
    assert "claim ceiling **analysis_only**" in report
    assert "never recoded" in report
    for figure_id, artifact in run.figures.items():
        assert artifact.png_path in report
        assert (out_dir / artifact.png_path).is_file()
        assert (out_dir / artifact.svg_path).is_file()
        assert figure_id in report


def test_export_is_deterministic_for_text_artifacts(tmp_path: Path) -> None:
    frame = make_example_cohort(1200)
    first = run_all(frame, example_spec(), tmp_path / "a", verbose=False)
    second = run_all(make_example_cohort(1200), example_spec(), tmp_path / "b", verbose=False)
    assert _text_hashes(tmp_path / "a") == _text_hashes(tmp_path / "b")
    assert first.receipt.manifest_sha256 == second.receipt.manifest_sha256


def test_small_cohort_raises_epv_flag_and_mandatory_sentence(tmp_path: Path) -> None:
    frame = make_example_cohort(320, seed=7)
    run = run_all(frame, example_spec(), tmp_path, verbose=False)
    flags = run.result.caveat_flags
    assert flags["epv"] is not None and flags["epv"] < example_spec().epv_minimum
    assert flags["epv_below_minimum"] is True
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "potentially overfitted" in report


def test_consistency_gate_refuses_tampered_key_metrics(tmp_path: Path, capsys) -> None:
    frame = make_example_cohort(1000)
    cohort = load_cohort(frame, example_spec(), verbose=False)
    result = run_analysis(cohort, work_dir=tmp_path, verbose=False)
    result.key_metrics["primary_or"] = float(result.key_metrics["primary_or"]) * 1.5
    with pytest.raises(ExportConsistencyError):
        export_all(result, tmp_path, figures=None)
    assert EXPORT_TOKEN not in capsys.readouterr().out
    assert not (tmp_path / "manifest.json").exists()


def test_cohort_contract_rejects_undeclared_levels_and_all_unknown() -> None:
    spec = example_spec()
    frame = make_example_cohort(400)
    bad = frame.copy()
    bad.loc[bad.index[0], "aki_stage_strict"] = 7.0
    with pytest.raises(CohortContractError, match="outside the declared"):
        load_cohort(bad, spec, verbose=False)
    unknown = frame.copy()
    unknown["aki_stage_strict"] = float("nan")
    with pytest.raises(CohortContractError, match="unknown exposure"):
        load_cohort(unknown, spec, verbose=False)
    missing = frame.drop(columns=["charlson"])
    with pytest.raises(CohortContractError, match="required column"):
        load_cohort(missing, spec, verbose=False)


def test_spec_rejects_incoherent_designs() -> None:
    base = example_spec().model_dump(mode="json")
    with pytest.raises(ValueError):
        LandmarkCategoricalSpec.model_validate({**base, "reference_level": "9"})
    with pytest.raises(ValueError):
        LandmarkCategoricalSpec.model_validate({**base, "primary_contrast_level": "0"})
    with pytest.raises(ValueError):
        LandmarkCategoricalSpec.model_validate({**base, "functional_form_covariates": ["sex"]})
    with pytest.raises(ValueError):
        CovariateSpec(name="sex", coding="binary", levels=["Female"], reference_level="Female")
    spec = LandmarkCategoricalSpec.model_validate({**base, "reference_level": 0, "primary_contrast_level": 3.0})
    assert spec.reference_level == "0" and spec.primary_contrast_level == "3"


def test_primary_failure_is_fail_closed(tmp_path: Path) -> None:
    spec = example_spec()
    frame = make_example_cohort(400)
    frame["death"] = 0.0  # no events: the declared model cannot be fitted
    cohort = load_cohort(frame, spec, verbose=False)
    with pytest.raises(AnalysisContractError):
        run_analysis(cohort, work_dir=tmp_path, verbose=False)


def test_provenance_is_declared_carried_or_explicitly_undeclared(tmp_path: Path) -> None:
    """The manifest names the data source; silence is recorded, never blank."""

    spec = example_spec()
    frame = make_example_cohort(600)
    # 1. A DataFrame that carries its own provenance keeps it (the synthetic example).
    carried = load_cohort(frame, spec, verbose=False)
    assert carried.provenance == example_truth()["provenance"]
    # 2. An explicit declaration wins over the frame's own label.
    declared = load_cohort(frame, spec, provenance="official_demo:eicu_demo_v2_0_1", verbose=False)
    assert declared.provenance == "official_demo:eicu_demo_v2_0_1"
    # 3. A file without a declaration is recorded as undeclared, in the manifest and the report.
    cohort_path = tmp_path / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    run = run_all(cohort_path, spec, tmp_path / "out", verbose=False)
    assert run.cohort.provenance == UNDECLARED_PROVENANCE
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_provenance"] == UNDECLARED_PROVENANCE
    assert f"Data provenance: **{UNDECLARED_PROVENANCE}**" in (tmp_path / "out" / "report.md").read_text(
        encoding="utf-8"
    )
    # 4. The CLI flag reaches the manifest.
    out_dir = tmp_path / "cli"
    assert (
        main(
            [
                "--cohort", str(cohort_path), "--spec", str(_write_spec(tmp_path, spec)),
                "--out", str(out_dir), "--provenance", "official_demo:mimic_iv_demo_v2_2", "--quiet",
            ]
        )
        == 0
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_provenance"] == "official_demo:mimic_iv_demo_v2_2"


def _write_spec(directory: Path, spec) -> Path:
    path = directory / "spec.json"
    path.write_text(json.dumps(spec.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_figures_use_reader_labels_not_identifiers(example_run) -> None:
    """Axis text names levels, predicates and refits the way the report does."""

    from easyicu.research_agent.skill_packages.landmark_categorical_association.scripts.generate_all_plots import (
        _flow_label,
        _variant_label,
    )

    run, out_dir = example_run
    spec = run.result.spec
    assert _flow_label("nonnegative_event_time") == "Non-negative event time"
    assert _flow_label("exposure_known") == "Exposure state at landmark"
    assert _flow_label("some_new_predicate") == "Some new predicate"
    assert _variant_label("primary", spec) == "Primary model"
    assert _variant_label("first_stay_only", spec) == "First ICU stay only"
    assert _variant_label("alternate_exposure__aki_stage_creat_strict", spec) == (
        "Alternate definition: Strict creatinine-domain stage"
    )
    assert _variant_label("functional_form__age", spec) == "Spline form: Age (years)"
    # The SVG is text: level rows carry the contrast only, the exposure name once.
    forest = (out_dir / "figure_adjusted_association_forest.svg").read_text(encoding="utf-8")
    assert "3 vs 0" in forest
    assert forest.count("Strict KDIGO AKI stage, 0-24 h") == 1
    flow = (out_dir / "figure_cohort_flow.svg").read_text(encoding="utf-8")
    assert "Alive at landmark" in flow and "alive_at_landmark" not in flow


def test_cli_runs_the_example(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli"
    spec_path = tmp_path / "spec.json"
    assert main(["--example", "--example-size", "900", "--out", str(out_dir), "--quiet", "--write-example-spec", str(spec_path)]) == 0
    assert (out_dir / "key_metrics.csv").is_file()
    LandmarkCategoricalSpec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))
