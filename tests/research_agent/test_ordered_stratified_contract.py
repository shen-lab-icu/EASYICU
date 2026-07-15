"""Closed-contract tests for agent-authored ordered-stratified analyses."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from easyicu.research_agent.audits.patterns import AnalysisPatternAuditor
from easyicu.research_agent.audits.validators import StatisticalValidator
from easyicu.research_agent.context import build_research_context
from easyicu.research_agent.methods.ordered_trends import (
    cochran_armitage_trend,
    jonckheere_terpstra_trend,
    wilson_interval,
)
from easyicu.research_agent.ordered_stratified_contract import (
    ordered_stratified_numeric_findings,
    ordered_stratified_script_findings,
    ordered_stratified_structure_findings,
)
from easyicu.research_agent.plan_utils import _step_contract_findings
from easyicu.research_agent.schema import AnalysisStep


def _step(*, figure_only: bool = False) -> AnalysisStep:
    return AnalysisStep(
        step_id=(
            "05_ordered_outcome_summary_figure"
            if figure_only
            else "05_ordered_outcome_summary"
        ),
        intent="Summarize two outcomes across ordered exposure levels.",
        inputs=["severity_band", "endpoint_flag", "duration_days"],
        expected_outputs=(
            ["figure:absolute_risk_by_level"]
            if figure_only
            else ["table:severity_stratified_outcomes", "test:ordinal_trend"]
        ),
        method="ordinal_stratified_descriptive_analysis",
    )


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "severity_band": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, np.nan],
            "endpoint_flag": [0, 1, 0, 0, 1, np.nan, 1, 1, 0, 1, 0],
            "duration_days": [1, 2, 3, 2, 3, 4, 4, 5, np.nan, 8, 9],
            # A tempting low-cardinality bystander must never enter the replay.
            "unplanned_band": [9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4],
        }
    )


def _contract() -> dict:
    return {
        "schema_version": "1.0",
        "ordered_exposure_column": "severity_band",
        "ordered_levels": [0, 1, 2],
        "cochran_armitage_scores": [0, 1, 2],
        "score_scheme": "consecutive_ordinal_ranks",
        "binary_outcome_column": "endpoint_flag",
        "continuous_outcome_column": "duration_days",
        "locked_cohort_n": 11,
        "valid_ordered_exposure_n": 10,
        "ci_method": "wilson_score",
        "ci_alpha": 0.05,
        "continuous_summary": "median_iqr",
        "quantile_method": "linear",
        "stratified_table": "severity_stratified_outcomes.csv",
        "trend_table": "ordinal_trend.csv",
        "tests": {
            "binary": {
                "test_id": "cochran_armitage",
                "alternative": "two-sided",
            },
            "continuous": {
                "test_id": "jonckheere_terpstra",
                "alternative": "two-sided",
            },
        },
        "multiplicity_policy": "holm_familywise",
        "multiplicity_family_size": 2,
    }


def _write_valid_outputs(out_dir: Path, cohort: pd.DataFrame) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = [0, 1, 2]
    rows = []
    events = []
    totals = []
    for order, level in enumerate(levels):
        level_mask = cohort["severity_band"] == level
        level_n = int(level_mask.sum())
        binary = pd.to_numeric(cohort.loc[level_mask, "endpoint_flag"], errors="coerce")
        binary = binary[binary.isin([0, 1])]
        event_n = int(binary.sum())
        binary_n = int(len(binary))
        events.append(event_n)
        totals.append(binary_n)
        ci = wilson_interval(event_n, binary_n)
        continuous = pd.to_numeric(
            cohort.loc[level_mask, "duration_days"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "level_value": level,
                "level_order": order,
                "level_n": level_n,
                "level_percentage": 100 * level_n / 10,
                "binary_outcome": "endpoint_flag",
                "binary_n": binary_n,
                "binary_missing_n": level_n - binary_n,
                "binary_event_n": event_n,
                "binary_risk": event_n / binary_n,
                "binary_percentage": 100 * event_n / binary_n,
                "binary_ci_low": ci.ci_low,
                "binary_ci_high": ci.ci_high,
                "binary_ci_method": "wilson_score",
                "binary_ci_alpha": 0.05,
                "continuous_outcome": "duration_days",
                "continuous_n": len(continuous),
                "continuous_missing_n": level_n - len(continuous),
                "continuous_median": continuous.median(),
                "continuous_q25": continuous.quantile(0.25),
                "continuous_q75": continuous.quantile(0.75),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "severity_stratified_outcomes.csv", index=False)

    ca = cochran_armitage_trend(
        events,
        totals,
        scores=[0, 1, 2],
        group_order=levels,
    )
    jt_mask = cohort["severity_band"].isin(levels) & np.isfinite(
        pd.to_numeric(cohort["duration_days"], errors="coerce")
    )
    jt = jonckheere_terpstra_trend(
        cohort.loc[jt_mask, "duration_days"].tolist(),
        cohort.loc[jt_mask, "severity_band"].tolist(),
        group_order=levels,
    )
    adjusted = multipletests([ca.p_value, jt.p_value], method="holm")[1]

    def trend_row(result, *, outcome: str, outcome_type: str, test_id: str):
        return {
            "outcome": outcome,
            "outcome_type": outcome_type,
            "test_id": test_id,
            "test_name": result.test_name,
            "alternative": result.alternative,
            "n": result.n,
            "levels_with_data": 3,
            "statistic": result.statistic,
            "statistic_name": result.statistic_type,
            "expected_statistic": result.expected_statistic,
            "variance": result.variance,
            "z_statistic": result.z_statistic,
            "chi_square": result.chi_square,
            "effect_size": result.effect_size,
            "effect_size_name": result.effect_size_name,
            "p_value": result.p_value,
            "adjusted_p": None,
            "p_value_reporting": result.p_value_reporting,
            "log_p_value": result.log_p_value,
            "negative_log10_p": result.negative_log10_p,
            "p_value_bounded": result.p_value_bounded,
            "tie_correction": result.tie_correction,
            "continuity_correction": result.continuity_correction,
            "implementation": result.implementation,
            "score_scheme": result.score_scheme,
            "family_id": "ordered_trend_outcomes",
            "family_size": 2,
            "prespecified": True,
            "multiplicity_policy": "holm_familywise",
            "status": "ok",
        }

    trend_rows = [
        trend_row(
            ca,
            outcome="endpoint_flag",
            outcome_type="binary",
            test_id="cochran_armitage",
        ),
        trend_row(
            jt,
            outcome="duration_days",
            outcome_type="continuous",
            test_id="jonckheere_terpstra",
        ),
    ]
    trend_rows[0]["adjusted_p"] = float(adjusted[0])
    trend_rows[1]["adjusted_p"] = float(adjusted[1])
    pd.DataFrame(trend_rows).to_csv(out_dir / "ordinal_trend.csv", index=False)
    return {"ordered_stratified_contract": _contract()}


def test_structure_contract_accepts_complete_case_neutral_declaration() -> None:
    summary = {"ordered_stratified_contract": _contract()}
    assert not ordered_stratified_structure_findings(step=_step(), step_summary=summary)


def test_structure_contract_rejects_missing_or_misaligned_scores() -> None:
    summary = {"ordered_stratified_contract": _contract()}
    summary["ordered_stratified_contract"]["cochran_armitage_scores"] = [0, 1]

    findings = ordered_stratified_structure_findings(step=_step(), step_summary=summary)

    assert any(f.detail["kind"] == "invalid_ca_scores" for f in findings)


def test_general_step_contract_delegates_to_ordered_structure_contract() -> None:
    findings = _step_contract_findings(step=_step(), step_summary={"status": "ok"})

    assert any(
        finding.validator == "ordered_stratified_contract"
        and finding.detail["kind"] == "missing_contract"
        for finding in findings
    )


def test_structure_contract_does_not_capture_rendering_child() -> None:
    assert not ordered_stratified_structure_findings(
        step=_step(figure_only=True), step_summary={}
    )


def test_script_contract_requires_all_three_validated_primitive_calls() -> None:
    findings = ordered_stratified_script_findings(
        step=_step(),
        script_text="""
from easyicu.research_agent.methods.ordered_trends import wilson_interval
wilson_interval(1, 2)
""",
    )

    missing = next(
        finding
        for finding in findings
        if finding.detail["kind"] == "missing_validated_primitive_call"
    )
    assert missing.detail["missing_calls"] == [
        "cochran_armitage_trend",
        "jonckheere_terpstra_trend",
    ]


def test_script_contract_rejects_spearman_only_for_controlled_method() -> None:
    code = """
wilson_interval(1, 2)
cochran_armitage_trend([1, 2], [3, 4])
jonckheere_terpstra_trend([1, 2], [0, 1], group_order=[0, 1])
spearmanr([0, 1], [1, 2])
"""
    controlled = ordered_stratified_script_findings(step=_step(), script_text=code)
    ordinary = ordered_stratified_script_findings(
        step=AnalysisStep(
            step_id="correlation",
            intent="Estimate an ordinary correlation.",
            method="correlation_analysis",
            expected_outputs=["table:correlation"],
        ),
        script_text=code,
    )

    assert any(f.detail["kind"] == "spearman_substituted_for_jt" for f in controlled)
    assert ordinary == []


def test_analysis_pattern_auditor_routes_controlled_tool_errors_to_usage_gate() -> None:
    context = build_research_context(
        research_question="Do two outcomes change across ordered severity groups?",
        cohort=_cohort(),
        cohort_name="synthetic",
        database="synthetic",
        target_outcome="endpoint_flag",
        primary_exposure="severity_band",
    )
    findings = AnalysisPatternAuditor().audit(
        context=context,
        step=_step(),
        script_text="from scipy.stats import spearmanr\nspearmanr([0, 1], [1, 2])",
    )

    kinds = {finding.detail.get("kind") for finding in findings}
    assert "missing_validated_primitive_call" in kinds
    assert "spearman_substituted_for_jt" in kinds


def test_numeric_contract_replays_complete_tables_from_locked_cohort(
    tmp_path: Path,
) -> None:
    cohort = _cohort()
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    summary = _write_valid_outputs(out_dir, cohort)

    findings = ordered_stratified_numeric_findings(
        cohort_path=cohort_path,
        step=_step(),
        out_dir=out_dir,
        step_summary=summary,
    )

    assert findings == []


@pytest.mark.parametrize(
    ("table_name", "column", "replacement", "expected_kind"),
    [
        (
            "severity_stratified_outcomes.csv",
            "binary_ci_high",
            0.999,
            "stratified_value_mismatch",
        ),
        (
            "severity_stratified_outcomes.csv",
            "continuous_median",
            99.0,
            "stratified_value_mismatch",
        ),
        ("ordinal_trend.csv", "statistic", -99.0, "trend_value_mismatch"),
        ("ordinal_trend.csv", "p_value", 0.0, "invalid_p_value"),
        ("ordinal_trend.csv", "adjusted_p", 0.7, "trend_value_mismatch"),
    ],
)
def test_numeric_contract_rejects_tampered_scientific_values(
    tmp_path: Path,
    table_name: str,
    column: str,
    replacement: float,
    expected_kind: str,
) -> None:
    cohort = _cohort()
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    summary = _write_valid_outputs(out_dir, cohort)
    table_path = out_dir / table_name
    table = pd.read_csv(table_path)
    table.loc[0, column] = replacement
    table.to_csv(table_path, index=False)

    findings = ordered_stratified_numeric_findings(
        cohort_path=cohort_path,
        step=_step(),
        out_dir=out_dir,
        step_summary=copy.deepcopy(summary),
    )

    assert any(f.detail["kind"] == expected_kind for f in findings)


def test_numeric_contract_rejects_spearman_label_even_with_jt_named_test_id(
    tmp_path: Path,
) -> None:
    cohort = _cohort()
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    summary = _write_valid_outputs(out_dir, cohort)
    trend_path = out_dir / "ordinal_trend.csv"
    trend = pd.read_csv(trend_path)
    trend.loc[trend["outcome"] == "duration_days", "test_name"] = (
        "Spearman rank test presented as a JT equivalent"
    )
    trend.to_csv(trend_path, index=False)

    findings = ordered_stratified_numeric_findings(
        cohort_path=cohort_path,
        step=_step(),
        out_dir=out_dir,
        step_summary=summary,
    )

    assert any(f.detail["kind"] == "spearman_substituted_for_jt" for f in findings)


def test_statistical_validator_delegates_to_locked_cohort_numeric_replay(
    tmp_path: Path,
) -> None:
    cohort = _cohort()
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    out_dir = tmp_path / "out"
    summary = _write_valid_outputs(out_dir, cohort)
    trend_path = out_dir / "ordinal_trend.csv"
    trend = pd.read_csv(trend_path)
    trend.loc[0, "p_value"] = 0.0
    trend.to_csv(trend_path, index=False)
    context = build_research_context(
        research_question="Do two outcomes change across ordered severity groups?",
        cohort=cohort,
        cohort_name="synthetic",
        database="synthetic",
        target_outcome="endpoint_flag",
        primary_exposure="severity_band",
    )

    findings = StatisticalValidator().audit(
        context=context,
        cohort_path=cohort_path,
        step=_step(),
        out_dir=out_dir,
        step_summary=summary,
    )

    assert any(
        finding.validator == "ordered_stratified_contract"
        and finding.detail["kind"] == "invalid_p_value"
        for finding in findings
    )


def test_numeric_replay_is_wired_before_the_existing_in_run_repair_gate() -> None:
    from easyicu.research_agent import pipeline_execute

    source = inspect.getsource(pipeline_execute)
    replay = source.index(
        "early_contract_findings += ordered_stratified_numeric_findings"
    )
    repair_gate = source.index("early_contract_errors =", replay)
    typed_ticket = source.index(
        "structured_repair_ticket = typed_repair_ticket(", repair_gate
    )

    assert replay < repair_gate
    # Every error returned by the replay now enters the aggregate typed repair
    # ticket. The retired validator-name string filter would have made this
    # contract depend on prose/routing text and could silently drop a new typed
    # occurrence.
    assert repair_gate < typed_ticket


def test_controlled_method_has_no_whole_step_deterministic_runner() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/easyicu/research_agent"
    runner_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package_root.glob("deterministic_*.py")
    )

    assert "ordinal_stratified_descriptive_analysis" not in runner_sources
