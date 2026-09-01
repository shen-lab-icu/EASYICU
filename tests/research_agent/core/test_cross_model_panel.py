"""Cross-model plan and robustness-panel concordance infrastructure."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from easyicu.research_agent.cohort.schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.evaluation.cross_model_panel import (
    BackendIdentity,
    FieldDisagreement,
    compare_panel_primaries,
    compare_plans,
    write_cross_model_report,
)
from easyicu.research_agent.robustness.panel import (
    PRIMARY_SPEC_ID,
    RobustnessPanel,
    RobustnessPanelRow,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def test_compare_plans_detects_cohort_disagreement() -> None:
    plans = {
        "mock_a": _plan(window_end=24),
        "mock_b": _plan(window_end=1),
    }
    report = compare_plans(plans)
    fields = {d.field_path: d for d in report.field_disagreements}
    assert fields["cohort.inclusion.time_windows"].all_agree is False


def test_compare_plans_full_agreement_rate_is_one() -> None:
    report = compare_plans({"mock_a": _plan(), "mock_b": _plan()})
    assert report.overall_agreement_rate == 1.0
    assert all(d.all_agree for d in report.field_disagreements)


def test_compare_panel_primaries_overlapping_cis_marked_concordant() -> None:
    report = compare_panel_primaries(
        {
            "mock_a": _panel(point=1.4, low=1.1, high=1.8),
            "mock_b": _panel(point=1.5, low=1.3, high=1.9),
        }
    )
    assert report.backends_concordant is True
    assert report.range_low == 1.1
    assert report.range_high == 1.9


def test_compare_panel_primaries_non_overlapping_cis_not_concordant() -> None:
    report = compare_panel_primaries(
        {
            "mock_a": _panel(point=1.1, low=1.0, high=1.2),
            "mock_b": _panel(point=2.0, low=1.8, high=2.2),
        }
    )
    assert report.backends_concordant is False


def test_cross_model_report_written_and_re_readable(tmp_path: Path) -> None:
    plan_conc = compare_plans({"mock_a": _plan(), "mock_b": _plan()})
    panel_conc = compare_panel_primaries(
        {
            "mock_a": _panel(point=1.4, low=1.1, high=1.8),
            "mock_b": _panel(point=1.5, low=1.3, high=1.9),
        }
    )
    path = write_cross_model_report(tmp_path, plan_conc, panel_conc)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["plan_concordance"]["overall_agreement_rate"] == 1.0
    assert payload["panel_primary_concordance"]["backends_concordant"] is True


def test_cli_dry_run_with_mock_backends(tmp_path: Path) -> None:
    backends_path = tmp_path / "backends.json"
    backends_path.write_text(
        json.dumps(
            {
                "backends": [
                    _backend("mock_a").to_dict(),
                    _backend("mock_b").to_dict(),
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_cross_model_check.py",
            "--backends",
            str(backends_path),
            "--mode",
            "plan_only",
            "--question",
            "Is age associated with mortality?",
            "--out-dir",
            str(out_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((out_dir / "cross_model_report.json").read_text())
    assert "plan_concordance" in payload


def test_cli_rejects_single_backend(tmp_path: Path) -> None:
    backends_path = tmp_path / "one_backend.json"
    backends_path.write_text(json.dumps([_backend("mock_a").to_dict()]), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "tools/run_cross_model_check.py",
            "--backends",
            str(backends_path),
            "--question",
            "Is age associated with mortality?",
            "--out-dir",
            str(tmp_path / "out"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "at least two backends" in result.stderr


def test_field_disagreement_serializes_with_complex_values() -> None:
    disagreement = FieldDisagreement(
        field_path="backend.identity",
        values_by_backend={"mock_a": _backend("mock_a")},
        all_agree=False,
    )
    payload = disagreement.to_dict()
    assert payload["values_by_backend"]["mock_a"]["name"] == "mock_a"


def _backend(name: str) -> BackendIdentity:
    return BackendIdentity(
        name=name,
        llm_provider="mock",
        llm_model="mock-model",
        prompt_pack_version="easyicu-research-agent-prompts/test",
        env_overrides={},
    )


def _plan(window_end: float = 24) -> AnalysisPlan:
    cohort = CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=TimeWindow(
                    anchor="icu_admit",
                    start_offset_hours=0,
                    end_offset_hours=window_end,
                ),
                aggregation="first",
                op=">=",
                value=18,
            ),
        ),
        exclusion=(),
        derived_from_named=None,
    )
    return AnalysisPlan(
        research_question="Is age associated with mortality?",
        cohort=cohort,
        steps=[
            AnalysisStep(
                step_id="01_primary",
                intent="Fit a primary model.",
                inputs=["age", "death"],
                expected_outputs=["primary_association"],
            )
        ],
    )


def _panel(*, point: float, low: float, high: float) -> RobustnessPanel:
    return RobustnessPanel.from_rows(
        [
            RobustnessPanelRow(
                spec_id=PRIMARY_SPEC_ID,
                axis="primary",
                n=100,
                point_estimate=point,
                ci_low=low,
                ci_high=high,
                se=0.1,
                evidence_id="primary",
                converged=True,
            )
        ]
    )
