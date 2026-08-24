"""Focused tests for the deterministic absolute-risk descriptive runner."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners.deterministic_descriptive import (
    _is_categorical,
    absolute_risk_context_code,
)
from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.schema import AnalysisStep
from easyicu.research_agent.reporting.writer_evidence import (
    _render_writer_evidence_digest,
)


def _execute_runner(
    run_dir: Path,
    *,
    cohort: pd.DataFrame,
    context: dict,
    inputs: list[str],
    step_id: str = "06_absolute_risk_context",
) -> tuple[dict, Path]:
    out_dir = run_dir / "steps" / step_id / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_path = run_dir / "cohort_analysis.parquet"
    cohort.to_parquet(cohort_path, index=False)
    (run_dir / "research_context.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    plan_name = "analysis_plan_revision_2.json"
    (run_dir / plan_name).write_text(
        json.dumps(
            {
                "research_question": "Structured descriptive-risk test",
                "steps": [
                    {
                        "step_id": step_id,
                        "intent": "Report exposure prevalence and absolute risk.",
                        "inputs": inputs,
                        "expected_outputs": ["table:exposure_outcome_summary"],
                        "method": "absolute_risk_context",
                        "icu_rule_refs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest_partial.json").write_text(
        json.dumps({"plan_path": plan_name}), encoding="utf-8"
    )

    env = {
        "STEP_OUT_DIR": str(out_dir),
        "COHORT_PARQUET": str(cohort_path),
        "OUTCOME_COL": str(context["target_outcome"]),
    }
    with patch.dict(os.environ, env, clear=False):
        code = absolute_risk_context_code()
        try:
            exec(compile(code, "<deterministic_absolute_risk>", "exec"), {})
        except SystemExit:
            pass

    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    return summary, out_dir


def _wilson(count: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = count / n
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n))
        / denom
    )
    return centre - half, centre + half


def test_numeric_cardinality_does_not_invent_categorical_exposure() -> None:
    values = pd.Series([0, 1, 0, 1], dtype="int64")

    assert _is_categorical({"variables": []}, "exposure", values) is False
    assert (
        _is_categorical(
            {
                "variables": [
                    {
                        "name": "exposure",
                        "dtype": "int64",
                        "is_ordinal": True,
                    }
                ]
            },
            "exposure",
            values,
        )
        is True
    )


def test_structured_inputs_produce_level_risk_source_states_and_continuous_summary(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame(
        {
            "stay_key": np.arange(8),
            "endpoint_flag": [0, 1, 1, 0, 1, 0, 1, 0],
            "organ_stage_max": [0, 1, 2, np.nan, np.nan, 1, 0, 2],
            "organ_stage_measured": [1, 1, 1, 0, 1, 0, 2, 1],
            "organ_stage_n": [1, 1, 1, 0, 1, 1, 1, 1],
            # Only eight unique values, but metadata declares this continuous.
            # The runner must not turn low sample cardinality into post-hoc bins.
            "marker_peak": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "unrelated_score": np.arange(8) + 20,
        }
    )
    context = {
        "primary_exposure": "organ_stage_max",
        "target_outcome": "endpoint_flag",
        "variables": [
            {"name": "stay_key", "role": "id", "dtype": "int64"},
            {
                "name": "endpoint_flag",
                "role": "outcome",
                "dtype": "int64",
                "is_ordinal": False,
                "observed_domain": {"is_binary": True},
            },
            {
                "name": "organ_stage_max",
                "role": "ordinal_score",
                "dtype": "float64",
                "is_ordinal": True,
                "ordinal_levels": [0, 1, 2],
                "observed_domain": {"is_binary": False},
            },
            {
                "name": "marker_peak",
                "role": "lab",
                "dtype": "float64",
                "is_ordinal": False,
                "observed_domain": {"is_binary": False, "n_unique": 8},
            },
            {
                "name": "unrelated_score",
                "role": "score",
                "dtype": "int64",
                "is_ordinal": True,
            },
        ],
    }
    summary, out_dir = _execute_runner(
        tmp_path,
        cohort=cohort,
        context=context,
        inputs=[
            "endpoint_flag",
            "organ_stage_max",
            "organ_stage_measured",
            "organ_stage_n",
            "marker_peak",
        ],
    )

    assert summary["status"] == "ok"
    assert summary["analysis_family"] == "absolute_risk_context"
    assert summary["exposure_columns"] == ["organ_stage_max", "marker_peak"]
    assert "unrelated_score" not in summary["exposure_columns"]
    assert summary["adjusted_effect"] is None
    reporting = summary["reportable_descriptive_results"]
    assert reporting["interpretation_ceiling"] == "descriptive_not_causal"
    assert reporting["overall_outcome"]["n"] == 8
    assert reporting["overall_outcome"]["event_n"] == 4
    assert reporting["overall_outcome"]["risk_pct"] == pytest.approx(50.0)
    stage_reporting = next(
        item
        for item in reporting["exposures"]
        if item["exposure"] == "organ_stage_max"
    )
    observed_reporting = next(
        item
        for item in stage_reporting["groups"]
        if item["group_type"] == "source_state"
        and item["group_value"] == "observed"
    )
    assert observed_reporting["n"] == 4
    assert observed_reporting["outcome_event_n"] == 2
    digest = _render_writer_evidence_digest(
        [
            {
                "step_id": "06_absolute_risk_context",
                "status": "ok",
                "step_summary": summary,
            }
        ]
    )
    assert '"reportable_descriptive_results"' in digest
    assert '"outcome_event_n": 2' in digest

    table = pd.read_csv(out_dir / "exposure_outcome_summary.csv")
    stage = table[table["exposure"] == "organ_stage_max"]
    level_prevalence = stage[
        (stage["group_type"] == "exposure_level")
        & (stage["estimate_type"] == "prevalence")
    ]
    assert set(level_prevalence["group_value"].astype(str)) == {"0", "1", "2"}
    # Only the genuine observed stage-0 row is a physiologic zero. Missing or
    # inconsistent rows are not silently imputed into level 0.
    assert int(level_prevalence.loc[level_prevalence["group_value"] == "0", "n"].iloc[0]) == 1

    source_prevalence = stage[
        (stage["group_type"] == "source_state")
        & (stage["estimate_type"] == "prevalence")
    ].set_index("group_value")
    assert source_prevalence["n"].astype(int).to_dict() == {
        "observed": 4,
        "no_source": 1,
        "measurement_missing": 1,
        "inconsistent": 2,
    }

    stage_two_risk = stage[
        (stage["group_type"] == "exposure_level")
        & (stage["group_value"] == "2")
        & (stage["estimate_type"] == "outcome_risk")
    ].iloc[0]
    expected_low, expected_high = _wilson(1, 2)
    assert int(stage_two_risk["n"]) == 2
    assert int(stage_two_risk["event_n"]) == 1
    assert stage_two_risk["outcome_risk"] == pytest.approx(0.5)
    assert stage_two_risk["ci_low"] == pytest.approx(expected_low)
    assert stage_two_risk["ci_high"] == pytest.approx(expected_high)

    marker = table[table["exposure"] == "marker_peak"]
    assert not (marker["group_type"] == "exposure_level").any()
    distribution = marker[marker["estimate_type"] == "continuous_distribution"].iloc[0]
    assert int(distribution["n"]) == 8
    assert distribution["median"] == pytest.approx(4.5)
    assert distribution["q25"] == pytest.approx(2.75)
    assert distribution["q75"] == pytest.approx(6.25)


def test_tidy_output_is_consumable_by_existing_association_context_renderer(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.pipeline import _association_descriptive_context

    cohort = pd.DataFrame(
        {
            "result_flag": [0, 1, 0, 1, 1, 0],
            "therapy_group": ["A", "A", "B", "B", "B", "A"],
        }
    )
    context = {
        "primary_exposure": "therapy_group",
        "target_outcome": "result_flag",
        "variables": [
            {
                "name": "result_flag",
                "role": "outcome",
                "dtype": "int64",
                "is_ordinal": False,
                "observed_domain": {"is_binary": True},
            },
            {
                "name": "therapy_group",
                "role": "categorical_exposure",
                "dtype": "object",
                "is_ordinal": False,
            },
        ],
    }
    summary, _out_dir = _execute_runner(
        tmp_path,
        cohort=cohort,
        context=context,
        inputs=["result_flag", "therapy_group"],
    )
    assert summary["status"] == "ok"

    figure_out = tmp_path / "steps" / "06_absolute_risk_context_figure" / "outputs"
    figure_out.mkdir(parents=True)
    result = _association_descriptive_context(
        run_dir=tmp_path,
        current_step_id="06_absolute_risk_context_figure",
        out_dir=figure_out,
    )
    assert result["has_prevalence"] is True
    assert result["has_outcome_risk"] is True
    assert (figure_out / "publication_figure_prevalence_source_data.csv").exists()
    assert (figure_out / "publication_figure_absolute_risk_source_data.csv").exists()


def test_tidy_output_renders_dedicated_absolute_risk_publication_bundle(
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.pipeline import (
        _render_publication_bundle_from_prior_outputs_for_step,
    )

    cohort = pd.DataFrame(
        {
            "endpoint_flag": [0, 1, 1, 0, 1, 0, 1, 0],
            "organ_stage_max": [0, 1, 2, np.nan, np.nan, 1, 0, 2],
            "organ_stage_measured": [1, 1, 1, 0, 1, 0, 2, 1],
            "organ_stage_n": [1, 1, 1, 0, 1, 1, 1, 1],
            "marker_peak": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    context = {
        "primary_exposure": "organ_stage_max",
        "target_outcome": "endpoint_flag",
        "variables": [
            {
                "name": "endpoint_flag",
                "role": "outcome",
                "dtype": "int64",
                "is_ordinal": False,
                "observed_domain": {"is_binary": True},
            },
            {
                "name": "organ_stage_max",
                "role": "ordinal_score",
                "dtype": "float64",
                "is_ordinal": True,
                "ordinal_levels": [0, 1, 2],
            },
            {
                "name": "marker_peak",
                "role": "lab",
                "dtype": "float64",
                "is_ordinal": False,
            },
        ],
    }
    _summary, _out_dir = _execute_runner(
        tmp_path,
        cohort=cohort,
        context=context,
        inputs=[
            "endpoint_flag",
            "organ_stage_max",
            "organ_stage_measured",
            "organ_stage_n",
            "marker_peak",
        ],
    )
    figure_step = "06_absolute_risk_context_figure"
    figure_out = tmp_path / "steps" / figure_step / "outputs"

    repair_id = _render_publication_bundle_from_prior_outputs_for_step(
        run_dir=tmp_path,
        current_step_id=figure_step,
        out_dir=figure_out,
        step_text=(
            "Render figure:absolute_risk without redefining the cohort, "
            "exposure, or outcome."
        ),
    )

    assert repair_id == "absolute_risk_publication_bundle_from_parent_outputs_v1"
    for suffix in ("png", "svg", "pdf", "tiff"):
        assert (figure_out / f"publication_figure.{suffix}").exists()
    contract = json.loads(
        (figure_out / "publication_figure.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "descriptive_result",
        "descriptive_result",
        "descriptive_result",
    ]
    assert [panel["metadata"]["chart_type"] for panel in contract["panels"]] == [
        "dot_interval_prevalence",
        "dot_interval_absolute_risk",
        "median_iqr",
    ]
    assert "eligibility" not in contract["core_claim"].lower()
    source_validator = FigureSourceDataValidator()
    assert source_validator.audit(
        step=AnalysisStep(
            step_id=figure_step,
            intent="Render figure:absolute_risk from its direct parent.",
        ),
        out_dir=figure_out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    ) == []
    assert not [
        finding
        for finding in FigureContractQualityValidator().audit(
            step=AnalysisStep(
                step_id=figure_step,
                intent="Render figure:absolute_risk from its direct parent.",
            ),
            out_dir=figure_out,
            run_dir=tmp_path,
            step_summary={"rendering_only": True},
        )
        if finding.severity == "error"
    ]

    risk_source_path = figure_out / "absolute_risk_outcome_source_data.csv"
    tampered = pd.read_csv(risk_source_path)
    tampered.loc[0, "event_n"] = float(tampered.loc[0, "event_n"]) + 1
    tampered.to_csv(risk_source_path, index=False)
    assert any(
        finding.severity == "error"
        for finding in source_validator.audit(
            step=AnalysisStep(
                step_id=figure_step,
                intent="Render figure:absolute_risk from its direct parent.",
            ),
            out_dir=figure_out,
            run_dir=tmp_path,
            step_summary={"rendering_only": True},
        )
    )


def test_nonbinary_outcome_blocks_without_creating_a_risk_table(
    tmp_path: Path,
) -> None:
    cohort = pd.DataFrame(
        {
            "continuous_endpoint": [1.2, 2.4, 3.6],
            "exposure_value": [0.1, 0.2, 0.3],
        }
    )
    context = {
        "primary_exposure": "exposure_value",
        "target_outcome": "continuous_endpoint",
        "variables": [],
    }
    summary, out_dir = _execute_runner(
        tmp_path,
        cohort=cohort,
        context=context,
        inputs=["continuous_endpoint", "exposure_value"],
    )

    assert summary["status"] == "blocked"
    assert "binary 0/1" in summary["blocking_reason"]
    assert summary["output_files"] == {}
    assert not (out_dir / "exposure_outcome_summary.csv").exists()
