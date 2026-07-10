"""The deterministic association forest-plot rescue must tolerate the OR/CI
column-name variants that free-model code emits (e.g. ``ci_lower``/``ci_upper``
rather than ``or_ci_low``/``or_ci_high``). Without this, a figure-only step
fails the whole run even though the parent step computed a valid odds ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.audits.validators import (
    FigureContractQualityValidator,
    FigureSourceDataValidator,
)
from easyicu.research_agent.pipeline import (
    _context_axis_label,
    _render_cohort_overlap_publication_bundle_from_prior_outputs as cohort_overlap_rescue,
    _render_missingness_publication_bundle_from_prior_outputs as missingness_rescue,
    _render_publication_bundle_from_prior_outputs_for_step as routed_rescue,
    _render_association_publication_bundle_from_prior_outputs as rescue,
    _render_sensitivity_publication_bundle_from_prior_outputs as sensitivity_rescue,
    _resolve_upstream_analysis_family,
    deterministic_figure_family_supported,
    deterministic_figure_family_supported_for_upstream,
)
from easyicu.research_agent.schema import AnalysisStep


def _make_parent_step(run_dir: Path, csv_name: str, columns: dict) -> None:
    out = run_dir / "steps" / "03_association_model" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(out / csv_name, index=False)


def test_context_axis_label_wraps_metric_group_pairs():
    assert _context_axis_label("Death Risk", "Sepsis-3 Negative") == (
        "Sepsis-3 Negative\nDeath Risk"
    )
    assert _context_axis_label("Exposure prevalence", "Sepsis-3 prevalence") == (
        "Sepsis-3\nprevalence"
    )


def test_missingness_rescue_recomputes_percentages_from_counts(tmp_path: Path):
    parent = (
        tmp_path
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "concept": ["resp", "lact", "sep3_sofa2"],
            "label": ["Respiratory rate", "Lactate", "Sepsis-3 source flag"],
            "n_total": [74829, 74829, 74829],
            "value_missing_n": [188, 30490, 0],
            "value_missing_pct": [0.2512394927, 40.7462347486, 0.0],
            "measured_one_n": [74641, 44339, 28229],
            "measured_one_pct": [99.7487605073, 59.2537652514, 37.7246789346],
            "value_present_but_measured_zero_n": [0, 0, 46600],
        }
    ).to_csv(parent / "missingness_measurement_audit.csv", index=False)
    out = (
        tmp_path
        / "steps"
        / "02_baseline_characteristics_and_data_quality_figure"
        / "outputs"
    )

    rid = missingness_rescue(
        run_dir=tmp_path,
        current_step_id="02_baseline_characteristics_and_data_quality_figure",
        out_dir=out,
    )

    assert rid == "missingness_publication_bundle_from_parent_outputs_v1"
    source = pd.read_csv(out / "missingness_measurement_panel_source_data.csv")
    resp = source[source["variable"] == "resp"].iloc[0]
    source_flag = source[source["variable"] == "sep3_sofa2"].iloc[0]
    assert resp["missing_pct"] == pytest.approx(0.2512394927)
    assert resp["measured_pct"] == pytest.approx(99.7487605073)
    assert source_flag["measured_pct"] == pytest.approx(100.0)
    contract = json.loads(
        (out / "missingness_measurement_panel.figure_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(contract["panels"]) == 2


def test_e3_ordered_stage_figure_step_is_deterministically_claimed(tmp_path: Path):
    # E3 regression: the deterministic ordinal runner emits a perfect
    # dose_response.csv, but the planner named the primary figure step
    # ``04_primary_ordered_stage_analysis_figure`` — which matched NO token group
    # (``ordered`` != ``ordinal``), so the forest fell to the LLM coder and
    # crashed, leaving primary_pub_fig_contracts=0 and failing the run closed.
    # The gate must now claim it AND route it to the association forest renderer.
    step_id = "04_primary_ordered_stage_analysis_figure"
    assert deterministic_figure_family_supported(step_id) is True

    parent = tmp_path / "steps" / "04_primary_ordered_stage_analysis" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 19593, 3621],
            "n_events": [2143, 1380, 2672, 1188],
            "event_rate": [0.0572, 0.0981, 0.1364, 0.3281],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.587, 2.119, 5.766],
            "or_ci_low": [1.0, 1.477, 1.993, 5.289],
            "or_ci_high": [1.0, 1.705, 2.253, 6.287],
        }
    ).to_csv(parent / "dose_response.csv", index=False)
    out = tmp_path / "steps" / step_id / "outputs"

    rid = routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out)
    # A non-None id proves the deterministic association/ordinal renderer claimed
    # the step (instead of the crashing LLM coder) and emitted a figure bundle.
    assert rid is not None


def test_survival_by_stage_figure_still_routes_to_survival(tmp_path: Path):
    # Anti-regression for the "ordered" token addition: a survival figure step
    # that happens to mention "stage" must NOT be stolen by the association
    # renderer. It contains no "ordered" token and keeps matching survival.
    assert (
        deterministic_figure_family_supported("05_survival_by_disease_stage_figure")
        is True
    )


def _write_parent_summary(run_dir: Path, parent_id: str, family: str) -> None:
    out = run_dir / "steps" / parent_id / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(
        json.dumps({"step": parent_id, "status": "ok", "analysis_family": family}),
        encoding="utf-8",
    )


def test_upstream_family_routes_token_free_primary_figure(tmp_path: Path):
    # E3 (2026-07-08) real-run regression: the planner named the primary forest
    # figure ``05_primary_stage_outcome_analysis_figure`` — it matches NO family
    # token (no association/ordinal/ordered/trend/gradient), so id-token routing
    # returned None and the forest fell to the LLM coder, which failed. But its
    # PARENT analysis step recorded analysis_family='association' and produced a
    # canonical dose_response.csv. Routing by the parent's PROVEN family must
    # claim the step and render the forest deterministically.
    step_id = "05_primary_stage_outcome_analysis_figure"
    parent_id = "05_primary_stage_outcome_analysis"

    # The id itself carries no family token -> the token gate is False ...
    assert deterministic_figure_family_supported(step_id) is False

    _write_parent_summary(tmp_path, parent_id, "association")
    parent = tmp_path / "steps" / parent_id / "outputs"
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 19593, 3621],
            "n_events": [2143, 1380, 2672, 1188],
            "event_rate": [0.0572, 0.0981, 0.1364, 0.3281],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.587, 2.119, 5.766],
            "or_ci_low": [1.0, 1.477, 1.993, 5.289],
            "or_ci_high": [1.0, 1.705, 2.253, 6.287],
        }
    ).to_csv(parent / "dose_response.csv", index=False)

    # ... but the parent-family fallback recognises + routes it.
    assert _resolve_upstream_analysis_family(tmp_path, step_id) == "association"
    assert deterministic_figure_family_supported_for_upstream(tmp_path, step_id) is True

    out = tmp_path / "steps" / step_id / "outputs"
    rid = routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out)
    assert rid is not None  # association forest renderer claimed + drew it


def test_descriptive_parent_supported_but_guarded_against_empty_figure(tmp_path: Path):
    # A descriptive/table-one renderer now EXISTS (deterministic descriptive
    # bundle), so a 'descriptive' parent is recognised by the family map. The old
    # "no empty figure" safety is preserved by the renderer's STRICT guard: with no
    # genuine table-one output present, routed_rescue returns None and the figure
    # falls through to its existing path rather than being force-drawn empty.
    step_id = "03_baseline_context_figure"
    _write_parent_summary(tmp_path, "03_baseline_context", "descriptive")
    assert _resolve_upstream_analysis_family(tmp_path, step_id) == "descriptive"
    assert (
        deterministic_figure_family_supported_for_upstream(tmp_path, step_id) is True
    )
    # No table-one output under the parent -> the strict guard declines (None).
    out = tmp_path / "steps" / step_id / "outputs"
    assert routed_rescue(run_dir=tmp_path, current_step_id=step_id, out_dir=out) is None
    # No parent summary at all -> also unsupported (no crash).
    assert (
        deterministic_figure_family_supported_for_upstream(tmp_path, "99_x_figure")
        is False
    )


def test_rescue_handles_ci_lower_upper_variant(tmp_path: Path):
    # free-model style column names: odds_ratio + ci_lower/ci_upper
    _make_parent_step(
        tmp_path,
        "adjusted_odds_ratios.csv",
        {
            "variable": ["const", "sepsis3", "age"],
            "odds_ratio": [0.01, 0.80, 1.03],
            "ci_lower": [0.0, 0.74, 1.01],
            "ci_upper": [0.1, 0.86, 1.05],
        },
    )
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(
        run_dir=tmp_path, current_step_id="03_association_model_figure", out_dir=out
    )
    assert rid is not None
    figs = {p.suffix for p in out.iterdir()}
    assert ".png" in figs and ".svg" in figs
    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B"]
    assert (out / "publication_figure_source_data.csv").exists()
    findings = FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    )
    assert not any(f.severity == "error" for f in findings), findings
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []

def test_rescue_handles_canonical_or_ci_columns(tmp_path: Path):
    # our deterministic fallback style: or_ci_low/or_ci_high
    _make_parent_step(
        tmp_path,
        "association_results.csv",
        {
            "variable": ["sepsis3"],
            "odds_ratio": [0.80],
            "or_ci_low": [0.74],
            "or_ci_high": [0.86],
        },
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rid = rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out)
    assert rid is not None


def test_rescue_promotes_prevalence_and_absolute_risk_context(tmp_path: Path):
    parent = tmp_path / "steps" / "03_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variable": ["const", "exposed", "age"],
            "odds_ratio": [0.10, 1.20, 1.04],
            "ci_lower": [0.02, 1.05, 1.02],
            "ci_upper": [0.50, 1.40, 1.06],
        }
    ).to_csv(parent / "adjusted_odds_ratios.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["exposed"],
            "definition": ["binary exposure"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence": [0.32],
            "prevalence_pct": [32.0],
            "ci_low": [0.291],
            "ci_high": [0.350],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "exposure_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "exposure_label": ["Exposure negative", "Exposure positive"],
            "n": [680, 320],
            "event_n": [61, 48],
            "outcome_risk": [0.0897, 0.1500],
            "outcome_risk_pct": [8.97, 15.0],
            "ci_low": [0.071, 0.115],
            "ci_high": [0.111, 0.193],
            "ci_low_pct": [7.1, 11.5],
            "ci_high_pct": [11.1, 19.3],
        }
    ).to_csv(parent / "outcome_by_exposure.csv", index=False)
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = rescue(
        run_dir=tmp_path,
        current_step_id="03_association_model_figure",
        out_dir=out,
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    contract = json.loads(
        (out / "publication_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "descriptive_result",
        "primary_estimand",
    ]
    assert contract["panels"][0]["metadata"]["chart_type"] == "dot_interval_absolute_risk"
    assert (out / "publication_figure_prevalence_source_data.csv").exists()
    assert (out / "publication_figure_absolute_risk_source_data.csv").exists()
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_rescue_uses_primary_summary_and_semantic_binary_risk_labels(tmp_path: Path):
    parent = tmp_path / "steps" / "03_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": ["primary_adjusted"],
            "outcome": ["death"],
            "exposure": ["treated"],
            "effect_scale": ["odds_ratio"],
            "point_estimate": [1.18],
            "ci_low": [1.04],
            "ci_high": [1.34],
        }
    ).to_csv(parent / "adjusted_association_death.csv", index=False)
    pd.DataFrame(
        {
            "term": ["treated", "age", "lab_missing"],
            "odds_ratio": [1.18, 1.03, 2.10],
            "ci_lower": [1.04, 1.01, 0.80],
            "ci_upper": [1.34, 1.05, 5.50],
        }
    ).to_csv(parent / "adjusted_association_death_full_coefficients.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["treated"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence_pct": [32.0],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "exposure_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "treated": [0, 1, "risk_difference_1_minus_0"],
            "n_total": [680, 320, 1000],
            "death_events": [61, 48, 109],
            "death_risk": [0.0897, 0.1500, 0.0603],
            "death_risk_ci_low": [0.071, 0.115, None],
            "death_risk_ci_high": [0.111, 0.193, None],
        }
    ).to_csv(parent / "outcome_by_exposure.csv", index=False)
    out = tmp_path / "steps" / "03_association_model_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = rescue(
        run_dir=tmp_path,
        current_step_id="03_association_model_figure",
        out_dir=out,
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    source = pd.read_csv(out / "publication_figure_source_data.csv")
    assert source["source_table"].tolist() == ["adjusted_association_death.csv"]
    assert source["exposure"].tolist() == ["treated"]
    absolute = pd.read_csv(out / "publication_figure_absolute_risk_source_data.csv")
    assert absolute["plot_group_label"].tolist() == [
        "Treated Negative",
        "Treated Positive",
    ]
    assert absolute["plot_ci_low_pct"].tolist() == pytest.approx([7.1, 11.5])
    contract = json.loads(
        (out / "publication_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    assert contract["panels"][1]["title"] == "Primary adjusted association"
    assert contract["panels"][1]["metadata"]["chart_type"] == "dot_interval"
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="03_association_model_figure",
            intent="Render the publication figure declared by step '03_association_model'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_routed_rescue_prioritizes_primary_association_over_missingness(
    tmp_path: Path,
):
    parent = (
        tmp_path
        / "steps"
        / "03_primary_prevalence_and_adjusted_association"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "term": ["const", "exposed"],
            "effect_scale": ["odds_ratio", "odds_ratio"],
            "estimate": [0.10, 1.20],
            "ci_low": [0.02, 1.05],
            "ci_high": [0.50, 1.40],
        }
    ).to_csv(parent / "adjusted_association_death.csv", index=False)
    pd.DataFrame(
        {
            "exposure": ["exposed"],
            "n_denominator": [1000],
            "n_positive": [320],
            "prevalence_pct": [32.0],
            "ci_low_pct": [29.1],
            "ci_high_pct": [35.0],
        }
    ).to_csv(parent / "sepsis3_prevalence.csv", index=False)
    pd.DataFrame(
        {
            "sepsis3_label": ["Exposure negative", "Exposure positive"],
            "n": [680, 320],
            "death_n": [61, 48],
            "death_risk_pct": [8.97, 15.0],
            "ci_low_pct": [7.1, 11.5],
            "ci_high_pct": [11.1, 19.3],
        }
    ).to_csv(parent / "outcome_by_sepsis3.csv", index=False)

    missingness_parent = (
        tmp_path
        / "steps"
        / "02_baseline_characteristics_and_data_quality"
        / "outputs"
    )
    missingness_parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "concept": ["lactate"],
            "label": ["Lactate"],
            "n_total": [1000],
            "value_missing_n": [400],
            "value_missing_pct": [40.0],
            "measured_one_n": [600],
            "measured_one_pct": [60.0],
        }
    ).to_csv(missingness_parent / "missingness_measurement_audit.csv", index=False)

    out = (
        tmp_path
        / "steps"
        / "03_primary_prevalence_and_adjusted_association_figure"
        / "outputs"
    )

    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="03_primary_prevalence_and_adjusted_association_figure",
        out_dir=out,
        step_text="Render primary result figure with missingness/data-quality context.",
    )

    assert rid == "association_publication_bundle_from_parent_outputs_v3"
    assert (out / "publication_figure.png").exists()
    assert not (out / "missingness_measurement_panel.png").exists()


def test_rescue_returns_none_without_or_ci_table(tmp_path: Path):
    _make_parent_step(
        tmp_path, "prevalence.csv", {"group": ["a"], "rate": [0.3]}
    )
    out = tmp_path / "steps" / "03_fig" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    assert rescue(run_dir=tmp_path, current_step_id="03_fig", out_dir=out) is None


# --- ordinal dose-response figure steps must reach the deterministic renderer ---
# E3 regression: when the LLM names its primary figure step "..._stage_gradient_
# analysis_figure" / "..._dose_response_figure" (instead of "...association...")
# the deterministic figure family/router did not recognise it, so the step fell
# through to LLM code that produced a corrupted source_data table (ci_low filled
# with the cohort count) which the figure-trace gate then rejected. The ordinal
# dose-response family is an association forest and must route to the association
# bundle renderer, which reads dose_response.csv and emits stage-keyed source data.


@pytest.mark.parametrize(
    "step_id",
    [
        "04_primary_stage_gradient_analysis_figure",
        "04_primary_dose_response_figure",
        "04_ordinal_trend_analysis_figure",
    ],
)
def test_ordinal_figure_step_is_family_supported(step_id: str):
    assert deterministic_figure_family_supported(step_id) is True


def test_graded_exposure_forest_keys_by_varying_level_not_constant_model(
    tmp_path: Path,
):
    # M1 regression: a single graded exposure keeps exposure_variable/model
    # CONSTANT across rows and varies by ordinal `level`. The renderer must label
    # and key rows by the varying `level`, not collapse every row to the constant
    # column (which drops the per-row trace key -> "no shared key").
    parent = tmp_path / "steps" / "04_primary_association_model" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "model": ["adjusted"] * 5,
            "exposure_variable": ["sofa2_liver_cat"] * 5,
            "level": [0, 1, 2, 3, 4],
            "odds_ratio": [1.0, 1.2122, 1.3638, 1.6002, 3.9035],
            "ci_low": [1.0, 1.1075, 1.1985, 1.3627, 3.3520],
            "ci_high": [1.0, 1.3269, 1.5520, 1.8791, 4.5458],
        }
    ).to_csv(parent / "primary_adjusted_odds_ratios.csv", index=False)

    out = tmp_path / "steps" / "04_primary_association_model_figure" / "outputs"
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="04_primary_association_model_figure",
        out_dir=out,
        step_text="Adjusted odds ratio per SOFA-2 liver category level.",
    )
    assert rid is not None
    src = pd.read_csv(out / "publication_figure_source_data.csv")
    # keyed by the varying level, 5 distinct rows (not collapsed to one label)
    assert "level" in src.columns
    assert src["level"].nunique() == 5
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=src,
        source_path=out / "publication_figure_source_data.csv",
        upstream_path=parent / "primary_adjusted_odds_ratios.csv",
    )
    assert res.get("ok") is True, res
    assert res.get("key_column") == "level", res


def test_ordinal_stage_gradient_figure_routes_to_association_renderer(tmp_path: Path):
    parent = (
        tmp_path / "steps" / "04_primary_stage_gradient_analysis" / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    # the deterministic ordinal runner's canonical dose_response.csv shape
    pd.DataFrame(
        {
            "stage": [0, 1, 2, 3],
            "n": [37433, 14061, 5200, 2100],
            "n_events": [2143, 1380, 780, 500],
            "event_rate": [0.0572, 0.0981, 0.150, 0.238],
            "is_reference": [True, False, False, False],
            "odds_ratio": [1.0, 1.5871617453700098, 2.51, 4.02],
            "or_ci_low": [1.0, 1.4771205, 2.30, 3.60],
            "or_ci_high": [1.0, 1.7054007, 2.74, 4.49],
            "or_p_value": [None, 2.08e-36, 1e-40, 1e-50],
        }
    ).to_csv(parent / "dose_response.csv", index=False)

    out = (
        tmp_path / "steps" / "04_primary_stage_gradient_analysis_figure" / "outputs"
    )
    rid = routed_rescue(
        run_dir=tmp_path,
        current_step_id="04_primary_stage_gradient_analysis_figure",
        out_dir=out,
        step_text="Render the adjusted odds-ratio gradient per KDIGO stage.",
    )
    # routed to a real deterministic renderer (NOT None -> not LLM-coded fallback)
    assert rid is not None, "ordinal figure fell through to LLM code"
    assert (out / "publication_figure.png").exists()
    # and the emitted source data traces to dose_response.csv on the `stage` key
    src = pd.read_csv(out / "publication_figure_source_data.csv")
    res = FigureSourceDataValidator._compare_source_to_upstream(
        source_df=src,
        source_path=out / "publication_figure_source_data.csv",
        upstream_path=parent / "dose_response.csv",
    )
    assert res.get("ok") is True, res


def test_cohort_overlap_rescue_writes_traceable_multipanel_bundle(tmp_path: Path):
    parent = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "definition_id": ["primary", "relax_temp", "tight_los"],
            "definition_label": [
                "Primary cohort",
                "Relax temperature requirement",
                "Tighten ICU length-of-stay threshold",
            ],
            "definition_type": ["primary", "alternative", "alternative"],
            "criteria": ["primary", "no temp", "los >=2"],
            "n_included": [100, 112, 72],
            "n_excluded": [50, 38, 78],
            "included_pct_of_rows": [66.7, 74.7, 48.0],
            "overlap_with_primary_n": [100, 100, 72],
            "overlap_with_primary_pct_of_primary": [100.0, 100.0, 72.0],
            "overlap_with_primary_pct_of_definition": [100.0, 89.3, 100.0],
            "moved_in_vs_primary_n": [0, 12, 0],
            "moved_out_vs_primary_n": [0, 0, 28],
        }
    ).to_csv(parent / "alternative_cohort_attrition.csv", index=False)
    rows = []
    sizes = {"primary": 100, "relax_temp": 112, "tight_los": 72}
    intersections = {
        ("primary", "primary"): 100,
        ("primary", "relax_temp"): 100,
        ("primary", "tight_los"): 72,
        ("relax_temp", "primary"): 100,
        ("relax_temp", "relax_temp"): 112,
        ("relax_temp", "tight_los"): 72,
        ("tight_los", "primary"): 72,
        ("tight_los", "relax_temp"): 72,
        ("tight_los", "tight_los"): 72,
    }
    for definition_a, n_a in sizes.items():
        for definition_b, n_b in sizes.items():
            intersection = intersections[(definition_a, definition_b)]
            union = n_a + n_b - intersection
            rows.append(
                {
                    "definition_a": definition_a,
                    "definition_b": definition_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "intersection_n": intersection,
                    "union_n": union,
                    "jaccard": intersection / union,
                    "a_in_b_pct": intersection / n_a * 100,
                    "b_in_a_pct": intersection / n_b * 100,
                }
            )
    pd.DataFrame(rows).to_csv(parent / "cohort_overlap_matrix.csv", index=False)

    out = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    out.mkdir(parents=True, exist_ok=True)

    rid = cohort_overlap_rescue(
        run_dir=tmp_path,
        current_step_id="04_alternative_eligibility_definitions_and_overlap_figure",
        out_dir=out,
    )

    assert rid == "cohort_overlap_publication_bundle_from_parent_outputs_v1"
    assert (out / "publication_figure.png").exists()
    assert (out / "publication_figure.svg").exists()
    assert (out / "publication_figure_definition_source_data.csv").exists()
    assert (out / "publication_figure_overlap_source_data.csv").exists()
    contract_path = out / "publication_figure.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    ) == []
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            intent=(
                "Render the publication figure declared by step "
                "'04_alternative_eligibility_definitions_and_overlap'."
            ),
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []


def test_cohort_overlap_rescue_shortens_sepsis3_derivable_definition_labels(
    tmp_path: Path,
):
    parent = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap"
        / "outputs"
    )
    parent.mkdir(parents=True, exist_ok=True)
    ids = [
        "primary_adult_los1_all_vitals_sepsis3_derivable",
        "alt_adult_no_los_all_vitals_sepsis3_derivable",
        "alt_adult_los1_three_of_four_vitals_sepsis3_derivable",
        "alt_adult_los1_no_temp_requirement_sepsis3_derivable",
        "alt_adult_los2_all_vitals_sepsis3_derivable",
    ]
    pd.DataFrame(
        {
            "definition_id": ids,
            "definition_label": [
                "Primary cohort",
                "Relax ICU length-of-stay threshold",
                "Relax vital completeness to >=3 of 4",
                "Relax temperature requirement",
                "Tighten ICU length-of-stay threshold",
            ],
            "definition_type": ["primary", "alternative", "alternative", "alternative", "alternative"],
            "n_included": [100, 100, 112, 111, 70],
            "n_excluded": [20, 20, 8, 9, 50],
            "included_pct_of_rows": [83.3, 83.3, 93.3, 92.5, 58.3],
            "overlap_with_primary_n": [100, 100, 100, 100, 70],
            "overlap_with_primary_pct_of_primary": [100, 100, 100, 100, 70],
            "overlap_with_primary_pct_of_definition": [100, 100, 89.3, 90.1, 100],
            "moved_in_vs_primary_n": [0, 0, 12, 11, 0],
            "moved_out_vs_primary_n": [0, 0, 0, 0, 30],
        }
    ).to_csv(parent / "alternative_cohort_attrition.csv", index=False)
    rows = []
    for a in ids:
        for b in ids:
            rows.append(
                {
                    "definition_a": a,
                    "definition_b": b,
                    "n_a": 100,
                    "n_b": 100,
                    "intersection_n": 100 if a == b else 80,
                    "union_n": 100 if a == b else 120,
                    "jaccard": 1.0 if a == b else 2 / 3,
                }
            )
    pd.DataFrame(rows).to_csv(parent / "cohort_overlap_matrix.csv", index=False)
    out = (
        tmp_path
        / "steps"
        / "04_alternative_eligibility_definitions_and_overlap_figure"
        / "outputs"
    )
    out.mkdir(parents=True, exist_ok=True)

    assert (
        cohort_overlap_rescue(
            run_dir=tmp_path,
            current_step_id="04_alternative_eligibility_definitions_and_overlap_figure",
            out_dir=out,
        )
        == "cohort_overlap_publication_bundle_from_parent_outputs_v1"
    )
    source = pd.read_csv(out / "publication_figure_definition_source_data.csv")
    assert source["display_label"].tolist() == [
        "Primary",
        "No LOS threshold",
        ">=3 of 4 vitals",
        "No temperature",
        "LOS >=2 d",
    ]


def test_sensitivity_rescue_writes_multipanel_contract_and_source_data(
    tmp_path: Path,
):
    parent = tmp_path / "steps" / "05_sensitivity_comparison" / "outputs"
    parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spec_id": ["primary", "alt_cohort", "risk_difference"],
            "axis": ["cohort", "cohort", "outcome"],
            "display_label": ["Primary cohort", "Alternative cohort", "Risk difference"],
            "effect_scale": ["OR", "OR", "RD"],
            "point_estimate": [1.12, 1.05, 0.03],
            "ci_low": [1.02, 0.95, 0.01],
            "ci_high": [1.24, 1.17, 0.05],
            "se": [0.05, 0.06, 0.01],
            "modeled_analytic_n": [1000, 920, 1000],
            "converged": [True, True, True],
        }
    ).to_csv(parent / "sensitivity_comparison.csv", index=False)
    out = tmp_path / "steps" / "05_sensitivity_comparison_figure" / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rid = sensitivity_rescue(
        run_dir=tmp_path,
        current_step_id="05_sensitivity_comparison_across_definitions_figure",
        out_dir=out,
    )

    assert rid == "sensitivity_publication_bundle_from_parent_outputs_v1"
    contract_path = out / "sensitivity_forest.figure_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [panel["panel_id"] for panel in contract["panels"]] == ["A", "B", "C"]
    assert (out / "sensitivity_forest_source_data.csv").exists()
    assert FigureContractQualityValidator().audit_contract_file(
        contract_path,
        manuscript_facing=True,
    ) == []
    source_findings = FigureSourceDataValidator().audit(
        step=AnalysisStep(
            step_id="05_sensitivity_comparison_figure",
            intent="Render the sensitivity figure declared by step '05_sensitivity_comparison'.",
        ),
        out_dir=out,
        run_dir=tmp_path,
        step_summary={"rendering_only": True},
    )
    assert source_findings == []

    routed_out = (
        tmp_path
        / "steps"
        / "05_sensitivity_comparison_across_definitions_figure_routed"
        / "outputs"
    )
    routed_out.mkdir(parents=True, exist_ok=True)
    routed_id = routed_rescue(
        run_dir=tmp_path,
        current_step_id="05_sensitivity_comparison_across_definitions_figure",
        out_dir=routed_out,
    )
    assert routed_id == "sensitivity_publication_bundle_from_parent_outputs_v1"
    assert (routed_out / "sensitivity_forest_source_data.csv").exists()
