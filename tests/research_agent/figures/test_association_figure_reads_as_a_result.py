"""The association publication figure reads as a result, not as an audit.

Observed risk is a point with its interval per exposure level, set apart from
the measured total and carrying events/n; a ratio forest uses a log axis and
prints each estimate; contrasts keep their plain "vs" text and the axes name
the exposure and outcome.  Audit-only status strips do not get the height of a
result panel.  The study here is a lactate tertile and ICU readmission.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import easyicu.research_agent as ra  # noqa: E402
from easyicu.research_agent.contracts.figure_plan import (  # noqa: E402
    ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
)
from easyicu.research_agent.execution.runners.association_publication_figure_renderer import (  # noqa: E402
    _absolute_risk_context,
    _draw_absolute_risk_points,
    _forest,
    _label,
)
from easyicu.research_agent.execution.runners.composite_descriptive_figure_executor import (  # noqa: E402
    run_composite_descriptive_figure,
)

_CJK = re.compile(r"[㐀-鿿]")


def _risk_rows() -> list[dict]:
    rows = []
    for group_type, value, n, events, share in (
        ("availability", "observed", 412, 37, 1.0),
        ("exposure_level", "1", 140, 8, 140 / 412),
        ("exposure_level", "2", 136, 12, 136 / 412),
        ("exposure_level", "3", 136, 17, 136 / 412),
    ):
        risk = events / n
        rows.append({
            "exposure": "lactate_tertile", "group_type": group_type,
            "group_value": value, "label": f"lactate_tertile = {value}",
            "estimate_type": "prevalence", "n": n, "event_n": None,
            "estimate": share, "ci_low": max(share - 0.05, 0.0),
            "ci_high": min(share + 0.05, 1.0),
        })
        rows.append({
            "exposure": "lactate_tertile", "group_type": group_type,
            "group_value": value, "label": f"lactate_tertile = {value}",
            "estimate_type": "outcome_risk", "n": n, "event_n": events,
            "estimate": risk, "ci_low": risk * 0.6, "ci_high": risk * 1.5,
        })
    return rows


def _frames() -> dict[str, pd.DataFrame]:
    return {
        "table:absolute_risk_context": pd.DataFrame(_risk_rows()),
        "table:adjusted_association_estimates": pd.DataFrame({
            "fit_status": ["fitted", "fitted"],
            "estimate": [1.4, 2.2],
            "ci_low": [1.1, 1.5],
            "ci_high": [1.8, 3.1],
            "effect_scale": ["odds_ratio", "odds_ratio"],
            "model_id": ["primary_adjusted", "primary_adjusted"],
            "contrast": ["2 vs 1", "3 vs 1"],
            "exposure": ["lactate_tertile", "lactate_tertile"],
            "outcome": ["icu_readmission", "icu_readmission"],
            "exposure_level": [2, 3],
            "reference_level": [1, 1],
        }),
        "table:robustness_matrix": pd.DataFrame({
            "spec_id": ["primary", "complete_case"],
            "point_estimate": [2.2, 2.1],
            "ci_low": [1.5, 1.4],
            "ci_high": [3.1, 3.0],
            "effect_scale": ["OR", "OR"],
            "converged": [True, True],
        }),
        "table:robustness_summary": pd.DataFrame({
            "axis": ["primary", "missing"],
            "total_specs": [1, 1],
            "converged_specs": [1, 1],
            "non_independent_specs": [0, 0],
            "range_low": [1.5, 1.4],
            "range_high": [3.1, 3.0],
        }),
    }


def _binding(key: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": path.name,
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key, "mode": "all_rows", "artifact_sha256": digest,
        },
        "identity_row": {
            "input_key": key, "declared_kind": "table", "product": product,
            "evidence_id": f"evidence_{product}", "sha256": digest,
        },
    }


def _render(tmp_path: Path) -> Path:
    bindings = {}
    for key, frame in _frames().items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
    context = ra.ResearchContext(
        research_question="Is the lactate tertile associated with ICU readmission?",
        cohort=ra.CohortDescriptor(
            cohort_name="Adult ICU stays", database="mimiciv", n_patients=380, n_stays=412,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="icu_readmission", description="ICU readmission",
                role="outcome", dtype="bool", source_concept="readmission",
            ),
            ra.ConceptDescriptor(
                name="lactate_tertile", description="lactate tertile",
                role="lab", dtype="int64", source_concept="lact",
            ),
        ],
        primary_exposure="lactate_tertile",
        target_outcome="icu_readmission",
    )
    context_path = tmp_path / "research_context.json"
    context_path.write_text(context.model_dump_json(), encoding="utf-8")
    out_dir = tmp_path / "outputs"
    run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs={
            "step_id": "readmission_figure",
            "inputs": bindings,
            "context": {
                "relative_path": context_path.name,
                "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            },
        },
        step_id="readmission_figure",
        figure_product="readmission_figure",
        input_keys=ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS,
        display_labels={"lactate_tertile": "乳酸三分位", "icu_readmission": "ICU再入院"},
    )
    return out_dir


def test_observed_risk_is_points_with_intervals_and_events_per_group() -> None:
    frame = _absolute_risk_context(pd.DataFrame(_risk_rows()))
    fig, ax = plt.subplots()
    try:
        title = _draw_absolute_risk_points(
            ax, frame, color="#E28E2C", neutral="#6F6F6F", level_label=str,
        )
        assert title == "Observed risk by exposure level"
        assert not ax.patches  # no bars
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "All measured\n37/412", "1\n8/140", "2\n12/136", "3\n17/136",
        ]
    finally:
        plt.close(fig)


def test_a_ratio_forest_is_logarithmic_and_prints_each_estimate() -> None:
    frame = _frames()["table:adjusted_association_estimates"]
    fig, ax = plt.subplots()
    try:
        _forest(
            ax, frame, estimate_column="estimate", label_column="contrast",
            title="Primary adjusted association", color="#0F4D92",
            label_formatter=_label,
        )
        assert ax.get_xscale() == "log"
        assert ax.get_xlabel() == "Odds ratio (95% CI, log scale)"
        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "2 vs 1", "3 vs 1",
        ]
        printed = {text.get_text() for text in ax.texts}
        assert {"1.40 (1.10–1.80)", "2.20 (1.50–3.10)"} <= printed
    finally:
        plt.close(fig)


def test_the_rendered_figure_names_exposure_and_outcome_in_english(tmp_path: Path) -> None:
    out_dir = _render(tmp_path)
    svg = (out_dir / "readmission_figure.svg").read_text(encoding="utf-8")
    contract = json.loads(
        (out_dir / "readmission_figure.figure_contract.json").read_text(encoding="utf-8")
    )

    for text in (
        "Lactate tertile", "ICU readmission (%)", "All measured", "37/412",
        "2 vs 1", "3 vs 1", "Odds ratio (95% CI, log scale)", "Complete case",
    ):
        assert text in svg, text
    # The exposure names both the risk panel's groups and the forest's rows.
    assert svg.count("Lactate tertile") >= 2
    assert not _CJK.search(svg)
    assert " Vs " not in svg
    assert contract["panels"][0]["title"] == "Observed risk by exposure level"
    assert "events/n" in contract["reader_caption"]


def test_audit_status_strips_do_not_take_a_result_panel_height(tmp_path: Path) -> None:
    out_dir = _render(tmp_path)
    height, width = mpimg.imread(out_dir / "readmission_figure.png").shape[:2]

    assert height / width < 0.85
