"""Publication layout must leave every scientific value and interval intact."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from easyicu.research_agent.contracts.figure_plan import FigurePresentationSpec  # noqa: E402
from easyicu.research_agent.execution.runners import landmark_association_figure_executor as owner  # noqa: E402
from easyicu.research_agent.figures.presentation import finish_presented_figure  # noqa: E402
from easyicu.research_agent.figures.publication import (  # noqa: E402
    PALETTE_CLINICAL,
    add_panel_label,
    save_publication_figure,
)
from easyicu.research_agent.schema import ResearchContext  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    with plt.rc_context():
        yield
        plt.close("all")


def _curve():
    return pd.DataFrame({
        "exposure": ["marker_mg_dl"] * 3,
        "exposure_value": [2.0, 4.0, 9.0],
        "reference_exposure_value": [3.0] * 3,
        "adjusted_odds_ratio": [1.4, 0.9, 0.7],
        "ci_low": [1.1, 0.8, 0.5], "ci_high": [1.7, 1.1, 0.9],
        "exposure_density_n": [2, 5, 3],
        "exposure_density_fraction": [0.2, 0.5, 0.3],
    })


def _forest(count=3):
    primary = _curve().iloc[:count].copy()
    sensitivity = primary.copy()
    for column in ("adjusted_odds_ratio", "ci_low", "ci_high"):
        sensitivity[column] *= 1.02
    fig, ax = plt.subplots(figsize=(2.2, 3.2))
    owner._draw_sensitivity_forest(
        ax, primary_curve=primary, exposure_column="exposure_value",
        reference_column="reference_exposure_value", sensitivity_contrasts=sensitivity,
        palette=PALETTE_CLINICAL,
    )
    return fig, ax, primary, sensitivity


def _assert_legend_clear(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = ax.get_legend().get_window_extent(renderer)
    for panel in fig.axes:
        assert not legend_box.overlaps(panel.get_window_extent(renderer))
        for line in panel.lines:
            assert not legend_box.overlaps(line.get_window_extent(renderer))
        for collection in panel.collections:
            # Errorbar LineCollections have no finite get_window_extent;
            # transform their data limits to test the actual interval bounds.
            bounds = collection.get_datalim(panel.transData).transformed(panel.transData)
            assert not legend_box.overlaps(bounds)
        for label in [panel.xaxis.label, panel.yaxis.label, *panel.texts]:
            if label.get_visible() and label.get_text():
                assert not legend_box.overlaps(label.get_window_extent(renderer))


def test_composite_forest_legend_cannot_cover_marks_or_intervals(tmp_path):
    fig, ax, _, _ = _forest(2)
    save_publication_figure(fig, tmp_path / "forest", formats=("png",), dpi=100)
    _assert_legend_clear(fig, ax)


@pytest.mark.parametrize("count", [2, 3])
def test_forest_pairs_are_centred_on_regular_rows_with_half_row_margins(count):
    _, ax, primary, sensitivity = _forest(count)
    ticks = ax.get_yticks()
    p, s = ax.containers
    np.testing.assert_allclose((p.lines[0].get_ydata() + s.lines[0].get_ydata()) / 2, ticks)
    np.testing.assert_allclose(np.diff(ticks), 1.0)
    assert ax.get_ylim() == pytest.approx((count - 0.5, -0.5))
    for container, source in ((p, primary), (s, sensitivity)):
        np.testing.assert_array_equal(container.lines[0].get_xdata(), source.adjusted_odds_ratio)
        segments = container.lines[2][0].get_segments()
        np.testing.assert_array_equal([segment[:, 0] for segment in segments], source[["ci_low", "ci_high"]])
    assert ax.get_xscale() == "log"


@pytest.mark.parametrize("coordinate", ["x", "y"])
@pytest.mark.parametrize("presented", [False, True])
def test_shared_export_keeps_all_interval_legends_outside_data(tmp_path, coordinate, presented):
    fig, axes = plt.subplots(2, 1, figsize=(3.2, 4.0), layout="constrained")
    for ax in axes:
        ax.errorbar([0.7, 1.3], [0.4, 0.9], **{f"{coordinate}err": [0.2, 0.1]}, label="Estimate")
        ax.set_xlabel("A labelled estimate axis")
        ax.legend(loc="upper right")
    if presented:
        finish_presented_figure(fig, FigurePresentationSpec(legend_location="upper right"))
    save_publication_figure(fig, tmp_path / "intervals", formats=("png",), dpi=100)
    for ax in axes:
        _assert_legend_clear(fig, ax)
    assert not axes[0].get_legend().get_window_extent().overlaps(axes[1].get_legend().get_window_extent())


@pytest.mark.parametrize(("field", "expected"), [
    ("map_min", "Minimum MAP (mmHg)"),
    ("sofa_max", "Maximum SOFA"),
    ("apache_max", "Maximum APACHE"),
    ("bun_min_mg_dl", "Minimum BUN (mg/dL)"),
    ("lact_max_mmol_l", "Maximum lactate (mmol/L)"),
])
def test_exposure_sentence_case_preserves_clinical_abbreviations(field, expected):
    assert owner._continuous_exposure_label(field) == expected


def test_distribution_uses_published_midpoint_bins_without_connecting_counts():
    fig, (ax, strip) = plt.subplots(2, 1, sharex=True)
    x = np.array([2.0, 4.0, 9.0])
    fractions = np.array([0.2, 0.5, 0.3])
    ax.plot(x, [0.3, 0.4, 0.5])
    owner._draw_exposure_distribution(
        strip, x=x, fractions=fractions, color="blue", exposure_label="Marker (mg/dL)",
    )
    assert not strip.lines
    assert len(strip.patches) == 3
    np.testing.assert_allclose([bar.get_x() for bar in strip.patches], [2.0, 3.0, 6.5])
    np.testing.assert_allclose([bar.get_width() for bar in strip.patches], [1.0, 3.5, 2.5])
    np.testing.assert_allclose([bar.get_height() for bar in strip.patches], 100 * fractions)
    assert strip.get_xlim() == ax.get_xlim() == (2.0, 9.0)
    fig.canvas.draw()
    np.testing.assert_allclose(strip.transData.transform([(2, 0), (9, 0)])[:, 0],
                               ax.transData.transform([(2, 0), (9, 0)])[:, 0])


@pytest.mark.parametrize("width", [2.0, 3.0])
def test_long_panel_titles_fit_above_axes_and_beside_panel_letters(tmp_path, width):
    fig, ax = plt.subplots(figsize=(width, 3.0))
    title = "Sensitivity to covariate functional form across prespecified contrasts"
    ax.set_title(title, loc="left", fontsize=9)
    ax.plot([0, 1], [0, 1])
    add_panel_label(ax, "c")
    save_publication_figure(fig, tmp_path / "title", formats=("png",), dpi=100)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = ax._left_title.get_window_extent(renderer)
    letter_box = ax.texts[0].get_window_extent(renderer)
    axis_box = ax.get_window_extent(renderer)
    assert " ".join(ax.get_title(loc="left").split()) == title
    assert title_box.width <= axis_box.width + 1
    assert title_box.y0 > axis_box.y1
    assert not title_box.overlaps(letter_box)


def _json_binding(root, name, payload):
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"relative_path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize(("label", "expected"), [
    ("ICU Readmission Within Seven Days", "ICU readmission within seven days"),
    ("In-hospital Mortality", "in-hospital mortality"),
])
def test_risk_axis_uses_the_bound_plan_outcome_without_changing_values(tmp_path, monkeypatch, label, expected):
    curve = _curve()
    risk = curve.rename(columns={"adjusted_odds_ratio": "adjusted_absolute_risk"}).copy()
    for column in ("adjusted_absolute_risk", "ci_low", "ci_high"):
        risk[column] /= 10
    frames = {"table:generic_landmark_rcs_curve": curve, "table:generic_adjusted_absolute_risk": risk}
    bindings = {}
    for key, frame in frames.items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        product = key.partition(":")[2]
        bindings[key] = {
            "declared_kind": "table", "evidence_kind": "table", "product": product,
            "relative_path": path.name, "sha256": digest, "evidence_id": product,
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
            "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
            "identity_row": {"input_key": key, "declared_kind": "table", "product": product,
                             "evidence_id": product, "sha256": digest},
        }
    context = ResearchContext(
        research_question="Synthetic marker association",
        cohort={"cohort_name": "Synthetic", "database": "synthetic", "n_stays": 10},
        target_outcome="endpoint", variables=[{
            "name": "endpoint", "description": "ICU readmission", "role": "outcome", "dtype": "bool",
        }],
    )
    manifest = {"step_id": "display", "inputs": bindings,
                "context": _json_binding(tmp_path, "context.json", context.model_dump(mode="json")),
                "plan": _json_binding(tmp_path, "plan.json", {"display_labels": {"endpoint": label}})}
    captured = []

    def capture(fig, *args, **kwargs):
        captured.append(fig)
        return {}

    monkeypatch.setattr(owner, "save_publication_figure", capture)
    owner.run_landmark_association_figure(
        out_dir=tmp_path / "out", run_dir=tmp_path, resolved_inputs=manifest,
        step_id="display", figure_product="display", input_keys=tuple(frames),
    )
    axis = captured[0].axes[1]
    assert axis.get_ylabel() == f"Model-standardised {expected} (%)"
    np.testing.assert_allclose(axis.lines[0].get_ydata(), risk.adjusted_absolute_risk * 100)


@pytest.mark.parametrize("changed", ["context", "plan", "step_id"])
def test_outcome_label_cannot_come_from_changed_bound_metadata(tmp_path, changed):
    from easyicu.research_agent.execution.runners.bound_variable_display import load_bound_outcome_label

    context = ResearchContext(
        research_question="Synthetic outcome",
        cohort={"cohort_name": "Synthetic", "database": "synthetic", "n_stays": 10},
        target_outcome="endpoint", variables=[{
            "name": "endpoint", "description": "ICU readmission", "role": "outcome", "dtype": "bool",
        }],
    )
    manifest = {"step_id": "display",
                "context": _json_binding(tmp_path, "context.json", context.model_dump(mode="json")),
                "plan": _json_binding(tmp_path, "plan.json", {"display_labels": {}})}
    assert load_bound_outcome_label(run_dir=tmp_path, resolved_inputs=manifest, step_id="display") == "ICU readmission"
    if changed == "step_id":
        manifest[changed] = "another_step"
    else:
        (tmp_path / manifest[changed]["relative_path"]).write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch|another step"):
        load_bound_outcome_label(run_dir=tmp_path, resolved_inputs=manifest, step_id="display")
