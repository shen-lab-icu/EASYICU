"""Synthetic recorded-resolution distributions, independent of study fixtures."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.execution.runners import landmark_spline_executor as producer
from easyicu.research_agent.figures import publication
from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
from easyicu.webserver.scientific_runtime_projection import compile_landmark_spline_runtime_projection


def _legacy_edges(grid):
    return np.concatenate(([grid[0]], (grid[:-1] + grid[1:]) / 2, [grid[-1]]))


def _synthetic_primary(tmp_path):
    rng = np.random.default_rng(726)
    values = np.tile(np.arange(1., 22.), 30)
    n = len(values)
    frame = pd.DataFrame({
        "marker": values, "event": rng.binomial(1, .3, n),
        "event_time": 72., "followup": 120., "age": rng.uniform(25, 85, n),
    })
    universe = tmp_path / "synthetic.parquet"
    frame.to_parquet(universe, index=False)
    specs = (
        PrespecifiedSensitivitySpec.model_validate({
            "spec_id": "timing", "axis": "timing", "strategy": "landmark",
            "landmark_hours": 24, "require_alive_at_landmark": True,
            "exclude_negative_event_times": True, "event_time_variable": "event_time",
            "observation_duration_variable": "followup", "observation_duration_unit": "hours",
        }),
        PrespecifiedSensitivitySpec.model_validate({
            "spec_id": "shape", "axis": "functional_form",
            "strategy": "restricted_cubic_spline", "execution_variables": ["marker"],
        }),
    )
    projection = compile_landmark_spline_runtime_projection(
        study={"covariate_selection": "exact"}, sensitivity_specs=specs,
        primary_exposure="marker", primary_exposure_source="marker", target_outcome="event",
        declared_covariates=("age",), covariate_operationalizations={}, target_is_event_status=True,
        universe_path=universe, scientific_configuration_sha256="d" * 64,
    )
    assert projection is not None
    return frame, projection


def test_published_integer_distribution_has_no_impossible_bins(tmp_path):
    frame, projection = _synthetic_primary(tmp_path)
    result = producer.run_landmark_spline_association(
        frame=frame, authority=projection.authority,
        runtime_projection_sha256=projection.projection_sha256, out_dir=tmp_path / "out",
    )
    curve = pd.read_csv(tmp_path / "out" / result["output_files"]["table:landmark_rcs_curve"])
    # On the read-only base this reads the actual comb-producing grid counts.
    if "exposure_distribution_json" in curve:
        counts = np.asarray(json.loads(curve.exposure_distribution_json.iloc[0])["counts"])
    else:
        counts = curve.exposure_density_n.to_numpy()
    assert np.all(counts[1:-1] > 0), counts.tolist()


@pytest.mark.parametrize("values,step", [
    (np.repeat(np.arange(-8., 13.), 5), 1.),
    (np.repeat(np.arange(41) * .5 - 3.25, 5), .5),
    (np.repeat(np.arange(41) * .3 + .17, 5), .3),
    (np.r_[np.arange(21.), np.repeat(10., 1000)], 1.),
    (np.arange(1001.), 1.),
])
def test_lattice_bins_have_half_step_edges_and_integral_widths(values, step):
    grid = np.linspace(values.min(), values.max(), 31)
    edges = publication.exposure_distribution_bin_edges(values, grid=grid)
    counts = np.histogram(values, bins=edges)[0]
    assert np.all(counts > 0)
    assert counts.sum() == len(values)
    assert len(counts) <= len(grid)
    coordinates = (edges - values.min()) / step + .5
    np.testing.assert_allclose(coordinates, np.round(coordinates), atol=1e-7, rtol=0)
    widths = np.diff(edges) / step
    np.testing.assert_allclose(widths, np.round(widths), atol=1e-7, rtol=0)
    assert np.all(widths >= 1 - 1e-7)
    np.testing.assert_allclose(widths, widths[0], atol=1e-7, rtol=0)


@pytest.mark.parametrize("values", [
    np.random.default_rng(9).normal(size=1000),
    np.r_[np.repeat(.2, 500), .61, 1.39, 2.78, 3.06],
    np.r_[np.arange(20.), 4.12345],
])
def test_continuous_values_keep_exact_existing_edges_and_counts(values):
    grid = np.linspace(values.min(), values.max(), 101)
    edges = publication.exposure_distribution_bin_edges(values, grid=grid)
    np.testing.assert_array_equal(edges, _legacy_edges(grid))
    np.testing.assert_array_equal(np.histogram(values, bins=edges)[0],
                                  np.histogram(values, bins=_legacy_edges(grid))[0])


def test_single_repeated_value_uses_one_bin_without_inventing_resolution():
    values = np.repeat(4.25, 500)
    edges = publication.exposure_distribution_bin_edges(values, grid=np.linspace(4., 5., 101))
    np.testing.assert_array_equal(edges, [4., 5.])
    np.testing.assert_array_equal(np.histogram(values, bins=edges)[0], [500])


def test_real_unobserved_lattice_levels_are_not_filled_in():
    values = np.r_[np.arange(5.), np.arange(8., 13.)]
    edges = publication.exposure_distribution_bin_edges(values, grid=np.linspace(0., 12., 101))
    counts = np.histogram(values, bins=edges)[0]
    np.testing.assert_array_equal(counts, [1] * 5 + [0] * 3 + [1] * 5)


@pytest.mark.parametrize("step,offset", [(1., 0.), (.5, -3.25), (.3, .17)])
def test_sparse_lattice_can_recover_a_step_absent_from_observed_gaps(step, offset):
    levels = np.array([0, 2, 5, 7, 10, 12])
    values = offset + step * levels
    grid = np.linspace(values.min(), values.max(), 101)
    edges = publication.exposure_distribution_bin_edges(values, grid=grid)
    np.testing.assert_allclose(edges, offset + step * (np.arange(14) - .5), atol=1e-7)
    counts = np.histogram(values, bins=edges)[0]
    np.testing.assert_array_equal(np.flatnonzero(counts), levels)
    # Empty bins reflect missing observations; every bin can hold a lattice value.
    assert np.all(np.histogram(offset + step * np.arange(13), bins=edges)[0] > 0)


def test_primary_estimates_and_legacy_grid_counts_stay_identical(tmp_path, monkeypatch):
    frame, projection = _synthetic_primary(tmp_path)
    kwargs = dict(frame=frame, authority=projection.authority,
                  runtime_projection_sha256=projection.projection_sha256)
    fixed = producer.run_landmark_spline_association(**kwargs, out_dir=tmp_path / "fixed")
    monkeypatch.setattr(publication, "exposure_distribution_bin_edges",
                        lambda values, *, grid: _legacy_edges(grid).tolist())
    before = producer.run_landmark_spline_association(**kwargs, out_dir=tmp_path / "before")
    assert fixed == before
    for name in ("landmark_rcs_curve.csv", "landmark_adjusted_absolute_risk.csv"):
        after = pd.read_csv(tmp_path / "fixed" / name)
        legacy = pd.read_csv(tmp_path / "before" / name)
        pd.testing.assert_frame_equal(after.drop(columns="exposure_distribution_json"),
                                      legacy.drop(columns="exposure_distribution_json"))
        assert after.exposure_distribution_json.nunique() == 1
        histogram = json.loads(after.exposure_distribution_json.iloc[0])
        assert sum(histogram["counts"]) == after.exposure_density_n.sum()
        edges = np.asarray(histogram["edges"])
        shown = frame.marker.between(after.exposure_value.min(), after.exposure_value.max())
        np.testing.assert_array_equal(histogram["counts"], np.histogram(frame.marker[shown], bins=edges)[0])


def _histogram_frame():
    return pd.DataFrame({
        "exposure_density_fraction": [.2, .6, .2], "exposure_density_n": [20, 60, 20],
        "exposure_distribution_json": [json.dumps({"edges": [.5, 2.5, 5.5], "counts": [35, 65]})] * 3,
    })


@pytest.mark.parametrize("histogram", [
    {"edges": [0., 1., 2.], "counts": [35, 64]},
    {"edges": [0., 1., 2.], "counts": [-1, 101]},
    {"edges": [0., 1., 2.], "counts": [35.5, 64.5]},
    {"edges": [0., 1., 2.], "counts": [0, 0]},
    {"edges": [0., 1., 1.], "counts": [35, 65]},
    {"edges": [0., 2., 1.], "counts": [35, 65]},
    {"edges": [0., float("nan"), 2.], "counts": [35, 65]},
    {"edges": [0., 1., 2.], "counts": [float("nan"), 65]},
    {"edges": [0., 1.], "counts": [35, 65]},
    {"edges": [[0., 1., 2.]], "counts": [[35, 65]]},
])
def test_published_histogram_rejects_invalid_geometry_or_counts(histogram):
    from easyicu.research_agent.execution.runners.landmark_association_figure_executor import _published_exposure_distribution

    frame = _histogram_frame()
    frame["exposure_distribution_json"] = json.dumps(histogram)
    with pytest.raises(ValueError, match="ordered edges and conserved counts"):
        _published_exposure_distribution(frame)


@pytest.mark.parametrize("replacement", [None, json.dumps({"edges": [1, 2, 3], "counts": [35, 65]})])
def test_curve_rows_cannot_disagree_about_the_distribution(replacement):
    from easyicu.research_agent.execution.runners.landmark_association_figure_executor import _published_exposure_distribution

    frame = _histogram_frame()
    frame.loc[1, "exposure_distribution_json"] = replacement
    with pytest.raises(ValueError, match="identical across curve rows"):
        _published_exposure_distribution(frame)


@pytest.mark.parametrize("mismatch", [None, "edges", "counts", "shape", "legacy"])
def test_renderer_uses_published_bins_in_both_shared_axis_strips(tmp_path, monkeypatch, mismatch):
    import matplotlib.pyplot as plt
    from easyicu.research_agent.execution.runners import landmark_association_figure_executor as renderer
    from tests.support.host_runner_parity import table_binding

    inputs = ("table:synthetic_rcs_curve", "table:synthetic_adjusted_absolute_risk")
    frames = {}
    bindings = {}
    for key, column, estimates in zip(
        inputs, ("adjusted_odds_ratio", "adjusted_absolute_risk"),
        ([.8, 1., 1.2], [.1, .2, .3]),
    ):
        frame = _histogram_frame().assign(
            exposure="marker", marker=[1., 3., 5.], reference_marker=3.,
            ci_low=np.asarray(estimates) * .8, ci_high=np.asarray(estimates) * 1.2,
        )
        frame[column] = estimates
        frames[key] = frame
        if mismatch and key == inputs[1]:
            if mismatch == "legacy":
                frame.drop(columns="exposure_distribution_json", inplace=True)
            else:
                different = {
                    "edges": {"edges": [.5, 3.5, 5.5], "counts": [35, 65]},
                    "counts": {"edges": [.5, 2.5, 5.5], "counts": [36, 64]},
                    "shape": {"edges": [.5, 1.5, 2.5, 5.5], "counts": [10, 25, 65]},
                }
                frame["exposure_distribution_json"] = json.dumps(different[mismatch])
        bindings[key] = table_binding(tmp_path, key, frame)
    captured = []

    def capture(fig, *args, **kwargs):
        captured.append(fig)
        return {}

    monkeypatch.setattr(renderer, "save_publication_figure", capture)
    arguments = dict(out_dir=tmp_path / "figure", run_dir=tmp_path,
                     resolved_inputs={"step_id": "display", "inputs": bindings},
                     step_id="display", figure_product="distribution", input_keys=inputs)
    try:
        if mismatch:
            with pytest.raises(ValueError, match="densities do not align"):
                renderer.run_landmark_association_figure(**arguments)
            return
        renderer.run_landmark_association_figure(**arguments)
        fig = captured[0]
        assert len(fig.axes) == 4
        for curve_ax, strip_ax in zip(fig.axes[:2], fig.axes[2:]):
            assert strip_ax.get_shared_x_axes().joined(curve_ax, strip_ax)
            assert strip_ax.get_xlim() == curve_ax.get_xlim() == (1., 5.)
            assert strip_ax.get_yscale() == "linear"
            np.testing.assert_allclose([bar.get_x() for bar in strip_ax.patches], [.5, 2.5])
            np.testing.assert_allclose([bar.get_width() for bar in strip_ax.patches], [2., 3.])
            np.testing.assert_allclose([bar.get_height() for bar in strip_ax.patches], [35., 65.])
        for ax, key, column, factor in zip(fig.axes[:2], inputs,
                                         ["adjusted_odds_ratio", "adjusted_absolute_risk"], [1, 100]):
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), frames[key].marker)
            np.testing.assert_array_equal(ax.lines[0].get_ydata(), frames[key][column] * factor)
    finally:
        plt.close("all")
