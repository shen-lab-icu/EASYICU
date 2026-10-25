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
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

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
    COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS,
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


def _receipt_sidecar(root, *, key, producer, name, digest):
    summary = root / "steps" / producer / "outputs/step_summary.json"
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    path = evidence / f"log_result_envelope_sidecar_fixture__result_envelope_sidecar__{producer}.json"
    path.write_text(json.dumps({
        "step_id": producer,
        "raw_summary_artifact_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "artifacts": [{"product_id": key, "relative_path": name, "sha256": digest}],
    }))
    return path


def _render(
    tmp_path: Path, *, frames=None, labels=None, receipts=None, provenance=None,
    landmark_hours=None,
) -> Path:
    bindings = {}
    for key, frame in (frames or _frames()).items():
        path = tmp_path / f"{key.partition(':')[2]}.csv"
        frame.to_csv(path, index=False)
        bindings[key] = _binding(key, frame, path)
        if receipts and key in receipts:
            producer = key.partition(":")[2]
            bindings[key]["produced_by_step"] = producer
            bindings[key]["identity_row"]["produced_by_step"] = producer
            output = tmp_path / "steps" / producer / "outputs"
            output.mkdir(parents=True)
            (output / path.name).write_bytes(path.read_bytes())
            (output / "step_summary.json").write_text(json.dumps({
                **receipts[key], "step_id": producer, "output_files": {key: path.name},
            }))
            _receipt_sidecar(tmp_path, key=key, producer=producer, name=path.name,
                             digest=bindings[key]["sha256"])
    context = ra.ResearchContext(
        research_question="Is the lactate tertile associated with ICU readmission?",
        cohort=ra.CohortDescriptor(
            cohort_name="Adult ICU stays", database="mimiciv", n_patients=380, n_stays=412,
            provenance=provenance or {},
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
        time_windows=([{
            "name": "exposure_window", "anchor": "icu_admission",
            "start_hours": 0, "end_hours": landmark_hours,
        }] if landmark_hours is not None else []),
    )
    context_path = tmp_path / "research_context.json"
    context_path.write_text(context.model_dump_json(), encoding="utf-8")
    manifest = {
        "step_id": "readmission_figure", "inputs": bindings,
        "context": {"relative_path": context_path.name,
                    "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest()},
    }
    (tmp_path / "resolved_inputs").mkdir()
    (tmp_path / "resolved_inputs/readmission_figure.json").write_text(json.dumps(manifest))
    out_dir = tmp_path / "outputs"
    run_composite_descriptive_figure(
        out_dir=out_dir,
        run_dir=tmp_path,
        resolved_inputs=manifest,
        step_id="readmission_figure",
        figure_product="readmission_figure",
        input_keys=(ABSOLUTE_RISK_ASSOCIATION_COMPOSITE_INPUTS
                    if "table:absolute_risk_context" in bindings
                    else COMPOSITE_ASSOCIATION_ROBUSTNESS_PUBLICATION_FIGURE_INPUTS),
        display_labels=labels or {
            "lactate_tertile": "乳酸三分位", "icu_readmission": "ICU再入院",
        },
    )
    return out_dir


def _svg_text(path):
    root = ElementTree.fromstring(path.read_text())
    return " ".join("".join(node.itertext()) for node in root.iter("{http://www.w3.org/2000/svg}text"))


def test_observed_risk_is_points_with_intervals_and_events_per_group() -> None:
    frame = _absolute_risk_context(pd.DataFrame(_risk_rows()))
    fig, ax = plt.subplots()
    try:
        title = _draw_absolute_risk_points(
            ax, frame, color="#E28E2C", neutral="#6F6F6F", level_label=str,
        )
        assert title == "Absolute risk by exposure level"
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


@pytest.mark.parametrize(("low", "high", "first_tick", "last_tick"), [
    (0.4, 1.2, 0.5, 1.1),
    (0.8, 3.1, 0.8, 3.0),
    (1.1, 12.0, 1.0, 10.0),
])
def test_ratio_forest_ticks_span_the_intervals_without_changing_marks(
    low, high, first_tick, last_tick,
):
    frame = pd.DataFrame({
        "estimate": [(low * high) ** 0.5], "ci_low": [low], "ci_high": [high],
        "effect_scale": ["odds_ratio"], "contrast": ["Synthetic contrast"],
    })
    original = frame.copy(deep=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    try:
        _forest(
            ax, frame, estimate_column="estimate", label_column="contrast",
            title="Synthetic association", color="#0F4D92", label_formatter=_label,
        )
        fig.canvas.draw()
        ticks = ax.get_xticks().tolist()
        assert ax.get_xscale() == "log"
        assert 1.0 in ticks
        assert 3 <= len(ticks) <= 5
        assert ticks[0] == first_tick
        assert ticks[-1] == last_tick
        assert all(min(low, 1.0) * 0.97 <= tick <= max(high, 1.0) * 1.03 for tick in ticks)
        texts = ax.get_xticklabels()
        assert [text.get_text() for text in texts] == [f"{tick:g}" for tick in ticks]
        boxes = [text.get_window_extent() for text in texts]
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
        assert ax.get_xlim() == pytest.approx((min(low, 1.0) / 1.3, max(high, 1.0) * 1.3))
        markers = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(markers) == 1
        assert markers[0].get_xdata().tolist() == original["estimate"].tolist()
        segments = [segment for collection in ax.collections for segment in collection.get_segments()]
        assert len(segments) == 1
        assert segments[0][:, 0].tolist() == pytest.approx([low, high])
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


def test_full_forest_row_labels_make_the_group_axis_title_redundant(tmp_path, captured_main):
    _render(tmp_path)
    [fig] = captured_main
    risk, forest = fig.axes
    assert risk.get_xlabel() == "Lactate tertile"
    assert forest.get_ylabel() == ""
    words = _figure_words(fig)
    assert "Lactate tertile 1 (reference)" in words
    assert "Lactate tertile 2 vs Lactate tertile 1" in words
    assert "Lactate tertile 3 vs Lactate tertile 1" in words


def test_the_rendered_figure_names_exposure_and_outcome_in_english(tmp_path: Path) -> None:
    out_dir = _render(tmp_path)
    svg = (out_dir / "readmission_figure.svg").read_text(encoding="utf-8")
    contract = json.loads(
        (out_dir / "readmission_figure.figure_contract.json").read_text(encoding="utf-8")
    )

    for text in (
        "Lactate tertile", "ICU readmission (%)", "All measured", "37/412",
        "Odds ratio (95% CI, log scale)",
    ):
        assert text in svg, text
    # The exposure names both the risk panel's groups and the forest's rows.
    assert svg.count("Lactate tertile") >= 2
    text = _svg_text(out_dir / "readmission_figure.svg")
    assert "Lactate tertile 2 vs Lactate tertile 1" in text
    assert "Lactate tertile 3 vs Lactate tertile 1" in text
    assert not _CJK.search(svg)
    assert " Vs " not in svg
    assert contract["panels"][0]["title"] == "Absolute risk by exposure level"
    assert "events/n" in contract["reader_caption"]
    width_pt = float(ElementTree.fromstring(svg).attrib["width"].removesuffix("pt"))
    assert width_pt / 72 * 25.4 == pytest.approx(183, abs=0.01)
    assert mpimg.imread(out_dir / "readmission_figure.png").shape[1] == 2161


def test_audit_status_strips_do_not_take_a_result_panel_height(tmp_path: Path) -> None:
    out_dir = _render(tmp_path)
    height, width = mpimg.imread(out_dir / "readmission_figure.png").shape[:2]

    assert height / width < 0.85


def test_the_risk_panel_carries_the_term_the_article_strategy_reads(tmp_path: Path) -> None:
    from easyicu.research_agent.figures.contracts import panel_text
    from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy

    out_dir = _render(tmp_path)
    contract = json.loads(
        (out_dir / "readmission_figure.figure_contract.json").read_text(encoding="utf-8")
    )
    context = ra.ResearchContext.model_validate_json(
        (tmp_path / "research_context.json").read_text(encoding="utf-8")
    )
    strategy = build_article_figure_strategy(context, analysis_family="association")
    [role] = [item for item in strategy.role_strategies if item.role == "descriptive_result"]
    [panel] = [item for item in contract["panels"] if item.get("role") == "descriptive_result"]

    assert any(term in panel_text(panel) for term in role.required_text_terms)


@pytest.mark.parametrize("path_kind", ["absolute_risk", "distribution"])
def test_both_risk_paths_name_both_sides_of_named_contrasts(tmp_path, path_kind):
    frames = _frames()
    levels = {"1": "low_band", "2": "middle_band", "3": "high_band"}
    risk = frames["table:absolute_risk_context"]
    risk["group_value"] = risk["group_value"].replace(levels)
    model = frames["table:adjusted_association_estimates"]
    for column in ("exposure_level", "reference_level"):
        model[column] = model[column].astype(str).replace(levels)
    if path_kind == "distribution":
        rows = risk.loc[risk["estimate_type"].eq("outcome_risk") & risk["group_type"].eq("exposure_level")]
        frames.pop("table:absolute_risk_context")
        frames["table:exposure_outcome_distribution"] = pd.DataFrame({
            "row_role": "exposure_level", "exposure_column": "lactate_tertile",
            "exposure_level": rows["group_value"], "n_rows": rows["n"],
            "exposure_denominator": 412, "exposure_pct": rows["n"] / 412 * 100,
            "outcome_events": rows["event_n"], "outcome_denominator": rows["n"],
            "outcome_rate_pct": rows["estimate"] * 100,
            "ci_low_pct": rows["ci_low"] * 100, "ci_high_pct": rows["ci_high"] * 100,
        })
    labels = {
        "lactate_tertile": "Marker band", "icu_readmission": "ICU readmission",
        "lactate_tertile=low_band": "Lower concentration",
        "lactate_tertile=middle_band": "Intermediate concentration",
        "lactate_tertile=high_band": "Higher concentration",
    }
    out = _render(tmp_path, frames=frames, labels=labels)
    svg = _svg_text(out / "readmission_figure.svg")
    assert "Intermediate concentration vs Lower concentration" in svg
    assert "Higher concentration vs Lower concentration" in svg
    assert "2 vs 1" not in svg


def test_long_risk_labels_are_measured_and_wrapped_without_overlap():
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    labels = {
        "1": "Lower concentration in the prespecified observation window",
        "2": "Intermediate concentration in the prespecified observation window",
        "3": "Higher concentration in the prespecified observation window",
    }
    try:
        _draw_absolute_risk_points(
            ax, _absolute_risk_context(pd.DataFrame(_risk_rows())),
            color="blue", neutral="grey", level_label=labels.__getitem__,
        )
        fig.canvas.draw()
        ticks = ax.get_xticklabels()
        boxes = [tick.get_window_extent(fig.canvas.get_renderer()) for tick in ticks]
        assert all(a.x1 < b.x0 for a, b in zip(boxes, boxes[1:]))
        assert all(tick.get_fontsize() >= 7 for tick in ticks)
        assert " ".join(ticks[2].get_text().split()) == labels["2"] + " 12/136"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(("source", "expected"), [
    ("Maximum below 5 mg/L, without maximum at least 9 mg/L", "Maximum <5 mg/L"),
    ("Neither maximum at least 9 mg/L nor maximum below 5 mg/L", "Maximum 5–<9 mg/L"),
    ("Maximum at least 9 mg/L", "Maximum ≥9 mg/L"),
    ("Minimum below 5 mg/L, without maximum at least 9 mg/L",
     "Minimum below 5 mg/L, without maximum at least 9 mg/L"),
])
def test_clinical_threshold_labels_keep_exact_semantics(tmp_path, source, expected):
    labels = {"lactate_tertile": "Marker band", "lactate_tertile=1": source}
    out = _render(tmp_path, labels=labels)
    assert expected in _svg_text(out / "readmission_figure.svg")


def test_a_restatement_and_coverage_are_supplementary_not_main_results(tmp_path):
    frames = _frames()
    matrix = frames["table:robustness_matrix"]
    matrix["independent_variant"] = [None, False]
    matrix.loc[1, ["point_estimate", "ci_low", "ci_high"]] = [2.2, 1.5, 3.1]
    frames["table:robustness_summary"]["non_independent_specs"] = [0, 1]
    out = _render(tmp_path, frames=frames)
    contract = json.loads((out / "readmission_figure.figure_contract.json").read_text())
    assert [p["role"] for p in contract["panels"]] == [
        "descriptive_result", "primary_estimand",
    ]
    svg = (out / "readmission_figure.svg").read_text()
    assert "Sensitivity-specification status" not in svg
    assert "Sensitivity-analysis coverage" not in svg
    audit = json.loads((out / "readmission_figure_audit.figure_contract.json").read_text())
    assert all(p["metadata"]["display_authority"] in {
        "audit_only", "specification_status_only",
    } for p in audit["panels"])
    assert "0/1" in (out / "readmission_figure_audit.svg").read_text()


def test_caption_states_bound_interval_methods_adjustment_and_denominators(tmp_path):
    frames = _frames()
    model = frames["table:adjusted_association_estimates"]
    for key, value in {
        "n": 412, "cluster_count": 380, "cluster_unit": "patient",
        "analysis_unit": "icu_stay", "variance_estimator": "cluster_robust",
        "covariates": "age;sex", "estimator_kind": "logistic",
    }.items():
        model[key] = value
    frames["table:absolute_risk_context"]["ci_method"] = "wilson"
    out = _render(tmp_path, frames=frames)
    caption = json.loads((out / "readmission_figure.figure_contract.json").read_text())["reader_caption"]
    for phrase in ("Wilson", "patient-cluster-robust", "Age", "Sex", "412 ICU stays",
                   "380 patient clusters", "not causal", "odds ratios"):
        assert phrase in caption


def test_caption_does_not_invent_missing_interval_or_population_facts(tmp_path):
    out = _render(tmp_path)
    caption = json.loads((out / "readmission_figure.figure_contract.json").read_text())["reader_caption"]
    assert "Wilson" not in caption
    assert "24 h" not in caption
    assert "380 patient clusters" not in caption
    assert "not reported" in caption


@pytest.mark.parametrize("hours", [8, 24])
def test_caption_binds_the_actual_landmark_and_exposure_time_origin(tmp_path, hours):
    frames = _frames()
    model = frames["table:adjusted_association_estimates"]
    for key, value in {
        "n": 412, "cluster_count": 380, "variance_estimator": "cluster_robust",
        "covariates": "age", "estimator_kind": "logistic",
    }.items():
        model[key] = value
    receipts = {
        "table:adjusted_association_estimates": {
            "n_total": 412,
            "landmark_runtime_receipt": {"landmark_hours": hours},
            "model_contracts": [{
                "model_id": "primary_adjusted", "dependence": {"cluster_unit": "patient"},
            }],
        },
        "table:absolute_risk_context": {"method": "Wilson 95% confidence intervals"},
    }
    grouping = {
        "variable": "lactate_tertile", "concept": "lact", "scale": "ordinal",
        "levels": [
            {"code": 1, "group": "g1", "label": "Lower concentration"},
            {"code": 2, "group": "g2", "label": "Intermediate concentration"},
            {"code": 3, "group": "g3", "label": "Higher concentration"},
        ],
        "parameters": {
            "concept": "lact", "window": {"start_hours": 0, "end_hours": hours},
            "source_columns": {"max": "marker_max"}, "scale": "ordinal",
            "unmeasured": "exclude",
            "groups": [
                {"id": "g1", "rule": {"op": "<", "summary": "max", "value": 5, "unit": "mg/L"}},
                {"id": "g3", "rule": {"op": ">=", "summary": "max", "value": 9, "unit": "mg/L"}},
                {"id": "g2", "rule": "otherwise"},
            ],
        },
    }
    out = _render(tmp_path, frames=frames, receipts=receipts, landmark_hours=hours,
                  provenance={"analysis_unit": "icu_stay", "exposure_group_labels": [grouping]})
    caption = json.loads((out / "readmission_figure.figure_contract.json").read_text())["reader_caption"]
    for text in (
        "Wilson", "patient-cluster-robust", "412 ICU stays", "380 patient clusters",
        f"alive and observed at the {hours} h landmark", f"0–{hours} h after ICU admission",
    ):
        assert text in caption


@pytest.mark.parametrize("names_step", [True, False])
@pytest.mark.parametrize("mutation", ["aggregate", "summary", "sidecar_link", "producer"])
def test_caption_refuses_stale_producer_output(tmp_path, mutation, names_step):
    from easyicu.research_agent.execution.runners.association_publication_figure_renderer import (
        _caption_producer_receipt,
    )
    from easyicu.research_agent.execution.runners.typed_input_binding import BoundTypedInput

    frame = _frames()["table:adjusted_association_estimates"]
    path = tmp_path / "bound.csv"
    frame.to_csv(path, index=False)
    key = "table:adjusted_association_estimates"
    binding = _binding(key, frame, path)
    binding["produced_by_step"] = "model"
    output = tmp_path / "steps/model/outputs"
    output.mkdir(parents=True)
    (output / "model.csv").write_bytes(path.read_bytes())
    (output / "step_summary.json").write_text(json.dumps({
        **({"step_id": "model"} if names_step else {}),
        "output_files": {key: "model.csv"}, "n_total": 412,
    }))
    sidecar = _receipt_sidecar(tmp_path, key=key, producer="model", name="model.csv",
                               digest=binding["sha256"])
    bound = BoundTypedInput(
        input_key=key, path=path, sha256=binding["sha256"], declared_kind="table",
        evidence_kind="table", product="adjusted_association_estimates", evidence_id="model",
        columns=tuple(frame.columns), row_count=len(frame), frame=frame, binding=binding,
    )
    assert _caption_producer_receipt(bound)["n_total"] == 412
    if mutation == "aggregate":
        path.write_text("estimate\n99\n")
    elif mutation == "summary":
        summary = output / "step_summary.json"
        summary.write_text(summary.read_text().replace('412', '999'))
    elif mutation == "producer":
        summary = output / "step_summary.json"
        summary.write_text(json.dumps({**json.loads(summary.read_text()), "step_id": "different_model"}))
        _receipt_sidecar(tmp_path, key=key, producer="model", name="model.csv",
                         digest=binding["sha256"])
    else:
        sidecar.write_text(sidecar.read_text().replace(binding["sha256"], "0" * 64))
    with pytest.raises(ValueError, match="caption producer"):
        _caption_producer_receipt(bound)


@pytest.mark.parametrize("mutation", ["context_digest", "input_digest"])
def test_caption_context_must_be_pinned_to_the_displayed_inputs(tmp_path, mutation):
    from easyicu.research_agent.execution.runners.association_publication_figure_renderer import (
        _caption_context,
    )
    from easyicu.research_agent.execution.runners.typed_input_binding import BoundTypedInput

    frame = _frames()["table:adjusted_association_estimates"]
    path = tmp_path / "bound.csv"
    frame.to_csv(path, index=False)
    key = "table:adjusted_association_estimates"
    binding = _binding(key, frame, path)
    bound = BoundTypedInput(
        input_key=key, path=path, sha256=binding["sha256"], declared_kind="table",
        evidence_kind="table", product="adjusted_association_estimates", evidence_id="model",
        columns=tuple(frame.columns), row_count=len(frame), frame=frame, binding=binding,
    )
    context = tmp_path / "context.json"
    context.write_text('{"cohort": {"database": "synthetic"}}')
    manifest = {"step_id": "figure", "inputs": {key: dict(binding)}, "context": {
        "relative_path": context.name, "sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
    }}
    directory = tmp_path / "resolved_inputs"
    directory.mkdir()
    manifest_path = directory / "figure.json"
    manifest_path.write_text(json.dumps(manifest))
    assert _caption_context({key: bound}, "figure")["cohort"]["database"] == "synthetic"
    if mutation == "context_digest":
        context.write_text('{"cohort": {"database": "stale"}}')
    else:
        manifest["inputs"][key]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest mismatch|does not bind"):
        _caption_context({key: bound}, "figure")


def _threshold_grouping(*, upper_unit="mg/L", upper_summary="max"):
    upper = f"{'Minimum' if upper_summary == 'min' else 'Maximum'} at least 9 {upper_unit}"
    names = (f"Maximum below 5 mg/L, without {upper.lower()}",
             f"Neither {upper.lower()} nor maximum below 5 mg/L", upper)
    return {
        "variable": "lactate_tertile", "concept": "lact", "scale": "ordinal",
        "levels": [
            {"code": code, "group": f"g{code}", "label": name}
            for code, name in enumerate(names, 1)
        ],
        "parameters": {
            "concept": "lact", "window": {"start_hours": 0, "end_hours": 8},
            "source_columns": {summary: f"marker_{summary}" for summary in {"max", upper_summary}},
            "scale": "ordinal", "unmeasured": "exclude",
            "groups": [
                {"id": "g3", "rule": {"op": ">=", "summary": upper_summary,
                                       "value": 9, "unit": upper_unit}},
                {"id": "g1", "rule": {"op": "<", "summary": "max", "value": 5, "unit": "mg/L"}},
                {"id": "g2", "rule": "otherwise"},
            ],
        },
    }


@pytest.fixture
def captured_main(monkeypatch):
    from easyicu.research_agent.execution.runners import association_publication_figure_renderer as owner

    figures = []

    def capture(fig, *args, **kwargs):
        fig.canvas.draw()
        figures.append(fig)
        return {}

    monkeypatch.setattr(owner, "save_publication_figure", capture)
    monkeypatch.setattr(owner, "save_association_coverage_audit", lambda **kwargs: {})
    yield figures
    for fig in figures:
        plt.close(fig)


def _figure_words(fig):
    from matplotlib.text import Text

    return [" ".join(text.get_text().split()) for text in fig.findobj(Text)
            if text.get_visible() and text.get_text()]


def test_shared_typed_measure_moves_to_axes_and_leaves_exact_ranges(tmp_path, captured_main):
    _render(tmp_path, provenance={"exposure_group_labels": [_threshold_grouping()]})
    [fig] = captured_main
    risk, forest = fig.axes
    assert risk.get_xlabel() == "Maximum lactate (mg/L)"
    assert forest.get_ylabel() == "Maximum lactate (mg/L)"
    assert [" ".join(t.get_text().split()) for t in risk.get_xticklabels()] == [
        "All measured 37/412", "<5 8/140", "5–<9 12/136", "≥9 17/136",
    ]
    assert {"5–<9 vs <5", "≥9 vs <5"} <= set(_figure_words(fig))


def test_forest_adds_one_typed_reference_without_an_interval_or_source_row(tmp_path, captured_main):
    frames = _frames()
    original = frames["table:adjusted_association_estimates"].copy(deep=True)
    out = _render(tmp_path, frames=frames,
                  provenance={"exposure_group_labels": [_threshold_grouping()]})
    [fig] = captured_main
    forest = fig.axes[1]
    assert _figure_words(fig).count("<5 (reference)") == 1
    markers = [(float(x), float(y)) for line in forest.lines if line.get_marker() == "o"
               for x, y in zip(line.get_xdata(), line.get_ydata())]
    assert markers == [(1.4, 1.0), (2.2, 2.0), (1.0, 0.0)]
    segments = [segment for collection in forest.collections for segment in collection.get_segments()]
    assert len(segments) == 2
    assert all(segment[0, 1] != 0 for segment in segments)
    assert "1.00" in _figure_words(fig)
    assert "1.00 (1.00–1.00)" not in _figure_words(fig)
    pd.testing.assert_frame_equal(frames["table:adjusted_association_estimates"], original)
    saved = pd.read_csv(out / "adjusted_association_estimates_source_data.csv")
    pd.testing.assert_frame_equal(saved[original.columns], original)


@pytest.mark.parametrize(("label", "expected"), [
    ("In hospital mortality", "In-hospital mortality (%)"),
    ("In-hospital mortality", "In-hospital mortality (%)"),
    ("ICU mortality", "ICU mortality (%)"),
    ("28-day mortality", "28-day mortality (%)"),
])
def test_outcome_grammar_preserves_the_bound_identity(tmp_path, captured_main, label, expected):
    frames = _frames()
    frames["table:adjusted_association_estimates"]["outcome"] = "event_flag"
    _render(tmp_path, frames=frames, labels={"event_flag": label})
    assert captured_main[0].axes[0].get_ylabel() == expected
    assert frames["table:adjusted_association_estimates"]["outcome"].eq("event_flag").all()


@pytest.mark.parametrize(("unit", "summary", "words"), [
    ("mg/dL", "max", ("mg/L", "mg/dL", "Maximum")),
    ("mg/L", "min", ("Minimum", "Maximum", "mg/L")),
])
def test_different_typed_units_or_summaries_keep_full_level_labels(
    tmp_path, captured_main, unit, summary, words,
):
    _render(tmp_path, provenance={"exposure_group_labels": [
        _threshold_grouping(upper_unit=unit, upper_summary=summary),
    ]})
    [fig] = captured_main
    ticks = " ".join(t.get_text() for t in fig.axes[0].get_xticklabels())
    assert all(word in ticks for word in words), repr(ticks)
    assert "Maximum" not in fig.axes[0].get_xlabel()
    assert "5–<9 vs <5" not in _figure_words(fig)


def test_matching_prose_alone_does_not_authorize_a_shared_measure(tmp_path, captured_main):
    _render(tmp_path, labels={
        "lactate_tertile": "Marker band",
        "lactate_tertile=1": "Maximum below 5 mg/L",
        "lactate_tertile=2": "Neither maximum below 5 mg/L nor maximum at least 9 mg/L",
        "lactate_tertile=3": "Maximum at least 9 mg/L",
    })
    [fig] = captured_main
    assert fig.axes[0].get_xlabel() == "Marker band"
    assert "Maximum 5–<9 mg/L vs Maximum <5 mg/L" in _figure_words(fig)


@pytest.mark.parametrize("mixed_units", [False, True])
def test_stacked_panels_use_the_page_width_without_borrowing_contrast_margins(
    tmp_path, captured_main, mixed_units,
):
    from matplotlib.text import Text
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    _render(tmp_path, provenance={"exposure_group_labels": [
        _threshold_grouping(upper_unit="mg/dL" if mixed_units else "mg/L"),
    ]})
    [fig] = captured_main
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    risk, forest = fig.axes
    assert risk.get_position().x0 < 0.16
    assert risk.get_position().width > 0.80
    assert risk.get_position().width == pytest.approx(forest.get_position().width, abs=0.005)
    assert risk.get_position().x0 == pytest.approx(forest.get_position().x0, abs=0.005)
    assert fig.get_tightbbox(renderer).width * 25.4 == pytest.approx(183, abs=0.1)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 7
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -0.5 and box.x1 <= fig.bbox.width + 0.5, text.get_text()
        assert box.y0 >= -0.5 and box.y1 <= fig.bbox.height + 0.5, text.get_text()
    ticks = [t.get_window_extent(renderer) for t in risk.get_xticklabels()]
    assert all(a.x1 < b.x0 for a, b in zip(ticks, ticks[1:]))
    row_texts = [t for t in forest.texts if "vs" in t.get_text() or "reference" in t.get_text()]
    estimates = [t for t in forest.texts if re.match(r"\d", t.get_text()) and t not in row_texts]
    assert row_texts
    assert all(not left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer))
               for left in row_texts for right in estimates)


@pytest.mark.parametrize(("upper_op", "lower_op", "expected"), [
    (">=", "<", ("<5", "5–<9", "≥9")),
    (">", "<=", ("≤5", ">5–≤9", ">9")),
])
@pytest.mark.parametrize("low_first", [False, True])
def test_factored_ranges_preserve_typed_boundary_inclusion(upper_op, lower_op, expected, low_first):
    from easyicu.research_agent.figures.display_labels import grouped_axis_labels

    grouping = _threshold_grouping()
    high, low, otherwise = grouping["parameters"]["groups"]
    high["rule"]["op"], low["rule"]["op"] = upper_op, lower_op
    grouping["parameters"]["groups"] = [low, high, otherwise] if low_first else [high, low, otherwise]
    result = grouped_axis_labels(grouping, {})
    assert tuple(result[f"lactate_tertile={code}"] for code in (1, 2, 3)) == expected


@pytest.mark.parametrize("defect", ["missing_unit", "empty_interval", "wrong_code", "wrong_concept"])
def test_incomplete_or_conflicting_typed_groups_cannot_factor_an_axis(defect):
    from easyicu.research_agent.figures.display_labels import grouped_axis_labels

    grouping = _threshold_grouping()
    groups = grouping["parameters"]["groups"]
    if defect == "missing_unit":
        for group in groups[:2]:
            group["rule"].pop("unit")
    elif defect == "empty_interval":
        groups[0]["rule"]["value"] = 4
    elif defect == "wrong_code":
        grouping["levels"][0]["code"] = 9
    else:
        grouping["concept"] = "another_marker"
    assert grouped_axis_labels(grouping, {}) == {}


@pytest.mark.parametrize("ambiguity", ["reference", "exposure", "missing"])
def test_a_forest_reference_requires_one_explicit_typed_identity(ambiguity):
    from easyicu.research_agent.execution.runners.association_publication_figure_renderer import (
        _forest_reference_label,
    )

    frame = _frames()["table:adjusted_association_estimates"]
    if ambiguity == "reference":
        frame.loc[1, "reference_level"] = 2
    elif ambiguity == "exposure":
        frame.loc[1, "exposure"] = "another_marker"
    else:
        frame = frame.drop(columns="reference_level")
    assert _forest_reference_label(frame, {}) is None


def test_a_nonratio_forest_does_not_gain_a_ratio_reference():
    frame = _frames()["table:adjusted_association_estimates"]
    frame["effect_scale"] = "mean_difference"
    fig, ax = plt.subplots()
    try:
        _forest(ax, frame, estimate_column="estimate", label_column="contrast",
                title="Difference", color="blue", reference_label="Lower")
        assert len(ax.get_yticks()) == 2
        assert "Lower (reference)" not in _figure_words(fig)
        assert ax.get_xscale() == "linear"
    finally:
        plt.close(fig)
