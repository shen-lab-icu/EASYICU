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
        f"alive and observed at the {hours} h landmark", f"0–{hours} h after ICU Admission",
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
