"""Deterministic export QA using synthetic sealed tables, never a research run.

Usage: PYTHONPATH=/absolute/checkout/src python tools/verify_figure_presentation.py --output /absolute/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.contracts.figure_plan import FigurePresentationSpec
from easyicu.research_agent.execution.figure_plan_binding import (
    validate_step_planned_figure_contract_binding,
)
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS,
)
from easyicu.research_agent.execution.runners.adjusted_association_figure_executor import (
    run_association_overview_figure,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS,
)
from easyicu.research_agent.figures.prediction import render_prediction_figure
from easyicu.research_agent.figures.publication import (
    make_figure_contract,
    save_publication_figure,
)
from easyicu.research_agent.gates.visual_qa import audit_svg_text_layout
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel(panel_id, role, chart, product, output, style):
    return dict(
        panel_id=panel_id,
        article_role=role,
        chart_type=chart,
        source_products=[f"table:{product}"],
        figure_output=f"figure:{output}",
        presentation=style.model_dump(mode="json"),
    )


def binding(root, product, frame):
    path = root / f"{product}.csv"
    frame.to_csv(path, index=False)
    digest = sha(path)
    key = f"table:{product}"
    return dict(
        declared_kind="table",
        evidence_kind="table",
        product=product,
        evidence_id=product,
        sha256=digest,
        relative_path=path.name,
        product_contract={
            "schema_version": "easyicu.host_typed_product.v4",
            "columns": list(frame.columns),
            "row_count": len(frame),
        },
        consumption_contract={
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
            "verified_row_count": len(frame),
        },
        identity_row={"input_key": key, "product": product, "sha256": digest},
    )


def verify(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    sources = output / "synthetic_sources"
    sources.mkdir(exist_ok=False)
    evidence = EvidenceStore(sources)
    frames = {
        "calibration_curve": pd.DataFrame(
            {
                "predicted": [0.1, 0.3, 0.5, 0.7, 0.9],
                "observed": [0.12, 0.28, 0.52, 0.68, 0.88],
            }
        ),
        "roc_curve": pd.DataFrame(
            {"fpr": [0, 0.1, 0.3, 0.6, 1], "tpr": [0, 0.4, 0.7, 0.9, 1]}
        ),
        "model_performance": pd.DataFrame(
            {
                "metric": ["auroc", "brier_score"],
                "value": [0.78, 0.16],
                "split": ["heldout", "heldout"],
            }
        ),
    }
    for name, frame in frames.items():
        path = sources / f"{name}.csv"
        frame.to_csv(path, index=False)
        evidence.register_file(
            kind="table",
            description="Synthetic engineering fixture; no scientific result",
            source_path=path,
            evidence_id=name,
            aliases=[name],
            producer="engineering_fixture",
            generation_mode="fixture",
        )
    context = ResearchContext(
        research_question="Synthetic model display QA",
        cohort=CohortDescriptor(
            cohort_name="synthetic_fixture",
            database="synthetic",
            n_patients=100,
            n_stays=100,
        ),
        variables=[],
        primary_exposure="synthetic exposure",
        target_outcome="synthetic outcome",
    )
    estimate = {key: "" for key in ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS}
    estimate.update(
        fit_status="fitted",
        estimate=1.4,
        ci_low=1.1,
        ci_high=1.8,
        effect_scale="odds_ratio",
        exposure="fixture_exposure",
        outcome="fixture_outcome",
        covariates="age;sex",
        estimator_kind="logistic",
        analysis_set="complete_case",
        n=100,
        n_events=25,
        requirement_id="fixture_effect",
        standard_error=0.1,
    )
    distribution = []
    for index, (level, count, events, risk) in enumerate(
        [("Unexposed", 60, 12, 20), ("Exposed", 40, 13, 32.5)]
    ):
        row = {key: "" for key in EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS}
        row.update(
            row_role="exposure_level",
            exposure_level_index=index,
            exposure_level=level,
            n_rows=count,
            exposure_denominator=100,
            exposure_pct=count,
            exposure_ci_low_pct=count - 8,
            exposure_ci_high_pct=count + 8,
            outcome_observed_n=count,
            outcome_missing_n=0,
            outcome_events=events,
            outcome_denominator=count,
            outcome_rate_pct=risk,
            ci_low_pct=risk - 5,
            ci_high_pct=risk + 5,
            missing_exposure_excluded_n=0,
        )
        distribution.append(row)
    bound = {
        "table:adjusted_association_estimates": binding(
            sources,
            "adjusted_association_estimates",
            pd.DataFrame([estimate], columns=ADJUSTED_ASSOCIATION_ESTIMATES_COLUMNS),
        ),
        "table:exposure_outcome_distribution": binding(
            sources,
            "exposure_outcome_distribution",
            pd.DataFrame(distribution, columns=EXPOSURE_OUTCOME_DISTRIBUTION_COLUMNS),
        ),
    }
    source_sha = {path.name: sha(path) for path in sources.glob("*.csv")}
    styles = {
        "paper": FigurePresentationSpec(
            layout="row",
            width_mm=183,
            height_mm=105,
            font_size=7.5,
            palette="grayscale",
            legend_location="outside bottom",
        ),
        "presentation": FigurePresentationSpec(
            layout="grid",
            width_mm=320,
            height_mm=230,
            font_size=15,
            palette="colorblind",
            legend_location="outside bottom",
        ),
    }
    projections = {}
    qa = []
    for name, style in styles.items():
        destination = output / name
        destination.mkdir(exist_ok=True)
        specs = [
            panel(
                "A",
                "calibration",
                "calibration_curve",
                "calibration_curve",
                "prediction",
                style,
            ),
            panel(
                "B", "model_performance", "roc_curve", "roc_curve", "prediction", style
            ),
            panel(
                "C",
                "validation",
                "validation_panel",
                "model_performance",
                "prediction",
                style,
            ),
        ]
        step = AnalysisStep(
            step_id="prediction",
            method="visualization",
            intent="Synthetic fixture only",
            inputs=[f"table:{product}" for product in frames],
            expected_outputs=["figure:prediction"],
            figure_panels=specs,
        )
        rendered = render_prediction_figure(
            context=context,
            plan=AnalysisPlan(
                research_question=context.research_question, steps=[step]
            ),
            evidence=evidence,
            run_dir=sources,
        )
        assert rendered is not None
        rendered.fig.suptitle(
            "Synthetic fixture · export QA only", fontsize=style.font_size
        )
        source_files = []
        for product, frame in rendered.source_frames.items():
            path = destination / f"prediction_{product}.csv"
            frame.to_csv(path, index=False)
            source_files.append(path.name)
        projections[name] = {path: sha(destination / path) for path in source_files}
        contract = make_figure_contract(
            figure_id="prediction",
            core_claim="Engineering fixture demonstrates identical numerical projections across presentation settings.",
            panels=rendered.panels,
            archetype="quantitative_grid",
            width_mm=style.width_mm,
            height_mm=style.height_mm,
            source_data=source_files,
            statistics_note="Synthetic precomputed points and metrics; no estimation, inference or clinical claim.",
        )
        save_publication_figure(
            rendered.fig,
            destination / "prediction",
            contract=contract,
            formats=("svg", "pdf", "png"),
            dpi=160,
        )
        import matplotlib.pyplot as plt

        plt.close(rendered.fig)
        overview_style = style.model_copy(
            update={"layout": "row" if name == "paper" else "column"}
        )
        overview_specs = [
            panel(
                "distribution",
                "descriptive_result",
                "event_rate_panel",
                "exposure_outcome_distribution",
                "association",
                overview_style,
            ),
            panel(
                "effect",
                "primary_estimand",
                "forest",
                "adjusted_association_estimates",
                "association",
                overview_style,
            ),
        ]
        overview_step = AnalysisStep(
            step_id="association",
            planned_analysis_role="auxiliary",
            method="visualization",
            intent="Synthetic fixture only",
            inputs=list(bound),
            expected_outputs=["figure:association"],
            figure_panels=overview_specs,
        )
        summary = run_association_overview_figure(
            out_dir=destination,
            run_dir=sources,
            resolved_inputs={"step_id": "association", "inputs": bound},
            step_id="association",
            figure_product="association",
            panel_specs=overview_specs,
        )
        assert not validate_step_planned_figure_contract_binding(
            step=overview_step, out_dir=destination, step_summary=summary
        )
        for path in summary["source_data_files"]:
            projections[name][path] = sha(destination / path)
        for product in ("prediction", "association"):
            findings = audit_svg_text_layout(destination / f"{product}.svg")
            qa.append(
                {
                    "delivery": name,
                    "figure": product,
                    "findings": [
                        finding.model_dump(mode="json") for finding in findings
                    ],
                }
            )
    assert projections["paper"] == projections["presentation"], (
        "numeric projections or source data changed with presentation"
    )
    assert source_sha == {path.name: sha(path) for path in sources.glob("*.csv")}, (
        "sealed input bytes changed"
    )
    report = {
        "schema_version": "easyicu.figure_presentation_engineering_qa/1",
        "scope": "synthetic deterministic export QA; no research run or publication authority",
        "source_sha256": source_sha,
        "projection_sha256": projections,
        "layout_qa": qa,
        "export_sha256": {
            str(path.relative_to(output)): sha(path)
            for name in styles
            for path in (output / name).iterdir()
            if path.suffix in {".svg", ".pdf", ".png"}
        },
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    errors = [
        finding
        for item in qa
        for finding in item["findings"]
        if finding["severity"] == "error"
    ]
    if errors:
        raise ValueError(f"Export layout errors: {errors}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output.resolve())
    print(
        json.dumps(
            {
                "exports": len(result["export_sha256"]),
                "identical_projections": True,
                "layout_findings": sum(
                    len(item["findings"]) for item in result["layout_qa"]
                ),
            }
        )
    )
