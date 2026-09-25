"""Which registered step figure becomes the article's main figure.

A deterministic figure step can export several surfaces: a composite, a
second main surface and a supplementary one.  The promotion must follow the
plan and the study family's main-figure strategy, not a role vocabulary in
which a supplementary decision curve outranked the calibration composite.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import easyicu.research_agent as ra
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.figures.publication import make_figure_contract
from easyicu.research_agent.figures.skill import (
    _select_existing_step_publication_figure_bundle,
    _step_publication_bundle_rank,
)
from easyicu.research_agent.planning.study_design import infer_study_design_family
from easyicu.research_agent.schema import VariableRole

STEP_ID = "visualization"


def _prediction_context() -> ra.ResearchContext:
    context = ra.ResearchContext(
        research_question="Predict a binary outcome from prespecified variables.",
        cohort=ra.CohortDescriptor(cohort_name="fixture", database="synthetic", n_stays=100),
        variables=[
            ra.ConceptDescriptor(name="age", dtype="float64"),
            ra.ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
        ],
        target_outcome="death",
    )
    assert infer_study_design_family(context) == "prediction"
    return context


def _panels(*specs: tuple[str, str, str, str]) -> list[dict]:
    return [
        {
            "panel_id": panel_id,
            "title": role,
            "role": role,
            "chart_type": chart_type,
            "claim": "The panel renders its registered source table.",
            "evidence_ids": ["source.csv"],
            "metadata": {"placement": placement, "chart_type": chart_type},
        }
        for panel_id, role, chart_type, placement in specs
    ]


COMPOSITE = _panels(
    ("a", "calibration", "calibration_curve", "main"),
    ("b", "model_performance", "roc_curve", "main"),
    ("c", "model_performance", "precision_recall_curve", "main"),
)
VALIDATED_COMPOSITE = [
    *COMPOSITE,
    *_panels(("d", "validation", "metric_dot_interval", "main")),
]
VALIDATION_SURFACE = _panels(
    ("a", "validation", "cohort_split_diagram", "main"),
    ("b", "validation", "metric_dot_interval", "main"),
)
DECISION_CURVE = _panels(("a", "clinical_utility", "decision_curve", "supplementary"))


def _bundle(stem: str, panels: list[dict], order: int) -> dict:
    return {
        "step_id": STEP_ID,
        "stem": stem,
        "order": order,
        "contract_payload": {"figure_id": f"figure:{stem}", "panels": panels},
    }


def _rank(bundles: list[dict], context=None) -> list[str]:
    return [
        bundle["stem"]
        for bundle in sorted(
            bundles, key=lambda item: _step_publication_bundle_rank(item, context=context)
        )
    ]


def test_a_supplementary_surface_never_becomes_the_main_figure() -> None:
    """Dev9 M2: the decision curve was promoted over calibration."""

    bundles = [
        _bundle("visualization", COMPOSITE, order=1),
        _bundle("visualization_supplementary_decision_curve", DECISION_CURVE, order=3),
    ]
    for context in (None, _prediction_context()):
        assert _rank(bundles, context)[0] == "visualization"


def test_the_bundle_carrying_the_hero_leads_among_equals() -> None:
    """Two main surfaces of one renderer: the calibration one leads, not the later one."""

    bundles = [
        _bundle("visualization", COMPOSITE, order=1),
        _bundle("visualization_validation_stability", VALIDATION_SURFACE, order=2),
        _bundle("visualization_supplementary_decision_curve", DECISION_CURVE, order=3),
    ]
    assert _rank(bundles, _prediction_context()) == [
        "visualization",
        "visualization_validation_stability",
        "visualization_supplementary_decision_curve",
    ]


def test_among_equals_the_hero_then_wider_coverage_leads() -> None:
    """Later registration no longer decides between two unready main surfaces."""

    calibration = _panels(("a", "calibration", "calibration_curve", "main"))
    performance = _panels(("a", "model_performance", "roc_curve", "main"))
    performance_and_validation = _panels(
        ("a", "model_performance", "roc_curve", "main"),
        ("b", "validation", "metric_dot_interval", "main"),
    )
    context = _prediction_context()
    # Equal coverage: the bundle with the family's hero role leads.
    assert _rank(
        [_bundle("hero", calibration, order=1), _bundle("later", performance, order=2)],
        context,
    )[0] == "hero"
    # No hero on either side: the bundle covering more family roles leads.
    assert _rank(
        [
            _bundle("wider", performance_and_validation, order=1),
            _bundle("later", performance, order=2),
        ],
        context,
    )[0] == "wider"


def test_a_bundle_the_family_strategy_accepts_comes_first() -> None:
    context = _prediction_context()
    main_utility = _panels(("a", "clinical_utility", "decision_curve", "main"))
    bundles = [
        _bundle("utility", main_utility, order=5),
        _bundle("visualization", VALIDATED_COMPOSITE, order=1),
    ]
    # Without the family, a main-placed primary-result role still ranks first.
    assert _rank(bundles)[0] == "utility"
    assert _rank(bundles, context)[0] == "visualization"


def _register_bundle(
    evidence: EvidenceStore, run_dir: Path, stem: str, panels: list[dict]
) -> None:
    outputs = run_dir / "steps" / STEP_ID / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    svg = outputs / f"{stem}.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>',
        encoding="utf-8",
    )
    png = outputs / f"{stem}.png"
    Image.new("RGB", (10, 10), "white").save(png)
    source = outputs / f"{stem}_source_data.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    contract = make_figure_contract(
        figure_id=f"figure:{stem}",
        core_claim="A registered surface renders its source tables.",
        panels=[{**panel, "evidence_ids": [source.name]} for panel in panels],
        source_data=[source.name],
        statistics_note="Source-bound display.",
    )
    contract_path = outputs / f"{stem}.figure_contract.json"
    contract_path.write_text(contract.to_json(indent=2), encoding="utf-8")
    metadata = {"figure_role": "publication_figure", "step_id": STEP_ID}
    for path, kind in ((svg, "figure"), (png, "figure"), (contract_path, "log"), (source, "table")):
        evidence.register_file(
            kind=kind,
            description="Registered step figure surface.",
            source_path=path,
            evidence_id=f"{kind}_{path.name.replace('.', '_')}",
            produced_by_step=STEP_ID,
            producer="runner",
            generation_mode="deterministic_standard",
            metadata=metadata if kind != "table" else {"step_id": STEP_ID},
        )


def test_the_registered_composite_is_selected_for_promotion(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    evidence = EvidenceStore(run_dir)
    for stem, panels in (
        ("visualization", COMPOSITE),
        ("visualization_validation_stability", VALIDATION_SURFACE),
        ("visualization_supplementary_decision_curve", DECISION_CURVE),
    ):
        _register_bundle(evidence, run_dir, stem, panels)

    bundle = _select_existing_step_publication_figure_bundle(
        evidence, context=_prediction_context()
    )

    assert bundle is not None
    assert (bundle["step_id"], bundle["stem"]) == (STEP_ID, "visualization")
