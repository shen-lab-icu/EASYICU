"""Materialize the Planner-final article-design authority bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..reporting.article_contract import build_article_analysis_contract
from .analysis_blueprint import build_analysis_blueprint
from .figure_strategy import build_article_figure_strategy
from .study_design import build_study_design_brief


@dataclass(frozen=True)
class FinalArticleDesignAuthority:
    """The four mutually consistent Planner-final design objects."""

    brief: Any
    contract: Any
    figure_strategy: Any
    blueprint: Any


def materialize_final_article_design_authority(
    *,
    context: Any,
    analysis_type: str,
    run_dir: Path,
    evidence: Any,
) -> FinalArticleDesignAuthority:
    """Build, persist, and register one final design bundle."""

    brief = build_study_design_brief(context, analysis_type=analysis_type)
    contract = build_article_analysis_contract(
        context,
        brief=brief,
        analysis_type=analysis_type,
    )
    figure_strategy = build_article_figure_strategy(
        context,
        analysis_family=brief.analysis_family,
    )
    blueprint = build_analysis_blueprint(
        context,
        brief=brief,
        contract=contract,
        figure_strategy=figure_strategy,
    )
    payloads = (
        ("study_design_brief_final", "study_design_brief.final.json", brief),
        (
            "article_analysis_contract_final",
            "article_analysis_contract.final.json",
            contract,
        ),
        (
            "article_figure_strategy_final",
            "article_figure_strategy.final.json",
            figure_strategy,
        ),
        ("analysis_blueprint_final", "analysis_blueprint.final.json", blueprint),
    )
    for evidence_id, filename, payload in payloads:
        path = run_dir / filename
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        if evidence.get(evidence_id) is None:
            evidence.register_file(
                kind="log",
                description=(
                    "Planner-final article design authority bound to "
                    f"analysis_type={analysis_type}."
                ),
                source_path=path,
                evidence_id=evidence_id,
                producer="planner_contract_finalizer",
                generation_mode="deterministic_skill",
            )
    return FinalArticleDesignAuthority(
        brief=brief,
        contract=contract,
        figure_strategy=figure_strategy,
        blueprint=blueprint,
    )


__all__ = [
    "FinalArticleDesignAuthority",
    "materialize_final_article_design_authority",
]
