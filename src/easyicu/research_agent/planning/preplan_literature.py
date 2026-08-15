"""Prepare and register the literature bundle used before planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from ..literature import LiteratureBundle, build_preplan_literature_bundle
from ..schema import ResearchContext


def prepare_preplan_literature(
    *,
    context: ResearchContext,
    run_dir: Path,
    evidence: Any,
    enable_pubmed: bool,
    pubmed_email: Optional[str],
    pubmed_api_key: Optional[str],
    enable_tavily: bool,
    tavily_api_key: Optional[str],
    tavily_retmax: int,
    tavily_include_domains: Sequence[str],
    bound_seed: Optional[LiteratureBundle] = None,
) -> LiteratureBundle:
    """Retrieve, persist, and register the pre-plan literature authority."""
    bundle = build_preplan_literature_bundle(
        context,
        enable_pubmed=enable_pubmed,
        pubmed_email=pubmed_email,
        pubmed_api_key=pubmed_api_key,
        enable_tavily=enable_tavily,
        tavily_api_key=tavily_api_key,
        tavily_retmax=tavily_retmax,
        tavily_include_domains=tavily_include_domains,
        bound_seed=bound_seed,
    )
    bundle_path = run_dir / "preplan_literature_bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    if evidence.get("preplan_literature_bundle") is None:
        evidence.register_file(
            kind="log",
            description=(
                "Pre-plan LiteratureBundle used to shape the hypothesis "
                "blueprint before planner execution."
            ),
            source_path=bundle_path,
            evidence_id="preplan_literature_bundle",
            producer="hypothesis_blueprint",
            generation_mode="deterministic_skill",
        )
    return bundle


__all__ = ["prepare_preplan_literature"]
