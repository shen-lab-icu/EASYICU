"""Study-design-aware publication figure renderers.

The base :mod:`figure_skill` deterministically renders the association-family
publication figure (forest + strata + missingness). That single template is
correct for association / descriptive studies but is scientifically wrong for
the other study-design families in ``figure_strategy``: a survival question
answered with an odds-ratio forest, or a prediction question with no ROC /
calibration curve, reads as the *same* figure regardless of the science.

This subpackage adds one deterministic, evidence-bound renderer per family so
the manuscript figure matches the study design:

* ``time_to_event``   -> Kaplan-Meier curve + Cox hazard-ratio forest + follow-up
* ``prediction``      -> ROC + calibration curve + performance metrics
* ``phenotyping``     -> cluster profile heatmap + stability + outcome-by-cluster
* ``causal_emulation``-> covariate-balance love plot + positivity + effect contrast

Each renderer reads the family tables the plan contract already asks the coder
to emit (``cox_summary``, ``model_performance``, ``cluster_characteristics``,
``covariate_balance`` ...) plus the curve-point tables added for KM/ROC/
calibration. A renderer returns ``None`` when its required source data is
absent, so the skill safely falls through to its existing behaviour and no
association-family run regresses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..evidence import EvidenceStore
from ..schema import AnalysisPlan, ResearchContext
from ..planning.study_design_playbook import StudyDesignFamily
from .base import RenderedFigure
from .causal import render_causal_figure
from .phenotype import render_phenotype_figure
from .prediction import render_prediction_figure
from .survival import render_survival_figure

# Families whose deterministic renderer lives here. ``association`` and
# ``descriptive`` are intentionally excluded: the base skill already renders
# them well and intercepting them would be a regression risk.
FAMILY_RENDERERS = {
    "time_to_event": render_survival_figure,
    "prediction": render_prediction_figure,
    "phenotyping": render_phenotype_figure,
    "causal_emulation": render_causal_figure,
}


def render_family_figure(
    family: StudyDesignFamily,
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    """Render the family-appropriate publication figure, or ``None``.

    ``None`` is returned both for families handled elsewhere (association /
    descriptive) and when a family renderer cannot find its required source
    evidence. Callers must treat ``None`` as "fall through to the existing
    association/promotion/skip ladder".
    """

    renderer = FAMILY_RENDERERS.get(str(family))
    if renderer is None:
        return None
    try:
        return renderer(
            context=context,
            plan=plan,
            evidence=evidence,
            run_dir=run_dir,
        )
    except Exception:
        # A renderer that raises must not crash the whole figure stage; the
        # skill falls through to its existing behaviour and records a skip.
        from .base import close_leaked_figures

        close_leaked_figures()
        return None


__all__ = [
    "FAMILY_RENDERERS",
    "RenderedFigure",
    "render_family_figure",
    "render_survival_figure",
    "render_prediction_figure",
    "render_phenotype_figure",
    "render_causal_figure",
]
