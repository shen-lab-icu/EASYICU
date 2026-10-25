"""Study-design-aware publication figure renderers.

The continuous-association renderer uses registered effect and absolute-risk
grids when they are available. The base :mod:`.skill` remains the fallback for
association studies without those tables. A single forest template is
scientifically wrong for the other study-design families in ``figure_strategy``: a survival question
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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..authority.evidence_store import EvidenceStore
    from ..schema import AnalysisPlan, ResearchContext
    from ..planning.study_design_playbook import StudyDesignFamily
    from .base import RenderedFigure


def __getattr__(name: str):
    """Load family renderers only when their public facade is requested.

    Every executor importing a style or contract helper initializes this
    package first. Those helpers do not need the family renderers or their
    evidence-store dependencies in each fresh interpreter.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .base import RenderedFigure
    from .association import render_continuous_association_figure
    from .causal import render_causal_figure
    from .phenotype import render_phenotype_figure
    from .prediction import render_prediction_figure
    from .survival import render_survival_figure

    # Keep the existing callable identities and mutable dispatch dictionary.
    renderers = {
        "association": render_continuous_association_figure,
        "time_to_event": render_survival_figure,
        "prediction": render_prediction_figure,
        "phenotyping": render_phenotype_figure,
        "causal_emulation": render_causal_figure,
    }
    exports = {renderer.__name__: renderer for renderer in renderers.values()}
    exports.update(FAMILY_RENDERERS=renderers, RenderedFigure=RenderedFigure)
    globals().update(exports)
    return exports[name]


def render_family_figure(
    family: StudyDesignFamily,
    *,
    context: ResearchContext,
    plan: AnalysisPlan,
    evidence: EvidenceStore,
    run_dir: Path,
) -> Optional[RenderedFigure]:
    """Render the family-appropriate publication figure, or ``None``.

    ``None`` is returned both for families handled elsewhere (descriptive) and
    when a family renderer cannot find its required source evidence. Callers
    must treat ``None`` as "fall through to the existing
    association/promotion/skip ladder".
    """

    renderers = globals().get("FAMILY_RENDERERS")
    if renderers is None:
        renderers = __getattr__("FAMILY_RENDERERS")
    renderer = renderers.get(str(family))
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
    "render_continuous_association_figure",
    "render_family_figure",
    "render_survival_figure",
    "render_prediction_figure",
    "render_phenotype_figure",
    "render_causal_figure",
]
