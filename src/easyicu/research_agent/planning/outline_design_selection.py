"""An outline's design selection, refused in words the Planner can act on.

Owner
-----
This module checks the design selection an outline carries against the run's
authority through the design owner
(:func:`.design_selection.validate_research_design_selection`) and turns the
owner's refusal into the outline's compile error, which the Planner reads
before it writes the outline again.

The owner's message names what is wrong.  A missing question anchor also
needs the rule it breaks: the outline prompt shows the anchors but not that
the selected design must list one among its required_variables, so a Planner
told only that "a run-specific question anchor" is missing writes the same
selection again.  The refusal therefore names each anchor the selected design
can list, that is each anchor among the run's allowed variables.
"""

from __future__ import annotations

from typing import Sequence

from .design_selection import (
    ResearchDesignSelectionError,
    validate_research_design_selection,
)
from .progressive_contract import ProgressivePlanCompileError, ProgressivePlanOutline

__all__ = ["validate_outline_design_selection"]

_ANCHOR_MISSING = "design_selection_question_anchor_missing"


def validate_outline_design_selection(
    outline: ProgressivePlanOutline,
    *,
    allowed_analysis_types: Sequence[str],
    allowed_variables: Sequence[str],
    allowed_literature_citation_keys: Sequence[str],
    question_anchors: Sequence[str],
    required: bool,
) -> None:
    """Refuse an outline whose design selection the design owner refuses."""

    try:
        validate_research_design_selection(
            outline.design_selection,
            selected_analysis_type=outline.analysis_type,
            allowed_analysis_types=allowed_analysis_types,
            allowed_variables=allowed_variables,
            allowed_literature_citation_keys=allowed_literature_citation_keys,
            question_anchors=question_anchors,
            required=required,
        )
    except ResearchDesignSelectionError as exc:
        message, findings = str(exc), ()
        if exc.reason_code == _ANCHOR_MISSING:
            allowed = set(allowed_variables)
            listable = [
                anchor
                for anchor in dict.fromkeys(
                    str(value or "").strip() for value in question_anchors
                )
                if anchor in allowed
            ]
            if listable:
                named = " or ".join(repr(anchor) for anchor in listable)
                message += (
                    f": list {named} among the selected design's required_variables"
                )
            findings = ({"question_anchors": listable},)
        raise ProgressivePlanCompileError(
            f"progressive_{exc.reason_code}",
            message,
            path=exc.path,
            findings=findings,
        ) from exc
