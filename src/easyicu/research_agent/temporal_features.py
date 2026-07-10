"""Compatibility import for saved scripts using the pre-package module path.

New code should import from :mod:`easyicu.research_agent.methods.temporal_features`.
"""

from .methods.temporal_features import (
    ID_COL,
    TIME_COL,
    incident_outcome_cohort,
    landmark_cohort,
    onset_times,
)

__all__ = [
    "ID_COL",
    "TIME_COL",
    "incident_outcome_cohort",
    "landmark_cohort",
    "onset_times",
]
