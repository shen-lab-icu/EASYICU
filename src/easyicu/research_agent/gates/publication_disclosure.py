"""Publication disclosure policy: the counts a manuscript reader is shown.

One owner holds the rule every count-bearing publication product follows: the
reader tables, the numbers the manuscript cites and the result tables a run
exports.  Content handed to an external service is a different purpose with
its own, stricter floor, owned by ``gates.figure_privacy`` (figure upload,
which reads count columns by this module's name rule) and ``mcp_policy``
(MCP responses); this module's floor never changes what may leave.

A *small cell* is a subject count from 1 to 10, the cell-size convention of
the CMS policy that most publication rules follow; zero is not a small cell.
What a run does with small cells depends on the licence of its data source:

* ``report`` - the source holds no patient data, so nothing is reviewed.
* ``report_with_review`` - the licence sets no numeric floor (the PhysioNet
  credentialed data use agreement and the AmsterdamUMCdb end user licence
  ask for reasonable care not to disclose identities).  Counts stay exact,
  as STROBE asks, and every small cell is listed for the person who signs
  the report off.
* ``suppress_small_cells`` - the licence requires suppression.  No product
  suppresses cells yet, so publication fails closed instead of printing them.

A source without a declared profile also fails closed.  This module imports
nothing beyond the standard library, so the execution-kernel owners that read
it (the survival suite's first-record-hours table, the figure privacy audit)
add only this file to the kernel's identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Optional

#: A subject count below this and above zero is a small cell.
SMALL_CELL_BELOW = 11

PROFILE_UNDECLARED_REASON = "publication_disclosure_profile_undeclared"
SUPPRESSION_UNAVAILABLE_REASON = "publication_small_cell_suppression_unavailable"


class PublicationDisclosureProfile(str, Enum):
    REPORT = "report"
    REPORT_WITH_REVIEW = "report_with_review"
    SUPPRESS_SMALL_CELLS = "suppress_small_cells"


@dataclass(frozen=True)
class SourceDisclosure:
    """The publication profile of one data source and the licence it follows."""

    source: str
    profile: Optional[PublicationDisclosureProfile]
    licence: str


_PHYSIONET_CREDENTIALED = "PhysioNet Credentialed Health Data Use Agreement"
_PHYSIONET_OPEN_DEMO = "PhysioNet open-access demo (Open Database License)"

#: Canonical data-source registry keys (``easyicu.databases.profiles``).
_SOURCE_PROFILES: dict[str, tuple[PublicationDisclosureProfile, str]] = {
    "miiv": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_CREDENTIALED),
    "mimic": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_CREDENTIALED),
    "eicu": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_CREDENTIALED),
    "hirid": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_CREDENTIALED),
    "sic": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_CREDENTIALED),
    "aumc": (
        PublicationDisclosureProfile.REPORT_WITH_REVIEW,
        "AmsterdamUMCdb end user licence agreement",
    ),
    "mimic_demo": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_OPEN_DEMO),
    "eicu_demo": (PublicationDisclosureProfile.REPORT_WITH_REVIEW, _PHYSIONET_OPEN_DEMO),
}

#: Source tags of generated data that describes no patient.
_NON_PATIENT_SOURCES = frozenset({"synthetic", "test", "mock", "fixture"})

_PROFILE_ORDER = {
    PublicationDisclosureProfile.REPORT: 0,
    PublicationDisclosureProfile.REPORT_WITH_REVIEW: 1,
    PublicationDisclosureProfile.SUPPRESS_SMALL_CELLS: 2,
}


def source_disclosure(source: str) -> SourceDisclosure:
    """The profile of ``source``, a canonical registry key or a non-patient tag.

    The caller resolves registry aliases first; an unknown source has no
    profile (``profile is None``) and publication fails closed on it.
    """

    key = str(source).strip().lower()
    if key in _SOURCE_PROFILES:
        profile, licence = _SOURCE_PROFILES[key]
        return SourceDisclosure(source=key, profile=profile, licence=licence)
    if key in _NON_PATIENT_SOURCES:
        return SourceDisclosure(
            source=key, profile=PublicationDisclosureProfile.REPORT, licence="no patient data",
        )
    return SourceDisclosure(source=key, profile=None, licence="undeclared")


def strictest_profile(
    disclosures: tuple[SourceDisclosure, ...],
) -> Optional[PublicationDisclosureProfile]:
    """The profile a study follows: its strictest source, ``None`` if one is undeclared."""

    if not disclosures or any(item.profile is None for item in disclosures):
        return None
    return max((item.profile for item in disclosures), key=_PROFILE_ORDER.__getitem__)


#: Names whose value counts subjects, records or events.
_COUNT_NAMES = frozenset({
    "at_risk",
    "cell_count",
    "complete_case_n",
    "count",
    "denominator",
    "denominator_n",
    "event_count",
    "event_n",
    "events",
    "excluded",
    "excluded_since_prior_stage",
    "group_events",
    "group_n",
    "group_size",
    "level_count",
    "missing_count",
    "missing_n",
    "n",
    "nonmissing_n",
    "sample_size",
    "stratum_n",
    "subgroup_n",
})

#: Count-shaped names that hold a method setting or a pipeline tally, never a
#: number of subjects.
_NON_SUBJECT_COUNT_NAMES = frozenset({
    "call_count",
    "code_repair_attempts",
    "concept_audit_error_count",
    "concept_repair_attempts",
    "error_count",
    "finding_count",
    "n_attempts",
    "n_bins",
    "n_boot",
    "n_bootstrap",
    "n_categories",
    "n_classes",
    "n_clusters",
    "n_components",
    "n_covariates",
    "n_cutpoints",
    "n_draws",
    "n_estimators",
    "n_features",
    "n_folds",
    "n_groups",
    "n_imputations",
    "n_intervals",
    "n_iter",
    "n_iterations",
    "n_jobs",
    "n_knots",
    "n_levels",
    "n_neighbors",
    "n_parameters",
    "n_params",
    "n_permutations",
    "n_predictors",
    "n_retries",
    "n_seeds",
    "n_splits",
    "n_steps",
    "n_strata",
    "n_timepoints",
    "n_trees",
    "n_variables",
    "n_windows",
    "retry_count",
    "step_count",
    "token_count",
    "warning_count",
})


def count_name_key(name: Any) -> str:
    """A field or column name in the form the count rule reads (``n_events``)."""

    text = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def is_subject_count_name(name: Any) -> bool:
    """Whether a field or column of this name counts subjects, records or events.

    The rule is deliberately wide: a missed count is worse than a listed
    setting, and every listed cell names its field so a reviewer can set a
    setting aside.
    """

    key = count_name_key(name)
    if not key or key in _NON_SUBJECT_COUNT_NAMES:
        return False
    return (
        key in _COUNT_NAMES
        or key.startswith("n_")
        or key.endswith(("_n", "_count", "_events", "_records"))
    )


def small_cell_value(value: Any) -> Optional[int]:
    """The integer ``value`` when it is a small cell (1 to 10), else ``None``.

    A boolean, ``None`` or any text that is not a number is no count.
    """

    try:
        number = Decimal(str(value).strip().replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    count = int(number)
    return count if 0 < count < SMALL_CELL_BELOW else None


__all__ = [
    "PROFILE_UNDECLARED_REASON",
    "PublicationDisclosureProfile",
    "SMALL_CELL_BELOW",
    "SUPPRESSION_UNAVAILABLE_REASON",
    "SourceDisclosure",
    "count_name_key",
    "is_subject_count_name",
    "small_cell_value",
    "source_disclosure",
    "strictest_profile",
]
