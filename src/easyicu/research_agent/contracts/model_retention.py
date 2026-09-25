"""How many cohort rows a primary model keeps, and which unmeasured rows it may keep.

A primary model that silently fits a minority of its cohort answers a
different question from the one reviewed: the rows it drops are the rows
whose covariates were not measured, and on routinely collected data that
selection usually tracks severity.  This module owns the thresholds a review
applies to that retention, the evidence record a review reads, and the one
rule deciding when an unmeasured covariate state can be estimated at all.

The rule is shared by the fit (``execution/model_matrix.py``) and by the
host's independent denominator audit (``audits/cross_step.py``), so both
reach the same rows from the same data without trusting each other.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

#: Below this share of evaluable rows the primary model answers another
#: question; the plan cannot be approved as it stands.
RETENTION_BLOCKER_BELOW = 0.50
#: Below this share the loss is material and has to be addressed or disclosed.
RETENTION_MAJOR_BELOW = 0.90
#: The fewest unmeasured rows an explicit unmeasured state is estimated from;
#: the same size the host suppresses in disclosed groups.
MISSING_CATEGORY_MIN_ROWS = 20
#: The measured missing share from which a family template keeps a
#: covariate's unmeasured rows as their own state instead of dropping them.
MISSING_CATEGORY_SHARE_THRESHOLD = 0.05

MISSING_CATEGORY_TOO_FEW_ROWS = "unmeasured_rows_below_minimum"
MISSING_CATEGORY_ONE_OUTCOME = "unmeasured_rows_share_one_outcome"


def estimable_missing_category_rows(
    *,
    rows: Any,
    unmeasured: Mapping[str, Any],
    outcome: Any,
) -> Tuple[Any, Dict[str, Tuple[int, str]]]:
    """Return the fitting rows and the listed covariates whose state is not estimable.

    ``rows`` are the rows every other requirement already admits (outcome,
    exposure and each unlisted covariate observed); ``unmeasured`` maps each
    listed covariate to where it was not measured; all share one index.  An
    unmeasured state is estimable from at least
    :data:`MISSING_CATEGORY_MIN_ROWS` rows and, for a 0/1 outcome, only when
    those rows hold both outcomes -- otherwise its coefficient has no finite
    maximum-likelihood value.  A state that is not estimable keeps nothing:
    its unmeasured rows leave the fit, which can shrink another state below
    the rule, so the rule is applied until nothing changes.  The result does
    not depend on the order of ``unmeasured``.
    """

    import pandas as pd

    kept = rows.astype(bool).copy()
    values = pd.to_numeric(outcome, errors="coerce")
    binary = bool(values.loc[kept].dropna().isin([0, 1]).all())
    demoted: Dict[str, Tuple[int, str]] = {}
    while True:
        newly: Dict[str, Tuple[int, str]] = {}
        for name in sorted(unmeasured):
            if name in demoted:
                continue
            missing = unmeasured[name].astype(bool) & kept
            n_missing = int(missing.sum())
            if n_missing == 0:
                continue
            if n_missing < MISSING_CATEGORY_MIN_ROWS:
                newly[name] = (n_missing, MISSING_CATEGORY_TOO_FEW_ROWS)
            elif binary and int(values.loc[missing].sum()) in {0, n_missing}:
                newly[name] = (n_missing, MISSING_CATEGORY_ONE_OUTCOME)
        if not newly:
            return kept, demoted
        demoted.update(newly)
        for name in newly:
            kept &= ~unmeasured[name].astype(bool)


class CovariateRetention(BaseModel):
    """One declared covariate's missingness among the evaluable rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    handling: Literal["drop_row", "unmeasured_category", "unmeasured_rows_dropped"]
    n_missing: int = Field(ge=0)
    missing_share: float = Field(ge=0.0, le=1.0)
    #: Outcome rates by measurement status; ``None`` below the disclosure size.
    outcome_rate_missing: Optional[float] = None
    outcome_rate_observed: Optional[float] = None
    #: Why an unmeasured state was not estimable, when it was not.
    reason_code: Optional[str] = None


class RequirementRetention(BaseModel):
    """What one primary model requirement keeps of its population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    policy: Literal["drop_missing_baseline", "explicit_missing_category"]
    #: Rows of the population the model is fitted in.
    population_n: int = Field(ge=0)
    #: Population rows whose outcome and exposure are both available.
    evaluable_n: int = Field(ge=0)
    #: Rows the model fits under its declared policy.
    model_n: int = Field(ge=0)
    #: Evaluable rows with every declared covariate measured.
    complete_case_n: int = Field(ge=0)
    #: ``model_n / evaluable_n``; ``None`` when nothing is evaluable.
    retention: Optional[float] = None
    complete_case_retention: Optional[float] = None
    outcome_rate_retained: Optional[float] = None
    outcome_rate_dropped: Optional[float] = None
    covariates: Tuple[CovariateRetention, ...] = ()


class PrimaryModelRetentionEvidence(BaseModel):
    """The review's measurement of primary-model retention on the sealed cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["easyicu.primary_model_retention/1"] = (
        "easyicu.primary_model_retention/1"
    )
    status: Literal[
        "measured",
        "rows_unavailable",
        "not_applicable",
        "not_evaluable",
        "probe_failed",
    ]
    reason_code: Optional[str] = None
    population_source: Optional[str] = None
    #: Digest of the cohort file measured, so a reused review can prove it
    #: still describes the rows execution will read.
    cohort_source_sha256: Optional[str] = None
    requirements: Tuple[RequirementRetention, ...] = ()

    def facts(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


__all__ = [
    "CovariateRetention",
    "MISSING_CATEGORY_MIN_ROWS",
    "MISSING_CATEGORY_ONE_OUTCOME",
    "MISSING_CATEGORY_SHARE_THRESHOLD",
    "MISSING_CATEGORY_TOO_FEW_ROWS",
    "PrimaryModelRetentionEvidence",
    "RETENTION_BLOCKER_BELOW",
    "RETENTION_MAJOR_BELOW",
    "RequirementRetention",
    "estimable_missing_category_rows",
]
