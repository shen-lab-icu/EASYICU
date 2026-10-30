"""The decision a job-starting host button answers, and its identity.

Six Copilot buttons start a governed job: generate or revise a plan, prepare
its data, approve or reject its review, retry its execution.  Each answers one
decision the researcher makes about one study.  The workflow projection offers
the decision each family can answer now; the browser echoes the one it acts on
with the submission, and the job submission owner (``host_action_jobs``)
compares it field by field with the decision offered then.  The fields are
reconciliation inputs, never authorization: the submission owners keep every
check they make.

A decision's key names it within its study, not within a conversation, so two
conversations of one project, or two tabs, start one job for it.  The key is
derived here from the decision and the submission's own options; the browser
never sends one.

Leaf module: it imports neither the Copilot package nor a job owner, so the
workflow projection, the submission owner and the job manager can all read it.
"""

from __future__ import annotations

import hashlib
import json
import re
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256 = r"^[0-9a-f]{64}$"
# No colon: identifiers are fields of a colon-joined decision key.
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$"
_OPTIONAL_IDENTIFIER = r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,159})?$"
_CODE = r"^[a-z][a-z0-9_]{2,120}$"
_KEY_DIGEST_CHARS = 16

HostActionCode = Literal[
    "auto_generate_plan",
    "generate_plan",
    "auto_revise_plan",
    "prepare_analysis_data",
    "review_raw_preparation",
    "execute_plan",
    "retry_analysis",
]
ReviewDecision = Literal["approved", "rejected"]


class PlanReviewDecision(BaseModel):
    """Approve or reject the plan review one run is paused at."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["plan_review"] = "plan_review"
    run_id: str = Field(pattern=_IDENTIFIER)
    scientific_configuration_sha256: str = Field(pattern=_SHA256)
    #: The paused requests: each review id with the digest of the authority
    #: it asks for (``review_authority_sha256``).
    review_authority_sha256: str = Field(pattern=_SHA256)


class PlanTransitionDecision(BaseModel):
    """Start the plan job of the action the workflow names next."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["plan_transition"] = "plan_transition"
    next_action_code: str = Field(pattern=_CODE)
    scientific_configuration_sha256: str = Field(pattern=_SHA256)
    #: The run the workflow reads its state from; empty before the first run.
    source_run_id: str = Field(default="", pattern=_OPTIONAL_IDENTIFIER)


class ExecutionRetryDecision(BaseModel):
    """Resume the execution of the run the workflow reads its state from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["execution_retry"] = "execution_retry"
    source_run_id: str = Field(pattern=_IDENTIFIER)
    gate_reason: str = Field(default="", max_length=200)
    scientific_configuration_sha256: str = Field(pattern=_SHA256)


class RawPreparationReviewDecision(BaseModel):
    """A user decision about one exact source preparation, never execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["raw_preparation_review"] = "raw_preparation_review"
    run_id: str = Field(pattern=_IDENTIFIER)
    scientific_configuration_sha256: str = Field(pattern=_SHA256)
    checkpoint_sha256: str = Field(pattern=_SHA256)
    scope_sha256: str = Field(pattern=_SHA256)
    review_authority_sha256: str = Field(pattern=_SHA256)
    nonce: str = Field(pattern=r"^[0-9a-f]{64}$")


HostActionDecision = Annotated[
    Union[PlanReviewDecision, PlanTransitionDecision, ExecutionRetryDecision, RawPreparationReviewDecision],
    Field(discriminator="family"),
]

#: The decision family each job-starting action answers.
ACTION_DECISION_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "auto_generate_plan": "plan_transition",
        "generate_plan": "plan_transition",
        "auto_revise_plan": "plan_transition",
        "prepare_analysis_data": "plan_transition",
        "review_raw_preparation": "raw_preparation_review",
        "execute_plan": "plan_review",
        "retry_analysis": "execution_retry",
    }
)


class HostDecisionOffers(BaseModel):
    """The decision each family can answer for the study now, if any.

    One state can offer several: a plan awaiting approval can also be
    regenerated, and a failed execution can be retried or planned afresh.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_review: Optional[PlanReviewDecision] = None
    plan_transition: Optional[PlanTransitionDecision] = None
    execution_retry: Optional[ExecutionRetryDecision] = None
    raw_preparation_review: Optional[RawPreparationReviewDecision] = None

    def for_family(
        self, family: str
    ) -> Optional[
        Union[PlanReviewDecision, PlanTransitionDecision, ExecutionRetryDecision, RawPreparationReviewDecision]
    ]:
        if family == "plan_review":
            return self.plan_review
        if family == "plan_transition":
            return self.plan_transition
        if family == "execution_retry":
            return self.execution_retry
        if family == "raw_preparation_review":
            return self.raw_preparation_review
        return None


class HostActionRequest(BaseModel):
    """The ``host_action`` a job submission carries from a host button."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1, max_length=160)
    project_id: str = Field(min_length=1, max_length=160)
    action_code: HostActionCode
    decision: HostActionDecision

    @model_validator(mode="after")
    def _decision_answers_its_action(self) -> "HostActionRequest":
        if self.decision.family != ACTION_DECISION_FAMILIES[self.action_code]:
            raise ValueError("host_action_decision_family_mismatch")
        return self


class HostActionTag(BaseModel):
    """The host decision a job answers, recorded on the job itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    project_id: str
    study_context_id: str
    action_code: HostActionCode
    decision_key: str
    action_id: str


class StartingEntry(BaseModel):
    """A decision whose job one submission is starting in this process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_context_id: str
    session_id: str
    action_code: HostActionCode
    decision_key: str
    action_id: str
    started_at: float


class HostActionStarting(BaseModel):
    """What the workflow shows while a decision's job is being started."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_code: HostActionCode
    started_at: float


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def review_authority_sha256(requests: Sequence[Any]) -> Optional[str]:
    """Digest a paused review's requests, or ``None`` when one is unbound."""

    pairs = sorted(
        (str(item.get("review_id") or ""), str(item.get("authority_sha256") or ""))
        for item in requests
        if isinstance(item, Mapping)
    )
    if not pairs or any(not review or not authority for review, authority in pairs):
        return None
    return _sha256_json(pairs)


def retry_options_sha256(
    *, report_only: bool, provider: str, credential_source: str
) -> str:
    """Digest the options a retry is submitted with.

    A retry answers a different decision when the researcher changes what it
    repairs or which model connection runs it, so these options are part of
    its key: a retry still running is never silently reused for another one.
    """

    return _sha256_json([bool(report_only), str(provider), str(credential_source)])


def decision_key(
    decision: Union[PlanReviewDecision, PlanTransitionDecision, ExecutionRetryDecision, RawPreparationReviewDecision],
    *,
    review_decision: str = "",
    retry_options: str = "",
) -> str:
    """Name one decision within its study.

    Digests are shortened to their first 16 hex characters: the key tells a
    study's decisions apart and stays readable in logs and replay rows.
    """

    if isinstance(decision, RawPreparationReviewDecision):
        if review_decision not in ("approved", "rejected"):
            raise ValueError("host_action_review_decision_required")
        # Unlike legacy display keys, the security-sensitive preparation key
        # retains every bit of the exact reviewed decision and its unique nonce.
        return ":".join(("raw-prep", decision.run_id, review_decision,
                         _sha256_json(decision.model_dump(mode="json")), decision.nonce))
    if isinstance(decision, PlanReviewDecision):
        if review_decision not in ("approved", "rejected"):
            raise ValueError("host_action_review_decision_required")
        return ":".join(
            (
                "review",
                decision.run_id,
                review_decision,
                decision.scientific_configuration_sha256[:_KEY_DIGEST_CHARS],
                decision.review_authority_sha256[:_KEY_DIGEST_CHARS],
            )
        )
    if isinstance(decision, PlanTransitionDecision):
        return ":".join(
            (
                "plan",
                decision.scientific_configuration_sha256[:_KEY_DIGEST_CHARS],
                decision.source_run_id or "-",
                decision.next_action_code,
            )
        )
    if not re.fullmatch(r"[0-9a-f]{64}", retry_options or ""):
        raise ValueError("host_action_retry_options_required")
    return ":".join(
        (
            "retry",
            decision.source_run_id,
            retry_options[:_KEY_DIGEST_CHARS],
            decision.gate_reason or "-",
        )
    )


class DecisionKeyFacts(BaseModel):
    """What one decision key names, read back from the key ``decision_key`` made."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["plan_review", "plan_transition", "execution_retry", "raw_preparation_review"]
    #: The reviewed run, the plan's source run ("" before the first run), or
    #: the retried run.
    run_id: str = ""
    review_decision: Literal["", "approved", "rejected"] = ""
    next_action_code: str = ""


_DIGEST_PREFIX = r"[0-9a-f]{%d}" % _KEY_DIGEST_CHARS
_KEY_IDENTIFIER = _IDENTIFIER[1:-1]
_REVIEW_KEY = re.compile(
    rf"review:(?P<run>{_KEY_IDENTIFIER}):(?P<decision>approved|rejected)"
    rf":{_DIGEST_PREFIX}:{_DIGEST_PREFIX}"
)
_PLAN_KEY = re.compile(
    rf"plan:{_DIGEST_PREFIX}:(?P<run>-|{_KEY_IDENTIFIER}):(?P<code>{_CODE[1:-1]})"
)
_RETRY_KEY = re.compile(rf"retry:(?P<run>{_KEY_IDENTIFIER}):{_DIGEST_PREFIX}:.+")
_RAW_PREP_KEY = re.compile(
    rf"raw-prep:(?P<run>{_KEY_IDENTIFIER}):(?P<decision>approved|rejected):[0-9a-f]{{64}}:[0-9a-f]{{64}}"
)


def decision_key_facts(action_key: Any) -> Optional[DecisionKeyFacts]:
    """Read a key ``decision_key`` made; ``None`` for any other key.

    Only the host makes these keys, in the request that starts a decision's
    job (``host_action_jobs``).  A row recorded under another key -- a
    browser-recorded row from before the host recorded them -- is not proof
    of the decision it may resemble.
    """

    key = str(action_key or "")
    raw_preparation = _RAW_PREP_KEY.fullmatch(key)
    if raw_preparation is not None:
        return DecisionKeyFacts(family="raw_preparation_review", run_id=raw_preparation["run"],
                                review_decision=raw_preparation["decision"])
    review = _REVIEW_KEY.fullmatch(key)
    if review is not None:
        return DecisionKeyFacts(
            family="plan_review",
            run_id=review["run"],
            review_decision=review["decision"],
        )
    plan = _PLAN_KEY.fullmatch(key)
    if plan is not None:
        return DecisionKeyFacts(
            family="plan_transition",
            run_id="" if plan["run"] == "-" else plan["run"],
            next_action_code=plan["code"],
        )
    retry = _RETRY_KEY.fullmatch(key)
    if retry is not None:
        return DecisionKeyFacts(family="execution_retry", run_id=retry["run"])
    return None


def host_action_id(session_id: str, action_code: str, action_key: str) -> str:
    """The replay row id of one host action in one conversation."""

    return (
        "host_"
        + hashlib.sha256(
            f"{session_id}\0{action_code}\0{action_key}".encode("utf-8")
        ).hexdigest()[:24]
    )


def mismatched_fields(
    echoed: Union[PlanReviewDecision, PlanTransitionDecision, ExecutionRetryDecision, RawPreparationReviewDecision],
    offered: Optional[
        Union[PlanReviewDecision, PlanTransitionDecision, ExecutionRetryDecision, RawPreparationReviewDecision]
    ],
) -> Tuple[str, ...]:
    """The fields in which an echoed decision differs from the one offered now.

    ``("family",)`` says the study offers no decision of that family now.
    """

    if offered is None or offered.family != echoed.family:
        return ("family",)
    mine = echoed.model_dump()
    theirs = offered.model_dump()
    return tuple(name for name in mine if mine[name] != theirs[name])


__all__ = [
    "ACTION_DECISION_FAMILIES",
    "DecisionKeyFacts",
    "ExecutionRetryDecision",
    "RawPreparationReviewDecision",
    "HostActionCode",
    "HostActionDecision",
    "HostActionRequest",
    "HostActionStarting",
    "HostActionTag",
    "HostDecisionOffers",
    "PlanReviewDecision",
    "PlanTransitionDecision",
    "ReviewDecision",
    "StartingEntry",
    "decision_key",
    "decision_key_facts",
    "host_action_id",
    "mismatched_fields",
    "retry_options_sha256",
    "review_authority_sha256",
]
