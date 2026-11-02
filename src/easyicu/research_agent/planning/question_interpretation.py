"""Interpret a question's explicit substance form and uncertainty requirements.

Owner
-----
Some substances are both a level measured and a drug, fluid or blood product
given: serum albumin and albumin given intravenously, a platelet count and a
platelet transfusion.  The Web reader (``webserver.study_intent``) reads which
form the question names from its words and records each decided reading in
the run's data constraints (:data:`QUESTION_SUBSTANCE_FORMS_KEY`).  A plan
whose primary exposure reads the substance in the other form answers another
question, so planning stops (:data:`SUBSTITUTED_REASON`) rather than analyse
it.  A substance the question names without saying which form is recorded by
no one here: the plan's choice of form is shown for review.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..schema import AnalysisPlan, ResearchContext

#: The data-constraints key the Web host records the readings under.
QUESTION_SUBSTANCE_FORMS_KEY = "question_substance_forms"
#: Why planning stops when the plan reads the other form.  Stable.
SUBSTITUTED_REASON = "question_substance_form_substituted"
#: Why a recorded reading is refused: the host wrote it, so it is the host's defect.
MALFORMED_REASON = "question_substance_forms_malformed"

_FORM_WORDS = {"given": "as given", "measured": "as a measured level"}


class QuestionSubstanceFormError(ValueError):
    """A typed refusal of this owner's inputs; ``reason_code`` names it."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code


class QuestionSubstanceForm(BaseModel):
    """A substance the question names in one form, and the columns of its other form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concepts: list[str] = Field(min_length=1, max_length=8)
    form: Literal["given", "measured"]
    other: list[str] = Field(min_length=1, max_length=8)
    evidence: str = Field(min_length=1, max_length=120)


def question_substance_forms(
    context: ResearchContext,
) -> tuple[QuestionSubstanceForm, ...]:
    """The forms the Web reader read in the question, from the run's data constraints.

    Absent on a run the Web did not launch, or whose question decides no
    substance's form.  A malformed record is refused, never read as none.
    """

    raw = (
        getattr(context.user_preferences, "data_constraints", None)
        if context.user_preferences
        else None
    )
    if not raw:
        return ()
    try:
        constraints = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError as exc:
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, "the run's data constraints are not JSON"
        ) from exc
    if not isinstance(constraints, dict):
        return ()
    entries = constraints.get(QUESTION_SUBSTANCE_FORMS_KEY)
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, f"{QUESTION_SUBSTANCE_FORMS_KEY} is not a list"
        )
    try:
        return tuple(QuestionSubstanceForm.model_validate(entry) for entry in entries)
    except ValidationError as exc:
        raise QuestionSubstanceFormError(
            MALFORMED_REASON, "an entry does not state one substance's form"
        ) from exc


@dataclass(frozen=True)
class SubstanceFormSubstitution:
    """The plan's primary exposure reads a substance in the form the question did not name."""

    stated: QuestionSubstanceForm
    used: str

    def message(self) -> str:
        other = "measured" if self.stated.form == "given" else "given"
        return (
            f"The question names {self.stated.evidence!r} {_FORM_WORDS[self.stated.form]} "
            f"({', '.join(self.stated.concepts)}); the plan's primary exposure "
            f"{self.used!r} reads it {_FORM_WORDS[other]}."
        )


def substance_form_substitutions(
    forms: Sequence[QuestionSubstanceForm],
    *,
    plan: AnalysisPlan,
    relatives: Callable[[str], frozenset[str]],
) -> tuple[SubstanceFormSubstitution, ...]:
    """Each primary exposure of ``plan`` that reads a named substance in its other form.

    ``relatives`` gives the run's columns related to a concept
    (``question_requirements.concept_relatives``): a column of the other form
    that no concept of the named form relates to is a substitution.
    """

    sources = dict.fromkeys(
        source
        for step in plan.steps
        for source in step.required_primary_exposure_sources()
    )
    found: list[SubstanceFormSubstitution] = []
    for stated in forms:
        named = frozenset().union(*(relatives(concept) for concept in stated.concepts))
        other = frozenset().union(*(relatives(concept) for concept in stated.other))
        found.extend(
            SubstanceFormSubstitution(stated=stated, used=source)
            for source in sources
            if source in other and source not in named
        )
    return tuple(found)


__all__ = [
    "MALFORMED_REASON",
    "QUESTION_SUBSTANCE_FORMS_KEY",
    "SUBSTITUTED_REASON",
    "QuestionSubstanceForm",
    "QuestionSubstanceFormError",
    "SubstanceFormSubstitution",
    "question_substance_forms",
    "substance_form_substitutions",
]


_CONTRACT_HEADER = re.compile(r"(?m)^Host-bound request to revise\b")
_MARKER = re.compile(r'\n\s*(\{\s*"schema_version"\s*:\s*"easyicu\.plan-change-request/1")')
# A confidence level can precede a named method (e.g. Wilson or
# Clopper-Pearson) before the interval phrase. Restrict the intervening text
# to established method names or an explicitly marked method so
# percentages attached to outcomes are not claimed as confidence levels.
_INTERVAL_METHOD = (
    r"(?:(?i:(?:patient[-\s]+)?cluster[-\s]+robust[-\s]+wald|"
    r"patient[-\s]+weighted[-\s]+hoeffding|hoeffding|wilson|wald|exact|bayesian|normal|percentile|bootstrap|"
    r"katz|jeffreys|profile[-–]likelihood|clopper[-–]pearson|"
    r"agresti[-–]coull|newcombe)|"
    r"[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,2}\s+(?i:method))"
    r"(?:\s+(?i:method|score|exact|bootstrap|based))?(?:\s*法)?"
)
_UNKNOWN_METHOD_NAME = (
    r"(?!(?i:male|female|mortality|survival|death|patients|participants)\b)"
    r"(?-i:[A-Z][a-z]+(?:[-–][A-Z][a-z]+)*)"
)
_INTERVAL = re.compile(
    r"置信区间|信赖区间|confidence\s+intervals?|"
    r"(?<![a-z])(?:\d{1,2}(?:\.\d+)?\s*%\s*)(?:"
    + _INTERVAL_METHOD + r"\s*)?ci(?:s)?\b|"
    r"(?<![a-z])(?:" + _INTERVAL_METHOD + r")\s+ci(?:s)?\b|"
    r"(?<![a-z])\d{1,2}(?:\.\d+)?\s*%\s+" + _UNKNOWN_METHOD_NAME + r"\s+ci(?:s)?\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"(?:不(?:再)?(?:需要|要求|提供|计算|报告|包含|展示|给出)|无需|不要|无须|不做)\s*$|"
    r"(?:\bno|\bwithout|\b(?:do\s+not|don't)\s+(?:need|report|calculate|provide|include)|"
    r"\bdoes\s+not\s+(?:need|require)|\bnot\s+requiring|\bneed\s+not\s+report)"
    r"\s*(?:an?\s+|the\s+)?$",
    re.IGNORECASE,
)
_PERCENT_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*(?:" + _INTERVAL_METHOD + r"\s*)?$"
)
_EXPLICIT_PERCENT_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*(?:%\s*的|percent\b)\s*(?:"
    + _INTERVAL_METHOD + r"\s*)?$", re.IGNORECASE,
)
_EXPLICIT_LEVEL_NEGATION = re.compile(
    r"不(?!只|仅)|无(?!论)|未|(?<!性)别|\b(?:no|not|never|without|neither|don't)\b",
    re.IGNORECASE,
)
_JOINED_PERCENT_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*(?:、|和|及|与|and)\s*"
    r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:" + _INTERVAL_METHOD + r"\s*)?$", re.IGNORECASE,
)
_PERCENT_JOINER_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s*(?:、|和|及|与|and)\s*$",
    re.IGNORECASE,
)
_ADDITIVE_LEVEL = re.compile(
    r"另外|额外|同时|也|新增|(?<!不)再|\b(?:add|also|additionally|in\s+addition)\b",
    re.IGNORECASE,
)
_PERCENT_AFTER = re.compile(
    r"^\s*(?:(?:of|at|to|改为|设为|为)\s*)?[(（]?\s*(\d{1,2}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_NEGATIVE_SUFFIX = re.compile(
    r"^\s*(?:不(?:需要|要求|计算|报告|提供)|无需|不是必须|"
    r"(?:is|are)\s+not\s+(?:needed|required|requested)|not\s+(?:needed|required))",
    re.IGNORECASE,
)
_METHOD_BEFORE = re.compile(r"(?P<method>" + _INTERVAL_METHOD + r")\s*$", re.IGNORECASE)
_METHOD_AFTER = re.compile(
    r"^\s*(?:using|with|via|by|采用|使用|选用|基于|改为)\s*(?P<method>"
    + _INTERVAL_METHOD + r")(?=$|[\s,.;，。；])", re.IGNORECASE,
)
_UNKNOWN_NAMED_BEFORE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*%\s+"
    r"(?P<method>" + _UNKNOWN_METHOD_NAME + r")\s*$"
)


def _interval_method(text: str) -> str:
    """Normalize an explicitly named method; unsupported names stay distinct."""
    value = re.sub(r"\s+", " ", text.casefold().strip()).replace("–", "-")
    value = re.sub(r"(?:\s+(?:method|score|exact|bootstrap|based)|\s*法)$", "", value)
    compact = re.sub(r"[-\s]+", "_", value)
    if compact in {"patient_cluster_robust_wald", "cluster_robust_wald"}:
        return "patient_cluster_robust_wald"
    if compact in {"patient_weighted_hoeffding", "hoeffding"}:
        return "patient_weighted_hoeffding"
    return compact


@dataclass(frozen=True)
class UncertaintyRequirement:
    requested: bool
    confidence_levels: tuple[float, ...]
    interval_methods: tuple[str, ...] = ()
    unresolved: bool = False


def amendment_user_messages(contract: str) -> tuple[str, ...]:
    """Extract only path-free user messages from the exact host-bound envelope.

    A malformed candidate is refused, never replaced by planner-authored text.
    Other contract forms, including plan repair prompts, carry no amendment.
    """
    if not contract:
        return ()
    decoder = json.JSONDecoder()
    messages: list[str] = []
    for header in _CONTRACT_HEADER.finditer(contract):
        match = _MARKER.search(contract, header.end(), min(len(contract), header.end() + 2_000))
        if match is None:
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        try:
            data, _ = decoder.raw_decode(contract[match.start(1):])
        except json.JSONDecodeError as exc:
            raise ValueError("host_bound_plan_revision_contract_unreadable") from exc
        if not isinstance(data, dict) or data.get("schema_version") != "easyicu.plan-change-request/1":
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        message = data.get("user_message")
        if not isinstance(message, str) or not 0 < len(message) <= 12_000:
            raise ValueError("host_bound_plan_revision_contract_unreadable")
        messages.append(message)
    return tuple(messages)


def requested_uncertainty(*texts: str) -> UncertaintyRequirement:
    """Find explicit interval requests, leaving generic uncertainty unspecified.

    A negation is checked directly before each mention, rather than suppressing
    unrelated positive requests in the same sentence. Requested confidence
    levels are captured only adjacent to the confidence-interval wording.
    """
    requested = False
    unresolved = False
    levels: set[float] = set()
    methods: set[str] = set()
    for text in texts:
        had_prior_requirement = requested
        normalized = unicodedata.normalize("NFKC", str(text or ""))
        abbreviation_unresolved = False
        abbreviation_pattern = (
            r"\bci\s+at\s+\d{1,2}(?:\.\d+)?\s*(?:%|percent\b)|"
            r"\d{1,2}(?:\.\d+)?\s*(?:%\s*的|percent)\s*ci\b"
        )
        for abbreviation in re.finditer(abbreviation_pattern, normalized, re.IGNORECASE):
            end = min((i for mark in ",，;；。.!?？"
                       if (i := normalized.find(mark, abbreviation.end())) >= 0), default=len(normalized))
            tail = normalized[abbreviation.end():end]
            defined = re.match(r"\s*(?:of\s+baseline\s+)?(?:is|means|represents)\s+(?:the\s+)?cardiac\s+index\b", tail, re.IGNORECASE)
            if not defined:
                abbreviation_unresolved = True
        change_unresolved = bool(
            re.search(r"(?:不要|不|不能|别)[^,，;；。]{0,30}(?:改为|改成)|\b(?:do\s+not|don't|never)\s+change(?:\s+it)?\s+to\b", normalized, re.IGNORECASE)
            or (had_prior_requirement and not _INTERVAL.search(normalized)
                and re.search(r"\d{1,2}(?:\.\d+)?\s*(?:%|percent\b)", normalized, re.IGNORECASE))
        )
        unresolved = unresolved or abbreviation_unresolved or change_unresolved
        # Separate user messages can amend a previously stated level, while
        # two levels named in the same message are both requirements.
        stated_levels: set[float] = set()
        saw_level = False
        stated_methods: set[str] = set()
        saw_method = False
        clarified = False
        explicit_withdrawal = False
        ambiguous_message = False
        bound_level_count = 0
        for match in _INTERVAL.finditer(normalized):
            before = normalized[max(0, match.start() - 35):match.start()]
            after = normalized[match.end():match.end() + 45]
            # Ignore an adjacent level when testing the negation itself.
            joined = _JOINED_PERCENT_BEFORE.search(before)
            embedded = re.match(r"(\d{1,2}(?:\.\d+)?)\s*%", match.group())
            prefix = _PERCENT_JOINER_BEFORE.search(before) if embedded else None
            before_value = _PERCENT_BEFORE.search(before)
            clause_prefix = re.split(r"[,，;；。.!?？]", before)[-1]
            if (re.search(r"不(?!只|仅)|无(?!论)|未|(?<!性)别|禁止|\b(?:no|not|never|without|don't)\b", clause_prefix, re.IGNORECASE)
                    and not _NEGATION.search(before)):
                unresolved = True
                ambiguous_message = True
            if re.match(r"\s*(?:禁止|暂时不用|没有必要|should\s+never\b|are\s+unnecessary\b)", after, re.IGNORECASE):
                unresolved = True
                ambiguous_message = True
            if before_value is None and re.match(r"置信区间|信赖区间|confidence\s+intervals?", match.group(), re.IGNORECASE):
                explicit = _EXPLICIT_PERCENT_BEFORE.search(before)
                prefix_text = before[:explicit.start()] if explicit else ""
                prefix_text = re.split(r"[,，;；。.!?？]", prefix_text)[-1]
                # Do not reinterpret a negated or ambiguous statement; this
                # narrow extension only recognizes an explicit positive level.
                if explicit:
                    multiple = re.search(r"\d+(?:\.\d+)?\s*(?:%|percent)?\s*(?:和|及|与|and)\s*$", prefix_text, re.IGNORECASE)
                    suffix = bool(after.strip(" \t\r\n。.!！?？"))
                    if _EXPLICIT_LEVEL_NEGATION.search(prefix_text) or multiple or suffix:
                        unresolved = True
                        ambiguous_message = True
                    else:
                        before_value = explicit
                        clarified = bool(re.search(r"(?:改为|改成|只(?:要|报告)|replace|only|instead)\s*$", prefix_text, re.IGNORECASE))
            interval_prefix = re.sub(r"ci(?:s)?\b$", "", match.group(), flags=re.IGNORECASE)
            method_match = _METHOD_BEFORE.search(interval_prefix) or _METHOD_BEFORE.search(before)
            method_match = method_match or _METHOD_AFTER.match(after)
            unknown_named = (_UNKNOWN_NAMED_BEFORE.search(interval_prefix)
                             or _UNKNOWN_NAMED_BEFORE.search(before)) if method_match is None else None
            method_match = method_match or unknown_named
            method = _interval_method(method_match.group("method")) if method_match else None
            adjacent = joined or prefix or before_value or (unknown_named if unknown_named and unknown_named.string == before else None) or (method_match if method_match and not embedded and method_match.string == before else None)
            negation_prefix = before[:adjacent.start()] if adjacent else before
            value = embedded or before_value or unknown_named or _PERCENT_AFTER.search(after)
            if joined:
                values = joined.groups()
            elif prefix and value:
                values = (prefix.group(1), value.group(1))
            else:
                values = (value.group(1),) if value else ()
            matches = ([(joined, 1), (joined, 2)] if joined else
                       [(prefix, 1), (value, 1)] if prefix and value else
                       [(value, 1)] if value else [])
            valid_values = []
            for token, group in matches:
                raw_level = token.group(group)
                position = token.start(group)
                source = token.string
                # An embedded abbreviation starts at the numeric token; in
                # that case inspect the original text just before the match.
                if token is embedded:
                    source = normalized
                    position += match.start()
                signed = bool(re.search(r"[-−﹣]\s*$", source[:position]))
                if signed or not 0 < float(raw_level) < 100:
                    unresolved = True
                    ambiguous_message = True
                else:
                    valid_values.append(raw_level)
            values = tuple(valid_values)
            bound_level_count += len(values)
            if _NEGATION.search(negation_prefix) or _NEGATIVE_SUFFIX.match(after):
                if not values and not method and len(_INTERVAL.findall(normalized)) == 1:
                    explicit_withdrawal = True
                # Removing one explicitly named level does not withdraw a
                # different level the researcher also requested.
                if values and levels:
                    levels.difference_update(float(value) / 100.0 for value in values)
                    requested = bool(levels)
                    stated_levels = set(levels)
                else:
                    requested = False
                    levels.clear()
                    stated_levels.clear()
                if method and methods:
                    methods.discard(method)
                    stated_methods = set(methods)
                else:
                    methods.clear()
                    stated_methods.clear()
                saw_level = False
                saw_method = False
                continue
            requested = True
            if method:
                if not saw_method:
                    if _ADDITIVE_LEVEL.search(before):
                        stated_methods.update(methods)
                    else:
                        methods.clear()
                    saw_method = True
                stated_methods.add(method)
                methods = set(stated_methods)
            for raw_level in values:
                level = float(raw_level) / 100.0
                if 0 < level < 1:
                    if not saw_level:
                        if _ADDITIVE_LEVEL.search(before):
                            stated_levels.update(levels)
                        else:
                            levels.clear()
                            stated_levels.clear()
                        saw_level = True
                    stated_levels.add(level)
                    levels = set(stated_levels)
        percent_count = len(re.findall(r"(?<![\d.])\d{1,3}(?:\.\d+)?\s*(?:%|-?\s*percent\b|per\s+cent\b)", normalized, re.IGNORECASE))
        if _INTERVAL.search(normalized) and percent_count > bound_level_count:
            unresolved = True
            ambiguous_message = True
        if (clarified or explicit_withdrawal) and not ambiguous_message and not abbreviation_unresolved and not change_unresolved:
            unresolved = False
    return UncertaintyRequirement(requested=requested, confidence_levels=tuple(sorted(levels)),
                                  interval_methods=tuple(sorted(methods)), unresolved=unresolved)


def verified_independent_patient_rows(context: Any) -> bool:
    """Prove one row per patient in an already authorized sealed cohort.

    A metadata-only plan, a claimed count, or a missing grouping is not a
    proof. This never reads the raw source or prepares a new cohort.
    """
    from pathlib import Path
    import pandas as pd
    from ..contracts.dependence import PatientGroupResolutionError, resolve_patient_groups
    from ..intake.materialized_metadata import (
        MaterializedMetadataError, load_verified_materialized_cohort_authority,
        read_verified_materialized_cohort_table,
    )
    from .dependence_authority import context_patient_group_authority

    provenance = context.cohort.provenance or {}
    if provenance.get("evidence_stage") == "metadata_only_planning":
        return False
    path = context.cohort_parquet
    grouping = context_patient_group_authority(context)
    if not isinstance(path, str) or not Path(path).is_absolute() or grouping is None:
        return False
    try:
        verified = load_verified_materialized_cohort_authority(Path(path))
        if (verified is None or verified.reference.sha256 != provenance.get("data_provenance_materialization_sha256")
                or verified.authority.cohort_sha256 != provenance.get("data_provenance_cohort_file_sha256")):
            return False
        table = read_verified_materialized_cohort_table(Path(path), verified=verified)
        if grouping.group_source not in table.column_names:
            return False
        values = table.column(grouping.group_source).to_pylist()
        if len(values) < 2 or any(pd.isna(value) for value in values):
            return False
        resolved = resolve_patient_groups(values, requirement=grouping)
        return resolved.cluster_count == len(values)
    except (OSError, ValueError, TypeError, MaterializedMetadataError, PatientGroupResolutionError):
        return False


def primary_distribution_gaps(
    plan: Mapping[str, Any], requirement: UncertaintyRequirement, *,
    independent_patient_rows_verified: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Compare a requested CI against the exact primary distribution contract.

    ``None`` means no applicable interval requirement. An empty first tuple
    means no typed distribution exists, which cannot establish CI coverage.
    Compare the deterministic effective method, including the repeated-unit
    branch. Source/independence authority remains a separate mandatory gate.
    """
    from .analysis_types import canonical_analysis_family

    if not requirement.requested or canonical_analysis_family(plan.get("analysis_type")) != "descriptive_epidemiology":
        return None
    steps = [s for s in plan.get("steps", ())
             if isinstance(s, Mapping) and isinstance(s.get("exposure_outcome_distribution_spec"), Mapping)]
    primary = [s for s in steps if s.get("planned_analysis_role") == "primary"]
    steps = primary or steps
    missing = []
    for step in steps:
        spec = step["exposure_outcome_distribution_spec"]
        level = spec.get("confidence_level")
        effective = (spec.get("repeated_unit_interval_method")
                     if spec.get("dependence") is not None else spec.get("interval_method"))
        accepted = {effective}
        if effective == "patient_cluster_robust_wald":
            accepted.add("wald")
        if (spec.get("interval_method") in {None, "none_counts_only"}
                or not effective
                or isinstance(level, bool) or not isinstance(level, (int, float))
                or not 0 < level < 1
                or any(abs(float(level) - requested) >= 1e-9 for requested in requirement.confidence_levels)
                or ("wilson" in requirement.interval_methods and effective == "wilson"
                    and not independent_patient_rows_verified)
                or any(method not in accepted for method in requirement.interval_methods)):
            missing.append(str(step.get("step_id") or ""))
    return tuple(str(step.get("step_id") or "") for step in steps), tuple(missing)

__all__ += ["UncertaintyRequirement", "amendment_user_messages", "requested_uncertainty", "primary_distribution_gaps"]
