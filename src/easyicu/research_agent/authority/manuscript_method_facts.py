"""Exact source-metadata statements, distinct from scientific result claims.

The pipeline's immutable typed research context supplies the recorded
definitions and windows.  A deterministic owner's sealed step summary supplies
the design it executed (``executed_method_design``): its time grid and
eligibility minimum, its model and selection rule.  These facts quote a
recorded or executed design; they do not validate a clinical definition,
infer a result, or exempt any value from numeric provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Sequence

from ..contracts.executed_method_design import (
    EXECUTED_METHOD_DESIGN_KEY,
    WHOLE_RISK_SET_REASON_WORDS,
    FixedWindowRepresentationDesign,
    LandmarkContinuousSurvivalDesign,
    LandmarkSurvivalDesign,
    LatentClassModelDesign,
    validate_executed_method_design,
)
from ..research_context.typed import (
    RESEARCH_CONTEXT_V2_SCHEMA_VERSION,
    RESEARCH_CONTEXT_V3_SCHEMA_VERSION,
    parse_research_context,
)
from ..schema import EvidenceRecord, RESEARCH_CONTEXT_SCHEMA_VERSION
from .runtime_artifacts import verified_run_evidence_path

# Every typed context version; a context prepared from a materialized extract
# is /3.  Anything else is a legacy untyped artifact and gains no authority.
_TYPED_CONTEXT_VERSIONS = frozenset({
    RESEARCH_CONTEXT_SCHEMA_VERSION,
    RESEARCH_CONTEXT_V2_SCHEMA_VERSION,
    RESEARCH_CONTEXT_V3_SCHEMA_VERSION,
})


class MethodFactAuthorityError(ValueError):
    """Recorded method metadata could not be reproduced from its exact source."""


def is_method_fact_candidate(text: str) -> bool:
    """Reserve the source-fact form so a citation cannot forge validation status."""

    return bool(
        re.search(
            r"\bRecorded [^:\n]{1,100} for the (?:selected exposure|primary outcome)\s*:",
            text,
            re.I,
        )
        or re.search(r"\bExecuted (?:time design|class model|survival design)\s*:", text, re.I)
    )


@dataclass(frozen=True)
class ManuscriptMethodFact:
    source_field: str
    text: str
    source_sha256: str
    evidence_id: str = "research_context"
    #: The windows an executed time design ran on, each (start, end) in hours
    #: from its time origin; reader prose may state a window only as one of them.
    executed_hour_spans: tuple[tuple[float, float], ...] = ()

    @property
    def scaffold(self) -> str:
        return f"{self.text} {{evidence:{self.evidence_id}}}."


def _quoted_source(value: str) -> str:
    text = " ".join(value.split()).strip()
    if not text or any(char in text for char in "{}`\\\n"):
        raise MethodFactAuthorityError("method source text contains unsupported markup")
    # Quote the source value rather than letting its text introduce Markdown
    # structure or pretend to be a fresh scientific assertion.
    text = re.sub(r"([\[\]*_])", r"\\\1", html.escape(text, quote=False))
    return "“" + text.replace("“", "‘").replace("”", "’") + "”"


def _window_text(value: str) -> str:
    match = re.fullmatch(
        r"([a-z][a-z_]+)\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]h", value
    )
    if match is None:
        return value
    anchor, start, end = match.groups()
    anchor = anchor.replace("_", " ").replace("icu", "ICU")
    return f"{start} to {end} hours relative to {anchor}"


def _reader_anchor(anchor: str) -> str:
    return " ".join("ICU" if word == "icu" else word for word in anchor.split("_"))


def _executed_hour_spans(design: object) -> tuple[tuple[float, float], ...]:
    """The windows a design ran on: its whole grid and each grid window, or the
    landmark, the prevalent-exposure and the exposure windows.  A class model
    runs on no time window."""

    if isinstance(design, FixedWindowRepresentationDesign):
        start, width = design.window_start_hours, design.window_width_hours
        cells = tuple(
            (float(start + index * width), float(start + (index + 1) * width))
            for index in range(design.n_windows)
        )
        return tuple(dict.fromkeys((
            (float(design.window_start_hours), float(design.window_end_hours)), *cells,
        )))
    if isinstance(design, LandmarkSurvivalDesign):
        cutoff = float(design.prevalent_exposure_cutoff_hours)
        window_end = float(design.exposure_window_end_hours)
        return tuple(dict.fromkeys((
            (0.0, float(design.landmark_hours)),
            (0.0, window_end),
            *(((0.0, cutoff),) if cutoff > 0 else ()),
            (cutoff, window_end),
            *(
                span
                for hour in design.prevalence_sensitivity_cutoffs_hours or ()
                for span in ((0.0, float(hour)), (float(hour), window_end))
            ),
        )))
    if isinstance(design, LandmarkContinuousSurvivalDesign):
        return tuple(dict.fromkeys((
            (0.0, float(design.landmark_hours)),
            (
                float(design.exposure_window_start_hours),
                float(design.exposure_window_end_hours),
            ),
        )))
    return ()


def _design_text(design: object) -> str:
    """One closed sentence per executed design kind."""

    if isinstance(design, FixedWindowRepresentationDesign):
        relation = "after" if design.window_start_hours >= 0 else "relative to"
        if design.window_evidence is None:
            eligibility = (
                f"with at least {design.minimum_observed_windows} observed windows"
            )
        else:
            eligibility = (
                "when at least one SOFA-2 score was available in at least "
                f"{design.minimum_observed_windows} windows"
            )
        return (
            "Executed time design: each coordinate was summarized by its maximum in "
            f"{design.n_windows} consecutive {design.window_width_hours}-hour windows "
            f"from {design.window_start_hours} to {design.window_end_hours} hours "
            f"{relation} {_reader_anchor(design.anchor)}, and a record entered the "
            f"model {eligibility}"
        )
    if isinstance(design, LandmarkSurvivalDesign):
        return _survival_design_text(design)
    if isinstance(design, LandmarkContinuousSurvivalDesign):
        return _continuous_survival_design_text(design)
    assert isinstance(design, LatentClassModelDesign)
    counts = design.candidate_class_counts
    if counts == list(range(counts[0], counts[-1] + 1)):
        grid = f"{counts[0]} to {counts[-1]}"
    else:
        grid = ", ".join(str(count) for count in counts[:-1]) + f" and {counts[-1]}"
    # A proportion, not a percentage: "5.00%" would bind to the class count 5
    # as readily as to the fraction 0.05 in the same summary.
    places = max(2, 1 - math.floor(math.log10(design.minimum_class_fraction)))
    proportion = f"{design.minimum_class_fraction:.{places}f}"
    if design.model_family == "latent_class_mixed_mode":
        model = (
            "a mixed-mode latent class model with categorical indicators "
            "(class-specific level probabilities) for the declared ordinal "
            "coordinates and Gaussian indicators for the pooled z-scores of any "
            f"continuous coordinate, fitted for {grid} classes"
        )
    else:
        model = (
            "a diagonal Gaussian latent class mixture fitted to pooled "
            f"coordinate-wise z-scores for {grid} classes"
        )
    return (
        f"Executed class model: {model}, with the class count chosen by the minimum "
        "Bayesian information criterion and a prespecified minimum class proportion "
        f"of {proportion} of records"
    )


def _days(values: Sequence[float]) -> str:
    text = [f"{value:g}" for value in values]
    return text[0] if len(text) == 1 else ", ".join(text[:-1]) + f" and {text[-1]}"


def _survival_design_text(design: LandmarkSurvivalDesign) -> str:
    """The landmark risk set, the model and the alternatives the suite ran."""

    # The onset column is the exposure source's first record of the exposure as
    # present (a suite signed before that: its first record of any value), not
    # a verified clinical onset: say so rather than "exposure began".  The PH
    # decision is the suite's typed rule (ProportionalHazardsTestOutcome): the
    # exposure term's test or the Bonferroni global test below alpha.
    present = design.exposure_onset_representation == "first_truthy_event_time"
    timing = (
        "the first time the exposure source recorded the exposure as present"
        if present
        else "the first recorded time of the exposure source"
    )
    recorded = "first recorded as present" if present else "first recorded"
    text = (
        "Executed survival design: the risk set comprised records alive and "
        f"observed at a landmark {design.landmark_hours:g} hours after "
        f"{_quoted_source(design.time_origin)}, with follow-up ending at day "
        f"{design.endpoint_horizon_days:g}; exposure timing was {timing}, which "
        "does not observe exposure begun before that record: exposed records "
        f"{recorded} at or before hour "
        f"{design.prevalent_exposure_cutoff_hours:g} were excluded, and those "
        f"{recorded} before hour {design.exposure_window_end_hours:g} formed the exposed group; "
        "a Cox proportional hazards model with Efron ties, adjusted for "
        f"{design.n_adjustment_covariates} prespecified covariates, estimated the "
        "exposure contrast with Wald intervals, and proportional hazards were "
        "tested with Schoenfeld residuals, judged violated when the test of the "
        "exposure term or a Bonferroni-adjusted global test over all model terms "
        f"rejected at a prespecified alpha of {design.proportional_hazards_alpha:g}"
    )
    # No result vocabulary ("hazard ratio", "confidence interval"): the numeric
    # binder would then accept only result fields for this sentence's numbers.
    if design.time_varying_cutpoints_days:
        text += (
            "; interval-specific contrasts came from a piecewise Cox model split "
            f"at days {_days(design.time_varying_cutpoints_days)} after the landmark"
        )
    if design.rmst_horizon_days is not None:
        text += (
            "; the restricted mean survival time difference was unadjusted over the "
            f"{design.rmst_horizon_days:g} days after the landmark"
        )
    if design.prevalence_sensitivity_cutoffs_hours:
        hours = [f"hour {hour:g}" for hour in design.prevalence_sensitivity_cutoffs_hours]
        analyses = (
            "a prespecified sensitivity analysis"
            if len(hours) == 1
            else "prespecified sensitivity analyses"
        )
        listed = hours[0] if len(hours) == 1 else (
            ", ".join(hours[:-1]) + f" or, separately, {hours[-1]}"
        )
        text += (
            f"; {analyses} of the prevalence definition also excluded the exposed "
            f"records {recorded} at or before {listed} and repeated the reported "
            "adjusted Cox contrast, with no record moved to the comparator group"
        )
    if design.n_adjustment_covariates:
        # The adjusted models drop records with a missing covariate; the
        # unadjusted estimates keep them.  Name both sets, without counts.
        models = "Cox models" if design.time_varying_cutpoints_days else "Cox model"
        whole = "the Kaplan-Meier curves" + (
            " and the restricted mean survival time difference"
            if design.rmst_horizon_days is not None else ""
        )
        text += (
            f"; the {models} used the records with complete covariate data, and "
            f"{whole} used the whole risk set"
        )
    return text


_WINDOW_SUMMARY_WORDS = {
    "max": "highest",
    "min": "lowest",
    "mean": "mean",
    "first": "first",
}


#: Why a prespecified interval model had no estimate, in the Methods' words.
_INTERVAL_NOT_ESTIMABLE_WORDS = {
    "follow_up_ends_by_final_cutpoint": "follow-up ended by its last cut point",
    "interval_without_event": "a follow-up interval had no event",
    "did_not_converge": "its fit did not converge",
    "invalid_contrast_variance": "an interval contrast had no valid variance",
    "non_finite_estimate": "its fit was not finite",
}


def _continuous_survival_design_text(design: LandmarkContinuousSurvivalDesign) -> str:
    """The landmark risk set of a continuous exposure, its model and its checks."""

    start = design.exposure_window_start_hours
    window = (
        f"in the first {design.exposure_window_end_hours:g} hours"
        if start == 0
        else f"from hour {start:g} to hour {design.exposure_window_end_hours:g}"
    )
    unit = (
        f" ({_quoted_source(design.exposure_unit)})"
        if design.exposure_unit is not None
        else ""
    )
    adjustment = (
        f", adjusted for {design.n_adjustment_covariates} prespecified covariates,"
        if design.n_adjustment_covariates
        else ""
    )
    cutpoints = _days(design.time_varying_cutpoints_days)
    split = f"split at days {cutpoints} after the landmark"
    reason = design.interval_model_not_estimable_reason
    intervals = (
        f"interval-specific associations came from a piecewise Cox model {split}"
        if reason is None
        else (
            f"a piecewise Cox model {split} was prespecified for interval-specific "
            f"associations and was not estimable, because {_INTERVAL_NOT_ESTIMABLE_WORDS[reason]}"
        )
    )
    grouping = (
        "grouped the risk set by exposure tertile"
        if design.descriptive_grouping_reason is None
        else (
            "described the whole risk set, without exposure tertiles, because "
            + WHOLE_RISK_SET_REASON_WORDS[design.descriptive_grouping_reason]
        )
    )
    # No result vocabulary ("hazard ratio", "confidence interval"): the numeric
    # binder would then accept only result fields for this sentence's numbers.
    text = (
        "Executed survival design: the risk set comprised records alive and "
        f"observed at a landmark {design.landmark_hours:g} hours after "
        f"{_quoted_source(design.time_origin)}, with follow-up ending at day "
        f"{design.endpoint_horizon_days:g}, and a recorded exposure value: the "
        f"{_WINDOW_SUMMARY_WORDS[design.exposure_window_summary]} value of the "
        f"exposure source {window} after that origin; a Cox proportional hazards "
        f"model with Efron ties{adjustment} estimated the association per "
        f"{design.exposure_increment:g} unit of the exposure's recorded "
        f"scale{unit} with Wald intervals, and proportional hazards were tested "
        "with Schoenfeld residuals, judged violated when the test of the exposure "
        "term or a Bonferroni-adjusted global test over all model terms rejected "
        f"at a prespecified alpha of {design.proportional_hazards_alpha:g}; "
        f"{intervals}; "
        "the linear exposure term was compared with a restricted cubic spline "
        "with knots at the "
        + ", ".join(f"{value:g}th" for value in design.spline_knot_percentiles[:-1])
        + f" and {design.spline_knot_percentiles[-1]:g}th percentiles of the "
        "exposure by a likelihood-ratio test; the descriptive tables and the "
        f"Kaplan-Meier curves {grouping}"
    )
    if design.n_adjustment_covariates:
        # The adjusted models drop records with a missing covariate; the
        # Kaplan-Meier curves keep them.  Name both sets, without counts.
        text += (
            "; the Cox models used the records with complete covariate data, "
            "and the Kaplan-Meier curves used the whole risk set"
        )
    return text


def _executed_design_facts(
    root: Path, records: Sequence[EvidenceRecord],
) -> list[ManuscriptMethodFact]:
    facts: list[ManuscriptMethodFact] = []
    for record in records:
        if not (
            record.kind == "statistic"
            and record.generation_mode == "deterministic_standard"
            and record.produced_by_step
            and Path(record.relative_path).name.endswith("step_summary.json")
        ):
            continue
        path = verified_run_evidence_path(root, record)
        if path is None:
            raise MethodFactAuthorityError("an executed design source has drifted")
        try:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != record.sha256:
                raise MethodFactAuthorityError("executed design changed while being read")
            summary = json.loads(payload)
            raw = summary.get(EXECUTED_METHOD_DESIGN_KEY) if isinstance(summary, dict) else None
            if raw is None:
                continue
            design = validate_executed_method_design(raw)
        except (OSError, UnicodeError, ValueError) as exc:
            raise MethodFactAuthorityError("an executed design cannot be reproduced") from exc
        facts.append(
            ManuscriptMethodFact(
                source_field=f"{record.produced_by_step}.{EXECUTED_METHOD_DESIGN_KEY}",
                text=_design_text(design),
                source_sha256=record.sha256,
                evidence_id=record.evidence_id,
                executed_hour_spans=_executed_hour_spans(design),
            )
        )
    return facts


def load_manuscript_method_facts(
    *,
    root: Path,
    records: Sequence[EvidenceRecord],
) -> tuple[ManuscriptMethodFact, ...]:
    return (*_context_facts(root, records), *_executed_design_facts(root, records))


def _context_facts(
    root: Path, records: Sequence[EvidenceRecord],
) -> tuple[ManuscriptMethodFact, ...]:
    sources = [record for record in records if record.evidence_id == "research_context"]
    if not sources:
        return ()
    if len(sources) != 1:
        raise MethodFactAuthorityError("method facts require one context source")
    source = sources[0]
    if (
        source.kind,
        source.producer,
        source.generation_mode,
        source.produced_by_step,
    ) != (
        "log",
        "pipeline",
        "system",
        None,
    ):
        raise MethodFactAuthorityError(
            "method facts require the pipeline context owner"
        )
    path = verified_run_evidence_path(root, source)
    if path is None:
        raise MethodFactAuthorityError(
            "method context source is missing or has drifted"
        )
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != source.sha256:
            raise MethodFactAuthorityError("method context changed while being read")
        raw = json.loads(payload)
        # Legacy untyped context artifacts gain no new authority.
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") not in _TYPED_CONTEXT_VERSIONS
        ):
            return ()
        context = parse_research_context(raw)
    except (OSError, UnicodeError, ValueError) as exc:
        raise MethodFactAuthorityError("method context cannot be reproduced") from exc

    facts: list[ManuscriptMethodFact] = []
    selected = (
        (context.primary_exposure, "selected exposure"),
        (context.target_outcome, "primary outcome"),
    )
    for name, role in selected:
        if name is None:
            continue
        matches = [
            (index, variable)
            for index, variable in enumerate(context.variables)
            if variable.name == name
        ]
        if len(matches) != 1:
            raise MethodFactAuthorityError(
                "selected method variable is missing or ambiguous"
            )
        index, variable = matches[0]

        def add(field: str, label: str, value: str | None) -> None:
            if value:
                facts.append(
                    ManuscriptMethodFact(
                        source_field=f"variables[{index}].{field}",
                        text=f"Recorded {label} for the {role}: {_quoted_source(value)}",
                        source_sha256=source.sha256,
                    )
                )

        add("description", "source definition", variable.description)
        if variable.analysis_window:
            window_role = (
                variable.analysis_window_role or "observation_window"
            ).replace("_", " ")
            add("analysis_window", window_role, _window_text(variable.analysis_window))
        definition = variable.clinical_definition
        if definition is not None:
            add(
                "clinical_definition.definition_time_anchor",
                "clinical definition time anchor",
                definition.definition_time_anchor.replace("_", " ")
                if definition.definition_time_anchor
                else None,
            )
            add(
                "clinical_definition.validation_status",
                "clinical-validation status",
                definition.validation_status.replace("_", " ")
                if definition.validation_status
                else None,
            )
            conformance = definition.database_conformance.get(context.cohort.database)
            add(
                "clinical_definition.database_conformance." + context.cohort.database,
                "source-database conformance",
                conformance.replace("_", " ") if conformance else None,
            )
    return tuple(facts)
