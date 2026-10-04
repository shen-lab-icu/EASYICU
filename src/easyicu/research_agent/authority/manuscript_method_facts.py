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
    FixedWindowRepresentationDesign,
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

    # The onset column is the exposure source's first recorded time, not a
    # verified clinical onset: say so rather than "exposure began".
    text = (
        "Executed survival design: the risk set comprised records alive and "
        f"observed at a landmark {design.landmark_hours:g} hours after "
        f"{_quoted_source(design.time_origin)}, with follow-up ending at day "
        f"{design.endpoint_horizon_days:g}; exposure timing was the first recorded "
        "time of the exposure source, which does not observe exposure begun before "
        "that record: exposed records first recorded at or before hour "
        f"{design.prevalent_exposure_cutoff_hours:g} were excluded, and those first "
        f"recorded by hour {design.exposure_window_end_hours:g} formed the exposed group; "
        "a Cox proportional hazards model with Efron ties, adjusted for "
        f"{design.n_adjustment_covariates} prespecified covariates, estimated the "
        "exposure contrast with Wald intervals, and proportional hazards were "
        "tested with Schoenfeld residuals at a prespecified alpha of "
        f"{design.proportional_hazards_alpha:g}"
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
