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
    LatentClassModelDesign,
    validate_executed_method_design,
)
from ..schema import EvidenceRecord, ResearchContext, RESEARCH_CONTEXT_SCHEMA_VERSION
from .runtime_artifacts import verified_run_evidence_path


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
        or re.search(r"\bExecuted (?:time design|class model)\s*:", text, re.I)
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
        return (
            "Executed time design: each coordinate was summarized by its maximum in "
            f"{design.n_windows} consecutive {design.window_width_hours}-hour windows "
            f"from {design.window_start_hours} to {design.window_end_hours} hours "
            f"{relation} {_reader_anchor(design.anchor)}, and a record entered the "
            f"model with at least {design.minimum_observed_windows} observed windows"
        )
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
    return (
        "Executed class model: a diagonal Gaussian latent class mixture fitted to "
        f"pooled coordinate-wise z-scores for {grid} classes, with the class count "
        "chosen by the minimum Bayesian information criterion and a prespecified "
        f"minimum class proportion of {proportion} of records"
    )


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
            or raw.get("schema_version") != RESEARCH_CONTEXT_SCHEMA_VERSION
        ):
            return ()
        context = ResearchContext.model_validate(raw)
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
