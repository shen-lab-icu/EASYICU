"""Reader facts compiled from verified counts-only distribution outputs.

The caller must first verify the executor envelope and source-summary digest.
This adapter copies recorded estimates; arithmetic below only checks agreement.
It neither fits a model nor grants clinical, inferential or publication authority.
"""

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping, Sequence

from ..contracts.descriptive_execution import exposure_outcome_distribution_result_receipt_valid
from ..authority.scientific_claims import ScientificClaim


@dataclass(frozen=True)
class DescriptiveReportFact:
    subsection: str
    text: str
    evidence_id: str
    source_sha256: str
    source_fields: tuple[str, ...]
    replaces_claim_ref: str | None = None
    cohort_n: int | None = None

    @property
    def scaffold(self) -> str:
        return f"{self.text} {{evidence:{self.evidence_id}}}."


def _count(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise ValueError("Descriptive fact requires a recorded integer count")
    return value


def _level(value: Any) -> str:
    if type(value) not in {bool, int, float, str}:
        raise ValueError("Unsupported typed distribution level")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite distribution level")
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _quoted(value: str) -> str:
    text = " ".join(value.split())
    if not text or len(text) > 400 or any(c in text for c in "{}[]`<>\\"):
        raise ValueError("Unsafe reader coordinate in descriptive fact")
    text = text.replace("*", "").replace("_", " ")
    return "“" + text.replace("“", "‘").replace("”", "’") + "”"


def _estimate(row: Mapping[str, Any], count: int, denominator: int) -> float:
    value = row.get("estimate_pct")
    if (
        type(value) not in {float, int} or not math.isfinite(value)
        or not 0 <= value <= 100
        or abs(value - 100 * count / denominator) > 1e-5
        or row.get("interval_method") != "none_counts_only"
        or row.get("covariance") != "none_counts_only"
        or any(row.get(key) is not None for key in (
            "ci_low_pct", "ci_high_pct", "confidence_level", "standard_error_pct", "cluster_count",
        ))
    ):
        raise ValueError("Counts-only report fact contradicts its recorded estimate")
    return float(value)


def compile_counts_only_report_facts(
    verified_records: Sequence[Mapping[str, Any]],
    *,
    evidence: Any,
    reader_display_labels: Mapping[str, str],
    scientific_claims: Sequence[ScientificClaim] = (),
) -> tuple[DescriptiveReportFact, ...]:
    """Compile the exact primary counts-only capability, not arbitrary tables."""

    facts: list[DescriptiveReportFact] = []
    for record in verified_records:
        summary = record.get("step_summary", {})
        if not exposure_outcome_distribution_result_receipt_valid(summary):
            continue
        if summary.get("interval_method") != "none_counts_only":
            continue
        if record.get("status") != "ok":
            raise ValueError("A failed record cannot supply report facts")
        source = evidence.get(str(record.get("step_summary_evidence_id") or ""))
        if source is None or source.produced_by_step != record.get("step_id"):
            raise ValueError("Report fact source does not belong to the verified step")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source.evidence_id):
            raise ValueError("Invalid report fact source id")
        estimates = summary["descriptive_estimates"]
        if estimates.get("dependence") is not None or estimates.get("risk_difference") is not None:
            raise ValueError("Counts-only report facts cannot introduce an inferential contrast")
        cohort_n = _count(summary.get("cohort_n"), positive=True)
        prevalence = estimates["exposure_prevalence"]
        outcomes = estimates["outcome_absolute_risks"]
        if len(prevalence) != len(outcomes):
            raise ValueError("Distribution fact level coverage differs")
        levels: dict[int, tuple[str, str, int]] = {}
        typed_levels: set[tuple[type, str]] = set()
        cohort_count = 0
        for position, row in enumerate(prevalence):
            index = _count(row.get("level_index"))
            level = _level(row.get("level"))
            identity = (type(row["level"]), level)
            if index in levels or identity in typed_levels:
                raise ValueError("Duplicate distribution fact level")
            typed_levels.add(identity)
            count = _count(row.get("n"))
            denominator = _count(row.get("denominator"), positive=True)
            if count > denominator or denominator != cohort_n:
                raise ValueError("Prevalence fact denominator differs from the cohort")
            estimate = _estimate(row, count, denominator)
            coordinate = f"{summary['exposure']}={level}"
            label = _quoted(reader_display_labels.get(coordinate, coordinate))
            levels[index] = (level, label, count)
            cohort_count += count
            prefix = f"descriptive_estimates.exposure_prevalence[{position}]"
            facts.append(DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=f"Exposure prevalence in the {label} group was {count:,} of {denominator:,} observations ({estimate:.2f}%)",
                evidence_id=source.evidence_id, source_sha256=source.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in ("level", "n", "denominator", "estimate_pct")),
                cohort_n=cohort_n,
            ))
        if cohort_count != cohort_n:
            raise ValueError("Prevalence fact counts do not partition the cohort")
        seen_outcomes: set[int] = set()
        outcome_label = _quoted(reader_display_labels.get(summary["outcome"], summary["outcome"]))
        for position, row in enumerate(outcomes):
            index = _count(row.get("level_index"))
            level = _level(row.get("level"))
            if index in seen_outcomes or index not in levels or level != levels[index][0]:
                raise ValueError("Outcome fact level does not match its prevalence level")
            seen_outcomes.add(index)
            _, label, count = levels[index]
            events = _count(row.get("events"))
            denominator = _count(row.get("denominator"), positive=True)
            if events > denominator or denominator != count:
                raise ValueError("Outcome fact denominator differs from its exposure group")
            estimate = _estimate(row, events, denominator)
            prefix = f"descriptive_estimates.outcome_absolute_risks[{position}]"
            matching_claims = [claim for claim in scientific_claims if (
                claim.evidence_id == source.evidence_id
                and claim.step_id == record["step_id"]
                and claim.claim_type == "descriptive_absolute_risk"
                and claim.exposure == f"{summary['exposure']}={level}"
                and claim.outcome == summary["outcome"]
            )]
            if len(matching_claims) > 1:
                raise ValueError("Ambiguous source-bound descriptive claim")
            facts.append(DescriptiveReportFact(
                subsection="Primary outcome",
                text=f"Observed {outcome_label} in the {label} group was {events:,} of {denominator:,} observations ({estimate:.2f}%)",
                evidence_id=source.evidence_id, source_sha256=source.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in ("level", "events", "denominator", "estimate_pct")),
                replaces_claim_ref=matching_claims[0].claim_ref if matching_claims else None,
            ))
    return tuple(facts)


def verified_descriptive_source_records(projected, evidence):
    """Recover the exact primary result contract from its sealed summary."""
    from .writer_evidence import _verified_evidence_json

    records = []
    for row in projected:
        summary = row.get("step_summary", {})
        if not (
            isinstance(summary, dict) and "descriptive_estimates" in summary
            and summary.get("analysis_role") == "primary"
            and summary.get("interval_method") == "none_counts_only"
        ):
            continue
        source = _verified_evidence_json(
            evidence, str(row.get("step_summary_evidence_id") or ""),
            exact_evidence_id=True, expected_kind="statistic",
        )
        records.append({**row, "step_summary": source})
    return records


def compile_primary_counts_only_report_facts(records, *, evidence, reader_display_labels):
    """Shared full-run/report-only admission; loose wrapper counts are not facts."""
    from ..audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
    from ..authority.scientific_claim_registry import load_registered_scientific_claims

    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records, evidence_store=evidence,
    )
    return compile_counts_only_report_facts(
        verified_descriptive_source_records(projected, evidence), evidence=evidence,
        reader_display_labels=reader_display_labels,
        scientific_claims=load_registered_scientific_claims(root=evidence.root, records=evidence.records()),
    )


def _primary_result_regions(manuscript: str):
    """Locate quantitative reporting sections, not interpretive sections.

    Discussion and Conclusion must explain the evidence, but need not repeat
    every count. Their actual claims still pass the scientific/numeric gates.
    """
    for section in ("Abstract", "Results"):
        match = re.search(rf"^## {section}[ \t]*\n(?P<body>.*?)(?=^##\s|\Z)", manuscript, re.M | re.S)
        if match is None:
            continue
        start, end = match.span("body")
        if section == "Abstract":
            block = re.search(r"^\*\*Results:\*\*(?P<body>.*?)(?=^\*\*[^*\n]+:\*\*|\Z)", match["body"], re.M | re.S)
            if block is None:
                continue
            start, end = start + block.start("body"), start + block.end("body")
        yield section, start, end


def _fact_present(body: str, fact: DescriptiveReportFact) -> bool:
    # This checks visibility, not numerical authority; STRICT binding follows.
    visible = re.sub(r"<!--.*?-->|```.*?```", "", body, flags=re.S)
    if fact.replaces_claim_ref and re.search(
        rf"^[ \t]*\{{claim:{re.escape(fact.replaces_claim_ref)}\}}[.!?]?[ \t]*$", visible, re.M,
    ):
        return True
    visible = re.sub(r"^\[\^claim_\d+\]:.*$", "", visible, flags=re.M)
    visible = re.sub(r"\[\^claim_\d+\]|\{evidence:[^}\n]+\}|\[[^\]]+\]\(evidence/[^\n)]*\)", "", visible)
    return " ".join(fact.text.split()) in " ".join(visible.split())


def missing_primary_result_facts(manuscript: str, facts: Sequence[DescriptiveReportFact]):
    """Check every admitted primary metric/level, not merely any result number."""
    if not facts:
        return {}
    regions = {section: manuscript[start:end] for section, start, end in _primary_result_regions(manuscript)}
    return {
        section: missing for section in ("Abstract", "Results")
        if (missing := tuple(fact for fact in facts if not _fact_present(regions.get(section, ""), fact)))
    }


def place_primary_result_summaries(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Carry verified primary counts into the abstract after scientific filtering.

    These are the same observed counts, not new effects, uncertainty estimates,
    literature comparisons or a substitute for interpretive review. Never
    manufacture Discussion or Conclusion content by appending result counts.
    """
    if not facts:
        return manuscript
    owned = {fact.scaffold for fact in facts}
    for section, start, end in reversed(tuple(_primary_result_regions(manuscript))):
        if section == "Results":  # The existing subsection owner handles Results.
            continue
        body = manuscript[start:end]
        seen: set[str] = set()
        lines = []
        for line in body.splitlines():
            if line.strip() in owned:
                if line.strip() in seen:
                    continue
                seen.add(line.strip())
            lines.append(line)
        body = "\n".join(lines) + ("\n" if body.endswith("\n") else "")
        missing = [fact.scaffold for fact in facts if not _fact_present(body, fact)]
        if missing:
            body = "\n\n" + "\n\n".join(missing) + "\n\n" + body.lstrip()
        manuscript = manuscript[:start] + body + manuscript[end:]
    return manuscript


def render_descriptive_report_claims(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Project admitted claim tokens only after the scientific grammar gate.

    Numeric binding still follows this display projection. Keep machine claim
    tokens during Writer repair, where they are the semantic authority.
    """
    # Replace only a complete token matched to the same verified source/level.
    # Other claims and model-authored sentences are not deduplicated by numbers
    # or similarity; an unrelated endpoint can have exactly the same count.
    for fact in facts:
        if fact.replaces_claim_ref:
            quantitative_regions = tuple(_primary_result_regions(manuscript))

            def project(match):
                if any(start <= match.start() < end for _, start, end in quantitative_regions):
                    return fact.scaffold
                # The source-bound claim authorizes both the count and its
                # interpretation ceiling. Never drop that ceiling in an
                # interpretive section when replacing a complete claim token.
                return fact.scaffold + " This was a descriptive, unadjusted, noncausal estimate."

            manuscript = re.sub(
                rf"^[ \t]*\{{claim:{re.escape(fact.replaces_claim_ref)}\}}[.!?]?[ \t]*$",
                project, manuscript, flags=re.M,
            )
    return place_primary_result_summaries(place_descriptive_report_facts(manuscript, facts), facts)


def place_descriptive_report_facts(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Place host-owned Results sentences after filtering, without model prose."""

    section = re.search(r"^## Results[ \t]*$", manuscript, re.M)
    if section is None or not facts:
        return manuscript
    following = re.search(r"^##\s+", manuscript[section.end():], re.M)
    end = section.end() + following.start() if following else len(manuscript)
    body = manuscript[section.end():end]
    seen: set[str] = set()
    owned_lines = {fact.scaffold for fact in facts}
    lines = []
    for line in body.splitlines():
        if line.strip() in owned_lines:
            if line.strip() in seen:
                continue
            seen.add(line.strip())
        lines.append(line)
    body = "\n".join(lines) + ("\n" if body.endswith("\n") else "")
    # Modern descriptive reports have one primary result section. Keep the
    # same registered facts together there instead of filling a false
    # association section with a figure pointer. Legacy layouts stay readable.
    descriptive_heading = re.search(r"^### Descriptive results[ \t]*$", body, re.M)
    destinations = (
        {"Descriptive results": facts} if descriptive_heading is not None else
        {name: tuple(fact for fact in facts if fact.subsection == name)
         for name in dict.fromkeys(fact.subsection for fact in facts)}
    )
    cohort_sources = tuple(fact for fact in facts if fact.cohort_n is not None)
    if descriptive_heading is not None and cohort_sources:
        # Copy the recorded cohort_n, not a number parsed from prose or an
        # outcome denominator. Different analysis cohorts cannot be collapsed
        # into one unqualified cohort count.
        counts = {_count(fact.cohort_n, positive=True) for fact in cohort_sources}
        if len(counts) == 1:
            source = cohort_sources[0]
            destinations["Cohort characteristics"] = (DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=f"The analysis cohort comprised {source.cohort_n:,} observations",
                evidence_id=source.evidence_id, source_sha256=source.source_sha256,
                source_fields=("cohort_n",),
            ),)
    for subsection, subsection_facts in destinations.items():
        heading = re.search(rf"^### {re.escape(subsection)}[ \t]*$", body, re.M)
        if heading is None:
            continue
        next_heading = re.search(r"^###\s+", body[heading.end():], re.M)
        stop = heading.end() + next_heading.start() if next_heading else len(body)
        existing = body[heading.end():stop].splitlines()
        missing = [fact.scaffold for fact in subsection_facts if fact.scaffold not in existing]
        if missing:
            body = body[:heading.end()] + "\n\n" + "\n\n".join(missing) + "\n" + body[heading.end():]
    return manuscript[:section.end()] + body + manuscript[end:]
