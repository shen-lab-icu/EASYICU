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


@dataclass(frozen=True)
class DescriptiveReportFact:
    subsection: str
    text: str
    evidence_id: str
    source_sha256: str
    source_fields: tuple[str, ...]

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
            facts.append(DescriptiveReportFact(
                subsection="Primary outcome",
                text=f"Observed {outcome_label} in the {label} group was {events:,} of {denominator:,} observations ({estimate:.2f}%)",
                evidence_id=source.evidence_id, source_sha256=source.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in ("level", "events", "denominator", "estimate_pct")),
            ))
    return tuple(facts)


def place_descriptive_report_facts(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Place host-owned sentences after strict filtering, without model prose."""

    section = re.search(r"^## Results[ \t]*$", manuscript, re.M)
    if section is None or not facts:
        return manuscript
    following = re.search(r"^##\s+", manuscript[section.end():], re.M)
    end = section.end() + following.start() if following else len(manuscript)
    body = manuscript[section.end():end]
    for subsection in dict.fromkeys(fact.subsection for fact in facts):
        heading = re.search(rf"^### {re.escape(subsection)}[ \t]*$", body, re.M)
        if heading is None:
            continue
        next_heading = re.search(r"^###\s+", body[heading.end():], re.M)
        stop = heading.end() + next_heading.start() if next_heading else len(body)
        existing = body[heading.end():stop].splitlines()
        missing = [fact.scaffold for fact in facts if fact.subsection == subsection and fact.scaffold not in existing]
        if missing:
            body = body[:heading.end()] + "\n\n" + "\n\n".join(missing) + "\n" + body[heading.end():]
    return manuscript[:section.end()] + body + manuscript[end:]
