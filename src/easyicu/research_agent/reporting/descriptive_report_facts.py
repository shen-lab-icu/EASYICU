"""Reader facts and question coverage from verified descriptive products.

The caller must first verify the executor envelope and source-summary digest.
This adapter copies recorded estimates; arithmetic below only checks agreement.
It neither fits a model nor grants clinical, inferential or publication authority.
"""

import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import math
import re
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from ..contracts.cohort_product_keys import is_closed_cohort_product_key
from ..contracts.primary_cohort import STUDY_POPULATION_PRODUCTS
from ..contracts.descriptive_execution import exposure_outcome_distribution_result_receipt_valid
from ..authority.manuscript_claim_policy import fact_sentence_visible, fact_sentence_in_results
from ..authority.scientific_claims import ScientificClaim
from ..authority.evidence_store import evidence_artifact_basename_stem
from ..authority.numeric_claim_identity import NumericClaim
from ..authority.runtime_artifacts import (
    current_successful_step_records, verified_run_evidence_path,
)
from ..contracts.continuous_outcome_summary import CONTINUOUS_OUTCOME_OUTPUT, continuous_outcome_columns
from ..contracts.exposure_group_rules import (
    grouping_measure_description, grouping_rule_texts, grouping_rule_values,
)
from ..authority.reader_numeric_display import round_reader_numeric_display
from ..orchestration.exposure_group_labels import exposure_group_label_rows
from ..schema import ValidationFinding
from .manuscript_surface import in_sentence_label
from .report_fact import DescriptiveReportFact
from .manuscript_provenance import verified_evidence_json as _verified_evidence_json

#: A host-issued unit a fact sentence may name after its count ("ICU stays",
#: "first ICU stays"): letters, spaces and hyphens only, so no number,
#: placeholder or markup can enter the sentence through it.
_READER_NOUN_PHRASE = re.compile(r"[A-Za-z][A-Za-z -]{0,60}[A-Za-z]")


def is_reader_noun_phrase(text: str) -> bool:
    """Whether ``text`` may follow a count in a host fact sentence."""

    return _READER_NOUN_PHRASE.fullmatch(" ".join(str(text).split())) is not None


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


def _reader_name(value: str) -> str:
    return _quoted(value)[1:-1]


def _analysis_unit(context) -> str:
    provenance = getattr(getattr(context, "cohort", None), "provenance", {}) or {}
    return {"icu_stay": "ICU stays", "patient": "patients"}.get(
        provenance.get("analysis_unit"), "records",
    )


def _join(values):
    return " and ".join(values) if len(values) < 3 else ", ".join(values[:-1]) + " and " + values[-1]


def _combined_facts(facts, text):
    """Keep each level's claim identity while sharing one source-bound sentence.

    Every projection carries all cells and citations printed in that sentence.
    Placement deduplicates the sentence, never the per-level scientific claims.
    """
    fields = tuple(dict.fromkeys(key for fact in facts for key in fact.source_fields))
    citations = tuple(dict.fromkeys(key for fact in facts for key in fact.supporting_evidence_ids))
    return [replace(fact, text=text, source_fields=fields, supporting_evidence_ids=citations)
            for fact in facts]


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


def _reviewed_wilson(row: Mapping[str, Any], count: int, denominator: int,
                     level: float) -> tuple[float, tuple[float, float, float]]:
    """Verify the copied interval against the reviewed independent-row design."""
    estimate, interval = _recorded_share(row, count, denominator)
    if (interval is None or row.get("interval_method") != "wilson"
            or row.get("covariance") != "binomial_independent"
            or abs(interval[2] - float(level)) > 1e-9):
        raise ValueError("Wilson interval differs from its reviewed plan")
    z = NormalDist().inv_cdf((1.0 + float(level)) / 2.0)
    p = count / denominator
    divisor = 1.0 + z * z / denominator
    centre = (p + z * z / (2.0 * denominator)) / divisor
    half_width = z * math.sqrt(p * (1.0 - p) / denominator + z * z / (4.0 * denominator ** 2)) / divisor
    expected = (max(0.0, 100.0 * (centre - half_width)), min(100.0, 100.0 * (centre + half_width)))
    if any(abs(actual - target) > 1e-5 for actual, target in zip(interval[:2], expected)):
        raise ValueError("Wilson interval contradicts its recorded numerator and denominator")
    return estimate, interval


def _writer_distribution_receipt(summary):
    """Admit the owner's observed-exposure subset only with its closed ledger.

    The execution receipt's whole-cohort policy remains unchanged; this is
    writer admission of an explicitly reported subset, never relabelling it
    as the full cohort.
    """
    if summary.get("analysis_set") != "exposure_observed_rows_within_bound_typed_cohort":
        return exposure_outcome_distribution_result_receipt_valid(summary)
    estimates = summary.get("descriptive_estimates", {})
    if estimates.get("analysis_set") != summary["analysis_set"]:
        raise ValueError("Exposure subset disagrees with its recorded estimates")
    reconciliation = summary.get("source_row_count_reconciliation", {})
    source = _count(reconciliation.get("source_rows"), positive=True)
    analyzed = _count(reconciliation.get("analyzed_rows"), positive=True)
    excluded = _count(reconciliation.get("excluded_missing_exposure_rows"))
    if source != analyzed + excluded or analyzed != _count(summary.get("cohort_n"), positive=True):
        raise ValueError("Observed exposure denominator does not reconcile to its source")
    return exposure_outcome_distribution_result_receipt_valid({
        **summary, "analysis_set": "bound_typed_cohort",
        "descriptive_estimates": {**estimates, "analysis_set": "bound_typed_cohort"},
    })


def compile_counts_only_report_facts(
    verified_records: Sequence[Mapping[str, Any]],
    *,
    evidence: Any,
    reader_display_labels: Mapping[str, str],
    scientific_claims: Sequence[ScientificClaim] = (),
    context: Any = None,
    manuscript_language: str = "en",
    plan: Any = None,
) -> tuple[DescriptiveReportFact, ...]:
    """Compile verified primary counts and reviewed independent Wilson intervals."""

    from .manuscript_labels import source_bound_manuscript_labels

    reader_display_labels, label_citations = manuscript_group_labels(evidence, source_bound_manuscript_labels(
        context, reader_display_labels, language=manuscript_language,
    ), language=manuscript_language)
    unit = _analysis_unit(context)
    grouping_record, _grouping_rows, parameters = _groupings(evidence)
    facts: list[DescriptiveReportFact] = []
    for record in verified_records:
        summary = record.get("step_summary", {})
        if not _writer_distribution_receipt(summary):
            continue
        method = summary.get("interval_method")
        if method not in {"none_counts_only", "wilson"}:
            continue
        if method == "wilson":
            estimates = summary["descriptive_estimates"]
            if (summary.get("effective_interval_method", method) != "wilson"
                    or estimates.get("dependence") is not None
                    or estimates.get("risk_difference") is not None):
                # Those designs have their own uncertainty/contrast semantics;
                # retain their existing unsupported-fact behavior.
                continue
            # A recorded interval is not proof that the reviewed plan asked
            # for that level, endpoint or independent-row method.
            matching = [step for step in getattr(plan, "steps", ())
                        if step.step_id == record.get("step_id")]
            spec = matching[0].exposure_outcome_distribution_spec if len(matching) == 1 else None
            reviewed_level = getattr(spec, "confidence_level", None)
            if (spec is None or matching[0].planned_analysis_role != "primary"
                    or getattr(spec, "interval_method", None) != "wilson"
                    or getattr(spec, "exposure", None) != summary.get("exposure")
                    or getattr(spec, "outcome", None) != summary.get("outcome")
                    or summary.get("effective_interval_method", method) != "wilson"
                    or type(reviewed_level) not in {float, int} or not 0 < reviewed_level < 1
                    or type(summary.get("confidence_level")) not in {float, int}
                    or abs(float(summary["confidence_level"]) - float(reviewed_level)) > 1e-9):
                raise ValueError("Wilson report fact differs from its reviewed plan")
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
        subset = summary.get("analysis_set") == "exposure_observed_rows_within_bound_typed_cohort"
        ledger = summary.get("source_row_count_reconciliation", {})
        analysis_n = ledger["source_rows"] if subset else cohort_n
        cohort_field = "source_row_count_reconciliation.source_rows" if subset else "cohort_n"
        exposure = summary["exposure"]
        measure = "exposure"
        window_phrase = ""
        if exposure in parameters:
            measure = in_sentence_label(_reader_name(
                grouping_measure_description(parameters[exposure]).split(" recorded", 1)[0]
            ))
            window_phrase = " in that window" if parameters[exposure]["window"] is not None else ""
            rules = grouping_rule_texts(parameters[exposure], include_measure=False)
            definition = "; ".join(
                _reader_name(text.replace("≤", "at most").replace("≥", "at least")
                             .replace("<", "below").replace(">", "above"))
                for text in rules.values()
            )
            facts.append(DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=f"Groups were defined from {in_sentence_label(grouping_measure_description(parameters[exposure]))}: {definition}",
                evidence_id=grouping_record.evidence_id, source_sha256=grouping_record.sha256,
                source_fields=(f"compiled.{exposure}.derivation",),
                required_result_sections=("Results",),
            ))
        prevalence = estimates["exposure_prevalence"]
        outcomes = estimates["outcome_absolute_risks"]
        if len(prevalence) != len(outcomes):
            raise ValueError("Distribution fact level coverage differs")
        levels: dict[int, tuple[str, str, int]] = {}
        typed_levels: set[tuple[type, str]] = set()
        cohort_count = 0
        shares = []
        share_parts = []
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
            if method == "wilson":
                estimate, interval = _reviewed_wilson(row, count, denominator, reviewed_level)
            else:
                estimate, interval = _estimate(row, count, denominator), None
            coordinate = f"{summary['exposure']}={level}"
            label = _reader_name(reader_display_labels.get(coordinate, coordinate))
            levels[index] = (level, label, count)
            cohort_count += count
            prefix = f"descriptive_estimates.exposure_prevalence[{position}]"
            interval_text = (f"; {100 * interval[2]:g}% Wilson CI, {interval[0]:.2f}% to {interval[1]:.2f}%"
                             if interval is not None else "")
            share_parts.append(f"{count:,} ({estimate:.2f}%{interval_text}) {'had ' if not share_parts else ''}{label}")
            shares.append(DescriptiveReportFact(
                subsection="Cohort characteristics",
                text="",
                evidence_id=source.evidence_id, source_sha256=source.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in (
                    "level", "n", "denominator", "estimate_pct",
                    *(("ci_low_pct", "ci_high_pct", "confidence_level") if interval is not None else ()),
                )),
                cohort_n=analysis_n, cohort_source_field=cohort_field, analysis_unit=unit,
                supporting_evidence_ids=(
                    *label_citations.get(coordinate, ()),
                    *((grouping_record.evidence_id,) if exposure in parameters else ()),
                ),
                step_id=record["step_id"], product_id="table:exposure_outcome_distribution",
                outcome="exposure_distribution",
            ))
        if cohort_count != cohort_n:
            raise ValueError("Prevalence fact counts do not partition the cohort")
        population = f"with {measure} recorded{window_phrase}" if subset else "in the analysis cohort"
        share_text = f"Of the {cohort_n:,} {unit} {population}, {_join(share_parts)}"
        if subset:
            share_text += (
                f"; {ledger['excluded_missing_exposure_rows']:,} of the {analysis_n:,} {unit} "
                f"had no {measure} recorded and are excluded from these shares"
            )
            shares = [replace(fact, source_fields=(*fact.source_fields, *(
                f"source_row_count_reconciliation.{key}"
                for key in ("analyzed_rows", "excluded_missing_exposure_rows", "source_rows")
            ))) for fact in shares]
        abstract_share_text = share_text
        if exposure in parameters:
            recorded_window = grouping_measure_description(parameters[exposure]).split(" recorded", 1)[1]
            abstract_share_text = share_text.replace(
                f"{measure} recorded{window_phrase}", f"{measure} recorded{recorded_window}", 1,
            )
        facts.extend(replace(fact, abstract_text=abstract_share_text)
                     for fact in _combined_facts(shares, share_text))
        seen_outcomes: set[int] = set()
        outcome_label = _reader_name(reader_display_labels.get(summary["outcome"], summary["outcome"]))
        outcome_facts, outcome_parts = [], []
        for position, row in enumerate(outcomes):
            index = _count(row.get("level_index"))
            level = _level(row.get("level"))
            if index in seen_outcomes or index not in levels or level != levels[index][0]:
                raise ValueError("Outcome fact level does not match its prevalence level")
            seen_outcomes.add(index)
            _, label, count = levels[index]
            events = _count(row.get("events"))
            denominator = _count(row.get("denominator"), positive=True)
            missing = _count(row.get("missing_n", 0))
            if (
                events > denominator or denominator > count or missing > count
                or (denominator < count and denominator + missing != count)
            ):
                raise ValueError("Outcome fact denominator differs from its exposure group")
            if method == "wilson":
                estimate, interval = _reviewed_wilson(row, events, denominator, reviewed_level)
            else:
                estimate, interval = _estimate(row, events, denominator), None
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
            interval_text = (f"; {100 * interval[2]:g}% Wilson CI, {interval[0]:.2f}% to {interval[1]:.2f}%"
                             if interval is not None else "")
            outcome_parts.append(f"{estimate:.2f}% ({events:,} of {denominator:,} {unit}{interval_text}) in {label}")
            if missing:
                outcome_parts[-1] += f"; {missing:,} {unit} had missing outcomes"
            outcome_facts.append(DescriptiveReportFact(
                subsection="Primary outcome",
                text="",
                evidence_id=source.evidence_id, source_sha256=source.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in ("level", "events", "denominator", "estimate_pct",
                    *(("ci_low_pct", "ci_high_pct", "confidence_level") if interval is not None else ())))
                + ((f"{prefix}.missing_n",) if "missing_n" in row else ()),
                replaces_claim_ref=matching_claims[0].claim_ref if matching_claims else None,
                outcome_label=outcome_label, group_label=label, estimate_pct=estimate,
                supporting_evidence_ids=label_citations.get(f"{summary['exposure']}={level}", ()),
                step_id=record["step_id"], product_id="table:exposure_outcome_distribution",
                outcome=summary["outcome"],
            ))
        facts.extend(_combined_facts(
            outcome_facts, f"{outcome_label[:1].upper() + outcome_label[1:]} was {_join(outcome_parts)}",
        ))
    return tuple(facts)


def verified_descriptive_source_records(projected, evidence):
    """Recover the exact primary result contract from its sealed summary."""

    records = []
    for row in projected:
        summary = row.get("step_summary", {})
        if not (
            isinstance(summary, dict) and "descriptive_estimates" in summary
            and summary.get("analysis_role") == "primary"
            and summary.get("interval_method") in {"none_counts_only", "wilson"}
        ):
            continue
        source = _verified_evidence_json(
            evidence, str(row.get("step_summary_evidence_id") or ""),
            exact_evidence_id=True, expected_kind="statistic",
        )
        records.append({**row, "step_summary": source})
    return records


def concise_results_word_floor(
    manuscript, *, evidence, records, reader_display_labels=None, context=None,
):
    """Size one counts-only result by its verified facts, never the question.

    A compact Results must state *every* host fact, including the denominator,
    group shares and outcome counts. Unsupported, stale, incomplete or more
    complex outputs retain the normal floor and all other manuscript gates.
    """
    if evidence is None or not records:
        return 200
    current = current_successful_step_records(records)
    result_records = [row for row in current if (
        row.get("step_summary", {}).get("analysis_role")
        in {"primary", "secondary", "sensitivity", "exploratory"}
    )]
    if len(result_records) != 1:
        return 200
    try:
        claims = evidence.authoritative_scientific_claims(records)
        if not claims or any(claim.claim_type != "descriptive_absolute_risk" for claim in claims):
            return 200
        verified = verified_descriptive_source_records(result_records, evidence)
        facts = compile_counts_only_report_facts(
            verified, evidence=evidence, reader_display_labels=reader_display_labels or {},
            scientific_claims=claims, context=context,
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError):
        return 200
    if not facts or not {claim.claim_ref for claim in claims}.issubset(
        {fact.replaces_claim_ref for fact in facts}
    ):
        return 200
    sentences = {fact.text for fact in facts if "Results" in fact.required_result_sections}
    if not sentences or not all(fact_sentence_in_results(manuscript, text) for text in sentences):
        return 200
    return min(200, sum(len(re.findall(r"\b[\w'-]+\b", text)) for text in sentences))


def _compile_grouped_table_one_cohort_report_facts(projected, evidence, *, context=None):
    """Copy the cohort count from a verified grouped Table 1 result.

    The sentence names the cohort Table 1 read: the closed analysis cohort when
    it is bound to one (after every exclusion and landmark), else the study
    cohort as extracted.  Neither is the source database's stay count.
    """


    facts = []
    for row in projected:
        summary = row.get("step_summary", {})
        if not (
            row.get("status") == "ok"
            and isinstance(summary, dict)
            and summary.get("status") == "ok"
            and summary.get("analysis_family") == "grouped_table_one"
        ):
            continue
        source_id = str(row.get("step_summary_evidence_id") or "")
        source = _verified_evidence_json(
            evidence, source_id, exact_evidence_id=True, expected_kind="statistic",
        )
        if source.get("analysis_family") != "grouped_table_one":
            raise ValueError("Grouped Table 1 source lost its analysis family")
        cohort_n = _count(source.get("cohort_n"), positive=True)
        variables = source.get("variables")
        outputs = source.get("output_files")
        if (
            not isinstance(variables, list) or not variables
            or any(not isinstance(name, str) or not name.strip() for name in variables)
            or not isinstance(outputs, dict) or not outputs.get("table:table_one")
        ):
            raise ValueError("Grouped Table 1 source lacks its reviewed roster or output")
        record = evidence.get(source_id)
        if record is None or record.produced_by_step != row.get("step_id"):
            raise ValueError("Grouped Table 1 source does not belong to the verified step")
        cohort = (
            "The analysis cohort"
            if is_closed_cohort_product_key(str(source.get("cohort_input_key") or ""))
            else "The study cohort"
        )
        facts.append(DescriptiveReportFact(
            subsection="Cohort characteristics",
            text=f"{cohort} included {cohort_n:,} {_analysis_unit(context) if context is not None else 'ICU stays'}",
            evidence_id=record.evidence_id,
            source_sha256=record.sha256,
            source_fields=("cohort_n",),
            required_result_sections=("Results",),
        ))
    return tuple(facts)


def _compile_survival_cohort_report_facts(projected, evidence):
    """Copy the landmark cohort counts from a verified signed survival suite.

    The suite owns its own Table 1, so no grouped Table 1 fact names its
    population.  The count is the suite's recorded landmark population: the
    stays still under observation at the landmark, in the analysis unit its
    typed reporting envelope declares.  When the complete-case adjusted models
    dropped records, their recorded count follows as a second fact.
    """

    from ..authority.continuous_survival_scientific_claims import (
        ContinuousSurvivalReporting,
        continuous_survival_reporting_requests_claims,
    )
    from ..authority.survival_scientific_claims import (
        SURVIVAL_REPORTING_KEY,
        SurvivalReporting,
        survival_reporting_requests_claims,
    )

    facts = []
    for row in projected:
        summary = row.get("step_summary", {})
        # The Writer projection keeps the envelope but not its schema version;
        # the verified source summary below is the typed authority.
        if not (
            row.get("status") == "ok"
            and isinstance(summary, dict)
            and summary.get("status") == "ok"
            and SURVIVAL_REPORTING_KEY in summary
        ):
            continue
        source_id = str(row.get("step_summary_evidence_id") or "")
        source = _verified_evidence_json(
            evidence, source_id, exact_evidence_id=True, expected_kind="statistic",
        )
        # A binary suite's /2 envelope or a continuous suite's /1 states the
        # analysis unit; a binary /1 envelope has no typed coordinates.
        if survival_reporting_requests_claims(source):
            reporting = SurvivalReporting.model_validate(source[SURVIVAL_REPORTING_KEY])
        elif continuous_survival_reporting_requests_claims(source):
            reporting = ContinuousSurvivalReporting.model_validate(
                source[SURVIVAL_REPORTING_KEY]
            )
        else:
            continue
        population = _count(source.get("n_landmark_population"), positive=True)
        if "n_source" in source and _count(source["n_source"], positive=True) < population:
            raise ValueError("Survival landmark population exceeds its source cohort")
        unit = " ".join(reporting.analysis_unit.split())
        if not is_reader_noun_phrase(unit):
            raise ValueError("Survival analysis unit is not a reader noun phrase")
        record = evidence.get(source_id)
        if record is None or record.produced_by_step != row.get("step_id"):
            raise ValueError("Survival source does not belong to the verified step")
        # A suite that records its risk set's events states each event count
        # beside its own denominator; a summary written before that keeps its
        # counts-only sentences.
        events = (
            _count(source["n_events_landmark_population"])
            if "n_events_landmark_population" in source else None
        )
        if events is not None and events > population:
            raise ValueError("Survival events exceed their landmark population")
        facts.append(DescriptiveReportFact(
            subsection="Cohort characteristics",
            text=f"The landmark analysis cohort included {population:,} {unit}" + (
                f", with {events:,} events during follow-up" if events is not None else ""
            ),
            evidence_id=record.evidence_id,
            source_sha256=record.sha256,
            source_fields=("n_landmark_population",) + (
                ("n_events_landmark_population",) if events is not None else ()
            ),
            required_result_sections=("Results",),
        ))
        # The adjusted Cox models are complete-case while Kaplan-Meier and the
        # restricted mean use the whole cohort, so one count cannot stand for
        # both: state the models' smaller set, and its events, beside the cohort.
        modelled = _count(source.get("n_complete_case"), positive=True)
        if modelled > population:
            raise ValueError("Survival model set exceeds its landmark population")
        if modelled < population:
            text = (
                f"Of the {population:,} {unit} in the landmark analysis cohort, "
                f"{modelled:,} had complete covariate data and entered the adjusted "
                "survival models"
            )
            fields: tuple[str, ...] = ("n_landmark_population", "n_complete_case")
            if events is not None:
                modelled_events = _count(source.get("n_events"))
                if modelled_events > min(modelled, events):
                    raise ValueError("Survival model events exceed their model set")
                text += f", with {modelled_events:,} events"
                fields += ("n_events",)
            facts.append(DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=text,
                evidence_id=record.evidence_id,
                source_sha256=record.sha256,
                source_fields=fields,
                required_result_sections=("Results",),
            ))
    return tuple(facts)


def _recorded_share(row: Mapping[str, Any], count: int, denominator: int):
    """Return a level's recorded share and, when recorded, its interval.

    The values are copied, never recomputed; arithmetic only checks that the
    share is the count over its denominator and that the interval holds it.
    """
    value = row.get("estimate_pct")
    if (
        type(value) not in {float, int} or not math.isfinite(value)
        or not 0 <= value <= 100
        or abs(value - 100 * count / denominator) > 1e-5
    ):
        raise ValueError("Occurrence fact contradicts its recorded share")
    bounds = tuple(row.get(key) for key in ("ci_low_pct", "ci_high_pct", "confidence_level"))
    if row.get("interval_method") == "none_counts_only":
        if any(bound is not None for bound in bounds):
            raise ValueError("A counts-only occurrence share cannot carry an interval")
        return float(value), None
    if any(type(bound) not in {float, int} or not math.isfinite(bound) for bound in bounds):
        raise ValueError("Occurrence share lacks its recorded interval")
    low, high, level = (float(bound) for bound in bounds)
    if not (0 <= low <= value <= high <= 100 and 0 < level < 1):
        raise ValueError("Occurrence interval does not hold its recorded share")
    return float(value), (low, high, level)


def _compile_study_population_occurrence_report_facts(projected, evidence, reader_display_labels, *, context=None):
    """Copy how often each exposure level occurs in the study cohort.

    A plan that asks how often its exposure occurs answers it on the host's
    republished study cohort (``STUDY_POPULATION_PRODUCTS``), not the narrower
    analysis cohort.  Each level's recorded count, share and interval become
    one Abstract and Results sentence, so the answer does not depend on the
    Writer quoting a secondary table.  The Writer's projection keeps typed
    scalars only, so the executor's interpretation class is read from the
    digest-verified source summary.
    """


    facts = []
    for row in projected:
        summary = row.get("step_summary", {})
        if not (
            row.get("status") == "ok"
            and isinstance(summary, dict)
            and summary.get("status") == "ok"
            and "descriptive_estimates" in summary
            and summary.get("typed_cohort_input") in STUDY_POPULATION_PRODUCTS
        ):
            continue
        source_id = str(row.get("step_summary_evidence_id") or "")
        source = _verified_evidence_json(
            evidence, source_id, exact_evidence_id=True, expected_kind="statistic",
        )
        if source.get("interpretation_class") != "exposure_outcome_distribution":
            continue
        record = evidence.get(source_id)
        if record is None or record.produced_by_step != row.get("step_id"):
            raise ValueError("Occurrence source does not belong to the verified step")
        if source.get("typed_cohort_input") not in STUDY_POPULATION_PRODUCTS:
            raise ValueError("Occurrence source lost its study-cohort binding")
        estimates = source.get("descriptive_estimates")
        if not isinstance(estimates, dict) or estimates.get("schema_version") != (
            "easyicu.exposure_outcome_descriptive_estimates/1"
        ):
            raise ValueError("Occurrence source lacks its descriptive estimates")
        prevalence = estimates.get("exposure_prevalence")
        if not isinstance(prevalence, list) or not prevalence:
            raise ValueError("Occurrence source reports no exposure level")
        exposure = str(source.get("exposure") or "")
        levels: set[str] = set()
        counts: list[int] = []
        denominators: set[int] = set()
        for position, level_row in enumerate(prevalence):
            if not isinstance(level_row, dict) or _count(level_row.get("level_index")) != position:
                raise ValueError("Occurrence levels are not in their recorded order")
            level = _level(level_row.get("level"))
            if level in levels:
                raise ValueError("Duplicate distribution fact level")
            levels.add(level)
            count = _count(level_row.get("n"))
            denominator = _count(level_row.get("denominator"), positive=True)
            if count > denominator:
                raise ValueError("Occurrence count exceeds its denominator")
            counts.append(count)
            denominators.add(denominator)
            share, interval = _recorded_share(level_row, count, denominator)
            label = _quoted(reader_display_labels.get(f"{exposure}={level}", f"{exposure}={level}"))
            prefix = f"descriptive_estimates.exposure_prevalence[{position}]"
            fields = ("level", "n", "denominator", "estimate_pct")
            recorded = f"{share:.2f}%"
            if interval is not None:
                low, high, confidence = interval
                recorded += f"; {100 * confidence:g}% CI, {low:.2f}% to {high:.2f}%"
                fields += ("ci_low_pct", "ci_high_pct", "confidence_level")
            facts.append(DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=(
                    f"In the study cohort, {count:,} of {denominator:,} {_analysis_unit(context)} "
                    f"({recorded}) were in the {label} group"
                ),
                evidence_id=record.evidence_id, source_sha256=record.sha256,
                source_fields=tuple(f"{prefix}.{key}" for key in fields),
            ))
        if len(denominators) != 1 or sum(counts) != denominators.pop():
            raise ValueError("Occurrence counts do not partition the study cohort")
    return tuple(facts)


def compile_primary_counts_only_report_facts(records, *, evidence, reader_display_labels,
                                           context=None, manuscript_language="en", plan=None):
    """Shared full-run/report-only admission; loose wrapper counts are not facts.

    Admits the grouped Table 1 cohort count, a signed survival suite's
    landmark cohort count, a primary counts-only or independent Wilson distribution, the study
    population's exposure occurrence and a prediction's comparison with an
    existing score (``benchmark_report_facts``), which reads ``plan``.
    """
    from ..audits.envelope_consumers import RegisteredOutputEnvelopeConsumer
    from ..authority.scientific_claim_registry import load_registered_scientific_claims
    from .benchmark_report_facts import compile_benchmark_report_facts
    from .manuscript_labels import source_bound_manuscript_labels

    projected = RegisteredOutputEnvelopeConsumer().authoritative_writer_records(
        records, evidence_store=evidence,
    )
    descriptive_facts = compile_counts_only_report_facts(
        verified_descriptive_source_records(projected, evidence), evidence=evidence,
        reader_display_labels=reader_display_labels,
        context=context, manuscript_language=manuscript_language,
        scientific_claims=load_registered_scientific_claims(root=evidence.root, records=evidence.records()),
        plan=plan,
    )
    return (
        *_compile_grouped_table_one_cohort_report_facts(projected, evidence, context=context),
        *_compile_survival_cohort_report_facts(projected, evidence),
        *descriptive_facts,
        *compile_continuous_outcome_report_facts(
            projected, evidence=evidence, reader_display_labels=reader_display_labels,
            context=context, manuscript_language=manuscript_language, plan=plan,
        ),
        *_compile_study_population_occurrence_report_facts(
            projected, evidence,
            source_bound_manuscript_labels(
                context, reader_display_labels, language=manuscript_language,
            ),
            context=context,
        ),
        *compile_benchmark_report_facts(
            projected, evidence=evidence, reader_display_labels=reader_display_labels,
            context=context, manuscript_language=manuscript_language, plan=plan,
        ),
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
    return fact_sentence_visible(body, fact.text)


def missing_primary_result_facts(manuscript: str, facts: Sequence[DescriptiveReportFact]):
    """Check every admitted primary metric/level, not merely any result number."""
    if not facts:
        return {}
    regions = {section: manuscript[start:end] for section, start, end in _primary_result_regions(manuscript)}
    return {
        section: missing for section in ("Abstract", "Results")
        if (missing := tuple(
            fact for fact in facts
            if section in fact.required_result_sections
            and not _fact_present(regions.get(section, ""), (
                replace(fact, text=fact.abstract_text)
                if section == "Abstract" and fact.abstract_text else fact
            ))
        ))
    }


def place_primary_result_summaries(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Carry verified primary counts into the abstract after scientific filtering.

    These are the same observed counts, not new effects, uncertainty estimates,
    literature comparisons or a substitute for interpretive review. Never
    manufacture Discussion or Conclusion content by appending result counts.
    """
    if not facts:
        return manuscript
    abstract_facts = tuple(
        replace(fact, text=fact.abstract_text) if fact.abstract_text else fact
        for fact in facts if "Abstract" in fact.required_result_sections
    )
    if not abstract_facts:
        return manuscript
    owned = {fact.scaffold for fact in abstract_facts}
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
        missing = list(dict.fromkeys(
            fact.scaffold for fact in abstract_facts
            if not _fact_present(body, fact)
        ))
        if missing:
            body = "\n\n" + "\n\n".join(missing) + "\n\n" + body.lstrip()
        manuscript = manuscript[:start] + body + manuscript[end:]
    return manuscript


def render_descriptive_report_claims(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Project admitted claim tokens only after the scientific grammar gate.

    Numeric binding still follows this display projection. Keep machine claim
    tokens during Writer repair, where they are the semantic authority.
    """
    from .manuscript_surface import deduplicate_claim_paragraphs

    manuscript = deduplicate_claim_paragraphs(manuscript)
    # Compress only a consecutive set of complete claims from one verified
    # endpoint/source in interpretive sections. Keep each group and percentage, without
    # repeating every numerator/denominator or inventing an effect contrast.
    regions = []
    for section in ("Discussion", "Conclusion"):
        match = re.search(rf"(?ms)^## {section}[ \t]*\n(?P<body>.*?)(?=^## |\Z)", manuscript)
        if match:
            regions.append(match.span("body"))
    abstract = re.search(r"(?ms)^## Abstract[ \t]*\n(?P<body>.*?)(?=^## |\Z)", manuscript)
    if abstract:
        conclusion = re.search(r"(?ms)^\*\*Conclusions:\*\*(?P<body>.*?)(?=^\*\*[^*\n]+:\*\*|\Z)", abstract["body"])
        if conclusion:
            regions.append(tuple(abstract.start("body") + offset for offset in conclusion.span("body")))
    for start, end in sorted(regions, reverse=True):
        body = manuscript[start:end]
        tokens = {fact.replaces_claim_ref: fact for fact in facts if fact.replaces_claim_ref}
        pattern = r"(?:^[ \t]*\{claim:[A-Za-z0-9_.-]+\}[.!?]?[ \t]*(?:\n|$)\s*)+"

        def summarize(match):
            refs = re.findall(r"\{claim:([^}]+)\}", match.group())
            if len(refs) == 1 and "### Descriptive results" not in manuscript:
                return match.group()
            selected = [tokens.get(ref) for ref in refs]
            if any(fact is None or fact.estimate_pct is None or not fact.group_label or not fact.outcome_label for fact in selected):
                return match.group()
            if len(set(refs)) != len(refs) or len({(fact.evidence_id, fact.source_sha256, fact.outcome_label) for fact in selected}) != 1:
                return match.group()
            # A Writer may choose only one group's token for its conclusion.
            # The same verified outcome product authorizes all its groups.
            owner = selected[0]
            selected = [fact for fact in tokens.values() if (
                fact.evidence_id, fact.source_sha256, fact.outcome_label
            ) == (owner.evidence_id, owner.source_sha256, owner.outcome_label)]
            if len(selected) < 2:
                return match.group()
            values = "; ".join(f"{fact.estimate_pct:.2f}% in the {fact.group_label} group" for fact in selected)
            citations = " ".join(f"{{evidence:{key}}}" for key in dict.fromkeys(
                key for fact in selected for key in (fact.evidence_id, *fact.supporting_evidence_ids)
            ))
            return (f"Observed {selected[0].outcome_label} proportions were {values} "
                    f"{citations}. "
                    "These are descriptive, unadjusted proportions; they do not establish an adjusted or causal effect.\n\n")

        body = re.sub(pattern, summarize, body, flags=re.M)
        manuscript = manuscript[:start] + body + manuscript[end:]
    # Replace only a complete token matched to the same verified source/level.
    # Other claims and model-authored sentences are not deduplicated by numbers
    # or similarity; an unrelated endpoint can have exactly the same count.
    for fact in facts:
        if fact.replaces_claim_ref:
            quantitative_regions = tuple(_primary_result_regions(manuscript))

            def project(match):
                if fact.abstract_text and any(
                    section == "Abstract" and start <= match.start() < end
                    for section, start, end in quantitative_regions
                ):
                    return replace(fact, text=fact.abstract_text).scaffold
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


def restore_descriptive_revision_claims(manuscript: str, facts: Sequence[DescriptiveReportFact]) -> str:
    """Re-admit exact prior rendered claims; all numbers must be rebound later.

    Older report revisions saved the bound draft instead of a canonical Writer
    draft. Remove its old footnote cache and recover only verbatim host claims.
    Free prose and changed values still face the ordinary scientific filter.
    """
    text = re.sub(r"(?m)^\[\^claim_\d+\]:.*$", "", manuscript)
    text = re.sub(r"\[\^claim_\d+\]", "", text)
    for fact in facts:
        if fact.replaces_claim_ref:
            prior = fact.scaffold + " This was a descriptive, unadjusted, noncausal estimate."
            text = re.sub(r"(?m)^[ \t]*" + re.escape(prior) + r"[ \t]*$",
                          lambda _: "{claim:" + fact.replaces_claim_ref + "}", text)
    return text


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
    destinations = {}
    for fact in facts:
        name = ("Descriptive results" if descriptive_heading is not None
                and fact.subsection in {"Cohort characteristics", "Primary outcome"}
                else fact.subsection)
        destinations.setdefault(name, []).append(fact)
    cohort_heading = re.search(r"^### Cohort characteristics[ \t]*$", body, re.M)
    cohort_sources = tuple(fact for fact in facts if fact.cohort_n is not None)
    if (descriptive_heading is not None or cohort_heading is not None) and cohort_sources:
        # Copy the recorded cohort_n, not a number parsed from prose or an
        # outcome denominator. Different analysis cohorts cannot be collapsed
        # into one unqualified cohort count.
        counts = {_count(fact.cohort_n, positive=True) for fact in cohort_sources}
        if len(counts) == 1:
            source = cohort_sources[0]
            cohort_fact = DescriptiveReportFact(
                subsection="Cohort characteristics",
                text=f"The analysis cohort comprised {source.cohort_n:,} {source.analysis_unit}",
                evidence_id=source.evidence_id, source_sha256=source.source_sha256,
                source_fields=(source.cohort_source_field,),
            )
            destinations["Cohort characteristics"] = (
                *destinations.get("Cohort characteristics", ()), cohort_fact,
            )
    for subsection, subsection_facts in destinations.items():
        heading = re.search(rf"^### {re.escape(subsection)}[ \t]*$", body, re.M)
        if heading is None:
            continue
        next_heading = re.search(r"^###\s+", body[heading.end():], re.M)
        stop = heading.end() + next_heading.start() if next_heading else len(body)
        existing = body[heading.end():stop].splitlines()
        missing = list(dict.fromkeys(fact.scaffold for fact in subsection_facts if fact.scaffold not in existing))
        if missing:
            body = body[:heading.end()] + "\n\n" + "\n\n".join(missing) + "\n" + body[heading.end():]
    return manuscript[:section.end()] + body + manuscript[end:]


_CONTINUOUS_FIELDS = ("group_n", "n_nonmissing", "missing_n", "median", "q25", "q75")
_COUNTS = _CONTINUOUS_FIELDS[:3]
_CJK = re.compile(r"[\u3400-\u9fff]")


def _read(evidence, record):
    path = verified_run_evidence_path(evidence.root, record)
    if path is None:
        raise ValueError("Descriptive writer source failed digest verification")
    return path.read_text(encoding="utf-8")


def _groupings(evidence):
    record = evidence.get("exposure_groupings")
    if record is None:
        return None, [], {}
    payload = json.loads(_read(evidence, record))
    rows = exposure_group_label_rows(payload, record_sha256=record.sha256)
    parameters = {
        item["variable"]: item["derivation"]
        for item in payload.get("compiled", {}).get("groupings", [])
        if item.get("disposition") == "applied"
    }
    return record, rows, parameters


def manuscript_group_labels(evidence, labels, *, language="en"):
    """The grouping owner's study labels, or its English rule when needed.

    Integer and float transports of the owner's integer codes name the same
    declared level. An unlabelled level remains code-only, even when a caller
    supplied an unsupported clinical translation.
    """
    named, citations = dict(labels), {}
    inequalities = (
        (("≤", "不高于"), ("≥", "不低于"), ("<", "低于"), (">", "高于"))
        if str(language).lower().startswith("zh") else
        (("≤", "at most"), ("≥", "at least"), ("<", "below"), (">", "above"))
    )
    record, groups, parameters = _groupings(evidence)
    for group in groups:
        variable = group["variable"]
        for level in group["levels"]:
            label = level["label"]
            if label and str(language).lower().startswith("en") and _CJK.search(label):
                label = grouping_rule_texts(parameters[variable], include_measure=False)[level["group"]]
            if label:
                # Keep inequalities readable without introducing HTML markup.
                for op, word in inequalities:
                    label = label.replace(op, word)
                if re.search(r"[{}\[\]`\\*_#|\n\r]", label):
                    raise ValueError("Grouping label cannot be carried in a result sentence")
            for code in (str(level["code"]), f"{level['code']:.1f}"):
                key = f"{variable}={code}"
                named[key] = label or key
                if label:
                    citations[key] = (record.evidence_id,)
    return named, citations


def _continuous_tables(records, evidence):
    for row in current_successful_step_records(records or ()):
        hint = row.get("step_summary", {})
        if not (
            hint.get("interpretation_class") == "continuous_outcome_summary"
            or hint.get("deterministic_standard_analysis") == "continuous_outcome_summary"
            or CONTINUOUS_OUTCOME_OUTPUT in hint.get("output_files", {})
        ):
            continue
        record = evidence.get(str(row.get("step_summary_evidence_id") or ""))
        if (
            record is None or record.kind != "statistic" or record.produced_by_step != row["step_id"]
            or record.evidence_id not in row.get("evidence_ids", ())
        ):
            raise ValueError("Continuous summary is not active evidence of its producing step")
        summary = json.loads(_read(evidence, record))
        if summary.get("interpretation_class") != "continuous_outcome_summary":
            continue
        if not (
            summary.get("status") == "ok" and summary.get("analysis_role") == "secondary"
            and summary.get("interpretation_ceiling") == "descriptive_unadjusted_not_causal"
            and summary.get("adjusted_effect") is None
            and summary.get("uncertainty_interval") is None and not summary.get("inferential_tests")
        ):
            raise ValueError("Continuous descriptive source exceeds its claim ceiling")
        filename = summary.get("output_files", {}).get(CONTINUOUS_OUTCOME_OUTPUT)
        if not filename:
            raise ValueError("Continuous summary names no owned table")
        tables = [
            item for item in evidence.records()
            if item.kind == "table" and item.produced_by_step == row["step_id"]
            and item.evidence_id in row.get("evidence_ids", ())
            and evidence_artifact_basename_stem(Path(item.relative_path), item.evidence_id) == Path(filename).stem
        ]
        if len(tables) != 1:
            raise ValueError("Continuous source needs exactly one active owned table")
        table = tables[0]
        rows = list(csv.DictReader(io.StringIO(_read(evidence, table))))
        seen = set()
        for position, cell in enumerate(rows):
            if cell.get("variable") not in summary.get("outcome_variables", ()):
                raise ValueError("Continuous table names an undeclared outcome")
            coordinate = (cell.get("variable"), cell.get("row_role"), cell.get("group"))
            if coordinate in seen:
                raise ValueError("Continuous table duplicates an outcome group")
            seen.add(coordinate)
            for key in _CONTINUOUS_FIELDS:
                raw = cell.get(key)
                value = float(raw) if raw not in (None, "") else None
                if value is not None and not math.isfinite(value):
                    raise ValueError("Nonfinite continuous result")
                if key in _COUNTS:
                    if value is None or value < 0 or not value.is_integer():
                        raise ValueError("Invalid continuous outcome count")
                    value = int(value)
                cell[key] = value
            if cell["n_nonmissing"] + cell["missing_n"] != cell["group_n"]:
                raise ValueError("Continuous outcome counts do not partition their group")
            quantiles = [cell[key] for key in ("q25", "median", "q75")]
            if cell["n_nonmissing"]:
                if any(x is None for x in quantiles) or quantiles != sorted(quantiles):
                    raise ValueError("Continuous quantiles are missing or unordered")
            elif any(x is not None for x in quantiles):
                raise ValueError("Unobserved outcome cannot supply quantiles")
            yield row["step_id"], summary, table, position, cell


def compile_continuous_outcome_report_facts(records, *, evidence, reader_display_labels,
                                           context=None, manuscript_language="en", plan=None):
    """Copy each declared continuous outcome's median, IQR, unit and counts."""
    from .manuscript_labels import source_bound_manuscript_labels

    labels, citations = manuscript_group_labels(evidence, source_bound_manuscript_labels(
        context, reader_display_labels, language=manuscript_language,
    ), language=manuscript_language)
    variables = {v.name: v for v in getattr(context, "variables", ())}
    planned = {step.step_id: step for step in getattr(plan, "steps", ())}
    grouped = {}
    unit_name = _analysis_unit(context)
    for step_id, summary, source, position, row in _continuous_tables(records, evidence):
        outcome = row["variable"]
        if plan is not None and (
            step_id not in planned or outcome not in continuous_outcome_columns(planned[step_id])[1:]
            or summary.get("grouping_variable") != continuous_outcome_columns(planned[step_id])[0]
        ):
            raise ValueError("Continuous result does not answer its declared plan outcome")
        if row["row_role"] not in {"overall", "exposure_level"}:
            raise ValueError("Unknown continuous result row role")
        coordinate = f"{summary['grouping_variable']}={row['group']}"
        group = _reader_name(labels.get(coordinate, coordinate))
        label = in_sentence_label(_reader_name(labels.get(outcome, outcome)))
        unit = row.get("unit") or getattr(variables.get(outcome), "unit", None) or ""
        if re.search(r"[{}\[\]`<>\\*_#|\n\r]", f"{label}{group}{unit}"):
            raise ValueError("Unsafe continuous reader label")
        # Use the shared reader precision, including its trailing-zero policy.
        values = {key: round_reader_numeric_display(f"{row[key]:.12f}")[0]
                  for key in ("median", "q25", "q75") if row[key] is not None}
        distribution = (
            f"{values['median']}{' ' + unit if unit else ''} (IQR {values['q25']}–{values['q75']})"
            if row["n_nonmissing"] else "unavailable (no observed values)"
        )
        if row["missing_n"]:
            distribution += (f" ({row['n_nonmissing']:,} observed; {row['missing_n']:,} missing "
                             f"of {row['group_n']:,} {unit_name})")
        role = "overall" if row["row_role"] == "overall" else "unmeasured" if row["group"] == "Missing" else "group"
        part = (f"{distribution} in {group}" if role == "group" else
                f"the overall median was {distribution}" if role == "overall" else
                f"the median among {row['group_n']:,} {unit_name} with unmeasured exposure was {distribution}")
        fact = DescriptiveReportFact(
            subsection="Secondary analyses", text="",
            evidence_id=source.evidence_id, source_sha256=source.sha256,
            source_fields=tuple(f"rows[{position}].{key}" for key in _CONTINUOUS_FIELDS),
            supporting_evidence_ids=citations.get(coordinate, ()),
            step_id=step_id, product_id=CONTINUOUS_OUTCOME_OUTPUT, outcome=outcome,
        )
        grouped.setdefault((step_id, source.evidence_id, outcome, label), []).append((role, part, fact))
    facts = []
    for (_step, _source, _outcome, label), rows in grouped.items():
        parts = [part for role, part, _fact in rows if role == "group"]
        tails = [part for wanted in ("overall", "unmeasured") for role, part, _fact in rows if role == wanted]
        text = f"Median {label} was {_join(parts)}" if parts else f"For {label}"
        if tails:
            text += ("; " if parts else ", ") + "; ".join(tails)
        facts.extend(_combined_facts([fact for _role, _part, fact in rows], text))
    return tuple(facts)


def descriptive_table_numeric_claims(evidence, records):
    """Numeric leaves from the same verified table/typed label records.

    These join the ordinary binder's candidate pool; wrong citations, changed
    values and stale/inactive products still fail the existing strict gate.
    """
    claims = []
    for step_id, _summary, source, position, row in _continuous_tables(records, evidence):
        for key in (*_CONTINUOUS_FIELDS, "group"):
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            claims.append(NumericClaim(
                value=str(row[key]), canonical=value, evidence_id=source.evidence_id,
                step_id=step_id, source_field=f"rows[{position}].{key}",
            ))
    record, groups, parameters = _groupings(evidence)
    for group in groups:
        values = {float(level["code"]) for level in group["levels"]}
        values.update(value for values in grouping_rule_values(parameters[group["variable"]]).values() for value in values)
        for value in sorted(values):
            claims.append(NumericClaim(
                value=str(value), canonical=value, evidence_id=record.evidence_id,
                step_id=record.evidence_id, source_field=f"compiled.{group['variable']}.derivation",
            ))
    return claims


def writer_question_coverage(manuscript, *, plan, facts, evidence):
    """A typed finding for every planned descriptive part without a bound result."""
    requirements = []
    for step in getattr(plan, "steps", ()):
        spec = step.exposure_outcome_distribution_spec
        if spec is not None and step.planned_analysis_role in {"primary", "secondary"}:
            for part in ("exposure_distribution", spec.outcome):
                requirements.append((step.step_id, "table:exposure_outcome_distribution", part))
        if CONTINUOUS_OUTCOME_OUTPUT in step.expected_outputs:
            requirements.extend((step.step_id, CONTINUOUS_OUTCOME_OUTPUT, outcome)
                                for outcome in continuous_outcome_columns(step)[1:])
    visible = re.sub(r"<!--.*?-->|```.*?```", "", manuscript, flags=re.S)
    results = re.search(r"(?ms)^## Results[ \t]*\n(.*?)(?=^## |\Z)", visible)
    body = results[1] if results else ""
    findings = []
    for step_id, product, part in requirements:
        candidates = [fact for fact in facts if
                      (fact.step_id, fact.product_id, fact.outcome) == (step_id, product, part)]
        covered = False
        for fact in candidates:
            source = evidence.get(fact.evidence_id)
            if source is None or source.sha256 != fact.source_sha256 or source.produced_by_step != step_id:
                continue
            for line in body.splitlines():
                citation = (f"{{evidence:{source.evidence_id}}}" in line or
                            f'({source.relative_path} "sha256={source.sha256[:8]}")' in line)
                if citation and fact_sentence_visible(line, fact.text):
                    covered = True
        if not covered:
            findings.append(ValidationFinding(
                validator="writer_question_coverage", severity="error",
                message="A planned descriptive question part has no result sentence bound to its product.",
                detail={"reason_code": "writer_question_part_uncovered", "step_id": step_id,
                        "product_id": product, "outcome": part},
            ))
    return findings
