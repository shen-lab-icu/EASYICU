from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import (
    EvidenceEnforcementError,
    EvidenceEnforcementMode,
    EvidenceStore,
)
from easyicu.research_agent.authority.manuscript_claim_policy import (
    expand_scientific_claim_tokens,
    filter_evidence_bound_scaffold,
    missing_scientific_claims_in_results,
    place_scientific_claim_tokens_in_results,
)
from easyicu.research_agent.authority.manuscript_method_facts import ManuscriptMethodFact
from easyicu.research_agent.authority.scientific_claims import ScientificClaim


def _claim() -> ScientificClaim:
    return ScientificClaim(
        claim_id="adjusted_association",
        claim_type="association",
        exposure="Lactate",
        outcome="hospital mortality",
        direction="positive",
        estimand="adjusted odds ratio",
        population="the primary cohort",
        analysis_role="primary",
        status="supported",
        adjusted_for=["age", "sex"],
        step_id="04_association",
        evidence_id="04_association_summary",
    )


def _resolver(ref: str):
    claim = _claim()
    return claim if ref == claim.claim_ref else None


def test_policy_accepts_only_a_complete_known_claim_token() -> None:
    result = filter_evidence_bound_scaffold(
        "{claim:04_association.adjusted_association}\n"
        "Lactate was higher {claim:04_association.adjusted_association}.",
        resolve_claim=_resolver,
    )

    assert result.scaffold == "{claim:04_association.adjusted_association}\n"
    assert result.filtered_sentences == (
        "Lactate was higher {claim:04_association.adjusted_association}.",
    )
    assert result.unsupported_scientific_claim_sentences == result.filtered_sentences


def test_quoted_context_heading_survives_without_ending_results() -> None:
    retained = (
        "## Results\n\n"
        "> ### Materials and methods\n"
        "> This section explains the prespecified study design.\n"
    )
    quoted_claim = "The filter eliminated false alerts."
    results_claim = "The patch eliminated duplicate alerts."
    filtered = filter_evidence_bound_scaffold(
        f"{retained}\n> ### {quoted_claim}\n\n{results_claim}\n",
        resolve_claim=_resolver,
    )

    assert filtered.scaffold == retained
    assert filtered.unsupported_scientific_claim_sentences == (quoted_claim, results_claim)


@pytest.mark.parametrize("separator", [" ", "\n"], ids=["inline", "separate-line"])
@pytest.mark.parametrize(
    "body",
    [
        "The patch eliminated duplicate alerts [@prior_2024].",
        "The patch eliminated duplicate alerts {claim:alerts.deduplicated}.",
    ],
    ids=["cited-prose", "embedded-claim"],
)
def test_rejected_abstract_body_leaves_its_label(separator, body) -> None:
    filtered = filter_evidence_bound_scaffold(
        f"## Abstract\n\n**Results:**{separator}{body}\n",
        resolve_claim=_resolver,
    )

    assert filtered.scaffold == "## Abstract\n\n**Results:**\n"
    assert filtered.unsupported_scientific_claim_sentences == (body,)


def test_partially_filtered_abstract_body_keeps_its_line_break_after_label() -> None:
    retained = "This study describes baseline characteristics."
    rejected = "The patch eliminated duplicate alerts."
    filtered = filter_evidence_bound_scaffold(
        f"## Abstract\n\n**Results:**\n{rejected} {retained}\n",
        resolve_claim=_resolver,
    )

    assert filtered.scaffold == f"## Abstract\n\n**Results:**\n{retained}\n"
    assert filtered.unsupported_scientific_claim_sentences == (rejected,)


@pytest.mark.parametrize(
    ("retained", "rejected_heading", "rejected_text"),
    [
        (
            "## Results\n\n{claim:04_association.adjusted_association}",
            "## Methods (Executed data preparation: queue reconciliation)",
            "## Methods (Executed data preparation: queue reconciliation)",
        ),
        (
            "## Abstract\n\n**Results:**",
            "### Methods: Executed data preparation: queue reconciliation",
            "### Methods: Executed data preparation: queue reconciliation",
        ),
        (
            "## Abstract\n\n**Results:**",
            "> ### Methods: Recorded units for the selected exposure: seconds",
            "> ### Methods: Recorded units for the selected exposure: seconds",
        ),
        (
            "## Results\n\n{claim:04_association.adjusted_association}",
            "## Discussion: Stale jobs fell by 42% (p = 0.001)",
            "Discussion: Stale jobs fell by 42% (p = 0.001)",
        ),
    ],
    ids=["results", "abstract", "quoted-abstract", "empirical-discussion"],
)
def test_rejected_heading_does_not_end_findings(retained, rejected_heading, rejected_text) -> None:
    assertion = "The revised scheduler cleared stale jobs [@scheduler_review]."
    filtered = filter_evidence_bound_scaffold(
        f"{retained}\n\n{rejected_heading}\n\n{assertion}", resolve_claim=_resolver,
    )
    second_pass = filter_evidence_bound_scaffold(filtered.scaffold, resolve_claim=_resolver)

    assert second_pass.scaffold == filtered.scaffold
    assert filtered.scaffold == retained + "\n"
    assert filtered.unsupported_scientific_claim_sentences == (rejected_text, assertion)


@pytest.mark.parametrize("separator", [" ", "\n"], ids=["inline", "separate-line"])
def test_rejected_abstract_heading_does_not_orphan_results_label(separator) -> None:
    heading = "## Abstract (Executed class model: queue profiles)"
    assertion = "The revised scheduler cleared stale jobs [@scheduler_review]."
    filtered = filter_evidence_bound_scaffold(
        f"{heading}\n\n**Results:**{separator}{assertion}", resolve_claim=_resolver,
    )

    assert filtered.scaffold == "\n"
    assert filtered.unsupported_scientific_claim_sentences == (heading, assertion)
    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver,
    ).scaffold == filtered.scaffold


@pytest.mark.parametrize(
    ("nested_heading", "rejected_heading", "rejected_text"),
    [
        (
            "",
            "## Introduction (Executed data preparation: queue reconciliation)",
            "## Introduction (Executed data preparation: queue reconciliation)",
        ),
        (
            "",
            "## Introduction: Stale jobs fell by 42% (p = 0.001)",
            "Introduction: Stale jobs fell by 42% (p = 0.001)",
        ),
        (
            "#### Results\n\n",
            "### Queue reconciliation",
            "Queue reconciliation",
        ),
        (
            "#### Results\n\n",
            "### {claim:queue_audit.unregistered}",
            "{claim:queue_audit.unregistered}",
        ),
    ],
    ids=["method-fact", "empirical-introduction", "restricted-heading", "unknown-claim"],
)
def test_rejected_heading_does_not_displace_exact_method_fact(
    nested_heading, rejected_heading, rejected_text,
) -> None:
    fact = ManuscriptMethodFact(
        source_field="exposure.units",
        text="Recorded units for the selected exposure: seconds",
        source_sha256="0" * 64,
    )
    retained = "## Methods\n\n### Variables\n\n" + nested_heading
    filtered = filter_evidence_bound_scaffold(
        f"{retained}{rejected_heading}\n{fact.scaffold}",
        resolve_claim=_resolver,
        method_facts=(fact,),
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver, method_facts=(fact,),
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == f"{retained}\n{fact.scaffold}\n"
    assert filtered.unsupported_scientific_claim_sentences == (rejected_text,)


@pytest.mark.parametrize(
    "rejected_line",
    [
        "> - **{label}:** Recorded units for the selected exposure: seconds",
        "{label}: Executed data preparation: queue reconciliation",
        "> - **{label}:** {{claim:queue_audit.unregistered}}",
    ],
    ids=["quoted-list", "unbolded", "unknown-claim"],
)
@pytest.mark.parametrize(
    ("active_label", "rejected_label"), [("Results", "Methods"), ("Methods", "Results")],
)
def test_dropped_abstract_label_keeps_the_active_scope(
    active_label, rejected_label, rejected_line,
) -> None:
    retained = f"## Abstract\n\n**{active_label}:**"
    assertion = "The revised scheduler cleared stale jobs [@scheduler_review]."
    filtered = filter_evidence_bound_scaffold(
        f"{retained}\n\n{rejected_line.format(label=rejected_label)}\n\n{assertion}",
        resolve_claim=_resolver,
    )
    expected = retained + "\n"
    if active_label == "Methods":
        expected = f"{retained}\n\n\n\n{assertion}\n"

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver,
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == expected


def test_retained_abstract_label_updates_scope_when_its_body_is_rejected() -> None:
    assertion = "The revised scheduler cleared stale jobs [@scheduler_review]."
    filtered = filter_evidence_bound_scaffold(
        "## Abstract\n\n**Results:**\n\n"
        "**Methods:** Results: Executed data preparation: queue reconciliation\n\n"
        f"{assertion}",
        resolve_claim=_resolver,
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver,
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == f"## Abstract\n\n**Results:**\n\n**Methods:**\n\n{assertion}\n"


@pytest.mark.parametrize(
    ("active_label", "discarded_label", "next_label", "retains_next"),
    [
        ("Results", "Methods: ", "", False),
        ("Methods", "Results: ", "", True),
        ("Methods", "", "Results: ", False),
        ("Methods", "", "Table 1: ", False),
        ("Results", "", "Methods: ", True),
        ("Methods", "Table 1: ", "", True),
    ],
    ids=[
        "discarded-methods", "discarded-results", "exposed-results", "exposed-caption",
        "exposed-methods", "discarded-caption",
    ],
)
def test_first_retained_sentence_determines_region_scope(
    active_label, discarded_label, next_label, retains_next,
) -> None:
    retained = f"## Abstract\n\n**{active_label}:**"
    rejected = f"{discarded_label}Archive volume shrank by 37% (p = 0.002)."
    following = f"{next_label}The migration tool shuffled archived records {{evidence:archive_log}}."
    resolve_evidence = {"archive_log"}.__contains__
    filtered = filter_evidence_bound_scaffold(
        f"{retained}\n\n{rejected} {following}",
        resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == (f"{retained}\n\n{following}" if retains_next else retained) + "\n"
    assert filtered.filtered_sentences == ((rejected,) if retains_next else (rejected, following))


@pytest.mark.parametrize(
    ("separator", "discarded"),
    [(" ", ""), ("\n", ""), ("\n", "The scanner copied 23 field notes.")],
    ids=["same-line", "soft-wrap", "soft-wrap-with-deletion"],
)
def test_first_retained_sentence_locks_scope_for_the_rest_of_the_region(separator, discarded) -> None:
    opener = (
        "## Abstract\n\n**Results:**\n\n"
        f"Methods: The archivist reviewed the staging directory.{separator}"
    )
    followup = "Results: The migration tool shuffled archived records [@migration_review].\n"
    scaffold = opener + (discarded + "\n" if discarded else "") + followup
    filtered = filter_evidence_bound_scaffold(scaffold, resolve_claim=_resolver)

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver,
    ).scaffold == filtered.scaffold == opener + followup
    assert filtered.filtered_sentences == ((discarded,) if discarded else ())


@pytest.mark.parametrize(
    ("retained_line", "discarded_body"),
    [
        ("Results: The cohort included 18 observations {evidence:cohort_log}.", ""),
        ("Table 1: The cohort included 18 observations {evidence:cohort_log}.", ""),
        ("**Results:**", " Storage volume fell by 37% (p = 0.002)."),
    ],
    ids=["results", "table-caption", "retained-abstract-label"],
)
@pytest.mark.parametrize("citation", ["{evidence:archive_log}", "[@archive_review]"])
def test_soft_wrapped_findings_keep_the_first_retained_scope(retained_line, discarded_body, citation):
    prefix = "## Abstract\n\n**Methods:**\n\n"
    unsupported = f"The scanner copied field notes to the catalog {citation}."
    resolve_evidence = {"cohort_log", "archive_log"}.__contains__
    filtered = filter_evidence_bound_scaffold(
        f"{prefix}{retained_line}{discarded_body}\n{unsupported}\n",
        resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == f"{prefix}{retained_line}\n"
    assert unsupported in filtered.unsupported_scientific_claim_sentences


@pytest.mark.parametrize("break_marker", ["\n", "  "], ids=["paragraph-break", "hard-line-break"])
def test_caption_restriction_ends_at_a_markdown_break(break_marker):
    scaffold = (
        "## Abstract\n\n**Methods:**\n\n"
        f"Table 1: The cohort included 18 observations {{evidence:cohort_log}}.{break_marker}\n"
        "The scanner copied field notes to the catalog.\n"
    )
    resolve_evidence = {"cohort_log"}.__contains__
    filtered = filter_evidence_bound_scaffold(
        scaffold, resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    ).scaffold == filtered.scaffold == scaffold


def test_exposed_caption_is_restricted_after_an_outer_methods_label() -> None:
    retained = "## Abstract\n\n**Methods:**"
    rejected = "Archive volume shrank by 37% (p = 0.002)."
    caption = "Table 1: The migration tool shuffled archived records {evidence:archive_log}."
    resolve_evidence = {"archive_log"}.__contains__
    filtered = filter_evidence_bound_scaffold(
        f"{retained} {rejected} {caption}",
        resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    )

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver, resolve_evidence=resolve_evidence,
    ).scaffold == filtered.scaffold == f"{retained}\n"
    assert filtered.filtered_sentences == (rejected, caption)


@pytest.mark.parametrize(("outer", "inner"), [("Results", "Methods"), ("Methods", "Results")])
def test_outer_abstract_label_overrides_leading_body_label(outer, inner) -> None:
    retained = f"## Abstract\n\n**{inner}:**\n\n**{outer}:**"
    body = f"{inner}: The migration tool shuffled archived records [@migration_review]."
    followup = "The archivist moved the index [@migration_review]."
    scaffold = f"{retained} {body}\n\n{followup}\n"
    filtered = filter_evidence_bound_scaffold(scaffold, resolve_claim=_resolver)

    assert filter_evidence_bound_scaffold(
        filtered.scaffold, resolve_claim=_resolver,
    ).scaffold == filtered.scaffold
    assert filtered.scaffold == (scaffold if outer == "Methods" else f"{retained}\n")
    assert filtered.filtered_sentences == (() if outer == "Methods" else (body, followup))


@pytest.mark.parametrize(
    "prefix",
    ["**Results:** ", "**Conclusions:** ", "> - **Results:** ", "**结果：** "],
)
def test_structured_abstract_label_preserves_known_claim_authority(prefix) -> None:
    claim = _claim()
    scaffold = prefix + claim.placeholder

    filtered = filter_evidence_bound_scaffold(scaffold, resolve_claim=_resolver)
    expanded = expand_scientific_claim_tokens(
        filtered.scaffold,
        resolve_claim=_resolver,
        current_evidence_ids={claim.evidence_id},
    )

    assert filtered.scaffold == scaffold + "\n"
    assert filtered.filtered_sentences == ()
    assert expanded.scaffold.startswith(prefix + claim.render_reader_text(
        include_estimate="Conclusions" not in prefix,
    ))
    assert f"{{evidence:{claim.evidence_id}}}" in expanded.scaffold
    assert expanded.missing_claim_refs == ()
    assert expanded.malformed_sentences == ()


@pytest.mark.parametrize("direction", ["positive", "negative", "no_clear_association"])
def test_conclusion_projects_bounded_interpretation_without_repeating_estimate(direction):
    claim = _claim().model_copy(update={
        "direction": direction, "point_estimate": 1.7,
        "interval_lower": 1.2, "interval_upper": 2.4,
        "exposure": "oxygen_index at 5 versus 2 mmHg",
        "population": "the complete-case records at the 24-hour landmark",
        "estimand": "odds ratio; this point contrast only, not a summary of the nonlinear curve; noncausal association",
    })
    sealed = claim.model_dump_json()
    raw = (
        "## Abstract\n\n**Results:** " + claim.placeholder
        + "\n\n**Conclusions:**\n\n" + claim.placeholder
        + "\n\n## Results\n\n" + claim.placeholder
        + "\n\n## Conclusion\n\n" + claim.placeholder
        + "\n\n## Supplementary results\n\n" + claim.placeholder
    )
    expanded = expand_scientific_claim_tokens(raw, resolve_claim=lambda _: claim).scaffold
    results = expanded.split("## Results\n", 1)[1].split("## Conclusion", 1)[0]
    conclusion = expanded.split("## Conclusion\n", 1)[1].split("## Supplementary", 1)[0]
    abstract = expanded.split("## Results\n", 1)[0]
    assert "1.7" in results and "95% CI" in results
    assert "1.7" not in conclusion and "95% CI" not in conclusion
    assert "1.7" not in abstract.split("**Conclusions:**", 1)[1]
    for coordinate in ("5 versus 2 mmHg", "24-hour landmark", "age and sex", "this point contrast only"):
        assert coordinate in conclusion
    assert "1.7" in expanded.split("## Supplementary results", 1)[1]
    assert claim.model_dump_json() == sealed


@pytest.mark.parametrize(
    "body",
    [
        "{claim:unknown.adjusted_association}",
        "Higher mortality {claim:04_association.adjusted_association}",
        "{claim:04_association.adjusted_association} and a causal benefit.",
        "{{claim:04_association.adjusted_association}}",
        "{claim:04_association.adjusted_association} {evidence:unrelated}",
    ],
)
def test_structured_abstract_label_does_not_authorize_unsupported_claims(body) -> None:
    filtered = filter_evidence_bound_scaffold(
        "**Results:** " + body,
        resolve_claim=_resolver,
    )

    assert filtered.scaffold == "\n"
    assert filtered.unsupported_scientific_claim_sentences


def test_structured_abstract_label_still_checks_current_evidence_membership() -> None:
    claim = _claim()
    expanded = expand_scientific_claim_tokens(
        "**Results:** " + claim.placeholder,
        resolve_claim=_resolver,
        current_evidence_ids=set(),
    )

    assert expanded.missing_claim_refs == (claim.claim_ref,)
    assert claim.render_reader_text() not in expanded.scaffold


def test_assertive_bold_label_cannot_hide_an_extra_scientific_claim() -> None:
    scaffold = "**Higher mortality:** " + _claim().placeholder

    filtered = filter_evidence_bound_scaffold(scaffold, resolve_claim=_resolver)
    expanded = expand_scientific_claim_tokens(scaffold, resolve_claim=_resolver)

    assert filtered.scaffold == "\n"
    assert filtered.unsupported_scientific_claim_sentences == (scaffold,)
    assert expanded.malformed_sentences == (scaffold,)


def test_policy_collapses_exact_host_prose_followed_by_duplicate_claim_token() -> None:
    claim = _claim()
    sentence = (
        f"{claim.render_text()} {{evidence:{claim.evidence_id}}}. "
        f"{{claim:{claim.claim_ref}}}"
    )

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == f"{{claim:{claim.claim_ref}}}\n"
    assert result.filtered_sentences == ()


def test_policy_does_not_canonicalize_paraphrase_plus_claim_token() -> None:
    claim = _claim()
    sentence = f"A different assertion. {{claim:{claim.claim_ref}}}"

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == "\n"
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_policy_preserves_metadata_but_filters_uncited_numeric_result() -> None:
    result = filter_evidence_bound_scaffold(
        "Data availability: generated scripts are available.\n"
        "Mortality was 20%.",
        resolve_claim=_resolver,
    )

    assert "Data availability" in result.scaffold
    assert "Mortality was 20%" not in result.scaffold
    assert result.removed_result_sentences == ("Mortality was 20%.",)


def test_policy_keeps_registered_numeric_fact_for_later_provenance_binding() -> None:
    sentence = "Mortality was 20% {evidence:outcome_rate}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref == "outcome_rate",
    )

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


def test_policy_rejects_numeric_fact_with_unregistered_evidence() -> None:
    sentence = "Mortality was 20% {evidence:unregistered}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda _ref: False,
    )

    assert result.scaffold == "\n"
    assert result.removed_result_sentences == (sentence,)


def test_unrelated_registered_evidence_cannot_launder_numeric_methods_sentence() -> None:
    sentence = "The primary model used 20 imputations {evidence:queued_unrelated_step}."
    scaffold = f"## Methods\n\n{sentence}"

    result = filter_evidence_bound_scaffold(
        scaffold,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref == "queued_unrelated_step",
    )

    assert sentence not in result.scaffold
    assert result.removed_result_sentences == (sentence,)


def test_registered_evidence_keeps_non_numeric_methods_prose() -> None:
    outcome = "In-hospital mortality was the primary outcome {evidence:outcome_definition}."
    missingness = "Missingness was addressed with multiple imputation {evidence:imputation_step}."
    numeric = "The primary model used 20 imputations {evidence:imputation_step}."
    scaffold = f"## Methods\n\n### Outcomes\n\n{outcome}\n{missingness}\n{numeric}"

    result = filter_evidence_bound_scaffold(
        scaffold,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref in {"outcome_definition", "imputation_step"},
    )

    assert outcome in result.scaffold
    assert missingness in result.scaffold
    assert numeric not in result.scaffold
    assert result.removed_result_sentences == (numeric,)


def test_registered_evidence_does_not_bypass_qualitative_claim_authority() -> None:
    sentence = "Patients had higher mortality {evidence:outcome_rate}."

    result = filter_evidence_bound_scaffold(
        sentence,
        resolve_claim=_resolver,
        resolve_evidence=lambda ref: ref == "outcome_rate",
    )

    assert result.scaffold == "\n"
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_exact_literature_key_keeps_only_nonnumeric_background_context() -> None:
    context = "Sepsis-3 mortality remains clinically important [@paper_2024]."
    numeric = "Prior mortality was 20% [@paper_2024]."

    kept = filter_evidence_bound_scaffold(context, resolve_claim=_resolver)
    blocked = filter_evidence_bound_scaffold(numeric, resolve_claim=_resolver)

    assert kept.scaffold == context + "\n"
    assert blocked.scaffold == "\n"


def test_grouped_exact_literature_keys_keep_nonnumeric_context() -> None:
    context = (
        "Cohort reporting followed observational guidance "
        "[@strobe_2007; @record_2015]."
    )

    result = filter_evidence_bound_scaffold(context, resolve_claim=_resolver)

    assert result.scaffold == context + "\n"
    assert result.filtered_sentences == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "Patients experienced excess mortality {evidence:unrelated_log}.",
        "Patients had higher mortality {evidence:unrelated_log}.",
    ],
)
def test_strict_policy_rejects_assertions_with_arbitrary_evidence_tokens(
    tmp_path, sentence
) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError) as blocked:
        store.enforce_evidence_bound_scaffold(sentence)

    detail = blocked.value.detail
    assert sentence in (
        detail["removed_sentences"]
        + detail["unsupported_scientific_claim_sentences"]
    )


def test_policy_preserves_provenance_metadata_with_evidence_token() -> None:
    sentence = (
        "Data availability: The reproducibility envelope and SHA-256 digest are "
        "available {evidence:run_manifest}."
    )

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


def test_policy_ignores_numeric_literature_key_but_not_numeric_result() -> None:
    context = (
        "The declared clinical framework used an exact run-bound source "
        "[@paper_2024]."
    )
    result = filter_evidence_bound_scaffold(context, resolve_claim=_resolver)
    blocked = filter_evidence_bound_scaffold(
        "Mortality was 20% [@paper_2024].",
        resolve_claim=_resolver,
    )

    assert result.scaffold == context + "\n"
    assert blocked.scaffold == "\n"
    assert blocked.removed_result_sentences == (
        "Mortality was 20% [@paper_2024].",
    )


def test_policy_preserves_scientific_noun_phrases_in_keyword_metadata() -> None:
    sentence = "Keywords: survival benefit, protective factors, ICU mortality"

    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert result.scaffold == sentence + "\n"
    assert result.filtered_sentences == ()


@pytest.mark.parametrize(
    "sentence",
    [
        "Treatment conferred a survival benefit.",
        "The exposure was protective against mortality.",
        "The intervention was harmful.",
        "Treated patients fared better.",
        "The exposure adversely affected outcomes.",
        "Treatment conferred benefit.",
        "The exposure was linked to harm.",
        "The findings suggested a benefit.",
    ],
)
def test_policy_rejects_common_qualitative_scientific_assertions(sentence) -> None:
    result = filter_evidence_bound_scaffold(sentence, resolve_claim=_resolver)

    assert sentence not in result.scaffold
    assert result.unsupported_scientific_claim_sentences == (sentence,)


def test_policy_applies_qualitative_claim_gate_to_markdown_headings() -> None:
    result = filter_evidence_bound_scaffold(
        "## Treatment conferred a survival benefit",
        resolve_claim=_resolver,
    )

    assert "Treatment conferred" not in result.scaffold
    assert result.unsupported_scientific_claim_sentences == (
        "Treatment conferred a survival benefit",
    )


def test_policy_preserves_versioned_clinical_term_in_structural_title() -> None:
    title = "# Retrospective ICU Study of Sepsis-3 and In-Hospital Mortality"

    result = filter_evidence_bound_scaffold(title, resolve_claim=_resolver)

    assert result.scaffold == title + "\n"
    assert result.filtered_sentences == ()


def test_versioned_term_does_not_hide_assertive_numeric_title() -> None:
    title = "# Sepsis-3 mortality was 20%"

    result = filter_evidence_bound_scaffold(title, resolve_claim=_resolver)

    assert result.scaffold == "\n"
    assert result.filtered_sentences


@pytest.mark.parametrize(
    "heading",
    [
        "## Treatment yielded better outcomes",
        "## The intervention was beneficial",
    ],
)
def test_policy_rejects_equivalent_unsupported_result_headings(heading) -> None:
    result = filter_evidence_bound_scaffold(heading, resolve_claim=_resolver)

    assert result.scaffold == "\n"
    assert result.filtered_sentences


@pytest.mark.parametrize(
    "sentence",
    [
        "Data availability: Scripts are available, and treatment conferred a survival benefit.",
        "Data availability: Patients experienced excess mortality.",
        "Generated scripts are available while exposed patients fared worse.",
        "Funding: The exposure was harmful to patients.",
    ],
)
def test_metadata_forms_cannot_hide_scientific_clauses(tmp_path, sentence) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError):
        store.enforce_evidence_bound_scaffold(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "See {Claim:04_association.adjusted_association}.",
        "See {claim:04_association.adjusted_association.",
        "See {claim:adjusted_association}.",
        "See {{claim:04_association.adjusted_association}}.",
        "See {Evidence:run_manifest}.",
        "See {evidence:run_manifest.",
        "See {evidence:run_manifest}}.",
        "See {evidence:}.",
    ],
)
def test_strict_policy_rejects_malformed_authority_placeholders(
    tmp_path, sentence
) -> None:
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)

    with pytest.raises(EvidenceEnforcementError) as blocked:
        store.enforce_evidence_bound_scaffold(sentence)

    assert blocked.value.detail["unsupported_scientific_claim_sentences"] == [
        sentence
    ]


def test_claim_expansion_preserves_markdown_prefix_and_binds_evidence() -> None:
    result = expand_scientific_claim_tokens(
        "> {claim:04_association.adjusted_association}",
        resolve_claim=_resolver,
    )

    assert result.scaffold.startswith(
        "> After adjustment for age and sex, Lactate"
    )
    assert "{evidence:04_association_summary}" in result.scaffold
    assert result.missing_claim_refs == ()
    assert result.malformed_sentences == ()


def test_claim_expansion_reports_missing_and_malformed_tokens() -> None:
    result = expand_scientific_claim_tokens(
        "{claim:99_missing.adjusted_association}\n"
        "Prose {claim:04_association.adjusted_association}.",
        resolve_claim=_resolver,
    )

    assert result.missing_claim_refs == ("99_missing.adjusted_association",)
    assert result.malformed_sentences == (
        "Prose {claim:04_association.adjusted_association}.",
    )


def test_host_claim_is_placed_in_results_even_when_writer_used_only_conclusion() -> None:
    claim = _claim()
    scaffold = (
        "## Results\n\n"
        "### Primary association\n\n"
        "The model included 100 stays {evidence:model}.\n\n"
        "## Conclusion\n\n"
        f"{claim.placeholder}\n"
    )

    placement = place_scientific_claim_tokens_in_results(
        scaffold,
        claims=[claim],
    )

    results = placement.scaffold.split("## Conclusion", 1)[0]
    assert claim.placeholder in results
    assert placement.inserted_claim_refs == (claim.claim_ref,)
    assert placement.missing_claim_refs == ()

    expanded = expand_scientific_claim_tokens(
        placement.scaffold,
        resolve_claim=_resolver,
    ).scaffold
    assert missing_scientific_claims_in_results(
        expanded,
        claims=[claim],
    ) == ()


def test_host_claim_placement_fails_closed_without_results_section() -> None:
    claim = _claim()
    placement = place_scientific_claim_tokens_in_results(
        f"## Conclusion\n\n{claim.placeholder}\n",
        claims=[claim],
    )

    assert placement.inserted_claim_refs == ()
    assert placement.missing_claim_refs == (claim.claim_ref,)


def test_host_claim_remains_present_after_deterministic_reader_rounding() -> None:
    claim = ScientificClaim(
        schema_version="easyicu.scientific_claim/2",
        claim_id="observed_absolute_risk_level_0",
        claim_type="descriptive_absolute_risk",
        exposure="group=0",
        outcome="death",
        direction="descriptive_only",
        estimand=(
            "observed absolute risk was 1/3 "
            "(33.333333 percent; counts only, no confidence interval)"
        ),
        population="the primary cohort",
        analysis_role="primary",
        status="supported",
        step_id="distribution",
        evidence_id="distribution_summary",
    )
    expanded = expand_scientific_claim_tokens(
        "## Results\n\n### Primary association\n\n" + claim.placeholder + "\n",
        resolve_claim=lambda ref: claim if ref == claim.claim_ref else None,
    ).scaffold
    assert "33.333333" not in expanded
    assert "33.333 percent" in expanded
    rounded = (
        expanded.replace("1/3", "1[^claim_1]/3[^claim_2]")
        .replace("33.333 percent", "33.333[^claim_3] percent")
    )

    assert missing_scientific_claims_in_results(
        rounded,
        claims=[claim],
    ) == ()


def test_claim_display_rounding_leaves_scaffold_provenance_and_model_numbers_intact():
    claim = _claim()
    canonical = "[^claim_1]: value=33.333333; field=estimate; evidence=source.123456"
    source_link = "[source](https://example.org/10.123456/record)"
    unbound = "A model-supplied value of 99.123456 must still face numeric binding."
    scaffold = "\n".join((claim.placeholder, canonical, source_link, unbound))
    expanded = expand_scientific_claim_tokens(scaffold, resolve_claim=_resolver)
    assert canonical in expanded.scaffold
    assert source_link in expanded.scaffold
    assert unbound in expanded.scaffold
    assert claim.render_reader_text() in expanded.scaffold


def test_writer_neutral_count_examples_use_the_same_closed_claim_grammar():
    manuscript = (
        "## Results\n\n"
        "The level 1 group included 60 stays (50% of the cohort) {evidence:summary}.\n\n"
        "The observed mortality was 12 of 60 stays (20%) in the level 1 group {evidence:summary}."
    )
    result = filter_evidence_bound_scaffold(
        manuscript, resolve_claim=lambda _: None, resolve_evidence=lambda ref: ref == "summary",
    )
    assert not result.filtered_sentences
    # Syntax admission is not value/source verification; the separate strict
    # binder must still validate every example value before a report is ready.
    tampered = manuscript.replace("included", "caused a reduction in")
    assert filter_evidence_bound_scaffold(
        tampered, resolve_claim=lambda _: None, resolve_evidence=lambda ref: True,
    ).filtered_sentences


@pytest.mark.parametrize("population, expected", [
    ("the bound typed cohort", "the analysis cohort"),
    ("rows with observed exposure in the bound typed cohort", "rows with observed exposure in the analysis cohort"),
])
def test_descriptive_reader_population_cannot_reintroduce_runtime_terms(population, expected):
    from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality

    claim = _claim().model_copy(update={
        "schema_version": "easyicu.scientific_claim/3",
        "claim_type": "descriptive_risk_difference", "direction": "descriptive_only",
        "adjusted_for": [], "point_estimate": 3.0, "interval_lower": 1.0,
        "interval_upper": 5.0, "confidence_level": 0.95,
        "interval_method": "linear_probability_wald", "effect_scale": "percentage_points",
        "exposure": "high_potassium=1 versus high_potassium=0",
        "outcome": "renal_readmission", "population": population,
    })
    original = claim.model_dump_json()
    draft = "\n\n".join(
        f"## {section}\n\n" + claim.placeholder
        for section in ("Abstract", "Discussion", "Conclusion")
    )
    reader = expand_scientific_claim_tokens(draft, resolve_claim=lambda _: claim).scaffold
    assert expected in reader
    assert not [finding for finding in audit_manuscript_quality(reader).findings
                if finding.code == "MANUSCRIPT_INTERNAL_TERM_EXPOSED"]
    assert claim.model_dump_json() == original
