from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    repair_registered_display_callouts,
    repair_reader_structure_from_existing_prose,
    render_reader_manuscript,
)
from easyicu.research_agent.reporting.write_phase import (
    _persist_manuscript_quality_artifacts,
)
from easyicu.research_agent.reporting.readiness import _MANUSCRIPT_ERROR_VALIDATORS


def _valid_manuscript() -> str:
    return """# Sepsis status and hospital mortality in a retrospective ICU cohort

**Keywords:** sepsis, intensive care, mortality, cohort, epidemiology

## Abstract

**Background:** Sepsis remains an important ICU syndrome.

**Methods:** We conducted a retrospective cohort analysis adjusted for age and sex.

**Results:** Sepsis status was associated with in-hospital mortality.

**Conclusions:** The association requires external validation.

## Introduction

Sepsis definitions and transparent cohort accounting matter for reproducible ICU research.

## Methods

### Study design and cohort
We conducted a retrospective ICU cohort study.

### Variables
The exposure was Sepsis-3 status and the outcome was in-hospital death.

### Statistical analysis
The adjustment set comprised age and sex. We used logistic regression.

### Software and reproducibility
Analyses were executed with versioned software and registered artifacts.

## Results

### Cohort characteristics
The cohort included eligible ICU stays.

### Primary outcome
The primary outcome was in-hospital death.

### Primary association
After adjustment for age and sex, Sepsis-3 status was associated with mortality.

### Sensitivity and subgroup analyses
Sensitivity analyses used the prespecified population.

## Discussion

The findings support an association but do not establish causation [@strobe_2007].

## Limitations

This observational, single-database analysis remains susceptible to residual confounding.

## Conclusion

Sepsis status was associated with in-hospital mortality and requires external validation.

## Data and code availability

Data and code availability require author verification before submission.

## Funding

Funding information requires author verification before submission.

## Ethics approval

Ethics information requires author verification before submission.

## Conflicts of interest

Conflict-of-interest information requires author verification before submission.

## Supplementary artifact release

The supplementary release inventory requires author verification before submission.
"""


def _codes(text: str) -> set[str]:
    return {finding.code for finding in audit_manuscript_quality(text).findings}


def test_complete_reader_facing_manuscript_passes() -> None:
    audit = audit_manuscript_quality(_valid_manuscript())

    assert audit.status == "pass"
    assert audit.schema_version == "manuscript-quality-audit-v3"
    assert audit.adjustment_sets == {
        "Methods": ("age", "sex"),
        "Results": ("age", "sex"),
    }
    assert not audit.findings


def test_registered_displays_must_be_called_out_in_results() -> None:
    audit = audit_manuscript_quality(
        _valid_manuscript(),
        expected_display_labels=("Table 1", "Figure 1"),
    )

    assert audit.status == "changes_required"
    assert audit.expected_display_labels == ("Table 1", "Figure 1")
    assert audit.observed_display_labels == ()
    assert (
        sum(
            finding.code == "MANUSCRIPT_DISPLAY_NOT_CALLED_OUT"
            for finding in audit.findings
        )
        == 2
    )


def test_registered_display_callouts_are_recorded() -> None:
    text = (
        _valid_manuscript()
        .replace(
            "The cohort included eligible ICU stays.",
            "The cohort included eligible ICU stays (Table 1).",
        )
        .replace(
            "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
            "After adjustment for age and sex, Sepsis-3 status was associated with mortality (Figure 1).",
        )
    )

    audit = audit_manuscript_quality(
        text,
        expected_display_labels=("Table 1", "Figure 1"),
    )

    assert audit.status == "pass"
    assert audit.observed_display_labels == ("Table 1", "Figure 1")


def test_registered_display_callouts_are_restored_without_inventing_results() -> None:
    text = _valid_manuscript().replace("Table 1 and Figure 1", "the displays")

    repaired, repairs = repair_registered_display_callouts(
        text,
        expected_display_labels=("Table 1", "Figure 1", "Figure 2"),
    )

    assert "Cohort characteristics are summarized in Table 1" in repaired
    assert "The principal study results are presented in Figure 1" in repaired
    assert "Figure 2" not in repaired
    assert [item["label"] for item in repairs] == ["Table 1", "Figure 1"]


def test_literature_year_does_not_create_unnamed_robustness_metric() -> None:
    text = _valid_manuscript().replace(
        "We used logistic regression.",
        "Flexible forms were assessed as robustness analyses [@splines_1989].",
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_METRIC_UNNAMED" not in {
        finding.code for finding in audit.findings
    }


def test_five_decimal_effect_estimate_is_overprecise() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        "The adjusted odds ratio was 1.60587.",
        1,
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_NUMERIC_OVERPRECISION" in {
        finding.code for finding in audit.findings
    }


def test_empty_conclusion_fails_closed() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.\n",
        "",
    )

    audit = audit_manuscript_quality(text)

    assert audit.status == "changes_required"
    assert "MANUSCRIPT_SECTION_EMPTY" in _codes(text)
    assert any(finding.section == "Conclusion" for finding in audit.findings)


def test_abstract_rejects_duplicated_conclusion() -> None:
    text = _valid_manuscript().replace(
        "**Conclusions:** The association requires external validation.",
        "**Conclusions:** Sepsis status was associated with in-hospital mortality.",
    )

    codes = _codes(text)
    assert "MANUSCRIPT_ABSTRACT_CONCLUSION_DUPLICATES_RESULTS" in codes


def test_methods_results_adjustment_conflict_is_reported() -> None:
    text = _valid_manuscript().replace(
        "After adjustment for age and sex, Sepsis-3 status was associated",
        "After adjustment for age and charlson_max, Sepsis-3 status was associated",
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" in _codes(text)
    assert audit.adjustment_sets["Methods"] == ("age", "sex")
    assert audit.adjustment_sets["Results"] == ("age", "charlson")


def test_trailing_adjustment_phrase_is_compared_with_methods() -> None:
    text = _valid_manuscript().replace(
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
        "The adjusted odds ratio was 1.61, after adjustment for age and "
        "Charlson comorbidity score.",
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" in _codes(text)
    assert audit.adjustment_sets["Results"] == (
        "age",
        "charlson",
    )

    raw = text.replace(
        "score.", "score {evidence:primary_association}.", 1
    )
    raw_audit = audit_manuscript_quality(raw)
    assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" in _codes(raw)
    assert raw_audit.adjustment_sets["Results"] == (
        "age",
        "charlson",
    )


def test_adjustment_aliases_and_materialisation_suffixes_are_equivalent() -> None:
    text = _valid_manuscript().replace(
        "The adjustment set comprised age and sex.",
        "The adjustment set comprised age and charlson_max.",
    ).replace(
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
        "The adjusted odds ratio was 1.61, after adjustment for patient age and "
        "Charlson comorbidity score.",
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" not in _codes(text)
    assert audit.adjustment_sets == {
        "Methods": ("age", "charlson"),
        "Results": ("age", "charlson"),
    }


def test_internal_runtime_terms_are_rejected_in_reader_facing_prose() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        "The `sep3_sofa2_max` result was reported in the bound typed cohort.",
        1,
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_INTERNAL_TERM_EXPOSED" in _codes(text)
    abstract_finding = next(
        finding
        for finding in audit.findings
        if finding.code == "MANUSCRIPT_INTERNAL_TERM_EXPOSED"
    )
    assert "`sep3_sofa2_max`" in abstract_finding.excerpts


def test_internal_runtime_terms_are_also_rejected_in_methods() -> None:
    text = _valid_manuscript().replace(
        "The exposure was Sepsis-3 status and the outcome was in-hospital death.",
        "The `sep3_sofa2_max` exposure was host-bound.",
    )

    audit = audit_manuscript_quality(text)

    finding = next(
        item
        for item in audit.findings
        if item.code == "MANUSCRIPT_INTERNAL_TERM_EXPOSED"
        and item.section == "Methods"
    )
    assert finding.severity == "error"
    assert "`sep3_sofa2_max`" in finding.excerpts


def test_unnamed_point_estimate_is_rejected() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        "The discrimination point estimate was 0.772.",
        1,
    )

    assert "MANUSCRIPT_METRIC_UNNAMED" in _codes(text)

    named = text.replace(
        "The discrimination point estimate was 0.772.",
        "The area under the receiver operating characteristic curve was 0.772.",
    )
    assert "MANUSCRIPT_METRIC_UNNAMED" not in _codes(named)


def test_unnamed_robustness_number_is_rejected() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        "The robustness values ranged from 0.61 to 0.64.",
        1,
    )

    assert "MANUSCRIPT_METRIC_UNNAMED" in _codes(text)

    named = text.replace(
        "The robustness values ranged from 0.61 to 0.64.",
        "The adjusted Rand index ranged from 0.61 to 0.64.",
    )
    assert "MANUSCRIPT_METRIC_UNNAMED" not in _codes(named)

    substring_false_positive = text.replace(
        "The robustness values ranged from 0.61 to 0.64.",
        "The robustness panel reported 0.61 across variants.",
    )
    assert "MANUSCRIPT_METRIC_UNNAMED" in _codes(substring_false_positive)

    unrelated_prior_number = text.replace(
        "The robustness values ranged from 0.61 to 0.64.",
        "Table 1 and the robustness panel were generated.",
    )
    assert "MANUSCRIPT_METRIC_UNNAMED" not in _codes(unrelated_prior_number)


def test_robustness_sample_and_convergence_counts_are_not_metrics() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        (
            "The robustness analysis included 94,425 ICU stays and yielded "
            "one converged specification."
        ),
        1,
    )

    assert "MANUSCRIPT_METRIC_UNNAMED" not in _codes(text)


def test_truncated_section_ending_is_rejected() -> None:
    text = _valid_manuscript().replace(
        "Analyses were executed with versioned software and registered artifacts.",
        "The analysis record included the robustness panel",
    )

    audit = audit_manuscript_quality(text)
    finding = next(
        item
        for item in audit.findings
        if item.code == "MANUSCRIPT_SECTION_TRUNCATED"
    )
    assert finding.section == "Methods"


def test_machine_precision_is_rejected_in_reader_facing_sections() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality.",
        "The odds ratio was 1.9600187955893984.",
        1,
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_NUMERIC_OVERPRECISION" in _codes(text)
    finding = next(
        item
        for item in audit.findings
        if item.code == "MANUSCRIPT_NUMERIC_OVERPRECISION"
    )
    assert "1.9600187955893984" in finding.excerpts

    repaired, repairs = repair_reader_structure_from_existing_prose(text)
    assert "1.960" not in repaired
    assert "1.96" in repaired
    assert "MANUSCRIPT_NUMERIC_OVERPRECISION" not in _codes(repaired)
    assert repairs[0] == {
        "code": "MANUSCRIPT_NUMERIC_DISPLAY_ROUNDED",
        "source": "existing_evidence_bound_numeric_prose",
        "count": "1",
    }


def test_exact_numeric_footnote_value_is_not_reader_overprecision() -> None:
    text = _valid_manuscript() + (
        "\n[^claim_1]: value=0.463888712444; step=context; field=missingness; "
        "evidence=context; display=46.4%; match=rounded_or_transformed\n"
    )

    assert "MANUSCRIPT_NUMERIC_OVERPRECISION" not in _codes(text)


def test_discussion_cannot_deny_a_reported_risk_difference() -> None:
    text = (
        _valid_manuscript()
        .replace(
            "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
            "After adjustment for age and sex, Sepsis-3 status was associated with mortality. "
            "The risk difference was 4.9 percentage points.",
        )
        .replace(
            "The findings support an association but do not establish causation [@strobe_2007].",
            "The findings support an association, but the evidence does not provide a basis "
            "for an absolute risk difference [@strobe_2007].",
        )
    )

    assert "MANUSCRIPT_REPORTED_RESULT_DISCLAIMED" in _codes(text)

    tokenised = text.replace(
        "The risk difference was 4.9 percentage points.",
        "{claim:distribution.prespecified_unadjusted_risk_difference}",
    )
    assert "MANUSCRIPT_REPORTED_RESULT_DISCLAIMED" in _codes(tokenised)


def test_structure_repair_relabels_existing_abstract_prose() -> None:
    manuscript = _valid_manuscript().replace(
        "**Background:** Sepsis remains an important ICU syndrome.",
        "Sepsis remains an important ICU syndrome {evidence:context}.",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    assert (
        "**Background:** Sepsis remains an important ICU syndrome "
        "{evidence:context}." in repaired
    )
    assert [item["code"] for item in repairs] == ["MANUSCRIPT_ABSTRACT_LABEL_RESTORED"]


def test_structure_repair_restores_background_from_existing_evidence_prose() -> None:
    manuscript = _valid_manuscript().replace(
        "**Background:** Sepsis remains an important ICU syndrome.",
        "",
    ).replace(
        "Sepsis definitions and transparent cohort accounting matter for reproducible ICU research.",
        "Sepsis definitions and transparent cohort accounting matter "
        "for reproducible ICU research {evidence:context}.",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    abstract = repaired.split("## Abstract", 1)[1].split("## Introduction", 1)[0]
    assert (
        "**Background:** Sepsis definitions and transparent cohort accounting matter "
        "for reproducible ICU research {evidence:context}." in abstract
    )
    assert [item["code"] for item in repairs] == [
        "MANUSCRIPT_ABSTRACT_BACKGROUND_RESTORED"
    ]


def test_structure_repair_restores_existing_results_prose_slots() -> None:
    manuscript = _valid_manuscript().replace(
        "**Results:** Sepsis status was associated with in-hospital mortality.",
        "Sepsis status was associated with in-hospital mortality {evidence:primary}.",
    ).replace(
        "### Primary outcome\nThe primary outcome was in-hospital death.",
        "### Primary outcome\n",
    ).replace(
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality "
        "{evidence:primary}.",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    assert (
        "**Results:** Sepsis status was associated with in-hospital mortality "
        "{evidence:primary}." in repaired
    )
    primary_outcome = repaired.split("### Primary outcome", 1)[1].split(
        "### Primary association", 1
    )[0]
    assert "{evidence:primary}" in primary_outcome
    assert [item["code"] for item in repairs] == [
        "MANUSCRIPT_PRIMARY_OUTCOME_RESTORED",
        "MANUSCRIPT_ABSTRACT_RESULTS_RELABELED",
    ]


def test_structure_repair_copies_results_evidence_to_empty_conclusion() -> None:
    manuscript = _valid_manuscript().replace(
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality.",
        "After adjustment for age and sex, Sepsis-3 status was associated with mortality "
        "{evidence:primary}.",
    )
    manuscript = manuscript.replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.",
        "",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    conclusion = repaired.split("## Conclusion", 1)[1]
    assert "{evidence:primary}" in conclusion
    assert [item["code"] for item in repairs] == ["MANUSCRIPT_CONCLUSION_RESTORED"]


def test_structure_repair_populates_empty_abstract_conclusions_from_claim() -> None:
    manuscript = _valid_manuscript().replace(
        "**Conclusions:** The association requires external validation.",
        "**Conclusions:**",
    ).replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.",
        "{claim:primary.adjusted_association}",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    abstract = repaired.split("## Abstract", 1)[1].split("## Introduction", 1)[0]
    assert (
        "**Conclusions:**\n\n{claim:primary.adjusted_association}" in abstract
    )
    assert [item["code"] for item in repairs] == [
        "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RESTORED"
    ]


def test_structure_repair_relabels_existing_post_results_abstract_prose() -> None:
    manuscript = _valid_manuscript().replace(
        "**Conclusions:** The association requires external validation.",
        "The association requires external validation.",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    assert "**Conclusions:** The association requires external validation." in repaired
    assert [item["code"] for item in repairs] == [
        "MANUSCRIPT_ABSTRACT_CONCLUSIONS_RELABELED"
    ]


def test_structure_repair_does_not_invent_conclusion_without_evidence() -> None:
    manuscript = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.",
        "",
    )

    repaired, repairs = repair_reader_structure_from_existing_prose(manuscript)

    assert repairs == ()
    assert "## Conclusion\n\n" in repaired


def test_reader_view_removes_audit_markup_but_preserves_scientific_text() -> None:
    text = (
        _valid_manuscript().replace(
            "eligible ICU stays.",
            "94,458[^claim_1] eligible ICU stays "
            '[cohort](evidence/cohort.json "sha256=abc").',
        )
        + "\n[^claim_1]: value=94458; evidence=cohort\n"
    )

    reader = render_reader_manuscript(text)

    assert "94,458 eligible ICU stays." in reader
    assert "evidence/cohort.json" not in reader
    assert "[^claim_1]" not in reader
    assert "[@strobe_2007]" in reader


def test_write_phase_persists_quality_gate_and_non_authoritative_reader(
    tmp_path,
) -> None:
    class EvidenceStub:
        def __init__(self) -> None:
            self.records: dict[str, dict[str, object]] = {}

        def get(self, evidence_id: str):
            return self.records.get(evidence_id)

        def register_file(self, **kwargs: object) -> None:
            self.records[str(kwargs["evidence_id"])] = dict(kwargs)

    evidence = EvidenceStub()
    findings = []
    invalid = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.\n",
        "",
    )

    errors = _persist_manuscript_quality_artifacts(
        bound=invalid,
        bound_evidence_id="manuscript_scaffold_bound",
        run_dir=tmp_path,
        evidence=evidence,
        findings=findings,
    )

    assert {item.code for item in errors} == {"MANUSCRIPT_SECTION_EMPTY"}
    assert findings[0].validator == "manuscript_quality"
    assert findings[0].severity == "error"
    assert (tmp_path / "manuscript_quality_audit.json").exists()
    assert (tmp_path / "manuscript_reader.md").exists()
    assert evidence.records["manuscript_reader"]["metadata"] == {
        "authoritative_manuscript": False,
        "source_evidence_id": "manuscript_scaffold_bound",
        "source_sha256": audit_manuscript_quality(invalid).source_sha256,
    }
    assert "manuscript_quality" in _MANUSCRIPT_ERROR_VALIDATORS
