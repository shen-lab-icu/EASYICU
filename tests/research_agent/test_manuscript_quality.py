from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
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
"""


def _codes(text: str) -> set[str]:
    return {finding.code for finding in audit_manuscript_quality(text).findings}


def test_complete_reader_facing_manuscript_passes() -> None:
    audit = audit_manuscript_quality(_valid_manuscript())

    assert audit.status == "pass"
    assert audit.adjustment_sets == {
        "Methods": ("age", "sex"),
        "Results": ("age", "sex"),
    }
    assert not audit.findings


def test_empty_conclusion_fails_closed() -> None:
    text = _valid_manuscript().replace(
        "Sepsis status was associated with in-hospital mortality and requires external validation.\n",
        "",
    )

    audit = audit_manuscript_quality(text)

    assert audit.status == "changes_required"
    assert "MANUSCRIPT_SECTION_EMPTY" in _codes(text)
    assert any(finding.section == "Conclusion" for finding in audit.findings)


def test_methods_results_adjustment_conflict_is_reported() -> None:
    text = _valid_manuscript().replace(
        "After adjustment for age and sex, Sepsis-3 status was associated",
        "After adjustment for age and charlson_max, Sepsis-3 status was associated",
    )

    audit = audit_manuscript_quality(text)

    assert "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT" in _codes(text)
    assert audit.adjustment_sets["Methods"] == ("age", "sex")
    assert audit.adjustment_sets["Results"] == ("age", "charlson_max")


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


def test_reader_view_removes_audit_markup_but_preserves_scientific_text() -> None:
    text = _valid_manuscript().replace(
        "eligible ICU stays.",
        "94,458[^claim_1] eligible ICU stays "
        "[cohort](evidence/cohort.json \"sha256=abc\").",
    ) + "\n[^claim_1]: value=94458; evidence=cohort\n"

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
