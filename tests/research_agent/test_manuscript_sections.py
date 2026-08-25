from __future__ import annotations

import pytest

from easyicu.research_agent.reporting.manuscript_sections import (
    MANUSCRIPT_SECTION_SPECS,
    ManuscriptReaderQualityContractError,
    ManuscriptSectionContractError,
    render_manuscript_sections,
)
from easyicu.research_agent.reporting.administrative_authority import (
    ManuscriptAdministrativeAuthority,
)


def test_manuscript_section_contract_has_fixed_publication_order() -> None:
    assert [spec.key for spec in MANUSCRIPT_SECTION_SPECS] == [
        "title",
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "limitations",
        "conclusion",
    ]
    assert [spec.max_tokens for spec in MANUSCRIPT_SECTION_SPECS] == [
        256,
        1024,
        4096,
        2048,
        2048,
        4096,
        1024,
        512,
    ]


def _minimal_valid_section(section_name: object) -> str:
    name = str(section_name)
    if name == "Title and Keywords":
        return """# Evidence-bound ICU retrospective cohort analysis

**Keywords:** intensive care, cohort, evidence, methods, reproducibility"""
    if name == "Abstract":
        return """## Abstract

**Background:** Evidence-bound reporting supports reproducible ICU research.

**Methods:** We conducted a retrospective cohort analysis.

**Results:** The registered analysis produced a bounded result.

**Conclusions:** Independent validation remains required."""
    if name == "Methods":
        return """## Methods

### Study design and cohort
Evidence-bound design prose.

### Variables
Evidence-bound variable prose.

### Statistical analysis
Evidence-bound analysis prose.

### Software and reproducibility
Evidence-bound software prose."""
    if name == "Results":
        return """## Results

### Cohort characteristics
Evidence-bound cohort prose.

### Primary outcome
Evidence-bound outcome prose.

### Primary association
Evidence-bound association prose.

### Sensitivity and subgroup analyses
Evidence-bound sensitivity prose."""
    return f"## {name}\n\nEvidence-bound {name.lower()} prose."


def test_manuscript_section_assembly_is_ordered_and_forwards_common_context() -> None:
    seen: list[tuple[str, object]] = []

    def call_section(**kwargs: object) -> str:
        seen.append((str(kwargs["section_name"]), kwargs["context"]))
        return _minimal_valid_section(kwargs["section_name"])

    rendered = render_manuscript_sections(
        call_section=call_section,
        common={"context": "sealed-context", "evidence_ids": ("e1",)},
    )

    expected_names = [spec.section_name for spec in MANUSCRIPT_SECTION_SPECS]
    expected_headings = [
        "# Evidence-bound ICU retrospective cohort analysis",
        *[f"## {name}" for name in expected_names[1:]],
    ]
    heading_positions = [rendered.index(heading) for heading in expected_headings]
    assert heading_positions == sorted(heading_positions)
    assert "requires author verification" in rendered
    assert "released alongside this manuscript" not in rendered
    assert "declare no conflicts" not in rendered
    assert seen == [(name, "sealed-context") for name in expected_names]


def test_manuscript_section_failure_stops_before_later_provider_calls() -> None:
    seen: list[str] = []

    def call_section(**kwargs: object) -> str:
        section_name = str(kwargs["section_name"])
        seen.append(section_name)
        if section_name == "Methods":
            raise RuntimeError("provider stop-loss")
        return _minimal_valid_section(section_name)

    with pytest.raises(RuntimeError, match="provider stop-loss"):
        render_manuscript_sections(call_section=call_section, common={})

    assert seen == ["Title and Keywords", "Abstract", "Introduction", "Methods"]


def test_manuscript_section_assembly_restores_missing_mechanical_heading() -> None:
    def call_section(**kwargs: object) -> str:
        if kwargs["section_name"] == "Conclusion":
            return "The evidence-bound conclusion sentence."
        return _minimal_valid_section(kwargs["section_name"])

    rendered = render_manuscript_sections(call_section=call_section, common={})

    assert "## Conclusion\n\nThe evidence-bound conclusion sentence." in rendered


def test_section_specs_keep_literature_and_evidence_boundaries() -> None:
    instructions = {spec.key: spec.instruction for spec in MANUSCRIPT_SECTION_SPECS}
    assert "direct-comparator" in instructions["introduction"]
    assert "method-source key" in instructions["methods"]
    assert "{evidence:id}" in instructions["results"]
    assert "reportable_secondary_results" in instructions["results"]
    assert "reportable_descriptive_results" in instructions["results"]
    assert "source concept's default aggregation" in instructions["methods"]
    assert "machine digest explicitly records" in instructions["limitations"]
    assert "specific population" in instructions["discussion"]
    assert "host owns those administrative facts" in instructions["conclusion"]
    assert "released alongside this manuscript" not in instructions["methods"]
    assert "Copy the executed adjustment set" in instructions["methods"]
    assert "Name the exact metric" in instructions["results"]
    assert "Do not list artifacts or praise the pipeline" in instructions["discussion"]
    for key in ("abstract", "introduction", "results", "discussion", "conclusion"):
        assert "raw snake_case" in instructions[key]


def test_incomplete_required_subsection_gets_one_targeted_retry() -> None:
    methods_calls = 0

    def call_section(**kwargs: object) -> str:
        nonlocal methods_calls
        if kwargs["section_name"] != "Methods":
            return _minimal_valid_section(kwargs["section_name"])
        methods_calls += 1
        if methods_calls == 1:
            return """## Methods

### Study design and cohort
Design prose.

### Variables
Variable prose.

### Statistical analysis

### Software and reproducibility
Software prose."""
        assert "STRUCTURAL CONTRACT REPAIR" in str(kwargs["instruction"])
        assert "`### Statistical analysis`" in str(kwargs["instruction"])
        return _minimal_valid_section("Methods")

    rendered = render_manuscript_sections(call_section=call_section, common={})

    assert methods_calls == 2
    assert "### Statistical analysis\nEvidence-bound analysis prose." in rendered


def test_incomplete_required_subsection_fails_closed_after_retry() -> None:
    seen: list[str] = []

    def call_section(**kwargs: object) -> str:
        section_name = str(kwargs["section_name"])
        seen.append(section_name)
        if section_name == "Methods":
            return "## Methods\n\n### Statistical analysis\n"
        return _minimal_valid_section(section_name)

    with pytest.raises(
        ManuscriptSectionContractError,
        match="missing or empty required subsections after one targeted retry",
    ):
        render_manuscript_sections(call_section=call_section, common={})

    assert seen == [
        "Title and Keywords",
        "Abstract",
        "Introduction",
        "Methods",
        "Methods",
    ]


def test_empty_section_body_gets_one_targeted_retry() -> None:
    conclusion_calls = 0

    def call_section(**kwargs: object) -> str:
        nonlocal conclusion_calls
        if kwargs["section_name"] != "Conclusion":
            return _minimal_valid_section(kwargs["section_name"])
        conclusion_calls += 1
        if conclusion_calls == 1:
            return "## Conclusion"
        assert "the main section body" in str(kwargs["instruction"])
        return "## Conclusion\n\nEvidence-bound conclusion prose."

    rendered = render_manuscript_sections(call_section=call_section, common={})

    assert conclusion_calls == 2
    assert "## Conclusion\n\nEvidence-bound conclusion prose." in rendered


def test_reader_quality_retries_only_abstract_with_missing_label() -> None:
    calls: list[str] = []
    abstract_calls = 0

    def call_section(**kwargs: object) -> str:
        nonlocal abstract_calls
        section_name = str(kwargs["section_name"])
        calls.append(section_name)
        if section_name != "Abstract":
            return _minimal_valid_section(section_name)
        abstract_calls += 1
        if abstract_calls == 1:
            return """## Abstract

**Background:** Bounded background.

**Methods:** Bounded methods.

**Results:** Bounded results.

**Conclusions:**"""
        assert "READER-QUALITY CONTRACT REPAIR" in str(kwargs["instruction"])
        assert "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY" in str(
            kwargs["instruction"]
        )
        return _minimal_valid_section("Abstract")

    rendered = render_manuscript_sections(call_section=call_section, common={})

    assert abstract_calls == 2
    assert calls.count("Introduction") == 1
    assert "**Conclusions:** Independent validation remains required." in rendered


def test_reader_quality_retries_methods_and_results_for_adjustment_conflict() -> None:
    methods_calls = 0
    results_calls = 0

    def call_section(**kwargs: object) -> str:
        nonlocal methods_calls, results_calls
        section_name = str(kwargs["section_name"])
        if section_name == "Methods":
            methods_calls += 1
            adjustment = (
                "age and sex" if methods_calls == 1 else "age and Charlson score"
            )
            return _minimal_valid_section("Methods").replace(
                "Evidence-bound analysis prose.",
                f"The adjustment set comprised {adjustment}. We used logistic regression.",
            )
        if section_name == "Results":
            results_calls += 1
            adjustment = "age and Charlson score"
            return _minimal_valid_section("Results").replace(
                "Evidence-bound association prose.",
                f"After adjustment for {adjustment}, exposure was associated with mortality.",
            )
        return _minimal_valid_section(section_name)

    rendered = render_manuscript_sections(call_section=call_section, common={})

    assert methods_calls == 2
    assert results_calls == 2
    assert "The adjustment set comprised age and Charlson score." in rendered


def test_reader_quality_fails_closed_when_targeted_retry_still_leaks_internal_term() -> (
    None
):
    def call_section(**kwargs: object) -> str:
        section_name = str(kwargs["section_name"])
        if section_name == "Discussion":
            return "## Discussion\n\nThe result remained host-bound."
        return _minimal_valid_section(section_name)

    with pytest.raises(
        ManuscriptReaderQualityContractError,
        match="MANUSCRIPT_INTERNAL_TERM_EXPOSED",
    ):
        render_manuscript_sections(call_section=call_section, common={})


def test_verified_administrative_authority_is_rendered_exactly() -> None:
    authority = ManuscriptAdministrativeAuthority.issue(
        authority_id="submission-metadata-v1",
        verified_by="corresponding author",
        verified_at="2026-08-14T02:00:00Z",
        data_and_code_availability="Verified data statement.",
        funding="Verified funding statement.",
        ethics="Verified ethics statement.",
        conflicts_of_interest="Verified disclosure statement.",
        artifact_release="Verified artifact inventory statement.",
    )

    rendered = render_manuscript_sections(
        call_section=lambda **kwargs: _minimal_valid_section(kwargs["section_name"]),
        common={},
        administrative_authority=authority,
    )

    assert "Verified data statement." in rendered
    assert "Verified funding statement." in rendered
    assert "Verified ethics statement." in rendered
    assert "Verified disclosure statement." in rendered
    assert "Verified artifact inventory statement." in rendered
