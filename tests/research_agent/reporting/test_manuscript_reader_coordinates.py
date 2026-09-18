"""Reader formatting preserves already resolved scientific coordinates."""

import pytest

from easyicu.research_agent.reporting.manuscript_quality import (
    audit_manuscript_quality,
    repair_reader_internal_phrases,
)


@pytest.mark.parametrize(
    "source",
    [
        "Patient age in years and Patient sex were described.",
        "patient age in years and patient sex were described.",
    ],
)
def test_readable_labels_are_not_expanded_again(source):
    result, repairs = repair_reader_internal_phrases(
        source, reader_display_labels={"age": "Patient age in years", "sex": "Patient sex"}
    )
    assert result == source
    assert repairs == ()


def test_label_repair_is_single_pass_and_idempotent():
    labels = {"group_code": "risk group", "risk": "Risk score", "age": "Patient age in years"}
    source = "group_code and age; risk. {evidence:age} [@risk] {claim:step.age}"
    expected = "risk group and Patient age in years; Risk score. {evidence:age} [@risk] {claim:step.age}"
    once, _ = repair_reader_internal_phrases(source, reader_display_labels=labels)
    twice, second_repairs = repair_reader_internal_phrases(once, reader_display_labels=labels)
    assert once == expected
    assert twice == expected
    assert second_repairs == ()


@pytest.mark.parametrize("label", ["Results", "Conclusions"])
def test_abstract_accepts_prose_after_label_line(label):
    manuscript = "## Abstract\n\n" + "\n\n".join(
        f"**{name}:**\nThe cohort was described." if name == label
        else f"**{name}:** The cohort was described."
        for name in ("Background", "Methods", "Results", "Conclusions")
    )
    findings = audit_manuscript_quality(manuscript).findings
    assert not [item for item in findings if item.code == "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY"]


@pytest.mark.parametrize("body", ["", "[source](evidence/source.json)", "<!-- no visible result -->"])
def test_abstract_does_not_borrow_prose_from_next_label(body):
    manuscript = (
        "## Abstract\n\n**Background:** Context is described.\n\n"
        "**Methods:** Methods are described.\n\n"
        f"**Results:**\n{body}\n\n"
        "**Conclusions:** Independent validation is required.\n\n"
        "## Introduction\nPrior research is described."
    )
    findings = audit_manuscript_quality(manuscript).findings
    assert any(
        item.code == "MANUSCRIPT_ABSTRACT_LABEL_MISSING_OR_EMPTY"
        and "Results" in item.excerpts
        for item in findings
    )
