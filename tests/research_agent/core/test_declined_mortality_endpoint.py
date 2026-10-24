"""Synthetic endpoint requests preserve the requested clinical setting."""
import pytest
from easyicu.research_agent.research_context.builder import _infer_outcome_semantics

@pytest.mark.parametrize(
    ("question", "semantic"),
    [
        ("报告院内死亡，不使用 ICU 死亡替代。", "hospital_mortality"),
        ("院内死亡以 hospitaldischargestatus 为准，不能用 ICU 死亡代替。", "hospital_mortality"),
        ("报告院内死亡，而不是ICU死亡。", "hospital_mortality"),
        ("不要使用 ICU 死亡，报告院内死亡。", "hospital_mortality"),
        ("Describe hospital mortality, not ICU mortality.", "hospital_mortality"),
        ("Use hospital mortality; do not substitute ICU mortality.", "hospital_mortality"),
        ("Use hospital mortality; cannot use ICU mortality.", "hospital_mortality"),
        ("Report hospital mortality rather than ICU mortality.", "hospital_mortality"),
        ("Report hospital mortality instead of ICU mortality.", "hospital_mortality"),
        ("报告 ICU 死亡，不能用院内死亡代替。", "icu_mortality"),
        ("Use ICU mortality, not hospital mortality.", "icu_mortality"),
        ("不是院内死亡，而是 ICU 死亡。", "icu_mortality"),
        ("Describe ICU mortality.", "icu_mortality"),
        ("Do not use hospital mortality, report ICU mortality.", "icu_mortality"),
        ("Do not exclude patients with ICU mortality.", "icu_mortality"),
        ("Not only ICU mortality but also hospital mortality.", "icu_mortality"),
        ("不能只报告院内死亡，也需要 ICU 死亡。", "icu_mortality"),
        ("Describe hospital mortality, but also report ICU mortality.", "icu_mortality"),
        ("不能用 ICU 死亡替代院内死亡，但还要单独报告 ICU 死亡。", "icu_mortality"),
    ],
)
def test_only_directly_rejected_endpoint_mentions_are_ignored(question, semantic):
    assert _infer_outcome_semantics(
        research_question=question, outcome_name="death",
    )["source_concept"] == semantic

