"""The quality audit flags a sentence said twice and an estimate without its value.

A survival manuscript stated its proportional-hazards decision word for word
in Results, Discussion and Conclusion, and closed its Conclusion with
"(adjusted hazard ratio for days 0 to 7 after the landmark)." after the value
had been dropped, and the deterministic audit passed both.  A long sentence
repeated outside the Abstract is now a warning (the Abstract restates results
by design), and a sentence-closing parenthesis that names an effect measure
without a value is an error in the Abstract's Results and Conclusions,
Results, Discussion and Conclusion.  Methods may still name an estimand.
"""

from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_quality import audit_manuscript_quality
from easyicu.research_agent.reporting.manuscript_surface import estimates_without_values

PH = (
    "The prespecified proportional-hazards test rejected the assumption, so the interval-specific "
    "hazard ratios replaced the constant hazard ratio."
)
LOST = "Exposure was associated with death (adjusted hazard ratio for days 0 to 7 after the landmark)."
VALUED = (
    "Exposure was associated with death (adjusted hazard ratio for days 0 to 7 after the landmark, "
    "2.08; 95% CI, 1.80 to 2.40)."
)


def _manuscript(*, abstract_results="Results were reported.", abstract_conclusions="Interpretation.",
                methods="Methods were prespecified.", results="Results were reported.",
                discussion="Discussion text.", conclusion="Conclusion text.") -> str:
    return "\n".join([
        "# Study", "", "**Keywords:** survival", "", "## Abstract", "",
        "**Background:** Background.", "", "**Methods:** Methods.", "",
        f"**Results:** {abstract_results}", "", f"**Conclusions:** {abstract_conclusions}", "",
        "## Introduction", "", "Introduction text.", "",
        "## Methods", "", methods, "", "## Results", "", results, "",
        "## Discussion", "", discussion, "", "## Limitations", "", "Limitations text.", "",
        "## Conclusion", "", conclusion, "",
    ])


def _findings(text: str, code: str):
    audit = audit_manuscript_quality(text, require_administrative_sections=False)
    return [finding for finding in audit.findings if finding.code == code]


def test_a_sentence_said_again_after_the_abstract_is_a_warning():
    findings = _findings(_manuscript(results=PH, discussion=PH, conclusion=PH), "MANUSCRIPT_REPEATED_SENTENCE")

    assert [(finding.section, finding.severity) for finding in findings] == [
        ("Discussion", "warning"), ("Conclusion", "warning"),
    ]
    assert findings[0].excerpts == (PH,)


def test_the_abstract_may_restate_a_result():
    assert _findings(_manuscript(abstract_results=PH, results=PH), "MANUSCRIPT_REPEATED_SENTENCE") == []


def test_a_short_sentence_may_recur():
    assert _findings(
        _manuscript(results="See Figure 1.", discussion="See Figure 1."), "MANUSCRIPT_REPEATED_SENTENCE",
    ) == []


def test_an_estimate_without_its_value_is_an_error_where_results_are_read():
    findings = _findings(
        _manuscript(abstract_conclusions=LOST, conclusion=LOST), "MANUSCRIPT_ESTIMATE_WITHOUT_VALUE",
    )

    assert [(finding.section, finding.severity) for finding in findings] == [
        ("Abstract", "error"), ("Conclusion", "error"),
    ]
    assert findings[0].excerpts == ("(adjusted hazard ratio for days 0 to 7 after the landmark)",)


def test_a_valued_estimate_or_a_methods_estimand_is_not_flagged():
    estimand = "The primary estimand was prespecified (adjusted hazard ratio over follow-up)."
    text = _manuscript(methods=estimand, results=VALUED, conclusion=VALUED)

    assert _findings(text, "MANUSCRIPT_ESTIMATE_WITHOUT_VALUE") == []


def test_only_a_sentence_closing_measure_reads_as_a_lost_value():
    assert estimates_without_values(
        "A time-varying effect (hazard ratio declining over follow-up) was seen."
    ) == ()
    assert estimates_without_values(
        "Exposure was associated with death, as estimated by the adjusted hazard ratio."
    ) == ()
    assert estimates_without_values("The groups differed (risk difference).") == ("(risk difference)",)
