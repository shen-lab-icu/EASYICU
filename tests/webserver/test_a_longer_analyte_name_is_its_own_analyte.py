"""A longer analyte name is its own analyte, never the shorter one inside it.

The intent reader maps clinical phrases onto catalog concepts, and its phrases
also matched inside longer names.  "Lactate" in "lactate dehydrogenase" and
"乳酸" in "乳酸脱氢酶" read a question about LDH as a question about lactate.
In the same way it read glycated haemoglobin as haemoglobin, a platelet-to-
lymphocyte ratio as platelets, direct bilirubin as total bilirubin, diastolic
blood pressure as systolic, a shock index as circulatory failure, and hospital
length of stay as ICU length of stay.

A reading inside another concept's longer catalog name now reads as that
concept.  A term of a ratio or a clearance that the catalog does not name is
not read, so the slot stays unread rather than naming the analyte it is
computed from.
"""

from __future__ import annotations

import pytest

from easyicu.webserver import study_intent


def _read(question: str) -> tuple:
    slots = study_intent.deterministic_intent(question)["slots"]
    return slots["exposure"]["value"], slots["outcome"]["value"]


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        ("乳酸脱氢酶与28天死亡率的关系", "ldh", "mort_28d"),
        ("Is lactate dehydrogenase associated with 28-day mortality?", "ldh", "mort_28d"),
        ("LDH and 28-day mortality in sepsis", "ldh", "mort_28d"),
        ("Is glycated hemoglobin associated with in-hospital mortality?", "hba1c", "death"),
        ("糖化血红蛋白与院内死亡", "hba1c", "death"),
        ("Is the platelet-to-lymphocyte ratio associated with 28-day mortality?", "plr", "mort_28d"),
        ("Is the platelet to lymphocyte ratio associated with 28-day mortality?", "plr", "mort_28d"),
        ("platelet/lymphocyte ratio and in-hospital mortality", "plr", "death"),
        ("血小板与淋巴细胞比值与28天死亡", "plr", "mort_28d"),
        ("Is direct bilirubin associated with ICU mortality?", "bili_dir", "death"),
        ("直接胆红素与死亡", "bili_dir", "death"),
        ("Is diastolic blood pressure associated with AKI?", "dbp", "aki"),
        ("休克指数与28天死亡", "shock_index", "mort_28d"),
        ("Is lactate associated with hospital length of stay?", "lact", "los_hosp"),
        # The longest name wins: not haemoglobin, nor mean corpuscular haemoglobin.
        ("Is mean corpuscular hemoglobin concentration associated with mortality?", "mchc", "death"),
    ],
)
def test_a_longer_name_reads_as_its_own_concept(question, exposure, outcome):
    assert _read(question) == (exposure, outcome)


@pytest.mark.parametrize(
    "question",
    [
        "Is lactate clearance associated with in-hospital mortality?",
        "乳酸清除率与院内死亡",
        "Is the ratio of BUN to creatinine associated with mortality?",
        "Is the lactate/pyruvate ratio associated with in-hospital mortality?",
    ],
)
def test_a_term_of_a_measure_the_catalog_does_not_name_is_not_read(question):
    exposure, _outcome = _read(question)
    assert exposure is None


@pytest.mark.parametrize(
    ("question", "exposure", "outcome"),
    [
        ("Is lactate associated with in-hospital mortality?", "lact", "death"),
        ("Is total bilirubin associated with ICU mortality?", "bili", "death"),
        ("Does creatinine predict 28-day mortality compared with lactate?", "crea", "mort_28d"),
        ("乳酸与死亡的比较", "lact", "death"),
        ("Is mean arterial pressure associated with AKI?", "map", "aki"),
        ("Is the PaO2/FiO2 ratio associated with in-hospital mortality?", "pafi", "death"),
    ],
)
def test_a_name_no_longer_name_contains_reads_as_before(question, exposure, outcome):
    assert _read(question) == (exposure, outcome)


def test_the_outcome_roster_reads_the_longer_name() -> None:
    assert study_intent.explicit_outcome_concepts(
        "Is lactate associated with hospital length of stay?"
    ) == ("los_hosp",)
    assert study_intent.explicit_outcome_concepts(
        "Is lactate associated with ICU length of stay?"
    ) == ("los_icu",)


def test_a_measurement_operation_binds_only_the_analyte_it_names() -> None:
    assert study_intent.explicit_exposure_aggregation("最高乳酸脱氢酶与死亡", concept_id="lact") is None
    ldh = study_intent.explicit_exposure_aggregation("最高乳酸脱氢酶与死亡", concept_id="ldh")
    assert (ldh.aggregation, ldh.evidence) == ("max", "最高乳酸脱氢酶")
    lactate = study_intent.explicit_exposure_aggregation("maximum lactate and death", concept_id="lact")
    assert (lactate.aggregation, lactate.evidence) == ("max", "maximum lactate")
