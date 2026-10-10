"""A substance both measured and given is read by what the question says of it.

The concept catalog holds some substances twice: as a level measured and as
what is given -- serum albumin and albumin given intravenously, a platelet
count and a platelet transfusion.  The Web reader (``webserver.study_intent``)
reads administration words (静脉、输注、给予、补充、开始、infusion、IV) as the
substance given and level words (血清、水平、浓度、计数、serum、level) as the
measurement; with neither it reads neither, and the exposure stays unread
rather than defaulting to the laboratory value.  A decided reading reaches
planning (``question_substance_forms``), where a plan that reads the other
form stops (``planning.question_substance_forms``).  Synthetic sentences only.
"""

from __future__ import annotations

import json
import re

import pytest

from easyicu.concept.catalog import CONCEPT_DICTIONARY, CONCEPT_GROUPS_INTERNAL
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.webserver import (
    agent_pipeline_runs,
    research_launch_scientific,
    study_intent,
)
from easyicu.webserver.study_intent import (
    deterministic_intent,
    named_study_concepts,
    substance_forms,
    substance_reading,
)
from tests.webserver.copilot.research_workflow_fixtures import complete_study

_TABLE = dict(study_intent._MEASURED_OR_GIVEN)
_LAB_MODULES = ("blood_gas", "chemistry", "hematology")
#: Catalog names that hold a laboratory value's name without being it given.
_NOT_GIVEN_FORMS = {
    ("na", "bicarbonate"),  # sodium bicarbonate is bicarbonate given
    ("cl", "ketamine"),  # 氯 in 氯胺酮 is no chloride
}
#: A substance given named otherwise than its level.
_NAMED_OTHERWISE = {("rbc", "packed_rbc"), ("glu", "dex"), ("glu", "dextrose50")}


def _contains(name: str, other: str) -> bool:
    return bool(
        re.search(rf"(?<![a-z]){re.escape(other.lower())}(?![a-z])", name.lower())
    )


def test_the_table_holds_each_catalog_substance_both_measured_and_given() -> None:
    labs = {
        concept
        for module in _LAB_MODULES
        for concept in CONCEPT_GROUPS_INTERNAL[module]
    }
    given = set(CONCEPT_GROUPS_INTERNAL["medications"])
    named_alike = {
        (lab, drug)
        for drug in given
        for lab in labs
        if lab in CONCEPT_DICTIONARY
        and (
            _contains(CONCEPT_DICTIONARY[drug][0], CONCEPT_DICTIONARY[lab][0])
            or CONCEPT_DICTIONARY[lab][1] in CONCEPT_DICTIONARY[drug][1]
        )
    }
    tabled = {(lab, drug) for lab, drugs in _TABLE.items() for drug in drugs}

    assert named_alike - _NOT_GIVEN_FORMS <= tabled
    assert tabled == (named_alike - _NOT_GIVEN_FORMS) | _NAMED_OTHERWISE
    assert set(_TABLE) <= labs
    assert {drug for drugs in _TABLE.values() for drug in drugs} <= given


_WORDS = {
    "alb": "albumin",
    "bicar": "bicarbonate",
    "ca": "calcium",
    "mg": "magnesium",
    "k": "potassium",
    "plt": "platelets",
    "rbc": "red blood cells",
    "glu": "glucose",
}


def _read(sentence: str, measured: str) -> tuple[str, ...]:
    word = _WORDS[measured]
    start = sentence.index(word)
    return substance_reading(
        sentence, concept_id=measured, start=start, end=start + len(word)
    )


@pytest.mark.parametrize("measured", sorted(_TABLE))
def test_each_substance_is_read_as_its_words_say(measured: str) -> None:
    word, given = _WORDS[measured], _TABLE[measured]

    infused = _read(
        f"Is intravenous {word} in the first day associated with death?", measured
    )
    started = _read(f"Does starting {word} within 24 hours change death?", measured)
    level = _read(
        f"Is the serum {word} level on day one associated with death?", measured
    )
    unsaid = _read(f"Is {word} associated with death?", measured)

    # Given: the one substance given, or every one when several are.
    assert infused == started == given
    assert level == (measured,)
    # Neither: every concept the words can mean, never the level alone.
    assert unsaid == (measured, *given)


def test_the_nearest_word_of_its_clause_decides() -> None:
    # Another analyte's level across a conjunction says nothing of this one.
    assert _read("Is albumin and serum lactate associated with death?", "alb") == (
        "alb",
        "albumin_iv",
    )
    # A word as near on each side says both: undecided.
    assert _read("serum albumin infusion", "alb") == ("alb", "albumin_iv")
    # A word before a clause break says nothing of what follows it.
    assert _read("IV fluids; is albumin associated with death?", "alb") == (
        "alb",
        "albumin_iv",
    )
    # "补钾": the verb joined to the substance it supplies.
    assert substance_reading("首日补钾与死亡", concept_id="k", start=3, end=4) == (
        "potassium_iv",
    )
    # A concept the catalog holds once is read as it is.
    assert substance_reading("serum lactate", concept_id="lact", start=6, end=13) == (
        "lact",
    )


def _read_word(sentence: str, measured: str, word: str) -> tuple[str, ...]:
    start = sentence.index(word)
    return substance_reading(
        sentence, concept_id=measured, start=start, end=start + len(word)
    )


@pytest.mark.parametrize(
    ("sentence", "measured", "word", "reading"),
    [
        pytest.param("首日血钾与院内死亡", "k", "钾", ("k",), id="blood-potassium"),
        pytest.param(
            "首日血糖与院内死亡", "glu", "血糖", ("glu",), id="its-name-blood-glucose"
        ),
        pytest.param(
            "首日血白蛋白与院内死亡", "alb", "白蛋白", ("alb",), id="blood-albumin"
        ),
        # 静脉血 is venous blood: the level is read in it, nothing is given.
        pytest.param("首日静脉血钾与院内死亡", "k", "钾", ("k",), id="venous-blood"),
        # 血 in a blood cell's own name says nothing of a level.
        pytest.param(
            "首日血小板与院内死亡",
            "plt",
            "血小板",
            ("plt", "platelets"),
            id="a-cells-name",
        ),
        pytest.param(
            "首日输注血小板与院内死亡",
            "plt",
            "血小板",
            ("platelets",),
            id="cells-given",
        ),
    ],
)
def test_blood_joined_to_a_substance_of_its_chemistry_names_its_level(
    sentence: str, measured: str, word: str, reading: tuple[str, ...]
) -> None:
    assert _read_word(sentence, measured, word) == reading


@pytest.mark.parametrize(
    ("sentence", "measured", "word", "reading"),
    [
        pytest.param(
            "Is albumin < 30 g/L at admission associated with death?",
            "alb",
            "albumin",
            ("alb",),
            id="g-per-litre",
        ),
        pytest.param(
            "首日白蛋白 <30 g/L 与院内死亡",
            "alb",
            "白蛋白",
            ("alb",),
            id="chinese-text",
        ),
        pytest.param(
            "Is platelets < 50×10^9/L on day one associated with death?",
            "plt",
            "platelets",
            ("plt",),
            id="cells-per-litre",
        ),
        pytest.param(
            "Is platelets < 50×10⁹/L on day one associated with death?",
            "plt",
            "platelets",
            ("plt",),
            id="cells-per-litre-superscript",
        ),
        pytest.param(
            "Is potassium below 3.5 mEq/L associated with arrhythmia?",
            "k",
            "potassium",
            ("k",),
            id="meq-per-litre",
        ),
        # Farther than a level word is read, inside its clause.
        pytest.param(
            "Is glucose in the first 24 hours under 70 mg/dL associated with death?",
            "glu",
            "glucose",
            ("glu",),
            id="later-in-the-clause",
        ),
        # A dose or a rate is no level.
        pytest.param(
            "Is potassium 20 mmol/h in the first day associated with death?",
            "k",
            "potassium",
            ("k", "potassium_iv"),
            id="a-rate",
        ),
        pytest.param(
            "Is albumin 25 g on day one associated with death?",
            "alb",
            "albumin",
            ("alb", "albumin_iv"),
            id="a-dose",
        ),
        # Another clause's value says nothing of this substance.
        pytest.param(
            "Is albumin associated with death, given lactate above 2 mmol/L?",
            "alb",
            "albumin",
            ("alb", "albumin_iv"),
            id="another-clause",
        ),
        pytest.param(
            "Given lactate above 2 mmol/L, is albumin associated with death?",
            "alb",
            "albumin",
            ("alb", "albumin_iv"),
            id="an-earlier-clause",
        ),
        # A nearer word of administration decides.
        pytest.param(
            "Is magnesium infusion at 2.0 mg/dL associated with death?",
            "mg",
            "magnesium",
            ("magnesium_iv",),
            id="nearer-given",
        ),
    ],
)
def test_a_value_in_a_unit_of_concentration_is_a_level(
    sentence: str, measured: str, word: str, reading: tuple[str, ...]
) -> None:
    assert _read_word(sentence, measured, word) == reading


@pytest.mark.parametrize(
    ("question", "slots"),
    [
        pytest.param(
            "成人 ICU 患者入科 24 小时内低血糖（血糖 <4.0 mmol/L）的比例是多少？"
            "低血糖、高血糖（>10 mmol/L）和血糖正常三组的院内死亡率有何不同？",
            {
                "analysis_family": None,
                "comparator": None,
                "exposure": None,
                "outcome": "death",
                "outcome_type": "binary",
                "population": "Adult ICU patients",
                "time_window_hours": 24,
            },
            id="glucose-groups",
        ),
        pytest.param(
            "Is albumin < 30 g/L at ICU admission associated with 28-day mortality?",
            {"exposure": "alb"},
            id="albumin-threshold",
        ),
        pytest.param(
            "Is platelet < 50×10^9/L associated with death in adult ICU patients?",
            {"exposure": "plt"},
            id="platelet-threshold",
        ),
        pytest.param(
            "成人 ICU 患者白蛋白 <30 g/L 与 28 天死亡",
            {"exposure": "alb"},
            id="albumin-threshold-chinese",
        ),
    ],
)
def test_a_level_stated_by_its_value_keeps_the_reading_it_had(
    question: str, slots: dict
) -> None:
    # Each reading is the one the reader gave before a substance was read
    # as measured or given: a level named by its value stays the level.
    read = deterministic_intent(question)["slots"]

    assert {name: read[name]["value"] for name in slots} == slots


@pytest.mark.parametrize(
    ("question", "exposure", "named"),
    [
        pytest.param(
            "Among adult ICU patients, is intravenous albumin given in the first "
            "24 hours associated with 28-day mortality?",
            "albumin_iv",
            (("albumin_iv",), "albumin"),
            id="given",
        ),
        pytest.param(
            "成人 ICU 患者入科首日血清白蛋白水平与院内死亡的关系",
            "alb",
            (("alb",), "白蛋白"),
            id="measured",
        ),
        pytest.param(
            "Is albumin associated with ICU length of stay among adult ICU patients?",
            None,
            (("alb", "albumin_iv"), "albumin"),
            id="unsaid",
        ),
        pytest.param(
            "首日输注血小板与 ICU 患者院内死亡的关系",
            "platelets",
            (("platelets",), "血小板"),
            id="transfused-platelets",
        ),
    ],
)
def test_the_reader_names_the_exposure_the_question_states(
    question: str, exposure: str | None, named: tuple
) -> None:
    slots = deterministic_intent(question)["slots"]

    assert slots["exposure"]["value"] == exposure
    assert named_study_concepts(question)[0] == named


def test_an_unsaid_substance_leaves_the_exposure_to_no_later_concept() -> None:
    question = "白蛋白与乳酸对 ICU 患者院内死亡的影响"

    assert deterministic_intent(question)["slots"]["exposure"]["value"] is None
    assert named_study_concepts(question)[:2] == (
        (("alb", "albumin_iv"), "白蛋白"),
        (("lact",), "乳酸"),
    )


def test_a_later_mention_decides_an_earlier_one() -> None:
    question = (
        "Is albumin, given intravenously as albumin infusion, associated with death?"
    )

    assert deterministic_intent(question)["slots"]["exposure"]["value"] == "albumin_iv"


def test_a_decided_reading_reaches_planning_with_its_other_form() -> None:
    question = (
        "在成人 ICU 住院中，比较入 ICU 后 24 小时内开始静脉输注白蛋白与不开始，"
        "28 天死亡风险有何差异？"
    )

    coordinates = research_launch_scientific._metadata_only_planning_coordinates(
        question=question, database="miiv"
    )

    assert substance_forms(question) == (
        {
            "concepts": ["albumin_iv"],
            "form": "given",
            "other": ["alb"],
            "evidence": "白蛋白",
        },
    )
    assert coordinates["question_substance_forms"] == [
        {
            "concepts": ["albumin_iv"],
            "form": "given",
            "other": ["alb"],
            "evidence": "白蛋白",
        }
    ]
    # A substance named without saying which form records none.
    assert substance_forms("Is albumin associated with death?") == ()
    # A level named as measured records the substance given as its other form.
    assert substance_forms("成人 ICU 患者入科首日血清白蛋白水平与院内死亡的关系") == (
        {
            "concepts": ["alb"],
            "form": "measured",
            "other": ["albumin_iv"],
            "evidence": "白蛋白",
        },
    )


def test_the_run_carries_the_reading_and_its_stop_says_what_to_change() -> None:
    form = {
        "concepts": ["albumin_iv"],
        "form": "given",
        "other": ["alb"],
        "evidence": "白蛋白",
    }

    preferences = agent_pipeline_runs._research_user_preferences(
        complete_study(), question_substance_forms=[form]
    )
    message = agent_pipeline_runs._progressive_compile_failure_message(
        ProgressivePlanCompileError(
            "progressive_question_substance_form_substituted",
            "The question names '白蛋白' as given (albumin_iv).",
        )
    )

    assert json.loads(preferences["data_constraints"])["question_substance_forms"] == [
        form
    ]
    assert "no analysis was run" in message
    assert "to study the other form, say so in the question" in message
    # The compiler's message names the study's words; the sentence does not.
    assert "白蛋白" not in message
