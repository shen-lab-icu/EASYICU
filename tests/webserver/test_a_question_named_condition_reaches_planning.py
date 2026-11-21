"""A recorded condition the question restricts its patients to reaches planning.

The launch reads the condition from the question's own words and passes it to
planning as a constraint, with the concepts that record it, the host's stated
reading first (S1757).  A researcher's recorded eligibility decision stands:
then nothing is passed.  Planning holds the plan to it
(``planning.stated_population``).
"""

from __future__ import annotations

import json

from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.research_launch_population import stated_population_constraints


def test_the_host_constraint_comes_from_the_question_unless_the_researcher_decided() -> None:
    question = "脓毒症 ICU 患者中，ICU 入科后首 24 小时峰值乳酸与院内死亡的关联。"
    found = stated_population_constraints({"question": question})
    assert found == [{"quote": "脓毒症", "concepts": ["sep3_sofa1", "sep3", "sep3_sofa2"]}]
    for state in ("confirmed", "declined"):
        decided = {"question": question, "cohort_eligibility_authority": {"selection_state": state}}
        assert stated_population_constraints(decided) == []
    assert stated_population_constraints({"question": "ICU 入科后首 24 小时峰值乳酸与院内死亡的关联。"}) == []
    # A study saved before its question was written passes nothing.
    assert stated_population_constraints({}) == []
    assert stated_population_constraints({"question": "  "}) == []


def test_the_launch_passes_the_condition_to_planning() -> None:
    question = "脓毒症 ICU 患者中，ICU 入科后首 24 小时峰值乳酸与院内死亡的关联。"
    window = {"hours": 24, "anchor": "ICU admission"}
    for study, expected in (
        ({"question": question, "time_window": window},
         [{"quote": "脓毒症", "concepts": ["sep3_sofa1", "sep3", "sep3_sofa2"]}]),
        ({"question": question, "time_window": window,
          "cohort_eligibility_authority": {"selection_state": "declined"}}, None),
    ):
        preferences = agent_pipeline_runs._research_user_preferences(study)
        assert json.loads(preferences["data_constraints"]).get("stated_population_conditions") == expected
