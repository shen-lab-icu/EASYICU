"""Synthetic unapproved outcome context for literature retrieval tests."""
from easyicu.research_agent.research_context.temporal_semantics import TimeWindowSemanticParser
from easyicu.research_agent.schema import CohortDescriptor, ConceptDescriptor, ResearchContext, TimeWindow, UserPreferences
from easyicu.webserver.agent_pipeline_runs import _research_user_preferences


LABEL = "从首次转入 ICU 或中间监护至病例关闭的时长"


def _context(question="SICdb 患者首次进入 ICU 或中间监护后前 24 小时乳酸与病例关闭时长的关联"):
    preferences = _research_user_preferences({"question": question, "outcome": LABEL})
    return ResearchContext(
        research_question=question,
        cohort=CohortDescriptor(cohort_name="synthetic", database="sic", n_stays=0),
        variables=[ConceptDescriptor(name="lact", dtype="float64")], primary_exposure="lact",
        user_preferences=UserPreferences(**preferences),
        temporal_constraints=TimeWindowSemanticParser().parse(question),
        time_windows=[TimeWindow(name="physical", anchor="icu_admission", start_hours=0, end_hours=24)],
    )
