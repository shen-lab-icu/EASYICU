"""Source-bound synthetic scientific plan fixture."""


def _real_plan():
    from tests.research_agent.planning.scientific_review_fixtures import _plan
    from easyicu.research_agent.planning.literature_contract import LiteratureDesignBinding
    original = _plan()
    steps = []
    for step in original.steps:
        if step.literature_citation_keys:
            step = step.model_copy(update={
                "literature_citation_keys": ["source_1"],
                "literature_design_bindings": [LiteratureDesignBinding(
                    citation_key="source_1", design_elements=["population"],
                    application="Describe the population of the original database resource.",
                )],
            })
        steps.append(step)
    return original.model_copy(update={"steps": steps})
