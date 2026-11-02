"""Synthetic original-question and amendment scientific review fixtures."""
import json
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from tests.support.descriptive_plan import _context, _plan


def _contract(message: str, reference: str = "") -> str:
    return "Host-bound request to revise the complete candidate plan. Changes follow.\n" + json.dumps({
        "schema_version": "easyicu.plan-change-request/1",
        "source_run_id": "run_one",
        "user_message": message,
        "reference_plans": [{"plan": {"research_question": reference}}],
    }, ensure_ascii=False)


def _review(question: str, *, amendment: str = "", plan=None):
    context = _context().model_copy(update={"research_question": question})
    plan = plan or _plan()
    plan = plan.model_copy(update={"research_question": question})
    return build_plan_scientific_review(
        context=context, plan=plan,
        bound_plan_revision_contract=_contract(amendment) if amendment else "",
    )
