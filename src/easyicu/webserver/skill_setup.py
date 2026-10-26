"""Apply attached methods through the study design and family-choice owners."""

from __future__ import annotations

from typing import Any, Mapping

from easyicu.extensions.method_attachments import method_constraints, question_family_for_method
from easyicu.webserver.question_family_check import check_question_family
from easyicu.webserver.study_family_design import family_design_for


def apply_skill_setup(study: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> dict:
    """Return only host-owned changes; the question and unrelated setup survive."""
    methods = method_constraints(study.get("attached_skills") or [])
    if not methods:
        return {}
    family = methods[0]["analysis_family"]
    question = str(study.get("question") or "").strip()
    if not question:
        return {"analysis_design": family_design_for(study, family)}
    check = check_question_family({"family": question_family_for_method(family), "words": question[:200]}, question)
    # A checked conversation reading can name a family the host's lexical
    # reader misses. Compare it too, using the existing two-reading record.
    for reading in (study.get("question_family_check") or {}, (previous or {}).get("question_family_check") or {}):
        if reading.get("words") and reading["words"] in question:
            other = reading.get("chosen_family") or reading.get("model_family")
            if reading.get("model_family") == check["model_family"]:
                other = reading.get("host_family")
            if other and other != check["model_family"] and check["host_family"] is None:
                check.update(host_family=other, status="host_disagrees")
                break
    for saved in (study.get("question_family_check") or {}, (previous or {}).get("question_family_check") or {}):
        if (saved.get("status") == "researcher_chose"
            and all(saved.get(key) == check.get(key) for key in ("model_family", "host_family", "words"))):
            check = dict(saved)
            if check["chosen_family"] != question_family_for_method(family):
                family = check["chosen_family"]
            break
    if check["status"] == "host_disagrees":
        design = {}  # The existing disagreement owner asks before planning.
    else:
        design = dict(study.get("analysis_design") or {})
        if design.get("analysis_family") != family:
            design = family_design_for(study, family)
    return {"question_family_check": check, "analysis_design": design}


def skill_planning_constraints(study: Mapping[str, Any]) -> dict:
    attachments = study.get("attached_skills") or []
    if not attachments:
        return {}
    methods = method_constraints(attachments)
    check = study.get("question_family_check") or {}
    if check.get("status") == "researcher_chose":
        methods = [item for item in methods if question_family_for_method(item["analysis_family"]) == check.get("chosen_family")]
    return {
        "attached_skills": attachments, "method_skill_constraints": methods,
        "question_family_check": check,
    }
