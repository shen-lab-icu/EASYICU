"""One project run's scientific review, read through the project binding authority."""
from easyicu.webserver.pi_copilot.contracts import PiCopilotError
from easyicu.webserver.pi_copilot.service import PiCopilotService


def run_scientific_review(service: PiCopilotService, *, project_id: str, run_id: str,
                          offset: int = 0, limit: int = 50) -> dict:
    from easyicu.webserver.scientific_review import read_scientific_review

    row = service._research_run_row(project_id, run_id)
    result = read_scientific_review(str(row.get("project_dir") or ""), offset=offset, limit=limit)
    if not result.get("ok"):
        raise PiCopilotError(
            str(result.get("error") or "scientific_review_unavailable"),
            "This run's scientific review cannot be read.",
            status_code=409 if result.get("error") == "scientific_review_privacy_blocked" else 400,
            details={"run_id": run_id},
        )
    return {"run_id": run_id, "review": result["review"]}
