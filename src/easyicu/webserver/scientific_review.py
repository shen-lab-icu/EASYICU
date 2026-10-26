"""Read-only scientific review of one run, including separately sealed appends.

Only existing browser-safe records enter the projection. No statistic, gate,
clinical judgment, execution authority or publication permission is computed.
"""

from __future__ import annotations

from easyicu.research_agent.contracts.frozen_payload import thaw_payload

from .scientific_review_items import MAX_PAGE, ReviewItems, mapping, page, rows, token
from .scientific_review_numbers import number_check
from .scientific_review_sections import review_sections


def project_scientific_review(payloads, *, artifacts=(), offset=0, limit=MAX_PAGE):
    """One bounded page per tab; counts cover all supplied records, not a page.

    ``number_check.coverage.answer`` explicitly names recorded answer inputs:
    the UI must use these source coordinates when rendering result numbers.
    It must not interpret the inventory as a prose audit of arbitrary replies.
    """
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_PAGE:
        raise ValueError("scientific_review_page_invalid")
    payloads = thaw_payload(payloads)
    context = mapping(payloads.get("run_context.json"))
    gate = mapping(mapping(payloads.get("quality_gate.json")).get("gate"))
    manifest = mapping(payloads.get("source_run_manifest.json"))
    checks = {mapping(row).get("id"): mapping(row).get("passed") for row in rows(gate.get("checks"))}
    finished = checks.get("execution_complete") is True or gate.get("status") in {"analysis_only", "reportable"}
    items = ReviewItems(artifacts)
    sections = review_sections(payloads, items)
    return {
        "schema_version": "easyicu.run-scientific-review/1", "run_id": token(context.get("run_id")),
        "status": "finished" if finished else "not_finished",
        "analysis_only": gate.get("status") == "analysis_only" or checks.get("paper_authorized") is not True,
        "publication_authorized": False,
        "append_source_run_id": token(mapping(manifest.get("append")).get("source_run_id")),
        **{key: page(value, offset=offset, limit=limit) for key, value in sections.items()},
        "number_check": number_check(payloads, items, offset=offset, limit=limit),
    }


def read_scientific_review(project_dir: str, *, offset=0, limit=MAX_PAGE) -> dict:
    """Use the existing bounded run reader and its privacy boundary; write nothing."""
    from .agent_runs import read_run_record
    from .run_artifact_disclosure import scan_browser_projection

    record = read_run_record(project_dir)
    if not record.ok:
        return {"ok": False, "error": record.error}
    payloads = thaw_payload(record.artifact_payloads)
    payloads["quality_gate.json"] = {"gate": record.gate.to_dict()}
    try:
        review = project_scientific_review(payloads, artifacts=[row.to_dict() for row in record.artifacts],
                                           offset=offset, limit=limit)
    except ValueError:
        return {"ok": False, "error": "scientific_review_page_invalid"}
    if not scan_browser_projection({"scientific_review": review})["passed"]:
        return {"ok": False, "error": "scientific_review_privacy_blocked"}
    return {"ok": True, "review": review}
