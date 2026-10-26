"""Review existing numeric bindings, never infer matches from prose or values.

Answer coordinates refer to the registered inputs of the run-answer renderer.
Manuscript counts are occurrences, as recorded by the Writer binder/reader.
Legacy numeric audits remain readable, including their explicit truncation.
"""

from __future__ import annotations

import hashlib
from typing import Any

from easyicu.research_agent.reporting.result_integrity import _plausibility_number
from easyicu.research_agent.reporting.manuscript_provenance import strip_numeric_provenance
from easyicu.research_agent.research_context.outbound import SUPPRESSED_COUNT, project_outbound_counts

from .scientific_review_items import ReviewItems, ReviewSource, digest, mapping, page, pointer, rows, token
from .numeric_evidence_audit import audit_numeric_evidence


def _leaves(value: Any, path: tuple = (), depth: int = 0):
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (*path, key), depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _leaves(child, (*path, index), depth + 1)
    elif value == SUPPRESSED_COUNT or _plausibility_number(value) is not None:
        yield path, value


def _number(items, value, field, source, *, surface, bound, occurrences=1, reason=""):
    shown = project_outbound_counts(value, key=field)
    status = "unbound" if not bound else "suppressed" if shown == SUPPRESSED_COUNT else "bound"
    return items.item("SR_NUMBER_" + status.upper(), shown, source, surface=surface,
                      status=status, binding_status="bound" if bound else "unbound",
                      field=field[:160], occurrences=occurrences, suppressed=shown == SUPPRESSED_COUNT, reason_code=reason)


def _answer(payloads, items):
    tables = mapping(payloads.get("result_tables.json"))
    performance = mapping(tables.get("prediction_performance"))
    typed_performance = performance.get("schema_version") == "easyicu.web-prediction-performance/1"
    output = []
    for index, raw in enumerate(rows(tables.get("tables"))):
        table = mapping(raw)
        if table.get("is_source_copy") is True:
            continue
        if typed_performance and table.get("evidence_id") == performance.get("evidence_id"):
            continue
        evidence_id, step_id = token(table.get("evidence_id")), token(table.get("produced_by_step"))
        verified = table.get("source_status") == "current" and bool(evidence_id and digest(table.get("sha256")))
        summaries = [key for key in ("descriptive_summary", "stratified_summary") if table.get(key)]
        if summaries:
            values = [(pointer("tables", index, key, *path), str(path[-1]), value)
                      for key in summaries for path, value in _leaves(table[key])]
        else:
            headers = rows(table.get("headers"))
            values = [(pointer("tables", index, "rows", ri, ci), str(headers[ci]), value)
                      for ri, row in enumerate(rows(table.get("rows")))
                      for ci, value in enumerate(rows(row)) if ci < len(headers)
                      and (value == SUPPRESSED_COUNT or _plausibility_number(value) is not None)]
        for location, field, value in values:
            output.append(_number(items, value, field, ReviewSource(
                "result_tables.json", location, evidence_id, step_id, table.get("sha256", "")),
                surface="answer", bound=verified, reason="" if verified else "source_binding_unavailable"))
    if typed_performance:
        for path, value in _leaves(dict(performance)):
            output.append(_number(items, value, str(path[-1]), ReviewSource(
                "result_tables.json", pointer("prediction_performance", *path),
                performance.get("evidence_id", ""), performance.get("step_id", "")),
                surface="answer", bound=bool(token(performance.get("evidence_id")))))
    # The append answer also displays this copied block. It has no value-level
    # evidence coordinates yet; equality with a table is not a binding.
    for path, value in _leaves(mapping(tables.get("append")).get("strata")):
        output.append(_number(items, value, str(path[-1]), ReviewSource(
            "result_tables.json", pointer("append", "strata", *path)), surface="answer",
            bound=False, reason="answer_source_coordinate_missing"))
    return output


def _manuscript(payloads, items):
    reader = mapping(payloads.get("manuscript_provenance.json"))
    draft = mapping(payloads.get("manuscript_draft.json"))
    text = draft.get("markdown_preview")
    preview = mapping(mapping(mapping(payloads.get("source_run_manifest.json")).get("scientific_review")).get("manuscript_preview"))
    preview_bound = bool(isinstance(text, str) and digest(reader.get("manuscript_sha256"))
                         and preview.get("manuscript_sha256") == reader.get("manuscript_sha256")
                         and preview.get("sha256") == hashlib.sha256(text.encode()).hexdigest())
    if isinstance(text, str) and text and reader and (
        reader.get("manuscript_sha256") != hashlib.sha256(text.encode()).hexdigest()
    ) and not preview_bound:
        return _unrecorded_draft(text, items), "unavailable"
    output = []
    for index, raw in enumerate(rows(reader.get("claims"))):
        claim = mapping(raw)
        evidence = mapping(claim.get("evidence"))
        count = claim.get("occurrence_count", 1)
        if type(count) is not int or count < 1:
            count = 1
        bound = claim.get("status") == "current" and bool(token(evidence.get("evidence_id")) and digest(evidence.get("sha256")))
        output.append(_number(items, claim.get("display_value"), str(claim.get("source_field") or ""), ReviewSource(
            "manuscript_provenance.json", pointer("claims", index), evidence.get("evidence_id", ""),
            claim.get("step_id", ""), evidence.get("sha256", ""), claim.get("source_json_pointer", "")), surface="manuscript", bound=bound,
            occurrences=count, reason="" if bound else "source_binding_unavailable"))
    receipt = mapping(reader.get("number_binding"))
    for index, value in enumerate(rows(receipt.get("untraced"))):
        output.append(_number(items, value, "", ReviewSource("manuscript_provenance.json", pointer("number_binding", "untraced", index)),
                              surface="manuscript", bound=False, reason="writer_numeric_binding_failed"))
    if receipt.get("status") == "recorded":
        return output, "recorded"
    # A claim list alone does not prove the absence of unbound numbers.
    audit = mapping(mapping(payloads.get("evidence_ledger.json")).get("numeric_evidence_audit"))
    if reader or not audit:
        if not reader and isinstance(text, str):
            output.extend(_unrecorded_draft(text, items))
        return output, "unavailable" if draft else "not_requested"
    for group in ("matches", "failures"):
        for index, raw in enumerate(rows(audit.get(group))):
            row = mapping(raw)
            values = [row["number"]] if "number" in row else rows(row.get("numbers"))
            for value in values:
                output.append(_number(items, value, str(row.get("evidence_path") or ""), ReviewSource(
                    "evidence_ledger.json", pointer("numeric_evidence_audit", group, index), row.get("evidence_id", "")),
                    surface="manuscript", bound=group == "matches", reason=token(row.get("reason"))))
    complete = len(rows(audit.get("matches"))) == audit.get("match_count") and len(rows(audit.get("failures"))) == audit.get("failure_count")
    return output, "recorded" if complete else "partial"


def _unrecorded_draft(text, items):
    """The existing native audit owns prose only when no current binding exists.

    No evidence is supplied: coincidentally equal table values cannot repair a
    missing Writer receipt. This exposes unknown numbers without inventing a
    new semantic matcher or claiming that a legacy draft passed the binder.
    """
    audit = audit_numeric_evidence({"manuscript_draft.json": {"sentences": [
        {"sentence_id": "unrecorded_draft", "text": strip_numeric_provenance(text), "evidence_ids": []},
    ]}})
    return [_number(items, value, "n", ReviewSource("manuscript_draft.json", "/markdown_preview"),
                    surface="manuscript", bound=False, reason="writer_binding_unavailable")
            for failure in audit["failures"] for value in rows(failure.get("numbers"))]


def number_check(payloads, items: ReviewItems, *, offset: int, limit: int) -> dict:
    answer = _answer(payloads, items)
    manuscript, coverage = _manuscript(payloads, items)
    all_items = [*answer, *manuscript]
    counts = {state: sum(row["occurrences"] for row in all_items if row["binding_status"] == state)
              for state in ("bound", "unbound")}
    counts["suppressed"] = sum(row["occurrences"] for row in all_items if row["suppressed"])
    defects = any(row["binding_status"] == "unbound" for row in all_items)
    all_items.sort(key=lambda row: row["binding_status"] != "unbound")
    return {**page(all_items, offset=offset, limit=limit), "counts": counts,
            "status": "defect" if defects else "incomplete" if coverage in {"unavailable", "partial"} else "recorded",
            "defect_codes": ["SR_NUMBER_UNBOUND"] if defects else [],
            "coverage": {"answer": "recorded_inputs" if answer else "unavailable", "manuscript": coverage},
            "suppressed_is_subset": True,
            "count_basis": "answer_input_cells_and_manuscript_occurrences"}
