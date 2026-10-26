"""Register and project only exact, digest-verified aggregate numeric leaves."""

from __future__ import annotations

import json
from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.canonical_json import sha256_file
from easyicu.research_agent.execution.runners.typed_input_binding import contained_regular_file

from .contracts import AnswerRow, AnswerSentence, EvidenceCoordinate, QuickAnswer, QuickStop
from .planning import card_digest


def record_answer(store, proposal, approval, cells):
    card = proposal.card
    receipt = store.register_json(
        kind="log", description="Approved quick-answer plan and source binding",
        payload={"card": card.model_dump(mode="json"), "approval": approval.model_dump(mode="json")},
        filename="quick_approval.json", producer="quick_answer", generation_mode="deterministic_host",
    )
    payload = {"source": card.source, "card_sha256": card.sha256, "rows": cells}
    record = store.register_json(
        kind="table", description="Deterministic quick-answer aggregate cells",
        payload=payload, filename="quick_result.json", produced_by_step="quick_answer",
        inputs=[receipt.evidence_id], producer="quick_answer", generation_mode="deterministic_host",
        metadata={"source_identity": card.source, "source_binding": card.plan.source_binding,
                  "source_data_sha256": card.plan.source_data_sha256,
                  "card_sha256": card.sha256, "aggregate_only": True},
    )
    store.register_step_summary_numerics(step_id="quick_answer", evidence_id=record.evidence_id,
                                       summary=payload, tolerance=0)
    return record


def project_quick_answer(*, run_dir: Path, proposal, evidence_id: str, sha256: str) -> QuickAnswer:
    """Run-answer host projection. IDs/digests come from the host's run receipt.

    No free-form answer text is accepted. Every sentence is rendered from one
    registered numeric coordinate, and every table value has the same binding.
    """
    store = EvidenceStore(run_dir)
    record = next((item for item in store.records() if item.evidence_id == evidence_id), None)
    card = proposal.card
    if (record is None or card is None or card_digest(card) != card.sha256 or record.sha256 != sha256
            or record.producer != "quick_answer" or record.produced_by_step != "quick_answer"
            or record.metadata.get("source_identity") != card.source
            or record.metadata.get("source_binding") != card.plan.source_binding
            or record.metadata.get("source_data_sha256") != card.plan.source_data_sha256
            or record.metadata.get("card_sha256") != card.sha256
            or record.metadata.get("aggregate_only") is not True):
        raise QuickStop("quick_evidence_binding_failed")
    path = contained_regular_file(Path(run_dir) / record.relative_path, Path(run_dir))
    if path is None or sha256_file(path) != record.sha256:
        raise QuickStop("quick_evidence_binding_failed")
    payload = json.loads(path.read_bytes())
    if payload.get("source") != card.source or payload.get("card_sha256") != card.sha256:
        raise QuickStop("quick_evidence_binding_failed")
    if len(record.inputs) != 1:
        raise QuickStop("quick_evidence_binding_failed")
    approval_record = store.get(record.inputs[0])
    approval_path = (contained_regular_file(Path(run_dir) / approval_record.relative_path, Path(run_dir))
                     if approval_record else None)
    if approval_path is None or sha256_file(approval_path) != approval_record.sha256:
        raise QuickStop("quick_evidence_binding_failed")
    approved = json.loads(approval_path.read_bytes())
    if (approved.get("card") != card.model_dump(mode="json")
            or approved.get("approval", {}).get("card_sha256") != card.sha256
            or card.capability.substitutions and approved.get("approval", {}).get("accept_substitutions") is not True):
        raise QuickStop("quick_evidence_binding_failed")
    claims = {(claim.evidence_id, claim.source_field): claim for claim in store.numeric_claims()
              if claim.step_id == "quick_answer"}
    rows, sentences = [], []
    for index, cell in enumerate(payload["rows"]):
        field = f"rows[{index}].value"
        claim = claims.get((evidence_id, field))
        if claim is None or claim.canonical != cell["value"] or claim.tolerance != 0:
            raise QuickStop("quick_evidence_binding_failed")
        coordinate = EvidenceCoordinate(evidence_id=evidence_id, sha256=sha256,
                                        source_field=field, source=card.source)
        rows.append(AnswerRow(**cell, evidence=coordinate))
        # Labels stay in the table (including numeric group labels and units).
        # This placeholder sentence has exactly one scientific number.
        sentences.append(AnswerSentence(text=f"The recorded value is {cell['value']}.",
                                        evidence=coordinate))
    return QuickAnswer(status="ok", reason="quick_answer_ready", source=card.source,
                       rows=tuple(rows), sentences=tuple(sentences), expand=proposal.expand)
