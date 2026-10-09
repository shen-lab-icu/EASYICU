"""Host result facts recorded for one manuscript, and the claims they carry.

Owner
-----
This module owns the record of the host fact sentences a bound manuscript
was written with, and the one rule by which such a sentence stands for a
scientific claim in the Results.  The fact names the claim, cites the
claim's evidence at its current digest, reads verbatim in the Results
section the claim check reads, and shows every number the claim states
(:mod:`..authority.reported_numbers`).

The write phase records the facts beside the manuscript, bound to the
manuscript's exact bytes, and applies the rule when it binds; the gate
re-reads that record and applies it again.  A record that is absent,
unverified, unreadable, of another schema or bound to other bytes carries
nothing, so the claim check falls back to each claim's own sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..authority.manuscript_claim_policy import fact_sentence_in_results
from ..authority.reported_numbers import bind_reported_numbers, sentence_numbers
from ..authority.runtime_artifacts import verified_run_evidence_path
from ..authority.scientific_claims import ScientificClaim

RESULT_FACTS_EVIDENCE_ID = "manuscript_result_facts_json"
RESULT_FACTS_SCHEMA = "easyicu.manuscript_result_facts/1"
_RESULT_FACTS_RECORD_RE = re.compile(
    re.escape(RESULT_FACTS_EVIDENCE_ID) + r"(?:_v\d+)?"
)


@dataclass(frozen=True)
class ResultFactCarriage:
    """The claims recorded facts carry (claim ref to fact index), and why
    every other fact that names a claim carries none."""

    carried: Mapping[str, int] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)


def manuscript_sha256(manuscript: str) -> str:
    return sha256(manuscript.encode("utf-8")).hexdigest()


def result_fact_rows(facts: Sequence[Any]) -> list[dict[str, Any]]:
    """Host facts as the record states them and the carriage rule reads them."""

    return [
        {
            "subsection": str(fact.subsection),
            "text": str(fact.text),
            "evidence_id": str(fact.evidence_id),
            "source_sha256": str(fact.source_sha256),
            "source_fields": list(fact.source_fields),
            "replaces_claim_ref": fact.replaces_claim_ref,
            "required_result_sections": list(fact.required_result_sections),
        }
        for fact in facts
    ]


def result_facts_payload(
    facts: Sequence[Any], *, manuscript_sha256: str
) -> dict[str, Any]:
    """The record of ``facts`` for the manuscript with this digest."""

    return {
        "schema_version": RESULT_FACTS_SCHEMA,
        "source_manuscript_sha256": manuscript_sha256,
        "facts": result_fact_rows(facts),
    }


def record_result_facts(
    manuscript: str, facts: Sequence[Any], *, run_dir: Path, evidence: Any
) -> None:
    """Record the host facts that carry claims, bound to these manuscript bytes.

    The claims gate reads this record when it judges the final manuscript; a
    manuscript with other bytes needs its own.
    """

    if not any(fact.replaces_claim_ref for fact in facts):
        return
    digest = manuscript_sha256(manuscript)
    path = Path(run_dir) / "manuscript_result_facts.json"
    path.write_text(
        json.dumps(
            result_facts_payload(facts, manuscript_sha256=digest),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    evidence.register_file(
        kind="log",
        description="Host result facts the bound manuscript states, bound to its bytes.",
        source_path=path,
        evidence_id=RESULT_FACTS_EVIDENCE_ID,
        producer="pipeline",
        generation_mode="system",
        metadata={"source_manuscript_sha256": digest},
        on_sha_change="new_id",
    )


def _decision(
    manuscript: str,
    fact: Mapping[str, Any],
    claim: ScientificClaim | None,
    evidence: Any,
) -> dict[str, Any]:
    if claim is None:
        return {"status": "claim_absent"}
    if fact.get("evidence_id") != claim.evidence_id:
        return {"status": "fact_evidence_mismatch"}
    source = evidence.get(claim.evidence_id)
    if source is None or fact.get("source_sha256") != source.sha256:
        return {"status": "fact_source_stale"}
    text = fact.get("text")
    if not isinstance(text, str) or not fact_sentence_in_results(manuscript, text):
        return {"status": "fact_not_in_results"}
    stated = claim.reported_numbers()
    if stated is None:
        return {"status": "claim_numbers_untyped"}
    binding = bind_reported_numbers(stated, sentence_numbers(text))
    if not binding.complete:
        return {
            "status": "fact_numbers_unbound",
            "numbers": [
                {"value": str(number.value), "unit": number.unit, "reason": reason}
                for number, reason in binding.unbound
            ],
        }
    return {"status": "carried"}


def result_fact_carriage(
    manuscript: str,
    *,
    facts: Sequence[Mapping[str, Any]],
    claims: Sequence[ScientificClaim],
    evidence: Any,
) -> ResultFactCarriage:
    """Which ``claims`` the fact sentences in ``manuscript`` carry."""

    claims_by_ref = {claim.claim_ref: claim for claim in claims}
    carried: dict[str, int] = {}
    decisions = []
    for index, fact in enumerate(facts):
        claim_ref = fact.get("replaces_claim_ref")
        if not isinstance(claim_ref, str) or not claim_ref:
            continue
        decision = (
            {"status": "claim_already_carried"}
            if claim_ref in carried
            else _decision(manuscript, fact, claims_by_ref.get(claim_ref), evidence)
        )
        if decision["status"] == "carried":
            carried[claim_ref] = index
        decisions.append({"fact_index": index, "claim_ref": claim_ref, **decision})
    return ResultFactCarriage(carried=carried, trace={"facts": decisions})


def recorded_result_fact_carriage(
    *,
    run_dir: Path,
    manuscript_text: str,
    evidence: Any,
    claims: Sequence[ScientificClaim],
) -> ResultFactCarriage:
    """Apply the carriage rule to the facts recorded for these exact bytes."""

    digest = manuscript_sha256(manuscript_text)
    records = [
        record
        for record in (
            evidence.records() if callable(getattr(evidence, "records", None)) else ()
        )
        if _RESULT_FACTS_RECORD_RE.fullmatch(record.evidence_id)
        and record.producer == "pipeline"
        and record.generation_mode == "system"
    ]
    current = [
        record
        for record in records
        if (record.metadata or {}).get("source_manuscript_sha256") == digest
    ]
    if not current:
        status = (
            "facts_record_for_other_manuscript" if records else "facts_record_absent"
        )
        return ResultFactCarriage(trace={"record": status})
    record = current[-1]

    def refused(status: str) -> ResultFactCarriage:
        return ResultFactCarriage(
            trace={"record": status, "evidence_id": record.evidence_id}
        )

    path = verified_run_evidence_path(run_dir, record)
    if path is None:
        return refused("facts_record_unverified")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return refused("facts_record_unreadable")
    facts = payload.get("facts") if isinstance(payload, dict) else None
    if (
        not isinstance(facts, list)
        or payload.get("schema_version") != RESULT_FACTS_SCHEMA
        or not all(isinstance(fact, dict) for fact in facts)
    ):
        return refused("facts_record_unknown_schema")
    if payload.get("source_manuscript_sha256") != digest:
        return refused("facts_record_for_other_manuscript")
    carriage = result_fact_carriage(
        manuscript_text, facts=facts, claims=claims, evidence=evidence
    )
    return ResultFactCarriage(
        carried=carriage.carried,
        trace={
            "record": "facts_record_read",
            "evidence_id": record.evidence_id,
            **carriage.trace,
        },
    )


__all__ = [
    "RESULT_FACTS_EVIDENCE_ID",
    "RESULT_FACTS_SCHEMA",
    "ResultFactCarriage",
    "manuscript_sha256",
    "record_result_facts",
    "recorded_result_fact_carriage",
    "result_fact_carriage",
    "result_fact_rows",
    "result_facts_payload",
]
