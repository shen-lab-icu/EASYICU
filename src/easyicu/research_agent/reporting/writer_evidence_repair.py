"""Bounded LLM decisions for manuscript sentences rejected by STRICT evidence.

This module owns a deliberately narrow post-draft protocol. The model cannot
rewrite a claim: it may only select registered evidence ids for the exact
sentence, replace a qualitative assertion with one exact host-issued claim, or
ask the host to drop it. The host applies the decision and runs the unchanged
strict evidence gate again.
"""

from __future__ import annotations

import json
from typing import List, Mapping, Optional, Sequence

from ..providers.protocol import LLMClient, LLMMessage
from ..providers.structured_retry import call_llm_with_structured_retry
from .writer_repair_decision import WriterRepairDecision


def _first_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object, ignoring braces in strings."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def decide_writer_evidence_repairs(
    llm: LLMClient,
    *,
    evidence_ids: Sequence[str],
    evidence_digest: Optional[str],
    missing_sentences: Sequence[str],
    scientific_claims: Optional[Mapping[str, str]] = None,
    claim_required_sentences: Sequence[str] = (),
    language: str = "en",
) -> List[WriterRepairDecision]:
    """Return validated cite/drop/host-claim decisions for rejected prose."""

    sentences = [str(sentence).strip() for sentence in missing_sentences]
    if not sentences:
        return []
    allowed_ids = tuple(
        dict.fromkeys(
            str(evidence_id).strip()
            for evidence_id in evidence_ids
            if str(evidence_id).strip()
        )
    )
    allowed_set = set(allowed_ids)
    if not allowed_ids:
        raise ValueError(
            "Writer evidence repair requires at least one registered evidence id"
        )
    claim_text_by_ref = {
        str(ref).strip(): " ".join(str(text or "").split())
        for ref, text in dict(scientific_claims or {}).items()
        if str(ref).strip() and " ".join(str(text or "").split())
    }
    claim_required = {
        str(sentence).strip()
        for sentence in claim_required_sentences
        if str(sentence).strip()
    }
    unknown_claim_required = claim_required - set(sentences)
    if unknown_claim_required:
        raise ValueError(
            "claim_required_sentences must be members of missing_sentences"
        )
    language_instruction = (
        "Keep the sentence language unchanged. Evidence ids remain ASCII."
        if str(language).lower().startswith(("zh", "cn", "chinese"))
        else "Keep the sentence language unchanged."
    )
    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are a manuscript evidence-repair classifier. Never rewrite, "
                "paraphrase, merge, or invent a scientific claim. Choose only among "
                "citing registered evidence for the exact original sentence, "
                "selecting one exact host-issued scientific claim, and dropping it."
            ),
        ),
        LLMMessage(
            role="user",
            content=(
                "STRICT EVIDENCE CITATION REPAIR.\n"
                "A deterministic gate rejected only the sentences listed below. "
                "For every sentence choose exactly one action:\n"
                "- `cite`: only when the machine evidence digest supports the "
                "entire original sentence; provide 1-3 ids from the allowed list.\n"
                "- `claim`: only for an item marked `requires_scientific_claim`; "
                "select exactly one ref from HOST-ISSUED SCIENTIFIC CLAIMS. The host "
                "will replace the whole original sentence; do not rewrite it.\n"
                "- `drop`: when support is absent, indirect, ambiguous, or the "
                "sentence merely narrates that a result was unavailable.\n"
                "Return one JSON object and no prose:\n"
                '{"decisions":[{"index":0,"action":"cite",'
                '"evidence_ids":["registered_id"]},{"index":1,'
                '"action":"claim","evidence_ids":[],"claim_ref":'
                '"step_id.claim_id"},{"index":2,"action":"drop",'
                '"evidence_ids":[]}]}\n'
                "Every input index must appear exactly once. Never invent an id.\n"
                "An item marked `requires_scientific_claim` may only use `claim` "
                "or `drop`; an ordinary evidence item may only use `cite` or `drop`.\n"
                f"{language_instruction}\n\n"
                "ALLOWED EVIDENCE IDS:\n"
                + json.dumps(allowed_ids, ensure_ascii=False)
                + "\n\nHOST-ISSUED SCIENTIFIC CLAIMS:\n"
                + json.dumps(claim_text_by_ref, ensure_ascii=False)
                + "\n\nFLAGGED SENTENCES:\n"
                + json.dumps(
                    [
                        {
                            "index": index,
                            "sentence": sentence,
                            "requires_scientific_claim": sentence
                            in claim_required,
                        }
                        for index, sentence in enumerate(sentences)
                    ],
                    ensure_ascii=False,
                )
                + "\n\nMACHINE EVIDENCE DIGEST:\n"
                + (evidence_digest or "(none)")
            ),
        ),
    ]

    def _parse(raw: str) -> List[WriterRepairDecision]:
        block = _first_json_object(raw)
        if block is None:
            raise ValueError("writer evidence repair returned no JSON object")
        payload = json.loads(block)
        raw_decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(raw_decisions, list):
            raise ValueError("writer evidence repair requires a decisions list")
        normalized: List[WriterRepairDecision] = []
        seen: set[int] = set()
        for item in raw_decisions:
            if not isinstance(item, dict):
                raise ValueError("each writer evidence decision must be an object")
            # Shape — index, action, and which fields each action may carry —
            # is the value type's invariant, checked in its constructor.
            decision = WriterRepairDecision.from_mapping(item)
            index = decision.index
            if index >= len(sentences) or index in seen:
                raise ValueError(
                    "writer evidence decision indexes must be unique and in range"
                )
            # What remains is what only this call knows: which ids are
            # registered for this run, and which sentences the host marked as
            # requiring a scientific claim.
            if any(
                evidence_id not in allowed_set
                for evidence_id in decision.evidence_ids
            ):
                raise ValueError(
                    "writer evidence repair selected an unregistered evidence id"
                )
            if decision.action == "cite" and sentences[index] in claim_required:
                raise ValueError(
                    "scientific-claim sentences cannot borrow evidence citations"
                )
            if decision.action == "claim":
                if sentences[index] not in claim_required:
                    raise ValueError(
                        "claim decisions are limited to scientific-claim sentences"
                    )
                if decision.claim_ref not in claim_text_by_ref:
                    raise ValueError(
                        "claim decisions require one registered host claim_ref"
                    )
            seen.add(index)
            normalized.append(decision)
        if seen != set(range(len(sentences))):
            raise ValueError(
                "writer evidence repair must decide every flagged sentence"
            )
        return sorted(normalized, key=lambda decision: decision.index)

    return call_llm_with_structured_retry(
        llm,
        messages,
        _parse,
        role="writer_evidence_citation_repair",
        max_retries=1,
        # A strict draft can flag the same unsupported assertion in Abstract,
        # Results, Discussion, and Conclusion.  Keep one bounded provider call,
        # but size its JSON budget to the number of decisions so the response
        # cannot truncate merely because every occurrence is adjudicated.
        max_tokens=max(1024, min(4096, 256 + len(sentences) * 48)),
        temperature=0.0,
        format_reminder=(
            "The JSON object must contain only `decisions`; every decision "
            "requires index, action, and evidence_ids; claim actions also require "
            "one exact claim_ref."
        ),
    )


__all__ = ["WriterRepairDecision", "decide_writer_evidence_repairs"]
