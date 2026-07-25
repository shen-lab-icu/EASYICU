"""Bounded LLM decisions for manuscript sentences rejected by STRICT evidence.

This module owns a deliberately narrow post-draft protocol. The model cannot
rewrite a claim: it may only select registered evidence ids for the exact
sentence or ask the host to drop it. The host applies the decision and runs the
unchanged strict evidence gate again.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from ..providers.protocol import LLMClient, LLMMessage
from ..providers.structured_retry import call_llm_with_structured_retry


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
    language: str = "en",
) -> List[Dict[str, object]]:
    """Return validated cite/drop decisions for every rejected sentence."""

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
                "paraphrase, merge, or add a scientific claim. Choose only between "
                "citing registered evidence for the exact original sentence and "
                "dropping it."
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
                "- `drop`: when support is absent, indirect, ambiguous, or the "
                "sentence merely narrates that a result was unavailable.\n"
                "Return one JSON object and no prose:\n"
                '{"decisions":[{"index":0,"action":"cite",'
                '"evidence_ids":["registered_id"]},{"index":1,'
                '"action":"drop","evidence_ids":[]}]}\n'
                "Every input index must appear exactly once. Never invent an id.\n"
                f"{language_instruction}\n\n"
                "ALLOWED EVIDENCE IDS:\n"
                + json.dumps(allowed_ids, ensure_ascii=False)
                + "\n\nFLAGGED SENTENCES:\n"
                + json.dumps(
                    [
                        {"index": index, "sentence": sentence}
                        for index, sentence in enumerate(sentences)
                    ],
                    ensure_ascii=False,
                )
                + "\n\nMACHINE EVIDENCE DIGEST:\n"
                + (evidence_digest or "(none)")
            ),
        ),
    ]

    def _parse(raw: str) -> List[Dict[str, object]]:
        block = _first_json_object(raw)
        if block is None:
            raise ValueError("writer evidence repair returned no JSON object")
        payload = json.loads(block)
        raw_decisions = payload.get("decisions") if isinstance(payload, dict) else None
        if not isinstance(raw_decisions, list):
            raise ValueError("writer evidence repair requires a decisions list")
        normalized: List[Dict[str, object]] = []
        seen: set[int] = set()
        for item in raw_decisions:
            if not isinstance(item, dict):
                raise ValueError("each writer evidence decision must be an object")
            index = item.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= len(sentences)
                or index in seen
            ):
                raise ValueError(
                    "writer evidence decision indexes must be unique and in range"
                )
            action = str(item.get("action") or "").strip().lower()
            raw_ids = item.get("evidence_ids", [])
            if not isinstance(raw_ids, list):
                raise ValueError("writer evidence_ids must be a list")
            selected_ids = tuple(
                dict.fromkeys(
                    str(evidence_id).strip()
                    for evidence_id in raw_ids
                    if str(evidence_id).strip()
                )
            )
            if any(evidence_id not in allowed_set for evidence_id in selected_ids):
                raise ValueError(
                    "writer evidence repair selected an unregistered evidence id"
                )
            if action == "cite":
                if not 1 <= len(selected_ids) <= 3:
                    raise ValueError(
                        "cite decisions require 1-3 registered evidence ids"
                    )
            elif action == "drop":
                if selected_ids:
                    raise ValueError("drop decisions cannot include evidence ids")
            else:
                raise ValueError("writer evidence action must be cite or drop")
            seen.add(index)
            normalized.append(
                {
                    "index": index,
                    "action": action,
                    "evidence_ids": list(selected_ids),
                }
            )
        if seen != set(range(len(sentences))):
            raise ValueError(
                "writer evidence repair must decide every flagged sentence"
            )
        return sorted(normalized, key=lambda item: int(item["index"]))

    return call_llm_with_structured_retry(
        llm,
        messages,
        _parse,
        role="writer_evidence_citation_repair",
        max_retries=1,
        max_tokens=1024,
        temperature=0.0,
        format_reminder=(
            "The JSON object must contain only `decisions`; every decision "
            "requires index, action, and evidence_ids."
        ),
    )


__all__ = ["decide_writer_evidence_repairs"]
