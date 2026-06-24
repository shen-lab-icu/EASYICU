"""Numeric evidence binding audit for native FastAPI agent artifacts.

The existing strict evidence gate checks that claims cite known artifacts. This
module checks the next safety layer: numeric claims must match concrete numeric
values inside the cited artifacts, within an explicit rounding tolerance.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Sequence

_AUDITABLE_DRAFT_KEYS = ("claims", "sentences")
_EXCLUDED_EVIDENCE = {"agent_plan.json", "manuscript_draft.json"}
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z_])(?P<sign>[+-]?)(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?:\s*(?P<unit>%|percent|percentage|pct))?",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_STOPWORDS = {
    "a",
    "across",
    "active",
    "all",
    "and",
    "are",
    "as",
    "at",
    "before",
    "bound",
    "contains",
    "draft",
    "export",
    "in",
    "is",
    "local",
    "of",
    "only",
    "snapshot",
    "the",
    "this",
    "to",
    "until",
    "was",
    "were",
    "with",
}
_PERCENT_TOKENS = {"pct", "percent", "percentage", "rate", "mortality", "female", "sepsis", "coverage"}
_COUNT_TOKENS = {"count", "counts", "n", "row", "rows", "stay", "stays", "module", "modules", "entities", "entity"}
_SYNONYMS = {
    "deceased": {"death", "mortality"},
    "death": {"deceased", "mortality"},
    "entity": {"entities", "stay", "stays", "cohort"},
    "entities": {"entity", "stay", "stays", "cohort"},
    "female": {"sex", "gender"},
    "los": {"length", "stay"},
    "mortality": {"death", "deceased"},
    "module": {"modules"},
    "modules": {"module"},
    "record": {"records", "row", "rows"},
    "records": {"record", "row", "rows"},
    "row": {"rows", "record", "records"},
    "rows": {"row", "record", "records"},
    "sepsis": {"sepsis3", "event"},
    "sepsis3": {"sepsis", "event"},
    "sofa": {"sofa2", "score"},
    "sofa2": {"sofa", "score"},
    "stay": {"stays", "entity", "entities", "cohort"},
    "stays": {"stay", "entity", "entities", "cohort"},
}


def audit_numeric_evidence(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate numeric claims in ``manuscript_draft.json`` against artifacts."""
    draft = artifacts.get("manuscript_draft.json") if isinstance(artifacts, dict) else {}
    if not isinstance(draft, dict):
        draft = {}
    facts = _collect_numeric_facts(artifacts)
    failures: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    numeric_claim_count = 0
    numeric_sentence_count = 0
    mention_count = 0
    claim_failures = 0
    sentence_failures = 0

    for row_type, rows in _draft_rows(draft):
        for row in rows:
            text = str(row.get("text") or "")
            mentions = _extract_numeric_mentions(text)
            if not mentions:
                continue
            if row_type == "claim":
                numeric_claim_count += 1
            else:
                numeric_sentence_count += 1
            owner = _row_owner(row, row_type)
            evidence_ids = [str(item) for item in row.get("evidence_ids") or [] if str(item)]
            if not evidence_ids:
                failure = {
                    "owner": owner,
                    "row_type": row_type,
                    "reason": "numeric_claim_missing_evidence_id",
                    "numbers": [m["raw"] for m in mentions],
                }
                failures.append(failure)
                if row_type == "claim":
                    claim_failures += 1
                else:
                    sentence_failures += 1
                mention_count += len(mentions)
                continue

            row_failed = False
            for evidence_id in evidence_ids:
                if evidence_id not in artifacts:
                    failures.append({
                        "owner": owner,
                        "row_type": row_type,
                        "reason": "missing_evidence",
                        "evidence_id": evidence_id,
                    })
                    row_failed = True
                elif evidence_id in _EXCLUDED_EVIDENCE:
                    failures.append({
                        "owner": owner,
                        "row_type": row_type,
                        "reason": "artifact_not_numeric_evidence_source",
                        "evidence_id": evidence_id,
                    })
                    row_failed = True
            evidence_facts = [fact for fact in facts if fact["artifact"] in evidence_ids]
            context_tokens = _context_tokens(text)
            for mention in mentions:
                mention_count += 1
                match = _match_mention(mention, evidence_facts, context_tokens)
                if match:
                    matches.append({
                        "owner": owner,
                        "row_type": row_type,
                        "number": mention["raw"],
                        "evidence_id": match["artifact"],
                        "evidence_path": match["path"],
                        "evidence_value": match["value"],
                        "tolerance": match["tolerance"],
                    })
                else:
                    failures.append({
                        "owner": owner,
                        "row_type": row_type,
                        "reason": "numeric_value_not_bound",
                        "number": mention["raw"],
                        "evidence_ids": evidence_ids,
                        "context_tokens": sorted(context_tokens)[:12],
                    })
                    row_failed = True
            if row_failed:
                if row_type == "claim":
                    claim_failures += 1
                else:
                    sentence_failures += 1

    passed = not failures
    return {
        "mode": "numeric_evidence",
        "passed": passed,
        "claims_passed": claim_failures == 0,
        "sentences_passed": sentence_failures == 0,
        "numeric_claim_count": numeric_claim_count,
        "numeric_sentence_count": numeric_sentence_count,
        "numeric_mention_count": mention_count,
        "match_count": len(matches),
        "failure_count": len(failures),
        "failures": failures[:50],
        "matches": matches[:50],
        "tolerance_policy": {
            "integer_count_exact": True,
            "rounded_decimal_absolute": "0.5 * 10^-displayed_decimals",
            "rounded_integer_for_decimal_facts_absolute": 0.5,
            "percentage_fraction_equivalence": True,
        },
    }


def _draft_rows(draft: Dict[str, Any]) -> Iterable[tuple[str, List[Dict[str, Any]]]]:
    for key in _AUDITABLE_DRAFT_KEYS:
        rows = draft.get(key)
        if not isinstance(rows, list):
            rows = []
        row_type = "claim" if key == "claims" else "sentence"
        yield row_type, [row for row in rows if isinstance(row, dict)]


def _row_owner(row: Dict[str, Any], row_type: str) -> str:
    if row_type == "claim":
        return str(row.get("claim_id") or row.get("id") or row.get("text") or "claim")
    return str(row.get("sentence_id") or row.get("id") or row.get("text") or "sentence")


def _extract_numeric_mentions(text: str) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(text or ""):
        start, end = match.span()
        if _is_embedded_label_number(text, start, end):
            continue
        sign = match.group("sign") or ""
        raw_number = match.group("number")
        if sign == "-" and start > 0 and text[start - 1].isdigit():
            sign = ""
        raw = (sign + raw_number + (match.group("unit") or "")).strip()
        value = float((sign + raw_number).replace(",", ""))
        number_text = raw_number.replace(",", "")
        decimals = len(number_text.split(".", 1)[1]) if "." in number_text else 0
        unit = (match.group("unit") or "").lower()
        mentions.append({
            "raw": raw,
            "value": value,
            "decimals": decimals,
            "unit": "percent" if unit in {"%", "percent", "percentage", "pct"} else "number",
        })
    return mentions


def _is_embedded_label_number(text: str, start: int, end: int) -> bool:
    prev_char = text[start - 1] if start > 0 else ""
    next_char = text[end] if end < len(text) else ""
    if prev_char.isalpha() or prev_char == "_":
        return True
    if next_char.isalpha() or next_char == "_":
        return True
    if prev_char == "-" and start > 1 and text[start - 2].isalpha():
        return True
    return False


def _collect_numeric_facts(artifacts: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for artifact, payload in (artifacts or {}).items():
        if artifact in _EXCLUDED_EVIDENCE:
            continue
        facts.extend(_walk_numeric_facts(payload, artifact, artifact))
    return facts


def _walk_numeric_facts(value: Any, artifact: str, path: str) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    if isinstance(value, bool):
        return facts
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        tokens = _context_tokens(path)
        facts.append({
            "artifact": artifact,
            "path": path,
            "value": float(value),
            "tokens": tokens,
            "unit": _fact_unit(tokens),
        })
    elif isinstance(value, dict):
        for key, child in value.items():
            facts.extend(_walk_numeric_facts(child, artifact, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            facts.extend(_walk_numeric_facts(child, artifact, f"{path}[{index}]"))
    return facts


def _context_tokens(text: str) -> set[str]:
    tokens = set()
    for match in _WORD_RE.finditer(text or ""):
        tokens.update(_token_parts(match.group(0)))
    tokens = {token for token in tokens if token and token not in _STOPWORDS and token != "json"}
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(_SYNONYMS.get(token, set()))
    return expanded


def _token_parts(token: str) -> set[str]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", str(token or "")) if part]
    return {_normalize_token(part) for part in parts}


def _normalize_token(token: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "", str(token or "").strip().lower())
    if len(out) > 3 and out.endswith("s"):
        out = out[:-1]
    return out


def _fact_unit(tokens: Sequence[str] | set[str]) -> str:
    token_set = set(tokens)
    if token_set & _PERCENT_TOKENS:
        return "percent"
    if token_set & _COUNT_TOKENS:
        return "count"
    return "number"


def _match_mention(
    mention: Dict[str, Any],
    facts: List[Dict[str, Any]],
    context_tokens: set[str],
) -> Dict[str, Any] | None:
    value_candidates = []
    context_candidates = []
    for fact in facts:
        tolerance, matched_value = _numeric_tolerance(mention, fact)
        if abs(float(mention["value"]) - matched_value) > tolerance:
            continue
        value_candidates.append(fact)
        if not _unit_compatible(mention, fact):
            continue
        if not (context_tokens & set(fact.get("tokens") or [])):
            continue
        context_candidates.append({**fact, "tolerance": tolerance})
    if context_candidates:
        return sorted(context_candidates, key=lambda item: (item["artifact"], item["path"]))[0]
    return None


def _unit_compatible(mention: Dict[str, Any], fact: Dict[str, Any]) -> bool:
    mention_unit = mention.get("unit")
    fact_unit = fact.get("unit")
    if mention_unit == "percent":
        return fact_unit in {"percent", "number"}
    if fact_unit == "percent":
        return False
    return True


def _numeric_tolerance(mention: Dict[str, Any], fact: Dict[str, Any]) -> tuple[float, float]:
    fact_value = float(fact["value"])
    if mention.get("unit") == "percent" and fact.get("unit") != "percent" and 0 <= fact_value <= 1:
        fact_value = fact_value * 100
    decimals = int(mention.get("decimals") or 0)
    if decimals > 0:
        tolerance = 0.5 * (10 ** -decimals) + 1e-9
    elif float(fact["value"]).is_integer():
        tolerance = 0.0
    else:
        tolerance = 0.5
    return tolerance, fact_value
