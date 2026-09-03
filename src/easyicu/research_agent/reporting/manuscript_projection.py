"""Typed, owner-issued projection of reportable results into manuscript prose.

Execution owners may attach one ``easyicu.manuscript_projection/1`` contract to
any ``reportable_*_results`` mapping.  This module resolves only declared text
and numeric paths.  It does not infer an estimand, choose a result, or calculate
a new statistic; the unchanged evidence and numeric binders remain the final
authority gates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


_SCHEMA_VERSION = "easyicu.manuscript_projection/1"
_REPORTABLE_KEY_RE = re.compile(r"^reportable_[a-z0-9_]+_results$")
_CLAIM_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_FORMAT_SPEC_RE = re.compile(r"^\.(?:[0-9]|1[0-2])[feg]$")
_PATH_TOKEN_RE = re.compile(r"(?:^|\.)([A-Za-z][A-Za-z0-9_]*)|\[(\d+)\]")


class ManuscriptProjectionError(ValueError):
    """A deterministic owner emitted an invalid manuscript projection."""


@dataclass(frozen=True)
class _Target:
    kind: str
    label: str


@dataclass(frozen=True)
class _Fragment:
    text: str | None = None
    numeric_path: str | None = None
    format_spec: str | None = None


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    targets: Tuple[_Target, ...]
    fragments: Tuple[_Fragment, ...]


def _strict_keys(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    coordinate: str,
) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ManuscriptProjectionError(
            f"{coordinate} contains unsupported field(s): {extra}"
        )


def _parse_target(payload: Any, *, coordinate: str) -> _Target:
    if not isinstance(payload, Mapping):
        raise ManuscriptProjectionError(f"{coordinate} must be a mapping")
    _strict_keys(
        payload,
        allowed=frozenset({"kind", "label"}),
        coordinate=coordinate,
    )
    kind = str(payload.get("kind") or "").strip()
    label = str(payload.get("label") or "").strip()
    if kind not in {"abstract_label", "markdown_heading"}:
        raise ManuscriptProjectionError(
            f"{coordinate}.kind must be abstract_label or markdown_heading"
        )
    if not label or len(label) > 120 or "\n" in label:
        raise ManuscriptProjectionError(f"{coordinate}.label is invalid")
    return _Target(kind=kind, label=label)


def _parse_fragment(payload: Any, *, coordinate: str) -> _Fragment:
    if not isinstance(payload, Mapping):
        raise ManuscriptProjectionError(f"{coordinate} must be a mapping")
    _strict_keys(
        payload,
        allowed=frozenset({"text", "numeric_path", "format_spec"}),
        coordinate=coordinate,
    )
    text = payload.get("text")
    numeric_path = payload.get("numeric_path")
    format_spec = payload.get("format_spec")
    if (text is None) == (numeric_path is None):
        raise ManuscriptProjectionError(
            f"{coordinate} must declare exactly one of text or numeric_path"
        )
    if text is not None:
        rendered = str(text)
        if not rendered or len(rendered) > 1000:
            raise ManuscriptProjectionError(f"{coordinate}.text is invalid")
        if format_spec is not None:
            raise ManuscriptProjectionError(
                f"{coordinate}.format_spec requires numeric_path"
            )
        return _Fragment(text=rendered)
    path = str(numeric_path or "").strip()
    spec = str(format_spec or ".6g").strip()
    if not path or not _FORMAT_SPEC_RE.fullmatch(spec):
        raise ManuscriptProjectionError(
            f"{coordinate} has invalid numeric_path or format_spec"
        )
    return _Fragment(numeric_path=path, format_spec=spec)


def _parse_contract(payload: Any, *, coordinate: str) -> Tuple[_Claim, ...]:
    if not isinstance(payload, Mapping):
        raise ManuscriptProjectionError(f"{coordinate} must be a mapping")
    _strict_keys(
        payload,
        allowed=frozenset({"schema_version", "claims"}),
        coordinate=coordinate,
    )
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ManuscriptProjectionError(f"{coordinate} has unsupported schema_version")
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ManuscriptProjectionError(f"{coordinate}.claims must be non-empty")
    claims: List[_Claim] = []
    seen: set[str] = set()
    for index, raw_claim in enumerate(raw_claims):
        claim_coordinate = f"{coordinate}.claims[{index}]"
        if not isinstance(raw_claim, Mapping):
            raise ManuscriptProjectionError(f"{claim_coordinate} must be a mapping")
        _strict_keys(
            raw_claim,
            allowed=frozenset({"claim_id", "targets", "fragments"}),
            coordinate=claim_coordinate,
        )
        claim_id = str(raw_claim.get("claim_id") or "").strip()
        if not _CLAIM_ID_RE.fullmatch(claim_id) or claim_id in seen:
            raise ManuscriptProjectionError(
                f"{claim_coordinate}.claim_id is invalid or duplicated"
            )
        seen.add(claim_id)
        raw_targets = raw_claim.get("targets")
        raw_fragments = raw_claim.get("fragments")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ManuscriptProjectionError(
                f"{claim_coordinate}.targets must be non-empty"
            )
        if not isinstance(raw_fragments, list) or not raw_fragments:
            raise ManuscriptProjectionError(
                f"{claim_coordinate}.fragments must be non-empty"
            )
        targets = tuple(
            _parse_target(item, coordinate=f"{claim_coordinate}.targets[{i}]")
            for i, item in enumerate(raw_targets)
        )
        fragments = tuple(
            _parse_fragment(item, coordinate=f"{claim_coordinate}.fragments[{i}]")
            for i, item in enumerate(raw_fragments)
        )
        if not any(fragment.numeric_path for fragment in fragments):
            raise ManuscriptProjectionError(
                f"{claim_coordinate} must contain a numeric fragment"
            )
        claims.append(_Claim(claim_id=claim_id, targets=targets, fragments=fragments))
    return tuple(claims)


def _resolve_numeric_path(root: Mapping[str, Any], path: str) -> float:
    tokens: List[str | int] = []
    rendered = ""
    for match in _PATH_TOKEN_RE.finditer(path):
        raw_index = match.group(2)
        token: str | int = int(raw_index) if raw_index is not None else match.group(1)
        tokens.append(token)
        rendered += (
            f"[{token}]"
            if isinstance(token, int)
            else (str(token) if not rendered else f".{token}")
        )
    if rendered != path or not tokens:
        raise ManuscriptProjectionError(f"invalid numeric path: {path}")
    current: Any = root
    for token in tokens:
        if isinstance(token, str):
            if not isinstance(current, Mapping) or token not in current:
                raise ManuscriptProjectionError(f"missing numeric path: {path}")
            current = current[token]
        else:
            if not isinstance(current, (list, tuple)) or token >= len(current):
                raise ManuscriptProjectionError(f"missing numeric path: {path}")
            current = current[token]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ManuscriptProjectionError(f"non-numeric projection path: {path}")
    value = float(current)
    if not math.isfinite(value):
        raise ManuscriptProjectionError(f"non-finite projection path: {path}")
    return value


def _render_claim(
    claim: _Claim, *, reporting: Mapping[str, Any]
) -> tuple[str, tuple[str, ...]]:
    parts: List[str] = []
    literals: List[str] = []
    for fragment in claim.fragments:
        if fragment.text is not None:
            parts.append(fragment.text)
            continue
        assert fragment.numeric_path is not None
        assert fragment.format_spec is not None
        literal = format(
            _resolve_numeric_path(reporting, fragment.numeric_path),
            fragment.format_spec,
        )
        parts.append(literal)
        literals.append(literal)
    sentence = "".join(parts).strip()
    if not sentence:
        raise ManuscriptProjectionError(f"claim {claim.claim_id} rendered empty text")
    return sentence, tuple(literals)


def _target_body_span(text: str, target: _Target) -> tuple[int, int] | None:
    if target.kind == "abstract_label":
        pattern = re.compile(
            rf"(?ms)(^\*\*{re.escape(target.label)}:\*\*\s*)(.*?)"
            r"(?=^\*\*[A-Za-z][^\n]*:\*\*|^## |\Z)"
        )
    else:
        pattern = re.compile(
            rf"(?ms)(^###\s+{re.escape(target.label)}\s*\n)(.*?)"
            r"(?=^### |^## |\Z)"
        )
    match = pattern.search(text)
    return (match.start(2), match.end(2)) if match is not None else None


def project_owner_issued_manuscript_claims(
    scaffold: str,
    *,
    per_step_records: Sequence[Mapping[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Insert missing deterministic owner claims declared by typed contracts."""

    projected = scaffold
    repairs: List[Dict[str, Any]] = []
    for record in per_step_records:
        summary = record.get("step_summary")
        if not isinstance(summary, Mapping):
            continue
        reportable_blocks = [
            (str(key), value)
            for key, value in summary.items()
            if _REPORTABLE_KEY_RE.fullmatch(str(key))
            and isinstance(value, Mapping)
            and "manuscript_projection" in value
        ]
        if not reportable_blocks:
            continue
        if record.get("generation_mode") != "deterministic_standard":
            raise ManuscriptProjectionError(
                "manuscript projection requires deterministic_standard authority"
            )
        evidence_id = str(record.get("step_summary_evidence_id") or "").strip()
        if not evidence_id:
            raise ManuscriptProjectionError(
                "manuscript projection requires step_summary_evidence_id"
            )
        for block_key, reporting in reportable_blocks:
            claims = _parse_contract(
                reporting["manuscript_projection"],
                coordinate=f"{record.get('step_id')}.{block_key}.manuscript_projection",
            )
            for claim in claims:
                sentence, literals = _render_claim(claim, reporting=reporting)
                sentence = sentence.rstrip(". ") + f" {{evidence:{evidence_id}}}."
                for target in claim.targets:
                    span = _target_body_span(projected, target)
                    if span is None:
                        raise ManuscriptProjectionError(
                            f"claim {claim.claim_id} target is absent: "
                            f"{target.kind}:{target.label}"
                        )
                    start, end = span
                    body = projected[start:end]
                    if all(literal in body for literal in literals):
                        continue
                    insertion = "\n\n" + sentence + "\n"
                    projected = projected[:end] + insertion + projected[end:]
                    repairs.append(
                        {
                            "reason_code": "owner_manuscript_claim_projected",
                            "step_id": str(record.get("step_id") or ""),
                            "evidence_id": evidence_id,
                            "reportable_block": block_key,
                            "claim_id": claim.claim_id,
                            "target_kind": target.kind,
                            "target_label": target.label,
                        }
                    )
    return projected, repairs


__all__ = [
    "ManuscriptProjectionError",
    "project_owner_issued_manuscript_claims",
]
