"""Typed, owner-issued projection of reportable results into manuscript prose.

Execution owners may attach one ``easyicu.manuscript_projection`` contract to
any ``reportable_*_results`` mapping.  This module resolves only declared text
and numeric paths.  It does not infer an estimand, choose a result, or calculate
a new statistic; the unchanged evidence and numeric binders remain the final
authority gates.

A ``/2`` contract may instead name a host scientific claim that the same
summary compiles; the projection then places that claim's token, the only
interpretive sentence the strict Results grammar admits.  A fragment claim
renders exactly one sentence, so its evidence citation covers every value.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


_SCHEMA_VERSIONS = frozenset({
    "easyicu.manuscript_projection/1", "easyicu.manuscript_projection/2",
})
_CLAIM_TOKEN_SCHEMA_VERSION = "easyicu.manuscript_projection/2"
# The strict filter splits prose at this boundary; a projected fragment claim
# that crossed it would leave every sentence but the last uncited.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?。！？]\s+")
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
    scientific_claim_id: str | None = None


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
    schema_version = payload.get("schema_version")
    if schema_version not in _SCHEMA_VERSIONS:
        raise ManuscriptProjectionError(f"{coordinate} has unsupported schema_version")
    claim_keys = {"claim_id", "targets", "fragments"}
    if schema_version == _CLAIM_TOKEN_SCHEMA_VERSION:
        claim_keys.add("scientific_claim_id")
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
            allowed=frozenset(claim_keys),
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
        targets = tuple(
            _parse_target(item, coordinate=f"{claim_coordinate}.targets[{i}]")
            for i, item in enumerate(raw_targets)
        )
        if "scientific_claim_id" in raw_claim:
            scientific_claim_id = str(raw_claim.get("scientific_claim_id") or "").strip()
            if raw_fragments is not None or not _CLAIM_ID_RE.fullmatch(scientific_claim_id):
                raise ManuscriptProjectionError(
                    f"{claim_coordinate} must name one scientific claim and no fragments"
                )
            claims.append(_Claim(
                claim_id=claim_id, targets=targets, fragments=(),
                scientific_claim_id=scientific_claim_id,
            ))
            continue
        if not isinstance(raw_fragments, list) or not raw_fragments:
            raise ManuscriptProjectionError(
                f"{claim_coordinate}.fragments must be non-empty"
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
    if _SENTENCE_BOUNDARY_RE.search(sentence):
        raise ManuscriptProjectionError(
            f"claim {claim.claim_id} renders more than one sentence; each "
            "projected sentence must carry its own evidence"
        )
    return sentence, tuple(literals)


def _target_body_span(text: str, target: _Target) -> tuple[int, int] | None:
    # The label matches in any case, as the quality gate reads abstract labels
    # and required subsections: a draft's "### Primary Association" is present.
    if target.kind == "abstract_label":
        pattern = re.compile(
            rf"(?ms)(^\*\*(?i:{re.escape(target.label)}):\*\*\s*)(.*?)"
            r"(?=^\*\*[A-Za-z][^\n]*:\*\*|^## |\Z)"
        )
    else:
        pattern = re.compile(
            rf"(?ms)(^###\s+(?i:{re.escape(target.label)})\s*\n)(.*?)"
            r"(?=^### |^## |\Z)"
        )
    match = pattern.search(text)
    return (match.start(2), match.end(2)) if match is not None else None


def _compiled_claim_ids(summary: Mapping[str, Any]) -> frozenset[str]:
    """The scientific claims the host compiles from one owner summary."""

    from ..authority.scientific_claims import derive_scientific_claim_drafts

    try:
        return frozenset(
            draft.claim_id for draft in derive_scientific_claim_drafts(dict(summary))
        )
    except ValueError as exc:
        raise ManuscriptProjectionError(
            f"the owner summary does not compile its scientific claims: {exc}"
        ) from exc


def project_owner_issued_manuscript_claims(
    scaffold: str,
    *,
    per_step_records: Sequence[Mapping[str, Any]],
    absent_targets: List[Dict[str, Any]] | None = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """Insert missing deterministic owner claims declared by typed contracts.

    A target section the draft lacks raises, unless the caller collects it in
    ``absent_targets``: a Writer draft without the section is the Writer's
    failure, not an invalid owner contract, and that caller reports it.
    """

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
        compiled_claim_ids: frozenset[str] | None = None
        for block_key, reporting in reportable_blocks:
            claims = _parse_contract(
                reporting["manuscript_projection"],
                coordinate=f"{record.get('step_id')}.{block_key}.manuscript_projection",
            )
            for claim in claims:
                if claim.scientific_claim_id is not None:
                    if compiled_claim_ids is None:
                        compiled_claim_ids = _compiled_claim_ids(summary)
                    step_id = str(record.get("step_id") or "").strip()
                    if not step_id or claim.scientific_claim_id not in compiled_claim_ids:
                        raise ManuscriptProjectionError(
                            f"claim {claim.claim_id} names scientific claim "
                            f"{claim.scientific_claim_id!r}, which this step's summary "
                            "does not compile"
                        )
                    # A bare token paragraph, as host claim placement writes it.
                    sentence = "{claim:" + f"{step_id}.{claim.scientific_claim_id}" + "}"
                    literals: tuple[str, ...] = (sentence,)
                else:
                    sentence, literals = _render_claim(claim, reporting=reporting)
                    sentence = sentence.rstrip(". ") + f" {{evidence:{evidence_id}}}."
                for target in claim.targets:
                    span = _target_body_span(projected, target)
                    if span is None and absent_targets is not None:
                        absent_targets.append(
                            {
                                "step_id": str(record.get("step_id") or ""),
                                "evidence_id": evidence_id,
                                "claim_id": claim.claim_id,
                                "target_kind": target.kind,
                                "target_label": target.label,
                            }
                        )
                        continue
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


#: The Writer section that holds each kind of projection target.
_TARGET_SECTION_KEYS = {"abstract_label": "abstract", "markdown_heading": "results"}


def with_absent_target_repairs(
    section_errors: Mapping[str, Tuple[str, ...]],
    absent_targets: Sequence[Mapping[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    """Section repairs plus one for each owner target the Writer draft lacks."""

    merged = {key: tuple(values) for key, values in section_errors.items()}
    for target in absent_targets:
        place = (
            f"the **{target['target_label']}:** label"
            if target["target_kind"] == "abstract_label"
            else f"the '### {target['target_label']}' subsection"
        )
        detail = (
            f"OWNER_CLAIM_TARGET_ABSENT: restore {place}; the signed owner of "
            f"{target['evidence_id']} places its result claims there."
        )
        key = _TARGET_SECTION_KEYS[target["target_kind"]]
        if detail not in merged.get(key, ()):
            merged[key] = (*merged.get(key, ()), detail)
    return merged


__all__ = [
    "ManuscriptProjectionError",
    "project_owner_issued_manuscript_claims",
    "with_absent_target_repairs",
]
