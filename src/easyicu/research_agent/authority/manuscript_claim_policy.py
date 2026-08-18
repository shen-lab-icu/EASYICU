"""Pure manuscript grammar for evidence-bound and scientific-claim prose.

The policy distinguishes structural Markdown, manuscript metadata, numeric
results, qualitative scientific assertions, and complete host-issued claim
tokens.  It receives a claim resolver from the evidence owner and never reads
or mutates EvidenceStore state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Optional

from .scientific_claims import ScientificClaim

ClaimResolver = Callable[[str], Optional[ScientificClaim]]


@dataclass(frozen=True)
class ScaffoldPolicyResult:
    scaffold: str
    filtered_sentences: tuple[str, ...]
    removed_result_sentences: tuple[str, ...]
    unsupported_scientific_claim_sentences: tuple[str, ...]


@dataclass(frozen=True)
class ScientificClaimExpansion:
    scaffold: str
    missing_claim_refs: tuple[str, ...]
    malformed_sentences: tuple[str, ...]


_RESULT_TOKEN_RE = re.compile(
    r"(\bOR\b|\bHR\b|\bRR\b|\bAUC\b|\bAUROC\b|\bBrier\b|\bcalibration\b|"
    r"\bdiscrimination\b|\bperformance\b|\brobust(?:ness)?\b|"
    r"\boverfitting\b|\bmiscalibration\b|\bmissingness\b|\bconsistent\b|"
    r"\bgeneralisa(?:bility|ble)\b|"
    r"\bgeneraliza(?:bility|ble)\b|"
    r"\bmedian\b|\bmean\b|\bincidence\b|\bmortality\b|\bhazard\b|"
    r"\bconfidence interval\b|\bCI\b|\bp\s*[<=>]|%|\d)",
    re.I,
)
_SCIENTIFIC_CLAIM_TOKEN_RE = re.compile(
    r"\{claim:(?P<ref>[A-Za-z0-9_-]+\.[a-z][a-z0-9_]*)\}"
)
_SCIENTIFIC_CLAIM_SENTENCE_RE = re.compile(
    r"\s*\{claim:(?P<ref>[A-Za-z0-9_-]+\.[a-z][a-z0-9_]*)\}[.!?。！？]?\s*"
)
_VALID_CLAIM_TOKEN_RE = re.compile(
    r"(?<!\{)\{claim:[A-Za-z0-9_-]+\.[a-z][a-z0-9_]*\}(?!\})"
)
_VALID_EVIDENCE_TOKEN_RE = re.compile(
    r"(?:\{\{evidence:[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\s*,\s*(?:evidence:)?[A-Za-z0-9][A-Za-z0-9_.-]*)*\}\}|"
    r"(?<!\{)\{evidence:[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\s*,\s*(?:evidence:)?[A-Za-z0-9][A-Za-z0-9_.-]*)*\}(?!\}))"
)
_LITERATURE_CITATION_MARKER_RE = re.compile(r"\[@[A-Za-z0-9_.:-]+\]")
_AUTHORITY_PLACEHOLDER_PREFIX_RE = re.compile(r"\{+\s*(?:claim|evidence)\s*:", re.I)
_QUALITATIVE_SCIENTIFIC_ASSERTION_RE = re.compile(
    r"\b(?:independently\s+)?associated\s+with\b|"
    r"\b(?:no\s+(?:clear\s+)?)?association\s+with\b|"
    r"\bcorrelat(?:ed|es|ing)\s+with\b|"
    r"\bpredict(?:ed|s|ing)\b|"
    r"\b(?:higher|lower|greater|less|elevated|unchanged|similar)\b|"
    r"\b(?:increas(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|"
    r"reduc(?:e|ed|es|ing)|declin(?:e|ed|es|ing)|"
    r"improv(?:e|ed|es|ing)|worsen(?:ed|s|ing)?)\b|"
    r"\b(?:consistent\s+with|may\s+reflect|could\s+reflect)\b|"
    r"\b(?:linked\s+to|suggest(?:ed|s|ing)(?:\s+(?:that|an?|the))?)\b|"
    r"\bconfer(?:red|s|ring)\s+(?:an?\s+)?(?:benefit|harm)\b|"
    r"\bsurvival\s+benefit\b|"
    r"\b(?:protective|harmful)(?:\s+(?:against|for|to))?\b|"
    r"\bfared\s+(?:better|worse)\b|"
    r"\bexperienced\s+(?:excess\s+)?(?:mortality|harm|benefit)\b|"
    r"\badversely\s+affect(?:ed|s|ing)\s+(?:the\s+)?outcomes?\b",
    re.I,
)
_MANUSCRIPT_METADATA_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?"
    r"(?:keywords?|key words|funding|conflicts?\s+of\s+interest|"
    r"data\s+(?:and\s+code\s+)?availability|code\s+availability|"
    r"ethics\s+approval|acknowledg(?:e)?ments?)"
    r"\s*(?:\*\*)?\s*[:：]",
    re.I,
)
_KEYWORD_ASSERTION_VERB_RE = re.compile(
    r"\b(?:associated\s+with|correlat(?:ed|es|ing)\s+with|predict(?:ed|s|ing)|"
    r"conferred|fared|experienced|adversely\s+affect(?:ed|s|ing)|"
    r"was|were|had|showed|demonstrated)\b",
    re.I,
)
_AVAILABILITY_BOILERPLATE_RE = re.compile(
    r"\b(?:generated scripts?|sha-?256|evidence store|reproducibility envelope|"
    r"strobe checklist|supplementary tables?|released alongside|available from|"
    r"data availability|code availability)\b",
    re.I,
)
_AVAILABILITY_ACTION_RE = re.compile(
    r"\b(?:released|available|deposited|archived|provided|shared|included)\b",
    re.I,
)
_HEADING_RESULT_ASSERTION_RE = re.compile(
    r"\b(?:higher|lower|greater|less|increas(?:e|ed|es|ing)|"
    r"decreas(?:e|ed|es|ing)|reduc(?:e|ed|es|ing)|elevated|"
    r"declin(?:e|ed|es|ing)|improv(?:e|ed|es|ing)|worsen(?:ed|s|ing)?|"
    r"associated|correlated|predicted|significant(?:ly)?|unchanged|similar|"
    r"beneficial|harmful|better|worse|benefit|harm|yield(?:ed|s|ing))\b",
    re.I,
)
_HEADING_RESULT_CONTEXT_RE = re.compile(
    r"\b(?:OR|HR|RR|AUC|AUROC|Brier|calibration|discrimination|"
    r"median|mean|incidence|mortality|hazard|confidence interval|CI|p)\b",
    re.I,
)
_HEADING_NUMERIC_RE = re.compile(r"(?:\d|%|\bp\s*[<=>])", re.I)
_HEADING_RESULT_VERB_RE = re.compile(
    r"\b(?:was|were|had|showed|demonstrated|differ(?:ed|s)?|varied)\b",
    re.I,
)


def _looks_manuscript_metadata_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    prefix_match = _MANUSCRIPT_METADATA_PREFIX_RE.search(stripped)
    metadata_form = bool(prefix_match) or bool(
        _AVAILABILITY_BOILERPLATE_RE.search(stripped)
        and _AVAILABILITY_ACTION_RE.search(stripped)
    )
    if not metadata_form:
        return False
    if prefix_match and "keyword" in prefix_match.group(0).lower():
        return _KEYWORD_ASSERTION_VERB_RE.search(stripped[prefix_match.end() :]) is None
    return not _contains_scientific_clause(stripped)


def _contains_scientific_clause(sentence: str) -> bool:
    if _QUALITATIVE_SCIENTIFIC_ASSERTION_RE.search(sentence):
        return True
    return bool(
        _HEADING_RESULT_CONTEXT_RE.search(sentence)
        and _HEADING_RESULT_VERB_RE.search(sentence)
    )


def _contains_malformed_authority_placeholder(sentence: str) -> bool:
    without_valid = _VALID_CLAIM_TOKEN_RE.sub("", sentence)
    without_valid = _VALID_EVIDENCE_TOKEN_RE.sub("", without_valid)
    return _AUTHORITY_PLACEHOLDER_PREFIX_RE.search(without_valid) is not None


def malformed_authority_placeholder_sentences(scaffold: str) -> tuple[str, ...]:
    """Return sentences containing claim/evidence candidates outside the grammar."""

    malformed: list[str] = []
    for line in scaffold.splitlines():
        for sentence in _split_sentences(line):
            if _contains_malformed_authority_placeholder(sentence):
                malformed.append(sentence.strip())
    return tuple(malformed)


def _looks_qualitative_scientific_assertion(sentence: str) -> bool:
    if _looks_manuscript_metadata_sentence(sentence):
        return False
    return bool(_QUALITATIVE_SCIENTIFIC_ASSERTION_RE.search(sentence))


def _looks_result_like_sentence(sentence: str) -> bool:
    if _looks_manuscript_metadata_sentence(sentence):
        return False
    # A year or version embedded only in an exact run-bound literature key is
    # citation metadata, not a reported manuscript value.  The literature
    # owner separately rejects unknown keys and missing section authority.
    # Strip only the closed ``[@key]`` marker; numeric or interpretive prose
    # surrounding it remains subject to the unchanged result/claim gates.
    prose = _LITERATURE_CITATION_MARKER_RE.sub("", sentence)
    return bool(_RESULT_TOKEN_RE.search(prose))


def _split_sentences(text: str) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+", text.strip())
        if part.strip()
    ]
    return parts or ([text.strip()] if text.strip() else [])


def _split_markdown_structure_prefix(line: str) -> tuple[str, str]:
    cursor = len(line) - len(line.lstrip())
    marker_re = re.compile(r"(?:>\s*|[-+*]\s+|\d+[.)]\s+)")
    while match := marker_re.match(line, cursor):
        cursor = match.end()
    return line[:cursor], line[cursor:]


def _split_markdown_heading_prefix(content: str) -> tuple[str, str]:
    match = re.match(r"^(\s*#{1,6}(?:\s+|$))", content)
    if match is None:
        return "", content
    return match.group(1), content[match.end() :]


def _heading_requires_evidence(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    semantic = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", stripped, count=1)
    if _contains_scientific_clause(semantic):
        return True
    if _HEADING_RESULT_ASSERTION_RE.search(semantic):
        return True
    if _HEADING_RESULT_CONTEXT_RE.search(semantic) and _HEADING_RESULT_VERB_RE.search(
        semantic
    ):
        return True
    return bool(
        _HEADING_NUMERIC_RE.search(semantic)
        and _HEADING_RESULT_CONTEXT_RE.search(semantic)
    )


def filter_evidence_bound_scaffold(
    scaffold: str,
    *,
    resolve_claim: ClaimResolver,
) -> ScaffoldPolicyResult:
    """Filter unsupported result prose while preserving Markdown structure."""

    removed: list[str] = []
    unsupported_scientific_claims: list[str] = []
    filtered_claims: list[str] = []
    filtered_lines: list[str] = []
    for raw_line in scaffold.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")) or not stripped:
            filtered_lines.append(line)
            continue
        structure_prefix, content = _split_markdown_structure_prefix(line)
        heading_prefix, heading_content = _split_markdown_heading_prefix(content)
        heading_requires_claim = False
        if heading_prefix:
            if not _heading_requires_evidence(heading_content):
                filtered_lines.append(line)
                continue
            heading_requires_claim = True
            structure_prefix += heading_prefix
            content = heading_content
        if not content.strip():
            filtered_lines.append(line)
            continue
        sentences = _split_sentences(content)
        if (
            len(sentences) == 1
            and not heading_requires_claim
            and not _looks_result_like_sentence(sentences[0])
            and not _looks_qualitative_scientific_assertion(sentences[0])
            and not _contains_malformed_authority_placeholder(sentences[0])
            and _SCIENTIFIC_CLAIM_TOKEN_RE.search(sentences[0]) is None
        ):
            filtered_lines.append(line)
            continue
        kept: list[str] = []
        for sentence in sentences:
            if _contains_malformed_authority_placeholder(sentence):
                rejected = sentence.strip()
                unsupported_scientific_claims.append(rejected)
                filtered_claims.append(rejected)
                continue
            claim_match = _SCIENTIFIC_CLAIM_SENTENCE_RE.fullmatch(sentence.strip())
            if claim_match is not None:
                if resolve_claim(claim_match.group("ref")) is None:
                    rejected = sentence.strip()
                    unsupported_scientific_claims.append(rejected)
                    filtered_claims.append(rejected)
                    continue
                kept.append(sentence.strip())
                continue
            if heading_requires_claim:
                rejected = sentence.strip()
                unsupported_scientific_claims.append(rejected)
                filtered_claims.append(rejected)
                continue
            if _SCIENTIFIC_CLAIM_TOKEN_RE.search(
                sentence
            ) or _looks_qualitative_scientific_assertion(sentence):
                rejected = sentence.strip()
                unsupported_scientific_claims.append(rejected)
                filtered_claims.append(rejected)
                continue
            if _looks_result_like_sentence(sentence):
                rejected = sentence.strip()
                removed.append(rejected)
                filtered_claims.append(rejected)
                continue
            kept.append(sentence.strip())
        kept_content = " ".join(part for part in kept if part).strip()
        filtered_lines.append(
            f"{structure_prefix}{kept_content}".rstrip() if kept_content else ""
        )
    return ScaffoldPolicyResult(
        scaffold="\n".join(filtered_lines).strip() + "\n",
        filtered_sentences=tuple(filtered_claims),
        removed_result_sentences=tuple(removed),
        unsupported_scientific_claim_sentences=tuple(
            unsupported_scientific_claims
        ),
    )


def expand_scientific_claim_tokens(
    scaffold: str,
    *,
    resolve_claim: ClaimResolver,
    current_evidence_ids: Optional[set[str]] = None,
) -> ScientificClaimExpansion:
    """Replace complete claim-token sentences with host-rendered prose."""

    out: list[str] = []
    missing: list[str] = []
    malformed: list[str] = []
    for raw_line in scaffold.splitlines():
        structure_prefix, content = _split_markdown_structure_prefix(raw_line)
        token_match = _SCIENTIFIC_CLAIM_SENTENCE_RE.fullmatch(content.strip())
        if token_match is None:
            if _SCIENTIFIC_CLAIM_TOKEN_RE.search(
                content
            ) or _contains_malformed_authority_placeholder(content):
                malformed.append(content.strip())
            out.append(raw_line)
            continue
        claim_ref = token_match.group("ref")
        claim = resolve_claim(claim_ref)
        if claim is None or (
            current_evidence_ids is not None
            and claim.evidence_id not in current_evidence_ids
        ):
            missing.append(claim_ref)
            out.append(f"{structure_prefix}[scientific claim missing: {claim_ref}]")
            continue
        out.append(
            f"{structure_prefix}{claim.render_text()} "
            f"{{evidence:{claim.evidence_id}}}"
        )
    return ScientificClaimExpansion(
        scaffold="\n".join(out),
        missing_claim_refs=tuple(missing),
        malformed_sentences=tuple(malformed),
    )


__all__ = [
    "ScaffoldPolicyResult",
    "ScientificClaimExpansion",
    "expand_scientific_claim_tokens",
    "filter_evidence_bound_scaffold",
    "malformed_authority_placeholder_sentences",
]
