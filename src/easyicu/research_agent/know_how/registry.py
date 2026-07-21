"""Strict card schema and deterministic offline retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..planning.study_design_playbook import StudyDesignFamily

CARD_SCHEMA_VERSION = "easyicu.research_know_how/1"
RETRIEVAL_SCHEMA_VERSION = "easyicu.research_know_how_retrieval/1"
MAX_CARD_BYTES = 24_000
MAX_PER_CARD_PROMPT_CHARS = 1_200
MAX_TOTAL_PROMPT_CHARS = 8_000
MAX_RETRIEVAL_HITS = 5

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "clinical",
        "data",
        "for",
        "from",
        "how",
        "in",
        "is",
        "icu",
        "outcome",
        "outcomes",
        "of",
        "on",
        "or",
        "patient",
        "patients",
        "study",
        "the",
        "to",
        "what",
        "which",
        "with",
    }
)


class KnowHowIntegrityError(ValueError):
    """A card set or persisted retrieval artifact is inconsistent."""


def _clean_text(value: object, *, label: str, max_length: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{label} must be non-empty")
    if len(text) > max_length:
        raise ValueError(f"{label} must be <= {max_length} characters")
    return text


def _clean_unique_strings(
    values: Iterable[object], *, label: str, max_items: int = 32
) -> list[str]:
    cleaned = [
        _clean_text(value, label=f"{label} item", max_length=500) for value in values
    ]
    if not cleaned:
        raise ValueError(f"{label} must contain at least one item")
    if len(cleaned) > max_items:
        raise ValueError(f"{label} must contain at most {max_items} items")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{label} items must be unique")
    return cleaned


class KnowHowCitation(BaseModel):
    """One verifiable source and the scope for which it is cited."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    title: str = Field(max_length=400)
    year: int = Field(ge=1900, le=2100)
    source_type: Literal[
        "guideline",
        "consensus",
        "reporting_standard",
        "methods",
        "representative_study",
    ]
    url: str
    doi: Optional[str] = Field(default=None, max_length=160)
    supports: list[str] = Field(min_length=1, max_length=12)

    @field_validator("citation_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _ID_RE.fullmatch(value):
            raise ValueError("citation_id must be a lowercase stable identifier")
        return value

    @field_validator("title")
    @classmethod
    def _valid_title(cls, value: str) -> str:
        return _clean_text(value, label="citation title", max_length=400)

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        value = str(value or "").strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("citation url must be an absolute HTTP(S) URL")
        return value

    @field_validator("doi")
    @classmethod
    def _valid_doi(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value.lower().startswith("10.") or "/" not in value:
            raise ValueError("doi must use canonical 10.x/... form")
        return value

    @field_validator("supports")
    @classmethod
    def _valid_supports(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, label="citation supports", max_items=12)


class KnowHowDesignCandidates(BaseModel):
    """Advisory design coordinates; none are execution authority."""

    model_config = ConfigDict(extra="forbid")

    population: str = Field(max_length=700)
    time_zero: str = Field(max_length=700)
    observation_window: str = Field(max_length=700)
    prediction_or_followup_window: str = Field(max_length=700)
    eligibility_candidates: list[str] = Field(min_length=1, max_length=20)
    exposure: str = Field(max_length=700)
    outcome: str = Field(max_length=700)
    estimand: str = Field(max_length=700)
    recommended_methods: list[str] = Field(min_length=1, max_length=20)
    sensitivity_analyses: list[str] = Field(min_length=1, max_length=20)

    @field_validator(
        "population",
        "time_zero",
        "observation_window",
        "prediction_or_followup_window",
        "exposure",
        "outcome",
        "estimand",
    )
    @classmethod
    def _valid_scalar(cls, value: str) -> str:
        return _clean_text(value, label="design candidate", max_length=700)

    @field_validator(
        "eligibility_candidates", "recommended_methods", "sensitivity_analyses"
    )
    @classmethod
    def _valid_lists(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(
            values, label="design candidate list", max_items=20
        )


class KnowHowCard(BaseModel):
    """A versioned ICU research-protocol card."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["easyicu.research_know_how/1"] = CARD_SCHEMA_VERSION
    card_id: str
    version: str
    title: str = Field(max_length=180)
    summary: str = Field(max_length=1_200)
    clinical_topics: list[str] = Field(min_length=1, max_length=20)
    study_families: list[StudyDesignFamily] = Field(min_length=1, max_length=6)
    applicable_databases: list[str] = Field(min_length=1, max_length=16)
    design_candidates: KnowHowDesignCandidates
    required_concepts: list[str] = Field(min_length=1, max_length=32)
    stop_conditions: list[str] = Field(min_length=1, max_length=20)
    citations: list[KnowHowCitation] = Field(min_length=2, max_length=12)
    review_status: Literal["curated_mvp"] = "curated_mvp"

    @field_validator("card_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _ID_RE.fullmatch(value):
            raise ValueError("card_id must be a lowercase stable identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("version must be semantic x.y.z")
        return value

    @field_validator("title")
    @classmethod
    def _valid_title(cls, value: str) -> str:
        return _clean_text(value, label="card title", max_length=180)

    @field_validator("summary")
    @classmethod
    def _valid_summary(cls, value: str) -> str:
        return _clean_text(value, label="card summary", max_length=1_200)

    @field_validator(
        "clinical_topics",
        "applicable_databases",
        "required_concepts",
        "stop_conditions",
    )
    @classmethod
    def _valid_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, label="card list", max_items=32)

    @field_validator("study_families")
    @classmethod
    def _unique_families(
        cls, values: list[StudyDesignFamily]
    ) -> list[StudyDesignFamily]:
        if len(values) != len(set(values)):
            raise ValueError("study_families must be unique")
        return values

    @model_validator(mode="after")
    def _unique_citations(self) -> "KnowHowCard":
        ids = [citation.citation_id for citation in self.citations]
        if len(ids) != len(set(ids)):
            raise ValueError("citation_id values must be unique within a card")
        return self


class KnowHowHit(BaseModel):
    """Serializable result of deterministic card retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: str
    version: str
    file_sha256: str
    score: float = Field(ge=0.0, le=1.0)
    match_reasons: list[str]
    unresolved_concepts: list[str]
    citation_ids: list[str]

    @field_validator("card_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _ID_RE.fullmatch(value):
            raise ValueError("card_id must be a lowercase stable identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        value = str(value or "").strip()
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("version must be semantic x.y.z")
        return value

    @field_validator("match_reasons", "citation_ids")
    @classmethod
    def _valid_nonempty_lists(cls, values: list[str]) -> list[str]:
        return _clean_unique_strings(values, label="know-how hit list", max_items=20)

    @field_validator("file_sha256")
    @classmethod
    def _valid_sha(cls, value: str) -> str:
        if not _SHA_RE.fullmatch(str(value or "")):
            raise ValueError("file_sha256 must be a lowercase SHA-256 digest")
        return value


@dataclass(frozen=True)
class _LoadedCard:
    card: KnowHowCard
    source: Any
    file_sha256: str


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class KnowHowRegistry:
    """Validated card registry with bounded deterministic retrieval."""

    def __init__(self, entries: Sequence[_LoadedCard]) -> None:
        by_id: dict[str, _LoadedCard] = {}
        for entry in entries:
            if entry.card.card_id in by_id:
                raise KnowHowIntegrityError(
                    f"duplicate know-how card_id: {entry.card.card_id}"
                )
            by_id[entry.card.card_id] = entry
        self._entries = dict(sorted(by_id.items()))

    @classmethod
    def load(
        cls,
        paths: Sequence[str | Path] = (),
        *,
        include_builtin: bool = True,
    ) -> "KnowHowRegistry":
        sources: list[Any] = []
        if include_builtin:
            builtin_dir = resources.files("easyicu").joinpath(
                "data", "research_know_how"
            )
            sources.extend(
                sorted(
                    (
                        item
                        for item in builtin_dir.iterdir()
                        if item.name.endswith(".json")
                    ),
                    key=lambda item: item.name,
                )
            )
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"know-how path does not exist: {path}")
            if path.is_dir():
                sources.extend(sorted(path.glob("*.json")))
            elif path.suffix.lower() == ".json":
                sources.append(path)
            else:
                raise KnowHowIntegrityError(
                    f"know-how path must be a JSON file or directory: {path}"
                )

        entries: list[_LoadedCard] = []
        for source in sources:
            raw = source.read_bytes()
            if len(raw) > MAX_CARD_BYTES:
                raise KnowHowIntegrityError(
                    f"know-how card exceeds {MAX_CARD_BYTES} bytes: {source}"
                )
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise KnowHowIntegrityError(
                    f"invalid know-how JSON at {source}: {exc}"
                ) from exc
            card = KnowHowCard.model_validate(payload)
            entries.append(
                _LoadedCard(
                    card=card,
                    source=source,
                    file_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
        if not entries:
            raise KnowHowIntegrityError("know-how registry contains no cards")
        return cls(entries)

    @property
    def cards(self) -> tuple[KnowHowCard, ...]:
        return tuple(entry.card for entry in self._entries.values())

    def get(self, card_id: str) -> KnowHowCard:
        try:
            return self._entries[card_id].card
        except KeyError as exc:
            raise KeyError(f"unknown know-how card_id: {card_id}") from exc

    def verify_hit_source(self, hit: KnowHowHit) -> None:
        entry = self._entries.get(hit.card_id)
        if entry is None:
            raise KnowHowIntegrityError(
                f"retrieval hit references unknown card: {hit.card_id}"
            )
        current_sha = hashlib.sha256(entry.source.read_bytes()).hexdigest()
        if current_sha != hit.file_sha256 or current_sha != entry.file_sha256:
            raise KnowHowIntegrityError(
                f"know-how source digest changed for {hit.card_id}"
            )

    def retrieve(
        self,
        *,
        query: str,
        study_family: Optional[str] = None,
        database: Optional[str] = None,
        available_concepts: Iterable[str] = (),
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> list[KnowHowHit]:
        if top_k < 0 or top_k > MAX_RETRIEVAL_HITS:
            raise ValueError(f"top_k must be between 0 and {MAX_RETRIEVAL_HITS}")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if top_k == 0:
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        available = {str(value).casefold() for value in available_concepts if value}
        family = str(study_family or "").casefold()
        db = str(database or "").casefold()
        scored: list[KnowHowHit] = []
        for entry in self._entries.values():
            card = entry.card
            indexed = " ".join(
                [
                    card.title,
                    card.summary,
                    *card.clinical_topics,
                    *card.required_concepts,
                ]
            )
            lexical = _jaccard(query_tokens, _tokens(indexed))
            # An analysis-family match alone must never retrieve an unrelated card.
            if lexical == 0.0:
                continue
            family_match = 1.0 if family and family in card.study_families else 0.0
            card_databases = {value.casefold() for value in card.applicable_databases}
            database_match = 1.0 if db and db in card_databases else 0.0
            required = {value.casefold() for value in card.required_concepts}
            concept_overlap = (
                len(required & available) / len(required) if required else 0.0
            )
            score = min(
                1.0,
                0.60 * lexical
                + 0.15 * family_match
                + 0.10 * database_match
                + 0.15 * concept_overlap,
            )
            if score < min_score:
                continue
            reasons = [f"lexical={lexical:.3f}"]
            if family_match:
                reasons.append(f"study_family={family}")
            if database_match:
                reasons.append(f"database={db}")
            if concept_overlap:
                reasons.append(f"concept_coverage={concept_overlap:.3f}")
            unresolved = sorted(required - available)
            scored.append(
                KnowHowHit(
                    card_id=card.card_id,
                    version=card.version,
                    file_sha256=entry.file_sha256,
                    score=round(score, 6),
                    match_reasons=reasons,
                    unresolved_concepts=unresolved,
                    citation_ids=[item.citation_id for item in card.citations],
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.card_id))
        return scored[:top_k]

    def render_prompt(
        self,
        hits: Sequence[KnowHowHit],
        *,
        per_card_limit: int = MAX_PER_CARD_PROMPT_CHARS,
        total_limit: int = MAX_TOTAL_PROMPT_CHARS,
    ) -> str:
        if per_card_limit < 300 or per_card_limit > MAX_PER_CARD_PROMPT_CHARS:
            raise ValueError(f"per_card_limit must be 300..{MAX_PER_CARD_PROMPT_CHARS}")
        if total_limit < 1_000 or total_limit > MAX_TOTAL_PROMPT_CHARS:
            raise ValueError(f"total_limit must be 1000..{MAX_TOTAL_PROMPT_CHARS}")
        header = (
            "# Retrieved Research Know-How (ADVISORY)\n\n"
            "These source-backed cards are design candidates, not scientific "
            "authority. Apply a candidate only when it matches the user's target "
            "population, time zero, estimand, and typed data. Never invent a "
            "missing field or silently exclude rows. Record adopted cards in "
            "top-level `know_how_refs`; leave the list empty when none apply.\n"
        )
        if not hits:
            return header + "\nNo cards met the configured relevance threshold.\n"
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            self.verify_hit_source(hit)
            card = self.get(hit.card_id)
            design = card.design_candidates
            sources = "; ".join(
                f"{item.citation_id}: {item.doi or item.url}" for item in card.citations
            )
            block = (
                f"\n## Card {index}: {card.title} [{card.card_id}@{card.version}]\n"
                f"score={hit.score:.3f}; sha256={hit.file_sha256}; "
                f"review_status={card.review_status}\n"
                f"required_concepts: {', '.join(card.required_concepts)}\n"
                f"unresolved_concepts: {', '.join(hit.unresolved_concepts) or 'none'}\n"
                f"summary: {card.summary}\n"
                f"population: {design.population}\n"
                f"time_zero: {design.time_zero}\n"
                f"windows: {design.observation_window}; "
                f"{design.prediction_or_followup_window}\n"
                f"estimand: {design.estimand}\n"
                f"methods: {'; '.join(design.recommended_methods)}\n"
                f"sensitivity: {'; '.join(design.sensitivity_analyses)}\n"
                f"stop_conditions: {'; '.join(card.stop_conditions)}\n"
                f"sources: {sources}\n"
            )
            if len(block) > per_card_limit:
                marker = (
                    "\n[card projection truncated; full card remains in registry]\n"
                )
                block = block[: per_card_limit - len(marker)].rstrip() + marker
            blocks.append(block)
        rendered = header + "".join(blocks)
        if len(rendered) > total_limit:
            marker = "\n[know-how prompt projection truncated]\n"
            rendered = rendered[: total_limit - len(marker)].rstrip() + marker
        return rendered

    def retrieval_receipt(
        self,
        *,
        query: str,
        study_family: Optional[str],
        database: Optional[str],
        available_concepts: Iterable[str],
        hits: Sequence[KnowHowHit],
        top_k: int,
        min_score: float,
    ) -> dict[str, Any]:
        for hit in hits:
            self.verify_hit_source(hit)
        return {
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
            "query": query,
            "study_family": study_family,
            "database": database,
            "available_concepts": sorted(
                {str(value) for value in available_concepts if value}
            ),
            "candidate_count": len(self._entries),
            "selected_count": len(hits),
            "top_k": top_k,
            "min_score": min_score,
            "selected": [hit.model_dump(mode="json") for hit in hits],
            "withheld_count": max(0, len(self._entries) - len(hits)),
            "authority": "advisory_only",
        }


__all__ = [
    "CARD_SCHEMA_VERSION",
    "MAX_CARD_BYTES",
    "MAX_PER_CARD_PROMPT_CHARS",
    "MAX_RETRIEVAL_HITS",
    "MAX_TOTAL_PROMPT_CHARS",
    "KnowHowCard",
    "KnowHowCitation",
    "KnowHowDesignCandidates",
    "KnowHowHit",
    "KnowHowIntegrityError",
    "KnowHowRegistry",
]
