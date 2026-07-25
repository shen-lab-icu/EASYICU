"""Three-tier database-feasibility classifier for idea-mining candidates.

The discovery report used to collapse every non-executable candidate into one
``db-cannot-do`` verdict, which hid the distinction the user actually needs:

* **executable** — both concepts resolve to the dictionary and have cohort data.
* **T1 (re-extract / derive)** — the construct IS a dictionary concept, but the
  current cohort/export lacks it (or it needs feature derivation). The fix is a
  re-extraction or a derived feature; no new clinical mapping required.
* **T2 (new concept authorable)** — the construct is NOT in the dictionary, but
  the *source database measures it* (a ``d_labitems`` / ``d_items`` itemid with a
  matching label exists). An AI-drafted concept definition + callback could
  expose it — pending human confirmation that the itemid truly represents it.
* **T3 (not in this database)** — the construct was never recorded in the source
  tables, so no dictionary extension can recover it; the honest verdict is
  "needs a different database / prospective data".

This module is a leaf: it must not import ``idea_mining`` (a module-boundary
test enforces that). It depends on the source-item catalog snapshot built by
``tools/build_source_item_catalog.py``, a small generic stopword set, and the
curated ``SYNONYM_GROUPS`` in the sibling leaf ``concept_catalog`` (reused so a
noradrenaline idea does not falsely miss a ``norepinephrine`` itemid).

Recall vs precision: source-item hits are CANDIDATES that always pass through a
human ``T2`` confirmation gate, so the safe bias here is *higher recall*. Two
recall aids expand the raw token overlap without touching the shared synonym
groups' novelty use:

* **synonym expansion** — when an idea term contains a full member phrase of a
  curated synonym group, the other members' tokens join the match set;
* **morphological folding** — a bounded ICU affix stemmer (``hyper``/``hypo``/
  ``dys`` prefixes, ``-emia``/``-aemia``/``-uria``/``-osis`` suffixes) lets
  ``hyperlactatemia`` reach a ``lactate`` itemid via a shared stem prefix. It
  only fires for tokens that actually carry such an affix, so non-morphological
  terms (``procalcitonin``, ``hemoperfusion``) are unaffected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set

from ..concept_catalog import SYNONYM_GROUPS

# Tokens too generic to be a clinically specific match: matching on these alone
# would over-call T2 ("we can build this") from research-design words.
_GENERIC_MATCH_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "the",
        "of",
        "in",
        "for",
        "with",
        "without",
        "versus",
        "vs",
        "different",
        "selected",
        "patient",
        "patients",
        "adult",
        "adults",
        "icu",
        "intensive",
        "care",
        "critically",
        "ill",
        "illness",
        "critical",
        "outcome",
        "outcomes",
        "mortality",
        "death",
        "survival",
        "prognosis",
        "score",
        "scores",
        "level",
        "levels",
        "value",
        "values",
        "status",
        "therapy",
        "therapies",
        "treatment",
        "treatments",
        "management",
        "strategy",
        "strategies",
        "timing",
        "duration",
        "durations",
        "dose",
        "doses",
        "intensity",
        "target",
        "targets",
        "assessment",
        "monitoring",
        "tool",
        "tools",
        "metric",
        "metrics",
        "efficacy",
        "safety",
        "early",
        "new",
        "novel",
        "risk",
        "factor",
        "factors",
        "association",
        "associated",
        "prediction",
        "predictor",
        "develop",
        "development",
        "onset",
        "use",
        "using",
        "based",
        "related",
        "general",
        "clinical",
        "acute",
        "chronic",
        "hospital",
        "unit",
        "stay",
        "phase",
        "first",
        "hour",
        "hours",
        "day",
        "days",
        "time",
        "window",
        "needs",
        "need",
        "needed",
        "study",
        "studies",
        "trial",
        "trials",
        "data",
        "measure",
        "measurement",
        "measurements",
        # method / descriptor qualifiers — matching on these alone is spurious
        # ("quantitative EEG" must not match "Quantitative G6PD").
        "quantitative",
        "qualitative",
        "continuous",
        "total",
        "ratio",
        "index",
        "rate",
        "count",
        "mean",
        "median",
        "peak",
        "trough",
        "baseline",
        "change",
        "delta",
        "trend",
        "trajectory",
        "point",
        "points",
        "function",
        "marker",
        "markers",
        "biomarker",
        "biomarkers",
        "parameter",
        "parameters",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

VALID_TIERS = ("executable", "T1_reextract", "T2_new_concept", "T3_not_in_db")
# precedence when the two sides disagree: a T3 blocker dominates everything.
_TIER_RANK = {
    "T3_not_in_db": 3,
    "T2_new_concept": 2,
    "T1_reextract": 1,
    "executable": 0,
}


@dataclass(frozen=True)
class SourceItemHit:
    itemid: int
    label: str
    table: str
    category: str
    matched_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SideTier:
    side: str  # "predictor" | "outcome"
    term: str
    resolved_concept: Optional[str]
    tier: str
    source_item_hits: tuple[SourceItemHit, ...] = ()


@dataclass(frozen=True)
class FeasibilityTier:
    tier: str
    predictor: SideTier
    outcome: SideTier
    human_note: str
    source_item_hits: tuple[SourceItemHit, ...] = field(default=())


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [
        tok
        for tok in _tokens(text)
        if len(tok) >= 3 and tok not in _GENERIC_MATCH_STOPWORDS
    ]


# Bounded ICU morphology: a clinical construct and the measured quantity often
# differ only by an affix ("hyperlactatemia" vs "lactate", "hypokalemia" vs
# "potassium"->"kalemia"). Folding is deliberately narrow — it fires only when a
# token actually carries one of these affixes — so ordinary terms are untouched.
_MORPH_PREFIXES = ("hyper", "hypo", "dys", "hypo")
_MORPH_SUFFIXES = ("aemia", "emia", "uria", "osis", "aemic", "emic")
_MORPH_MIN_STEM = 5


def _affix_stem(token: str) -> tuple[str, bool]:
    """Strip one leading + one trailing clinical affix. Returns (stem, changed)."""
    stem = token
    changed = False
    for prefix in _MORPH_PREFIXES:
        if stem.startswith(prefix) and len(stem) - len(prefix) >= 4:
            stem = stem[len(prefix) :]
            changed = True
            break
    for suffix in _MORPH_SUFFIXES:
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
            stem = stem[: -len(suffix)]
            changed = True
            break
    return stem, changed


def _morph_matches(term_token: str, label_token: str) -> bool:
    """True when the two tokens share a morphological stem prefix.

    Only fires when at least one side carried a clinical affix, and the shorter
    stem (>= _MORPH_MIN_STEM chars) is a prefix of the longer — so
    "hyperlactatemia" -> "lactat" matches "lactate", but no affix-free pair is
    fuzzily merged.
    """
    a, a_changed = _affix_stem(term_token)
    b, b_changed = _affix_stem(label_token)
    if not (a_changed or b_changed):
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= _MORPH_MIN_STEM and long.startswith(short)


def _synonym_expansion_tokens(raw_tokens: Set[str]) -> Set[str]:
    """Tokens contributed by any curated synonym group the term already matches.

    A group is triggered only when one of its member phrases is a full token
    subset of the term (single-word members by presence, multi-word members by
    all-tokens-present), then every member's content tokens join the match set.
    """
    extra: Set[str] = set()
    for group in SYNONYM_GROUPS:
        triggered = any(
            phrase_tokens and phrase_tokens <= raw_tokens
            for phrase_tokens in (set(_tokens(member)) for member in group)
        )
        if not triggered:
            continue
        for member in group:
            extra.update(
                tok
                for tok in _tokens(member)
                if len(tok) >= 3 and tok not in _GENERIC_MATCH_STOPWORDS
            )
    return extra


class SourceItemIndex:
    """Keyword/abbreviation index over a frozen source-item catalog snapshot."""

    def __init__(self, items: Sequence[dict]) -> None:
        self._items: List[dict] = []
        for raw in items:
            label = str(raw.get("label") or "").strip()
            if not label:
                continue
            self._items.append(
                {
                    "itemid": int(raw.get("itemid") or 0),
                    "label": label,
                    "table": str(raw.get("table") or ""),
                    "category": str(raw.get("category") or ""),
                    "abbrev": str(raw.get("abbrev") or "").strip().lower(),
                    # carried through for the deterministic concept-proposal
                    # gates (specimen / role / unit), not used by matching.
                    "fluid": str(raw.get("fluid") or ""),
                    "param_type": str(raw.get("param_type") or ""),
                    "unitname": str(raw.get("unitname") or ""),
                    "_label_tokens": set(_tokens(label)),
                }
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "SourceItemIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload.get("items", []))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def match(self, term: str, *, limit: int = 6) -> List[SourceItemHit]:
        """Return source items whose label/abbrev share specific tokens with term.

        Conservative on generic words (so research-design phrasing does not
        over-call T2) but recall-oriented on clinically specific tokens (so a
        real measurable construct is not falsely called T3). Hits are CANDIDATES
        requiring human confirmation, never an assertion that the itemid IS the
        concept.
        """
        raw = set(_tokens(term))
        wanted = set(_content_tokens(term))
        wanted |= _synonym_expansion_tokens(raw)
        if not wanted:
            return []
        # tokens that carry a clinical affix are eligible for morphological fold.
        morph_terms = [tok for tok in wanted if _affix_stem(tok)[1]]
        hits: List[tuple[float, SourceItemHit]] = []
        for item in self._items:
            label_tokens = item["_label_tokens"]
            shared = wanted & label_tokens
            abbrev = item["abbrev"]
            abbrev_hit = bool(abbrev) and abbrev in wanted
            morph_shared: Set[str] = set()
            if morph_terms:
                for term_tok in morph_terms:
                    for label_tok in label_tokens:
                        if label_tok not in shared and _morph_matches(
                            term_tok, label_tok
                        ):
                            morph_shared.add(label_tok)
            if not shared and not abbrev_hit and not morph_shared:
                continue
            matched = tuple(
                sorted(shared | morph_shared | ({abbrev} if abbrev_hit else set()))
            )
            # score: exact specific tokens dominate; abbrev high; morphological
            # (fuzzy) matches count half so exact hits always rank first.
            score = len(shared) + (2 if abbrev_hit else 0) + 0.5 * len(morph_shared)
            hits.append(
                (
                    score,
                    SourceItemHit(
                        itemid=item["itemid"],
                        label=item["label"],
                        table=item["table"],
                        category=item["category"],
                        matched_tokens=matched,
                    ),
                )
            )
        hits.sort(key=lambda pair: (-pair[0], pair[1].itemid))
        return [hit for _score, hit in hits[:limit]]


def _classify_side(
    side: str,
    term: str,
    resolved_concept: Optional[str],
    *,
    executable: bool,
    source_index: Optional[SourceItemIndex],
) -> SideTier:
    if resolved_concept is not None:
        # concept is in the dictionary; if the candidate is still blocked the
        # gap is extraction/derivation, not clinical mapping.
        tier = "executable" if executable else "T1_reextract"
        return SideTier(
            side=side, term=term, resolved_concept=resolved_concept, tier=tier
        )
    hits = source_index.match(term) if source_index is not None else []
    if hits:
        return SideTier(
            side=side,
            term=term,
            resolved_concept=None,
            tier="T2_new_concept",
            source_item_hits=tuple(hits),
        )
    return SideTier(side=side, term=term, resolved_concept=None, tier="T3_not_in_db")


def classify_feasibility_tier(
    candidate: Any,
    *,
    source_index: Optional[SourceItemIndex],
) -> FeasibilityTier:
    """Classify an executable-hypothesis candidate into the four-tier scheme.

    ``candidate`` is duck-typed (an ``ExecutableHypothesisCandidate``): it must
    expose ``resolved_predictor_concept``, ``resolved_outcome_concept``,
    ``predictor_label``, ``outcome_label`` and ``executable``.
    """
    executable = bool(getattr(candidate, "executable", False))
    pred = _classify_side(
        "predictor",
        getattr(candidate, "predictor_label", "") or "",
        getattr(candidate, "resolved_predictor_concept", None),
        executable=executable,
        source_index=source_index,
    )
    out = _classify_side(
        "outcome",
        getattr(candidate, "outcome_label", "") or "",
        getattr(candidate, "resolved_outcome_concept", None),
        executable=executable,
        source_index=source_index,
    )
    if executable:
        overall = "executable"
    else:
        overall = (
            pred.tier if _TIER_RANK[pred.tier] >= _TIER_RANK[out.tier] else out.tier
        )
        if overall == "executable":
            # not executable but both sides resolved → blocked downstream (data /
            # feature derivation): treat as re-extract/derive.
            overall = "T1_reextract"
    all_hits = tuple(pred.source_item_hits) + tuple(out.source_item_hits)
    return FeasibilityTier(
        tier=overall,
        predictor=pred,
        outcome=out,
        human_note=_human_note(overall, pred, out),
        source_item_hits=all_hits,
    )


def _human_note(overall: str, pred: SideTier, out: SideTier) -> str:
    if overall == "executable":
        return "executable on the current cohort"
    if overall == "T1_reextract":
        sides = [s.resolved_concept for s in (pred, out) if s.resolved_concept]
        return (
            "T1 in-dictionary — (re)extract or derive the concept(s): "
            + ", ".join(sides)
            if sides
            else "T1 in-dictionary — (re)extract/derive before execution"
        )
    if overall == "T2_new_concept":
        ex = ""
        for s in (pred, out):
            if s.tier == "T2_new_concept" and s.source_item_hits:
                h = s.source_item_hits[0]
                ex = f" e.g. itemid {h.itemid} '{h.label}' ({h.table})"
                break
        return (
            "T2 not in dictionary, but the source database measures it — an "
            "AI-drafted concept definition is authorable (human-confirm the "
            "itemid)." + ex
        )
    return (
        "T3 not measured in this database — needs a different database or "
        "prospective data; no dictionary extension can recover it"
    )
