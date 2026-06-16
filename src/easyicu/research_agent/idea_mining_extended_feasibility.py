"""Extended database-feasibility reconsideration for idea-mining.

The base feasibility gate marks a candidate ``db-cannot-do`` whenever its
predictor / outcome / analysis constructs do not resolve against the *current
cohort export*. That over-rejects two cases the user explicitly wants surfaced
(2026-06-16):

* **Case 1 — ICD-derivable cohort.** The blocking element is the
  *population* (a disease with no derived concept, e.g. traumatic brain injury,
  Legionnaires' disease). It can be defined by the agent from ICD diagnosis
  codes — *provided the ICD definition is accurate*. We propose a candidate
  code-set + coverage + a reliability flag and ALWAYS require human
  confirmation; we never auto-execute. Known under-coded diagnoses (e.g.
  ICU-acquired weakness / critical-illness myopathy) are flagged unreliable.

* **Case 2 — dictionary-definable feature.** The construct IS a concept in the
  EasyICU dictionary, just not in the current export — either available for the
  current database (the agent must re-extract it first) or available only for
  *another* database (the study is executable there). This is a re-extract /
  cross-database route, not a database limit.

This module only ever DOWNGRADES a ``db-cannot-do`` to ``hold`` with an
actionable reason; it never promotes anything to executable/go and never
fabricates joint coverage. A genuinely-absent exposure or outcome (no concept
in any dictionary and no ICD definition) stays ``db-cannot-do`` (fail-closed).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

# Real (non-demo) databases EasyICU supports.
_REAL_DBS = ("miiv", "mimic", "eicu", "aumc", "hirid", "sic")

# Tokens that carry no disease/population signal -- dropped before ICD matching
# so "severe Legionnaires' disease" matches on "legionnaires", not "severe".
_COHORT_STOPWORDS = {
    "the",
    "of",
    "in",
    "and",
    "with",
    "for",
    "to",
    "a",
    "an",
    "or",
    "on",
    "patients",
    "patient",
    "critically",
    "ill",
    "icu",
    "disease",
    "diseases",
    "care",
    "intensive",
    "unit",
    "acquired",
    "adult",
    "adults",
    "severe",
    "acute",
    "chronic",
    "suspected",
    "risk",
    "who",
    "underwent",
    "their",
    "admission",
    "admitted",
    "complicated",
    "study",
    "population",
    "cohort",
}

# Cohorts that have no single coherent ICD family (broad umbrella categories):
# we may still propose codes, but flag low confidence -> human must curate.
_BROAD_CATEGORY_TOKENS = {
    "autoimmune",
    "immunocompromised",
    "immunosuppressed",
    "malignancy",
    "comorbidity",
    "comorbidities",
    "frailty",
    "polypharmacy",
}

# Diagnoses known to be severely under-coded in administrative ICD data; an
# ICD-based cohort/outcome for these is an unreliable proxy.
_UNDERCODED_PATTERNS = (
    "weakness",
    "myopathy",
    "polyneuropathy",
    "critical illness neuropathy",
    "icu-acquired",
    "delirium",
    "encephalopathy",
    "dysphagia",
    "post-intensive care",
    "frailty",
)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _content_tokens(text: str) -> FrozenSet[str]:
    return frozenset(t for t in _tokens(text) if t not in _COHORT_STOPWORDS)


@dataclass(frozen=True)
class CrossDbConceptHit:
    """A blocking construct that resolves to a dictionary concept somewhere."""

    term: str
    concept: str
    databases: FrozenSet[str]
    in_current_db: bool


@dataclass(frozen=True)
class IcdCohortProposal:
    """A proposed ICD code-set for a population, for HUMAN confirmation."""

    population: str
    matched_codes: Tuple[Tuple[str, str, int], ...]  # (icd_code, title, n_hadm)
    family_prefixes: Tuple[str, ...]
    total_hadm: int
    confident: bool
    reliability: str  # "review_required" | "needs_curation" | "unreliable_undercoded"
    requires_human_confirm: bool = True


@dataclass
class ExtendedFeasibilityVerdict:
    decision: str  # always "hold" here
    reason: str
    case: str  # "icd_cohort" | "reextract_current_db" | "other_db"
    metadata: Dict[str, object] = field(default_factory=dict)


class ExtendedFeasibilityIndex:
    """Reconsiders ``db-cannot-do`` verdicts for Case 1 / Case 2."""

    def __init__(
        self,
        *,
        current_db: str,
        concept_db_map: Mapping[str, FrozenSet[str]],
        full_lookup: Mapping[str, str],
        export_concepts: FrozenSet[str],
        icd_codes: Sequence[Tuple[str, int, str, int]],
    ) -> None:
        self.current_db = current_db
        self._concept_db_map = dict(concept_db_map)
        self._full_lookup = dict(full_lookup)
        self._export_concepts = frozenset(export_concepts)
        # ICD title index: (content_tokens, code, title, n_hadm)
        self._icd_index: List[Tuple[FrozenSet[str], str, str, int]] = [
            (_content_tokens(title), code, title, n_hadm)
            for code, _ver, title, n_hadm in icd_codes
        ]
        # document frequency of each token across ICD titles -> pick the rarest
        # (most specific) token of a population as the required "anchor".
        df: Dict[str, int] = {}
        for toks, _c, _t, _n in self._icd_index:
            for tok in toks:
                df[tok] = df.get(tok, 0) + 1
        self._token_df = df

    # ---- construction -------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        current_db: str,
        available_concepts: Sequence[object],
        data_dir: Optional[Path] = None,
        icd_catalog_path: Optional[Path] = None,
        concept_aliases: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> "ExtendedFeasibilityIndex":
        from .idea_mining import _build_concept_lookup  # lazy: avoid import cycle
        from .idea_mining import normalize_concept_name

        if data_dir is None:
            import easyicu

            data_dir = Path(easyicu.__file__).resolve().parent / "data"

        concept_db_map: Dict[str, FrozenSet[str]] = {}
        for fname in ("concept-dict.json", "sofa2-dict.json"):
            fp = Path(data_dir) / fname
            if not fp.exists():
                continue
            raw = json.loads(fp.read_text(encoding="utf-8"))
            for name, defn in raw.items():
                if not isinstance(defn, dict):
                    continue
                srcs = defn.get("sources")
                dbs = (
                    {db for db in srcs if not db.endswith("_demo")}
                    if isinstance(srcs, dict)
                    else set()
                )
                if not dbs:
                    continue
                canon = normalize_concept_name(name)
                concept_db_map[canon] = frozenset(dbs) | concept_db_map.get(
                    canon, frozenset()
                )

        full_lookup = _build_concept_lookup(
            list(concept_db_map.keys()), concept_aliases=concept_aliases
        )

        export_concepts = set()
        for item in available_concepts:
            name = (
                getattr(item, "source_concept", None)
                or getattr(item, "name", None)
                or str(item)
            )
            export_concepts.add(normalize_concept_name(str(name)))

        icd_codes: List[Tuple[str, int, str, int]] = []
        if icd_catalog_path is None:
            icd_catalog_path = (
                Path(easyicu.__file__).resolve().parent.parent.parent
                / "benchmark"
                / "icd_cohort_catalog_miiv.json"
            )
        if Path(icd_catalog_path).exists():
            cat = json.loads(Path(icd_catalog_path).read_text(encoding="utf-8"))
            for rec in cat.get("codes", []):
                icd_codes.append(
                    (
                        str(rec["icd_code"]),
                        int(rec.get("icd_version", 0)),
                        str(rec["long_title"]),
                        int(rec.get("n_hadm", 0)),
                    )
                )

        return cls(
            current_db=current_db,
            concept_db_map=concept_db_map,
            full_lookup=full_lookup,
            export_concepts=frozenset(export_concepts),
            icd_codes=icd_codes,
        )

    # ---- Case 2: dictionary / cross-DB reachability -------------------
    def resolve_construct_cross_db(self, term: str) -> Optional[CrossDbConceptHit]:
        if not term or not str(term).strip():
            return None
        from .idea_mining import _resolve_concept  # lazy

        concept = _resolve_concept(str(term), self._full_lookup)
        if concept is None:
            return None
        if concept in self._export_concepts:
            return None  # already available in the current export -> not blocking
        dbs = self._concept_db_map.get(concept)
        if not dbs:
            return None
        return CrossDbConceptHit(
            term=str(term),
            concept=concept,
            databases=dbs,
            in_current_db=self.current_db in dbs,
        )

    # ---- Case 1: ICD-derivable cohort --------------------------------
    def propose_cohort_icd(self, population: str) -> Optional[IcdCohortProposal]:
        content = _content_tokens(population)
        if not content or not self._icd_index:
            return None
        # Anchor on the rarest (most specific) content token: it MUST appear in
        # the title (so "severe Legionnaires' disease" anchors on "legionnaires",
        # never on "severe"). Then require at least half of the content tokens to
        # match, so extra population descriptors ("hospitalized") don't block a
        # real disease match.
        known = [t for t in content if t in self._token_df]
        if not known:
            return None
        anchor = min(known, key=lambda t: self._token_df[t])
        need = max(1, (len(content) + 1) // 2)
        matches = [
            (code, title, n)
            for (toks, code, title, n) in self._icd_index
            if anchor in toks and len(content & toks) >= need
        ]
        if not matches:
            return None
        matches.sort(key=lambda x: x[2], reverse=True)
        prefixes = tuple(sorted({c[:3] for c, _t, _n in matches}))
        total = sum(n for _c, _t, n in matches)

        pop_l = population.lower()
        undercoded = any(p in pop_l for p in _UNDERCODED_PATTERNS)
        broad = bool(content & _BROAD_CATEGORY_TOKENS)
        # confident: a coherent definition exists -- either >=2 specific content
        # tokens all matched, or a single specific (non-broad) disease token.
        confident = (
            (not broad)
            and (not undercoded)
            and (len(content) >= 2 or len(matches) <= 40)
        )
        reliability = (
            "unreliable_undercoded"
            if undercoded
            else ("needs_curation" if (broad or not confident) else "review_required")
        )
        return IcdCohortProposal(
            population=population,
            matched_codes=tuple(matches[:25]),
            family_prefixes=prefixes,
            total_hadm=total,
            confident=confident,
            reliability=reliability,
        )

    # ---- orchestration ------------------------------------------------
    def reconsider(
        self, *, idea: object, candidate: Optional[object]
    ) -> Optional[ExtendedFeasibilityVerdict]:
        """Return a hold-downgrade for a db-cannot-do verdict, or None.

        Only downgrades; never promotes to executable. Exposure/outcome that
        resolve nowhere stay db-cannot-do.
        """
        # Gather the blocking constructs from the literature idea.
        pred = getattr(idea, "exposure_or_predictor", "") or ""
        out = getattr(idea, "outcome", "") or ""
        analysis = list(getattr(idea, "analysis_concepts", []) or [])
        population = getattr(idea, "population", "") or ""

        # Case 2 first: a construct that IS a dictionary concept (re-extract /
        # other-db) is the strongest actionable route.
        cross_hits: List[CrossDbConceptHit] = []
        for term in [pred, out, *analysis]:
            hit = self.resolve_construct_cross_db(term)
            if hit is not None:
                cross_hits.append(hit)
        if cross_hits:
            in_cur = [h for h in cross_hits if h.in_current_db]
            if in_cur:
                names = ", ".join(sorted({h.concept for h in in_cur}))
                return ExtendedFeasibilityVerdict(
                    decision="hold",
                    reason=(
                        f"construct(s) are dictionary concepts for {self.current_db} "
                        f"but absent from the current export ({names}); agent should "
                        f"re-extract them first, then re-probe feasibility"
                    ),
                    case="reextract_current_db",
                    metadata={
                        "concepts": names,
                        "hits": [
                            {
                                "term": h.term,
                                "concept": h.concept,
                                "databases": sorted(h.databases),
                            }
                            for h in in_cur
                        ],
                    },
                )
            # only available in other databases
            other = cross_hits
            dbs = sorted({db for h in other for db in h.databases})
            names = ", ".join(sorted({h.concept for h in other}))
            return ExtendedFeasibilityVerdict(
                decision="hold",
                reason=(
                    f"construct(s) are dictionary concepts only for other databases "
                    f"({names} available in {', '.join(dbs)}); executable there after "
                    f"extraction, not on {self.current_db}"
                ),
                case="other_db",
                metadata={
                    "concepts": names,
                    "databases": dbs,
                    "hits": [
                        {
                            "term": h.term,
                            "concept": h.concept,
                            "databases": sorted(h.databases),
                        }
                        for h in other
                    ],
                },
            )

        # Case 1: the cohort/population is ICD-derivable. Only meaningful when the
        # exposure/outcome are NOT themselves the unrecoverable blocker -- but we
        # surface it regardless as a human-confirm route for the population, with
        # explicit QC flags. We do NOT claim executability.
        proposal = self.propose_cohort_icd(population)
        if proposal is not None:
            note = (
                "cohort is ICD-derivable; propose a candidate code-set for HUMAN "
                "confirmation (verify ICD accuracy before use)"
            )
            if proposal.reliability == "unreliable_undercoded":
                note += "; WARNING: this diagnosis is under-coded in ICD -- proxy unreliable"
            elif not proposal.confident:
                note += "; broad/uncertain category -- human must curate the code-set"
            return ExtendedFeasibilityVerdict(
                decision="hold",
                reason=note,
                case="icd_cohort",
                metadata={
                    "population": population,
                    "family_prefixes": list(proposal.family_prefixes),
                    "total_hadm": proposal.total_hadm,
                    "confident": proposal.confident,
                    "reliability": proposal.reliability,
                    "requires_human_confirm": True,
                    "example_codes": [
                        {"icd_code": c, "title": t, "n_hadm": n}
                        for c, t, n in proposal.matched_codes[:8]
                    ],
                },
            )
        return None
