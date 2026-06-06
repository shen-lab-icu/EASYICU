"""Dictionary-driven concept catalog for idea-mining.

Why this exists
---------------
The idea-mining dry run needs three pieces of metadata to turn a literature
phrase into an executable hypothesis:

1. **available concepts** — what the database actually carries;
2. **concept aliases** — a map from concept *keys* (e.g. ``norepi_rate``) to the
   *literature phrasing* a paper would use (e.g. "norepinephrine"), so the
   resolver can bind an LLM-written exposure/outcome to a concept;
3. **outcome determinability** — which concepts are 0/1-determinable outcomes
   (mortality, RRT, AKI ...) so the feasibility probe can use them.

In early upper-bound runs all three were hand-coded per benchmark. They do not
need to be: EasyICU's own concept dictionaries already encode them. Each concept
carries a human-readable ``description`` (the alias source), a ``category`` and
a ``class_name`` (the outcome-type signal), and ``unit``/``min``/``max`` (the
"this is a numeric, not a logical concept" signal). This module derives the
three layers from ``concept-dict.json`` + ``sofa2-dict.json`` so the discovery
pipeline is dictionary-backed instead of hand-curated.

Case-neutrality
---------------
Everything here is derived from the shared dictionaries plus a small table of
EasyICU's own derived concepts (KDIGO AKI, circulatory failure, sepsis-3, urine
output rates) that live in code rather than the JSON dictionaries. No single
benchmark case / disease / score is privileged; ``extra_aliases`` is the
caller-supplied escape hatch for benchmark-local synonyms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .concept_availability import normalize_concept_name

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DICTS = ("concept-dict.json", "sofa2-dict.json")

# Generic trailing/structural words to strip when deriving a short alias from a
# concept description (so "norepinephrine rate" also yields "norepinephrine").
_ALIAS_STRIP_WORDS = {
    "rate", "dur", "duration", "dose", "dosage", "in", "use", "windows",
    "window", "indicator", "ind", "level", "score", "value", "measurement",
    "of", "the", "per", "total", "cumulative", "current",
}

# EasyICU derived concepts that are produced in code (KDIGO AKI, circulatory
# failure, sepsis-3, urine-output rates) and are absent from the JSON
# dictionaries. (aliases, is_binary_outcome). Stable EasyICU domain knowledge.
DERIVED_CONCEPT_HINTS: Dict[str, Tuple[List[str], bool]] = {
    "aki": (["acute kidney injury", "AKI"], True),
    "aki_stage": (["AKI stage", "KDIGO stage", "acute kidney injury stage"], True),
    "aki_stage_creat": (["AKI stage by creatinine"], True),
    "aki_stage_rrt": (
        ["AKI requiring renal replacement therapy", "AKI requiring dialysis"],
        True,
    ),
    "aki_stage_uo": (["AKI stage by urine output"], True),
    "circ_event": (["circulatory event"], True),
    "circ_failure": (["circulatory failure", "shock"], True),
    "creat_low_past_48hr": (["baseline creatinine past 48 hours"], False),
    "creat_low_past_7day": (["baseline creatinine past 7 days"], False),
    "sep3_sofa1": (["sepsis", "sepsis-3", "septic shock"], True),
    "uo_rt_6hr": (["urine output rate 6 hours"], False),
    "uo_rt_12hr": (["urine output rate 12 hours"], False),
    "uo_rt_24hr": (["urine output rate 24 hours"], False),
}

# Universal medical-terminology equivalences: UK/US spelling, common brand and
# abbreviation variants. NOT benchmark-case specific — a group is merged into a
# concept's aliases only when one of its phrases already matches that concept's
# dictionary-derived alias, so no group is hard-bound to a concept key. This
# generalises across the whole dictionary (any concept whose description names
# "norepinephrine" automatically also answers to "noradrenaline", etc.).
SYNONYM_GROUPS: Tuple[frozenset, ...] = (
    frozenset({"norepinephrine", "noradrenaline"}),
    frozenset({"epinephrine", "adrenaline"}),
    frozenset({"vasopressin", "antidiuretic hormone", "arginine vasopressin", "argipressin"}),
    frozenset({"phenylephrine", "neosynephrine"}),
    frozenset({"dobutamine", "dobutrex"}),
    frozenset({"dopamine", "intropin"}),
    frozenset({"milrinone", "primacor"}),
    frozenset({
        "renal replacement therapy", "dialysis", "hemodialysis", "haemodialysis",
        "continuous renal replacement therapy", "crrt", "hemofiltration",
        "haemofiltration", "cvvh", "cvvhd",
    }),
    frozenset({"mechanical ventilation", "invasive ventilation", "invasive mechanical ventilation"}),
    frozenset({"positive end-expiratory pressure", "positive end expiratory pressure", "peep"}),
    frozenset({"horowitz index", "p/f ratio", "pf ratio", "pao2/fio2", "pao2/fio2 ratio", "oxygenation index"}),
    frozenset({"sao2/fio2", "s/f ratio", "sf ratio"}),
    frozenset({"rdw", "red cell distribution width", "red blood cell distribution width"}),
    frozenset({
        "nlr", "neutrophil lymphocyte ratio", "neutrophil-lymphocyte ratio",
        "neutrophil-to-lymphocyte ratio", "neutrophil to lymphocyte ratio",
    }),
    frozenset({"acetaminophen", "paracetamol"}),
    frozenset({"furosemide", "frusemide"}),
    frozenset({"acute kidney injury", "acute renal failure"}),
    frozenset({"glasgow coma scale", "gcs", "glasgow coma score"}),
)

# Words that signal a parenthetical fragment is a description, dosing, route, or
# formula clause rather than a usable synonym/brand name.
_BRACKET_NOISE = {
    "route", "include", "includes", "including", "correction", "administration",
    "resuscitation", "edema", "oedema", "covid", "compatibility", "alias",
    "etc", "bolus", "infusion", "continuous", "colloid", "antiplatelet",
    "inhibitor", "antiarrhythmic", "blocker", "sedation", "any", "via", "such",
    "other", "iv", "po", "oral", "intravenous", "and", "or", "for", "with",
}


@dataclass(frozen=True)
class ConceptCatalog:
    """Dictionary-derived metadata for the idea-mining feasibility probe."""

    available_concepts: Tuple[str, ...]
    concept_aliases: Dict[str, List[str]] = field(default_factory=dict)
    # plain dicts: run_idea_mining_dry_run accepts Mapping[str, Mapping] here,
    # so we avoid importing the (in-flux) idea_mining module.
    outcome_determinability: Dict[str, Dict[str, str]] = field(default_factory=dict)


def _load_dicts(dict_paths: Optional[Sequence[str | Path]]) -> Dict[str, dict]:
    merged: Dict[str, dict] = {}
    for name in dict_paths or _DEFAULT_DICTS:
        path = Path(name)
        if not path.is_absolute():
            path = _DATA_DIR / path
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    merged.setdefault(key, value)
    return merged


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _aliases_from_description(description: str) -> List[str]:
    desc = _clean(description)
    if not desc:
        return []
    low = desc.lower()
    aliases: List[str] = [low]
    tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
    core = [t for t in tokens if t not in _ALIAS_STRIP_WORDS]
    if core and core != tokens:
        aliases.append(" ".join(core))
    # mortality phrasings: "in hospital mortality" -> "hospital mortality", "mortality"
    if "mortality" in tokens:
        aliases.append("mortality")
        aliases.extend(
            [
                "ICU mortality",
                "intensive care unit mortality",
                "intensive-care unit mortality",
                "intensive care mortality",
                "critical care mortality",
            ]
        )
        idx = tokens.index("mortality")
        if idx > 0 and tokens[idx - 1] not in _ALIAS_STRIP_WORDS:
            aliases.append(f"{tokens[idx - 1]} mortality")
        if "hospital" in tokens:
            aliases.append("in-hospital mortality")
    aliases.extend(_bracket_aliases(desc))
    return _dedup(a for a in aliases if len(a) > 2)


def _bracket_aliases(description: str) -> List[str]:
    """Extract usable synonyms/brand names/mode acronyms from parentheses.

    "furosemide (Lasix)" -> "Lasix"; "dexmedetomidine (Precedex/Dexdor)" ->
    "Precedex", "Dexdor"; "(IMV/NIV/HFNC/CPAP/BiPAP)" -> each mode. Dosing,
    routes, percentages, and formulae ("(5%, 20%)", "(Na - Cl - HCO3)",
    "(any route)") are filtered out: a fragment must be purely alphabetic and
    carry no noise word.
    """
    out: List[str] = []
    for inner in re.findall(r"\(([^)]+)\)", description):
        for part in re.split(r"[/,;]", inner):
            name = part.strip().strip(".")
            if len(name) < 3 or not re.fullmatch(r"[A-Za-z][A-Za-z \-]+", name):
                continue
            if any(word in _BRACKET_NOISE for word in name.lower().split()):
                continue
            out.append(name)
    return out


def _expand_synonyms(aliases: List[str]) -> List[str]:
    """Merge a universal synonym group into the alias list when any of its
    phrases already matches one of the concept's dictionary-derived aliases."""
    if not aliases:
        return aliases
    lower = {a.lower() for a in aliases}
    extra: List[str] = []
    for group in SYNONYM_GROUPS:
        if lower & group:
            extra.extend(sorted(group))
    return _dedup([*aliases, *extra]) if extra else aliases


def _drop_ambiguous_outcome_aliases(
    aliases: Dict[str, List[str]],
    outcome_keys: Iterable[str],
) -> Dict[str, List[str]]:
    """Suppress aliases that would bind one outcome phrase to multiple outcomes."""
    guarded = set(outcome_keys)
    if not guarded:
        return aliases
    owners: Dict[str, set[str]] = {}
    for key, key_aliases in aliases.items():
        if key not in guarded:
            continue
        for alias in key_aliases:
            owners.setdefault(_clean(alias).lower(), set()).add(key)
    ambiguous = {alias for alias, keys in owners.items() if len(keys) > 1}
    if not ambiguous:
        return aliases
    return {
        key: [
            alias
            for alias in key_aliases
            if key not in guarded or _clean(alias).lower() not in ambiguous
        ]
        for key, key_aliases in aliases.items()
    }


def _dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = _clean(item).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(_clean(item))
    return out


def _is_binary_outcome(concept: Mapping) -> bool:
    """Return whether a dictionary concept is safe as a 0/1 outcome.

    Logical concepts outside the outcome category can be valid exposures or
    eligibility variables, but should not automatically pass the idea-mining
    outcome gate.
    """
    if any(k in concept for k in ("unit", "min", "max")):
        return False  # numeric / continuous
    desc = str(concept.get("description", "")).lower()
    if "score" in desc or "component" in desc:
        return False  # ordinal severity scale, not 0/1
    if any(word in desc for word in ("administration", "infusion", "dose", " rate")):
        return False  # treatment/exposure concept, not an outcome
    return concept.get("category") == "outcome"


def load_concept_catalog(
    *,
    restrict_to: Optional[Iterable[str]] = None,
    dict_paths: Optional[Sequence[str | Path]] = None,
    extra_aliases: Optional[Mapping[str, Sequence[str]]] = None,
) -> ConceptCatalog:
    """Build a :class:`ConceptCatalog` from the EasyICU concept dictionaries.

    Parameters
    ----------
    restrict_to:
        If given (e.g. a manifest's ``selected_concepts`` or a prepared cohort's
        columns), the catalog is limited to these keys — derived concepts not in
        the JSON dictionaries are still included via ``DERIVED_CONCEPT_HINTS``.
        If omitted, every dictionary concept is exposed.
    dict_paths:
        Override the default ``concept-dict.json`` + ``sofa2-dict.json``.
    extra_aliases:
        Caller-supplied benchmark-local synonyms, merged on top of the derived
        ones (escape hatch for phrasing the dictionary does not cover).
    """
    dicts = _load_dicts(dict_paths)

    if restrict_to is not None:
        keys = _dedup(str(k) for k in restrict_to)
    else:
        keys = _dedup([*dicts.keys(), *DERIVED_CONCEPT_HINTS.keys()])

    aliases: Dict[str, List[str]] = {}
    outcomes: Dict[str, Dict[str, str]] = {}

    def _declare_outcome(key: str) -> None:
        # The resolver canonicalizes concept keys (e.g. aki -> kdigo_aki), so we
        # register the determinability spec under BOTH the raw and canonical key
        # to guarantee the feasibility probe's lookup hits regardless of which
        # form the resolved outcome takes.
        norm = normalize_concept_name(key)
        spec = {"outcome": norm, "status": "known_0_1"}
        outcomes[key] = spec
        if norm and norm != key:
            outcomes[norm] = spec

    for key in keys:
        concept = dicts.get(key)
        if concept is not None:
            derived = _expand_synonyms(
                _aliases_from_description(concept.get("description", ""))
            )
            if derived:
                aliases[key] = derived
            if _is_binary_outcome(concept):
                _declare_outcome(key)
        elif key in DERIVED_CONCEPT_HINTS:
            hint_aliases, is_binary = DERIVED_CONCEPT_HINTS[key]
            if hint_aliases:
                aliases[key] = _expand_synonyms(_dedup(hint_aliases))
            if is_binary:
                _declare_outcome(key)

    if extra_aliases:
        for key, extra in extra_aliases.items():
            merged = [*aliases.get(str(key), []), *[str(e) for e in extra]]
            aliases[str(key)] = _dedup(merged)

    aliases = _drop_ambiguous_outcome_aliases(aliases, outcomes.keys())

    return ConceptCatalog(
        available_concepts=tuple(keys),
        concept_aliases=aliases,
        outcome_determinability=outcomes,
    )


__all__ = [
    "ConceptCatalog",
    "DERIVED_CONCEPT_HINTS",
    "load_concept_catalog",
]
