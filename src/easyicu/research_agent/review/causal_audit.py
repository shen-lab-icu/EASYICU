"""Causal-claim auditor (O18).

A deterministic guardrail that sits between the writer and the bound
manuscript. It answers one question for every primary effect the run
reported: *is this being sold as causal, and if so, does the
artefact carry the minimum support needed to do so?*

Policy
------

1. **Every primary effect registered in the run (OR / HR / risk
   difference / AUROC) is labelled.** A deterministic label is
   attached to each effect:

   * ``"associational"`` — the default. A *regression coefficient* is
     fit on observational ICU data with no explicit identification
     strategy (no DAG, no positivity diagnostic, no negative-control
     outcome, no E-value, no IPTW / TMLE / g-computation).
   * ``"causal_explicit"`` — the user / skill declared a causal
     estimand (IPTW, TMLE, g-computation, instrumental variable) and
     the run produced the required supporting artefacts.
   * ``"causal_overclaimed"`` — the user / skill declared causality
     but the artefacts are missing; this is an ``error`` finding and
     the manuscript cannot be bound without a downgrade.

2. **The writer output is scanned for causal language.** Words like
   ``cause``, ``effect of``, ``increases ... by``, ``reduces ... by``,
   ``attributable to`` are flagged. Matches against sentences that
   are themselves bound to an ``associational`` effect trigger a
   ``warning``. Matches against sentences bound to a
   ``causal_overclaimed`` effect trigger an ``error`` that blocks the
   bound manuscript from being marked final.

3. **Optional hook for DoWhy / causallib / EconML.** The module does
   not import any causal library. A user / skill may attach an
   ``identification_strategy`` dict to a registered statistic
   artefact; the auditor then requires the listed support artefacts
   (DAG, positivity diagnostic, negative control, E-value, sensitivity
   analysis) to be registered before it will upgrade the label. This
   is the extension point the future CausalSkill will plug into.

Nothing in this module imports pandas / sklearn / DoWhy. It operates
on :class:`EvidenceRecord` metadata and the bound manuscript string.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

# Words that make a claim causal. Ordered roughly from strongest to weakest.
# Each entry is (pattern, severity) where severity is applied on top of the
# effect label (associational → warning, causal_overclaimed → error).
_CAUSAL_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bcaus(?:e|es|ed|ing)\b", "strong"),
    (r"\bcausal\b", "strong"),
    (r"\battributable to\b", "strong"),
    (r"\beffect of\b", "moderate"),
    (r"\bdue to\b", "moderate"),
    (r"\bresulting in\b", "moderate"),
    (r"\blead(?:s|ing)? to\b", "moderate"),
    (r"\bdrives?\b", "moderate"),
    (r"\bimprove(?:s|d|ment of)\b", "weak"),
    (r"\breduce(?:s|d|ction of)\b", "weak"),
    (r"\bincrease(?:s|d) (?:[A-Za-z_]+ )?(?:by|to)\b", "moderate"),
    (r"\bdecrease(?:s|d) (?:[A-Za-z_]+ )?(?:by|to)\b", "moderate"),
)


# Evidence-metadata flag that a skill / user can set to declare the step
# was designed as a causal estimand. Present on
# ``EvidenceRecord.metadata["identification_strategy"]`` — a dict with
# at least a ``method`` (``"iptw"``, ``"tmle"``, ``"g_computation"``,
# ``"instrumental_variable"``, ``"did"``) and optionally a list of
# required support ids under ``"supporting_evidence_ids"``.
_SUPPORT_DEFAULTS: Dict[str, Tuple[str, ...]] = {
    "iptw": ("dag", "positivity_diagnostic", "negative_control", "e_value"),
    "tmle": ("dag", "positivity_diagnostic", "e_value"),
    "g_computation": ("dag", "positivity_diagnostic"),
    "instrumental_variable": ("first_stage_diagnostic", "exclusion_restriction"),
    "did": ("parallel_trends", "placebo_test"),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EffectLabel:
    """One row of the causal label table."""

    evidence_id: str
    artefact_path: str
    estimand: str  # "odds_ratio", "hazard_ratio", "risk_difference", "auroc", ...
    label: str  # "associational" | "causal_explicit" | "causal_overclaimed"
    rationale: str
    identification_strategy: Optional[str] = None
    missing_supports: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "artefact_path": self.artefact_path,
            "estimand": self.estimand,
            "label": self.label,
            "rationale": self.rationale,
            "identification_strategy": self.identification_strategy,
            "missing_supports": list(self.missing_supports),
        }


@dataclass
class CausalLanguageHit:
    """A causal-language match flagged in the bound manuscript."""

    sentence: str
    pattern: str
    strength: str
    severity: str  # "warning" | "error"
    linked_evidence_ids: List[str] = field(default_factory=list)
    linked_effect_labels: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "pattern": self.pattern,
            "strength": self.strength,
            "severity": self.severity,
            "linked_evidence_ids": list(self.linked_evidence_ids),
            "linked_effect_labels": list(self.linked_effect_labels),
        }


@dataclass
class CausalAuditReport:
    """Full result of the run-wide causal audit."""

    effect_labels: List[EffectLabel] = field(default_factory=list)
    language_hits: List[CausalLanguageHit] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        n_eff = len(self.effect_labels)
        assoc = sum(1 for e in self.effect_labels if e.label == "associational")
        explicit = sum(1 for e in self.effect_labels if e.label == "causal_explicit")
        overclaim = sum(1 for e in self.effect_labels if e.label == "causal_overclaimed")
        warnings_ = sum(1 for h in self.language_hits if h.severity == "warning")
        errors = sum(1 for h in self.language_hits if h.severity == "error")
        return {
            "n_effects_labelled": n_eff,
            "n_associational": assoc,
            "n_causal_explicit": explicit,
            "n_causal_overclaimed": overclaim,
            "n_language_warnings": warnings_,
            "n_language_errors": errors,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "effect_labels": [e.to_json() for e in self.effect_labels],
                    "language_hits": [h.to_json() for h in self.language_hits],
                    "summary": self.summary(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    def write_markdown(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        lines = [
            "# Causal-claim audit (O18)",
            "",
            f"- Effects labelled: **{s['n_effects_labelled']}**",
            f"- Associational (default): **{s['n_associational']}**",
            f"- Causal, explicit strategy: **{s['n_causal_explicit']}**",
            f"- Causal, overclaimed (missing support): **{s['n_causal_overclaimed']}**",
            f"- Language warnings (causal prose over associational effect): **{s['n_language_warnings']}**",
            f"- Language errors (causal prose over overclaimed effect): **{s['n_language_errors']}**",
        ]
        if self.effect_labels:
            lines += [
                "",
                "## Effect labels",
                "",
                "| evidence | estimand | label | strategy | missing support |",
                "|---|---|---|---|---|",
            ]
            for e in self.effect_labels:
                lines.append(
                    "| {evid} | {est} | {lab} | {strat} | {miss} |".format(
                        evid=e.evidence_id,
                        est=e.estimand,
                        lab=e.label,
                        strat=e.identification_strategy or "",
                        miss=", ".join(e.missing_supports) or "—",
                    )
                )
        if self.language_hits:
            lines += [
                "",
                "## Causal-language hits",
                "",
                "| severity | pattern | sentence |",
                "|---|---|---|",
            ]
            for h in self.language_hits:
                lines.append(
                    "| {sev} | `{pat}` | {sent} |".format(
                        sev=h.severity,
                        pat=h.pattern,
                        sent=(h.sentence[:120].replace("|", "/")).strip(),
                    )
                )
        else:
            lines += [
                "",
                "No causal language was detected in the bound manuscript.",
            ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Effect labelling
# ---------------------------------------------------------------------------


_ESTIMAND_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("primary_association", "odds_ratio"),
    ("logistic", "odds_ratio"),
    ("odds_ratio", "odds_ratio"),
    ("cox", "hazard_ratio"),
    ("hazard_ratio", "hazard_ratio"),
    ("risk_difference", "risk_difference"),
    ("auroc", "auroc"),
    ("calibration", "calibration_slope"),
    ("prediction", "auroc"),
)


def _guess_estimand(evidence_id: str, description: str) -> str:
    text = (evidence_id + " " + description).lower()
    for needle, estimand in _ESTIMAND_KEYWORDS:
        if needle in text:
            return estimand
    return "other"


def label_effects(
    *, evidence_records: Iterable[Any], run_dir: Path
) -> List[EffectLabel]:
    """Classify every effect artefact into associational vs causal.

    ``evidence_records`` is ``EvidenceStore.records()``. Only records
    with kind ``table`` / ``statistic`` whose id or description looks
    like an effect estimate are considered; descriptive tables (Table
    One, missingness profiles) are ignored.
    """
    labels: List[EffectLabel] = []
    available_ids = set()
    for rec in evidence_records:
        available_ids.add(getattr(rec, "evidence_id", None))
    for rec in evidence_records:
        kind = getattr(rec, "kind", None)
        if kind not in {"table", "statistic"}:
            continue
        evidence_id = getattr(rec, "evidence_id", "") or ""
        desc = getattr(rec, "description", "") or ""
        estimand = _guess_estimand(evidence_id, desc)
        if estimand == "other":
            # Not an effect estimate; skip without label.
            continue
        metadata = getattr(rec, "metadata", {}) or {}
        id_strategy = metadata.get("identification_strategy")
        if not id_strategy:
            labels.append(
                EffectLabel(
                    evidence_id=evidence_id,
                    artefact_path=getattr(rec, "relative_path", "") or "",
                    estimand=estimand,
                    label="associational",
                    rationale=(
                        "No identification_strategy metadata on the effect artefact; "
                        "treated as an association on observational data."
                    ),
                )
            )
            continue
        method = str(id_strategy.get("method", "")).lower()
        required = tuple(
            id_strategy.get("supporting_evidence_ids")
            or _SUPPORT_DEFAULTS.get(method, ())
        )
        missing = [r for r in required if r not in available_ids]
        if missing:
            labels.append(
                EffectLabel(
                    evidence_id=evidence_id,
                    artefact_path=getattr(rec, "relative_path", "") or "",
                    estimand=estimand,
                    label="causal_overclaimed",
                    rationale=(
                        f"identification_strategy={method} declared but required "
                        f"support artefacts are missing: {sorted(missing)}."
                    ),
                    identification_strategy=method,
                    missing_supports=sorted(missing),
                )
            )
        else:
            labels.append(
                EffectLabel(
                    evidence_id=evidence_id,
                    artefact_path=getattr(rec, "relative_path", "") or "",
                    estimand=estimand,
                    label="causal_explicit",
                    rationale=(
                        f"identification_strategy={method} with all required "
                        "support artefacts registered."
                    ),
                    identification_strategy=method,
                )
            )
    return labels


# ---------------------------------------------------------------------------
# Manuscript scan
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_EVIDENCE_ID_RE = re.compile(r"\{evidence:([^}]+)\}")


def _iter_sentences(text: str) -> Iterable[str]:
    # Drop markdown code fences and HTML comments so scan doesn't fire on them.
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.S)
    for raw in _SENTENCE_SPLIT_RE.split(cleaned):
        s = raw.strip()
        if s:
            yield s


def scan_manuscript_for_causal_language(
    *, bound_manuscript: str, effect_labels: Sequence[EffectLabel]
) -> List[CausalLanguageHit]:
    """Find causal-language sentences and grade them against effect labels."""
    label_by_id = {e.evidence_id: e.label for e in effect_labels}
    hits: List[CausalLanguageHit] = []
    for sentence in _iter_sentences(bound_manuscript):
        linked_ids = _EVIDENCE_ID_RE.findall(sentence)
        for pattern, strength in _CAUSAL_PATTERNS:
            if not re.search(pattern, sentence, flags=re.IGNORECASE):
                continue
            linked_labels = [label_by_id.get(i) for i in linked_ids if label_by_id.get(i)]
            # Default severity is warning. Escalate to error only if the
            # sentence cites a causal_overclaimed effect.
            severity = "warning"
            if any(lbl == "causal_overclaimed" for lbl in linked_labels):
                severity = "error"
            elif linked_labels and all(lbl == "causal_explicit" for lbl in linked_labels):
                # Explicit causal estimand + causal language is the
                # happy path; we record it as info only.
                severity = "info"
            if severity == "info":
                # Skip recording — happy path is not worth manuscript clutter.
                continue
            if not linked_labels and strength == "weak":
                # Unlinked ``improves``/``reduces`` phrases without an
                # effect citation are too noisy to flag as warnings.
                continue
            hits.append(
                CausalLanguageHit(
                    sentence=sentence,
                    pattern=pattern,
                    strength=strength,
                    severity=severity,
                    linked_evidence_ids=list(dict.fromkeys(linked_ids)),
                    linked_effect_labels=list(
                        dict.fromkeys(label for label in linked_labels if label)
                    ),
                )
            )
            # Only record the strongest match per sentence; the first match
            # from the ordered pattern list wins.
            break
    return hits


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_causal_audit(
    *,
    evidence_records: Iterable[Any],
    run_dir: Path,
    bound_manuscript: str,
) -> CausalAuditReport:
    """One-call driver that returns a full :class:`CausalAuditReport`."""
    labels = label_effects(evidence_records=evidence_records, run_dir=run_dir)
    hits = scan_manuscript_for_causal_language(
        bound_manuscript=bound_manuscript, effect_labels=labels,
    )
    return CausalAuditReport(effect_labels=labels, language_hits=hits)


__all__ = [
    "CausalAuditReport",
    "CausalLanguageHit",
    "EffectLabel",
    "label_effects",
    "run_causal_audit",
    "scan_manuscript_for_causal_language",
]
