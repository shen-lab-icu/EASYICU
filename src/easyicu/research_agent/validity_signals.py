"""Gold-free, kind-routed deterministic result-validity signals.

Implements the manuscript's Fig.3 evaluation redesign (`Fig3_评估重设_E1H3...`):
``result_validity`` does NOT depend on a frozen numeric reference (an optional
future bonus). The baseline is a set of **deterministic, per-task-kind validity
checks** read from a run's artifacts — the same "deterministic, no-LLM-judge"
stance that differentiates EasyICU.

CRITICAL design rule (the difference between this and a *reporting* score):
a validity signal must read an actual VALUE and judge correctness against a
*standard / objective* threshold — NOT merely check that an artifact with the
right name exists. "There is a balance table" is reporting; "max |SMD| < 0.1, so
covariates ARE balanced" is validity. A pure presence check belongs in
``reporting_completeness``, not here.

Consequences enforced below:
* Each signal returns ``pass`` (done, and correct), ``fail`` (a central
  methodological check was done wrong OR an objective requirement is absent), or
  ``na`` (genuinely not applicable, or present-but-unreadable). ``na`` is
  excluded from the score.
* Only the small set of checks that are (i) value-readable and (ii) *central* to
  the result's validity for that kind are scored. Kinds without such an
  artifact-readable check stay unscored (``[]`` → ``None`` subscore) — honest,
  not a fabricated pass. Threshold-free quality judgments (clustering silhouette
  "good enough") are deliberately NOT scored: the degenerate case is a Fail via
  the phenotype *teeth* upstream; a non-degenerate one stays NA.
* The subscore is graded (``passes / (passes + fails)``), so an objective check
  that was skipped or failed pulls the dimension below Full rather than collapsing
  to a 1.0-or-None binary.

Impartiality ([[feedback_rules_must_be_impartial]]): only objective requirements
(no train/test leakage, covariate balance for a causal estimate, positivity for a
weighted estimand) gate. Defensible analytical choices are never failed.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .authority.runtime_artifacts import (
    current_run_evidence_paths,
    current_successful_step_records,
    load_run_artifact_authority,
)

__all__ = [
    "ValiditySignal",
    "assess_validity_signals",
    "validity_positive_subscore",
    "BALANCE_SMD_THRESHOLD",
]

# Standard "good covariate balance" threshold (Austin 2009; widely used in the
# ICU causal-inference literature). Objective, not arbitrary — so failing it is a
# defensible Fail, unlike a clustering silhouette which has no consensus cutoff.
BALANCE_SMD_THRESHOLD = 0.1


@dataclass(frozen=True)
class ValiditySignal:
    name: str
    status: str  # "pass" | "fail" | "na"
    detail: str = ""


# ---------------------------------------------------------------------------
# Artifact readers
# ---------------------------------------------------------------------------


def _iter_step_summaries(run_dir: Path) -> List[Mapping[str, object]]:
    authority = load_run_artifact_authority(run_dir)
    if authority is not None:
        records = authority.get("per_step_records")
        records = records if isinstance(records, list) else []
        return [
            summary
            for record in current_successful_step_records(records)
            if isinstance((summary := record.get("step_summary")), Mapping)
        ]
    out: List[Mapping[str, object]] = []
    for pat in ("steps/*/outputs/step_summary.json", "steps/*/step_summary.json"):
        for p in run_dir.glob(pat):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(doc, dict):
                out.append(doc)
    return out


def _summary_get(summaries: Sequence[Mapping[str, object]], *keys: str) -> object:
    for doc in summaries:
        for k in keys:
            if k in doc and doc[k] is not None:
                return doc[k]
    return None


def _find_artifacts(run_dir: Path, *substrings: str) -> List[Path]:
    subs = tuple(s.lower() for s in substrings)
    hits: List[Path] = []
    current_paths = current_run_evidence_paths(run_dir)
    candidates = current_paths if current_paths is not None else run_dir.rglob("*")
    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in {".csv", ".json"}:
            continue
        name = p.name.lower()
        if any(s in name for s in subs):
            hits.append(p)
    return hits


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, ValueError):
        return []


def _floats(rows: Sequence[Mapping[str, str]], *col_aliases: str) -> List[float]:
    aliases = {a.lower() for a in col_aliases}
    out: List[float] = []
    for r in rows:
        for k, v in r.items():
            if (k or "").strip().lower() in aliases and v not in (None, ""):
                try:
                    out.append(abs(float(v)))
                except (TypeError, ValueError):
                    pass
                break
    return out


# ---------------------------------------------------------------------------
# Per-kind detectors — VALUE-based correctness, not artifact presence
# ---------------------------------------------------------------------------


# An ID-level split unit (patient or stay/admission) is leak-free at the patient
# level either directly (patient/subject) or when the cohort is one-stay-per-patient
# (stay/admission). A row/observation-level split mixes a patient across train and
# test → leakage.
_PATIENT_UNIT_TOKENS = ("patient", "subject", "uniquepid")
_STAY_UNIT_TOKENS = ("stay", "icustay", "hadm", "admission", "encounter")
# Tokens for an explicit row/observation-level (leaky) split. NOTE: bare "row"
# is deliberately NOT here — it false-matches granularity attestations like
# "one row per stay" / "the file is one row per patient" (which describe table
# shape, not a row-level split unit) and wrongly failed M2's stay-level split.
_ROW_UNIT_TOKENS = ("observation", "record", "row-level", "row level", "row_level")

# The split unit is most reliably read from the phrase describing what the split
# was performed ON ("split on stay_id", "grouped by subject_id") or the
# "<unit>-level split" form — not from justification prose elsewhere in the
# sentence ("interpreted as patient-separated", "one row per stay"), which
# otherwise contaminates a simple token scan.
_SPLIT_CONTEXT_RE = re.compile(
    r"(?:split|splitting|partition\w*|hold[-\s]?out|grouped|group|stratif\w*)"
    r"[^.;]*?\b(?:on|by|using|per|across|at)\s+"
    r"(?:the\s+|each\s+|every\s+|unique\s+|distinct\s+|individual\s+|patient[-\s]?level\s+)*"
    r"([a-z][a-z_]*)",
    re.IGNORECASE,
)
_LEVEL_CONTEXT_RE = re.compile(r"\b([a-z][a-z_]*)[-\s](?:level|based)\b", re.IGNORECASE)
# "one row per stay" / "per observation" granularity attestations must not be
# read as a row-level split when scanning loose prose.
_ROW_GRANULARITY_RE = re.compile(
    r"\b(?:one\s+|single\s+)?rows?\s+per\b|\bper\s+rows?\b", re.IGNORECASE
)


def _token_to_unit(token: str) -> str:
    """Map a single captured identifier (e.g. ``stay_id``, ``subject_id``,
    ``observation``) to a split unit. For a captured identifier (not loose
    prose) ``row``/``rows`` is an unambiguous row-level unit."""
    t = (token or "").lower()
    if "row" in t or any(tok in t for tok in _ROW_UNIT_TOKENS):
        return "row"
    if any(tok in t for tok in _PATIENT_UNIT_TOKENS):
        return "patient"
    if any(tok in t for tok in _STAY_UNIT_TOKENS):
        return "stay"
    return ""
# Phrases by which the agent attests the stay-level split is patient-equivalent.
_PATIENT_EQUIV_PHRASES = (
    "one stay per patient",
    "one row per patient",
    "one admission per patient",
    "single stay per patient",
    "equivalent to patient-level",
    "equivalent to a patient-level",
    "patient-level split",
)


def _scan_split_evidence(
    summaries: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Aggregate train/test split evidence across the agent's varied vocabulary.

    The agent records the split under different keys/shapes run to run
    (``split_integrity`` mapping, a ``split_strategy`` string or mapping, a
    free-text ``*_split_limitation`` note, ``train_n``/``test_n`` counts). Reading
    only one schema false-NAs a perfectly valid split, so we scan every step
    summary for the leakage-relevant facts: the split UNIT, any recorded patient
    OVERLAP count, whether a held-out partition exists, and whether the agent
    attested stay==patient equivalence.
    """
    unit = ""
    overlap_n: Optional[int] = None
    has_heldout = False
    patient_equiv = False

    def _note_unit(text: str) -> None:
        nonlocal unit
        if unit:
            return
        t = text.lower()
        # 1) Prefer the unit the split was explicitly performed ON / the
        #    "<unit>-level" form — robust to justification prose in the sentence.
        for rx in (_SPLIT_CONTEXT_RE, _LEVEL_CONTEXT_RE):
            for m in rx.finditer(t):
                u = _token_to_unit(m.group(1))
                if u:
                    unit = u
                    return
        # 2) Fall back to a loose scan, but neutralise "row per <x>" granularity
        #    attestations first so they do not read as a row-level split unit. A
        #    word-bounded bare "row"/"rows" still counts (structured split_unit
        #    field, "split on rows"); "borrow"/"narrow" do not.
        guarded = _ROW_GRANULARITY_RE.sub(" ", t)
        if re.search(r"\brows?\b", guarded) or any(
            tok in guarded for tok in _ROW_UNIT_TOKENS
        ):
            unit = "row"
        elif any(tok in guarded for tok in _PATIENT_UNIT_TOKENS):
            unit = "patient"
        elif any(tok in guarded for tok in _STAY_UNIT_TOKENS):
            unit = "stay"

    def _note_overlap(val: object) -> None:
        nonlocal overlap_n
        try:
            n = int(float(val))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        overlap_n = n if overlap_n is None else max(overlap_n, n)

    for doc in summaries:
        for k, v in doc.items():
            kl = (k or "").lower()
            relevant = any(
                t in kl for t in ("split", "overlap", "train", "test", "leak")
            )
            if isinstance(v, Mapping):
                if relevant or any(("unit" in (sk or "").lower()) for sk in v.keys()):
                    u = str(v.get("split_unit") or v.get("unit") or v.get("type") or "")
                    if u:
                        _note_unit(u)
                    if "test_size" in v or "n_test" in v or "test_n" in v:
                        has_heldout = True
                    for ok in ("patient_overlap_n", "overlap_n", "n_overlap"):
                        if v.get(ok) is not None:
                            _note_overlap(v.get(ok))
                    blob = json.dumps(v).lower()
                    if any(p in blob for p in _PATIENT_EQUIV_PHRASES):
                        patient_equiv = True
            elif isinstance(v, str) and relevant:
                _note_unit(v)
                if any(p in v.lower() for p in _PATIENT_EQUIV_PHRASES):
                    patient_equiv = True
                if "held-out" in v.lower() or "held out" in v.lower():
                    has_heldout = True
            elif relevant and ("overlap" in kl) and v is not None:
                _note_overlap(v)
            elif kl in ("train_n", "n_train", "test_n", "n_test") and v is not None:
                has_heldout = True

    return {
        "unit": unit,
        "overlap_n": overlap_n,
        "has_heldout": has_heldout,
        "patient_equiv": patient_equiv,
    }


def _mortality_prediction_signals(
    run_dir: Path, summaries: Sequence[Mapping[str, object]]
) -> List[ValiditySignal]:
    # Central validity property of a prediction AUROC: the train/test split keeps
    # every patient on ONE side (no patient overlap). Leakage inflates
    # discrimination — an objective Fail. We read the leakage facts from whatever
    # vocabulary the agent used (see _scan_split_evidence), not one fixed schema.
    ev = _scan_split_evidence(summaries)
    unit = str(ev["unit"])
    overlap_n = ev["overlap_n"]
    name = "patient_level_split_no_overlap"

    # Explicit patient overlap > 0 → leakage, objective Fail.
    if isinstance(overlap_n, int) and overlap_n > 0:
        return [ValiditySignal(name, "fail", f"patient overlap={overlap_n}")]
    # Explicit row/observation-level split → leakage, objective Fail.
    if unit == "row":
        return [ValiditySignal(name, "fail", "row-level split unit")]
    # Leak-free established: patient/subject unit, OR stay unit attested
    # patient-equivalent (one-stay-per-patient), OR an explicit overlap==0.
    if unit == "patient" or (unit == "stay" and ev["patient_equiv"]) or overlap_n == 0:
        detail = f"unit={unit or 'patient'}"
        if unit == "stay" and ev["patient_equiv"]:
            detail += ", attested one-stay-per-patient (patient-equivalent)"
        elif overlap_n == 0:
            detail += ", overlap=0"
        return [ValiditySignal(name, "pass", detail)]
    # A stay-level split with no patient-equivalence attestation and no overlap
    # count is defensible but not VERIFIABLY leak-free here — do not fabricate a
    # pass, do not false-fail. Same for no readable split evidence at all.
    if unit == "stay" and ev["has_heldout"]:
        return [
            ValiditySignal(
                name, "na", "stay-level held-out split; patient overlap not verifiable"
            )
        ]
    return [ValiditySignal(name, "na", "no readable split-integrity evidence")]


# Markers that the causal design is *balance-based* (weighting / matching) — the
# only family for which covariate balance (SMD) is a required diagnostic. A
# g-computation / outcome-regression / TMLE design produces no SMD balance table
# and MUST NOT be failed for lacking one (that would impose one analytical
# paradigm on a defensible alternative — [[feedback_rules_must_be_impartial]]).
_WEIGHTING_MATCHING_MARKERS = (
    "iptw",
    "ipw",
    "propensity",
    "ps_score",
    "ps_weight",
    "weights",
    "matched",
    "matching",
)


def _uses_weighting_or_matching(run_dir: Path) -> bool:
    return bool(_find_artifacts(run_dir, *_WEIGHTING_MATCHING_MARKERS))


def _causal_signals(
    run_dir: Path, summaries: Sequence[Mapping[str, object]]
) -> List[ValiditySignal]:
    out: List[ValiditySignal] = []

    # 1) Covariate balance ACHIEVED on the POST-ADJUSTMENT set — that is the table
    #    that must be balanced for the weighted/matched estimand. Critically, exclude
    #    the unweighted/crude table (it is expected to be imbalanced, so scoring it
    #    would be a false Fail) — note "weighted" is a substring of "unweighted".
    def _is_adjusted_balance(p: Path) -> bool:
        n = p.name.lower()
        if any(
            t in n for t in ("unweight", "unadjust", "crude", "raw", "_pre", "before")
        ):
            return False
        return any(
            t in n for t in ("weight", "adjust", "iptw", "ipw", "matched", "post")
        )

    all_bal = _find_artifacts(run_dir, "balance", "smd")
    adjusted = [p for p in all_bal if _is_adjusted_balance(p)]
    smds: List[float] = []
    for p in sorted(adjusted):
        smds = _floats(_read_csv_rows(p), "abs_smd", "smd", "std_mean_diff", "smd_abs")
        if smds:
            break
    if smds:
        worst = max(smds)
        out.append(
            ValiditySignal(
                "covariate_balance_achieved",
                "pass" if worst < BALANCE_SMD_THRESHOLD else "fail",
                f"max|SMD|={worst:.3f} on adjusted set (threshold {BALANCE_SMD_THRESHOLD})",
            )
        )
    elif not all_bal:
        # No balance assessment anywhere. Only a Fail if the design is actually
        # balance-based (weighting/matching), where balance is a REQUIRED check it
        # skipped. For a non-balance-based causal design (g-computation / outcome
        # regression / TMLE) balance is not the relevant diagnostic, so demanding
        # it would impose one paradigm — stay na (impartial), not fail.
        if _uses_weighting_or_matching(run_dir):
            out.append(
                ValiditySignal(
                    "covariate_balance_achieved",
                    "fail",
                    "weighting/matching design with no balance/SMD assessment",
                )
            )
        else:
            out.append(
                ValiditySignal(
                    "covariate_balance_achieved",
                    "na",
                    "no balance table; design is not weighting/matching-based "
                    "(balance not a required diagnostic)",
                )
            )
    else:
        # Balance artifacts exist but the post-adjustment one is not machine-readable
        # in our columns — do not guess correctness from the crude table.
        out.append(
            ValiditySignal(
                "covariate_balance_achieved", "na", "post-adjustment SMD not readable"
            )
        )

    # 2) Positivity / overlap assessed with a readable verdict. Presence of a
    #    diagnostic alone is reporting; we score only when a pass/violation verdict
    #    is machine-readable, else na (we do not infer correctness from a filename).
    pos_paths = _find_artifacts(
        run_dir, "positivity_diagnostics", "positivity", "overlap_decision", "overlap"
    )
    verdict: Optional[str] = None
    for p in sorted(pos_paths):
        rows = _read_csv_rows(p)
        for r in rows:
            for k, v in r.items():
                kl = (k or "").strip().lower()
                vl = str(v or "").strip().lower()
                if any(
                    t in kl
                    for t in ("positivity", "overlap", "decision", "verdict", "status")
                ):
                    if any(t in vl for t in ("violat", "fail", "no_overlap", "poor")):
                        verdict = "fail"
                    elif any(
                        t in vl
                        for t in ("ok", "pass", "satisf", "adequate", "good", "holds")
                    ):
                        verdict = verdict or "pass"
        if verdict:
            break
    if verdict:
        out.append(
            ValiditySignal("positivity_assessed", verdict, "from diagnostic verdict")
        )
    else:
        out.append(
            ValiditySignal("positivity_assessed", "na", "no machine-readable verdict")
        )

    return out


# Detectors are registered only for kinds where a value-readable, central validity
# check exists. Every other kind intentionally has NO entry → stays NA (its teeth,
# e.g. overadjustment / phenotype degeneracy, still cap Fail upstream).
_KIND_DETECTORS: Dict[
    str, Callable[[Path, Sequence[Mapping[str, object]]], List[ValiditySignal]]
] = {
    "mortality_prediction": _mortality_prediction_signals,
    "causal_inference": _causal_signals,
}


def assess_validity_signals(kind: Optional[str], run_dir: Path) -> List[ValiditySignal]:
    """Value-based, gold-free validity signals for a run of the given task kind.

    Returns ``[]`` for a kind with no implementable central validity check — the
    dimension then falls back to the error/gold logic in ``score_result_validity``
    (Fail on a teeth error, else honestly unscored).
    """
    detector = _KIND_DETECTORS.get((kind or "").strip())
    if detector is None:
        return []
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    return detector(run_dir, _iter_step_summaries(run_dir))


def validity_positive_subscore(
    signals: Sequence[ValiditySignal],
) -> Optional[float]:
    """``passes / (passes + fails)`` over assessed signals; ``None`` if all ``na``.

    Graded — a skipped/failed objective check pulls the score down rather than
    collapsing to 1.0-or-None. ``na`` is excluded so an unverifiable check neither
    inflates nor deflates. ``None`` keeps the dimension honestly unscored.
    """
    passes = sum(1 for s in signals if s.status == "pass")
    fails = sum(1 for s in signals if s.status == "fail")
    assessed = passes + fails
    if assessed == 0:
        return None
    return passes / assessed
