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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

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
    for p in run_dir.rglob("*"):
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


def _mortality_prediction_signals(
    run_dir: Path, summaries: Sequence[Mapping[str, object]]
) -> List[ValiditySignal]:
    # Central validity property of a prediction AUROC: the train/test split is at
    # the patient level with NO patient overlap. A row-level split (or overlap) is
    # leakage that inflates discrimination — an objective Fail. A prediction model
    # that reports no split integrity at all has skipped a required check → fail.
    split = _summary_get(summaries, "split_integrity")
    if isinstance(split, Mapping):
        unit = str(split.get("split_unit") or "").lower()
        overlap = split.get("patient_overlap_n")
        patient_unit = any(t in unit for t in ("patient", "stay", "subject"))
        try:
            overlap_n = int(float(overlap)) if overlap is not None else None
        except (TypeError, ValueError):
            overlap_n = None
        if patient_unit and overlap_n == 0:
            return [
                ValiditySignal(
                    "patient_level_split_no_overlap", "pass", f"unit={unit}, overlap=0"
                )
            ]
        if overlap_n is not None and overlap_n > 0:
            return [
                ValiditySignal(
                    "patient_level_split_no_overlap",
                    "fail",
                    f"patient overlap={overlap_n}",
                )
            ]
        if not patient_unit and unit:
            # An explicitly NON-patient (row-level) split is leakage: an objective Fail.
            return [
                ValiditySignal(
                    "patient_level_split_no_overlap",
                    "fail",
                    f"row-level split unit ({unit})",
                )
            ]
        return [
            ValiditySignal(
                "patient_level_split_no_overlap", "na", "split metadata unreadable"
            )
        ]
    # No split-integrity metadata in the field we read. We CANNOT conclude the run
    # skipped the split (it may record it elsewhere) — absence in our format is not
    # evidence of wrongness, so stay na rather than false-fail.
    return [
        ValiditySignal(
            "patient_level_split_no_overlap",
            "na",
            "no readable split-integrity metadata",
        )
    ]


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
