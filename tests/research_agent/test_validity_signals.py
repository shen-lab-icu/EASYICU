"""Gold-free, kind-routed VALUE-based result-validity signals.

These lock the two properties the design demands and that a naive presence-check
version got wrong:
* signals read an actual VALUE and judge correctness against a standard threshold
  (split overlap == 0, adjusted-set max|SMD| < 0.1), never "a file exists";
* the subscore is graded (passes / (passes + fails)), never collapsed to 1.0-or-None;
* a kind with no value-readable central check stays unscored (``[]`` → ``None``) —
  no fabricated pass (guards the "healthy clustering stays unscored" rule).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from easyicu.research_agent.validity_signals import (
    BALANCE_SMD_THRESHOLD,
    ValiditySignal,
    assess_validity_signals,
    validity_positive_subscore,
)


def _write_model_summary(run_dir: Path, split_integrity) -> None:
    out = run_dir / "steps" / "01_model_training" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"auroc": 0.8}
    if split_integrity is not None:
        payload["split_integrity"] = split_integrity
    (out / "step_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_balance(run_dir: Path, name: str, abs_smds) -> None:
    out = run_dir / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    with (out / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["feature", "abs_smd"])
        w.writeheader()
        for i, v in enumerate(abs_smds):
            w.writerow({"feature": f"x{i}", "abs_smd": v})


# ---------------------------------------------------------------------------
# M2 patient-level split (value: overlap)
# ---------------------------------------------------------------------------


def test_split_no_overlap_passes(tmp_path):
    _write_model_summary(tmp_path, {"split_unit": "stay_id", "patient_overlap_n": 0})
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert [(s.name, s.status) for s in sig] == [
        ("patient_level_split_no_overlap", "pass")
    ]
    assert validity_positive_subscore(sig) == 1.0


def test_split_with_patient_overlap_fails(tmp_path):
    _write_model_summary(tmp_path, {"split_unit": "patient", "patient_overlap_n": 12})
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "fail"
    assert validity_positive_subscore(sig) == 0.0


def test_row_level_split_fails(tmp_path):
    _write_model_summary(tmp_path, {"split_unit": "row", "patient_overlap_n": 0})
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "fail"


def test_missing_split_metadata_is_na_not_false_fail(tmp_path):
    # Absence in our field is NOT evidence the run skipped the split — stay na so a
    # correctly-split run that records it elsewhere is not false-failed.
    _write_model_summary(tmp_path, None)
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "na"
    assert validity_positive_subscore(sig) is None


def _write_named_summary(run_dir, step, payload):
    out = run_dir / "steps" / step / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "step_summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_split_strategy_vocabulary_with_patient_equivalence_passes(tmp_path):
    # The real M2 vocabulary: a `split_strategy` (not `split_integrity`) at stay
    # level, plus an explicit attestation that one stay == one patient. This is a
    # genuinely leak-free split; the reader must recognise it (was a false NA).
    _write_named_summary(
        tmp_path,
        "02_model_training",
        {
            "auroc": 0.82,
            "split_strategy": "held-out split on stay_id as a proxy for patient-level",
            "train_n": 59863,
            "test_n": 14966,
        },
    )
    _write_named_summary(
        tmp_path,
        "02_audit",
        {
            "benchmark_specific_split_limitation": (
                "Only stay_id is available; benchmark metadata report one stay per "
                "patient; therefore split by stay_id is equivalent to patient-level."
            )
        },
    )
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "pass"
    assert validity_positive_subscore(sig) == 1.0


def test_stay_level_heldout_without_equivalence_is_na_not_pass(tmp_path):
    # Impartiality / no fabrication: a stay-level held-out split with NO attestation
    # that the cohort is one-stay-per-patient is NOT verifiably leak-free (a
    # multi-stay patient could straddle train/test). Must be na, never a free pass.
    _write_named_summary(
        tmp_path,
        "02_model_training",
        {
            "auroc": 0.82,
            "split_strategy": {"unit": "stay_id", "test_size": 0.2},
            "train_n": 100,
            "test_n": 25,
        },
    )
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "na"
    assert validity_positive_subscore(sig) is None


def test_split_strategy_mapping_explicit_overlap_zero_passes(tmp_path):
    _write_named_summary(
        tmp_path,
        "02_model_training",
        {"split_strategy": {"unit": "stay_id", "patient_overlap_n": 0}},
    )
    sig = assess_validity_signals("mortality_prediction", tmp_path)
    assert sig[0].status == "pass"


# ---------------------------------------------------------------------------
# H2 covariate balance (value: adjusted-set max|SMD|, NOT the unweighted table)
# ---------------------------------------------------------------------------


def test_balance_reads_adjusted_not_unweighted_table(tmp_path):
    # The unweighted table is expected to be imbalanced; scoring it would be a false
    # Fail. "weighted" is a substring of "unweighted" — the detector must not match it.
    _write_balance(tmp_path, "unweighted_baseline_balance.csv", [0.79, 0.5, 0.4])
    _write_balance(tmp_path, "weighted_baseline_balance.csv", [0.05, 0.04, 0.06])
    sig = assess_validity_signals("causal_inference", tmp_path)
    bal = next(s for s in sig if s.name == "covariate_balance_achieved")
    assert bal.status == "pass"  # adjusted set is balanced
    assert "0.06" in bal.detail and "0.79" not in bal.detail


def test_balance_residual_imbalance_fails(tmp_path):
    _write_balance(tmp_path, "weighted_baseline_balance.csv", [0.05, 0.226, 0.04])
    sig = assess_validity_signals("causal_inference", tmp_path)
    bal = next(s for s in sig if s.name == "covariate_balance_achieved")
    assert bal.status == "fail"
    assert BALANCE_SMD_THRESHOLD == 0.1


def test_balance_only_unweighted_present_is_na(tmp_path):
    # Cannot judge the adjusted estimand's balance from the crude table alone.
    _write_balance(tmp_path, "unweighted_baseline_balance.csv", [0.79, 0.5])
    sig = assess_validity_signals("causal_inference", tmp_path)
    bal = next(s for s in sig if s.name == "covariate_balance_achieved")
    assert bal.status == "na"


def test_no_balance_with_weighting_design_fails(tmp_path):
    # A weighting/matching design (here: an IPTW weights artifact) for which NO
    # balance table exists skipped a REQUIRED check -> objective Fail.
    ev = tmp_path / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "iptw_weights.csv").write_text("id,weight\n1,0.9\n", encoding="utf-8")
    sig = assess_validity_signals("causal_inference", tmp_path)
    bal = next(s for s in sig if s.name == "covariate_balance_achieved")
    assert bal.status == "fail"


def test_no_balance_without_weighting_design_is_na_not_fail(tmp_path):
    # No balance table AND no weighting/matching evidence: the causal estimate may
    # be g-computation / outcome regression / TMLE, which does not produce an SMD
    # table. Demanding one would impose a paradigm -> must stay NA, not Fail
    # (impartiality: never fail a defensible analytical choice).
    (tmp_path / "evidence").mkdir(parents=True, exist_ok=True)
    sig = assess_validity_signals("causal_inference", tmp_path)
    bal = next(s for s in sig if s.name == "covariate_balance_achieved")
    assert bal.status == "na"


def test_causal_graded_subscore_balance_fail_positivity_pass(tmp_path):
    _write_balance(tmp_path, "weighted_baseline_balance.csv", [0.226])
    out = tmp_path / "evidence"
    with (out / "positivity_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "overlap_decision"])
        w.writeheader()
        w.writerow({"metric": "ps_overlap", "overlap_decision": "adequate"})
    sig = assess_validity_signals("causal_inference", tmp_path)
    # balance fail + positivity pass -> 1/2 graded, NOT collapsed to 1.0/None
    assert validity_positive_subscore(sig) == 0.5


# ---------------------------------------------------------------------------
# The don't-fabricate-a-pass guarantee: kinds with no value-readable check -> NA
# ---------------------------------------------------------------------------


def test_clustering_has_no_positive_signal(tmp_path):
    # Clustering validity needs a threshold the design refuses to impose; degeneracy
    # is a Fail via the phenotype teeth, a healthy partition stays unscored here.
    (tmp_path / "cluster_validity.json").write_text(
        json.dumps({"silhouette": 0.41}), encoding="utf-8"
    )
    sig = assess_validity_signals("subphenotype_clustering", tmp_path)
    assert sig == []
    assert validity_positive_subscore(sig) is None


def test_unmapped_kinds_stay_unscored(tmp_path):
    for kind in (
        "descriptive_association",
        "ordinal_dose_response",
        "missingness_robustness",
        "survival_analysis",
        "longitudinal_trajectory_analysis",
    ):
        assert assess_validity_signals(kind, tmp_path) == []


def test_subscore_excludes_na_returns_none_when_all_na():
    sigs = [ValiditySignal("a", "na"), ValiditySignal("b", "na")]
    assert validity_positive_subscore(sigs) is None
