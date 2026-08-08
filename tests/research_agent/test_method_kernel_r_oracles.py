"""In-tree statistical kernels agree with the reference implementations.

EasyICU implements several statistics itself rather than shelling out to R --
DeLong's AUC comparison, the E-value, the Schoenfeld PH test (this one wraps
lifelines). Self-implemented is defensible; *unvalidated* self-implemented is
not, and the previous tests only checked internal consistency (AUC agrees with
sklearn, bootstrap SE is sane, ties behave). That establishes the code does
what its author meant, not that what its author meant is the statistic the
literature names.

These compare against values produced by ``survival::cox.zph``,
``pROC::roc.test(method="delong")`` and ``EValue::evalues.RR`` on exactly the
bytes in ``data/oracle_*.csv``. The R values are frozen in
``data/method_kernel_oracles.json`` so CI needs no R; regenerate with
``tools/generate_method_kernel_oracles.py``.

Two things this does NOT cover, stated here so the file is not read as more
than it is:

* the PH ``global`` row -- ours is a Bonferroni family-wise summary and
  ``cox.zph``'s is the joint Grambsch-Therneau chi-square, which are different
  statistics by design (see ``methods/ph_schoenfeld.py``). The per-covariate
  rows, which the survival receipt's exposure verdict now reads, are compared.
  Lifelines and R do not produce identical null-covariate p-values when a
  different covariate is strongly non-PH, so the external contract is the
  per-covariate decision at alpha=0.05 plus 5% agreement for the deliberately
  violated covariate's chi-square -- not invented bitwise equivalence.
* ``methods.rmst`` and ``methods.decision_curve`` -- ``survRM2`` and
  ``dcurves``/``rmda`` are not installed, so those two kernels remain
  unvalidated against any external reference.
* the E-value oracle covers the RR formula only. The observed-prevalence
  OR-to-RR conversion is an EasyICU contract and is not claimed to match
  ``EValue::evalues.OR(rare=FALSE)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.methods.sensitivity import compute_e_value


DATA = Path(__file__).parent / "data"
ORACLE = json.loads((DATA / "method_kernel_oracles.json").read_text(encoding="utf-8"))


PH_FIXTURES = {
    "proportional": "oracle_survival.csv",
    "exposure_nonph": "oracle_survival_exposure_nonph.csv",
    "nuisance_nonph": "oracle_survival_nuisance_nonph.csv",
}


@pytest.fixture(scope="module")
def roc_frame() -> pd.DataFrame:
    return pd.read_csv(DATA / "oracle_roc.csv")


# --- Schoenfeld PH <-> survival::cox.zph -------------------------------------


@pytest.mark.parametrize("scenario", tuple(PH_FIXTURES))
def test_per_covariate_ph_decisions_match_r_cox_zph(scenario: str) -> None:
    ph_schoenfeld = pytest.importorskip(
        "easyicu.research_agent.methods.ph_schoenfeld",
        reason="ph_schoenfeld requires lifelines",
    )
    pytest.importorskip("lifelines")

    frame = pd.read_csv(DATA / PH_FIXTURES[scenario])
    table = ph_schoenfeld.ph_test(
        frame,
        duration_col="time",
        event_col="event",
        covariates=["exposure", "nuisance"],
        time_transform="km",
    )
    by_covariate = {str(row["covariate"]): row for _, row in table.iterrows()}
    for name, expected in ORACLE["ph_scenarios"][scenario].items():
        assert name in by_covariate, name
        assert (float(by_covariate[name]["p_value"]) < 0.05) is (
            expected["p"] < 0.05
        )
    violated = {
        "exposure_nonph": "exposure",
        "nuisance_nonph": "nuisance",
    }.get(scenario)
    if violated is not None:
        assert float(by_covariate[violated]["test_statistic"]) == pytest.approx(
            ORACLE["ph_scenarios"][scenario][violated]["chisq"], rel=0.05
        )

    assert ORACLE["_provenance"]["ph_reference_scope"].startswith(
        "per-covariate alpha=0.05 decision"
    )


def test_the_ph_global_row_is_deliberately_not_the_r_global() -> None:
    """The Bonferroni summary is a different statistic, not a broken one."""

    assert "Bonferroni" in ORACLE["_provenance"]["ph_global_excluded_because"]


def test_frozen_r_oracles_really_span_exposure_and_nuisance_nonph() -> None:
    """This test runs even when the optional Python lifelines adapter is absent."""

    scenarios = ORACLE["ph_scenarios"]
    assert scenarios["proportional"]["exposure"]["p"] > 0.05
    assert scenarios["proportional"]["nuisance"]["p"] > 0.05
    assert scenarios["exposure_nonph"]["exposure"]["p"] < 0.01
    assert scenarios["exposure_nonph"]["nuisance"]["p"] > 0.05
    assert scenarios["nuisance_nonph"]["exposure"]["p"] > 0.05
    assert scenarios["nuisance_nonph"]["nuisance"]["p"] < 0.01


# --- DeLong <-> pROC::roc.test(method="delong") ------------------------------


def test_auc_and_delong_comparison_match_r_proc(roc_frame) -> None:
    delong = pytest.importorskip("easyicu.research_agent.methods.delong_auc")

    label = roc_frame["label"].to_numpy()
    a = roc_frame["score_a"].to_numpy()
    b = roc_frame["score_b"].to_numpy()

    auc_a, auc_b, z, p_value = delong.delong_test(label, a, b)
    assert auc_a == pytest.approx(ORACLE["auc_a"], rel=1e-6)
    assert auc_b == pytest.approx(ORACLE["auc_b"], rel=1e-6)
    # DeLong's z is signed by argument order; compare magnitude.
    assert abs(z) == pytest.approx(abs(ORACLE["delong_z"]), rel=1e-3)
    assert p_value == pytest.approx(ORACLE["delong_p"], rel=1e-3)


# --- E-value <-> EValue::evalues.RR ------------------------------------------


def test_rr_e_value_point_and_ci_match_r_evalue() -> None:
    assert ORACLE["_provenance"]["evalue_reference_scope"].startswith("RR formula")
    result = compute_e_value(estimate=3.9, estimate_type="rr", ci=(2.8, 5.4))
    assert float(result.e_value) == pytest.approx(ORACLE["evalue_point"], rel=1e-6)
    assert float(result.e_value_lower_bound) == pytest.approx(
        ORACLE["evalue_lower"], rel=1e-6
    )
