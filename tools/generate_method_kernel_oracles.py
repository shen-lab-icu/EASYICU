"""Regenerate the R-oracle goldens for EasyICU's in-tree statistical kernels.

Run manually when a kernel changes; CI compares against the frozen JSON and
never needs R installed.

    Rscript --version            # 4.x, with survival / pROC / EValue
    python tools/generate_method_kernel_oracles.py

What this does and does not establish
-------------------------------------
It checks that our implementations agree numerically with the reference
implementations clinicians and reviewers actually cite:

* ``methods.ph_schoenfeld.ph_test``   <-> ``survival::cox.zph``
* ``methods.delong_auc``              <-> ``pROC::roc.test(method="delong")``
* RR branch of ``methods.sensitivity.compute_e_value`` <-> ``EValue::evalues.RR``

Two deliberate exclusions, recorded rather than quietly skipped:

* The PH ``global`` row is NOT compared. Ours is a Bonferroni family-wise
  summary, min(1, k * min_j p_j); ``cox.zph``'s is the joint Grambsch-Therneau
  chi-square on k df. They answer different questions and are not expected to
  match. The per-covariate alpha=0.05 decisions are compared, and the targeted
  non-PH covariate's chi-square must agree within 5%. Lifelines and R do not
  give identical null-covariate p-values when a different covariate is strongly
  non-PH, so the fixture does not claim bitwise numerical equivalence there.
* ``methods.rmst`` and ``methods.decision_curve`` have no oracle here because
  ``survRM2`` and ``dcurves``/``rmda`` are not installed on this machine. They
  remain unvalidated against an external reference; do not read this file as
  covering them.
* The E-value comparison covers the RR formula only. It does not validate
  EasyICU's observed-prevalence Zhang--Yu OR-to-RR conversion against
  ``EValue::evalues.OR(rare=FALSE)``, which is a different contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLDEN = ROOT / "tests" / "research_agent" / "data" / "method_kernel_oracles.json"
FIXTURE_DIR = ROOT / "tests" / "research_agent" / "data"


def _piecewise_survival_frame(
    *,
    seed: int,
    exposure_log_hr: tuple[float, float],
    nuisance_log_hr: tuple[float, float],
) -> pd.DataFrame:
    """Simulate a two-interval piecewise-exponential survival process."""

    rng = np.random.default_rng(seed)
    n = 1800
    exposure = rng.integers(0, 2, n).astype(float)
    nuisance = rng.integers(0, 2, n).astype(float)
    split = 12.0
    baseline_hazard = 0.025
    early_hazard = baseline_hazard * np.exp(
        exposure_log_hr[0] * exposure + nuisance_log_hr[0] * nuisance
    )
    late_hazard = baseline_hazard * np.exp(
        exposure_log_hr[1] * exposure + nuisance_log_hr[1] * nuisance
    )
    cumulative_target = rng.exponential(scale=1.0, size=n)
    event_time = np.where(
        cumulative_target <= early_hazard * split,
        cumulative_target / early_hazard,
        split + (cumulative_target - early_hazard * split) / late_hazard,
    )
    censor_time = rng.uniform(18.0, 55.0, size=n)
    event = (event_time <= censor_time).astype(int)
    time = np.minimum(event_time, censor_time).round(6)
    return pd.DataFrame(
        {
            "time": time,
            "event": event,
            "exposure": exposure,
            "nuisance": nuisance,
        }
    )


def _survival_frames() -> dict[str, pd.DataFrame]:
    stable_exposure = (np.log(1.6), np.log(1.6))
    stable_nuisance = (np.log(1.3), np.log(1.3))
    crossing = (np.log(4.0), np.log(0.25))
    return {
        "proportional": _piecewise_survival_frame(
            seed=20260807,
            exposure_log_hr=stable_exposure,
            nuisance_log_hr=stable_nuisance,
        ),
        "exposure_nonph": _piecewise_survival_frame(
            seed=20260808,
            exposure_log_hr=crossing,
            nuisance_log_hr=stable_nuisance,
        ),
        "nuisance_nonph": _piecewise_survival_frame(
            seed=20260809,
            exposure_log_hr=stable_exposure,
            nuisance_log_hr=crossing,
        ),
    }


def _roc_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    label = rng.integers(0, 2, n)
    score_a = rng.normal(label * 1.1, 1.0, n).round(6)
    score_b = rng.normal(label * 0.6, 1.0, n).round(6)
    return pd.DataFrame({"label": label, "score_a": score_a, "score_b": score_b})


R_SCRIPT = r"""
suppressMessages({library(survival); library(pROC); library(EValue)})
args <- commandArgs(trailingOnly = TRUE)
roc_df <- read.csv(args[4])

ph_table <- function(path) {
  surv <- read.csv(path)
  fit <- coxph(Surv(time, event) ~ exposure + nuisance, data = surv)
  z <- cox.zph(fit, transform = "km")
  tab <- as.data.frame(z$table)
  ph <- list()
  for (nm in rownames(tab)) {
    if (nm == "GLOBAL") next
    ph[[nm]] <- list(chisq = unname(tab[nm, "chisq"]), p = unname(tab[nm, "p"]))
  }
  ph
}
ph <- list(
  proportional = ph_table(args[1]),
  exposure_nonph = ph_table(args[2]),
  nuisance_nonph = ph_table(args[3])
)

r1 <- roc(roc_df$label, roc_df$score_a, quiet = TRUE)
r2 <- roc(roc_df$label, roc_df$score_b, quiet = TRUE)
dl <- roc.test(r1, r2, method = "delong")

ev <- evalues.RR(est = 3.9, lo = 2.8, hi = 5.4)

cat(jsonlite::toJSON(list(
  ph_scenarios = ph,
  auc_a = unname(as.numeric(auc(r1))),
  auc_b = unname(as.numeric(auc(r2))),
  delong_p = unname(dl$p.value),
  delong_z = unname(dl$statistic),
  evalue_point = unname(ev["E-values", "point"]),
  evalue_lower = unname(ev["E-values", "lower"])
), auto_unbox = TRUE, digits = 12))
"""


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    survival_frames, roc = _survival_frames(), _roc_frame()
    survival_paths = {
        "proportional": FIXTURE_DIR / "oracle_survival.csv",
        "exposure_nonph": FIXTURE_DIR / "oracle_survival_exposure_nonph.csv",
        "nuisance_nonph": FIXTURE_DIR / "oracle_survival_nuisance_nonph.csv",
    }
    roc_csv = FIXTURE_DIR / "oracle_roc.csv"
    for name, frame in survival_frames.items():
        frame.to_csv(survival_paths[name], index=False)
    roc.to_csv(roc_csv, index=False)

    script = FIXTURE_DIR / "_oracle.R"
    script.write_text(R_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [
            "Rscript",
            str(script),
            str(survival_paths["proportional"]),
            str(survival_paths["exposure_nonph"]),
            str(survival_paths["nuisance_nonph"]),
            str(roc_csv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    script.unlink(missing_ok=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    payload = json.loads(proc.stdout)
    payload["_provenance"] = {
        "generator": "tools/generate_method_kernel_oracles.py",
        "r_packages": ["survival", "pROC", "EValue"],
        "not_covered": ["methods.rmst", "methods.decision_curve"],
        "ph_global_excluded_because": (
            "ours is Bonferroni min(1, k*min_p); cox.zph's is the joint "
            "Grambsch-Therneau chi-square on k df -- different statistics"
        ),
        "ph_scenarios": {
            "proportional": "constant exposure and nuisance log-hazard ratios",
            "exposure_nonph": "piecewise exposure HR 4.0 early and 0.25 late",
            "nuisance_nonph": "piecewise nuisance HR 4.0 early and 0.25 late",
        },
        "ph_reference_scope": (
            "per-covariate alpha=0.05 decision; deliberately violated "
            "covariate chi-square within 5% between lifelines and R cox.zph"
        ),
        "evalue_reference_scope": "RR formula only (EValue::evalues.RR)",
    }
    GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
