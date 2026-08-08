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
* ``methods.sensitivity.compute_e_value`` <-> ``EValue::evalues.RR``

Two deliberate exclusions, recorded rather than quietly skipped:

* The PH ``global`` row is NOT compared. Ours is a Bonferroni family-wise
  summary, min(1, k * min_j p_j); ``cox.zph``'s is the joint Grambsch-Therneau
  chi-square on k df. They answer different questions and are not expected to
  match. The per-covariate rows -- which is what the exposure verdict now reads
  -- are compared exactly.
* ``methods.rmst`` and ``methods.decision_curve`` have no oracle here because
  ``survRM2`` and ``dcurves``/``rmda`` are not installed on this machine. They
  remain unvalidated against an external reference; do not read this file as
  covering them.
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


def _survival_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20260807)
    n = 400
    treatment = rng.integers(0, 2, n).astype(float)
    age = rng.normal(65, 12, n).round(3)
    # A deliberately time-varying treatment effect so the PH test has something
    # to find: hazard depends on treatment early and reverses late.
    base = rng.exponential(scale=1.0, size=n)
    time = np.where(treatment > 0, base * 0.6, base * 1.4) * 30.0
    horizon = 60.0
    event = (time <= horizon).astype(int)
    time = np.minimum(time, horizon).round(4)
    return pd.DataFrame(
        {"time": time, "event": event, "treatment": treatment, "age": age}
    )


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
surv <- read.csv(args[1]); roc_df <- read.csv(args[2])

fit <- coxph(Surv(time, event) ~ treatment + age, data = surv)
z <- cox.zph(fit, transform = "km")
tab <- as.data.frame(z$table)
ph <- list()
for (nm in rownames(tab)) {
  if (nm == "GLOBAL") next
  ph[[nm]] <- list(chisq = unname(tab[nm, "chisq"]), p = unname(tab[nm, "p"]))
}

r1 <- roc(roc_df$label, roc_df$score_a, quiet = TRUE)
r2 <- roc(roc_df$label, roc_df$score_b, quiet = TRUE)
dl <- roc.test(r1, r2, method = "delong")

ev <- evalues.RR(est = 3.9, lo = 2.8, hi = 5.4)

cat(jsonlite::toJSON(list(
  ph_per_covariate = ph,
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
    surv, roc = _survival_frame(), _roc_frame()
    surv_csv = FIXTURE_DIR / "oracle_survival.csv"
    roc_csv = FIXTURE_DIR / "oracle_roc.csv"
    surv.to_csv(surv_csv, index=False)
    roc.to_csv(roc_csv, index=False)

    script = FIXTURE_DIR / "_oracle.R"
    script.write_text(R_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        ["Rscript", str(script), str(surv_csv), str(roc_csv)],
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
    }
    GOLDEN.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
