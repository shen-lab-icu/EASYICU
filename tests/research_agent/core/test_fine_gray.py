"""Fine-Gray adapter compared with a separate local survival reference fit."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.r_runtime import run_rscript


def test_fine_gray_agrees_with_r_reference(tmp_path):
    from easyicu.research_agent.methods.competing_risks import fit_fine_gray

    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("local Rscript is not installed")
    rng = np.random.default_rng(942)
    data = pd.DataFrame({"t": rng.uniform(0.1, 20, 160),
                         "event": rng.choice([0, 1, 2], 160),
                         "x": rng.integers(0, 2, 160), "z": rng.normal(size=160)})
    data.to_csv(tmp_path / "input.csv", index=False)
    reference = run_rscript(rscript=rscript, script=r'''
library(survival)
args <- commandArgs(TRUE)
d <- read.csv(args[1])
d$id <- seq_len(nrow(d))
d$status <- factor(d$event, levels=0:2, labels=c("censor", "target", "competing"))
expanded <- finegray(Surv(t, status) ~ x + z + id, data=d, etype="target")
fit <- coxph(Surv(fgstart, fgstop, fgstatus) ~ x + z, data=expanded,
             weights=fgwt, cluster=id, ties="efron")
s <- summary(fit)
write.csv(data.frame(term=rownames(s$coefficients), coef=s$coefficients[,"coef"],
                    se=s$coefficients[,"robust se"], p=s$coefficients[,"Pr(>|z|)"]),
          args[2], row.names=FALSE)
''', args=[str(tmp_path / "input.csv"), str(tmp_path / "reference.csv")], timeout_s=30)
    assert reference.returncode == 0, reference.stderr
    expected = pd.read_csv(tmp_path / "reference.csv")
    result = fit_fine_gray(data, duration_col="t", event_col="event", covariates=["x", "z"])
    assert result.estimates.term.tolist() == expected.term.tolist()
    assert result.estimates.coefficient.to_numpy() == pytest.approx(expected.coef, abs=1e-9)
    assert result.estimates.standard_error.to_numpy() == pytest.approx(expected.se, abs=1e-9)
    assert result.estimates.p_value.to_numpy() == pytest.approx(expected.p, abs=1e-9)
    assert result.estimates.hazard_ratio.to_numpy() == pytest.approx(np.exp(expected.coef))
    assert result.receipt["survival_version"]


@pytest.mark.parametrize("failure,reason", [
    ("timeout", "fine_gray_fit_timeout"),
    ("warning", "fine_gray_fit_warning"),
    ("failed", "fine_gray_fit_failed"),
])
def test_fine_gray_does_not_emit_a_fit_after_runtime_failure(monkeypatch, failure, reason):
    from easyicu.research_agent.methods import competing_risks as kernel
    from easyicu.research_agent.r_runtime import RScriptResult
    data = pd.DataFrame({"t": [1, 2, 3, 4], "event": [0, 1, 2, 1], "x": [0, 1, 0, 1]})
    monkeypatch.setattr(kernel.shutil, "which", lambda _: "/synthetic/Rscript")
    monkeypatch.setattr(kernel, "run_rscript", lambda **_: RScriptResult(
        returncode=1, timed_out=failure == "timeout",
        stderr="EASYICU_FINE_GRAY_WARNING: did not converge" if failure == "warning" else "failed",
    ))
    with pytest.raises(kernel.FineGrayError) as exc:
        kernel.fit_fine_gray(data, duration_col="t", event_col="event", covariates=["x"])
    assert exc.value.code == reason
