"""A published Wald bound is checked within the rounding it was published with.

MIMIC-IV walk 11-03: a patient-clustered 「按性别分组」 mortality table stopped
with "risk-difference interval is not its point estimate plus/minus the
declared Wald-normal uncertainty".  The producer rounds each bound from the
full-precision estimate and standard error; every consumer rebuilt it from the
published, rounded values within a fixed 2e-6, which their roundings can
exceed (about one synthetic clustered table in nine).  The executor, the
figure renderer and the claim authority now share one bound for that
rounding, and still refuse an interval whose arithmetic is wrong.  Synthetic
cohorts only.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.authority.descriptive_scientific_claims import (
    derive_descriptive_claim_payloads,
)
from easyicu.research_agent.methods.cluster_proportion import published_wald_tolerance

from . import test_exposure_outcome_distribution_executor as executor_case
from . import test_exposure_outcome_distribution_render as render_case

_DEPENDENCE = {
    "variance_estimator": "cluster_robust", "cluster_unit": "patient",
    "group_source": "opaque_row_identity", "group_derivation": "prefix_before_delimiter",
    "delimiter": ":s",
}
_CONTRAST = {"reference_exposure_level": 0, "comparison_exposure_level": 1}


def _cohort(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for patient in range(rng.randint(20, 120)):
        exposed = rng.randint(0, 1)
        for stay in range(rng.randint(1, 3)):
            rows.append({
                executor_case.EXPOSURE: exposed,
                executor_case.OUTCOME: int(rng.random() < 0.15 + 0.1 * exposed),
                "opaque_row_identity": f"p{patient:03d}:s{stay + 1}",
            })
    return pd.DataFrame(rows)


def test_the_bound_is_the_published_rounding_and_no_wider() -> None:
    z95 = statistics.NormalDist().inv_cdf(0.975)
    assert 2e-6 < published_wald_tolerance(z95) < 2.5e-6
    assert published_wald_tolerance(statistics.NormalDist().inv_cdf(0.995)) < 3e-6


# Seeds whose tables the fixed 2e-6 refused (outcome risk or risk difference).
@pytest.mark.parametrize("seed", [16, 38, 45, 57, 75])
def test_a_clustered_table_passes_every_consumer(monkeypatch, tmp_path: Path, seed: int) -> None:
    summary = executor_case._run(
        monkeypatch, tmp_path / "executor", _cohort(seed),
        risk_difference_contrast=_CONTRAST, dependence=_DEPENDENCE,
    )
    assert summary["status"] == "ok"
    assert derive_descriptive_claim_payloads(summary)
    table = render_case._produced_table(
        tmp_path / "render", monkeypatch,
        frame=_cohort(seed).rename(columns={
            executor_case.EXPOSURE: render_case.EXPOSURE,
            executor_case.OUTCOME: render_case.OUTCOME,
        }),
        spec_updates={"risk_difference_contrast": _CONTRAST, "dependence": _DEPENDENCE},
    )
    run_dir, manifest = render_case._bound(tmp_path / "render", table)
    render_case._render(run_dir, manifest, tmp_path / "render" / "out")


def test_a_wrong_interval_is_still_refused(monkeypatch, tmp_path: Path) -> None:
    table = render_case._produced_table(
        tmp_path, monkeypatch,
        frame=_cohort(16).rename(columns={
            executor_case.EXPOSURE: render_case.EXPOSURE,
            executor_case.OUTCOME: render_case.OUTCOME,
        }),
        spec_updates={"risk_difference_contrast": _CONTRAST, "dependence": _DEPENDENCE},
    )
    produced = pd.read_csv(table)
    produced["risk_difference_ci_high_pct"] = produced["risk_difference_ci_high_pct"] + 1e-5
    produced.to_csv(table, index=False)
    run_dir, manifest = render_case._bound(tmp_path, table)
    with pytest.raises(ValueError, match="Wald-normal arithmetic"):
        render_case._render(run_dir, manifest, tmp_path / "out")
