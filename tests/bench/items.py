"""Benchmark fixtures — 6 canonical ICU research questions (T2.1).

Each :class:`BenchItem` declares everything the harness needs to score
one run end-to-end:

* a research question (the planner sees this verbatim),
* a synthetic cohort generator (deterministic; baked-in pitfall),
* the expected primary predictor and direction of association,
* the substrings that should appear in validator findings when the
  ICU-aware rules fire correctly.

The 6 items are drawn from canonical ICU literature and sit in the
"narrow but representative" zone the EHRFlowBench paper argues for:
each one is small enough to cohort-build deterministically, but each
one trips a *different* ICU rule — so a generic agent that lucks
into one will not luck into all six.

| key                     | predictor → outcome     | pitfall the agent must respect |
|-------------------------|-------------------------|--------------------------------|
| sofa2_mortality         | sofa2 → death           | ordinal; sofa2==0 is missingness |
| aki_kdigo_mortality     | kdigo_stage → death     | ordinal stage; right-skewed creat |
| lactate_mortality       | lact → death            | log-skew; report median, not mean |
| vasopressor_mortality   | vaso → death            | binary intervention windowing   |
| map_mortality           | map → death (negative!) | continuous; sign matters        |
| gcs_mortality           | gcs → death (negative!) | ordinal; never mean()           |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple


@dataclass
class BenchItem:
    """One benchmark question + its synthetic cohort + expected answer."""

    key: str
    name: str
    research_question: str
    target_outcome: str
    primary_predictor: str
    expected_or_direction: int  # +1 (positive) or -1 (negative association)

    # Returns a (cohort_dataframe, label) tuple. Deterministic given seed.
    cohort_factory: Callable[[int], "object"]

    # Substrings (case-insensitive) that a correctly-running ICU-aware
    # agent should surface in the validator findings list. Each string
    # only needs to appear in ONE finding to count as a hit.
    expected_finding_substrings: List[str] = field(default_factory=list)

    # Optional: the inclusion-criteria phrasing for the manuscript.
    inclusion_criteria: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cohort factories. Each returns a pandas DataFrame with one row per stay.
# The factories live here (not in the harness) so reviewers can read the
# ground truth alongside the question.
# ---------------------------------------------------------------------------


def _common_demographics(rng, n: int):
    age = rng.normal(65, 14, n).clip(18, 95)
    sex = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])
    return age, sex


def _sofa2_cohort(seed: int):
    """The canonical SOFA-2 mortality cohort with the missingness anomaly."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = 1000
    age, sex = _common_demographics(rng, n)
    base = rng.integers(1, 14, size=n, endpoint=False)
    miss = rng.random(n) < 0.10
    truly_low = rng.random(n) < 0.04
    sofa2 = np.where(miss, 0, np.where(truly_low, 0, base))
    logit = -3.5 + 0.18 * sofa2 + 0.012 * (age - 65) + np.where(miss, 1.5, 0.0)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    los = rng.gamma(2.0, 1.5 + 0.15 * sofa2, size=n).clip(0.1, 60)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex, "sofa2": sofa2,
        "los_icu": los, "death": death,
    })


def _aki_cohort(seed: int):
    """KDIGO AKI stage → mortality, with a right-skewed creatinine."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 17)
    n = 1000
    age, sex = _common_demographics(rng, n)
    # KDIGO stage 0–3 with prevalence 0.55/0.20/0.15/0.10
    stage = rng.choice([0, 1, 2, 3], size=n, p=[0.55, 0.20, 0.15, 0.10])
    creat = rng.lognormal(0.05 + 0.35 * stage, 0.4, size=n).clip(0.1, 12)
    logit = -3.0 + 0.6 * stage + 0.012 * (age - 65)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex,
        "kdigo_stage": stage.astype(int),
        "creat": creat,
        "death": death,
    })


def _lactate_cohort(seed: int):
    """Lactate trajectory → mortality. Right-skewed lab classic."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 29)
    n = 1000
    age, sex = _common_demographics(rng, n)
    lact = rng.lognormal(0.7, 0.65, size=n).clip(0.5, 25)
    logit = -3.5 + 0.30 * np.log(lact) + 0.010 * (age - 65)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex,
        "lact": lact,
        "death": death,
    })


def _vasopressor_cohort(seed: int):
    """Any-vasopressor exposure → mortality. Binary intervention."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 41)
    n = 1000
    age, sex = _common_demographics(rng, n)
    vaso = (rng.random(n) < 0.32).astype(int)
    logit = -3.0 + 1.1 * vaso + 0.012 * (age - 65)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex,
        "vaso": vaso,
        "death": death,
    })


def _map_cohort(seed: int):
    """Mean arterial pressure → mortality (NEGATIVE association)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 53)
    n = 1000
    age, sex = _common_demographics(rng, n)
    map_v = rng.normal(75, 14, size=n).clip(40, 130)
    logit = -3.0 - 0.04 * (map_v - 75) + 0.012 * (age - 65)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex,
        "map": map_v,
        "death": death,
    })


def _gcs_cohort(seed: int):
    """Glasgow Coma Scale → mortality (NEGATIVE; ordinal pitfall)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed + 61)
    n = 1000
    age, sex = _common_demographics(rng, n)
    # GCS 3–15; skew toward 13–15
    gcs = rng.choice(range(3, 16), size=n,
                     p=np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06,
                                 0.07, 0.08, 0.09, 0.12, 0.18, 0.21]) /
                       np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06,
                                 0.07, 0.08, 0.09, 0.12, 0.18, 0.21]).sum())
    logit = -3.5 - 0.20 * (gcs - 11) + 0.012 * (age - 65)
    death = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "stay_id": np.arange(1, n + 1),
        "age": age, "sex": sex,
        "gcs": gcs.astype(int),
        "death": death,
    })


# ---------------------------------------------------------------------------
# Bench items
# ---------------------------------------------------------------------------


BENCH_ITEMS: List[BenchItem] = [
    BenchItem(
        key="sofa2_mortality",
        name="Admission SOFA-2 → ICU mortality",
        research_question="Is admission SOFA-2 score associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="sofa2",
        expected_or_direction=+1,
        cohort_factory=_sofa2_cohort,
        expected_finding_substrings=[
            "non-monotonic", "sofa2", "missingness",
        ],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years",
                            "ICU LoS ≥ 6 hours"],
    ),
    BenchItem(
        key="aki_kdigo_mortality",
        name="KDIGO AKI stage → ICU mortality",
        research_question="Is peak first-24h KDIGO AKI stage associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="kdigo_stage",
        expected_or_direction=+1,
        cohort_factory=_aki_cohort,
        expected_finding_substrings=["creat"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
    ),
    BenchItem(
        key="lactate_mortality",
        name="Admission lactate → ICU mortality",
        research_question="Is admission lactate associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="lact",
        expected_or_direction=+1,
        cohort_factory=_lactate_cohort,
        expected_finding_substrings=["lact"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
    ),
    BenchItem(
        key="vasopressor_mortality",
        name="Any-vasopressor exposure → ICU mortality",
        research_question="Is any-vasopressor exposure within the first 24h associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="vaso",
        expected_or_direction=+1,
        cohort_factory=_vasopressor_cohort,
        expected_finding_substrings=[],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
    ),
    BenchItem(
        key="map_mortality",
        name="Mean arterial pressure → ICU mortality (negative)",
        research_question="Is admission mean arterial pressure associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="map",
        expected_or_direction=-1,
        cohort_factory=_map_cohort,
        expected_finding_substrings=[],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
    ),
    BenchItem(
        key="gcs_mortality",
        name="Worst GCS → ICU mortality (negative; ordinal pitfall)",
        research_question="Is the worst Glasgow Coma Scale within the first 24h associated with ICU mortality?",
        target_outcome="death",
        primary_predictor="gcs",
        expected_or_direction=-1,
        cohort_factory=_gcs_cohort,
        expected_finding_substrings=["gcs", "ordinal"],
        inclusion_criteria=["First ICU admission", "Age ≥ 18 years"],
    ),
]


__all__ = ["BENCH_ITEMS", "BenchItem"]
