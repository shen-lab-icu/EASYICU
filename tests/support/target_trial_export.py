"""A synthetic export a target trial is extracted from, shared by its tests.

Test modules may not import one another (``tests/governance/test_test_organization.py``).
The stays of ``synthetic_target_trial_cohort`` are written as a typed native
export -- a stay-level outcome module, the ages, and one medication record
per started vasoactive drug and one lactate per measured stay before time
zero -- and acquired as Data Extraction acquires a trial: covariates summarized
over the hours before time zero, each treatment's onset read through the
grace period (``CompiledTargetTrial.acquisition_windows``), and the death,
its time, the ICU stay and the follow-up the emulation reads.  Each stay is
its patient's first, as the host's first-stay receipt states.  The trial asks
whether starting a vasoactive drug within six hours of time zero, six hours
after ICU admission, changes death by day 28 among adults with a lactate of
at least 2 mmol/L before time zero, adjusting for age and that lactate.
Zero real patient rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from easyicu.research_agent.acquisition.foundation import acquire_universe_for_question
from easyicu.research_agent.intake.materialized_metadata import (
    FIRST_ICU_STAY_RESTRICTION_SCHEMA,
)
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.planning.target_trial_spec import TargetTrialSpec
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.schema import ResearchContext
from tests.support.native_outcome_export import typed_native_export
from tests.support.target_trial import (
    GRACE,
    TIME_ZERO,
    synthetic_target_trial_cohort,
    target_trial_spec,
)

TREATMENTS = ("vaso_ind", "other_vaso")
QUESTION = (
    "Does starting a vasoactive drug within six hours of time zero change death "
    "by day 28?"
)
#: The windows a trial extraction reads (``acquisition_windows``).
COHORT_WINDOW = (0.0, float(TIME_ZERO))
ONSET_WINDOWS = {concept: (0.0, float(TIME_ZERO + GRACE)) for concept in TREATMENTS}


def target_trial_export(root: Path, *, n: int = 600, seed: int = 20261009) -> Path:
    """Write the synthetic stays as a typed native export at ``root``."""

    cohort = synthetic_target_trial_cohort(n=n, seed=seed)
    stays = cohort["stay_id"].astype(int)
    outcome = pd.DataFrame(
        {
            "stay_id": stays,
            "charttime": 0.0,
            "death": cohort["death"].astype(bool),
            "death_time": cohort["death_time"],
            "los_icu": cohort["los_icu"],
            "followup_days_28d": cohort["followup_days_28d"],
            "mort_28d": cohort["mort_28d"].astype(bool),
        }
    )
    rows: list[dict[str, Any]] = []
    for stay, norepinephrine, vasopressin, lactate in zip(
        stays,
        cohort["norepinephrine_onset_time"],
        cohort["vasopressin_onset_time"],
        cohort["lactate_max"],
    ):
        # Every stay is charted at admission, started or not.
        rows.append(
            {
                "stay_id": stay,
                "charttime": 0.0,
                "vaso_ind": False,
                "other_vaso": False,
                "lact": np.nan,
            }
        )
        for concept, onset in (
            ("vaso_ind", norepinephrine),
            ("other_vaso", vasopressin),
        ):
            if not np.isnan(onset):
                rows.append(
                    {
                        "stay_id": stay,
                        "charttime": float(onset),
                        "vaso_ind": concept == "vaso_ind",
                        "other_vaso": concept == "other_vaso",
                        "lact": np.nan,
                    }
                )
        if not np.isnan(lactate):
            rows.append(
                {
                    "stay_id": stay,
                    "charttime": 1.0,
                    "vaso_ind": False,
                    "other_vaso": False,
                    "lact": float(lactate),
                }
            )
    medications = pd.DataFrame(rows)
    for concept in TREATMENTS:
        medications[concept] = pd.array(medications[concept], dtype="boolean")
    return typed_native_export(
        root,
        outcome=outcome,
        outcome_concepts=["death", "los_icu", "followup_days_28d", "mort_28d"],
        longitudinal=medications,
        longitudinal_concepts=[*TREATMENTS, "lact"],
        statics=pd.DataFrame({"stay_id": stays, "age": cohort["age"]}),
    )


def acquire_target_trial(
    export: Path,
    output_dir: Path,
    *,
    cohort_window: tuple[float, float] = COHORT_WINDOW,
    onset_windows: Optional[dict[str, tuple[float, float]]] = None,
) -> Path:
    """The universe Data Extraction acquires for the trial; its parquet path."""

    result = acquire_universe_for_question(
        export_dir=export,
        question=QUESTION,
        llm=ScriptedMockLLMClient([]),
        output_dir=output_dir,
        target_outcome="mort_28d",
        outcome_concepts=["mort_28d", "death"],
        # The ICU stay and the follow-up are each stay's own value.
        required_feature_concepts=[*TREATMENTS, "lact", "los_icu", "followup_days_28d"],
        static_concepts=["age"],
        concept_selection_authority="host_exact",
        cohort_window=cohort_window,
        emit_trajectory=False,
        event_onset_windows=onset_windows,
    )
    if result.blocked or result.universe_path is None:
        raise AssertionError("the synthetic trial extraction was blocked")
    return Path(result.universe_path)


def export_context(universe: Path) -> ResearchContext:
    """The research context a run builds on the acquired universe.

    The host kept each patient's first ICU stay, so the bootstrap resamples
    stays; its receipt is what the dependence owner reads.
    """

    context = build_research_context(
        research_question=QUESTION,
        cohort=universe,
        cohort_name="trial",
        database="miiv",
        target_outcome="mort_28d",
        id_columns=("stay_id",),
        outcome_columns=("mort_28d",),
    )
    receipt = {
        "schema_version": FIRST_ICU_STAY_RESTRICTION_SCHEMA,
        "coordinate_sha256": "f" * 64,
    }
    return context.model_copy(
        update={
            "cohort": context.cohort.model_copy(
                update={
                    "provenance": {
                        **context.cohort.provenance,
                        "first_icu_stay_restriction": receipt,
                    }
                }
            )
        }
    )


def export_trial_spec() -> TargetTrialSpec:
    """The trial over the export: indicated by lactate, adjusted for age and it."""

    return target_trial_spec(
        indication={
            "quote": "a lactate of at least 2 mmol/L",
            "source": "question",
            "criterion_ids": ["c2"],
        },
        confounders=[
            {
                "name": "age",
                "source": "question",
                "clinical_rationale": "Older patients are started later and die more often.",
            },
            {
                "name": "lact_max",
                "source": "conversation",
                "clinical_rationale": "A higher lactate prompts the start and predicts death.",
            },
        ],
    )


def export_trial_population() -> PopulationSpec:
    return PopulationSpec.model_validate(
        {
            "criteria": [
                {
                    "id": "c1",
                    "source": "question",
                    "role": "include",
                    "quote": "adults",
                    "kind": "age_years",
                    "min_years": 18,
                },
                {
                    "id": "c2",
                    "source": "question",
                    "role": "include",
                    "quote": "a lactate of at least 2 mmol/L",
                    "kind": "measurement",
                    "concept": "lact",
                    "summary": "max",
                    "window": {"start_hours": 0, "end_hours": TIME_ZERO},
                    "op": ">=",
                    "value": 2.0,
                    "unit": "mmol/L",
                },
            ]
        }
    )


__all__ = [
    "COHORT_WINDOW",
    "ONSET_WINDOWS",
    "QUESTION",
    "TREATMENTS",
    "acquire_target_trial",
    "export_context",
    "export_trial_population",
    "export_trial_spec",
    "target_trial_export",
]
