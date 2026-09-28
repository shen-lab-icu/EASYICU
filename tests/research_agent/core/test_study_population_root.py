"""The host root that republishes the study cohort under its own name.

A landmark primary analysis runs on the stays alive and observed at the
landmark.  A step that must report on the population the study selected reads
that population from the host's interpretation-free root, which copies the run
cohort byte for byte under ``cohort:study_population`` -- a name that is never
the locked primary cohort, so no binding surface substitutes or projects it.
The legacy root that publishes ``table:analysis_cohort`` renders exactly as
before.  Every frame here is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from easyicu.research_agent.contracts.primary_cohort import (
    STUDY_POPULATION_PRODUCTS,
    is_host_bound_cohort_publisher,
    step_cohort_population,
    study_population_product_for,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    exposure_outcome_distribution_executor_owns_step,
    run_exposure_outcome_distribution_from_env,
)
from easyicu.research_agent.execution.runners.host_bound_cohort_executor import (
    host_bound_cohort_executor_code,
    host_bound_cohort_executor_owns_step,
)
from easyicu.research_agent.planning.cohort_contract import CohortDefinition
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

#: The legacy root's rendered script, digest pinned before the study
#: population existed: publishing a second product must not change it.
LEGACY_ROOT_SCRIPT_SHA256 = "14819dbcaf01e5152614eeba922c811a7d1eb0ee5228c963110418f061d39388"


def _root(product: str = "cohort:study_population", **updates) -> AnalysisStep:
    payload = {
        "step_id": "study_population",
        "intent": "Publish the run cohort.",
        "method": "host_materialized_locked_cohort",
        "planned_analysis_role": "auxiliary",
        "inputs": [],
        "expected_outputs": [product],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _distribution(cohort_input: str = "cohort:study_population") -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "exposure_occurrence",
            "intent": "Report how often each exposure level occurs in the study cohort.",
            "method": "descriptive",
            "planned_analysis_role": "secondary",
            "inputs": ["stage", "death", cohort_input],
            "expected_outputs": ["table:exposure_outcome_distribution"],
            "scientific_capability": "descriptive_exposure_outcome_distribution_v1",
            "descriptive_claim": {
                "unresolved_limitations": ["post_baseline_exposure_opportunity_unresolved"]
            },
            "exposure_outcome_distribution_spec": _SPEC,
        }
    )


_SPEC = {
    "exposure": "stage",
    "exposure_levels": [0, 1, 2],
    "outcome": "death",
    "outcome_levels": [0, 1],
    "outcome_positive_value": 1,
    "level_match_policy": "exact_typed",
    "denominator_policy": "all_declared_rows",
    "missing_exposure_policy": "exclude_from_denominator",
    "missing_outcome_policy": "fail_closed",
    "confidence_level": 0.95,
}


# ---------------------------------------------------------------------------
# Naming: the study population is never the locked cohort
# ---------------------------------------------------------------------------


def test_the_study_population_takes_the_spelling_the_locked_cohort_cannot_claim() -> None:
    assert study_population_product_for("adult_icu_stays") == "cohort:study_population"
    assert study_population_product_for("study_population") == (
        "cohort:eligible_study_population"
    )
    assert study_population_product_for("Study Population") == (
        "cohort:eligible_study_population"
    )


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "product", ["table:analysis_cohort", *STUDY_POPULATION_PRODUCTS]
)
def test_the_root_owns_exactly_one_published_product(product) -> None:
    assert host_bound_cohort_executor_owns_step(_root(product))
    assert is_host_bound_cohort_publisher(_root(product))


@pytest.mark.parametrize(
    "updates",
    [
        {"expected_outputs": ["cohort:study_population", "table:cohort_flow"]},
        {"expected_outputs": ["cohort:some_other_population"]},
        {"inputs": ["artifact:analysis_cohort"]},
        {"method": "cohort_definition_and_attrition"},
        {"planned_analysis_role": "secondary"},
    ],
)
def test_any_other_shape_is_not_the_root(updates) -> None:
    step = _root(**updates)
    assert not host_bound_cohort_executor_owns_step(step)


def test_the_legacy_root_renders_exactly_as_before() -> None:
    legacy = _root("table:analysis_cohort", step_id="00_cohort", intent="Publish the run cohort.")
    script = host_bound_cohort_executor_code(legacy)

    assert hashlib.sha256(script.encode()).hexdigest() == LEGACY_ROOT_SCRIPT_SHA256


def test_the_study_root_copies_the_run_cohort_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "run_cohort.parquet"
    pd.DataFrame({"stay_id": [1, 2, 3], "stage": [0, 2, 1], "death": [0, 1, 0]}).to_parquet(
        source, index=False
    )
    out_dir = tmp_path / "steps" / "study_population" / "outputs"
    script = tmp_path / "root.py"
    script.write_text(host_bound_cohort_executor_code(_root()), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        env={**os.environ, "COHORT_PARQUET": str(source), "STEP_OUT_DIR": str(out_dir)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    published = out_dir / "study_population.parquet"
    assert published.read_bytes() == source.read_bytes()
    summary = json.loads((out_dir / "step_summary.json").read_text(encoding="utf-8"))
    assert summary["output_files"] == {"cohort:study_population": "study_population.parquet"}
    assert summary["n_source"] == 3
    assert summary["source_sha256"] == summary["output_sha256"]


# ---------------------------------------------------------------------------
# Which population a step's cohort input carries
# ---------------------------------------------------------------------------


def _plan(*steps: AnalysisStep, cohort_name: str = "adult_icu_stays") -> AnalysisPlan:
    return AnalysisPlan(
        research_question="How often does the exposure occur?",
        cohort=CohortDefinition(name=cohort_name, locked_at="2026-09-26T00:00:00Z"),
        steps=list(steps),
    )


def _cohort_step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "cohort_definition",
        "intent": "Select the analysis cohort and record its flow.",
        "method": "cohort_definition_and_attrition",
        "planned_analysis_role": "auxiliary",
        "inputs": ["stay_id"],
        "expected_outputs": ["artifact:analysis_cohort", "table:cohort_flow"],
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def test_the_root_and_an_unsigned_cohort_step_carry_the_study_cohort() -> None:
    root_plan = _plan(_root(), _distribution())
    assert step_cohort_population(step=root_plan.steps[1], plan=root_plan) == "study_cohort"

    unsigned = _plan(_cohort_step(), _distribution("artifact:analysis_cohort"))
    assert step_cohort_population(step=unsigned.steps[1], plan=unsigned) == "study_cohort"


def test_a_signed_runtime_cohort_is_restricted() -> None:
    signed = _plan(
        _cohort_step(icu_rule_refs=["scientific_runtime_contract:" + "a" * 64]),
        _distribution("artifact:analysis_cohort"),
    )

    assert step_cohort_population(step=signed.steps[1], plan=signed) == "restricted"


def test_no_producer_or_two_producers_name_no_population() -> None:
    orphan = _plan(_distribution())
    assert step_cohort_population(step=orphan.steps[0], plan=orphan) == "unbound"

    twice = _plan(_root(), _root(step_id="study_population_again"), _distribution())
    assert step_cohort_population(step=twice.steps[2], plan=twice) == "ambiguous"


# ---------------------------------------------------------------------------
# The distribution reads the republished cohort through its typed input
# ---------------------------------------------------------------------------


def test_the_distribution_reads_the_study_population_it_is_bound_to(
    monkeypatch, tmp_path: Path
) -> None:
    frame = pd.DataFrame(
        {
            "stay_id": range(1, 9),
            "stage": [0, 0, 1, 1, 2, 2, None, 0],
            "death": [0, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "exposure_occurrence" / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "steps" / "study_population" / "outputs" / "study_population.parquet"
    cohort_path.parent.mkdir(parents=True)
    frame.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    identity = {
        "input_key": "cohort:study_population",
        "declared_kind": "cohort",
        "product": "study_population",
        "evidence_id": "ev-study-population",
        "sha256": digest,
    }
    manifest = run_dir / "resolved_inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "step_id": "exposure_occurrence",
                "inputs": {
                    "cohort:study_population": {
                        "relative_path": str(cohort_path.relative_to(run_dir)),
                        "sha256": digest,
                        "declared_kind": "cohort",
                        "product": "study_population",
                        "evidence_id": "ev-study-population",
                        "identity_row": identity,
                        "product_contract": {
                            "columns": list(frame.columns),
                            "row_count": int(len(frame)),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(manifest))
    assert exposure_outcome_distribution_executor_owns_step(_distribution())

    summary = run_exposure_outcome_distribution_from_env(
        spec_payload=_SPEC, typed_cohort_input="cohort:study_population"
    )

    assert summary["typed_cohort_input"] == "cohort:study_population"
    prevalence = summary["descriptive_estimates"]["exposure_prevalence"]
    assert [(row["level"], row["n"], row["denominator"]) for row in prevalence] == [
        (0, 3, 7),
        (1, 2, 7),
        (2, 2, 7),
    ]
    # The unclassified row leaves the denominator and its count is reported.
    assert summary["source_row_count_reconciliation"]["excluded_missing_exposure_rows"] == 1
