"""A prediction is compared with an existing score on the same validation stays.

The comparison reads the primary's sealed scores and the cohort's comparator
columns, aligned by the primary's own row identity.  One owner decides what
a comparator is (a probability of an outcome, or a score whose direction it
states) and the window it was computed over
(``planning.benchmark_comparator``); the intervals follow how the compared
stays depend.  Synthetic data only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.contracts.executor_stop import (
    EXECUTOR_STOP_RECORD_NAME,
    ExecutorStop,
)
from easyicu.research_agent.contracts.prediction_execution import (
    PREDICTION_BENCHMARK_ACTION,
    static_prediction_benchmark_columns,
    static_prediction_owns_step,
)
from easyicu.research_agent.planning.benchmark_comparator import (
    COMPARATOR_READINGS,
    benchmark_comparator_facts,
)
from easyicu.research_agent.execution.runners.prediction_model_executor import (
    PREDICTION_SCORES_PRODUCT,
    BenchmarkComparisonError,
    prediction_model_consumed_input_keys,
    prediction_model_executor_code,
    run_prediction_benchmark_comparison,
    run_prediction_model,
    run_prediction_score_analysis,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    UserPreferences,
    VariableRole,
)

_PROBABILITY = "apache_iv_pred_hosp_mort"
_SCORE = "apache_iv"


def _context(
    row_count: int,
    *,
    outcome_concept: str = "death",
    hours: float = 24.0,
    extra_variables: tuple[ConceptDescriptor, ...] = (),
):
    return ResearchContext(
        research_question="Predict in-hospital death and compare with APACHE IVa.",
        cohort=CohortDescriptor(
            cohort_name="benchmark_fixture",
            database="synthetic",
            n_stays=row_count,
            id_columns=["patient_stay_id"],
            outcome_columns=["death"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "a" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="age", dtype="float64"),
            ConceptDescriptor(name="marker", dtype="float64"),
            ConceptDescriptor(name=_PROBABILITY, dtype="float64", source_concept=_PROBABILITY),
            ConceptDescriptor(name=_SCORE, dtype="float64", source_concept=_SCORE),
            ConceptDescriptor(name="hr", dtype="float64", source_concept="hr"),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                source_concept=outcome_concept,
                observed_domain={"n_unique": 2, "is_binary": True, "levels": [0, 1]},
            ),
            *extra_variables,
        ],
        target_outcome="death",
        user_preferences=UserPreferences(
            data_constraints=json.dumps(
                {
                    "materialization_window": {
                        "role": "outer_observation_window",
                        "anchor": "icu_admission",
                        "hours": hours,
                    }
                }
            )
        ),
    )


def _frame(*, repeats: bool) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    patients = 160
    stays = np.where(np.arange(patients) % 3 == 0, 2, 1) if repeats else np.ones(patients, int)
    subject = np.repeat(np.arange(patients), stays)
    stay = np.concatenate([np.arange(count) + 1 for count in stays])
    age = rng.normal(64, 12, subject.size)
    marker = rng.normal(0, 1, subject.size)
    risk = -1.6 + 0.03 * (age - 60) + 0.9 * marker
    death = rng.binomial(1, 1.0 / (1.0 + np.exp(-risk)))
    death[:10] = np.arange(10) % 2
    probability = 1.0 / (1.0 + np.exp(-(risk + rng.normal(0, 1.2, subject.size))))
    frame = pd.DataFrame(
        {
            "patient_stay_id": [f"p{p}:s{s}" for p, s in zip(subject, stay, strict=True)],
            "age": age,
            "marker": marker,
            "death": death,
            _PROBABILITY: probability,
            _SCORE: np.round(40 + 60 * probability + rng.normal(0, 8, subject.size)),
            "hr": rng.normal(90, 15, subject.size),
        }
    )
    frame.loc[frame.index[::11], _PROBABILITY] = np.nan
    return frame


def _binding(key: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "produced_by_step": f"source_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {"input_key": key, "mode": "all_rows", "artifact_sha256": digest},
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "produced_by_step": f"source_{product}",
            "sha256": digest,
        },
    }


def _compare(tmp_path: Path, frame: pd.DataFrame, comparators, **context_fields):
    (tmp_path / "research_context.json").write_text(
        _context(len(frame), **context_fields).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    primary_dir = tmp_path / "primary"
    run_prediction_model(
        frame=frame,
        declared_columns=("age", "marker", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=primary_dir,
        run_dir=tmp_path,
        step_id="primary_model",
    )
    scores = pd.read_csv(primary_dir / "prediction_scores.csv")
    summary = run_prediction_benchmark_comparison(
        frame=frame,
        comparator_columns=comparators,
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=tmp_path / "benchmark",
        run_dir=tmp_path,
        resolved_inputs={
            "step_id": "benchmark",
            "inputs": {
                PREDICTION_SCORES_PRODUCT: _binding(
                    PREDICTION_SCORES_PRODUCT, scores, primary_dir / "prediction_scores.csv"
                )
            },
        },
        step_id="benchmark",
    )
    table = pd.read_csv(
        tmp_path / "benchmark" / "benchmark_comparison.csv", float_precision="round_trip"
    )
    return summary, table, scores


def _step(**fields) -> AnalysisStep:
    values = {
        "step_id": "benchmark",
        "planned_analysis_role": "secondary",
        "intent": "Compare the model with APACHE IVa on the same validation stays.",
        "inputs": [_PROBABILITY, "artifact:analysis_cohort", PREDICTION_SCORES_PRODUCT],
        "expected_outputs": ["table:benchmark_comparison"],
        "method": "paired AUROC comparison",
        "scientific_action_id": PREDICTION_BENCHMARK_ACTION,
    }
    values.update(fields)
    return AnalysisStep(**values)


def test_the_static_prediction_owner_claims_only_the_exact_comparison_shape() -> None:
    step = _step()
    assert static_prediction_owns_step(step)
    assert static_prediction_benchmark_columns(step) == (_PROBABILITY,)
    assert prediction_model_consumed_input_keys(step) == (
        "artifact:analysis_cohort",
        PREDICTION_SCORES_PRODUCT,
    )
    code = prediction_model_executor_code(step)
    assert "run_prediction_benchmark_comparison" in code
    # The comparison reads the primary's cohort, which is one of two typed inputs.
    assert "typed_cohort_input='artifact:analysis_cohort'" in code
    # No comparator, too many, the scores missing or first, a typed input
    # before a comparator, another typed input in the cohort's or the scores'
    # place, or the primary's role:
    for wrong in (
        _step(inputs=["artifact:analysis_cohort", PREDICTION_SCORES_PRODUCT]),
        _step(inputs=["a", "b", "c", "d", "artifact:analysis_cohort", PREDICTION_SCORES_PRODUCT]),
        _step(inputs=[_PROBABILITY, "artifact:analysis_cohort"]),
        _step(inputs=[_PROBABILITY, PREDICTION_SCORES_PRODUCT, "artifact:analysis_cohort"]),
        _step(inputs=[_PROBABILITY, "table:validation", PREDICTION_SCORES_PRODUCT]),
        _step(inputs=["artifact:analysis_cohort", _PROBABILITY, PREDICTION_SCORES_PRODUCT]),
        _step(inputs=[_PROBABILITY, "artifact:analysis_cohort", "table:validation"]),
        _step(planned_analysis_role="primary"),
    ):
        assert not static_prediction_owns_step(wrong)
    with pytest.raises(RuntimeError):
        run_prediction_score_analysis(
            action_id=PREDICTION_BENCHMARK_ACTION,
            out_dir=Path("."),
            run_dir=Path("."),
            resolved_inputs={},
            step_id="benchmark",
        )


def test_a_probability_of_the_outcome_is_compared_in_discrimination_and_calibration(
    tmp_path: Path,
) -> None:
    summary, table, scores = _compare(tmp_path, _frame(repeats=False), (_PROBABILITY,))

    validation = scores.loc[scores["split"].eq("validation")]
    auroc = table.loc[table["metric"].eq("auroc")].iloc[0]
    assert list(table["metric"]) == [
        "auroc",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
    ]
    assert auroc["comparator_kind"] == "probability"
    assert auroc["interval_method"] == "delong_paired_normal_95pct"
    assert auroc["validation_n"] == len(validation)
    assert auroc["comparison_n"] + auroc["comparator_missing_n"] == len(validation)
    assert auroc["comparator_missing_n"] > 0
    assert auroc["difference"] == pytest.approx(auroc["model_value"] - auroc["comparator_value"])
    assert auroc["difference_ci_low"] < auroc["difference"] < auroc["difference_ci_high"]
    assert not np.isnan(auroc["p_value"])
    assert auroc["calibration_status"] == "compared"
    assert auroc["outcome_concept"] == "death" and auroc["comparator_predicts"] == "death"
    # APACHE IVa reads the first ICU day; this model predicts at 24 h.
    assert auroc["comparator_information_window"] == "icu_admission[0,24]h"
    assert auroc["information_window_relation"] == "same"
    assert not bool(auroc["information_window_differs"])
    reported = summary["reportable_benchmark_comparison"]["comparisons"][0]
    assert reported["auroc_difference"] == pytest.approx(auroc["difference"])
    # The reportable block holds the table's own values, under the names the
    # manuscript audit reads (``*.auroc``, ``*.auroc_ci_low``, ``*.brier_score``).
    brier = table.loc[table["metric"].eq("brier_score")].iloc[0]
    for side, prefix in (("model", "model_"), ("comparator", "comparator_")):
        assert set(reported[side]) == {
            "auroc", "auroc_ci_low", "auroc_ci_high",
            "brier_score", "calibration_intercept", "calibration_slope",
        }
        assert reported[side]["auroc"] == auroc[f"{prefix}value"]
        assert reported[side]["auroc_ci_low"] == auroc[f"{prefix}ci_low"]
        assert reported[side]["brier_score"] == brier[f"{prefix}value"]


def test_repeat_stays_compare_by_patient_resampling(tmp_path: Path) -> None:
    _summary, table, _scores = _compare(tmp_path, _frame(repeats=True), (_PROBABILITY,))

    auroc = table.loc[table["metric"].eq("auroc")].iloc[0]
    assert auroc["interval_method"] == "patient_stratified_bootstrap_percentile_95pct"
    assert auroc["bootstrap_n"] == 2000
    assert np.isnan(auroc["p_value"]) and np.isnan(auroc["z"])
    assert auroc["comparison_subject_n"] < auroc["comparison_n"]


@pytest.mark.parametrize(
    "comparator, outcome_concept, hours, reason, relation",
    [
        (_SCORE, "death", 24.0, "score_scale", "same"),
        (_PROBABILITY, "death_icu", 24.0, "predicts_another_outcome", "same"),
        (_PROBABILITY, "death", 6.0, "", "comparator_ends_after_prediction_time"),
        (_PROBABILITY, "death", 48.0, "", "comparator_ends_before_prediction_time"),
    ],
)
def test_what_the_comparison_cannot_claim_is_stated(
    tmp_path: Path,
    comparator: str,
    outcome_concept: str,
    hours: float,
    reason: str,
    relation: str,
) -> None:
    _summary, table, _scores = _compare(
        tmp_path,
        _frame(repeats=False),
        (comparator,),
        outcome_concept=outcome_concept,
        hours=hours,
    )

    first = table.iloc[0]
    assert first["metric"] == "auroc"
    assert (first["calibration_reason"] if isinstance(first["calibration_reason"], str) else "") == reason
    assert bool((table["metric"] != "auroc").any()) is (reason == "")
    assert first["information_window_relation"] == relation
    assert bool(first["information_window_differs"]) is (relation != "same")


def test_a_summary_column_is_compared_over_its_own_window(tmp_path: Path) -> None:
    # The worst SOFA over the first 6 h: a score whose dictionary entry states
    # no window, summarized over the window the context records for it.
    frame = _frame(repeats=False)
    frame["sofa_max"] = np.round(frame[_SCORE] / 10)
    _summary, table, _scores = _compare(
        tmp_path,
        frame,
        ("sofa_max",),
        extra_variables=(
            ConceptDescriptor(
                name="sofa_max", dtype="float64", source_concept="sofa",
                analysis_window="icu_admission[0,6]h",
            ),
        ),
    )

    [auroc] = table.to_dict("records")
    assert (auroc["comparator_concept"], auroc["comparator_kind"]) == ("sofa", "score")
    assert auroc["comparator_information_window"] == "icu_admission[0,6]h"
    assert auroc["information_window_relation"] == "comparator_ends_before_prediction_time"


def test_a_probability_outside_zero_and_one_stops_the_comparison(tmp_path: Path) -> None:
    frame = _frame(repeats=False)
    frame.loc[frame.index[5], _PROBABILITY] = 1.5
    with pytest.raises(BenchmarkComparisonError) as stopped:
        _compare(tmp_path, frame, (_PROBABILITY,))
    assert stopped.value.code == "benchmark_probability_out_of_range"


def test_a_column_the_dictionary_does_not_orient_is_not_a_comparator(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkComparisonError) as stopped:
        _compare(tmp_path, _frame(repeats=False), ("hr",))
    assert stopped.value.code == "benchmark_comparator_unsupported"


def _recorded_stop(out_dir: Path) -> dict:
    return json.loads((out_dir / EXECUTOR_STOP_RECORD_NAME).read_text(encoding="utf-8"))


def test_a_comparator_no_validation_stay_records_stops_the_comparison(tmp_path: Path) -> None:
    # A score its source lists but holds no value of: there is nothing to
    # compare the model with, whatever the earlier split rows hold.
    frame = _frame(repeats=False)
    frame[_PROBABILITY] = np.nan
    with pytest.raises(ExecutorStop) as stopped:
        _compare(tmp_path, frame, (_PROBABILITY,))
    assert stopped.value.reason_code == "benchmark_comparator_unobserved"
    assert repr(_PROBABILITY) in str(stopped.value)
    assert _recorded_stop(tmp_path / "benchmark")["reason_code"] == (
        "benchmark_comparator_unobserved"
    )


def test_a_predictor_no_development_stay_records_stops_the_model(tmp_path: Path) -> None:
    # The imputer would drop the predictor with only a warning, so the model
    # fitted would not be the one planned.  Its validation values do not help.
    frame = _frame(repeats=False)
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
    _summary, _table, scores = _compare(tmp_path / "first", frame, (_SCORE,))
    development = set(scores.loc[scores["split"].eq("development"), "unit_id"].astype(str))
    # The owner names a stay by its identity's type and value.
    in_development = frame["patient_stay_id"].map(lambda value: f"str:{value!r}")
    frame.loc[in_development.isin(development), "marker"] = np.nan
    assert frame["marker"].notna().any()
    with pytest.raises(ExecutorStop) as stopped:
        _compare(tmp_path / "second", frame, (_SCORE,))
    assert stopped.value.reason_code == "prediction_predictor_unobserved"
    assert "'marker'" in str(stopped.value)
    assert _recorded_stop(tmp_path / "second" / "primary")["reason_code"] == (
        "prediction_predictor_unobserved"
    )


def test_what_a_comparator_is_is_read_beside_the_sealed_dictionary() -> None:
    """The comparator readings name dictionary concepts and add to them only.

    The dictionary is sealed with its extraction release, so a comparison's
    readings stay with their owner: each names a concept the dictionary
    defines, and the dictionary states none of them itself.
    """

    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=True)
    for concept, reading in COMPARATOR_READINGS.items():
        definition = dictionary.get(concept)
        assert definition is not None, concept
        assert set(reading) <= {"risk_direction", "predicts", "analysis_window"}
        assert not any(getattr(definition, key, None) for key in reading), concept
    probability = benchmark_comparator_facts(_PROBABILITY)
    assert (probability.kind, probability.predicts) == ("probability", "death")
    assert benchmark_comparator_facts(_SCORE).kind == "score"
    assert benchmark_comparator_facts("gcs").kind is None
