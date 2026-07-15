"""Primary analysis-cohort input-role and integrity regressions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from easyicu.research_agent.cohort_schema import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.declared_product_contract import (
    primary_analysis_cohort_integrity_findings,
    primary_analysis_cohort_producer_uses_universe,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _definition() -> CohortDefinition:
    window = TimeWindow(
        anchor="icu_admit",
        start_offset_hours=0,
        end_offset_hours=24,
    )
    return CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate(
                concept_id="age",
                time_window=window,
                aggregation="first",
                op=">=",
                value=18,
            ),
            ConceptPredicate(
                concept_id="los_icu",
                time_window=window,
                aggregation="first",
                op=">=",
                value=1,
            ),
        ),
    )


def _cohort_step(*, method: str = "cohort_definition_and_attrition") -> AnalysisStep:
    return AnalysisStep(
        step_id="01_cohort",
        intent="Apply the locked primary cohort and report attrition.",
        inputs=["stay_id", "age", "los_icu"],
        expected_outputs=[
            "artifact:analysis_cohort",
            "table:cohort_flow",
            "table:cohort_attrition",
        ],
        method=method,
    )


def _plan(step: AnalysisStep, *extra_steps: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan(
        research_question="Describe a locked ICU analysis cohort.",
        cohort=_definition(),
        steps=[step, *extra_steps],
    )


def _write_authorities(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    universe = pd.DataFrame(
        {
            "stay_id": list(range(1, 9)),
            "age": [17, 18, 30, 40, 50, 60, 70, 80],
            "los_icu": [2.0, 0.5, 1.0, 2.0, 0.2, 3.0, 1.5, 0.8],
            "death": [0, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    authoritative = universe[universe["stay_id"].isin([3, 4, 6, 7])].copy()
    universe_path = tmp_path / "cohort.parquet"
    authoritative_path = tmp_path / "cohort_analysis.parquet"
    universe.to_parquet(universe_path, index=False)
    authoritative.to_parquet(authoritative_path, index=False)
    return universe_path, authoritative_path, authoritative


def _write_outputs(
    out_dir: Path,
    *,
    produced: pd.DataFrame,
    universe_n: int,
    final_n: int,
) -> dict:
    out_dir.mkdir(parents=True)
    produced.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    flow = pd.DataFrame(
        {
            "criterion": ["universe", "adult", "final_analysis_cohort"],
            "n_remaining": [universe_n, 7 if universe_n == 8 else final_n, final_n],
            "n_excluded_at_step": [
                0,
                1 if universe_n == 8 else 0,
                3 if universe_n == 8 else 0,
            ],
        }
    )
    flow.to_csv(out_dir / "cohort_flow.csv", index=False)
    flow.to_csv(out_dir / "cohort_attrition.csv", index=False)
    return {
        "status": "completed",
        "n_universe": universe_n,
        "n_final_analysis_cohort": final_n,
        "output_files": [
            "analysis_cohort.parquet",
            "cohort_flow.csv",
            "cohort_attrition.csv",
        ],
    }


def test_unique_closed_primary_cohort_producer_uses_raw_universe() -> None:
    step = _cohort_step()

    assert primary_analysis_cohort_producer_uses_universe(step=step, plan=_plan(step))


def test_plain_attrition_role_uses_raw_universe() -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )

    assert primary_analysis_cohort_producer_uses_universe(step=step, plan=_plan(step))


def test_universe_binding_rejects_alternative_effect_and_duplicate_producers() -> None:
    for method in (
        "cohort_definition_sensitivity",
        "mixed_effects_regression",
    ):
        step = _cohort_step(method=method)
        assert not primary_analysis_cohort_producer_uses_universe(
            step=step, plan=_plan(step)
        )

    step = _cohort_step()
    duplicate = AnalysisStep(
        step_id="02_duplicate",
        intent="Duplicate cohort output.",
        expected_outputs=["dataset:analysis_cohort"],
        method="cohort_definition",
    )
    assert not primary_analysis_cohort_producer_uses_universe(
        step=step, plan=_plan(step, duplicate)
    )


def test_ambiguous_or_mixed_primary_cohort_owner_fails_closed(tmp_path: Path) -> None:
    step = _cohort_step()
    duplicate = AnalysisStep(
        step_id="02_duplicate",
        intent="Duplicate cohort output.",
        expected_outputs=["dataset:analysis_cohort"],
        method="cohort_definition",
    )
    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=_plan(step, duplicate),
        step_summary={},
        out_dir=tmp_path,
        universe_path=tmp_path / "cohort.parquet",
        authoritative_cohort_path=tmp_path / "cohort_analysis.parquet",
    )
    assert findings[0].detail["issue"] == "primary_cohort_product_owner_ambiguous"

    foreign = step.model_copy(
        update={
            "expected_outputs": [
                *step.expected_outputs,
                "statistic:primary_effect",
            ]
        }
    )
    findings = primary_analysis_cohort_integrity_findings(
        step=foreign,
        plan=_plan(foreign),
        step_summary={},
        out_dir=tmp_path,
        universe_path=tmp_path / "cohort.parquet",
        authoritative_cohort_path=tmp_path / "cohort_analysis.parquet",
    )
    assert findings[0].detail["issue"] == "primary_cohort_product_owner_ambiguous"


def test_truthful_cohort_and_legacy_free_text_attrition_pass(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative.assign(derived_stage=[0, 1, 2, 3]),
        universe_n=8,
        final_n=4,
    )

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_exact_typed_product_keys_may_bind_noncanonical_filenames(
    tmp_path: Path,
) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    renames = {
        "analysis_cohort.parquet": "eligible_rows.parquet",
        "cohort_flow.csv": "flow_values.csv",
        "cohort_attrition.csv": "attrition_values.csv",
    }
    for before, after in renames.items():
        (out_dir / before).rename(out_dir / after)
    summary["output_files"] = {
        "artifact:analysis_cohort": renames["analysis_cohort.parquet"],
        "table:cohort_flow": renames["cohort_flow.csv"],
        "table:cohort_attrition": renames["cohort_attrition.csv"],
    }

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_filtered_cohort_cannot_masquerade_as_universe(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=4,
        final_n=4,
    )

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["issue"] == "cohort_denominator_mismatch"
    assert findings[0].detail["expected_universe_n"] == 8


def test_summary_synonymous_denominators_must_agree(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    summary["universe_n"] = 999

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "cohort_denominator_fields_disagree"


def test_summary_synonymous_denominators_must_all_be_integral(
    tmp_path: Path,
) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    summary["final_cohort_n"] = "not-a-count"

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "cohort_denominator_fields_nonintegral"


def test_matching_summary_synonymous_denominators_pass(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    summary.update(
        {
            "universe_n": 8.0,
            "n_input_universe": "8",
            "n_analysis_cohort": 4.0,
            "n_final_cohort": "4",
            "final_cohort_n": 4,
        }
    )

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_same_n_wrong_cohort_row_identities_fail_closed(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, _authoritative = _write_authorities(tmp_path)
    wrong = pd.read_parquet(universe_path).iloc[[0, 1, 4, 7]].copy()
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(out_dir, produced=wrong, universe_n=8, final_n=4)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert len(findings) == 1
    assert findings[0].detail["issue"] == "analysis_cohort_identity_mismatch"


def test_same_ids_with_changed_authoritative_values_fail_closed(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    changed = authoritative.reset_index(drop=True).copy()
    changed.loc[0, "death"] = 1 - int(changed.loc[0, "death"])
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(out_dir, produced=changed, universe_n=8, final_n=4)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert len(findings) == 1
    assert findings[0].detail["issue"] == "analysis_cohort_value_mismatch"


def test_primary_cohort_alias_participates_in_plan_contracts() -> None:
    from easyicu.research_agent.plan_utils import (
        _cohort_change_contract_applies,
        _plan_expects_analysis_cohort,
    )

    step = _cohort_step()
    assert _plan_expects_analysis_cohort(_plan(step))
    assert _cohort_change_contract_applies(step)


def test_remaining_attrition_must_match_each_locked_predicate(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    forged = pd.DataFrame(
        {
            "criterion": ["universe", "adult", "final_analysis_cohort"],
            "n_remaining": [8, 6, 4],
            "n_excluded_at_step": [0, 2, 2],
        }
    )
    forged.to_csv(out_dir / "cohort_flow.csv", index=False)
    forged.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_stage_counts_mismatch"


def test_remaining_attrition_requires_rowwise_conservation(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    forged = pd.DataFrame(
        {
            "criterion": ["universe", "adult", "final_analysis_cohort"],
            "n_remaining": [8, 7, 4],
            "n_excluded_at_step": [0, 0, 4],
        }
    )
    forged.to_csv(out_dir / "cohort_flow.csv", index=False)
    forged.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_transitions_do_not_conserve"


def test_detailed_remaining_rows_schema_is_verified(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    detailed = pd.DataFrame(
        {
            "criterion_id": ["universe", "include_01_age", "include_02_los_icu"],
            "n_at_start_rows": [8, 8, 7],
            "n_remaining_rows": [8, 7, 4],
            "n_excluded_rows": [0, 1, 3],
        }
    )
    detailed.to_csv(out_dir / "cohort_flow.csv", index=False)
    detailed.to_csv(out_dir / "cohort_attrition.csv", index=False)

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_sequential_attrition_cannot_swap_planner_predicate_ids(
    tmp_path: Path,
) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    swapped = pd.DataFrame(
        {
            "criterion_id": [
                "universe",
                "include_02_los_icu",
                "include_01_age",
            ],
            "n_remaining_rows": [8, 7, 4],
            "n_excluded_rows": [0, 1, 3],
        }
    )
    swapped.to_csv(out_dir / "cohort_flow.csv", index=False)
    swapped.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_sequence_rule_ids_mismatch"


def test_sequential_attrition_synonymous_remaining_counts_must_agree(
    tmp_path: Path,
) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    contradictory = pd.DataFrame(
        {
            "criterion_id": ["universe", "include_01_age", "include_02_los_icu"],
            "n_remaining": [8, 7, 4],
            "n_remaining_rows": [8, 999, 4],
            "n_excluded_rows": [0, 1, 3],
        }
    )
    contradictory.to_csv(out_dir / "cohort_flow.csv", index=False)
    contradictory.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_count_columns_disagree"


def test_sequential_attrition_synonymous_exclusion_counts_must_agree(
    tmp_path: Path,
) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    contradictory = pd.DataFrame(
        {
            "criterion_id": ["universe", "include_01_age", "include_02_los_icu"],
            "n_remaining_rows": [8, 7, 4],
            "n_excluded_at_step": [0, 1, 3],
            "n_excluded_rows": [0, 3, 1],
        }
    )
    contradictory.to_csv(out_dir / "cohort_flow.csv", index=False)
    contradictory.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_count_columns_disagree"


def test_sequential_attrition_matching_synonymous_counts_pass(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    truthful = pd.DataFrame(
        {
            "criterion_id": ["universe", "include_01_age", "include_02_los_icu"],
            "attrition_category": [
                "universe",
                "include_01_age",
                "include_02_los_icu",
            ],
            "n_remaining": [8, 7, 4],
            "n_remaining_rows": [8, 7, 4],
            "n_excluded_at_step": [0, 1, 3],
            "n_removed_from_prior_stage": [0, 1, 3],
            "n_excluded_rows": [0, 1, 3],
        }
    )
    truthful.to_csv(out_dir / "cohort_flow.csv", index=False)
    truthful.to_csv(out_dir / "cohort_attrition.csv", index=False)

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_canonical_attrition_identity_columns_must_agree(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    contradictory = pd.DataFrame(
        {
            "criterion_id": ["universe", "include_01_age", "include_02_los_icu"],
            "attrition_category": [
                "universe",
                "include_02_los_icu",
                "include_01_age",
            ],
            "n_remaining_rows": [8, 7, 4],
            "n_excluded_rows": [0, 1, 3],
        }
    )
    contradictory.to_csv(out_dir / "cohort_flow.csv", index=False)
    contradictory.to_csv(out_dir / "cohort_attrition.csv", index=False)

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_identity_columns_disagree"


def test_explicit_terminal_cohort_row_is_allowed(tmp_path: Path) -> None:
    step = _cohort_step()
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    summary = _write_outputs(
        out_dir,
        produced=authoritative,
        universe_n=8,
        final_n=4,
    )
    flow = pd.DataFrame(
        {
            "criterion": [
                "universe",
                "adult",
                "icu_length_of_stay",
                "final_analysis_cohort",
            ],
            "n_remaining": [8, 7, 4, 4],
            "n_excluded_at_step": [0, 1, 3, 0],
        }
    )
    flow.to_csv(out_dir / "cohort_flow.csv", index=False)
    flow.to_csv(out_dir / "cohort_attrition.csv", index=False)

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_bare_n_rows_sequence_cannot_hide_forged_intermediate_count(
    tmp_path: Path,
) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:cohort_denominator",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame({"n_rows": [8, 999, 4]}).to_csv(
        out_dir / "denominators.csv", index=False
    )
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:cohort_denominator": "denominators.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_stage_counts_mismatch"


def test_single_row_denominator_synonymous_fields_must_agree(tmp_path: Path) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:cohort_denominator",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "n_universe": [8],
            "universe_n": [999],
            "n_analysis_cohort": [4],
            "final_cohort_n": [4],
        }
    ).to_csv(out_dir / "denominators.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:cohort_denominator": "denominators.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "cohort_denominator_fields_disagree"


def test_partition_attrition_must_conserve_excluded_rows(tmp_path: Path) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "include_01_age",
                "include_02_los_icu",
                "primary_analysis_cohort",
            ],
            "n": [8, 1, 999, 4],
            "status": ["denominator", "excluded", "excluded", "retained"],
        }
    ).to_csv(out_dir / "attrition.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:attrition": "attrition.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_partitions_do_not_conserve"


def test_truthful_partition_attrition_passes(tmp_path: Path) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "include_02_los_icu",
                "include_01_age",
                "primary_analysis_cohort",
            ],
            "n": [8, 3, 1, 4],
            "status": ["denominator", "excluded", "excluded", "retained"],
            "partition_status": [
                " Denominator ",
                "EXCLUDED",
                "excluded",
                "Retained",
            ],
            "row_role": ["denominator", "excluded", "excluded", "retained"],
            "role": ["denominator", "excluded", "excluded", "retained"],
        }
    ).to_csv(out_dir / "attrition.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:attrition": "attrition.csv",
        },
    }

    assert (
        primary_analysis_cohort_integrity_findings(
            step=step,
            plan=plan,
            step_summary=summary,
            out_dir=out_dir,
            universe_path=universe_path,
            authoritative_cohort_path=authoritative_path,
        )
        == []
    )


def test_partition_attrition_cannot_swap_rule_labels(tmp_path: Path) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "include_02_los_icu",
                "include_01_age",
                "primary_analysis_cohort",
            ],
            "n": [8, 1, 3, 4],
            "status": ["denominator", "excluded", "excluded", "retained"],
        }
    ).to_csv(out_dir / "attrition.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:attrition": "attrition.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_partition_rule_counts_mismatch"


def test_partition_attrition_synonymous_count_columns_must_agree(
    tmp_path: Path,
) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "include_01_age",
                "include_02_los_icu",
                "primary_analysis_cohort",
            ],
            "n": [8, 1, 3, 4],
            "n_rows": [8, 3, 1, 4],
            "status": ["denominator", "excluded", "excluded", "retained"],
        }
    ).to_csv(out_dir / "attrition.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:attrition": "attrition.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_count_columns_disagree"


def test_partition_attrition_role_aliases_must_agree(tmp_path: Path) -> None:
    step = _cohort_step().model_copy(
        update={
            "expected_outputs": [
                "artifact:analysis_cohort",
                "table:attrition",
            ]
        }
    )
    plan = _plan(step)
    universe_path, authoritative_path, authoritative = _write_authorities(tmp_path)
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True)
    authoritative.to_parquet(out_dir / "analysis_cohort.parquet", index=False)
    pd.DataFrame(
        {
            "attrition_category": [
                "universe",
                "include_01_age",
                "include_02_los_icu",
                "primary_analysis_cohort",
            ],
            "n": [8, 1, 3, 4],
            "status": ["denominator", "excluded", "excluded", "retained"],
            "partition_status": [
                "denominator",
                "excluded",
                "retained",
                "excluded",
            ],
        }
    ).to_csv(out_dir / "attrition.csv", index=False)
    summary = {
        "status": "completed",
        "n_universe": 8,
        "n_final_analysis_cohort": 4,
        "output_files": {
            "artifact:analysis_cohort": "analysis_cohort.parquet",
            "table:attrition": "attrition.csv",
        },
    }

    findings = primary_analysis_cohort_integrity_findings(
        step=step,
        plan=plan,
        step_summary=summary,
        out_dir=out_dir,
        universe_path=universe_path,
        authoritative_cohort_path=authoritative_path,
    )

    assert findings[0].detail["issue"] == "attrition_role_columns_disagree"
