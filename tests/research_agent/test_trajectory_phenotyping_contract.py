"""Generic fixed-window trajectory representation and phenotyping contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.audits.patterns import AnalysisPatternAuditor
from easyicu.research_agent.agents.agentic_coder import AgenticCoderAgent
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.gates.method_compatibility import (
    detect_forbidden_pattern_usage,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    TemporalConstraint,
    VariableRole,
)
from easyicu.research_agent.trajectory.contract import (
    TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS,
    _stability_findings,
    infer_fixed_window_trajectory_metadata,
    trajectory_phenotyping_artifact_findings,
    trajectory_phenotyping_contract_applies,
    trajectory_future_imputation_detected,
    trajectory_zero_imputation_detected,
)
from easyicu.research_agent.plan_utils import (
    _enforce_advanced_plan_contract,
    _step_contract_findings,
    _step_contract_repair_guidance,
)


def _trajectory_context(*, fractional: bool) -> ResearchContext:
    values = (
        pd.Series([0.0, 0.5, 1.25])
        if fractional
        else pd.Series([0, 1, 2], dtype="int64")
    )
    variables = []
    for start, end in ((0, 6), (6, 12)):
        name = f"severity_state_h{start}_{end}"
        variables.append(
            ConceptDescriptor(
                name=name,
                role=VariableRole.ORDINAL_SCORE,
                dtype=str(values.dtype),
                is_ordinal=True,
                fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
                    column_name=name,
                    values=values,
                    source_scale="ordinal",
                ),
            )
        )
    return ResearchContext(
        research_question="Discover phenotypes from fixed-window trajectories.",
        cohort=CohortDescriptor(
            cohort_name="trajectory-test",
            database="test",
            n_patients=3,
            n_stays=3,
            id_columns=["stay_id"],
        ),
        variables=variables,
    )


def _clustering_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="trajectory_phenotyping",
        intent="Discover trajectory phenotypes.",
        inputs=["severity_state_h0_6", "severity_state_h6_12"],
        method="kmeans_clustering",
        expected_outputs=[
            "table:cluster_assignments",
            "statistic:cluster_count",
        ],
    )


def _scripts(*, zero_impute: bool = False) -> tuple[str, str]:
    fill = ".fillna(0)" if zero_impute else ".dropna()"
    common = (
        "from sklearn.cluster import KMeans\n"
        "from sklearn.preprocessing import StandardScaler\n"
    )
    explicit = common + (
        'cols = ["severity_state_h0_6", "severity_state_h6_12"]\n'
        f"X = StandardScaler().fit_transform(df[cols]{fill})\n"
        "labels = KMeans(n_clusters=2, random_state=7).fit_predict(X)\n"
    )
    dynamic = common + (
        'cols = [c for c in df.columns if c.startswith("severity_state_")]\n'
        f"X = StandardScaler().fit_transform(df[cols]{fill})\n"
        "labels = KMeans(n_clusters=2, random_state=7).fit_predict(X)\n"
    )
    return explicit, dynamic


def test_context_exposes_generic_fixed_window_metadata(tmp_path: Path):
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "physiology_h0_6": [0.0, 0.5, 1.25],
            "physiology_h6_12": [0.25, 0.75, 1.5],
            "death": [0, 1, 0],
        }
    )
    path = tmp_path / "cohort.parquet"
    cohort.to_parquet(path, index=False)

    context = build_research_context(
        research_question=(
            "Discover phenotypes from trajectories binned from ICU admission."
        ),
        cohort=path,
        cohort_name="trajectory-test",
        database="test",
        target_outcome="death",
    )

    metadata = context.variable("physiology_h6_12").fixed_window_trajectory
    assert metadata is not None
    assert metadata.family == "physiology"
    assert metadata.window_start_hours == 6
    assert metadata.window_end_hours == 12
    assert metadata.window_width_hours == 6
    assert metadata.time_axis == "relative_hours"
    assert metadata.anchor is None
    assert metadata.representation_kind == "fractional_window_summary"
    assert {item.anchor_event for item in context.temporal_constraints} == {
        "icu_admission"
    }


def test_fractional_and_raw_ordinal_window_representations_are_distinct():
    fractional = infer_fixed_window_trajectory_metadata(
        column_name="severity_state_h0_6",
        values=pd.Series([0.0, 0.5, 1.25]),
        source_scale="ordinal",
    )
    raw = infer_fixed_window_trajectory_metadata(
        column_name="severity_state_h0_6",
        values=pd.Series([0, 1, 2]),
        source_scale="ordinal",
    )

    assert fractional is not None and raw is not None
    assert fractional.source_scale == raw.source_scale == "ordinal"
    assert fractional.representation_kind == "fractional_window_summary"
    assert raw.representation_kind == "discrete_window_state"


def test_fixed_window_metadata_generalizes_across_unseen_families():
    for family in ("organ_burden", "physiology_index"):
        metadata = infer_fixed_window_trajectory_metadata(
            column_name=f"{family}_h12_24",
            values=pd.Series([0.25, 0.75, 1.5]),
            source_scale="continuous",
        )
        assert metadata is not None
        assert metadata.family == family
        assert metadata.window_start_hours == 12
        assert metadata.window_end_hours == 24
        assert metadata.representation_kind == "fractional_window_summary"


def test_trajectory_contract_requires_two_ordered_bins_from_one_selected_family():
    context = _trajectory_context(fractional=True)
    step = _clustering_step()
    assert trajectory_phenotyping_contract_applies(context=context, step=step)

    one_bin = context.model_copy(update={"variables": context.variables[:1]})
    assert not trajectory_phenotyping_contract_applies(context=one_bin, step=step)

    other_name = "other_state_h6_12"
    mixed = context.model_copy(
        update={
            "variables": [
                context.variables[0],
                ConceptDescriptor(
                    name=other_name,
                    role=VariableRole.ORDINAL_SCORE,
                    dtype="float64",
                    is_ordinal=True,
                    fixed_window_trajectory=infer_fixed_window_trajectory_metadata(
                        column_name=other_name,
                        values=pd.Series([0.0, 0.5, 1.0]),
                        source_scale="ordinal",
                    ),
                ),
            ]
        }
    )
    mixed_step = step.model_copy(
        update={"inputs": [context.variables[0].name, other_name]}
    )
    assert not trajectory_phenotyping_contract_applies(
        context=mixed,
        step=mixed_step,
    )
    assert not trajectory_phenotyping_contract_applies(
        context=context,
        step=step.model_copy(update={"inputs": []}),
    )


def test_literal_and_dynamic_selection_have_identical_method_compatibility():
    step = _clustering_step()
    explicit, dynamic = _scripts()

    fractional = _trajectory_context(fractional=True)
    for code in (explicit, dynamic):
        assert detect_forbidden_pattern_usage(code, fractional, step) == []
        errors = [
            finding
            for finding in AnalysisPatternAuditor().audit(
                context=fractional,
                script_text=code,
                step=step,
            )
            if finding.severity == "error"
        ]
        assert errors == []

    raw = _trajectory_context(fractional=False)
    for code in (explicit, dynamic):
        violations = detect_forbidden_pattern_usage(code, raw, step)
        assert any(item["kind"] == "ordinal" for item in violations)
        errors = [
            finding
            for finding in AnalysisPatternAuditor().audit(
                context=raw,
                script_text=code,
                step=step,
            )
            if finding.severity == "error"
        ]
        assert any("distance-based" in finding.message for finding in errors)


def test_trajectory_zero_imputation_is_error_for_literal_and_dynamic_selection():
    context = _trajectory_context(fractional=True)
    step = _clustering_step()
    for code in _scripts(zero_impute=True):
        violations = detect_forbidden_pattern_usage(code, context, step)
        assert any(
            item["matched_patterns"] == ["zero_imputation"] for item in violations
        )
        errors = [
            finding
            for finding in AnalysisPatternAuditor().audit(
                context=context,
                script_text=code,
                step=step,
            )
            if finding.severity == "error"
        ]
        assert any(
            finding.validator == "trajectory_representation_contract"
            for finding in errors
        )

    loc_dynamic = _scripts(zero_impute=True)[1].replace(
        "df[cols].fillna(0)", "df.loc[:, cols].fillna(0)"
    )
    assert trajectory_zero_imputation_detected(
        loc_dynamic,
        trajectory_columns=["severity_state_h0_6", "severity_state_h6_12"],
    )
    assert trajectory_zero_imputation_detected(
        'cols = ["severity_state_h0_6"]\nX = df[cols]\nX = np.nan_to_num(X)',
        trajectory_columns=["severity_state_h0_6"],
    )
    for script in (
        'X = df.filter(regex=r"^severity_state_h\\d+_\\d+$").fillna(0)',
        'X = df.filter(like="severity_state_").fillna(0)',
        (
            'cols = [c for c in df.columns if "severity_state_" in c]\n'
            "X = df.loc[:, cols].fillna(0)"
        ),
        (
            'cols = [c for c in df.columns if re.match(r"^severity_state_h", c)]\n'
            "X = df[cols].fillna(0)"
        ),
    ):
        assert trajectory_zero_imputation_detected(
            script,
            trajectory_columns=["severity_state_h0_6", "severity_state_h6_12"],
        ), script


def test_unrelated_column_zero_fill_is_not_a_trajectory_error():
    context = _trajectory_context(fractional=True)
    step = _clustering_step()
    code = (
        "from sklearn.cluster import KMeans\n"
        "from sklearn.preprocessing import StandardScaler\n"
        'cols = [c for c in df.columns if c.startswith("severity_state_")]\n'
        'age = df["age"].fillna(0)\n'
        "X = StandardScaler().fit_transform(df[cols].dropna())\n"
        "labels = KMeans(n_clusters=2, random_state=7).fit_predict(X)\n"
    )

    assert detect_forbidden_pattern_usage(code, context, step) == []
    findings = AnalysisPatternAuditor().audit(
        context=context,
        script_text=code,
        step=step,
    )
    assert not any(
        finding.validator == "trajectory_representation_contract"
        for finding in findings
    )
    assert not trajectory_zero_imputation_detected(
        'cols = ["severity_state_h0_6"]\nX = df[cols].fillna(method="ffill", limit=0)',
        trajectory_columns=["severity_state_h0_6"],
    )
    assert not trajectory_zero_imputation_detected(
        'X = df["age"]\nX = np.nan_to_num(X)',
        trajectory_columns=["severity_state_h0_6"],
    )


def test_trajectory_future_looking_imputation_is_error():
    context = _trajectory_context(fractional=True)
    step = _clustering_step()
    for code in (
        'cols = ["severity_state_h0_6", "severity_state_h6_12"]\n'
        "X = df[cols].bfill()",
        'cols = ["severity_state_h0_6", "severity_state_h6_12"]\n'
        'X = df[cols].fillna(method="backfill")',
        'cols = ["severity_state_h0_6", "severity_state_h6_12"]\n'
        "X = df[cols].interpolate()",
    ):
        assert trajectory_future_imputation_detected(
            code,
            trajectory_columns=["severity_state_h0_6", "severity_state_h6_12"],
        )
        findings = AnalysisPatternAuditor().audit(
            context=context,
            script_text=code,
            step=step,
        )
        assert any(
            finding.detail.get("kind") == "trajectory_future_imputation"
            and finding.severity == "error"
            for finding in findings
        )

    assert not trajectory_future_imputation_detected(
        'cols = ["severity_state_h0_6"]\nX = df[cols].ffill()',
        trajectory_columns=["severity_state_h0_6"],
    )


def _selection_manifest() -> dict:
    return {
        "criterion": "Bayesian information criterion",
        "selection_rule": "minimum",
        "direction": "minimize",
        "selected_n_clusters": 2,
        "candidates": [
            {"n_clusters": 1, "criterion_value": 125.0},
            {"n_clusters": 2, "criterion_value": 100.0},
            {"n_clusters": 3, "criterion_value": 112.0},
        ],
        "rationale": "Selected the finite minimum among the evaluated candidates.",
        "candidate_range_boundary_rule": "allow_upper_boundary",
        "candidate_range_boundary_reason_code": None,
    }


def test_plan_and_agentic_coder_receive_only_the_generic_trajectory_contract():
    context = _trajectory_context(fractional=True).model_copy(
        update={"target_outcome": "death"}
    )
    step = _clustering_step()
    plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[step],
    )

    revised, _ = _enforce_advanced_plan_contract(plan=plan, context=context)
    outputs = set(revised.steps[0].expected_outputs)
    assert set(TRAJECTORY_PHENOTYPING_REQUIRED_OUTPUTS).issubset(outputs)
    assert "manifest:cluster_selection" in outputs
    assert "statistic:silhouette_score" not in outputs
    assert "table:outcome_by_cluster" not in outputs
    prompt = AgenticCoderAgent(object())._build_prompt(context, revised.steps[0])
    assert "trajectory_missingness_policy.json" in prompt
    assert "cluster_stability_assignments.csv" in prompt
    assert "outcome_by_cluster.csv" not in prompt
    assert "anchor_provenance" in prompt

    cross_sectional = context.model_copy(
        update={
            "variables": [
                ConceptDescriptor(
                    name="age",
                    role=VariableRole.DEMOGRAPHIC,
                    dtype="float64",
                )
            ]
        }
    )
    cross_plan = AnalysisPlan(
        research_question=context.research_question,
        analysis_type="trajectory_clustering",
        steps=[
            step.model_copy(
                update={"inputs": ["age"]},
            )
        ],
    )
    cross_revised, _ = _enforce_advanced_plan_contract(
        plan=cross_plan,
        context=cross_sectional,
    )
    assert "table:trajectory_membership" not in set(
        cross_revised.steps[0].expected_outputs
    )


def test_clustering_contract_accepts_agent_native_selection_without_silhouette():
    step = AnalysisStep(
        step_id="phenotyping",
        intent="Discover phenotypes with the agent-selected mixture model.",
        method="gaussian_mixture_model",
        expected_outputs=[
            "table:cluster_assignments",
            "statistic:cluster_count",
            "manifest:cluster_selection",
        ],
    )
    findings = _step_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "cluster_count": 2,
            "cluster_selection": _selection_manifest(),
        },
    )
    assert not any("clustering summary" in item.message for item in findings)

    incomplete = _step_contract_findings(
        step=step,
        step_summary={"status": "ok", "cluster_count": 2},
    )
    assert any("clustering summary" in item.message for item in incomplete)
    path_only = _step_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "cluster_count": 2,
            "cluster_selection_path": "cluster_selection.json",
        },
    )
    assert any("clustering summary" in item.message for item in path_only)
    contradictory = _step_contract_findings(
        step=step,
        step_summary={
            "status": "ok",
            "cluster_count": 3,
            "cluster_selection": _selection_manifest(),
        },
    )
    assert any("clustering summary" in item.message for item in contradictory)
    guidance = _step_contract_repair_guidance(
        step=step,
        step_summary={"status": "error"},
        code="",
    )
    assert "BIC/AIC/ICL" in guidance
    assert "full `cluster_selection`" in guidance


def _write_truthful_bundle(tmp_path: Path):
    from sklearn.metrics import adjusted_rand_score

    tmp_path.mkdir(parents=True, exist_ok=True)
    cohort = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5, 6, 7],
            "severity_h0_6": [0.0, 0.5, 1.0, 5.0, 5.5, 6.0, float("nan")],
            "severity_h6_12": [
                1.0,
                1.5,
                float("nan"),
                6.0,
                6.5,
                float("nan"),
                float("nan"),
            ],
            "severity_h12_18": [2.0, 2.5, 3.0, 7.0, 7.5, 8.0, float("nan")],
            "death": [0, 0, 0, 1, 1, 1, 1],
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    cohort.to_parquet(cohort_path, index=False)
    context = build_research_context(
        research_question=(
            "Discover phenotypes from trajectories binned from ICU admission."
        ),
        cohort=cohort_path,
        cohort_name="trajectory-test",
        database="test",
        target_outcome="death",
    )
    step = AnalysisStep(
        step_id="trajectory_phenotyping",
        intent="Discover fixed-window trajectory phenotypes and characterize outcomes.",
        inputs=["severity_h0_6", "severity_h6_12"],
        method="gaussian_mixture_model",
        expected_outputs=[
            "table:cluster_assignments",
            "table:trajectory_profiles",
            "table:cluster_stability",
            "table:outcome_by_cluster",
        ],
    )
    out_dir = tmp_path / "steps" / step.step_id / "outputs"
    out_dir.mkdir(parents=True)
    (out_dir.parent / "analysis.py").write_text(
        "# Agent-owned model; no zero imputation.\n", encoding="utf-8"
    )
    policy = {
        "id_column": "stay_id",
        "observation_family": "severity",
        "observation_columns": ["severity_h0_6", "severity_h6_12"],
        "min_observed_windows": 1,
        "profile_columns": ["severity_h0_6", "severity_h6_12"],
        "profile_summary_statistic": "mean",
        "clustering_method": "GaussianMixture",
        "n_clusters": 2,
        "time_axis": "relative_hours",
        "anchor": "icu_admission",
        "anchor_provenance": "task_contract",
        "anchor_source": "temporal_constraints.relative_to_anchor",
        "trailing_na_policy": {
            "zero_imputation": False,
            "eligibility_uses_observed_window_count": True,
            "profile_summaries_ignore_missing": True,
        },
    }
    (out_dir / "trajectory_missingness_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    selection_manifest = _selection_manifest()
    (out_dir / "cluster_selection.json").write_text(
        json.dumps(selection_manifest), encoding="utf-8"
    )
    observed = cohort[["severity_h0_6", "severity_h6_12"]].notna().sum(axis=1)
    included = observed >= 1
    pd.DataFrame(
        {
            "stay_id": cohort["stay_id"],
            "observed_window_count": observed,
            "meets_min_observed_windows": included,
            "included_in_clustering": included,
            "exclusion_reason": [
                "" if value else "below_minimum" for value in included
            ],
        }
    ).to_csv(out_dir / "trajectory_membership.csv", index=False)
    assignments = pd.DataFrame(
        {"stay_id": [1, 2, 3, 4, 5, 6], "cluster": [0, 0, 0, 1, 1, 1]}
    )
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame(
        {
            "metric": [
                "input_cohort",
                "meets_min_observed_windows",
                "excluded_insufficient_windows",
                "included_in_clustering",
            ],
            "n": [7, 6, 1, 6],
        }
    ).to_csv(out_dir / "cohort_flow.csv", index=False)
    pd.DataFrame({"cluster": [0, 1], "n": [3, 3]}).to_csv(
        out_dir / "cluster_sizes.csv", index=False
    )

    merged = assignments.merge(cohort, on="stay_id", how="left")
    profile_rows = []
    for cluster, group in merged.groupby("cluster"):
        for column, start, end in (
            ("severity_h0_6", 0, 6),
            ("severity_h6_12", 6, 12),
        ):
            values = group[column].dropna()
            profile_rows.append(
                {
                    "cluster": cluster,
                    "source_column": column,
                    "window_start_hours": start,
                    "window_end_hours": end,
                    "summary_statistic": "mean",
                    "value": values.mean(),
                    "n_observed": len(values),
                }
            )
    pd.DataFrame(profile_rows).to_csv(out_dir / "trajectory_profiles.csv", index=False)

    stability_assignment_rows = []
    stability_rows = []
    reference_by_id = assignments.set_index("stay_id")["cluster"].to_dict()
    for resample_id, refit_model_id, seed, sample_ids, labels in (
        ("seed_1", "refit_model_1", 101, [1, 2, 4, 5], [0, 0, 1, 1]),
        ("seed_2", "refit_model_2", 202, [2, 3, 5, 6], [1, 1, 0, 0]),
    ):
        references = [reference_by_id[stay_id] for stay_id in sample_ids]
        for stay_id, reference, resampled in zip(
            sample_ids, references, labels, strict=True
        ):
            stability_assignment_rows.append(
                {
                    "resample_id": resample_id,
                    "stay_id": stay_id,
                    "reference_cluster": reference,
                    "resampled_cluster": resampled,
                }
            )
        stability_rows.append(
            {
                "resample_id": resample_id,
                "n_overlap": len(sample_ids),
                "adjusted_rand_index": adjusted_rand_score(references, labels),
                "clustering_method": "GaussianMixture",
                "refit_model_id": refit_model_id,
                "seed": seed,
                "sampling_method": "subsample_without_replacement",
                "sample_n": len(sample_ids),
                "sample_id_hash": hashlib.sha256(
                    "\n".join(sorted(map(str, sample_ids))).encode("utf-8")
                ).hexdigest(),
            }
        )
    pd.DataFrame(stability_assignment_rows).to_csv(
        out_dir / "cluster_stability_assignments.csv", index=False
    )
    pd.DataFrame(stability_rows).to_csv(out_dir / "cluster_stability.csv", index=False)

    outcome_rows = []
    for cluster, group in merged.groupby("cluster"):
        values = group["death"].dropna()
        outcome_rows.append(
            {
                "cluster": cluster,
                "n": len(group),
                "outcome_n": len(values),
                "event_n": int(values.sum()),
                "outcome_rate": values.mean(),
            }
        )
    pd.DataFrame(outcome_rows).to_csv(out_dir / "outcome_by_cluster.csv", index=False)
    summary = {
        "status": "ok",
        "clustering_method": "GaussianMixture",
        "n_clusters": 2,
        "min_observed_windows": 1,
        "cluster_selection": selection_manifest,
    }
    return context, cohort_path, step, out_dir, summary


def _artifact_errors(bundle):
    context, cohort_path, step, out_dir, summary = bundle
    return [
        finding
        for finding in trajectory_phenotyping_artifact_findings(
            context=context,
            cohort_path=cohort_path,
            step=step,
            out_dir=out_dir,
            step_summary=summary,
        )
        if finding.severity == "error"
    ]


def test_truthful_trajectory_artifact_bundle_replays(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    assert {item.anchor_event for item in bundle[0].temporal_constraints} == {
        "icu_admission"
    }
    assert _artifact_errors(bundle) == []


def test_missing_assignment_artifact_blocks(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    (bundle[3] / "cluster_assignments.csv").unlink()
    errors = _artifact_errors(bundle)
    assert errors[0].detail["kind"] == "missing_trajectory_artifacts"


def test_policy_observation_columns_equal_selected_family_inputs(tmp_path: Path):
    mutations = (
        ["severity_h0_6"],
        ["severity_h6_12", "severity_h0_6"],
        ["severity_h0_6", "severity_h6_12", "severity_h12_18"],
    )
    for index, columns in enumerate(mutations):
        bundle = _write_truthful_bundle(tmp_path / f"observation_binding_{index}")
        path = bundle[3] / "trajectory_missingness_policy.json"
        policy = json.loads(path.read_text(encoding="utf-8"))
        policy["observation_columns"] = columns
        path.write_text(json.dumps(policy), encoding="utf-8")
        errors = _artifact_errors(bundle)
        assert errors[0].detail["kind"] == "invalid_trajectory_columns"
        assert errors[0].detail["observation_binding_policy"] == "ordered_equality"

    bundle = _write_truthful_bundle(tmp_path / "family_switch")
    path = bundle[3] / "trajectory_missingness_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["observation_family"] = "unselected_family"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert _artifact_errors(bundle)[0].detail["kind"] == "invalid_trajectory_columns"


def test_task_contract_anchor_mismatch_blocks(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    path = bundle[3] / "trajectory_missingness_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["anchor"] = "hospital_admission"
    path.write_text(json.dumps(policy), encoding="utf-8")

    errors = _artifact_errors(bundle)
    assert errors[0].detail["kind"] == "trajectory_anchor_mismatch"


def test_anchor_is_agent_declared_when_context_has_no_unique_anchor(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    context, cohort_path, step, out_dir, summary = bundle
    context = context.model_copy(update={"temporal_constraints": []})
    path = out_dir / "trajectory_missingness_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy.update(
        {
            "anchor": "study_time_zero",
            "anchor_provenance": "agent_declared",
            "anchor_source": "analysis_plan",
        }
    )
    path.write_text(json.dumps(policy), encoding="utf-8")

    assert _artifact_errors((context, cohort_path, step, out_dir, summary)) == []


def test_unrelated_temporal_constraint_cannot_bind_trajectory_anchor(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    context, cohort_path, step, out_dir, summary = bundle
    unrelated = TemporalConstraint(
        raw_text="within 48h after hospital admission",
        relation="within_after",
        anchor_event="hospital_admission",
        start_hours=0,
        end_hours=48,
        executable_repr="within_after|anchor=hospital_admission|hours=48",
    )
    context = context.model_copy(update={"temporal_constraints": [unrelated]})
    path = out_dir / "trajectory_missingness_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy.update(
        {
            "anchor_provenance": "agent_declared",
            "anchor_source": "analysis_plan",
        }
    )
    path.write_text(json.dumps(policy), encoding="utf-8")

    assert _artifact_errors((context, cohort_path, step, out_dir, summary)) == []


def test_conflicting_relative_task_anchors_fail_closed(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    context, cohort_path, step, out_dir, summary = bundle
    constraints = list(context.temporal_constraints) + [
        TemporalConstraint(
            raw_text="relative to hospital admission",
            relation="relative_to_anchor",
            anchor_event="hospital_admission",
            executable_repr="relative_to_anchor|anchor=hospital_admission",
        )
    ]
    context = context.model_copy(update={"temporal_constraints": constraints})
    errors = _artifact_errors((context, cohort_path, step, out_dir, summary))
    assert errors[0].detail["kind"] == "trajectory_anchor_contract_conflict"
    assert errors[0].detail["conflicting_anchors"] == [
        "hospital_admission",
        "icu_admission",
    ]


def test_outcome_artifact_is_conditional_on_agent_plan(tmp_path: Path):
    context, cohort_path, step, out_dir, summary = _write_truthful_bundle(tmp_path)
    assert "outcome_by_cluster.csv" in AgenticCoderAgent(object())._build_prompt(
        context,
        step,
    )
    no_outcome_step = step.model_copy(
        update={
            "expected_outputs": [
                value
                for value in step.expected_outputs
                if value != "table:outcome_by_cluster"
            ]
        }
    )
    assert "outcome_by_cluster.csv" not in AgenticCoderAgent(object())._build_prompt(
        context,
        no_outcome_step,
    )
    (out_dir / "outcome_by_cluster.csv").unlink()
    assert (
        _artifact_errors((context, cohort_path, no_outcome_step, out_dir, summary))
        == []
    )

    errors = _artifact_errors((context, cohort_path, step, out_dir, summary))
    assert errors[0].detail["kind"] == "missing_trajectory_artifacts"


def test_cluster_selection_manifest_rejects_forged_or_thin_evidence(tmp_path: Path):
    mutations = (
        (
            lambda payload: payload.update({"candidates": payload["candidates"][:1]}),
            "invalid_cluster_selection_manifest",
        ),
        (
            lambda payload: payload["candidates"][2].update({"criterion_value": 90.0}),
            "cluster_selection_replay_mismatch",
        ),
        (
            lambda payload: payload.update({"selected_n_clusters": 3}),
            "cluster_selection_replay_mismatch",
        ),
        (
            lambda payload: payload.update(
                {
                    "selection_rule": "elbow",
                    "direction": "not_applicable",
                    "rationale": "",
                }
            ),
            "invalid_cluster_selection_manifest",
        ),
    )
    for index, (mutate, expected_kind) in enumerate(mutations):
        bundle = _write_truthful_bundle(tmp_path / f"selection_{index}")
        path = bundle[3] / "cluster_selection.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        kinds = {finding.detail["kind"] for finding in _artifact_errors(bundle)}
        assert expected_kind in kinds


def test_elbow_selection_with_rationale_remains_agent_owned(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    path = bundle[3] / "cluster_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    selection.update(
        {
            "selection_rule": "elbow",
            "direction": "not_applicable",
            "rationale": (
                "The incremental criterion improvement flattened after k=2, "
                "so the agent selected the parsimonious elbow."
            ),
        }
    )
    # Deliberately make a different candidate the numeric minimum: elbow is a
    # declared scientific judgment, not a framework-selected optimum.
    selection["candidates"][2]["criterion_value"] = 90.0
    path.write_text(json.dumps(selection), encoding="utf-8")
    bundle[4]["cluster_selection"] = selection

    assert _artifact_errors(bundle) == []


def test_foreign_or_missing_assignment_id_blocks(tmp_path: Path):
    bundle = _write_truthful_bundle(tmp_path)
    path = bundle[3] / "cluster_assignments.csv"
    assignments = pd.read_csv(path)
    assignments.loc[0, "stay_id"] = 999
    assignments.to_csv(path, index=False)
    errors = _artifact_errors(bundle)
    assert errors[0].detail["kind"] == "cluster_assignments_membership_mismatch"


def test_forged_cluster_size_profile_outcome_and_stability_are_caught(
    tmp_path: Path,
):
    mutations = (
        ("cluster_sizes.csv", "n", "cluster_sizes_mismatch"),
        ("trajectory_profiles.csv", "value", "trajectory_profiles_mismatch"),
        ("outcome_by_cluster.csv", "event_n", "outcome_by_cluster_replay_mismatch"),
        (
            "cluster_stability.csv",
            "adjusted_rand_index",
            "cluster_stability_replay_mismatch",
        ),
    )
    for index, (filename, column, expected_kind) in enumerate(mutations):
        case_dir = tmp_path / f"case_{index}"
        bundle = _write_truthful_bundle(case_dir)
        path = bundle[3] / filename
        frame = pd.read_csv(path)
        frame.loc[0, column] = float(frame.loc[0, column]) + 1
        frame.to_csv(path, index=False)
        kinds = {finding.detail["kind"] for finding in _artifact_errors(bundle)}
        assert expected_kind in kinds, (filename, kinds)


def test_reported_refit_provenance_is_replayed_and_bound(tmp_path: Path):
    mutations = (
        ("drop_refit", "trajectory_artifact_schema_missing"),
        ("duplicate_refit", "cluster_stability_provenance_invalid"),
        ("duplicate_seed", "cluster_stability_provenance_invalid"),
        ("duplicate_sample_hash", "cluster_stability_provenance_invalid"),
        ("forged_count", "cluster_stability_replay_mismatch"),
        ("forged_hash", "cluster_stability_replay_mismatch"),
    )
    for index, (mutation, expected_kind) in enumerate(mutations):
        bundle = _write_truthful_bundle(tmp_path / f"stability_{index}")
        path = bundle[3] / "cluster_stability.csv"
        frame = pd.read_csv(path)
        if mutation == "drop_refit":
            frame = frame.drop(columns=["refit_model_id"])
        elif mutation == "duplicate_refit":
            frame.loc[1, "refit_model_id"] = frame.loc[0, "refit_model_id"]
        elif mutation == "duplicate_seed":
            frame.loc[1, "seed"] = frame.loc[0, "seed"]
        elif mutation == "duplicate_sample_hash":
            frame.loc[1, "sample_id_hash"] = frame.loc[0, "sample_id_hash"]
        elif mutation == "forged_count":
            frame.loc[0, "sample_n"] = int(frame.loc[0, "sample_n"]) + 1
        else:
            frame.loc[0, "sample_id_hash"] = "0" * 64
        frame.to_csv(path, index=False)
        kinds = {finding.detail["kind"] for finding in _artifact_errors(bundle)}
        assert expected_kind in kinds, (mutation, kinds)


def test_stability_resample_may_omit_a_rare_declared_cluster():
    from sklearn.metrics import adjusted_rand_score

    cluster_by_id = {
        "1": "a",
        "2": "a",
        "3": "b",
        "4": "b",
        "5": "c",
        "6": "c",
    }
    assignment_rows = []
    summary_rows = []
    for resample_id, refit_id, seed, sample_ids, labels in (
        ("r1", "model_1", 11, ["1", "2", "3", "4"], ["x", "x", "y", "y"]),
        ("r2", "model_2", 22, ["1", "3", "4"], ["x", "y", "y"]),
    ):
        reference = [cluster_by_id[value] for value in sample_ids]
        for row_id, ref, label in zip(sample_ids, reference, labels, strict=True):
            assignment_rows.append(
                {
                    "resample_id": resample_id,
                    "stay_id": row_id,
                    "reference_cluster": ref,
                    "resampled_cluster": label,
                }
            )
        summary_rows.append(
            {
                "resample_id": resample_id,
                "n_overlap": len(sample_ids),
                "adjusted_rand_index": adjusted_rand_score(reference, labels),
                "clustering_method": "agent_selected_method",
                "refit_model_id": refit_id,
                "seed": seed,
                "sampling_method": "subsample_without_replacement",
                "sample_n": len(sample_ids),
                "sample_id_hash": hashlib.sha256(
                    "\n".join(sorted(sample_ids)).encode("utf-8")
                ).hexdigest(),
            }
        )

    assert (
        _stability_findings(
            step=AnalysisStep(
                step_id="phenotyping",
                intent="Assess agent-planned phenotyping stability.",
            ),
            id_column="stay_id",
            method="agent_selected_method",
            n_clusters=3,
            cluster_by_id=cluster_by_id,
            stability=pd.DataFrame(summary_rows),
            stability_assignments=pd.DataFrame(assignment_rows),
        )
        == []
    )


def test_duplicate_or_extra_size_and_flow_rows_are_rejected(tmp_path: Path):
    mutations = (
        ("cluster_sizes.csv", "cluster_sizes_mismatch", {"cluster": 0, "n": 3}),
        ("cluster_sizes.csv", "cluster_sizes_mismatch", {"cluster": 9, "n": 1}),
        (
            "cohort_flow.csv",
            "trajectory_cohort_flow_mismatch",
            {"metric": "input_cohort", "n": 7},
        ),
        (
            "cohort_flow.csv",
            "trajectory_cohort_flow_mismatch",
            {"metric": "unplanned_extra", "n": 1},
        ),
    )
    for index, (filename, expected_kind, extra_row) in enumerate(mutations):
        bundle = _write_truthful_bundle(tmp_path / f"duplicate_{index}")
        path = bundle[3] / filename
        frame = pd.read_csv(path)
        frame = pd.concat([frame, pd.DataFrame([extra_row])], ignore_index=True)
        frame.to_csv(path, index=False)
        kinds = {finding.detail["kind"] for finding in _artifact_errors(bundle)}
        assert expected_kind in kinds
