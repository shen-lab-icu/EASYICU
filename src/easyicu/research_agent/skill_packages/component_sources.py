"""Exact, read-only method-component source bindings.

Unlike a workflow, a component must never inherit a family's entire runner or
all kernels imported by that runner. These coordinates name inspected compute
functions or reusable helpers. Adapter coordinates come from the adapter owner;
they identify dispatch, not a separate algorithm for every action.
"""

from __future__ import annotations

import ast
from typing import Any

from ..planning.method_adapter_catalog import get_method_adapter_contract
from .host_sources import (
    SourceDefinition, _Sources, _PACKAGE_ROOT, _PREFIX, _DEFINITIONS, _SUPPORT_FUNCTIONS,
    _implementation_files, implementation_metadata,
)


# Only exact functions are selected. In particular Cox/PH do not inherit CIF,
# prediction calibration does not inherit reclassification, and mixed-effects
# does not inherit GEE merely because they share an executor module.
_COMPONENT_FUNCTIONS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "data_preparation.outlier_rules": (("methods.missing", ("apply_outliers",)),),
    "data_preparation.window_aggregation": (("methods.missing", ("aggregate_windows",)),),
    "data_preparation.unit_harmonisation": (("methods.missing", ("database_preparation_status",)),),
    "data_preparation.missing_data": (("methods.missing_data", ("multiple_impute", "rubin_pool")),),
    "data_preparation.scaling_coding": (("methods.missing", ("standardize_columns",)),
                                          ("execution.model_matrix", ("compile_model_terms",))),
    "time_to_event.cox_hr": (("execution.runners.survival_primary_executor", ("_fit_declared_cox",)),),
    "time_to_event.km_logrank": (("execution.runners.survival_primary_executor", ("logrank_by_group",)),
                                ("figures.base", ("km_estimate",))),
    "time_to_event.ph_check": (("methods.ph_schoenfeld", ("ph_test",)),),
    "time_to_event.rmst": (("methods.rmst", ("rmst", "rmst_difference")),),
    "time_to_event.competing_risks_cif": (("methods.competing_risks", ("estimate_cif", "fit_fine_gray")),
                                         ("methods.gray_test", ("gray_test",))),
    "causal_emulation.covariate_balance": (("methods.covariate_balance", ("compute_covariate_balance",)),
                                          ("execution.runners.covariate_balance_executor", ("run_covariate_balance",))),
    "causal_emulation.positivity_overlap": (("methods.doubly_robust", ("positivity_diagnostics",)),),
    "causal_emulation.propensity_adjustment": (("methods.propensity_weighting", (
        "estimate_propensity_scores", "match_nearest_neighbor", "compute_iptw_weights")),),
    "causal_emulation.evalue": (("methods.sensitivity", ("compute_e_value",)),),
    "causal_emulation.negative_control": (("methods.sensitivity", ("run_negative_control_check",)),),
    "causal_emulation.doubly_robust": (("methods.doubly_robust", ("aiptw_ate",)),
                                      ("methods.survival_inputs", ("event_time_reconciliation_receipt",))),
    "causal_emulation.mediation": (("methods.mediation", ("mediate",)),),
    "causal_emulation.g_methods": (("methods.msm", ("fit_longitudinal_msm", "stabilized_iptw")),
                                   ("methods.gformula", ("gformula_ate",))),
    "association.mixed_effects": (("methods.mixed_effects", ("fit_mixed_effects",)),),
    "association.gee": (("methods.gee", ("fit_gee",)),),
    "association.ordinal_trend": (("methods.ordered_trends", ("cochran_armitage_trend", "jonckheere_terpstra_trend")),
                                   ("execution.runners.ordered_stratified_executor", ("_adjusted_trend",)),
                                   ("execution.runners.adjusted_association_executor", ("run_adjusted_association_from_env",)),
                                   ("robustness.estimators", ("fit_estimator",))),
    "association.adjusted_association": (("execution.runners.adjusted_association_executor", ("run_adjusted_association_from_env",)),
                                          ("robustness.estimators", ("fit_estimator",))),
    "association.proportional_odds": (("methods.ordered_trends", ("fit_outcome_model", "proportional_odds_score")),),
    "association.free_days": (("execution.runners.adjusted_association_executor", ("prepare_outcome", "fit_declared_outcome")),),
    "association.count_outcome": (("methods.ordered_trends", ("fit_outcome_model",)),),
    "association.effect_modification": (("execution.runners.association_effect_modification", ("fit_baseline_association",)),
                                         ("robustness.estimators", ("fit_estimator",))),
    "association.multiple_testing": (("methods.multiple_testing", ("build_multiple_testing_report",)),),
    "association.rcs_spline": (("methods.rcs_dose_response", ("rcs_basis", "rcs_fit", "nonlinearity_wald_test", "predict_curve")),),
    "association.evalue": (("methods.sensitivity", ("compute_e_value",)),),
    "association.robustness_panel": (("execution.runners.deterministic_robustness", ("_run_robustness_preflight_from_env",)),
                                      ("robustness.estimators", ("fit_robustness_rows_from_records",))),
    "prediction.discrimination_calibration": (("execution.runners.prediction_model_executor", ("run_prediction_model",)),
                                               ("methods.prediction_validation", ("evaluate_binary_predictions",))),
    "prediction.calibration_metrics": (("execution.runners.prediction_model_executor", ("run_prediction_score_analysis",)),
                                        ("methods.prediction_validation", ("evaluate_binary_predictions", "_calibration_model"))),
    "prediction.delong_ci": (("methods.delong_auc", ("delong_auc_ci", "delong_difference", "delong_test")),),
    "prediction.benchmark_comparison": (("execution.runners.prediction_model_executor", ("run_prediction_benchmark_comparison",)),
                                         ("methods.delong_auc", ("delong_difference",)),
                                         ("methods.prediction_validation", ("calibration_in_the_large", "recalibrate_score"))),
    "prediction.decision_curve": (("execution.runners.prediction_model_executor", ("run_prediction_score_analysis",)),
                                   ("methods.decision_curve", ("net_benefit_curve",))),
    "prediction.threshold_metrics": (("methods.prediction_validation", ("evaluate_binary_predictions", "_threshold_metrics")),),
    "prediction.subgroup_fairness": (("methods.prediction_subgroup", ("evaluate_prediction_subgroups",)),
                                      ("execution.runners.prediction_subgroup_executor", ("run_prediction_subgroup_step",)),
                                      ("methods.prediction_validation", ("evaluate_binary_predictions",))),
    "prediction.feature_attribution": (("methods.shap_attribution", ("shap_attribute",)),),
    "prediction.internal_validation": (("execution.runners.prediction_model_executor", ("run_prediction_score_analysis", "_bootstrap_pipeline_validation")),
                                        ("methods.prediction_validation", ("bootstrap_optimism_validation",))),
    "prediction.conformal_intervals": (("methods.conformal", ("conformal_calibrate", "conformal_predict_sets", "conformal_evaluate")),),
    "prediction.external_validation": (("execution.runners.prediction_validation_executor", ("run_external_validation_step",)),
                                        ("methods.prediction_validation", ("run_external_validation",))),
    "prediction.reclassification": (("methods.reclassification", ("categorical_nri", "continuous_nri", "idi")),),
    "prediction.dynamic_prediction": (("methods.dynamic_prediction", ("build_landmark_feature_matrix", "attach_landmark_outcomes", "evaluate_landmark_probabilities")),
                                       ("methods.temporal_features", ("landmark_cohort",))),
    "phenotyping.cluster_solution": (("execution.runners.cross_sectional_phenotyping_executor", ("run_primary_phenotyping",)),),
    "phenotyping.k_selection": (("execution.runners.cross_sectional_phenotyping_executor", ("run_phenotyping_diagnostic", "_candidate_scores")),),
    "phenotyping.cluster_stability": (("execution.runners.cross_sectional_phenotyping_executor", ("run_phenotyping_diagnostic",)),),
    "phenotyping.trajectory_cluster_stability": (("execution.runners.trajectory_stability_executor", ("run_trajectory_stability",)),),
    "phenotyping.cluster_sizes": (("methods.cluster_sizes", ("compute_cluster_sizes",)),
                                  ("execution.runners.cluster_sizes_executor", ("run_cluster_sizes",))),
    "phenotyping.outcome_by_cluster": (("execution.runners.phenotype_comparison_executor", (
        "phenotype_comparison_executor_code", "run_phenotype_comparison")),),
    "phenotyping.early_subtype_assignment": (("methods.subtype_assignment", ("fit_and_evaluate_early_subtype_assignment",)),),
    "phenotyping.mixed_effects_growth": (("methods.mixed_effects", ("fit_mixed_effects",)),),
    "descriptive.table_one": (("methods.table_one", ("build_grouped_table_one", "table_one_spec_sha256")),),
    "cross_database.meta_analysis": (("replication.metrics", ("pool_database_estimates",)),
                                     ("methods.sensitivity", ("compute_meta_analysis",))),
}

def method_component_implementation(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return exact action evidence, without importing or executing any kernel."""
    action = row["id"]
    adapter = get_method_adapter_contract(action)
    fixed = row["implementation"] == "deterministic" or (adapter is not None and adapter.scope == "full_action")
    mode = "fixed_host" if fixed else "generated_code"
    bindings = {}
    if adapter is not None and action != "phenotyping.outcome_by_cluster":
        # This callable renders a sealed script. The computing functions below
        # remain the primary source evidence, even for a full action adapter.
        bindings[SourceDefinition(adapter.owner_module, adapter.owner_entrypoint)] = "helper"
    for module, names in _COMPONENT_FUNCTIONS.get(action, ()):
        for name in names:
            # These are exact computing functions, not inherited family roots.
            role = "kernel" if fixed else "helper"
            if module.startswith("execution.runners") and name.startswith("run_"):
                role = "entrypoint" if fixed else "helper"
            if name.endswith(("_code", "_scaffold", "_sha256")) or SourceDefinition(_PREFIX + module, name) in _SUPPORT_FUNCTIONS:
                role = "helper"
            if action == "cross_database.meta_analysis" and name == "pool_database_estimates":
                role = "entrypoint"
            if action == "prediction.decision_curve" and module == "methods.decision_curve":
                # The fixed score-analysis route computes net benefit inline;
                # this separately reusable kernel is not called by that route.
                role = "helper"
            bindings[SourceDefinition(_PREFIX + module, name)] = role
    if action == "data_preparation.unit_harmonisation":
        mode = "owner_projection"
        summary_en = "Read-only status of the frozen concept-dictionary conversion rules captured by extraction. The listed function inspects provenance; it does not convert or reconvert materialized values."
        summary_zh = "只读查看提取时保留的概念字典单位转换规则与状态。所列函数检查来源记录，不执行换算，也不对已提取数值再次换算。"
        bindings = {source: "helper" for source in bindings}
    elif mode == "generated_code":
        summary_en = "The model writes this method's study-specific code from the reviewed plan. Listed functions are available helpers or a controlled built-in subcontract, not the generated study implementation. Code-origin gates are shown; there is no saved generated code before a run."
        summary_zh = "模型按已审阅方案编写本方法的研究代码。所列函数是可用辅助函数或受控内置子合同，不是本次生成的分析实现。这里同时列出代码来源检查；运行前尚无已保存的生成代码。"
        bindings.update({
            SourceDefinition(_PREFIX + "execution.initial_code_policy", "resolve_initial_code_origin"): "gate",
            SourceDefinition(_PREFIX + "execution.step_execution", "initial_code_origin"): "gate",
        })
    elif not bindings:
        mode = "unavailable"
        summary_en = "The registry names a shared runner, but no reliable action-specific function binding is available. It is not presented as an independent algorithm."
        summary_zh = "注册表列有共享运行器，但尚无可靠的本动作函数绑定，不能将其展示为独立算法。"
    else:
        summary_en = "Existing host computing functions and, where registered, the exact action adapter. A shared entrypoint selects reviewed subcontracts; it is not a separate algorithm for each action. Source excerpts may include local support code."
        summary_zh = "所列为既有宿主计算函数，以及已登记的精确动作适配器。共享入口按已审阅子合同分派，并非每个动作各有独立算法；源码片段可能包含同文件辅助代码。"
    sources = _Sources(_PACKAGE_ROOT)
    selected = {}
    pending = list(bindings)
    while pending:
        source = pending.pop(0)
        if source in selected:
            continue
        node = sources.definition(source)
        selected[source] = None
        local = {child.name for child in sources.read(source.module)[2].body if isinstance(child, _DEFINITIONS)}
        referenced = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        pending.extend(SourceDefinition(source.module, name) for name in sorted(local & referenced))
    module_order = {module: index for index, module in enumerate(dict.fromkeys(source.module for source in selected))}
    definitions = sorted(selected, key=lambda source: (module_order[source.module], sources.definition(source).lineno))
    files = _implementation_files(definitions, sources)
    return files, implementation_metadata(mode=mode, summary_en=summary_en, summary_zh=summary_zh,
                                         bindings=bindings, files=files)
