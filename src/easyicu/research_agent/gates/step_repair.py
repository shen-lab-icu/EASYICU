"""Focused repair guidance for deterministic step-contract findings."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

from ..contracts.ordered_stratified import is_ordered_stratified_analysis_step
from ..contracts.primary_cohort import _primary_analysis_cohort_attrition_candidate
from ..contracts.step_families import (
    _clustering_contract_applies,
    _cohort_change_contract_applies,
    _effect_contract_applies,
    _prediction_contract_applies,
)
from ..schema import AnalysisStep

def _primary_analysis_cohort_canonical_schema_rules(
    step: AnalysisStep,
) -> tuple[str, ...]:
    """Return schema rules only for the closed primary-cohort product family.

    These rules describe how to render Planner-owned eligibility decisions; they
    do not choose or modify any cohort predicate.  Exact typed products are the
    routing authority so benchmark prose, step ids, and clinical variable names
    cannot activate the contract.
    """

    if not _primary_analysis_cohort_attrition_candidate(step):
        return ()
    return (
        "Write the declared primary analysis-cohort product with every physical "
        "column from the host-authoritative locked cohort, preserving its exact "
        "ordered row identity and values; additional derived columns are allowed, "
        "but authoritative columns may not be dropped or changed.",
        "Write exact top-level integer fields `n_universe` and "
        "`n_final_analysis_cohort` in step_summary.json; do not hide either "
        "denominator in a nested mapping or under an approximate alias.",
        "For every declared cohort-flow or cohort-attrition table, write exactly "
        "one first `universe` row followed by exactly one row for every "
        "Planner-owned inclusion predicate and then every Planner-owned exclusion "
        "predicate, preserving their declared order. Do not split a predicate "
        "into an additional missingness, unknown-status, or complete-case row.",
        "Each such table must contain the canonical columns `criterion_id`, "
        "`n_at_start_rows`, `n_remaining_rows`, and `n_excluded_rows`. The universe "
        "row starts and remains at `n_universe` with zero excluded; every later "
        "row starts at the previous row's remaining count and satisfies "
        "n_excluded_rows = n_at_start_rows - n_remaining_rows.",
        "Set `criterion_id` to exactly `universe` for the first row. For predicates, "
        "use `{include|exclude}_{order:02d}_{normalized_concept_id}`, with one "
        "1-based order across the Planner inclusion list followed by the exclusion "
        "list; normalize concept_id to lowercase ASCII tokens separated by single "
        "underscores. Use the identical ordered ids and counts in every declared "
        "flow/attrition table.",
    )


def _cohort_predicate_partition_safety_rules(
    step: AnalysisStep,
) -> tuple[str, ...]:
    """Render mechanical safety rules for a declared cohort-flow owner.

    The rules make Planner-owned predicates executable without choosing their
    scientific meaning.  Routing relies on the same closed method/product
    contract used by the host cohort-change gate; prose, benchmark names, and
    variable names cannot activate it.
    """

    if not _cohort_change_contract_applies(step):
        return ()
    return (
        "Before evaluating each Planner-owned numeric eligibility predicate, "
        "coerce its declared value explicitly and build a finite-value mask; "
        "a non-null check alone is insufficient because positive or negative "
        "infinity can otherwise satisfy a threshold.",
        "Never allow a missing, unparseable, or non-finite value to satisfy a "
        "numeric eligibility predicate. Apply only the missing/invalid policy "
        "already declared by the Planner or host contract. If such values are "
        "observed and no policy authorizes their retained/excluded placement, "
        "fail the cohort step closed instead of inventing a scientific rule.",
        "At every predicate stage, construct retained and excluded masks that "
        "are mutually exclusive and exhaustive over the rows at that stage, "
        "and assert n_at_start_rows = n_remaining_rows + n_excluded_rows before "
        "writing outputs. Any optional missing/invalid diagnostic categories "
        "must also be mutually exclusive and exhaustive and must not become "
        "additional Planner predicates.",
    )


def _step_contract_repair_guidance(
    *,
    step: AnalysisStep,
    step_summary: Dict[str, Any],
    code: str,
    input_bindings: Optional[Mapping[str, Any]] = None,
) -> str:
    guidance: List[str] = []
    if not isinstance(step_summary, dict):
        # Hosted models sometimes emit a bare string as step_summary
        # when the generated code prints JSON as stdout. Treat non-dict
        # summaries as empty for repair guidance so we never crash in
        # the middle of the loop.
        step_summary = {}
    predictor = str(
        step_summary.get("primary_predictor") or step_summary.get("predictor") or ""
    ).strip()
    summary_text = json.dumps(step_summary or {}, ensure_ascii=False, default=str)
    assignment_binding = (
        input_bindings.get("artifact:assignment_model")
        if isinstance(input_bindings, Mapping)
        else None
    )
    assignment_contract = (
        assignment_binding.get("product_contract")
        if isinstance(assignment_binding, Mapping)
        else None
    )
    assignment_models = (
        assignment_contract.get("models")
        if isinstance(assignment_contract, Mapping)
        else None
    )
    if isinstance(input_bindings, Mapping):
        exact_input_keys = sorted(str(key) for key in input_bindings)
        if exact_input_keys:
            guidance.append(
                "Every step_summary.input_bindings receipt must use one of these "
                "exact host-resolved typed input keys, with no aliases or raw-column "
                "receipts: " + json.dumps(exact_input_keys, ensure_ascii=False)
            )
        else:
            guidance.append(
                "This step has no host-resolved typed inputs. Omit "
                "step_summary.input_bindings or write an empty list; raw Planner "
                "columns are bound separately and must not be reported as "
                "`raw:<column>` receipts."
            )
    if isinstance(assignment_models, list) and len(assignment_models) > 1:
        roster = [
            {
                key: model.get(key)
                for key in (
                    "model_id",
                    "analysis_set",
                    "fit_status",
                    "propensity_score_column",
                    "weight_column",
                )
                if model.get(key) is not None
            }
            for model in assignment_models
            if isinstance(model, Mapping)
        ]
        guidance.append(
            "The digest-bound assignment product is a Planner-owned model roster, "
            "not an ambiguous list from which the engine may choose a primary model. "
            "If its contract declares `diagnostic_model_id` or `selected_model_id`, "
            "use that exact entry. Otherwise compute and report the planned diagnostic "
            "separately for every fitted roster entry, keyed by its `model_id` and "
            "`analysis_set`; do not choose the first row, collapse variants, refit, or "
            "imply that one is primary. Preserve each entry's exact declared propensity "
            "and weight columns and its own analysis-set denominator. "
            "Current typed roster facts: "
            + json.dumps(roster, ensure_ascii=False, sort_keys=True)
        )
    guidance.extend(_primary_analysis_cohort_canonical_schema_rules(step))
    guidance.extend(_cohort_predicate_partition_safety_rules(step))
    if is_ordered_stratified_analysis_step(step):
        guidance.append(
            "Keep this as an agent-authored ordered-stratified analysis, but call "
            "the documented wilson_interval, cochran_armitage_trend, and "
            "jonckheere_terpstra_trend primitives. Use explicit CA scores, "
            "individual-level values for JT, nonzero bounded p-values with log-p "
            "metadata, one two-test Holm family, the canonical flat CSV columns, "
            "and a complete ordered_stratified_contract declaration. Spearman "
            "must not be substituted or relabelled as JT."
        )
    if predictor and predictor in summary_text:
        guidance.append(
            f"The machine summary identifies `{predictor}` as the primary predictor. "
            f"The repaired script must include `{predictor}` in the fitted design matrix."
        )
        lookup_patterns = (
            f"result.params['{predictor}'",
            f'result.params["{predictor}"',
            f"result.conf_int().loc['{predictor}'",
            f'result.conf_int().loc["{predictor}"',
            f"result.pvalues['{predictor}'",
            f'result.pvalues["{predictor}"',
            f"coef_table.loc['{predictor}'",
            f'coef_table.loc["{predictor}"',
        )
        if any(pattern in code for pattern in lookup_patterns):
            guidance.append(
                f"The previous script read model results for `{predictor}`. "
                f"Before fitting, build `x_cols` so `{predictor}` is present in `X.columns`; "
                "otherwise statsmodels will fit a model that cannot report the requested coefficient."
            )
    if "pd.get_dummies" in code and "drop_first" in code:
        guidance.append(
            "The previous script used dummy encoding. Rebuild the predictor list after "
            "dummy encoding: primary predictor + numeric covariates + generated dummy columns."
        )
    if (
        (
            (step_summary or {}).get("n_total") == 0
            or "zero-size array" in summary_text.lower()
            or "empty" in summary_text.lower()
        )
        and "pd.to_numeric" in code
        and "sex" in code
    ):
        guidance.append(
            "The previous script appears to have dropped the entire cohort by applying "
            "`pd.to_numeric(..., errors='coerce')` to `sex` before dummy encoding. "
            "Repair preprocessing by dummy-encoding `sex` first, rebuilding `x_cols`, "
            "then numeric-coercing only `[outcome] + x_cols` and dropping missing rows "
            "with that rebuilt list."
        )
        guidance.append(
            "Do not keep a null estimate summary for this contract failure. The repair "
            "should produce a numeric odds ratio when enough non-missing rows/events exist."
        )
    if (
        "pandas data cast to numpy dtype of object" in summary_text.lower()
        or "dtype of object" in summary_text.lower()
    ) and ("sm.logit(" in code.lower() or "pd.get_dummies" in code):
        guidance.append(
            "The prior script passed an object-dtype design matrix into statsmodels. "
            "After `pd.get_dummies(...)`, rebuild the predictor frame and convert every "
            "column in `X` with `pd.to_numeric(..., errors='coerce')`, cast boolean "
            "dummy columns to int when needed, and fit `sm.Logit(y, X.astype(float))`."
        )
        guidance.append(
            "Check the final design matrix dtypes before fitting and keep only rows with "
            "non-missing numeric predictors/outcome so the repaired script writes a "
            "non-null odds ratio."
        )
    expected = " ".join(str(item).lower() for item in (step.expected_outputs or []))
    effect_required = _effect_contract_applies(step)
    if effect_required:
        guidance.append(
            "This association step must write a non-null numeric primary effect "
            "estimate in step_summary.json, such as `adjusted_or`, `primary_or`, "
            "`odds_ratio`, or `primary_association_estimate`. Do not satisfy the "
            "contract by leaving association fields null."
        )
    prediction_required = _prediction_contract_applies(step)
    if prediction_required:
        guidance.append(
            "This prediction step must produce numeric AUROC and Brier/calibration metrics "
            "in step_summary.json (for example `cv_auroc_mean` and `brier_score`). "
            "Do not return only null metrics unless validation is truly impossible."
        )
        if "could not convert string to float" in summary_text.lower() or (
            "passthrough" in code and "onehot" in code.lower()
        ):
            guidance.append(
                "The failure indicates a categorical variable reached a numeric estimator. "
                "Use a scikit-learn ColumnTransformer with numeric features in a median-impute/"
                "scale branch and categorical features in a most-frequent-impute + "
                "OneHotEncoder(handle_unknown='ignore', sparse_output=False) branch. "
                "Never use `('onehot', 'passthrough')` for the categorical branch."
            )
        if "pd.to_numeric" in code and "categorical" in code.lower():
            guidance.append(
                "Do not numeric-coerce the full mixed feature frame. Keep categorical "
                "columns such as sex as object/string until the categorical transformer "
                "encodes them."
            )
    if "simpleimputer does not support data with dtype bool" in summary_text.lower():
        guidance.append(
            "A boolean dummy column reached SimpleImputer. Cast boolean dummy columns "
            "to int before fitting scikit-learn pipelines, or route them through a "
            "numeric branch with median imputation after conversion."
        )
    clustering_required = _clustering_contract_applies(
        method=str(step.method or ""),
        step_id=str(step.step_id or ""),
        intent=str(step.intent or ""),
        expected_outputs=step.expected_outputs or [],
    )
    if clustering_required:
        guidance.append(
            "This clustering step must write the selected `cluster_count` (or "
            "`n_clusters`) and its agent-declared native selection/stability "
            "evidence in step_summary.json. Record a full `cluster_selection` "
            "mapping (criterion, rule/direction, selected k, and at least two "
            "finite candidate values), or a substantive `cluster_stability` "
            "mapping with at least two resamples and a finite stability metric. "
            "A bare criterion string or artifact path does not satisfy this gate. "
            "Use the method-appropriate evidence (for example BIC/AIC/ICL, gap "
            "statistic, resampling stability, or silhouette when appropriate)."
        )
        guidance.append(
            "Use one of these exact top-level step_summary JSON shapes, populated "
            "with the values this script actually evaluated. Non-binding selection "
            "example: `\"cluster_selection\": {\"criterion\": "
            "\"silhouette_score\", \"selection_rule\": \"maximum\", "
            "\"direction\": \"maximize\", \"selected_n_clusters\": 2, "
            "\"candidates\": [{\"n_clusters\": 2, \"criterion_value\": 0.31}, "
            "{\"n_clusters\": 3, \"criterion_value\": 0.27}], \"rationale\": "
            "\"selected the evaluated maximum\"}`. Stability alternative: "
            "`\"cluster_stability\": {\"selected_n_clusters\": 2, "
            "\"n_resamples\": 3, \"mean_adjusted_rand_index\": 0.93}`. "
            "Replace every example value with the truthful selected k, criterion "
            "values, repeat count, and stability metric; do not leave only sibling "
            "scalars such as `selected_silhouette` or `selected_stability_ari`."
        )
        guidance.append(
            "Keep clustering self-contained: create labels, cluster characteristics, "
            "method/selection metadata, and the clustering figure inside this "
            "script. Add descriptive outcomes only when the plan declares them; "
            "do not rely on labels saved by another step."
        )
        guidance.append(
            "Also save a table artefact named `cluster_characteristics.csv` and "
            "the declared cluster-selection manifest so manuscript evidence aliases bind."
        )
    if "figure:" in expected:
        declared_figure_stems = [
            str(item).split(":", 1)[1].strip()
            for item in (step.expected_outputs or [])
            if str(item).strip().lower().startswith("figure:")
            and str(item).split(":", 1)[1].strip()
        ]
        guidance.append(
            "This step declares a figure output. Save a real figure file such as PNG/SVG/"
            "PDF/TIFF and record its path in step_summary.json using a key such as "
            "`figure_path`, `figure_file`, or `figure_files`."
        )
        for stem in declared_figure_stems:
            quoted_stem = json.dumps(stem, ensure_ascii=False)
            guidance.append(
                f"For declared `figure:{stem}`, call the host helper directly as "
                "`saved = save_publication_figure(fig=fig, out_dir=out_dir, "
                f"stem={quoted_stem}, contract=contract)`. It writes the canonical "
                f"same-stem companion `{stem}.figure_contract.json`; record "
                "`saved[\"contract\"]` in step_summary.json. Do not manually write, "
                f"rename, or advertise the underscore alias "
                f"`{stem}_figure_contract.json`; it must not replace the canonical "
                "dot-suffix companion."
            )
        guidance.append(
            "In every top-level FigureContract, `source_data` must be one local CSV "
            "basename string or a flat list of local CSV basename strings from the "
            "current step output directory. Never write a dict, list of dicts, "
            "evidence object, absolute path, or path metadata there; put evidence ids "
            "in panel `evidence_ids` and other provenance in step_summary metadata."
        )
        guidance.append(
            "Build figure source-data CSVs from the actual plotted analytic rows "
            "before rendering, preserving a host-verifiable source row/key and the "
            "upstream value columns; never reconstruct source data from Matplotlib "
            "Axes, collections, lines, patches, artist coordinates, or rendered "
            "pixels: a canvas is not scientific provenance. Every panel contract "
            "must declare a panel title, reader-facing claim, role, local "
            "source-data basename(s), source columns, and the evidence ids that bind "
            "those rows. Do not invent a synthetic source table such as a rendered "
            "panel; if the plotted values cannot be traced to a declared analytic "
            "table or statistic, fail closed instead of emitting the figure."
        )
    if not guidance:
        guidance.append(
            "Repair the script so each expected output is written as machine-readable "
            "numbers in step_summary.json, or write a precise skipped/error reason."
        )
    return "\n".join(f"- {item}" for item in guidance)

