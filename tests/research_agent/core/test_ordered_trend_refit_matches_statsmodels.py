"""The adjusted ordered-trend refit is the primary model with one per-level term.

An ordered three-level exposure, a binary outcome and two covariates: the
ordered-trend owner refits the primary's own estimator (logistic for an odds
ratio, modified Poisson for a risk ratio, cluster-robust when the primary is)
with the exposure coded 0, 1, 2.  Its per-level ratio, interval and Wald
p-value must equal an independent statsmodels fit on the same complete rows;
the replay catches an edited trend row or envelope; and the result reaches a
secondary claim and the Writer's evidence.  Synthetic cohorts only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip("statsmodels.api")
stats = pytest.importorskip("scipy.stats")

from easyicu.research_agent.authority.scientific_claims import (  # noqa: E402
    bind_scientific_claim_drafts,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.contracts.model_terms import ModelTermSpec  # noqa: E402
from easyicu.research_agent.contracts.model_tokens import (  # noqa: E402
    ASSOCIATION_LOGIT_ESTIMATOR,
    ASSOCIATION_OLS_ESTIMATOR,
    ASSOCIATION_RR_ESTIMATOR,
)
from easyicu.research_agent.contracts.ordered_stratified import (  # noqa: E402
    ordered_stratified_numeric_findings,
    ordered_stratified_script_findings,
    ordered_stratified_structure_findings,
)
from easyicu.research_agent.contracts.ordered_stratified_spec import (  # noqa: E402
    OrderedTrendUnsupported,
    ordered_trend_model_for,
)
from easyicu.research_agent.execution.runners.ordered_stratified_executor import (  # noqa: E402
    ordered_stratified_spec_for_step,
    run_ordered_stratified_from_env,
)
from easyicu.research_agent.reporting.writer_evidence import (  # noqa: E402
    _render_writer_evidence_digest,
)
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep  # noqa: E402
from tests.support.host_runner_cases import ordered_stratified_primary_binding  # noqa: E402

LEVELS = ["1", "2", "3"]


def _cohort(*, log_link: bool, n: int = 1200, seed: int = 20261026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    band = rng.choice([1, 2, 3], size=n, p=[0.45, 0.35, 0.20]).astype(float)
    age = rng.normal(64.0, 12.0, size=n)
    sex = rng.integers(0, 2, size=n).astype(float)
    linear = 0.45 * (band - 1) + 0.02 * (age - 64.0) + 0.25 * sex
    risk = np.exp(-2.4 + 0.6 * linear) if log_link else 1 / (1 + np.exp(-(-1.6 + linear)))
    death = (rng.random(n) < np.clip(risk, 0.0, 0.95)).astype(float)
    frame = pd.DataFrame({
        "patient_id": rng.integers(0, n // 2, size=n).astype(str),
        "sofa_band": band, "death": death, "age": age, "sex": sex,
    })
    # Rows the models must drop: an unknown band, an unknown outcome, unknown ages.
    frame.loc[[3, 17], "sofa_band"] = np.nan
    frame.loc[[5], "death"] = np.nan
    # Enough unmeasured ages (both outcomes among them) for a missing-category
    # state to be estimable when a primary keeps it.
    frame.loc[list(range(40, 1200, 30)), "age"] = np.nan
    return frame


def _plan(
    *, method_family: str, clustered: bool, missing_category: bool = False,
) -> tuple[AnalysisPlan, AnalysisStep]:
    requirement = {
        "requirement_id": "primary",
        "outcome": "death",
        "outcome_type": "binary",
        "method_family": method_family,
        "exposure_source": "sofa_band",
        "analysis_role": "primary",
        "analysis_set": "complete_case",
        "covariates": ["age", "sex"],
        "model_terms": [
            {"name": "sofa_band", "role": "exposure", "coding": "categorical", "levels": LEVELS,
             "reference_level": "1", "transform": "treatment_contrast"},
            {"name": "age", "role": "covariate", "coding": "continuous", "transform": "identity"},
            {"name": "sex", "role": "covariate", "coding": "binary", "levels": ["0", "1"],
             "reference_level": "0", "transform": "treatment_contrast"},
        ],
        "exposure_levels": LEVELS,
        "exposure_reference_level": "1",
        "primary_contrast_level": "3",
    }
    if clustered:
        requirement["dependence"] = {"group_source": "patient_id", "group_derivation": "identity"}
    if missing_category:
        # Unmeasured age is kept as its own state, so the model keeps those rows.
        requirement["analysis_set"] = "source_aware"
        requirement["baseline_missing_handling"] = {"covariates": ["age"]}
    primary = AnalysisStep.model_validate({
        "step_id": "primary_model",
        "planned_analysis_role": "primary",
        "intent": "Fit the declared adjusted primary model.",
        "inputs": ["artifact:analysis_cohort", "sofa_band", "death", "age", "sex"],
        "expected_outputs": ["table:adjusted_association_estimates"],
        "method": "adjusted_association_models",
        "model_requirements": [requirement],
    })
    trend = AnalysisStep.model_validate({
        "step_id": "ordinal_trend",
        "planned_analysis_role": "secondary",
        "intent": "Estimate the ordered trend of death across the declared bands.",
        "inputs": ["artifact:analysis_cohort", "table:adjusted_association_estimates", "sofa_band", "death"],
        "expected_outputs": ["table:ordinal_trend"],
        "method": "ordinal_stratified_descriptive_analysis",
        "scientific_action_id": "association.ordinal_trend",
    })
    return AnalysisPlan(research_question="Synthetic ordered trend", steps=[primary, trend]), trend


def _bind(
    tmp_path: Path, frame: pd.DataFrame, monkeypatch, plan: AnalysisPlan,
    primary_update: dict | None = None,
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "ordinal_trend" / "outputs"
    out_dir.mkdir(parents=True)
    cohort_path = run_dir / "cohort.parquet"
    frame.to_parquet(cohort_path, index=False)
    digest = hashlib.sha256(cohort_path.read_bytes()).hexdigest()
    identity = {
        "input_key": "artifact:analysis_cohort", "declared_kind": "artifact",
        "product": "analysis_cohort", "evidence_id": "ev-cohort", "sha256": digest,
    }
    resolved = {"step_id": "ordinal_trend", "inputs": {
        "artifact:analysis_cohort": {
            "relative_path": "cohort.parquet", "sha256": digest, "declared_kind": "artifact",
            "product": "analysis_cohort", "evidence_id": "ev-cohort", "identity_row": identity,
            "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        },
        "table:adjusted_association_estimates": ordered_stratified_primary_binding(
            plan, frame, run_dir, requirement_update=primary_update,
        ),
    }}
    resolved_path = run_dir / "resolved_inputs.json"
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    monkeypatch.setenv("STEP_OUT_DIR", str(out_dir))
    monkeypatch.setenv("EASYICU_RUN_DIR", str(run_dir))
    monkeypatch.setenv("EASYICU_RESOLVED_INPUTS_JSON", str(resolved_path))
    return out_dir, cohort_path


def _run(
    tmp_path, monkeypatch, *, method_family: str, clustered: bool = False,
    missing_category: bool = False, primary_update: dict | None = None,
):
    frame = _cohort(log_link=method_family == ASSOCIATION_RR_ESTIMATOR)
    plan, step = _plan(
        method_family=method_family, clustered=clustered, missing_category=missing_category,
    )
    out_dir, cohort_path = _bind(tmp_path, frame, monkeypatch, plan, primary_update)
    spec = ordered_stratified_spec_for_step(step, plan=plan)
    assert spec is not None
    summary = run_ordered_stratified_from_env(
        spec_payload=spec.model_dump(mode="json"),
        typed_cohort_input="artifact:analysis_cohort",
        analysis_role="secondary",
    )
    return frame, step, out_dir, cohort_path, json.loads(json.dumps(summary))


def _independent_fit(frame: pd.DataFrame, *, method_family: str, clustered: bool):
    rows = frame.dropna(subset=["sofa_band", "death", "age", "sex"])
    design = pd.DataFrame({
        "const": 1.0,
        "per_level": rows["sofa_band"].map({1.0: 0.0, 2.0: 1.0, 3.0: 2.0}),
        "age": rows["age"],
        "sex": rows["sex"],
    })
    covariance = (
        {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(rows["patient_id"])[0]}}
        if clustered else {}
    )
    if method_family == ASSOCIATION_RR_ESTIMATOR:
        result = sm.GLM(rows["death"], design, family=sm.families.Poisson()).fit(
            **(covariance or {"cov_type": "HC0"})
        )
    else:
        result = sm.Logit(rows["death"], design).fit(disp=False, **covariance)
    beta = float(result.params["per_level"])
    se = float(result.bse["per_level"])
    critical = float(stats.norm.ppf(0.975))
    return {
        "n": len(rows), "events": int(rows["death"].sum()),
        "estimate": math.exp(beta),
        "ci_low": math.exp(beta - critical * se), "ci_high": math.exp(beta + critical * se),
        "standard_error": se, "p_value": float(result.pvalues["per_level"]),
    }


@pytest.mark.parametrize(
    ("method_family", "clustered", "scale"),
    [
        (ASSOCIATION_LOGIT_ESTIMATOR, False, "odds_ratio"),
        (ASSOCIATION_RR_ESTIMATOR, False, "risk_ratio"),
        (ASSOCIATION_LOGIT_ESTIMATOR, True, "odds_ratio"),
        (ASSOCIATION_RR_ESTIMATOR, True, "risk_ratio"),
    ],
    ids=["logit", "modified_poisson", "logit_clustered", "modified_poisson_clustered"],
)
def test_the_trend_refit_matches_an_independent_statsmodels_fit(
    tmp_path, monkeypatch, method_family, clustered, scale,
):
    frame, step, out_dir, cohort_path, summary = _run(
        tmp_path, monkeypatch, method_family=method_family, clustered=clustered,
    )
    expected = _independent_fit(frame, method_family=method_family, clustered=clustered)
    record = summary["ordered_stratified_contract"]["adjusted_trend"]

    assert record["effect_scale"] == scale
    assert record["covariates"] == ["age", "sex"]
    assert record["level_scores"] == [0, 1, 2]
    assert record["variance_estimator"] == (
        "cluster_robust" if clustered
        else "sandwich_robust" if method_family == ASSOCIATION_RR_ESTIMATOR
        else "model_based"
    )
    assert (record["n"], record["n_events"]) == (expected["n"], expected["events"])
    for key in ("estimate", "ci_low", "ci_high", "standard_error", "p_value"):
        assert record[key] == pytest.approx(expected[key], rel=1e-6), key
    assert expected["p_value"] < 0.05  # the synthetic gradient is real

    trend = pd.read_csv(out_dir / "ordinal_trend.csv")
    assert list(trend["test_id"]) == ["cochran_armitage", "adjusted_ordinal_linear_trend"]
    adjusted = trend.iloc[1]
    assert adjusted["effect_size_name"] == f"adjusted_{scale}_per_level"
    assert adjusted["estimate"] == pytest.approx(expected["estimate"], rel=1e-6)
    assert adjusted["p_value"] == pytest.approx(expected["p_value"], rel=1e-6)
    assert pd.isna(adjusted["adjusted_p"])  # model-based: outside the Holm family
    assert trend.iloc[0]["family_size"] == 1
    # Only the declared products sit at the top of the step's outputs.
    assert sorted(path.name for path in out_dir.iterdir() if path.is_file()) == [
        "ordered_stratified_outcomes.csv", "ordinal_trend.csv", "step_summary.json",
    ]

    assert not ordered_stratified_structure_findings(step=step, step_summary=summary)
    assert not ordered_stratified_numeric_findings(
        cohort_path=cohort_path, step=step, out_dir=out_dir, step_summary=summary,
    )


def test_an_edited_trend_row_or_envelope_fails_the_replay(tmp_path, monkeypatch):
    _, step, out_dir, cohort_path, summary = _run(
        tmp_path, monkeypatch, method_family=ASSOCIATION_LOGIT_ESTIMATOR,
    )
    trend_path = out_dir / "ordinal_trend.csv"
    original = pd.read_csv(trend_path)
    edited = original.copy()
    edited.loc[1, "estimate"] = float(edited.loc[1, "estimate"]) * 1.1
    edited.to_csv(trend_path, index=False)
    assert ordered_stratified_numeric_findings(
        cohort_path=cohort_path, step=step, out_dir=out_dir, step_summary=summary,
    )
    original.drop(index=1).to_csv(trend_path, index=False)
    assert ordered_stratified_numeric_findings(
        cohort_path=cohort_path, step=step, out_dir=out_dir, step_summary=summary,
    )
    original.to_csv(trend_path, index=False)

    for path, value in (
        (("reportable_trend_results", "estimate"), 9.9),
        (("reportable_trend_results", "effect_scale"), "risk_ratio"),
        (("reportable_secondary_results", "adjusted_trend", "ci_high"), 9.9),
        # No ceiling may read causal, and the crude test may not read adjusted.
        (("reportable_secondary_results", "adjusted_trend", "interpretation_ceiling"),
         "secondary_adjusted_causal_effect"),
        (("reportable_secondary_results", "interpretation_ceiling"), "secondary_causal_effect"),
        (("reportable_secondary_results", "binary_trend", "adjustment"), "adjusted"),
    ):
        tampered = json.loads(json.dumps(summary))
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        kinds = {item.detail["kind"] for item in ordered_stratified_structure_findings(step=step, step_summary=tampered)}
        assert "trend_reporting_mismatch" in kinds, path
    dropped = json.loads(json.dumps(summary))
    dropped["ordered_stratified_contract"]["adjusted_trend"].pop("standard_error")
    kinds = {item.detail["kind"] for item in ordered_stratified_structure_findings(step=step, step_summary=dropped)}
    assert "invalid_adjusted_trend" in kinds


@pytest.mark.parametrize(
    ("method_family", "estimand"),
    [
        (ASSOCIATION_LOGIT_ESTIMATOR, "adjusted odds ratio per one-level increase across its ordered levels"),
        (ASSOCIATION_RR_ESTIMATOR, "adjusted risk ratio per one-level increase across its ordered levels"),
    ],
)
def test_the_trend_reaches_a_secondary_claim_and_the_writer(tmp_path, monkeypatch, method_family, estimand):
    _, step, _, _, summary = _run(tmp_path, monkeypatch, method_family=method_family)
    record = summary["ordered_stratified_contract"]["adjusted_trend"]

    [draft] = derive_scientific_claim_drafts(summary)
    assert draft.analysis_role == "secondary"
    assert draft.estimand == estimand
    assert draft.direction == "positive"
    assert draft.adjusted_for == ["age", "sex"]
    assert (draft.point_estimate, draft.interval_lower, draft.interval_upper) == (
        record["estimate"], record["ci_low"], record["ci_high"],
    )
    [claim] = bind_scientific_claim_drafts(
        [draft.model_dump(mode="json")], step_id=step.step_id, evidence_id="trend_summary",
    )
    sentence = claim.render_reader_text()
    assert sentence.startswith("After adjustment for age and sex, sofa band was positively associated")
    assert estimand in sentence

    digest = _render_writer_evidence_digest(
        [{"step_id": step.step_id, "status": "ok", "step_summary": summary}]
    )
    assert '"adjusted_trend"' in digest
    assert '"schema_version": "easyicu.ordered_stratified_reporting/2"' in digest

    # A crude-only summary (contract 1.0 era) stays readable and claims nothing.
    legacy = {key: value for key, value in summary.items() if key != "reportable_trend_results"}
    assert derive_scientific_claim_drafts(legacy) == []


def test_the_trend_claim_survives_strict_numeric_binding(tmp_path, monkeypatch):
    """The claim's ratio and interval bind to the step summary's own numbers."""

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.manuscript_post import (
        bind_numeric_values,
        drop_untraceable_numeric_sentences,
    )

    _, step, _, _, summary = _run(tmp_path / "fit", monkeypatch, method_family=ASSOCIATION_RR_ESTIMATOR)
    store = EvidenceStore(tmp_path / "evidence", enforcement_mode="strict")
    record = store.register_json(
        kind="statistic", description="Ordered trend", payload=summary, filename="summary.json",
        evidence_id="ordinal_trend_summary", produced_by_step=step.step_id,
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id=step.step_id, evidence_id=record.evidence_id, summary=summary,
    )
    [claim] = store.scientific_claims()
    records = [{
        "step_id": step.step_id, "status": "ok", "generation_mode": "deterministic_standard",
        "step_summary": summary, "step_summary_evidence_id": record.evidence_id,
        "evidence_ids": [record.evidence_id],
    }]
    bound = store.bind_manuscript(
        "## Results\n\n### Ordered trend\n\n" + claim.placeholder, per_step_records=records,
    )
    filtered, removed = drop_untraceable_numeric_sentences(
        bound, evidence=store, per_step_records=records,
    )
    _, bindings, untraced = bind_numeric_values(bound, evidence=store, per_step_records=records)

    assert "adjusted risk ratio per one-level increase" in bound
    assert removed == [] and filtered == bound
    assert not untraced
    trend = summary["reportable_trend_results"]
    assert {round(float(item.canonical), 3) for item in bindings.values()} >= {
        round(trend["estimate"], 3), round(trend["lower"], 3), round(trend["upper"], 3),
    }


def test_the_refit_keeps_the_primarys_missing_category_rows(tmp_path, monkeypatch):
    """A primary that keeps unmeasured age as its own state: the refit keeps those rows too."""

    frame, step, out_dir, cohort_path, summary = _run(
        tmp_path, monkeypatch, method_family=ASSOCIATION_LOGIT_ESTIMATOR, missing_category=True,
    )
    record = summary["ordered_stratified_contract"]["adjusted_trend"]
    kept = frame.dropna(subset=["sofa_band", "death", "sex"])
    assert len(kept) > len(frame.dropna(subset=["sofa_band", "death", "age", "sex"]))
    assert record["n"] == record["primary_n"] == len(kept)
    assert record["n_events"] == record["primary_n_events"] == int(kept["death"].sum())
    assert record["primary_model_id"] == "primary"
    assert not ordered_stratified_structure_findings(step=step, step_summary=summary)
    assert not ordered_stratified_numeric_findings(
        cohort_path=cohort_path, step=step, out_dir=out_dir, step_summary=summary,
    )
    unbound = json.loads(json.dumps(summary))
    unbound["ordered_stratified_contract"]["adjusted_trend"]["primary_n"] -= 2
    kinds = {
        item.detail["kind"]
        for item in ordered_stratified_structure_findings(step=step, step_summary=unbound)
    }
    assert "invalid_adjusted_trend" in kinds


@pytest.mark.parametrize(
    ("missing_category", "primary_update", "differing"),
    [
        # The primary dropped the unmeasured-age rows the refit keeps.
        (True, {"baseline_missing_handling": None}, "n, n_events"),
        # The primary was adjusted for age alone.
        (False, {
            "covariates": ["age"],
            "model_terms": [
                ModelTermSpec(name="sofa_band", role="exposure", coding="categorical", levels=LEVELS,
                              reference_level="1", transform="treatment_contrast"),
                ModelTermSpec(name="age", role="covariate", coding="continuous", transform="identity"),
            ],
        }, "covariates"),
    ],
    ids=["missing_policy_dropped", "roster_changed"],
)
def test_a_refit_that_is_not_the_primary_model_fails_the_step(
    tmp_path, monkeypatch, missing_category, primary_update, differing,
):
    with pytest.raises(RuntimeError, match="not the primary model with one per-level term") as failed:
        _run(
            tmp_path, monkeypatch, method_family=ASSOCIATION_LOGIT_ESTIMATOR,
            missing_category=missing_category, primary_update=primary_update,
        )
    assert differing in str(failed.value)


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        ({"covariates": None}, "no adjustment set"),
        ({"effect_modifiers": [
            ModelTermSpec(name="sex", role="covariate", coding="binary", levels=["0", "1"],
                          reference_level="0", transform="treatment_contrast"),
        ]}, "effect modifiers"),
        ({"method_family": ASSOCIATION_OLS_ESTIMATOR}, "neither the logistic nor the modified-Poisson"),
        ({"model_terms": [
            ModelTermSpec(name="sofa_band", role="exposure", coding="binary", levels=["1", "2"],
                          reference_level="1", transform="treatment_contrast"),
        ]}, "at least three ordered levels"),
    ],
    ids=["no_roster", "effect_modifiers", "linear_estimator", "two_levels"],
)
def test_the_compiler_and_the_owner_refuse_the_same_primaries(update, reason):
    """One predicate: the compiler stops on it, and the owner never owns such a step."""

    plan, step = _plan(method_family=ASSOCIATION_LOGIT_ESTIMATOR, clustered=False)
    primary = plan.steps[0]
    requirement = primary.model_requirements[0].model_copy(update=update)
    with pytest.raises(OrderedTrendUnsupported, match=reason):
        ordered_trend_model_for(requirement)
    changed = plan.model_copy(update={"steps": [
        primary.model_copy(update={"model_requirements": [requirement]}), step,
    ]})
    assert ordered_stratified_spec_for_step(step, plan=changed) is None
    assert ordered_trend_model_for(primary.model_requirements[0]).effect_measure == "odds_ratio"


def test_a_null_trend_claims_no_clear_association(tmp_path, monkeypatch):
    """The claim's direction is read from its interval, never from the point estimate."""

    _, _, _, _, summary = _run(tmp_path, monkeypatch, method_family=ASSOCIATION_LOGIT_ESTIMATOR)
    for (estimate, lower, upper), direction in (
        ((1.2, 0.9, 1.6), "no_clear_association"),
        ((0.8, 0.6, 1.1), "no_clear_association"),
        ((0.7, 0.5, 0.9), "negative"),
        ((1.3, 1.1, 1.5), "positive"),
    ):
        edited = json.loads(json.dumps(summary))
        edited["reportable_trend_results"].update(estimate=estimate, lower=lower, upper=upper)
        [draft] = derive_scientific_claim_drafts(edited)
        assert draft.direction == direction, (estimate, lower, upper)


def test_the_writer_reads_a_v2_envelope_only_when_it_is_complete(tmp_path, monkeypatch):
    _, step, _, _, summary = _run(tmp_path, monkeypatch, method_family=ASSOCIATION_LOGIT_ESTIMATOR)

    def digest(payload):
        return _render_writer_evidence_digest(
            [{"step_id": step.step_id, "status": "ok", "step_summary": payload}]
        )

    assert '"reportable_secondary_results"' in digest(summary)
    for part in ("binary_level_summaries", "binary_trend", "adjusted_trend"):
        edited = json.loads(json.dumps(summary))
        edited["reportable_secondary_results"].pop(part)
        assert '"reportable_secondary_results"' not in digest(edited), part
    # A named continuous outcome needs its summaries and test.
    edited = json.loads(json.dumps(summary))
    edited["reportable_secondary_results"]["continuous_outcome"] = "los_days"
    assert '"reportable_secondary_results"' not in digest(edited)


def test_a_binary_only_script_needs_no_jonckheere_terpstra():
    _, step = _plan(method_family=ASSOCIATION_LOGIT_ESTIMATOR, clustered=False)
    code = "wilson_interval(1, 2)\ncochran_armitage_trend([1, 2, 3], [4, 5, 6])\n"
    assert not ordered_stratified_script_findings(step=step, script_text=code)
    continuous = step.model_copy(update={"inputs": [*step.inputs, "los_days"]})
    [finding] = ordered_stratified_script_findings(step=continuous, script_text=code)
    assert finding.detail["missing_calls"] == ["jonckheere_terpstra_trend"]
