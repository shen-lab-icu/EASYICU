"""Tests for reporting-guideline checklists (O16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _EvRec:
    def __init__(self, evidence_id):
        self.evidence_id = evidence_id


def _recs(*ids):
    return [_EvRec(i) for i in ids]


# ---------------------------------------------------------------------------
# STROBE
# ---------------------------------------------------------------------------


def test_strobe_empty_inputs_are_mostly_open(ra):
    report = ra.build_strobe_checklist(
        evidence_records=[],
        bound_manuscript="",
    )
    assert report.name == "STROBE"
    # All items instantiated.
    assert len(report.items) == 22
    # With no evidence and no manuscript, nothing should be addressed.
    assert all(i.status in {"open", "partial"} for i in report.items)
    summary = report.summary()
    assert summary["n_addressed"] == 0


def test_strobe_common_ids_fill_key_methods_items(ra):
    recs = _recs(
        "analysis_plan",
        "research_context",
        "table_one",
        "missingness",
        "outcome_rate",
        "primary_association",
        "multiple_testing_report",
        "causal_audit_report",
        "manuscript_scaffold_bound",
        "literature_bundle",
        "hypothesis_blueprint",
        "cohort_audit",
    )
    scaffold = (
        "This retrospective cohort study analysed ICU patients. "
        "The study was supported by grant 123."
    )
    report = ra.build_strobe_checklist(
        evidence_records=recs,
        bound_manuscript=scaffold,
    )
    by_id = {i.item_id: i for i in report.items}
    # A cross-section of items should now be addressed.
    assert by_id["3"].status == "addressed"  # objectives/hypotheses
    assert by_id["4"].status == "addressed"  # design elements
    assert by_id["6"].status == "addressed"  # eligibility
    assert by_id["9"].status == "addressed"  # sources of bias
    assert by_id["12e"].status == "addressed"  # sensitivity
    assert by_id["15"].status == "addressed"  # outcome events
    assert by_id["16"].status == "addressed"  # primary association
    # Funding keyword picked up.
    assert by_id["22"].status == "addressed"
    # "retrospective cohort" keyword satisfies item 1a.
    assert by_id["1a"].status == "addressed"


def test_strobe_partial_when_only_some_requirements_present(ra):
    # Item 1a has both required evidence (manuscript_scaffold_bound) and
    # required keywords (cohort/observational/retrospective/case-control).
    # With only the evidence present, status should be "partial".
    recs = _recs("manuscript_scaffold_bound")
    report = ra.build_strobe_checklist(evidence_records=recs, bound_manuscript="")
    by_id = {i.item_id: i for i in report.items}
    assert by_id["1a"].status == "partial"


# ---------------------------------------------------------------------------
# TRIPOD+AI
# ---------------------------------------------------------------------------


def test_tripod_ai_has_27ish_items_and_coverage_grows_with_evidence(ra):
    blank = ra.build_tripod_ai_checklist(evidence_records=[], bound_manuscript="")
    assert blank.name == "TRIPOD+AI"
    assert len(blank.items) >= 25
    assert blank.summary()["coverage"] < 0.2

    full = ra.build_tripod_ai_checklist(
        evidence_records=_recs(
            "manuscript_scaffold_bound",
            "literature_bundle",
            "hypothesis_blueprint",
            "analysis_plan",
            "research_context",
            "table_one",
            "outcome_rate",
            "missingness",
            "model_performance",
            "cross_database_summary",
            "multiple_testing_report",
            "causal_audit_report",
            "reproducibility_envelope",
        ),
        bound_manuscript=(
            "Prediction model with AUROC and calibration. Subgroup by age "
            "and sex. Funding: institutional grant. Registered on ClinicalTrials.gov."
        ),
    )
    assert full.summary()["coverage"] > blank.summary()["coverage"]


def test_choose_checklist_selects_tripod_for_prediction_family(ra):
    assert "tripod_ai" in ra.choose_checklist("prediction_study")
    assert "tripod_ai" in ra.choose_checklist("validation_study")
    # Generic / association should get STROBE only.
    assert ra.choose_checklist("association_study") == ("strobe",)
    assert ra.choose_checklist(None) == ("strobe",)


def test_choose_checklist_routes_phenotype_to_internal_core(ra):
    # Clustering / trajectory have no EQUATOR guideline -> internal core, not
    # STROBE. The family key "trajectory_clustering" must be caught before the
    # prediction branch.
    assert ra.choose_checklist("trajectory_clustering") == ("internal_phenotype",)
    assert ra.choose_checklist("subphenotype clustering") == ("internal_phenotype",)
    assert ra.choose_checklist("phenotype discovery") == ("internal_phenotype",)


def test_checklist_names_for_kind_is_authoritative(ra):
    assert ra.checklist_names_for_kind("subphenotype_clustering") == (
        "internal_phenotype",
    )
    assert ra.checklist_names_for_kind("longitudinal_trajectory_analysis") == (
        "internal_phenotype",
    )
    assert "tripod_ai" in ra.checklist_names_for_kind("mortality_prediction")
    # Observational / unknown kinds default to STROBE.
    assert ra.checklist_names_for_kind("sepsis_onset") == ("strobe",)
    assert ra.checklist_names_for_kind(None) == ("strobe",)


def test_internal_phenotype_core_marks_trajectory_items_applicable_by_run(ra):
    # Cross-sectional clustering run: the trajectory-only item (P9) is N/A and
    # excluded from the denominator, not counted as an open failure.
    cross = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript="k-means clustering; silhouette and bootstrap stability",
    )
    p9 = {i.item_id: i for i in cross.items}["P9"]
    assert p9.status == "not_applicable"
    assert cross.summary()["n_not_applicable"] == 1

    # Longitudinal run: the trajectory keywords now apply and resolve P9.
    longit = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript="group-based trajectory model (GBTM) over time with BIC",
    )
    p9b = {i.item_id: i for i in longit.items}["P9"]
    assert p9b.status == "addressed"


def test_internal_phenotype_kind_overrides_fragile_trajectory_wording(ra):
    # M3 regression: a cross-sectional subphenotype_clustering run whose agent
    # MISLABELS the analysis "trajectory clustering" (step name) and whose prose
    # mentions a generic "clinical trajectory" must still mark the longitudinal
    # item P9 not-applicable — the authoritative task_kind wins over wording.
    mislabelled = (
        "We performed phenotype trajectory clustering on first-24h features. "
        "Decisions are made before a stable clinical trajectory is apparent. "
        "[01_phenotype_trajectory_clustering](evidence/x.json) k-means, silhouette."
    )
    cross = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript=mislabelled,
        task_kind="subphenotype_clustering",
    )
    assert {i.item_id: i for i in cross.items}["P9"].status == "not_applicable"

    # And the kind-unknown fallback must NOT flip to longitudinal on the generic
    # word "trajectory" alone (only on an explicit longitudinal-modelling cue).
    fallback = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript=mislabelled,
        task_kind=None,
    )
    assert {i.item_id: i for i in fallback.items}["P9"].status == "not_applicable"

    # A genuine longitudinal kind keeps P9 applicable even with terse prose.
    longit = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript="k-means on cross-sectional features",  # no cue in prose
        task_kind="longitudinal_trajectory_analysis",
    )
    assert {i.item_id: i for i in longit.items}["P9"].status != "not_applicable"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_markdown_includes_coverage_line(ra):
    report = ra.build_strobe_checklist(
        evidence_records=_recs("table_one"), bound_manuscript=""
    )
    md = report.to_markdown()
    assert "Coverage:" in md
    assert "| Item | Section | Statement | Status | Evidence |" in md


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_strobe_by_default(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reporting_checklist_strobe.md").exists()
    assert (run_dir / "reporting_checklist_strobe.json").exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
    assert "reporting_checklist_strobe" in ev_ids
    assert "reporting_checklist_strobe_json" in ev_ids
    # Finding emitted
    findings = [
        f for f in manifest["findings"] if f["validator"] == "reporting_checklist"
    ]
    assert len(findings) >= 1


def test_pipeline_checklist_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reporting_checklist=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "reporting_checklist_strobe.md").exists()


def test_pipeline_explicit_override_selects_tripod_too(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        reporting_checklist_names=["strobe", "tripod_ai"],
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reporting_checklist_strobe.md").exists()
    assert (run_dir / "reporting_checklist_tripod_ai.md").exists()


def test_pipeline_emits_internal_phenotype_core_when_requested(
    ra, synthetic_cohort, tmp_path
):
    # A phenotype-discovery run emits the internal core so its reporting
    # dimension is no longer permanently unscored.
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        reporting_checklist_names=["internal_phenotype"],
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reporting_checklist_internal_phenotype.md").exists()
    assert (run_dir / "reporting_checklist_internal_phenotype.json").exists()


# ---------------------------------------------------------------------------
# Semantic alias recovery (hashed evidence ids -> real artefact names)
# ---------------------------------------------------------------------------


class _EvRecRP:
    """Evidence record with the REAL shape: hashed id + relative_path."""

    def __init__(self, evidence_id, relative_path=None):
        self.evidence_id = evidence_id
        self.relative_path = relative_path


def test_strobe_credits_real_artefact_names_not_only_hashed_ids(ra):
    # Regression for the systematic false-open: step-output artefacts carry
    # hashed ids (table_cohort_attrition_492925c7) while the checklist keyed on
    # clean tokens, so flow/estimate items never matched. The matcher must
    # recover the agent's real artefact name from id/relative_path.
    recs = [
        _EvRecRP(
            "table_cohort_attrition_492925c7",
            "evidence/table_cohort_attrition_492925c7__cohort_attrition.csv",
        ),
        _EvRecRP(
            "table_final_results_summary_8a99df86",
            "evidence/table_final_results_summary_8a99df86__final_results_summary.csv",
        ),
        _EvRecRP("manuscript_scaffold_bound", "evidence/manuscript_scaffold_bound.md"),
    ]
    report = ra.build_strobe_checklist(
        evidence_records=recs,
        bound_manuscript="A retrospective cohort study.",
    )
    by_id = {i.item_id: i for i in report.items}
    # 13a (participant flow) credited by the real cohort_attrition artefact...
    assert by_id["13a"].status == "addressed"
    # ...16 (estimates) credited by the real final_results_summary artefact.
    assert by_id["16"].status == "addressed"
    # But a genuinely absent baseline table / outcome incidence stays honestly
    # open — semantic recovery must not fabricate credit.
    assert by_id["14a"].status == "open"
    assert by_id["15"].status == "open"


def test_strobe_alias_matched_by_suffixed_artifact_name(ra):
    # Regression (E1 12c): the agent emits a missing-data artefact under a
    # descriptive suffix (missingness_summary / missingness_profile); the
    # checklist item keyed on the bare token "missingness" must credit it via a
    # `_`-delimited prefix match, not false-open on exact membership.
    recs = [
        _EvRecRP(
            "table_missingness_summary_b2f04651",
            "evidence/table_missingness_summary_b2f04651__missingness_summary.csv",
        ),
    ]
    report = ra.build_strobe_checklist(
        evidence_records=recs, bound_manuscript="A retrospective cohort study."
    )
    by_id = {i.item_id: i for i in report.items}
    assert by_id["12c"].status == "addressed"  # missing-data item credited


def test_strobe_alias_prefix_does_not_overcredit_unrelated_token(ra):
    # Impartiality: a different artefact that merely shares letters must NOT
    # satisfy an item — the prefix match is `_`-delimited, not a substring.
    from easyicu.research_agent.reporting.reporting_checklist import _alias_satisfied

    assert _alias_satisfied("missingness", {"missingness_summary"}) is True
    assert _alias_satisfied("missingness", {"completeness_audit"}) is False
    assert _alias_satisfied("table_one", {"table_one_locked_cohort"}) is True
    assert _alias_satisfied("table_one", {"table_two_summary"}) is False


def test_strobe_13a_not_satisfied_by_table_one_alone(ra):
    # 13a is the participant-flow item; a baseline table must NOT silently
    # satisfy it (the prior mis-specified ``table_one`` alias did exactly that).
    recs = _recs("table_one")
    report = ra.build_strobe_checklist(evidence_records=recs, bound_manuscript="")
    by_id = {i.item_id: i for i in report.items}
    assert by_id["13a"].status == "open"
    assert by_id["14a"].status == "addressed"


# ---------------------------------------------------------------------------
# Keyword stem matching (Fix I): truncated stems must match inflected forms,
# short whole-word keywords must stay exact.
# ---------------------------------------------------------------------------


def test_keyword_stem_matches_inflected_forms(ra):
    from easyicu.research_agent.reporting.reporting_checklist import _keyword_hit

    # Truncated stems (length>=6, single lowercase token) match as word prefixes,
    # so a writer who DID describe scaling is credited. A trailing \b previously
    # made these impossible (\bstandardi\b never matches "standardized").
    assert _keyword_hit("features were standardized before clustering", "standardi")
    assert _keyword_hit("we applied z-score normalization", "normaliz")
    assert _keyword_hit("inputs were normalised per feature", "normalis")
    assert _keyword_hit("a reproducibility check across seeds", "reproducib")


def test_keyword_short_tokens_stay_exact_no_overmatch(ra):
    from easyicu.research_agent.reporting.reporting_checklist import _keyword_hit

    # Short / non-stem keywords keep exact whole-word matching so the stem rule
    # cannot, e.g., credit the BIC model-selection keyword against "bicarbonate".
    assert _keyword_hit("model selection used the BIC", "bic")
    assert not _keyword_hit("serum bicarbonate was 24 mmol/L", "bic")
    assert not _keyword_hit("the agency reviewed the protocol", "age")
    assert _keyword_hit("patient age was recorded", "age")
    # Hyphenated / multiword keywords are unaffected (kept exact).
    assert _keyword_hit("a z-score transform", "z-score")


def test_phenotype_p3_credited_when_manuscript_describes_scaling(ra):
    # Regression (M3/H3 P3): the writer reported standardisation ("normalized" /
    # "standardized") but the truncated-stem keyword never matched, so a real
    # scaling description was scored open. With stem prefix matching it resolves.
    report = ra.build_internal_phenotype_checklist(
        evidence_records=[],
        bound_manuscript=(
            "Features were normalized with z-score standardization and log1p "
            "before k-means clustering; silhouette and bootstrap stability."
        ),
        task_kind="subphenotype_clustering",
    )
    assert {i.item_id: i for i in report.items}["P3"].status == "addressed"


# ---------------------------------------------------------------------------
# P8 phenotype-native aliases (Fix J)
# ---------------------------------------------------------------------------


def test_phenotype_p8_credited_by_native_cluster_artifacts(ra):
    # The clustering agent emits cluster_characteristics / cluster_mortality, not
    # the generic STROBE names (primary_association / outcome_rate). P8 must
    # credit those phenotype-native artefacts (variable profile + outcome compare).
    recs = [
        _EvRecRP(
            "table_cluster_characteristics_acd6546f",
            "evidence/table_cluster_characteristics_acd6546f__cluster_characteristics.csv",
        ),
        _EvRecRP(
            "table_cluster_mortality_8e775741",
            "evidence/table_cluster_mortality_8e775741__cluster_mortality.csv",
        ),
    ]
    report = ra.build_internal_phenotype_checklist(
        evidence_records=recs,
        bound_manuscript="k-means clustering",
        task_kind="subphenotype_clustering",
    )
    assert {i.item_id: i for i in report.items}["P8"].status == "addressed"


# ---------------------------------------------------------------------------
# STROBE item 12d design-conditional N/A (Fix K)
# ---------------------------------------------------------------------------


def test_strobe_12d_not_applicable_for_cross_sectional_kinds(ra):
    # Loss to follow-up cannot apply to a fixed-endpoint / point-treatment design
    # with no longitudinal follow-up: STROBE 12d is "if applicable", so it is N/A
    # (removed from the denominator), not a penalised open.
    for kind in ("mortality_prediction", "sepsis_onset", "causal_inference"):
        report = ra.build_strobe_checklist(
            evidence_records=[], bound_manuscript="", task_kind=kind
        )
        item = {i.item_id: i for i in report.items}["12d"]
        assert item.status == "not_applicable", kind


def test_tripod_internal_validation_and_imbalance_and_calibration_figure_credited(ra):
    # Regression (M2): the prediction run genuinely handled class imbalance,
    # used a held-out patient-level split (internal validation), and emitted a
    # calibration/reliability FIGURE — but the TRIPOD items keyed on phrasings /
    # cross-database aliases the run did not use, so all three stayed open
    # (reporting 0.78). They must be credited from what the run actually produced.
    recs = [
        _EvRecRP(
            "figure_discrimination_calibration_16b4237d",
            "evidence/figure_discrimination_calibration_16b4237d__discrimination_calibration.png",
        ),
    ]
    manuscript = (
        "We addressed class-imbalance with imbalance-aware metrics. "
        "Discrimination and calibration were evaluated in a held-out patient-level split."
    )
    report = ra.build_tripod_ai_checklist(
        evidence_records=recs, bound_manuscript=manuscript
    )
    by_id = {i.item_id: i for i in report.items}
    assert by_id["9"].status == "addressed"   # class imbalance (bare stem)
    assert by_id["11"].status == "addressed"  # internal validation (held-out split)
    assert by_id["17"].status == "addressed"  # calibration figure
    # External validation (16) stays honestly open for a single-database study.
    assert by_id["16"].status == "open"


def test_tripod_items_not_satisfied_without_the_content(ra):
    # Negative: with neither the figure nor the prose, the same items stay open
    # (the fix credits real content, it does not blanket-pass).
    report = ra.build_tripod_ai_checklist(evidence_records=[], bound_manuscript="")
    by_id = {i.item_id: i for i in report.items}
    assert by_id["9"].status == "open"
    assert by_id["11"].status == "open"
    assert by_id["17"].status == "open"


def test_strobe_12d_applicable_for_followup_and_unknown_kinds(ra):
    # A survival kind genuinely has follow-up -> 12d stays applicable (open when
    # the writer did not address censoring; we never auto-N/A a follow-up design).
    surv = ra.build_strobe_checklist(
        evidence_records=[], bound_manuscript="", task_kind="survival_analysis"
    )
    assert {i.item_id: i for i in surv.items}["12d"].status == "open"
    # Unknown kind (None) stays applicable for backward compatibility.
    unknown = ra.build_strobe_checklist(
        evidence_records=[], bound_manuscript="", task_kind=None
    )
    assert {i.item_id: i for i in unknown.items}["12d"].status != "not_applicable"
