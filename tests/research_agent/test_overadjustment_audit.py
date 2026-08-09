"""Overadjustment hard-block: a primary model that conditioned on a constituent
of a composite/derived exposure is an objective design error, routed through the
same in-run repair loop the exposure-contract auditor uses (re-fit, no restart).

Twin of test_exposure_contract_audit.py: that one enforces the model MUST use
the exposure; this one enforces it must NOT adjust for the exposure's own
constituents.
"""

import csv
from pathlib import Path
from types import SimpleNamespace

from easyicu.research_agent.plan_utils import (
    _name_intends_covariates,
    _primary_exposure_overadjustment_findings,
    read_adjustment_covariates,
    read_model_covariate_names,
)
from easyicu.research_agent.schema import AnalysisStep, PlannedModelRequirement


def _step(
    step_id="06_primary_association",
    *,
    primary_source=None,
    source_role="primary",
):
    model_requirements = []
    if primary_source is not None:
        model_requirements.append(
            PlannedModelRequirement(
                requirement_id=f"{source_role}_model",
                outcome="death",
                outcome_type="binary",
                method_family="logistic_regression",
                exposure_source=primary_source,
                analysis_role=source_role,
                analysis_set="complete_case",
                required_for_step_success=True,
            )
        )
    return AnalysisStep(
        step_id=step_id,
        intent="Estimate the adjusted association.",
        planned_analysis_role=(
            "primary"
            if not model_requirements or source_role == "primary"
            else "auxiliary"
        ),
        method=(
            "adjusted_association_models"
            if model_requirements
            else "logistic_regression"
        ),
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=model_requirements,
    )


def _ctx(required="sepsis3"):
    return SimpleNamespace(primary_exposure=required)


def _write_coef_table(out_dir: Path, variables, *, name="primary_association.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "coef", "odds_ratio"])
        w.writeheader()
        for v in variables:
            w.writerow({"variable": v, "coef": "0.1", "odds_ratio": "1.1"})


# ---------------------------------------------------------------------------
# read_model_covariate_names — content-based coefficient-table detection
# ---------------------------------------------------------------------------


def test_reader_detects_coef_table_regardless_of_filename(tmp_path: Path):
    # Real runs name this primary_association.csv / model_coefficients.csv, not
    # regression_results.csv: detection must ride the column contract.
    _write_coef_table(
        tmp_path, ["const", "age", "sofa_max"], name="model_coefficients.csv"
    )
    assert read_model_covariate_names(tmp_path) == ["age", "sofa_max"]  # const dropped


def test_reader_detects_coef_table_with_term_identifier_column(tmp_path: Path):
    # R broom::tidy and many hand-rolled statsmodels exports name the identifier
    # column ``term`` (+ odds_ratio/coefficient), not ``variable``. The real E1
    # run wrote exactly this header and the overadjustment guard read [] -> 0
    # fires -> the overadjustment slipped through. Detection must ride the id
    # column being any standard name, not literally ``variable``.
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "adjusted_odds_ratios.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(
            fh, fieldnames=["model_id", "term", "coefficient", "odds_ratio"]
        )
        w.writeheader()
        for v in ["const", "sepsis3", "age", "map_first"]:
            w.writerow(
                {"model_id": "m1", "term": v, "coefficient": "0.1", "odds_ratio": "1.1"}
            )
    assert read_model_covariate_names(tmp_path) == ["sepsis3", "age", "map_first"]


def test_reader_excludes_structured_exposure_term_from_adjustment_set(
    tmp_path: Path,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "coefficients.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["term", "term_role", "source_variable", "odds_ratio"]
        )
        w.writeheader()
        w.writerow(
            {
                "term": "sep3_sofa2_max",
                "term_role": "exposure",
                "source_variable": "sep3_sofa2_max",
                "odds_ratio": "1.6",
            }
        )
        w.writerow(
            {
                "term": "age",
                "term_role": "adjustment",
                "source_variable": "age",
                "odds_ratio": "1.02",
            }
        )

    assert read_model_covariate_names(tmp_path) == ["age"]


def test_reader_detects_predictor_identifier_column(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "coefs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["predictor", "beta"])
        w.writeheader()
        w.writerow({"predictor": "age", "beta": "0.02"})
    assert read_model_covariate_names(tmp_path) == ["age"]


def test_reader_ignores_term_table_without_coefficient_column(tmp_path: Path):
    # A ``term`` column alone (no coef-value column) is not a model table — the
    # value-column requirement is what keeps the broadened id set safe.
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "glossary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["term", "definition"])
        w.writeheader()
        w.writerow({"term": "sofa_max", "definition": "max SOFA in window"})
    assert read_model_covariate_names(tmp_path) == []


def test_reader_ignores_non_model_variable_tables(tmp_path: Path):
    # missingness.csv has a `variable` column but no coefficient column: it must
    # NOT inject phantom covariates into the overadjustment check.
    tmp_path.mkdir(parents=True, exist_ok=True)
    with (tmp_path / "missingness.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variable", "missing_frac"])
        w.writeheader()
        w.writerow({"variable": "sofa_max", "missing_frac": "0.66"})
    assert read_model_covariate_names(tmp_path) == []


def test_reader_missing_dir_degrades_to_empty(tmp_path: Path):
    assert read_model_covariate_names(tmp_path / "does_not_exist") == []


# ---------------------------------------------------------------------------
# _primary_exposure_overadjustment_findings — the hard block
# ---------------------------------------------------------------------------


def test_flags_overadjustment_for_exposure_constituent(tmp_path: Path):
    # Sepsis-3 is defined via SOFA, so adjusting for SOFA is overadjustment.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sofa_max"])
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "error"
    assert f.detail["kind"] == "overadjustment"
    assert f.detail["exposure"] == "sepsis3"
    assert f.detail["offending_covariates"] == ["sofa_max"]
    assert "sofa_max" in f.message


def test_overadjustment_detector_does_not_match_incidental_substrings():
    from easyicu.research_agent.icu_rules import detect_overadjustment

    covariates = [
        "acute_pancreatitis",
        "history_of_pancreatitis",
        "increase_from_baseline",
        "mapped_diagnosis",
        "age",
    ]

    assert detect_overadjustment("sofa", covariates) == []
    assert detect_overadjustment("kdigo", covariates) == []
    assert detect_overadjustment("sepsis3", covariates) == []


def test_overadjustment_detector_keeps_explicit_tokens_aliases_and_suffixes():
    from easyicu.research_agent.icu_rules import detect_overadjustment

    offenders = detect_overadjustment(
        "sepsis3",
        [
            "crea_first",
            "baseline_creatinine",
            "mean_arterial_pressure",
            "sofa_renal",
            "age",
        ],
    )

    assert offenders == [
        "crea_first",
        "baseline_creatinine",
        "mean_arterial_pressure",
        "sofa_renal",
    ]


def test_exposure_row_itself_is_not_flagged(tmp_path: Path):
    # The exposure appears in its own coefficient table; that is correct, not
    # overadjustment, and must not be flagged.
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sex"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
        )
        == []
    )


def test_planner_operational_primary_source_is_not_overadjustment(tmp_path: Path):
    # Real E1 shape: ResearchContext preserves the clinical concept while the
    # typed model roster binds a row-aligned operational representation.  The
    # operational predictor is the exposure itself, not an adjusted-for SOFA
    # constituent merely because its column name contains ``sofa``.
    _write_coef_table(
        tmp_path,
        ["const", "sep3_sofa2_max", "age", "sex", "charlson_max"],
        name="adjusted_association_estimates.csv",
    )
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(primary_source="sep3_sofa2_max"),
            context=_ctx("sepsis3"),
            out_dir=tmp_path,
        )
        == []
    )


def test_operational_source_exemption_does_not_hide_real_constituent(
    tmp_path: Path,
):
    _write_coef_table(
        tmp_path,
        ["const", "sep3_sofa2_max", "age", "sofa_renal"],
        name="adjusted_association_estimates.csv",
    )
    findings = _primary_exposure_overadjustment_findings(
        step=_step(primary_source="sep3_sofa2_max"),
        context=_ctx("sepsis3"),
        out_dir=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].detail["offending_covariates"] == ["sofa_renal"]


def test_secondary_source_cannot_exempt_a_primary_constituent(tmp_path: Path):
    _write_coef_table(tmp_path, ["const", "sepsis3", "age", "sofa_max"])
    findings = _primary_exposure_overadjustment_findings(
        step=_step(primary_source="sofa_max", source_role="secondary"),
        context=_ctx("sepsis3"),
        out_dir=tmp_path,
    )
    assert len(findings) == 1
    assert findings[0].detail["offending_covariates"] == ["sofa_max"]


def test_no_flag_without_required_exposure(tmp_path: Path):
    # Question names no exposure -> nothing to enforce (never inferred).
    _write_coef_table(tmp_path, ["const", "age", "sofa_max"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx(None), out_dir=tmp_path
        )
        == []
    )


def test_no_flag_when_no_coefficient_table(tmp_path: Path):
    # No model output yet -> silent, not a guess.
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
        )
        == []
    )


def test_unresolvable_derived_exposure_emits_caution_not_error(tmp_path: Path):
    # NEWS is a callback score with an empty dependency closure: the
    # deterministic check is blind, so instead of silently passing, a non-gating
    # caution (warning) is emitted to prompt manual verification.
    _write_coef_table(tmp_path, ["const", "news", "age", "sex", "heart_rate"])
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("news"), out_dir=tmp_path
    )
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "warning"  # caution, never the gating "error"
    assert f.detail["kind"] == "overadjustment_caution"
    assert f.detail["exposure"] == "news"


def test_non_derived_exposure_emits_nothing(tmp_path: Path):
    # A raw lab exposure is not derived -> no caution, no error (silent).
    _write_coef_table(tmp_path, ["const", "lact", "age", "sex"])
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("lact"), out_dir=tmp_path
        )
        == []
    )


# ---------------------------------------------------------------------------
# read_adjustment_covariates — recover the adjustment set from analysis code
# when the run reports only a model-level OR summary (no coefficient table).
# This closes the auditor-blindness gap surfaced by the bench_run14 E1 run.
# General by construction: it recovers column *names*; whether any is an
# exposure constituent is left to the dictionary-driven detector.
# ---------------------------------------------------------------------------


def _write_code(out_dir: Path, body: str, *, name="analysis.py"):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(body, encoding="utf-8")


def test_coef_table_preferred_over_code(tmp_path: Path):
    # Ground truth wins: when a coefficient table exists it is used as-is and the
    # code is not consulted.
    _write_coef_table(tmp_path, ["const", "age", "sofa_max"])
    _write_code(tmp_path, "covariates = ['map_max', 'lact_max']\n")
    assert read_adjustment_covariates(tmp_path) == ["age", "sofa_max"]


def test_recover_covariates_from_intent_named_list(tmp_path: Path):
    # No coefficient table — only a model-level summary would exist. The
    # adjustment set is recovered from a covariate-intent list literal.
    _write_code(
        tmp_path,
        "covariate_cols = ['hr_max', 'map_max', 'lact_max']\n"
        "model = sm.Logit(y, X).fit()\n",
    )
    assert read_adjustment_covariates(tmp_path) == ["hr_max", "map_max", "lact_max"]


def test_recover_covariates_from_formula(tmp_path: Path):
    _write_code(
        tmp_path,
        "res = smf.logit('death ~ sepsis3 + age + C(sex) + map_max', df).fit()\n",
    )
    got = read_adjustment_covariates(tmp_path)
    assert got == ["sepsis3", "age", "sex", "map_max"]


def test_code_recovery_then_overadjustment_fires(tmp_path: Path):
    # End-to-end: summary-only run (no coef table) + a covariate list that
    # includes map (a Sepsis-3 constituent via SOFA) -> the in-run auditor now
    # flags it instead of staying blind.
    _write_code(
        tmp_path,
        "covariates = ['hr_max', 'map_max', 'resp_max', 'lact_max']\n",
    )
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
    )
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["offending_covariates"] == ["map_max"]


def test_prose_tilde_string_is_not_a_formula(tmp_path: Path):
    # A note containing "~" as "approximately" must NOT be parsed as a formula
    # (its RHS terms are not identifiers) -> nothing recovered, no junk tokens.
    _write_code(tmp_path, "note = 'adjusted OR ~1.11, identical (coding bug)'\n")
    assert read_adjustment_covariates(tmp_path) == []


def test_all_model_vars_list_does_not_leak_outcome(tmp_path: Path):
    # A list named for ALL model variables bundles the outcome with predictors;
    # it must NOT be treated as the adjustment set, or the outcome would leak in
    # and trip a spurious outcome-leakage error. Only predictor/covariate-side
    # names are recovered.
    _write_code(tmp_path, "model_vars = ['sepsis3', 'age', 'death']\n")
    assert read_adjustment_covariates(tmp_path) == []


# ---------------------------------------------------------------------------
# Negation-named exclusion lists must NOT be read as the adjustment set.
# Regression for the real E3 KDIGO missingness-audit false positive: that step
# fits no model but enumerates the renal KDIGO-component fields in a constant
# named RENAL_SOURCE_NOT_ADJUSTED (documenting the *exclusion*). The auditor
# substring-matched "adjust" inside "not_adjusted", read the excluded renal
# fields as the adjustment set, and failed a fully-correct audit — driving the
# replan loop to exhaustion. A name that *means* "not in the model" never
# intends the adjustment set.
# ---------------------------------------------------------------------------


def test_name_intends_covariates_rejects_negation():
    # Genuine adjustment-set names still intend covariates ...
    assert _name_intends_covariates("covariates")
    assert _name_intends_covariates("PRIMARY_ADJUSTMENT_SET")
    assert _name_intends_covariates("confounders")
    assert _name_intends_covariates("x_cols")
    # ... but exclusion/negation-named lists never do.
    assert not _name_intends_covariates("RENAL_SOURCE_NOT_ADJUSTED")
    assert not _name_intends_covariates("excluded_covariates")
    assert not _name_intends_covariates("covariates_excluded_for_overadjustment")
    assert not _name_intends_covariates("non_adjustment_fields")
    assert not _name_intends_covariates("unadjusted_model_columns")


def test_exclusion_named_list_not_read_as_adjustment_set(tmp_path: Path):
    # A step that declares BOTH a genuine adjustment set and an exclusion list of
    # exposure constituents (named to document the exclusion) must recover only
    # the genuine set — never the excluded constituents.
    _write_code(
        tmp_path,
        "PRIMARY_ADJUSTMENT_SET = ['age', 'sex', 'adm']\n"
        "RENAL_SOURCE_NOT_ADJUSTED = ['crea_first', 'urine24_first', 'sofa_max']\n",
    )
    assert read_adjustment_covariates(tmp_path) == ["age", "sex", "adm"]


def test_exclusion_named_constituent_does_not_false_positive(tmp_path: Path):
    # End-to-end mirror of the real bug with the well-supported sepsis3/sofa_max
    # constituent pair: sofa_max sits ONLY in an exclusion-named list, so the
    # overadjustment auditor must stay silent (before the fix it read sofa_max as
    # adjusted-for and raised a phantom error that fail-closed the run).
    _write_code(
        tmp_path,
        "covariates = ['age', 'sex']\n"
        "sofa_excluded_to_avoid_overadjustment = ['sofa_max']\n",
    )
    assert read_adjustment_covariates(tmp_path) == ["age", "sex"]
    assert (
        _primary_exposure_overadjustment_findings(
            step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
        )
        == []
    )


def test_genuine_constituent_still_fires_after_negation_guard(tmp_path: Path):
    # The fix is surgical: a constituent sitting in a genuine (non-negated)
    # adjustment list must STILL raise the overadjustment error — the negation
    # guard only spares exclusion-named lists, it never blinds the auditor.
    _write_code(tmp_path, "covariates_final = ['age', 'sofa_max']\n")
    findings = _primary_exposure_overadjustment_findings(
        step=_step(), context=_ctx("sepsis3"), out_dir=tmp_path
    )
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["offending_covariates"] == ["sofa_max"]


# ---------------------------------------------------------------------------
# Prevention layer: the replanner must also see the methodological principles
# (the planner already did via 94cf8db; the replanner revises the model spec,
# so dropping the guard there is exactly where overadjustment can re-enter).
# ---------------------------------------------------------------------------


def test_replanner_injects_methodological_principles(monkeypatch):
    from easyicu.research_agent.agents import core as A
    from easyicu.research_agent.providers import structured_retry as SR
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    # The overadjustment principle must actually be in the shared guide. The
    # guide renders each principle's `principle` text (not its rationale), so
    # match that wording rather than the word "overadjustment".
    assert "neither constituents nor downstream" in A._PRINCIPLES_GUIDE.lower()

    plan = AnalysisPlan(
        research_question="q", steps=[AnalysisStep(step_id="01", intent="x")]
    )
    captured = {}

    def _fake_retry(llm, messages, parser, **kwargs):
        captured["system"] = messages[0].content
        return plan

    # Sidestep the heavy ResearchContext rendering; we only assert the system
    # message the replanner builds, not the user prompt.
    monkeypatch.setattr(A, "scoped_planner_context", lambda ctx: ctx)
    monkeypatch.setattr(A, "_format_context", lambda ctx, **_kwargs: "CTX")
    monkeypatch.setattr(A, "planner_variable_catalog", lambda *_args: "CATALOG")
    monkeypatch.setattr(SR, "call_llm_with_structured_retry", _fake_retry)

    out = A.ReplannerAgent(llm=object()).run(
        context=object(), current_plan=plan, probe_summary={}, completed_step_records=[]
    )
    # The replanner returns a revised copy (revision bumped), so assert on
    # content, not identity. The point of the test is the captured system msg.
    assert out.research_question == "q"
    assert "neither constituents nor downstream" in captured["system"].lower()
