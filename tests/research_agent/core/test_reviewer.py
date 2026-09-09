"""Tests for the three-role simulated reviewer round (O15)."""

from __future__ import annotations

import json
import hashlib
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


class _EvRec:
    def __init__(
        self,
        evidence_id,
        *,
        produced_by_step=None,
        description=None,
        relative_path=None,
        kind="table",
        metadata=None,
        sha256=None,
    ):
        self.evidence_id = evidence_id
        self.produced_by_step = produced_by_step
        self.description = description
        self.relative_path = relative_path
        self.kind = kind
        self.metadata = metadata or {}
        self.sha256 = sha256


class _Finding:
    def __init__(self, validator, severity, message, detail=None):
        self.validator = validator
        self.severity = severity
        self.message = message
        self.detail = detail or {}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def _registered_record(run_dir, evidence_id, *, filename=None, content="{}", **kwargs):
    relative = f"evidence/{evidence_id}__{filename or evidence_id + '.json'}"
    path = run_dir / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return _EvRec(
        evidence_id, relative_path=relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **kwargs,
    )


@pytest.fixture
def bound_primary(ra, tmp_path):
    from easyicu.research_agent.reporting.reviewer import ReviewerPrimaryResultBinding

    # Sanitized native product/registration shape. No fit or patient data.
    product = "table:landmark_rcs_contrasts"
    step_id = "primary_adjusted_association"
    result = _registered_record(
        tmp_path, "table_step_artifact_7addebf5129174e4",
        filename="landmark_rcs_contrasts.csv", content="contrast,estimate\nA,1.2\n",
        produced_by_step=step_id, metadata={"diagnostic_only": False},
    )
    summary = _registered_record(
        tmp_path, "statistic_step_summary_96e49f5541b34e84",
        filename="step_summary.json", produced_by_step=step_id, kind="statistic",
        content=json.dumps({
            "status": "ok", "output_files": {product: "landmark_rcs_contrasts.csv"},
        }),
    )
    ledger = {
        "step_id": step_id, "status": "ok", "planned_analysis_role": "primary",
        "analysis_request": {"step": {
            "step_id": step_id, "planned_analysis_role": "primary",
            "expected_outputs": [product],
        }},
        "evidence_ids": [result.evidence_id, summary.evidence_id],
        "step_summary_evidence_id": summary.evidence_id,
    }
    return {
        "run_dir": tmp_path,
        "evidence_records": [result, summary],
        "per_step_records": [ledger],
        "primary_result_bindings": [ReviewerPrimaryResultBinding(
            product=product, evidence_id=result.evidence_id, sha256=result.sha256,
            produced_by_step=step_id, analysis_role="primary", claim_ceiling="reportable",
        )],
    }


def _effect_comments(report):
    return [c for critique in report.critiques for c in critique.comments if c.topic == "effect_estimate"]


def test_clean_run_recommends_accept(ra, bound_primary):
    for evidence_id in (
        "missingness", "multiple_testing_report", "reporting_checklist_strobe",
        "reproducibility_envelope", "preplan_literature_bundle",
    ):
        bound_primary["evidence_records"].append(_registered_record(
            bound_primary["run_dir"], evidence_id, kind="log",
        ))
    findings = [
        _Finding(
            "multiple_testing",
            "info",
            "Ran BH-FDR across 3 tests at alpha=0.050",
        ),
        _Finding(
            "reporting_checklist",
            "info",
            "STROBE coverage 80%",
            detail={"coverage": 0.8, "n_addressed": 18},
        ),
        _Finding("causal_audit", "info", "Labelled 1 effect(s)"),
    ]
    report = ra.run_reviewer_round(**bound_primary, findings=findings)
    # Every role accepts; no reject / major.
    assert report.aggregated_recommendation() == "accept"
    for c in report.critiques:
        assert c.recommendation() == "accept"


def test_missing_primary_estimate_triggers_statistician_major(ra):
    recs = [_EvRec("table_one")]
    report = ra.run_reviewer_round(evidence_records=recs, findings=[])
    stats = next(c for c in report.critiques if c.reviewer == "statistician")
    assert any(
        c.topic == "effect_estimate" and c.severity == "major" for c in stats.comments
    )


def test_primary_prose_is_not_authority_but_missingness_remains_detectable(ra):
    recs = [
        _EvRec(
            "statistic_step_summary_a029c2dd",
            produced_by_step="06_association_model",
            description="Step summary containing primary association odds_ratio.",
        ),
        _EvRec(
            "table_report_missingness_summary_1a19c00e",
            produced_by_step="02_missingness_audit",
            description="Table report_missingness_summary from the missingness profile.",
        ),
        _EvRec("multiple_testing_report"),
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=[])
    stats = next(c for c in report.critiques if c.reviewer == "statistician")
    assert any(c.topic == "effect_estimate" for c in stats.comments)
    assert not any(c.topic == "missingness" for c in stats.comments)


def test_prediction_filename_without_primary_authority_is_insufficient(ra):
    recs = [
        _EvRec(
            "table_step_artifact_d40d4c7a",
            produced_by_step="05_primary_discrimination",
            description="Table prediction_performance from the primary step.",
            relative_path=(
                "table_step_artifact_d40d4c7a__prediction_performance.csv"
            ),
        ),
        _EvRec("missingness"),
        _EvRec("reproducibility_envelope", kind="log"),
    ]

    report = ra.run_reviewer_round(evidence_records=recs, findings=[])
    stats = next(c for c in report.critiques if c.reviewer == "statistician")
    meth = next(c for c in report.critiques if c.reviewer == "methodologist")
    assert any(c.topic == "effect_estimate" for c in stats.comments)
    assert not any(c.topic == "reproducibility" for c in meth.comments)


@pytest.mark.parametrize("product_name", ["landmark_rcs_contrasts", "prediction_performance", "registered_scientific_result"])
def test_verified_product_identity_does_not_depend_on_reviewer_keywords(ra, bound_primary, product_name):
    binding = bound_primary["primary_result_bindings"][0]
    result, summary = bound_primary["evidence_records"]
    product = f"table:{product_name}"
    result = _registered_record(
        bound_primary["run_dir"], result.evidence_id,
        filename=f"{product_name}.csv", content="result\n1.2\n",
        produced_by_step=binding.produced_by_step,
    )
    summary = _registered_record(
        bound_primary["run_dir"], summary.evidence_id, filename="step_summary.json",
        kind="statistic", produced_by_step=binding.produced_by_step,
        content=json.dumps({"status": "ok", "output_files": {product: f"{product_name}.csv"}}),
    )
    bound_primary["evidence_records"] = [result, summary]
    bound_primary["per_step_records"][0]["analysis_request"]["step"]["expected_outputs"] = [product]
    bound_primary["primary_result_bindings"] = [replace(binding, product=product, sha256=result.sha256)]
    assert not _effect_comments(ra.run_reviewer_round(**bound_primary, findings=[]))


@pytest.mark.parametrize("failure", [
    "analysis_only", "secondary_product", "secondary_step", "role_drift",
    "diagnostic", "metadata_ceiling", "unregistered", "undeclared", "other_producer",
    "result_tampered", "summary_tampered", "later_failed_attempt", "missing_context",
    "self_claimed_binding", "binding_digest_drift", "malformed_ceiling", "runtime_unvalidated",
])
def test_primary_recognition_fails_closed(ra, bound_primary, failure):
    result, summary = bound_primary["evidence_records"]
    record = bound_primary["per_step_records"][0]
    binding = bound_primary["primary_result_bindings"][0]
    if failure == "analysis_only":
        binding = replace(binding, claim_ceiling="analysis_only")
    elif failure == "secondary_product":
        binding = replace(binding, analysis_role="sensitivity")
    elif failure == "secondary_step":
        record["planned_analysis_role"] = "secondary"
        record["analysis_request"]["step"]["planned_analysis_role"] = "secondary"
    elif failure == "role_drift":
        record["analysis_request"]["step"]["planned_analysis_role"] = "secondary"
    elif failure == "diagnostic":
        result.metadata["diagnostic_only"] = True
    elif failure == "metadata_ceiling":
        result.metadata["claim_ceiling"] = "analysis_only"
    elif failure == "malformed_ceiling":
        result.metadata["claim_ceiling"] = {"status": "reportable"}
    elif failure == "runtime_unvalidated":
        record["analysis_validated"] = False
    elif failure == "unregistered":
        record["evidence_ids"].remove(result.evidence_id)
    elif failure == "undeclared":
        record["analysis_request"]["step"]["expected_outputs"] = []
    elif failure == "other_producer":
        result.produced_by_step = "secondary_step"
    elif failure in {"result_tampered", "summary_tampered"}:
        target = result if failure == "result_tampered" else summary
        (bound_primary["run_dir"] / target.relative_path).write_text("{}")
    elif failure == "later_failed_attempt":
        bound_primary["per_step_records"].append({**record, "status": "failed"})
    elif failure == "missing_context":
        bound_primary["per_step_records"] = None
    elif failure == "self_claimed_binding":
        binding = dict(vars(binding))
    elif failure == "binding_digest_drift":
        binding = replace(binding, sha256="0" * 64)
    bound_primary["primary_result_bindings"] = [binding]
    comments = _effect_comments(ra.run_reviewer_round(**bound_primary, findings=[]))
    assert len(comments) == 1
    assert comments[0].severity == "major"
    assert "could not be verified" in comments[0].message


@pytest.mark.parametrize("evidence_id", ["preplan_literature_bundle", "literature_bundle"])
def test_registered_literature_bundle_is_recognized(ra, tmp_path, evidence_id):
    record = _registered_record(tmp_path, evidence_id, kind="log")
    report = ra.run_reviewer_round(evidence_records=[record], findings=[], run_dir=tmp_path)
    assert not any(c.topic == "literature" for q in report.critiques for c in q.comments)


@pytest.mark.parametrize("failure", ["metadata_keyword", "wrong_kind", "step_owned", "tampered"])
def test_literature_registration_cannot_be_spoofed_by_metadata(ra, tmp_path, failure):
    record = _registered_record(tmp_path, "preplan_literature_bundle", kind="log")
    if failure == "metadata_keyword":
        record.evidence_id = "unrelated_log"
        record.metadata["alias"] = "literature_bundle"
        record.description = "preplan_literature_bundle"
    elif failure == "wrong_kind":
        record.kind = "table"
    elif failure == "step_owned":
        record.produced_by_step = "coder"
    elif failure == "tampered":
        (tmp_path / record.relative_path).write_text("changed")
    report = ra.run_reviewer_round(evidence_records=[record], findings=[], run_dir=tmp_path)
    assert any(c.topic == "literature" for q in report.critiques for c in q.comments)


@pytest.mark.parametrize("validator,severity,detail", [
    ("scientific_maturity", "warning", {"paper_authorization_allowed": False}),
    ("primary_runtime", "error", {}),
    ("manuscript_numeric_auditor", "error", {}),
])
def test_unresolved_run_blocker_prevents_accept(ra, bound_primary, validator, severity, detail):
    findings = [
        _Finding(
            validator, severity, "Current authority remains blocked.", detail=detail,
        )
    ]

    report = ra.run_reviewer_round(**bound_primary, findings=findings)
    assert not _effect_comments(report)
    meth = next(c for c in report.critiques if c.reviewer == "methodologist")
    assert any(c.topic == "scientific_gate" for c in meth.comments)
    assert report.aggregated_recommendation() == "major_revision"


@pytest.mark.parametrize("representation", ["object", "mapping"])
def test_development_lineage_warning_preserves_nonpaper_identity(ra, bound_primary, representation):
    finding = _Finding(
        "development_runtime_lineage", "warning", "Development-only lineage.",
        {"paper_authority": False, "diagnostic_only": True},
    )
    if representation == "mapping":
        finding = vars(finding)
    report = ra.run_reviewer_round(**bound_primary, findings=[finding])
    comments = next(c for c in report.critiques if c.reviewer == "methodologist").comments
    assert not any(c.topic == "scientific_gate" for c in comments)
    scope = next(c for c in comments if c.topic == "publication_scope")
    assert scope.severity == "info"
    assert "no paper authority" in scope.message


@pytest.mark.parametrize("difference", [
    "error", "other_validator", "not_diagnostic", "analysis_invalid",
    "reportability_denied", "paper_authorization_denied", "coexisting_error",
])
def test_development_lineage_exception_does_not_hide_real_blocks(ra, bound_primary, difference):
    finding = _Finding(
        "development_runtime_lineage", "warning", "Development-only lineage.",
        {"paper_authority": False, "diagnostic_only": True},
    )
    if difference == "error":
        finding.severity = "error"
    elif difference == "other_validator":
        finding.validator = "scientific_maturity"
    elif difference == "not_diagnostic":
        finding.detail["diagnostic_only"] = False
    elif difference == "analysis_invalid":
        finding.detail["analysis_validated"] = False
    elif difference == "reportability_denied":
        finding.detail["reportability_allowed"] = False
    elif difference == "paper_authorization_denied":
        finding.detail["paper_authorization_allowed"] = False
    findings = [finding]
    if difference == "coexisting_error":
        findings.append(_Finding("manuscript_numeric_auditor", "error", "Current manuscript fails."))
    report = ra.run_reviewer_round(**bound_primary, findings=findings)
    assert any(c.topic == "scientific_gate" for q in report.critiques for c in q.comments)
    assert report.aggregated_recommendation() == "major_revision"


@pytest.fixture(scope="module")
def native_reviewer_replay(ra):
    """Opt-in read-only replay of registered metadata; never instantiate a live store.

    The store constructor can repair projections. Use its existing read-only
    snapshot loader instead, and expose only the consumer's records/aliases API.
    No cohort or raw patient rows are loaded; no fit, Provider or writes occur.
    """
    from easyicu.research_agent.authority.evidence_snapshot import load_current_evidence_snapshot
    from easyicu.research_agent.authority.runtime_artifacts import (
        current_step_records, load_run_artifact_authority, verified_run_evidence_path,
    )

    configured = os.environ.get("EASYICU_REVIEWER_REPLAY_RUN_DIR")
    if not configured:
        pytest.skip("set EASYICU_REVIEWER_REPLAY_RUN_DIR for registered native metadata replay")
    root = Path(configured)
    snapshot = load_current_evidence_snapshot(root)
    records = [ra.EvidenceRecord.model_validate(value) for value in snapshot.records]
    ledger = load_run_artifact_authority(root)["per_step_records"]
    primary = next(record for record in current_step_records(ledger) if record.get("planned_analysis_role") == "primary")
    receipt = primary["step_summary"]["scientific_runtime_receipt"]
    # Select only a digest-verified registered execution configuration matching
    # the producing receipt, never credentials, progress notes or raw files.
    for record in reversed(records):
        if not record.evidence_id.startswith("execution_runtime_revision_"):
            continue
        path = verified_run_evidence_path(root, record)
        if path is None:
            continue
        config = json.loads(path.read_text())["approved_config"]
        authority = config.get("current_case_scientific_runtime_authority") or {}
        projection = config.get("scientific_runtime_projection_sha256")
        if (
            authority.get("execution_contract_sha256") == receipt["execution_contract_sha256"]
            and projection == receipt["runtime_projection_sha256"]
        ):
            return root, records, snapshot.aliases, ledger, authority, projection
    pytest.fail("registered producing runtime configuration was not verified")


def _replay_arguments(native_reviewer_replay):
    root, records, aliases, ledger, authority, projection = deepcopy(native_reviewer_replay)
    return {
        "evidence_store": SimpleNamespace(root=root, records=lambda: records, aliases=lambda: aliases),
        "per_step_records": ledger,
        "current_case_scientific_runtime_authority": authority,
        "scientific_runtime_projection_sha256": projection,
    }


def test_native_metadata_replay_recognizes_primary_and_literature(ra, native_reviewer_replay):
    from easyicu.research_agent.reporting.reviewer import derive_reviewer_primary_result_bindings

    args = _replay_arguments(native_reviewer_replay)
    bindings = derive_reviewer_primary_result_bindings(**args)
    assert {binding.product for binding in bindings} == {
        "table:landmark_rcs_curve", "table:landmark_rcs_contrasts",
    }
    store = args["evidence_store"]
    report = ra.run_reviewer_round(
        evidence_records=store.records(), per_step_records=args["per_step_records"],
        primary_result_bindings=bindings, run_dir=store.root, findings=[],
    )
    assert not _effect_comments(report)  # primaryidentityrecognized
    assert not any(c.topic == "literature" for q in report.critiques for c in q.comments)
    # This is an identity assertion, never an independent scientific acceptance.
    assert "Simulated reviewer report" in report.to_markdown()


@pytest.mark.parametrize("failure", [
    "nonprimary", "failed", "analysis_only", "runtime_unvalidated", "projection_drift",
    "receipt_drift", "sidecar_missing", "sidecar_digest_drift", "result_digest_drift",
    "capability_drift", "missing_runtime_authority", "only_sensitivity_registered",
])
def test_native_metadata_replay_refuses_unverified_results(ra, native_reviewer_replay, failure):
    from easyicu.research_agent.authority.runtime_artifacts import current_step_records
    from easyicu.research_agent.reporting.reviewer import derive_reviewer_primary_result_bindings

    args = _replay_arguments(native_reviewer_replay)
    primary = next(record for record in current_step_records(args["per_step_records"]) if record.get("planned_analysis_role") == "primary")
    if failure == "nonprimary":
        primary["planned_analysis_role"] = "secondary"
        primary["analysis_request"]["step"]["planned_analysis_role"] = "secondary"
    elif failure == "failed":
        args["per_step_records"].append({**primary, "status": "failed"})
    elif failure == "analysis_only":
        primary["claim_ceiling"] = "analysis_only"
    elif failure == "runtime_unvalidated":
        primary["analysis_validated"] = False
    elif failure == "projection_drift":
        args["scientific_runtime_projection_sha256"] = "0" * 64
    elif failure == "receipt_drift":
        primary["step_summary"]["scientific_runtime_receipt"]["execution_contract_sha256"] = "0" * 64
    elif failure in {"sidecar_missing", "sidecar_digest_drift", "result_digest_drift", "only_sensitivity_registered"}:
        records = args["evidence_store"].records()
        target = next(record for record in records if record.produced_by_step == primary["step_id"] and (
            record.evidence_id.startswith("log_result_envelope_sidecar_")
            if failure.startswith("sidecar") else record.relative_path.endswith("__landmark_rcs_contrasts.csv")
        ))
        if failure == "only_sensitivity_registered":
            primary["evidence_ids"] = [eid for eid in primary["evidence_ids"] if not any(
                item.evidence_id == eid and item.relative_path.endswith(("__landmark_rcs_curve.csv", "__landmark_rcs_contrasts.csv"))
                for item in records
            )]
        elif failure == "sidecar_missing":
            primary["evidence_ids"].remove(target.evidence_id)
        else:
            target.sha256 = "0" * 64
    elif failure == "capability_drift":
        primary["analysis_request"]["step"]["scientific_capability"] = "association_general_v1"
    elif failure == "missing_runtime_authority":
        args["current_case_scientific_runtime_authority"] = None
    assert derive_reviewer_primary_result_bindings(**args) == []


def test_causal_error_triggers_clinician_reject(ra):
    recs = [_EvRec("primary_association"), _EvRec("causal_audit_report")]
    findings = [
        _Finding(
            "causal_audit",
            "error",
            "Causal language over strong pattern caus cited [causal_overclaimed]",
        )
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=findings)
    clin = next(c for c in report.critiques if c.reviewer == "clinician")
    assert clin.recommendation() == "reject"
    assert report.aggregated_recommendation() == "reject"


def test_low_checklist_coverage_triggers_methodologist_major(ra):
    recs = [_EvRec("primary_association"), _EvRec("reporting_checklist_strobe")]
    findings = [
        _Finding(
            "reporting_checklist",
            "info",
            "STROBE coverage 30%",
            detail={"coverage": 0.3, "n_addressed": 6},
        )
    ]
    report = ra.run_reviewer_round(evidence_records=recs, findings=findings)
    meth = next(c for c in report.critiques if c.reviewer == "methodologist")
    assert any(
        c.topic == "reporting_guideline" and c.severity == "major" for c in meth.comments
    )


def test_markdown_contains_per_role_header(ra):
    report = ra.run_reviewer_round(evidence_records=[], findings=[])
    md = report.to_markdown()
    assert "## Statistician" in md
    assert "## Clinician" in md
    assert "## Methodologist" in md


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _write_cohort(df, tmp_path):
    path = tmp_path / "cohort.parquet"
    df.to_parquet(path)
    return path


def test_pipeline_writes_reviewer_report_by_default(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reproducibility_envelope=True,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert (run_dir / "reviewer_report.md").exists()
    assert (run_dir / "reviewer_report.json").exists()
    manifest = json.loads(Path(result.manifest_path).read_text())
    ev_ids = {r["evidence_id"] for r in manifest["evidence"]}
    assert "reviewer_report" in ev_ids
    assert "reviewer_report_json" in ev_ids
    assert "reproducibility_envelope" in ev_ids
    findings = [f for f in manifest["findings"] if f["validator"] == "reviewer_round"]
    assert len(findings) == 1
    # Read the structured report and make sure three reviewers appear.
    payload = json.loads((run_dir / "reviewer_report.json").read_text())
    assert len(payload["critiques"]) == 3
    assert {c["reviewer"] for c in payload["critiques"]} == {
        "statistician",
        "clinician",
        "methodologist",
    }
    methodologist = next(
        critique
        for critique in payload["critiques"]
        if critique["reviewer"] == "methodologist"
    )
    assert not any(
        comment["topic"] == "reproducibility"
        for comment in methodologist["comments"]
    )


def test_pipeline_reviewer_can_be_disabled(ra, synthetic_cohort, tmp_path):
    cohort_path = _write_cohort(synthetic_cohort, tmp_path)
    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path / "out",
        llm=ra.MockLLMClient(),
        enable_reviewer_round=False,
    )
    result = pipeline.run(
        skill="association_analysis",
        cohort=cohort_path,
        database="miiv",
    )
    run_dir = Path(result.manifest_path).parent
    assert not (run_dir / "reviewer_report.md").exists()
