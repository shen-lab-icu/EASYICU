"""Paper-aware replication mode tests."""

from __future__ import annotations

import json
from pathlib import Path


def _mock_association_paper() -> str:
    return """Title: Admission SOFA-2 score and ICU mortality in a retrospective cohort

Methods
We performed a retrospective ICU cohort study of adult ICU patients.
The primary exposure was SOFA-2 score at admission.
The primary outcome was ICU mortality.
We used multivariable logistic regression adjusted for age and sex.

Results
Table 1 summarised baseline characteristics.
Higher SOFA-2 was associated with ICU mortality (OR 1.18, 95% CI 1.07-1.31, p=0.001).
The cohort included N=800 patients.
Figure 1 shows the primary association.
"""


def test_parse_paper_profile_extracts_supported_design_and_claims(ra):
    profile = ra.parse_paper_profile(_mock_association_paper())
    assert profile.paper_type == "association"
    assert profile.target_outcome == "ICU mortality"
    assert "sofa" in (profile.primary_exposure or "").lower()
    assert profile.primary_analysis_method
    assert profile.key_claims
    assert any(claim.metric == "OR" for claim in profile.key_claims)


def test_build_replication_spec_flags_unsupported_design(ra):
    paper = """Title: ICU imaging biomarker model

Methods
We trained a deep model on bedside imaging and clinical notes.
The primary outcome was ICU mortality.
"""
    profile = ra.parse_paper_profile(paper)
    spec, deviation = ra.build_paper_replication_spec(profile)
    assert profile.paper_type == "unsupported_or_underspecified"
    assert deviation.supported is False
    assert spec.unmappable_items or deviation.items


def test_pipeline_reproduce_paper_writes_replication_artefacts(
    ra, synthetic_cohort, tmp_path: Path
):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.reproduce_paper(
        paper=_mock_association_paper(),
        cohort=synthetic_cohort,
        database="synthetic",
        mode="replication",
    )

    run_dir = Path(result.workdir)
    for name in (
        "paper_profile.json",
        "replication_spec.json",
        "paper_claim_ledger.csv",
        "replication_comparison.csv",
        "replication_report.md",
        "deviation_report.md",
    ):
        assert (run_dir / name).exists(), name

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    readiness = manifest["readiness"]
    assert readiness["design_reproduced"] is True
    assert readiness["paper_claims_parsed"] is True
    assert readiness["result_alignment_audited"] is True
    assert readiness["replication_report_ready"] is True
    assert readiness["showcase_manuscript_ready"] is False
    assert not (run_dir / "manuscript_ready.md").exists()
    run_status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert run_status["status"] == "replication_ready"

    comparison = pipeline.compare_with_paper(
        paper=_mock_association_paper(),
        result=result,
    )
    assert comparison["rows"]
    assert comparison["rows"][0]["alignment_status"] in {
        "aligned",
        "directionally_aligned",
        "not_comparable",
        "not_aligned",
    }


def test_manuscript_mode_does_not_override_an_unreportable_scientific_capability(
    ra, synthetic_cohort, tmp_path: Path
):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.reproduce_paper(
        paper=_mock_association_paper(),
        cohort=synthetic_cohort,
        database="synthetic",
        mode="manuscript",
    )

    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    readiness = manifest["readiness"]
    assert readiness["scientific_capability_claim_ceiling_allows_reportable"] is False
    assert readiness["showcase_manuscript_ready"] is False
    assert not (run_dir / "manuscript_ready.md").exists()


def test_pipeline_reproduce_paper_fail_closed_for_unsupported_source(
    ra, synthetic_cohort, tmp_path: Path
):
    pipeline = ra.ResearchAgentPipeline(workdir=tmp_path, llm=ra.MockLLMClient())
    result = pipeline.reproduce_paper(
        paper="""Title: Imaging ICU outcome model

Methods
We used imaging, waveform data, and clinical notes to train a deep model.
The primary outcome was ICU mortality.
""",
        cohort=synthetic_cohort,
        database="synthetic",
        mode="manuscript",
    )

    run_dir = Path(result.workdir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    readiness = manifest["readiness"]
    assert readiness["design_reproduced"] is False
    assert readiness["showcase_manuscript_ready"] is False
    assert not (run_dir / "manuscript_ready.md").exists()
