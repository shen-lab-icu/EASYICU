"""Reader confidence is checked against inputs, never another renderer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.authority.manuscript_claim_policy import (
    missing_scientific_claims_in_results,
)
from easyicu.research_agent.authority.scientific_claim_registry import (
    ScientificClaimRegistryError,
    load_registered_scientific_claims,
    validate_scientific_claim_registration,
)
from easyicu.research_agent.authority.scientific_claims import (
    ScientificClaim,
    ScientificClaimDraft,
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.schema import EvidenceRecord

FIXTURE = Path(__file__).parent / "fixtures" / "scientific_claim_v2_90.json"


def _summary(confidence: float | None) -> dict:
    summary = json.loads(FIXTURE.read_text())["summary"]
    estimates = summary["descriptive_estimates"]
    if confidence is None:
        estimates["dependence"] = None
        estimates["risk_difference"] = None
        for item in estimates["outcome_absolute_risks"]:
            for field in (
                "standard_error_pct",
                "ci_low_pct",
                "ci_high_pct",
                "confidence_level",
                "cluster_count",
            ):
                item[field] = None
            item.update(
                interval_method="none_counts_only", covariance="none_counts_only"
            )
    else:
        z = NormalDist().inv_cdf((1 + confidence) / 2)
        for item in [
            *estimates["outcome_absolute_risks"],
            estimates["risk_difference"],
        ]:
            item["confidence_level"] = confidence
            item["ci_low_pct"] = item["estimate_pct"] - z * item["standard_error_pct"]
            item["ci_high_pct"] = item["estimate_pct"] + z * item["standard_error_pct"]
    return summary


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.99])
def test_registered_intervals_keep_the_input_level_in_every_claim_location(
    tmp_path: Path,
    confidence: float,
) -> None:
    summary = _summary(confidence)
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary))
    store = EvidenceStore(tmp_path, enforcement_mode="strict")
    record = store.register_file(
        kind="statistic",
        description="synthetic interval review",
        source_path=source,
        evidence_id="synthetic_summary",
        produced_by_step="02_describe",
        generation_mode="deterministic_standard",
    )
    store.register_step_summary_numerics(
        step_id="02_describe",
        evidence_id=record.evidence_id,
        summary=summary,
    )
    claims = store.scientific_claims()
    inputs = [
        *summary["descriptive_estimates"]["outcome_absolute_risks"],
        summary["descriptive_estimates"]["risk_difference"],
    ]
    for claim, original in zip(claims, inputs, strict=True):
        assert claim.schema_version == "easyicu.scientific_claim/3"
        assert claim.confidence_level == confidence
        assert claim.interval_method == original["interval_method"]
        assert claim.point_estimate == original["estimate_pct"]
        assert claim.interval_lower == pytest.approx(original["ci_low_pct"])
        assert claim.interval_upper == pytest.approx(original["ci_high_pct"])
        expected_scale = (
            "percent"
            if claim.claim_type == "descriptive_absolute_risk"
            else "percentage_points"
        )
        assert claim.effect_scale == expected_scale
        assert f"{confidence * 100:g}% CI" in claim.render_text()
        assert f"{confidence * 100:g}% CI" in claim.render_reader_text()
    tokens = "\n\n".join(claim.placeholder for claim in claims)
    scaffold = (
        f"## Abstract\n{tokens}\n## Results\n{tokens}\n## Figure legends\n{tokens}\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    bound = store.bind_manuscript(filtered)
    assert removed == []
    assert bound.count(f"{confidence * 100:g}% CI") == 9
    if confidence != 0.95:
        assert "95% CI" not in bound
    assert missing_scientific_claims_in_results(bound, claims=claims) == ()
    corrupted = bound.replace(f"{confidence * 100:g}% CI", "80% CI")
    assert len(missing_scientific_claims_in_results(corrupted, claims=claims)) == 3


def test_counts_only_claims_do_not_acquire_interval_authority() -> None:
    for draft in derive_scientific_claim_drafts(_summary(None)):
        claim = ScientificClaim(
            **draft.model_dump(), step_id="02_describe", evidence_id="summary"
        )
        assert claim.confidence_level is None
        assert claim.interval_lower is None
        assert "% CI" not in claim.render_reader_text()
        assert "counts only, no confidence interval" in claim.render_reader_text()


@pytest.mark.parametrize(
    "updates",
    [
        {"confidence_level": None},
        {"confidence_level": 0.5},
        {"confidence_level": float("nan")},
        {"confidence_level": float("inf")},
        {"interval_method": None},
        {"effect_scale": "percentage_points"},
        {"interval_lower": None},
        {"interval_method": "linear_probability_wald"},
    ],
)
def test_structured_interval_refuses_partial_or_contradictory_semantics(
    updates,
) -> None:
    payload = derive_scientific_claim_drafts(_summary(0.90))[0].model_dump()
    with pytest.raises(ValueError):
        ScientificClaimDraft.model_validate({**payload, **updates})


def _legacy_record(tmp_path: Path) -> tuple[dict, EvidenceRecord]:
    fixture = json.loads(FIXTURE.read_text())
    data = json.dumps(fixture["summary"]).encode()
    (tmp_path / "summary.json").write_bytes(data)
    record = EvidenceRecord(
        evidence_id="synthetic_summary",
        kind="statistic",
        description="legacy",
        relative_path="summary.json",
        sha256=hashlib.sha256(data).hexdigest(),
        produced_by_step="02_describe",
        generation_mode="deterministic_standard",
        metadata={"scientific_claims": fixture["claims"]},
    )
    return fixture["summary"], record


def test_legacy_source_recovers_confidence_without_rewriting_sealed_metadata(
    tmp_path: Path,
) -> None:
    summary, record = _legacy_record(tmp_path)
    before = copy.deepcopy(record.model_dump())
    claims = load_registered_scientific_claims(root=tmp_path, records=[record])
    assert all(claim.confidence_level == 0.90 for claim in claims)
    assert all("90% CI" in claim.render_reader_text() for claim in claims)
    registration = validate_scientific_claim_registration(
        root=tmp_path,
        record=record,
        step_id="02_describe",
        summary=summary,
        drafts=derive_scientific_claim_drafts(summary),
    )
    assert not registration.attach_metadata
    assert record.model_dump() == before


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_registration_cannot_borrow_a_summary_for_different_interval_drafts(
    tmp_path: Path,
    legacy_metadata: bool,
) -> None:
    summary, record = _legacy_record(tmp_path)
    if not legacy_metadata:
        record.metadata = {}
    with pytest.raises(ScientificClaimRegistryError, match="drafts differ"):
        validate_scientific_claim_registration(
            root=tmp_path,
            record=record,
            step_id="02_describe",
            summary=summary,
            drafts=derive_scientific_claim_drafts(_summary(0.99)),
        )


@pytest.mark.parametrize("tamper", ["metadata", "source", "missing_confidence"])
def test_legacy_replay_does_not_infer_or_bypass_source_authority(
    tmp_path: Path, tamper: str
) -> None:
    summary, record = _legacy_record(tmp_path)
    if tamper == "metadata":
        record.metadata["scientific_claims"][0]["estimand"] = "changed claim"
    elif tamper == "source":
        (tmp_path / record.relative_path).write_text("{}")
    else:
        del summary["descriptive_estimates"]["outcome_absolute_risks"][0][
            "confidence_level"
        ]
        data = json.dumps(summary).encode()
        (tmp_path / record.relative_path).write_bytes(data)
        record.sha256 = hashlib.sha256(data).hexdigest()
    with pytest.raises(ScientificClaimRegistryError):
        load_registered_scientific_claims(root=tmp_path, records=[record])


def test_legacy_interval_without_registered_source_cannot_render_as_95_percent() -> (
    None
):
    raw = json.loads(FIXTURE.read_text())["claims"][0]
    claim = ScientificClaim.model_validate(raw)
    assert claim.model_dump(mode="json") == raw
    with pytest.raises(ValueError, match="structured confidence authority"):
        claim.render_reader_text()
