from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.scientific_claim_registry import (
    ScientificClaimRegistryError,
    load_registered_scientific_claims,
    validate_scientific_claim_registration,
)
from easyicu.research_agent.authority.scientific_claims import (
    derive_scientific_claim_drafts,
)
from easyicu.research_agent.schema import EvidenceRecord


def _summary() -> dict:
    return {
        "interpretation_class": "adjusted_association",
        "exposure": "Lactate",
        "outcome": "hospital mortality",
        "effect_scale": "odds_ratio",
        "primary_estimate_interval": [1.1, 1.8],
        "analysis_set": "primary_cohort",
        "analysis_role": "primary",
        "adjustment_covariates": ["age", "sex"],
    }


def _record(tmp_path: Path, summary: dict) -> EvidenceRecord:
    path = tmp_path / "summary.json"
    payload = json.dumps(summary).encode("utf-8")
    path.write_bytes(payload)
    return EvidenceRecord(
        evidence_id="04_association_summary",
        kind="statistic",
        description="summary",
        relative_path=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
        produced_by_step="04_association",
        generation_mode="deterministic_standard",
    )


def test_registry_validates_bytes_and_returns_immutable_registration(
    tmp_path: Path,
) -> None:
    summary = _summary()
    record = _record(tmp_path, summary)

    registration = validate_scientific_claim_registration(
        root=tmp_path,
        record=record,
        step_id="04_association",
        summary=summary,
        drafts=derive_scientific_claim_drafts(summary),
    )

    assert registration.attach_metadata is True
    assert tuple(claim.claim_ref for claim in registration.claims) == (
        "04_association.adjusted_association",
    )


def test_registry_rejects_digest_drift(tmp_path: Path) -> None:
    summary = _summary()
    record = _record(tmp_path, summary)
    (tmp_path / record.relative_path).write_text("{}", encoding="utf-8")

    with pytest.raises(ScientificClaimRegistryError, match="digest drifted"):
        validate_scientific_claim_registration(
            root=tmp_path,
            record=record,
            step_id="04_association",
            summary=summary,
            drafts=derive_scientific_claim_drafts(summary),
        )


def test_registry_rederives_claims_instead_of_trusting_metadata(
    tmp_path: Path,
) -> None:
    summary = _summary()
    record = _record(tmp_path, summary)
    registration = validate_scientific_claim_registration(
        root=tmp_path,
        record=record,
        step_id="04_association",
        summary=summary,
        drafts=derive_scientific_claim_drafts(summary),
    )
    record.metadata = {
        "scientific_claims": [
            claim.model_dump(mode="json") for claim in registration.claims
        ]
    }

    claims = load_registered_scientific_claims(root=tmp_path, records=[record])

    assert [claim.exposure for claim in claims] == ["Lactate"]
    record.metadata["scientific_claims"][0]["exposure"] = "SOFA"
    with pytest.raises(
        ScientificClaimRegistryError,
        match="differs from host derivation",
    ):
        load_registered_scientific_claims(root=tmp_path, records=[record])
