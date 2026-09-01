from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.reporting.administrative_authority import (
    ManuscriptAdministrativeAuthority,
    load_manuscript_administrative_authority,
    render_manuscript_administrative_sections,
)


def _authority() -> ManuscriptAdministrativeAuthority:
    return ManuscriptAdministrativeAuthority.issue(
        authority_id="submission-metadata-v1",
        verified_by="corresponding author",
        verified_at="2026-08-14T02:00:00Z",
        data_and_code_availability="Verified data statement.",
        funding="Verified funding statement.",
        ethics="Verified ethics statement.",
        conflicts_of_interest="Verified disclosure statement.",
        artifact_release="Verified artifact inventory statement.",
    )


def test_unverified_administrative_sections_never_invent_release_or_disclosure() -> None:
    rendered = render_manuscript_administrative_sections(None)

    assert "requires author verification" in rendered
    assert "released alongside" not in rendered
    assert "declare no conflicts" not in rendered


def test_administrative_authority_digest_detects_statement_drift() -> None:
    payload = _authority().model_dump(mode="json")
    payload["conflicts_of_interest"] = "No conflicts."

    with pytest.raises(ValidationError, match="digest mismatch"):
        ManuscriptAdministrativeAuthority.model_validate(payload)


def test_run_authority_loads_only_exact_digest_bound_payload(tmp_path) -> None:
    authority = _authority()
    path = tmp_path / "authorities" / "manuscript_administrative_authority.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(authority.model_dump(mode="json")),
        encoding="utf-8",
    )

    loaded = load_manuscript_administrative_authority(tmp_path)

    assert loaded == authority
    path.write_text(path.read_text().replace("Verified funding", "Changed funding"))
    assert load_manuscript_administrative_authority(tmp_path) is None
