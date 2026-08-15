"""Host-owned authority for manuscript administrative statements.

Scientific evidence cannot establish author conflicts, funding, ethics approval,
or what files will actually be released with a submission.  This small contract
keeps those facts outside the Writer model and makes the unverified state
explicit rather than filling it with plausible-sounding boilerplate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


ADMINISTRATIVE_AUTHORITY_SCHEMA_VERSION = (
    "easyicu.manuscript_administrative_authority/1"
)


def _payload_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ManuscriptAdministrativeAuthority(BaseModel):
    """Exact author/repository statements approved outside the LLM pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[ADMINISTRATIVE_AUTHORITY_SCHEMA_VERSION] = (
        ADMINISTRATIVE_AUTHORITY_SCHEMA_VERSION
    )
    authority_id: str = Field(min_length=1, max_length=160)
    verified_by: str = Field(min_length=1, max_length=240)
    verified_at: str = Field(min_length=1, max_length=80)
    data_and_code_availability: str = Field(min_length=1, max_length=4000)
    funding: str = Field(min_length=1, max_length=2000)
    ethics: str = Field(min_length=1, max_length=2000)
    conflicts_of_interest: str = Field(min_length=1, max_length=2000)
    artifact_release: str = Field(min_length=1, max_length=3000)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def issue(
        cls,
        *,
        authority_id: str,
        verified_by: str,
        verified_at: str,
        data_and_code_availability: str,
        funding: str,
        ethics: str,
        conflicts_of_interest: str,
        artifact_release: str,
    ) -> "ManuscriptAdministrativeAuthority":
        payload: dict[str, object] = {
            "schema_version": ADMINISTRATIVE_AUTHORITY_SCHEMA_VERSION,
            "authority_id": authority_id,
            "verified_by": verified_by,
            "verified_at": verified_at,
            "data_and_code_availability": data_and_code_availability,
            "funding": funding,
            "ethics": ethics,
            "conflicts_of_interest": conflicts_of_interest,
            "artifact_release": artifact_release,
        }
        return cls(**payload, authority_sha256=_payload_sha256(payload))

    @model_validator(mode="after")
    def _verify_digest(self) -> "ManuscriptAdministrativeAuthority":
        payload = self.model_dump(exclude={"authority_sha256"}, mode="json")
        if _payload_sha256(payload) != self.authority_sha256:
            raise ValueError("manuscript administrative authority digest mismatch")
        return self


def load_manuscript_administrative_authority(
    run_dir: Path,
) -> Optional[ManuscriptAdministrativeAuthority]:
    """Load the optional host-issued run authority; malformed data fails closed."""

    path = run_dir / "authorities" / "manuscript_administrative_authority.json"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ManuscriptAdministrativeAuthority.model_validate(payload)
    except Exception:
        return None


def render_manuscript_administrative_sections(
    authority: Optional[ManuscriptAdministrativeAuthority],
) -> str:
    """Render exact host statements or explicit author-action placeholders."""

    if authority is None:
        return "\n\n".join(
            (
                "## Data and code availability\n"
                "Data and code availability require verification against the "
                "source-data licence and the final release inventory before submission.",
                "## Funding\n"
                "Funding information requires author verification before submission.",
                "## Ethics approval\n"
                "Ethics and data-access statements require author verification before "
                "submission.",
                "## Conflicts of interest\n"
                "Conflict-of-interest information requires author verification before "
                "submission.",
                "## Supplementary artifact release\n"
                "The supplementary artifact release inventory requires repository-owner "
                "verification before submission.",
            )
        )
    return "\n\n".join(
        (
            "## Data and code availability\n" + authority.data_and_code_availability,
            "## Funding\n" + authority.funding,
            "## Ethics approval\n" + authority.ethics,
            "## Conflicts of interest\n" + authority.conflicts_of_interest,
            "## Supplementary artifact release\n" + authority.artifact_release,
        )
    )


__all__ = [
    "ADMINISTRATIVE_AUTHORITY_SCHEMA_VERSION",
    "ManuscriptAdministrativeAuthority",
    "load_manuscript_administrative_authority",
    "render_manuscript_administrative_sections",
]
