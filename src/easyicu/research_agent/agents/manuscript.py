"""Human-supervised manuscript drafting agent."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ..providers.protocol import LLMClient
from ..reporting.administrative_authority import ManuscriptAdministrativeAuthority
from ..schema import (
    ClinicalSemanticsResolution,
    EvidenceRef,
    ManuscriptDraftPacket,
    ResearchContext,
)
from .reporting import WriterAgent


class ManuscriptAgent:
    """Draft-only manuscript agent that stays human-supervised for discussion."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        language: str = "en",
        nature_writing_enabled: bool = True,
        user_writing_advisory: str = "",
    ) -> None:
        self.llm = llm
        self.language = language
        self.nature_writing_enabled = bool(nature_writing_enabled)
        self.user_writing_advisory = str(user_writing_advisory or "")

    def build_packet(
        self,
        *,
        context: ResearchContext,
        semantics: ClinicalSemanticsResolution,
        evidence_refs: Sequence[EvidenceRef],
        findings: Sequence[str],
        caveats: Sequence[str],
    ) -> ManuscriptDraftPacket:
        return ManuscriptDraftPacket(
            title=context.research_question,
            abstract_focus=context.target_outcome,
            analysis_family=semantics.analysis_family,
            evidence_refs=list(evidence_refs),
            findings=list(findings),
            caveats=list(caveats),
        )

    def run(
        self,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str] = None,
        literature_digest: Optional[str] = None,
        reader_display_labels: Optional[Mapping[str, str]] = None,
        administrative_authority: ManuscriptAdministrativeAuthority | None = None,
    ) -> str:
        return WriterAgent(
            self.llm,
            language=self.language,
            nature_writing_enabled=self.nature_writing_enabled,
            user_writing_advisory=self.user_writing_advisory,
        ).run(
            context=context,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
            literature_digest=literature_digest,
            reader_display_labels=reader_display_labels,
            administrative_authority=administrative_authority,
        )

    def repair_existing(
        self,
        manuscript: str,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str] = None,
        literature_digest: Optional[str] = None,
        reader_display_labels: Optional[Mapping[str, str]] = None,
        administrative_authority: ManuscriptAdministrativeAuthority | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        return WriterAgent(
            self.llm,
            language=self.language,
            nature_writing_enabled=self.nature_writing_enabled,
            user_writing_advisory=self.user_writing_advisory,
        ).repair_existing(
            manuscript,
            context=context,
            evidence_ids=evidence_ids,
            evidence_digest=evidence_digest,
            literature_digest=literature_digest,
            reader_display_labels=reader_display_labels,
            administrative_authority=administrative_authority,
        )
