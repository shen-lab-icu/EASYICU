"""AnalyzerAgent and WriterAgent (interpretation + manuscript)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from ..providers.protocol import LLMClient, LLMMessage
from ..providers.factory import authorized_complete
from ..research_context.prompt_scope import (
    scoped_coder_context,
    scoped_reporting_context,
)
from ..authority.provider_budget import (
    StepProviderCallBudget,
    complete_with_provider_budget,
)
from ..schema import (
    AnalysisStep,
    ResearchContext,
)
from ..reporting.manuscript_sections import render_manuscript_sections

from ._support import _NATURE_WRITING_GUIDE, _SYSTEM_GUIDE, _WRITER_GUIDE, _format_context, _strip_code_fence
from .coder import _coder_prompt_payload_bytes

# ---------------------------------------------------------------------------
# Analyzer (interpretation)
# ---------------------------------------------------------------------------


_ANALYZER_PROMPT_BYTE_LIMIT = 48_000
_WRITER_PROMPT_BYTE_LIMIT = 64_000


class ReportingPromptBudgetError(RuntimeError):
    """A lossless Analyzer/Writer request exceeds its transport envelope."""

    def __init__(self, *, role: str, actual_bytes: int, limit_bytes: int) -> None:
        self.role = str(role)
        self.actual_bytes = int(actual_bytes)
        self.limit_bytes = int(limit_bytes)
        super().__init__(
            f"{self.role} prompt transport budget exceeded: "
            f"{self.actual_bytes} > {self.limit_bytes} bytes. "
            "No evidence digest or binding scientific coordinate was truncated; "
            "reduce the role-scoped projection or split the evidence digest."
        )


def _enforce_reporting_prompt_budget(
    messages: Sequence[LLMMessage],
    *,
    role: str,
    limit_bytes: int,
) -> None:
    actual_bytes = _coder_prompt_payload_bytes(messages)
    if actual_bytes > int(limit_bytes):
        raise ReportingPromptBudgetError(
            role=role,
            actual_bytes=actual_bytes,
            limit_bytes=limit_bytes,
        )


class AnalyzerAgent:
    """Turns step outputs into a short, evidence-grounded interpretation."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        *,
        context: ResearchContext,
        step: AnalysisStep,
        step_summary: Dict[str, Any],
        evidence_ids: Sequence[str],
        provider_budget: Optional[StepProviderCallBudget] = None,
    ) -> str:
        from ..research_context.outbound import project_outbound_step_summary

        safe_step_summary = project_outbound_step_summary(step_summary)
        reporting_context = scoped_coder_context(
            context,
            step,
            max_variables=20,
        )
        messages = [
            LLMMessage(role="system", content=_SYSTEM_GUIDE),
            LLMMessage(
                role="user",
                content=(
                    f"INTERPRET the results of step {step.step_id}.\n"
                    f"Step intent: {step.intent}\n"
                    "Numeric summary (host-projected, machine-readable): "
                    f"{json.dumps(safe_step_summary, default=str)}\n"
                    f"Evidence ids you may cite verbatim: {list(evidence_ids)}\n\n"
                    "Constraints:\n"
                    "- Cite at least one evidence_id for every numeric claim, "
                    "in the form {evidence:<id>}.\n"
                    "- Do not introduce numbers that are not in the summary.\n"
                    "- 4 sentences max. No clinical recommendations.\n\n"
                    "RESEARCH CONTEXT:\n"
                    + _format_context(
                        reporting_context,
                        include_method_constraints=False,
                    )
                ),
            ),
        ]
        _enforce_reporting_prompt_budget(
            messages,
            role="Analyzer",
            limit_bytes=_ANALYZER_PROMPT_BYTE_LIMIT,
        )
        return complete_with_provider_budget(
            budget=provider_budget,
            category="analyzer",
            call=lambda: authorized_complete(
                self.llm, messages, max_tokens=512, temperature=0.2
            ),
        ).strip()


# ---------------------------------------------------------------------------
# Writer (manuscript scaffolder)
# ---------------------------------------------------------------------------


class WriterAgent:
    """Produces a full manuscript by writing each section in a separate
    LLM call, then concatenating. This avoids the "lazy middle"
    problem where small models truncate Introduction / Discussion.

    Each section call gets:
    - a role-scoped research context (study coordinates, not every source column),
    - the machine evidence digest (numbers to cite),
    - the list of available evidence ids,
    - a section-specific instruction with word-count target.

    The downstream causal-audit + critic loop reject drafts that use
    causal language for associations or cite non-existent evidence ids.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        language: str = "en",
        nature_writing_enabled: bool = True,
        user_writing_advisory: str = "",
    ) -> None:
        self.llm = llm
        lang = (language or "en").lower()
        self.language = "zh" if lang.startswith(("zh", "cn", "chinese")) else "en"
        self.nature_writing_enabled = bool(nature_writing_enabled)
        self.user_writing_advisory = str(user_writing_advisory or "")

    def _call_section(
        self,
        *,
        section_name: str,
        instruction: str,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str],
        literature_digest: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> str:
        lang_inst = _writer_language_instruction(self.language)
        evidence_list = (
            ", ".join(str(eid) for eid in evidence_ids) if evidence_ids else "(none)"
        )
        reporting_context = scoped_reporting_context(context)
        messages = [
            LLMMessage(
                role="system",
                content=(
                    _SYSTEM_GUIDE
                    + _WRITER_GUIDE
                    + (
                        "\n\n" + _NATURE_WRITING_GUIDE
                        if self.nature_writing_enabled
                        else ""
                    )
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"Write ONLY the **{section_name}** section of an ICU research "
                    "manuscript in markdown. Do NOT write any other section.\n\n"
                    f"{instruction}\n\n"
                    f"{lang_inst}\n\n"
                    + (
                        self.user_writing_advisory + "\n\n"
                        if self.user_writing_advisory
                        else ""
                    )
                    + "CITATION RULE:\n"
                    "- `{evidence:<id>}` is an inline citation (like a footnote number).\n"
                    "- Write the actual number in prose, then cite: "
                    "`mortality was 12% {evidence:outcome_rate}`.\n"
                    "- Use exactly single braces: `{evidence:<id>}`, not "
                    "`{{evidence:<id>}}`.\n"
                    "- Every current-study empirical sentence about cohort composition, "
                    "exposure prevalence, outcome frequency, model estimates, "
                    "sensitivity/robustness, missingness, or data quality must include "
                    "at least one `{evidence:<id>}` citation.\n"
                    "- Keywords, Data/code availability, Funding, and Conflicts of "
                    "interest are manuscript metadata and do not need evidence citations.\n"
                    "- NEVER use a placeholder as a noun. If a number is unavailable, omit the sentence.\n"
                    f"- Only use ids from this list: {evidence_list}\n\n"
                    "NUMERIC AUTHORITY RULE:\n"
                    "- Copy a current-study number only when that exact literal value "
                    "appears in the MACHINE EVIDENCE DIGEST, and use the exact "
                    "`{evidence:<id>}` citation shown for its owning fact.\n"
                    "- Do not calculate, infer, transform, round, or reconstruct a new "
                    "count, rate, difference, interval, or other numeric value from "
                    "values in the digest or RESEARCH CONTEXT.\n"
                    "- RESEARCH CONTEXT supplies study semantics only; it never "
                    "authorizes a numeric claim. A derived value is usable only when "
                    "the host has registered it explicitly in the evidence digest.\n"
                    "- If an exact value-and-owner pair is absent, omit the numeric "
                    "sentence rather than estimating it.\n\n"
                    "SCIENTIFIC CLAIM RULE:\n"
                    "- The machine digest may contain a `host-authorized scientific "
                    "claims` block. For any current-study direction, comparison, or "
                    "qualitative interpretation covered by that block, output the exact "
                    "`{claim:<step>.<claim>}` token as the complete standalone sentence.\n"
                    "- Do not paraphrase a host claim and do not replace `{claim:...}` "
                    "with `{evidence:...}`. Evidence citations authorize numeric facts; "
                    "they do not authorize independently worded scientific conclusions.\n"
                    "- A claim token cannot be attached to a heading, label, or other "
                    "prose. If no exact host claim applies, omit the qualitative "
                    "assertion.\n\n"
                    "LITERATURE RULE:\n"
                    "- Cite prior work only with an exact `[@key]` from the "
                    "run-bound literature digest below.\n"
                    "- Claims about prior studies or plausible clinical mechanisms "
                    "require an exact `[@key]`; a current-run evidence id cannot "
                    "substitute for literature support. A sentence comparing the "
                    "current result with prior work needs both citation types.\n"
                    "- Statements about mechanisms, strengths, or limitations must "
                    "use the exact applicable source: `[@key]` for prior knowledge "
                    "and `{evidence:<id>}` for a current-run empirical fact. Omit "
                    "the statement when neither supplied source supports it.\n"
                    "- If the digest contains a `direct_comparator`, Introduction and "
                    "Discussion must each cite at least one such key. Methods must cite "
                    "at least one `method:<layer>` key when one is available.\n"
                    "- When the digest contains `Run-bound typed methodology "
                    "applications`, Methods must cite at least one exact key from that "
                    "block next to the specific design or reporting choice it governs.\n"
                    "- `{evidence:literature_prisma}` supports the search process, "
                    "not what an individual paper found.\n"
                    "- Never invent an author, paper, comparison, mechanism, or "
                    "citation key; omit unsupported literature claims.\n\n"
                    "OUTPUT DISCIPLINE:\n"
                    "- Output ONLY finished, publishable manuscript prose. Do NOT include "
                    "your reasoning, planning, working notes, or meta-commentary about the "
                    "evidence digest — e.g. no lines such as 'First, extract ...', 'Let me "
                    "...', 'Actually, X says ...', a trailing 'no number?', or bullet lists "
                    "that restate the digest.\n"
                    "- If you cannot support a sentence from the listed evidence, omit it "
                    "silently; do not narrate the gap.\n\n"
                    "LANGUAGE POLICY:\n"
                    "- Use ONLY associational phrasing. Forbidden: 'caused by', 'causal', "
                    "'attributable to', 'effect of', 'due to', 'leads to', 'drives'.\n"
                    "- Allowed: 'was associated with', 'correlated with', 'observed alongside', "
                    "'consistent with', 'may reflect'.\n\n"
                    "SECTION-SPECIFIC LENGTH TARGET:\n"
                    f"- {section_name}: follow the requested length and paragraph structure exactly.\n\n"
                    "MACHINE EVIDENCE DIGEST:\n"
                    + (evidence_digest or "(none)")
                    + "\n\nRUN-BOUND LITERATURE DIGEST:\n"
                    + (literature_digest or "(none)")
                    + "\n\nRESEARCH CONTEXT:\n"
                    + _format_context(
                        reporting_context,
                        include_method_constraints=False,
                    )
                ),
            ),
        ]
        _enforce_reporting_prompt_budget(
            messages,
            role="Writer",
            limit_bytes=_WRITER_PROMPT_BYTE_LIMIT,
        )
        raw = authorized_complete(
            self.llm, messages, max_tokens=max_tokens, temperature=0.3
        ).strip()
        return _strip_code_fence(raw)

    def run(
        self,
        *,
        context: ResearchContext,
        evidence_ids: Sequence[str],
        evidence_digest: Optional[str] = None,
        literature_digest: Optional[str] = None,
    ) -> str:
        return render_manuscript_sections(
            call_section=self._call_section,
            common={
                "context": context,
                "evidence_ids": evidence_ids,
                "evidence_digest": evidence_digest,
                "literature_digest": literature_digest,
            },
        )


def _writer_language_instruction(language: str) -> str:
    if language == "zh":
        return (
            "OUTPUT LANGUAGE: zh / Simplified Chinese. Keep section headings "
            "as markdown headings. Preserve every `{evidence:<id>}` placeholder "
            "and `{claim:<step>.<claim>}` token exactly as ASCII; do not translate "
            "evidence ids, claim refs, filenames, variable "
            "names, or code-like tokens."
        )
    return (
        "OUTPUT LANGUAGE: en / English. Preserve every `{evidence:<id>}` "
        "placeholder and `{claim:<step>.<claim>}` token exactly as ASCII."
    )
