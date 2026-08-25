"""Deterministic section specification and assembly for manuscript writing.

The writer agent owns one model call per section.  This module owns which
manuscript sections are requested, their bounded instructions, budget-safe
dispatch, required-subsection validation, one targeted structural retry, and
fixed-order assembly.  Keeping that contract outside the agent prevents the
model-facing class from also becoming a manuscript workflow coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping

from .administrative_authority import (
    ManuscriptAdministrativeAuthority,
    render_manuscript_administrative_sections,
)


@dataclass(frozen=True)
class ManuscriptSectionSpec:
    """One immutable writer request in final manuscript order."""

    key: str
    section_name: str
    instruction: str
    max_tokens: int
    required_subsections: tuple[str, ...] = ()


class ManuscriptSectionContractError(RuntimeError):
    """A writer section remained structurally incomplete after one retry."""

    def __init__(self, *, section_name: str, missing_subsections: tuple[str, ...]):
        self.section_name = section_name
        self.missing_subsections = missing_subsections
        super().__init__(
            f"Writer section {section_name!r} has missing or empty required "
            f"subsections after one targeted retry: {', '.join(missing_subsections)}"
        )


class ManuscriptReaderQualityContractError(RuntimeError):
    """A targeted section retry did not close deterministic reader errors."""

    def __init__(self, *, findings: tuple[tuple[str, str, str], ...]):
        self.findings = findings
        detail = "; ".join(
            f"{code} ({section}): {message}" for code, section, message in findings
        )
        super().__init__(
            "Writer sections still fail deterministic reader-quality checks "
            f"after one targeted retry: {detail}"
        )


MANUSCRIPT_SECTION_SPECS = (
    ManuscriptSectionSpec(
        key="title",
        section_name="Title and Keywords",
        instruction=(
            "Write:\n"
            "1. `# <title>` — 12-20 words, include study design + cohort + "
            "primary scientific question, but no numeric result, effect direction, "
            "association claim, or verdict.\n"
            "2. On the next line: `**Keywords:** keyword1, keyword2, ...` "
            "(5-7 keywords).\n"
            "Nothing else."
        ),
        max_tokens=256,
    ),
    ManuscriptSectionSpec(
        key="abstract",
        section_name="Abstract",
        instruction=(
            "Write `## Abstract` with four labelled paragraphs:\n"
            "- **Background:** 2-3 sentences (clinical importance, knowledge "
            "gap).\n"
            "- **Methods:** 3-4 sentences (cohort, design, primary analysis, "
            "ICU-aware aggregation).\n"
            "- **Results:** 4-5 sentences (N, outcome incidence, primary effect "
            "size with 95% CI and p, one supporting finding).\n"
            "- **Conclusions:** 1-2 sentences (associational phrasing only, call "
            "for validation).\n"
            "Use an exact standalone `{claim:<step>.<claim>}` sentence for any "
            "current-study qualitative direction or comparison when the machine "
            "digest supplies one; do not paraphrase it.\n"
            "Use reader-facing clinical labels; do not expose raw snake_case "
            "identifiers, internal reason codes, or host/runtime terminology.\n"
            "Target: 200-300 words total."
        ),
        max_tokens=1024,
    ),
    ManuscriptSectionSpec(
        key="introduction",
        section_name="Introduction",
        instruction=(
            "Write `## Introduction` with 4-5 paragraphs (900-1200 words "
            "total):\n"
            "- Para 1: Clinical importance of the ICU question and why it "
            "matters now.\n"
            "- Para 2: Prior evidence on the key predictor / score / exposure. "
            "Use evidence ids from the digest when possible and cite literature "
            "if available.\n"
            "- Para 3: What prior studies did well, and where they still leave "
            "uncertainty.\n"
            "- Para 4: The specific gap in the literature that this study "
            "addresses.\n"
            "- Para 5: One sentence on the objective, one sentence on the "
            "hypothesis, and one sentence on the expected contribution.\n"
            "Requirements: write full prose (no bullets), avoid generic filler, "
            "and include at least one evidence citation or literature citation "
            "in each paragraph when evidence is available. Do not collapse the "
            "introduction into two sentences. Cite at least one exact "
            "direct-comparator key when the literature digest provides one. "
            "Use reader-facing clinical labels; do not expose raw snake_case "
            "identifiers, internal reason codes, or host/runtime terminology."
        ),
        max_tokens=4096,
    ),
    ManuscriptSectionSpec(
        key="methods",
        section_name="Methods",
        instruction=(
            "Write `## Methods` with sub-sections:\n"
            "### Study design and cohort\n"
            "  Database, setting (ICU type), inclusion/exclusion criteria, time "
            "period.\n"
            "### Variables\n"
            "  Primary predictor, outcome, covariates. For each, state only the "
            "host-bound materialized representation and analysis window. A "
            "precomputed maximum, minimum, mean, or first value must not be "
            "reinterpreted using the source concept's default aggregation rule.\n"
            "### Statistical analysis\n"
            "  Model family (logistic regression / Cox / clustering), adjustment "
            "set, sensitivity analyses (multiple-testing correction, subgroup "
            "analysis, ICU-rule-specific strata or missingness-pattern audits "
            "raised by the research context).\n"
            "  Copy the executed adjustment set from the machine digest exactly "
            "and keep that same set in Results; do not infer or substitute a "
            "different covariate.\n"
            "### Software and reproducibility\n"
            "  State that analyses were conducted through the EasyICU "
            "research-agent pipeline and describe only run artifacts named in "
            "the machine digest. Do not claim that any artifact is released, "
            "public, or supplementary material; the host owns release facts.\n"
            "Target: 400-600 words. Cite at least one exact method-source key "
            "from the literature digest when available; do not cite a disease "
            "definition paper as statistical-method authority."
        ),
        max_tokens=2048,
        required_subsections=(
            "Study design and cohort",
            "Variables",
            "Statistical analysis",
            "Software and reproducibility",
        ),
    ),
    ManuscriptSectionSpec(
        key="results",
        section_name="Results",
        instruction=(
            "Write `## Results` with sub-sections:\n"
            "### Cohort characteristics\n"
            "  N, key demographics. When `table_one` is available, call the "
            "display `Table 1` in prose and cite {evidence:table_one}.\n"
            "### Primary outcome\n"
            "  Incidence, cite {evidence:outcome_rate}.\n"
            "### Primary association\n"
            "  Effect size, 95% CI, p-value, cite "
            "{evidence:primary_association} or {evidence:model_performance}.\n"
            "  Name the exact metric and contrast for every reported value; "
            "never use an unlabelled `point estimate`, `score`, or `range`.\n"
            "  When the machine digest supplies a host-authorized scientific "
            "claim, use its exact `{claim:<step>.<claim>}` token as a standalone "
            "sentence instead of independently wording the direction.\n"
            "### Sensitivity and subgroup analyses\n"
            "  Multiple-testing result, subgroup heterogeneity, E-value if "
            "available. When the machine digest supplies a "
            "`reportable_descriptive_results` block, report its overall outcome "
            "count/risk and the question-relevant exposure-source or exposure-level "
            "groups, including their supplied uncertainty, without calculating "
            "contrasts. When the machine digest supplies a "
            "`reportable_secondary_results` block, report its level-specific "
            "continuous-outcome medians and IQRs plus the named adjusted trend "
            "test here. Preserve its interpretation ceiling; do not calculate "
            "or infer an unsupported direction.\n"
            "### ICU-specific quality control\n"
            "  Report any ICU-rule-specific finding raised by the "
            "research-context validators (e.g. a stratum where a derived score "
            "collapsed to a degenerate value, a missingness pattern that "
            "violated an aggregation rule). Cite the corresponding registered "
            "evidence id. Omit this subsection if no such finding was produced.\n"
            "When `publication_figure_contract` is available, call the canonical "
            "result display `Figure 1` at least once in the Results subsection "
            "whose claim it supports. Do not invent a table or figure callout "
            "when its evidence id is absent.\n"
            "Target: 400-600 words. Every numeric claim MUST have an "
            "{evidence:id} citation. Use reader-facing clinical labels; do not "
            "expose raw snake_case identifiers, internal reason codes, or "
            "host/runtime terminology."
        ),
        max_tokens=2048,
        required_subsections=(
            "Cohort characteristics",
            "Primary outcome",
            "Primary association",
            "Sensitivity and subgroup analyses",
        ),
    ),
    ManuscriptSectionSpec(
        key="discussion",
        section_name="Discussion",
        instruction=(
            "Write `## Discussion` with 5 paragraphs (900-1300 words total):\n"
            "- Para 1: Restate the main finding and interpret it cautiously in "
            "the context of the results.\n"
            "- Para 2: Compare with prior literature and explain where this "
            "study agrees or diverges.\n"
            "- Para 3: Discuss plausible mechanisms using only associational "
            "language ('may reflect', 'could be consistent with', 'one possible "
            "explanation').\n"
            "- Para 4: Clinical implications, limits to generalisability, and why "
            "the result should not be over-interpreted.\n"
            "- Para 5: Methodological strengths, evidence traceability, ICU-aware "
            "rules, and the reproducibility conditions that materially affect "
            "interpretation. Do not list artifacts or praise the pipeline.\n"
            "Requirements: full prose only, no bullets, no recommendations or "
            "causal claims, and at least one evidence citation or literature "
            "citation in each paragraph when available. Do not collapse the "
            "discussion into a two-sentence stub. Cite a screened direct "
            "comparator when available, and state the specific population, "
            "time-zero, estimand, or analysis difference instead of claiming "
            "novelty from database choice alone. Use reader-facing clinical "
            "labels; do not expose raw snake_case identifiers, internal reason "
            "codes, or host/runtime terminology. Avoid generic prose that could "
            "be copied unchanged into an unrelated ICU study."
        ),
        max_tokens=4096,
    ),
    ManuscriptSectionSpec(
        key="limitations",
        section_name="Limitations",
        instruction=(
            "Write `## Limitations` — one paragraph, 150-250 words. Include at "
            "least:\n"
            "1. Observational design → no causal inference, residual "
            "confounding.\n"
            "2. Single database cohort → limited external "
            "generalisability.\n"
            "3. One ICU-specific limitation drawn from the registered evidence, "
            "expressed in concept-neutral terms — e.g. component-level "
            "missingness in a derived-score input, ordinal-score aggregation "
            "choice, time-window definition. Do not invent an ICU limitation "
            "that is not supported by a registered audit finding.\n"
            "4. Automated-pipeline limitation: execution artifacts and adapters "
            "require independent review. Mention LLM-generated analysis code only "
            "when the machine digest explicitly records that generation mode."
        ),
        max_tokens=1024,
    ),
    ManuscriptSectionSpec(
        key="conclusion",
        section_name="Conclusion",
        instruction=(
            "## Conclusion\n"
            "1-2 sentences. Associational phrasing. Each conclusion sentence "
            "must either be one exact host-authorized claim token or cite at "
            "least one registered evidence id. End with a call for "
            "prospective / external validation only if it can be tied to "
            "sensitivity, limitation, or validation evidence.\n\n"
            "If a host-authorized scientific claim is supplied, use its exact "
            "standalone `{claim:<step>.<claim>}` token for the current-study "
            "conclusion; the host will render and cite it. Do not write funding, "
            "ethics, conflicts, data/code availability, or release statements; "
            "the host owns those administrative facts. Use reader-facing "
            "clinical labels; do not expose raw snake_case identifiers, internal "
            "reason codes, or host/runtime terminology."
        ),
        max_tokens=512,
    ),
)


MANUSCRIPT_WRITER_CONTRACT_VERSION = "4"


def manuscript_writer_contract_sha256() -> str:
    """Return the stable identity of the reader-facing Writer contract.

    Report-only resume may reuse a prior free-form scaffold only while both
    the executed analysis ledger and this contract remain unchanged.  The
    explicit version covers shared Writer policy outside the section specs;
    it must be bumped whenever that policy changes.
    """

    payload = {
        "contract_version": MANUSCRIPT_WRITER_CONTRACT_VERSION,
        "sections": [
            {
                "key": spec.key,
                "section_name": spec.section_name,
                "instruction": spec.instruction,
                "max_tokens": spec.max_tokens,
                "required_subsections": list(spec.required_subsections),
            }
            for spec in MANUSCRIPT_SECTION_SPECS
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ensure_section_heading(spec: ManuscriptSectionSpec, text: str) -> str:
    """Restore only the mechanical heading owned by the section contract."""

    stripped = str(text or "").strip()
    if not stripped:
        return stripped
    if spec.key == "title":
        if stripped.startswith("# "):
            return stripped
        first, *rest = stripped.splitlines()
        return "\n".join([f"# {first.lstrip('# ').strip()}", *rest]).strip()
    expected = f"## {spec.section_name}"
    if any(line.strip() == expected for line in stripped.splitlines()):
        return stripped
    return f"{expected}\n\n{stripped}"


def _missing_required_subsections(
    spec: ManuscriptSectionSpec,
    text: str,
) -> tuple[str, ...]:
    """Return required level-three headings that are absent or have no prose."""

    lines = str(text or "").splitlines()
    missing: list[str] = []
    if spec.key != "title":
        prose = "\n".join(
            line for line in lines if not line.lstrip().startswith("#") and line.strip()
        )
        prose = re.sub(r"<!--.*?-->", "", prose, flags=re.S)
        if not re.search(r"[A-Za-z]{2,}", prose):
            missing.append("section body")
    for subsection in spec.required_subsections:
        expected = f"### {subsection}".casefold()
        heading_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().casefold() == expected
            ),
            None,
        )
        if heading_index is None:
            missing.append(subsection)
            continue
        next_heading_index = next(
            (
                index
                for index in range(heading_index + 1, len(lines))
                if lines[index].lstrip().startswith("#")
            ),
            len(lines),
        )
        if not any(
            line.strip() for line in lines[heading_index + 1 : next_heading_index]
        ):
            missing.append(subsection)
    return tuple(missing)


def _assemble_scientific_sections(sections: Mapping[str, str]) -> str:
    return "\n\n".join(
        sections[spec.key].strip()
        for spec in MANUSCRIPT_SECTION_SPECS
        if sections.get(spec.key, "").strip()
    )


def _quality_repair_specs(
    scientific: str,
    *,
    expected_display_labels: tuple[str, ...] = (),
) -> tuple[tuple[ManuscriptSectionSpec, str], ...]:
    """Map deterministic manuscript findings to their section owners."""

    from .manuscript_quality import audit_manuscript_quality

    by_key = {spec.key: spec for spec in MANUSCRIPT_SECTION_SPECS}
    section_keys = {
        "Title": ("title",),
        "Abstract": ("abstract",),
        "Introduction": ("introduction",),
        "Methods": ("methods",),
        "Results": ("results",),
        "Discussion": ("discussion",),
        "Limitations": ("limitations",),
        "Conclusion": ("conclusion",),
        "Methods/Results": ("methods", "results"),
    }
    messages: dict[str, list[str]] = {}
    for finding in audit_manuscript_quality(
        scientific,
        expected_display_labels=expected_display_labels,
        require_administrative_sections=False,
    ).findings:
        if finding.severity != "error":
            continue
        owner_keys = section_keys.get(finding.section, ())
        if finding.code == "MANUSCRIPT_ADJUSTMENT_SET_CONFLICT":
            # Results are the executed-result reporting surface.  When the two
            # sections disagree, repair the Methods description to the exact
            # machine digest instead of paying for a second free-form Results
            # rewrite that could move the reported estimate.
            owner_keys = ("methods",)
        for key in owner_keys:
            detail = f"{finding.code}: {finding.message}"
            if finding.excerpts:
                detail += " Offending text: " + "; ".join(finding.excerpts)
            messages.setdefault(key, []).append(detail)
    return tuple(
        (by_key[key], "\n".join(f"- {message}" for message in values))
        for key, values in messages.items()
    )


def _remaining_quality_errors(
    scientific: str,
    *,
    expected_display_labels: tuple[str, ...] = (),
) -> tuple[tuple[str, str, str], ...]:
    from .manuscript_quality import audit_manuscript_quality

    return tuple(
        (finding.code, finding.section, finding.message)
        for finding in audit_manuscript_quality(
            scientific,
            expected_display_labels=expected_display_labels,
            require_administrative_sections=False,
        ).findings
        if finding.severity == "error"
    )


def _existing_scientific_sections(manuscript: str) -> dict[str, str]:
    """Project an existing scaffold back onto the eight Writer owners."""

    text = str(manuscript or "")
    sections: dict[str, str] = {}
    first_level_two = re.search(r"^##\s+", text, flags=re.M)
    title = text[: first_level_two.start() if first_level_two else len(text)].strip()
    if title:
        sections["title"] = title
    matches = list(re.finditer(r"^##\s+([^\n]+?)\s*$", text, flags=re.M))
    key_by_name = {
        spec.section_name.casefold(): spec.key
        for spec in MANUSCRIPT_SECTION_SPECS
        if spec.key != "title"
    }
    for index, match in enumerate(matches):
        key = key_by_name.get(match.group(1).strip().casefold())
        if key is None:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[key] = text[match.start() : end].strip()
    return sections


def repair_existing_manuscript_sections(
    manuscript: str,
    *,
    call_section: Callable[..., str],
    common: Mapping[str, Any],
    administrative_authority: ManuscriptAdministrativeAuthority | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Regenerate only section owners named by deterministic quality errors."""

    from .manuscript_quality import repair_reader_structure_from_existing_prose

    manuscript, _structural_repairs = repair_reader_structure_from_existing_prose(
        manuscript
    )
    sections = _existing_scientific_sections(manuscript)
    from .manuscript_quality import expected_manuscript_display_labels

    display_labels = expected_manuscript_display_labels(
        tuple(common.get("evidence_ids") or ())
    )
    repaired_keys: list[str] = []
    scientific = _assemble_scientific_sections(sections)
    for attempt in range(2):
        repair_specs = _quality_repair_specs(
            scientific,
            expected_display_labels=display_labels,
        )
        if not repair_specs:
            administrative = render_manuscript_administrative_sections(
                administrative_authority
            )
            return "\n\n".join((scientific, administrative)), tuple(repaired_keys)
        for spec, error_detail in repair_specs:
            repair_instruction = (
                spec.instruction
                + "\n\nREADER-QUALITY CONTRACT MIGRATION:\n"
                + "The prior verified draft failed these deterministic checks owned "
                + f"by this section:\n{error_detail}\n"
                + "Regenerate the complete section from the same machine evidence. "
                + "Resolve every listed error without adding an unsupported result, "
                + "changing the executed method, exposing runtime identifiers, or "
                + "mentioning this migration. Preserve all required headings and "
                + "labels."
                + (
                    " This is the final bounded repair attempt; verify every "
                    "offending term is absent before returning."
                    if attempt == 1
                    else ""
                )
            )
            repaired = _ensure_section_heading(
                spec,
                call_section(
                    section_name=spec.section_name,
                    instruction=repair_instruction,
                    max_tokens=spec.max_tokens,
                    **common,
                ),
            )
            missing_subsections = _missing_required_subsections(spec, repaired)
            if missing_subsections:
                raise ManuscriptSectionContractError(
                    section_name=spec.section_name,
                    missing_subsections=missing_subsections,
                )
            sections[spec.key] = repaired
            if spec.key not in repaired_keys:
                repaired_keys.append(spec.key)
        scientific = _assemble_scientific_sections(sections)

    remaining = _remaining_quality_errors(
        scientific,
        expected_display_labels=display_labels,
    )
    if remaining:
        raise ManuscriptReaderQualityContractError(findings=remaining)
    administrative = render_manuscript_administrative_sections(administrative_authority)
    return "\n\n".join((scientific, administrative)), tuple(repaired_keys)


def render_manuscript_sections(
    *,
    call_section: Callable[..., str],
    common: Mapping[str, Any],
    administrative_authority: ManuscriptAdministrativeAuthority | None = None,
) -> str:
    """Dispatch scientific sections and append host-owned administrative facts.

    Section calls are deliberately sequential.  A Provider transport whose
    output cap is not enforceable must reserve its conservative worst-case
    usage before every request.  Dispatching all sections concurrently makes
    those reservations overlap and can exhaust a run stop-loss even when the
    completed sections are inexpensive.  Fixed-order dispatch keeps at most
    one reservation in flight and stops immediately on the first failure.
    """

    sections: dict[str, str] = {}
    from .manuscript_quality import expected_manuscript_display_labels

    display_labels = expected_manuscript_display_labels(
        tuple(common.get("evidence_ids") or ())
    )
    for spec in MANUSCRIPT_SECTION_SPECS:
        section = _ensure_section_heading(
            spec,
            call_section(
                section_name=spec.section_name,
                instruction=spec.instruction,
                max_tokens=spec.max_tokens,
                **common,
            ),
        )
        missing_subsections = _missing_required_subsections(spec, section)
        if missing_subsections:
            retry_instruction = (
                spec.instruction
                + "\n\nSTRUCTURAL CONTRACT REPAIR:\n"
                + "The previous draft omitted or left empty these required "
                + "subsections: "
                + ", ".join(
                    (
                        "the main section body"
                        if subsection == "section body"
                        else f"`### {subsection}`"
                    )
                    for subsection in missing_subsections
                )
                + ". Regenerate the complete section. Every listed subsection "
                + "must contain evidence-bound manuscript prose. Do not mention "
                + "the repair or the previous draft."
            )
            section = _ensure_section_heading(
                spec,
                call_section(
                    section_name=spec.section_name,
                    instruction=retry_instruction,
                    max_tokens=spec.max_tokens,
                    **common,
                ),
            )
            missing_subsections = _missing_required_subsections(spec, section)
        if missing_subsections:
            raise ManuscriptSectionContractError(
                section_name=spec.section_name,
                missing_subsections=missing_subsections,
            )
        sections[spec.key] = section

    scientific = _assemble_scientific_sections(sections)
    from .manuscript_quality import repair_reader_structure_from_existing_prose

    scientific, _structural_repairs = repair_reader_structure_from_existing_prose(
        scientific
    )
    sections = _existing_scientific_sections(scientific)
    for spec, error_detail in _quality_repair_specs(
        scientific,
        expected_display_labels=display_labels,
    ):
        repair_instruction = (
            spec.instruction
            + "\n\nREADER-QUALITY CONTRACT REPAIR:\n"
            + "The assembled draft failed these deterministic checks owned by "
            + f"this section:\n{error_detail}\n"
            + "Regenerate the complete section from the same machine evidence. "
            + "Resolve every listed error without adding an unsupported result, "
            + "changing the executed method, exposing runtime identifiers, or "
            + "mentioning this repair. Preserve all required headings and labels."
        )
        repaired = _ensure_section_heading(
            spec,
            call_section(
                section_name=spec.section_name,
                instruction=repair_instruction,
                max_tokens=spec.max_tokens,
                **common,
            ),
        )
        missing_subsections = _missing_required_subsections(spec, repaired)
        if missing_subsections:
            raise ManuscriptSectionContractError(
                section_name=spec.section_name,
                missing_subsections=missing_subsections,
            )
        sections[spec.key] = repaired

    scientific = _assemble_scientific_sections(sections)
    remaining = _remaining_quality_errors(
        scientific,
        expected_display_labels=display_labels,
    )
    if remaining:
        raise ManuscriptReaderQualityContractError(findings=remaining)
    administrative = render_manuscript_administrative_sections(administrative_authority)
    return "\n\n".join(part for part in (scientific, administrative) if part)


__all__ = [
    "MANUSCRIPT_SECTION_SPECS",
    "MANUSCRIPT_WRITER_CONTRACT_VERSION",
    "ManuscriptReaderQualityContractError",
    "ManuscriptSectionContractError",
    "ManuscriptSectionSpec",
    "manuscript_writer_contract_sha256",
    "repair_existing_manuscript_sections",
    "render_manuscript_sections",
]
