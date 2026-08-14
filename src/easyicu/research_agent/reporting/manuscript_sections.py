"""Deterministic section specification and assembly for manuscript writing.

The writer agent owns one model call.  This module owns which manuscript
sections are requested, their bounded instructions, concurrent dispatch, and
fixed-order assembly.  Keeping that contract outside the agent prevents the
model-facing class from also becoming a manuscript workflow coordinator.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ManuscriptSectionSpec:
    """One immutable writer request in final manuscript order."""

    key: str
    section_name: str
    instruction: str
    max_tokens: int


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
            "direct-comparator key when the literature digest provides one."
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
            "  Primary predictor, outcome, covariates. For each, state the "
            "ICU-aware aggregation rule (from the research context: ordinal → "
            "max-in-window, labs → median, etc.).\n"
            "### Statistical analysis\n"
            "  Model family (logistic regression / Cox / clustering), adjustment "
            "set, sensitivity analyses (multiple-testing correction, subgroup "
            "analysis, ICU-rule-specific strata or missingness-pattern audits "
            "raised by the research context).\n"
            "### Software and reproducibility\n"
            "  One sentence: 'Analyses were conducted through the EasyICU "
            "research-agent pipeline; the full reproducibility envelope "
            "(prompt/response SHA-256 hashes, per-step scripts, and dependency "
            "lockfile) is released as supplementary material.'\n"
            "Target: 400-600 words. Cite at least one exact method-source key "
            "from the literature digest when available; do not cite a disease "
            "definition paper as statistical-method authority."
        ),
        max_tokens=2048,
    ),
    ManuscriptSectionSpec(
        key="results",
        section_name="Results",
        instruction=(
            "Write `## Results` with sub-sections:\n"
            "### Cohort characteristics\n"
            "  N, key demographics, cite {evidence:table_one} if available.\n"
            "### Primary outcome\n"
            "  Incidence, cite {evidence:outcome_rate}.\n"
            "### Primary association\n"
            "  Effect size, 95% CI, p-value, cite "
            "{evidence:primary_association} or {evidence:model_performance}.\n"
            "  When the machine digest supplies a host-authorized scientific "
            "claim, use its exact `{claim:<step>.<claim>}` token as a standalone "
            "sentence instead of independently wording the direction.\n"
            "### Sensitivity and subgroup analyses\n"
            "  Multiple-testing result, subgroup heterogeneity, E-value if "
            "available.\n"
            "### ICU-specific quality control\n"
            "  Report any ICU-rule-specific finding raised by the "
            "research-context validators (e.g. a stratum where a derived score "
            "collapsed to a degenerate value, a missingness pattern that "
            "violated an aggregation rule). Cite the corresponding registered "
            "evidence id. Omit this subsection if no such finding was produced.\n"
            "Target: 400-600 words. Every numeric claim MUST have an "
            "{evidence:id} citation."
        ),
        max_tokens=2048,
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
            "- Para 5: Strengths of the pipeline, evidence traceability, "
            "ICU-aware rules, and reproducibility.\n"
            "Requirements: full prose only, no bullets, no recommendations or "
            "causal claims, and at least one evidence citation or literature "
            "citation in each paragraph when available. Do not collapse the "
            "discussion into a two-sentence stub. Cite a screened direct "
            "comparator when available, and state the specific population, "
            "time-zero, estimand, or analysis difference instead of claiming "
            "novelty from database choice alone."
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
            "2. Single synthetic/database cohort → limited external "
            "generalisability.\n"
            "3. One ICU-specific limitation drawn from the registered evidence, "
            "expressed in concept-neutral terms — e.g. component-level "
            "missingness in a derived-score input, ordinal-score aggregation "
            "choice, time-window definition. Do not invent an ICU limitation "
            "that is not supported by a registered audit finding.\n"
            "4. LLM-in-the-loop limitation (generated code was audited but not "
            "manually reviewed line-by-line)."
        ),
        max_tokens=1024,
    ),
    ManuscriptSectionSpec(
        key="conclusion",
        section_name="Conclusion, Data availability, Funding, COI",
        instruction=(
            "Write these sections exactly:\n"
            "## Conclusion\n"
            "1-2 sentences. Associational phrasing. Each conclusion sentence "
            "must either be one exact host-authorized claim token or cite at "
            "least one registered evidence id. End with a call for "
            "prospective / external validation only if it can be tied to "
            "sensitivity, limitation, or validation evidence.\n\n"
            "If a host-authorized scientific claim is supplied, use its exact "
            "standalone `{claim:<step>.<claim>}` token for the current-study "
            "conclusion; the host will render and cite it.\n\n"
            "## Data and code availability\n"
            "'The cohort, generated scripts, SHA-256 evidence store, "
            "reproducibility envelope, STROBE checklist, and supplementary "
            "tables are released alongside this manuscript.'\n\n"
            "## Funding\n"
            "'Funding information was not available to the analysis agent and "
            "should be completed by the authors before journal submission.'\n\n"
            "## Conflicts of interest\n"
            "'The authors declare no conflicts of interest.'"
        ),
        max_tokens=512,
    ),
)


def render_manuscript_sections(
    *,
    call_section: Callable[..., str],
    common: Mapping[str, Any],
) -> str:
    """Dispatch independent sections concurrently and assemble fixed order."""

    with ThreadPoolExecutor(max_workers=len(MANUSCRIPT_SECTION_SPECS)) as executor:
        futures = [
            executor.submit(
                call_section,
                section_name=spec.section_name,
                instruction=spec.instruction,
                max_tokens=spec.max_tokens,
                **common,
            )
            for spec in MANUSCRIPT_SECTION_SPECS
        ]
        sections = [future.result() for future in futures]
    return "\n\n".join(section.strip() for section in sections if section.strip())


__all__ = [
    "MANUSCRIPT_SECTION_SPECS",
    "ManuscriptSectionSpec",
    "render_manuscript_sections",
]
