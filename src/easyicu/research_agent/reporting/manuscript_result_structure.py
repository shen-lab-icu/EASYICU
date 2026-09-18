"""One plan-derived Results structure shared by Writer and reader validation.

This is a reporting requirement, not an execution receipt: a planned analysis
must have a result or an explicit supported failure, never an invented finding.
No plan means legacy structure remains readable without inferring a new family.
"""

from ..planning.analysis_types import canonical_analysis_family
from ..schema import AnalysisPlan
from ..contracts.manuscript_result_structure import (
    COHORT_RESULT_HEADING,
    DEFAULT_PRIMARY_RESULT_HEADING,
    PRIMARY_RESULT_HEADINGS,
    PRIMARY_RESULT_HEADINGS_BY_FAMILY,
    RESULT_HEADINGS_BY_ROLE,
)

__all__ = ["PRIMARY_RESULT_HEADINGS", "required_result_subsections", "result_section_instruction"]


def required_result_subsections(plan: AnalysisPlan) -> tuple[str, ...]:
    """Derive headings from explicit plan family and scientific step roles."""
    primary = PRIMARY_RESULT_HEADINGS_BY_FAMILY.get(
        canonical_analysis_family(plan.analysis_type), DEFAULT_PRIMARY_RESULT_HEADING,
    )
    sections = [COHORT_RESULT_HEADING, primary]
    roles = {step.planned_analysis_role for step in plan.steps}
    sections.extend(heading for role, heading in RESULT_HEADINGS_BY_ROLE.items() if role in roles)
    return tuple(sections)


def result_section_instruction(plan: AnalysisPlan) -> str:
    """Use the same requirements at initial drafting and every repair."""
    headings = required_result_subsections(plan)
    roles = {step.planned_analysis_role for step in plan.steps}
    role_guidance: list[str] = []
    if "secondary" in roles and "Secondary analyses" in headings:
        role_guidance.append(
            "In Secondary analyses report the prespecified secondary results from "
            "the machine digest's `reportable_descriptive_results` or "
            "`reportable_secondary_results` blocks: the overall outcome count and "
            "risk, the question-relevant exposure-source or exposure-level groups, "
            "and their supplied uncertainty. When an executed step supplies neither "
            "block, write one sentence stating only that the prespecified secondary "
            "analysis was executed and its registered results are bound to the named "
            "evidence id; never leave this subsection empty and never invent numbers."
        )
    if "sensitivity" in roles and "Sensitivity and subgroup analyses" in headings:
        role_guidance.append(
            "In Sensitivity and subgroup analyses report the supplied sensitivity or "
            "subgroup results, including any registered multiplicity, heterogeneity "
            "or E-value output, from the same machine digest. When the digest records "
            "zero sensitivity result rows, report only that registered count with its "
            "exact evidence id and never claim stability, convergence, or a performed "
            "analysis."
        )
    return (
        "Write `## Results` with these required subsections, in order:\n"
        + "\n".join(f"### {heading}" for heading in headings)
        + "\n"
        + ("\n".join(role_guidance) + "\n" if role_guidance else "")
        + "In Cohort characteristics report the analysis unit, cohort count and "
        "relevant baseline summaries. Cite Table 1 when registered. In the primary "
        "results subsection answer the original question using every supplied "
        "primary metric, endpoint and comparison level. A figure callout alone "
        "does not report a result. Cite Figure 1 where its evidence supports the "
        "result, only when publication_figure_contract is registered.\n"
        "Report only executed results from the machine digest. For descriptive "
        "counts-only results use observed events, denominators and proportions; "
        "do not imply an estimated association. For models, name the exact metric "
        "and contrast; include effect sizes, confidence intervals and p-values "
        "only when explicitly supplied and authorized. Preserve the confidence "
        "level and exact test role; a nonlinearity or calibration test is not a "
        "primary association test. For prediction report supplied discrimination "
        "and calibration, distinguishing training and validation. For clustering "
        "keep fitting features separate from clinical/outcome comparisons.\n"
        "Use exact standalone `{claim:<step>.<claim>}` sentences for qualitative "
        "directions. Never copy, extend or paraphrase their digest text. For "
        "reportable_survival_results retain the RMST horizon, both group RMSTs, "
        "signed difference, supplied uncertainty and interval-specific estimates. "
        "Do not invent a constant hazard ratio. For reportable_secondary_results "
        "report all requested outcomes with their named metrics and ceilings.\n"
        "Include Secondary analyses and Sensitivity and subgroup analyses only "
        "when required above or when the digest supplies an executed result. "
        "For a required but unexecuted step, describe only its supported failure "
        "or absence, never success, stability or convergence. A zero result-row "
        "count does not establish that an analysis was unplanned or unnecessary. "
        "Do not add an empty section or a synthetic zero-count filler.\n"
        "ICU-specific quality control may be included only for recorded findings. "
        "Use reader-facing clinical labels, not raw snake_case identifiers or "
        "runtime terminology. Every numeric fact needs its exact {evidence:id}; "
        "never invent an alias, test or result. Use integer counts, two-decimal "
        "percentages and at most three decimals for effects and confidence limits. "
        "Target: 400-600 words, without padding unavailable analyses."
    )
