"""Shared question-to-display rules; reviewed plans retain scientific authority."""

RESEARCH_DISPLAY_RULES = (
    "Carry every requested endpoint through plan, execution and report; keep unmet requirements explicit. Select analyses and diagnostics by study design, not paper appearance.",
    "For each display bind its question, population/unit, group/reference, time window, numerator/denominator, estimand/uncertainty when applicable, source and placement.",
    "Table 1: groups are columns; characteristics/categories are rows. Preserve N, planned summaries, missingness and authorized comparisons.",
    "One denominator belongs in text or a table header. A flow diagram needs real selection stages; never infer upstream exclusions. Preserve the source ledger.",
    "Do not fill figure quotas or repeat the same result in every format. Use main displays for the question and supplementary displays for supporting diagnostics.",
)
RESEARCH_DISPLAY_GUIDE = "RESEARCH DISPLAY COVERAGE:\n" + "\n".join(
    "- " + rule for rule in RESEARCH_DISPLAY_RULES
)
