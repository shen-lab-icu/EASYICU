"""Compatibility exports for figure-owned article display policy.

Display placement is consumed while figure contracts are built, so the
implementation belongs below Reporting in ``figures.contracts``. This module
keeps the established import path without making lower layers depend on the
Reporting package.
"""

from ..figures.contracts import (
    ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN,
    ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION,
    ARTICLE_DISPLAY_POLICY_VALIDATOR,
    ARTICLE_DISPLAY_PURPOSE_CONFLICT,
    ARTICLE_DISPLAY_ROLE_UNSUPPORTED,
    ArticleDisplayDecision,
    ArticleDisplayPolicyError,
    ArticleDisplayPolicyRequest,
    DisplayPlacement,
    DisplayPurpose,
    decide_article_display,
)

__all__ = [
    "ARTICLE_DISPLAY_FAILED_CLOSED_RESULT_FORBIDDEN",
    "ARTICLE_DISPLAY_POLICY_SCHEMA_VERSION",
    "ARTICLE_DISPLAY_POLICY_VALIDATOR",
    "ARTICLE_DISPLAY_PURPOSE_CONFLICT",
    "ARTICLE_DISPLAY_ROLE_UNSUPPORTED",
    "ArticleDisplayDecision",
    "ArticleDisplayPolicyError",
    "ArticleDisplayPolicyRequest",
    "DisplayPlacement",
    "DisplayPurpose",
    "decide_article_display",
]
