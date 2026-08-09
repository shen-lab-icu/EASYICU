"""Adversarial canary corpus for ``_NUMERIC_IN_PROSE_RE``.

The numeric-in-prose matcher feeds value-level provenance binding
(``bind_numeric_values``) and STRICT-mode enforcement. These cases pin the
behaviour that matters for real (and Chinese / approximate / interval)
manuscript prose, so a future regex tweak that silently widens or narrows
coverage trips a test instead of a reviewer.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.authority.evidence_store import _NUMERIC_IN_PROSE_RE


def _matches(text: str) -> list[str]:
    return [m.group("value") for m in _NUMERIC_IN_PROSE_RE.finditer(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        # Decimals are bound regardless of an approximation prefix — the
        # *number* is what carries provenance, not the hedge word.
        ("approximately 1.4 fold higher", ["1.4"]),
        ("the HR was ~1.29", ["1.29"]),
        # Chinese approximation / interval prose: CJK chars never interfere
        # with the lookbehind/lookahead, even with no separating space.
        ("校正后 OR 约为 1.42（95% CI 1.11–1.50）", ["1.42", "1.11", "1.50"]),
        ("约0.05", ["0.05"]),
        ("死亡率约 12.5%，n=1,234", ["12.5%", "1,234"]),
        ("12.5%、7.3% 和 4.1%", ["12.5%", "7.3%", "4.1%"]),
        # Interval expressions: both endpoints bind (en-dash or " to ").
        ("AUROC 0.766 to 0.812", ["0.766", "0.812"]),
        ("1.2–1.5 range", ["1.2", "1.5"]),
        # Short data percentages now bind (previously a provenance blind spot).
        ("mortality was 23%", ["23%"]),
        ("8% decline", ["8%"]),
        ("下降了 5%", ["5%"]),
        ("100% adherence", ["100%"]),
        # Counted short integers bind (2026-07-25). Small subgroup sizes, death
        # counts and event counts are the numbers a reviewer checks first and
        # the ones a writer most often gets wrong; leaving them unbound left
        # "the subgroup included 42 patients" unverifiable while every decimal
        # in the same sentence was value-checked. Binding is gated on the
        # counting phrase, never on the digits alone.
        ("n=42 patients enrolled", ["42"]),
        ("n = 8 for the primary model", ["8"]),
        ("There were 8 deaths and 17 events", ["8", "17"]),
        ("recruited across 3 sites", ["3"]),
    ],
)
def test_canary_matches(text: str, expected: list[str]) -> None:
    assert _matches(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Confidence / credible-interval *levels* are labels, not claims.
        "at the 90% confidence level",
        "99% credible interval",
        "95% CI 1.11 to 1.50",  # the level is skipped; the bounds still bind
        "95%CI without a space",
        # Identifier / structural collisions the 2-digit rejection protects.
        # These stay unbound even though counted short integers now bind: none
        # of them is followed by a counted noun or preceded by "n =".
        "SOFA-2 score",
        "Section 4 results",
        "Figure 2 and Table 1",
        "the 30-day endpoint",
        "between 8-12 events",  # a range bound is not a count
        "ARDS-3 phenotype",
    ],
)
def test_canary_skips_levels_and_identifiers(text: str) -> None:
    bound = _matches(text)
    # No bare 1-2 digit integer or confidence-level percent leaks through.
    assert "90%" not in bound
    assert "99%" not in bound
    assert "95%" not in bound
    assert "42" not in bound
    assert "2" not in bound
    assert "4" not in bound
    assert "3" not in bound
