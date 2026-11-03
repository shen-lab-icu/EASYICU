"""Parenthetical synonyms come from a concept's clinical head, not its notes."""

from easyicu.research_agent.concept_catalog import _bracket_aliases


def test_bracket_aliases_ignore_the_implementation_tail():
    assert _bracket_aliases("furosemide (Lasix) — dosed (MIIV/PACU) per hour") == ["Lasix"]
    assert _bracket_aliases(
        "fentanyl rate — opioid infusion (excludes sufentanil)"
    ) == []
    assert _bracket_aliases("ventilation mode (IMV/NIV)") == ["IMV", "NIV"]
