"""The Writer is asked for claim tokens in the shape the strict gate keeps.

The gate removes a whole line when a claim token shares it with other text,
so a paraphrase cannot ride next to a host claim.  The Writer was asked for a
"standalone sentence" and placed tokens inside Discussion and Limitations
paragraphs; each such paragraph was deleted with all its prose, and an empty
Limitations failed the manuscript audit.  The contract now names the unit the
gate enforces, and keeps result tokens out of Limitations.

Generic manuscripts only; no study's values.
"""

from __future__ import annotations

from easyicu.research_agent.authority.manuscript_claim_policy import (
    SCIENTIFIC_CLAIM_WRITER_RULES,
    filter_evidence_bound_scaffold,
)
from easyicu.research_agent.reporting.manuscript_sections import MANUSCRIPT_SECTION_SPECS

REF = "primary_model.adjusted_association"
TOKEN = "{claim:" + REF + "}"
OWN_LINE = ("own line", "own paragraph")


class _Claim:
    claim_ref = REF

    @staticmethod
    def render_text() -> str:
        return "A host-rendered association sentence."


def _resolve(ref: str):
    return _Claim() if ref == REF else None


def test_the_gate_keeps_a_token_paragraph_and_deletes_a_shared_line():
    prose = "These estimates come from routinely collected records [@record_2015]."
    shared = f"## Discussion\n\n{prose} {TOKEN} The context matters.\n"
    separate = f"## Discussion\n\n{TOKEN}\n\n{prose} The context matters.\n"

    assert filter_evidence_bound_scaffold(shared, resolve_claim=_resolve).scaffold.strip() == "## Discussion"
    kept = filter_evidence_bound_scaffold(separate, resolve_claim=_resolve)
    assert kept.filtered_sentences == ()
    assert TOKEN in kept.scaffold and prose in kept.scaffold


def test_the_shared_rule_names_the_unit_the_gate_keeps():
    assert "alone on its own line" in SCIENTIFIC_CLAIM_WRITER_RULES
    assert "deletes any line on which a claim token shares space with other text" in SCIENTIFIC_CLAIM_WRITER_RULES


def test_every_section_that_may_carry_a_token_asks_for_it_on_its_own_line():
    asking = {
        spec.key: spec.instruction
        for spec in MANUSCRIPT_SECTION_SPECS
        if "{claim:" in spec.instruction and spec.key != "limitations"
    }

    assert {"abstract", "results", "discussion", "conclusion"} <= set(asking)
    for key, instruction in asking.items():
        assert any(phrase in instruction for phrase in OWN_LINE), key


def test_limitations_carry_no_result_token():
    (limitations,) = [spec for spec in MANUSCRIPT_SECTION_SPECS if spec.key == "limitations"]

    assert "Do not place `{claim:...}` tokens here" in limitations.instruction
