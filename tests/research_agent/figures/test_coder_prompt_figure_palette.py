"""The house figure palette must be stated as a contract, not left implied.

``apply_publication_style`` applies the Nature-style rcParams *and returns the
palette*, but the prompt only ever emphasised keeping the return value of
``save_publication_figure``. A recorded four-panel figure therefore applied the
style and then hard-coded seven of its own hexes, shipping four unrelated
colour schemes inside one figure while ``PALETTE_CLINICAL`` sat unused in the
same call. The instruction is the behaviour, so it is locked here.
"""

from __future__ import annotations


def _coder_prompt() -> str:
    from easyicu.research_agent.providers.prompts import load_prompt_pack

    return load_prompt_pack()["coder"]


def test_coder_prompt_binds_figure_colour_to_the_returned_palette() -> None:
    coder_prompt = _coder_prompt()

    assert "KEEP THE RETURN VALUE OF `apply_publication_style` TOO" in coder_prompt
    assert "palette = apply_publication_style(fig)" in coder_prompt
    assert "the only colour source you may use" in coder_prompt
    # the refusal is explicit about every channel a figure can leak colour on
    for banned in ("hex literal", "named matplotlib colour", "`cmap` name"):
        assert banned in coder_prompt, banned
    # and about consistency across panels, which is what actually broke
    assert "one colour per series meaning, not per panel" in coder_prompt


def test_the_prompt_names_palette_keys_that_actually_exist() -> None:
    """A prompt that names a key the palette does not carry teaches a KeyError."""

    import re

    from easyicu.research_agent.figures.publication import PALETTE_CLINICAL

    coder_prompt = _coder_prompt()
    named = set(re.findall(r'palette\["([a-z_]+)"\]', coder_prompt))
    assert named, "the prompt should show the palette being indexed"
    unknown = sorted(named - set(PALETTE_CLINICAL))
    assert not unknown, f"prompt names palette keys that do not exist: {unknown}"


def test_the_palette_still_covers_the_roles_the_prompt_promises() -> None:
    from easyicu.research_agent.figures.publication import PALETTE_CLINICAL

    for role in ("baseline", "blue", "orange", "teal", "red", "neutral"):
        assert role in PALETTE_CLINICAL, role
    # soft variants exist for fills and bands, which the prompt points at
    assert any(key.endswith("_soft") for key in PALETTE_CLINICAL)
