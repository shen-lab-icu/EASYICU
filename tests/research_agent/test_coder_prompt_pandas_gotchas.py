"""Prompt guardrails for pandas idioms seen in real LLM pilot runs."""

from __future__ import annotations


def test_coder_prompt_names_pandas_categorical_codes_gotcha() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "PANDAS IDIOM GOTCHAS" in coder_prompt
    assert "pd.Categorical(x).cat.codes" in coder_prompt
    assert "pd.Categorical(x).codes" in coder_prompt
    assert 'pd.Series(x).astype("category").cat.codes' in coder_prompt
