"""Prompt guardrails for pandas idioms seen in real LLM pilot runs."""

from __future__ import annotations


def test_coder_prompt_names_pandas_categorical_codes_gotcha() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "PANDAS IDIOM GOTCHAS" in coder_prompt
    assert "pd.Categorical(x).cat.codes" in coder_prompt
    assert "pd.Categorical(x).codes" in coder_prompt
    assert 'pd.Series(x).astype("category").cat.codes' in coder_prompt


def test_coder_prompt_keeps_sparse_event_negatives_in_exposure_denominator() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert (
        "Do not define the analytic cohort for a binary event/exposure" in coder_prompt
    )
    assert "<concept>_measured == 1" in coder_prompt
    assert "event-negative" in coder_prompt
    assert "untriggered" in coder_prompt


def test_coder_prompt_keeps_long_provenance_notes_out_of_plot_canvas() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Do not place long audit/provenance/reporting notes" in coder_prompt
    assert "inside result plots" in coder_prompt
    assert "write the full notes as a separate table" in coder_prompt


def test_coder_prompt_blocks_mixed_effect_scales_on_one_forest_axis() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "Never mix incompatible effect scales on a single forest-plot axis" in (
        coder_prompt
    )
    assert "risk differences" in coder_prompt
    assert "split" in coder_prompt
    assert "the plot by `effect_scale`" in coder_prompt
    assert "Use reader-facing labels in figures" in coder_prompt


def test_coder_prompt_prevents_resume_evidence_polluting_figure_rendering() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "For rendering-only figure steps" in coder_prompt
    assert "explicitly named upstream" in coder_prompt
    assert "previous figure source-data CSVs" in coder_prompt
    assert "robustness panels" in coder_prompt
    assert "render from" in coder_prompt
    assert "that table alone" in coder_prompt


def test_coder_prompt_keeps_footnotes_and_raw_ids_out_of_result_figures() -> None:
    from easyicu.research_agent.prompts import load_prompt_pack

    coder_prompt = load_prompt_pack()["coder"]

    assert "remove suffixes like `_modeled_or`" in coder_prompt
    assert "replace underscores with spaces" in coder_prompt
    assert "Do not draw figure captions, long footnotes" in coder_prompt
    assert "inside the" in coder_prompt
    assert "saved plotting canvas" in coder_prompt
    assert "so it cannot be clipped" in coder_prompt
