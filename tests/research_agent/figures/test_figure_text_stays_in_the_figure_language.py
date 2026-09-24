"""Figure text is English, so every label a figure shows is English as well.

A Planner may write display labels in the language of the user's question.
Inside an English figure they mix scripts, so the figure uses each variable's
English source description instead, and names an event time after its event.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import easyicu.research_agent as ra  # noqa: E402
from easyicu.research_agent.execution.runners.bound_variable_display import (  # noqa: E402
    load_bound_variable_descriptions,
)
from easyicu.research_agent.figures.display_labels import (  # noqa: E402
    display_label,
    figure_language_labels,
    label_lookup,
    scoped_label_lookup,
)
from easyicu.research_agent.figures.presentation import wrap_figure_label  # noqa: E402

_CJK = re.compile(r"[㐀-鿿]")


def _context_manifest(run_dir: Path) -> dict:
    context = ra.ResearchContext(
        research_question="Is the lactate tertile associated with ICU readmission?",
        cohort=ra.CohortDescriptor(
            cohort_name="Adult ICU stays", database="mimiciv", n_patients=380, n_stays=412,
        ),
        variables=[
            ra.ConceptDescriptor(
                name="icu_readmission", description="ICU readmission", role="outcome",
                dtype="bool", source_concept="readmission",
            ),
            ra.ConceptDescriptor(
                name="readmission_time_hours", description="ICU readmission", role="time",
                dtype="float64", unit="h", source_concept="readmission",
            ),
            ra.ConceptDescriptor(
                name="lactate_tertile", description="lactate tertile", role="lab",
                dtype="int64", source_concept="lact",
            ),
        ],
        primary_exposure="lactate_tertile",
        target_outcome="icu_readmission",
    )
    path = run_dir / "research_context.json"
    path.write_text(context.model_dump_json(), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"context": {"relative_path": path.name, "sha256": digest}}


def test_a_label_in_another_script_becomes_the_english_source_description() -> None:
    labels = figure_language_labels(
        {
            "lactate_tertile": "乳酸三分位",
            "icu_readmission": "ICU再入院",
            "lactate_tertile=1": "低乳酸",
            "age": "Age (years)",
            "vasopressor_any": "是否使用升压药",
        },
        {"lactate_tertile": "lactate tertile", "icu_readmission": "ICU readmission"},
    )

    assert labels == {
        "lactate_tertile": "Lactate tertile",
        "icu_readmission": "ICU readmission",
        "age": "Age (years)",
    }
    assert not any(_CJK.search(label) for label in labels.values())


def test_label_lookups_never_return_text_in_another_script() -> None:
    labels = {
        "vasopressor_any": "是否使用升压药",
        "vasopressor_any=1": "使用升压药",
        "mort_28d": "28-day mortality",
        "sex=1": "Male",
    }

    assert label_lookup("vasopressor_any", labels) is None
    assert scoped_label_lookup("vasopressor_any", 1, labels) is None
    assert label_lookup("mort_28d", labels) == "28-day mortality"
    assert scoped_label_lookup("sex", 1, labels) == "Male"
    assert display_label("vasopressor_any", labels) == "Vasopressor Any"


def test_an_event_time_is_named_after_its_event(tmp_path: Path) -> None:
    descriptions = load_bound_variable_descriptions(
        run_dir=tmp_path, resolved_inputs=_context_manifest(tmp_path),
    )

    assert descriptions["icu_readmission"] == "ICU readmission"
    assert descriptions["readmission_time_hours"] == "time to ICU readmission (h)"
    assert descriptions["lactate_tertile"] == "lactate tertile"
    assert figure_language_labels(
        {"readmission_time_hours": "再入院时间（小时）", "icu_readmission": "ICU再入院"},
        descriptions,
    ) == {
        "readmission_time_hours": "Time to ICU readmission (h)",
        "icu_readmission": "ICU readmission",
    }


def test_figure_descriptions_come_only_from_the_digest_bound_context(tmp_path: Path) -> None:
    manifest = _context_manifest(tmp_path)
    manifest["context"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest mismatch"):
        load_bound_variable_descriptions(run_dir=tmp_path, resolved_inputs=manifest)
    assert load_bound_variable_descriptions(run_dir=tmp_path, resolved_inputs={}) == {}


def test_wrapping_keeps_punctuation_with_its_word() -> None:
    fig, ax = plt.subplots()
    renderer = fig.canvas.get_renderer()
    font = ax.yaxis.label.get_fontproperties()
    font.set_size(7.0)

    def width(text: str) -> float:
        return renderer.get_text_width_height_descent(text, font, False)[0]

    try:
        # The column ends just before the punctuation in whatever font renders,
        # so splitting the punctuation from its word would start a line with it.
        for label, before, intact in (
            ("Maximum serum lactate (arterial)", "Maximum serum lactate (arterial", "(arterial)"),
            ("Measurement availability: Serum lactate", "Measurement availability", "availability:"),
        ):
            lines = wrap_figure_label(
                label, renderer=renderer, font=font, width=width(before) + 0.5,
            ).split("\n")
            assert len(lines) > 1
            assert any(intact in line for line in lines)
            assert not any(line.startswith((")", ":")) for line in lines)
            assert " ".join(lines) == label
        cjk = "最高血清乳酸浓度动脉血样本"
        wrapped = wrap_figure_label(
            cjk, renderer=renderer, font=font, width=width(cjk[:5]) + 0.5,
        )
        assert "\n" in wrapped
        assert wrapped.replace("\n", "") == cjk
    finally:
        plt.close(fig)
