from __future__ import annotations

from easyicu.research_agent.reporting.manuscript_sections import (
    MANUSCRIPT_SECTION_SPECS,
    render_manuscript_sections,
)


def test_manuscript_section_contract_has_fixed_publication_order() -> None:
    assert [spec.key for spec in MANUSCRIPT_SECTION_SPECS] == [
        "title",
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "limitations",
        "conclusion",
    ]
    assert [spec.max_tokens for spec in MANUSCRIPT_SECTION_SPECS] == [
        256,
        1024,
        4096,
        2048,
        2048,
        4096,
        1024,
        512,
    ]


def test_manuscript_section_assembly_is_ordered_and_forwards_common_context() -> None:
    seen: list[tuple[str, object]] = []

    def call_section(**kwargs: object) -> str:
        seen.append((str(kwargs["section_name"]), kwargs["context"]))
        return f"## {kwargs['section_name']}"

    rendered = render_manuscript_sections(
        call_section=call_section,
        common={"context": "sealed-context", "evidence_ids": ("e1",)},
    )

    expected_names = [spec.section_name for spec in MANUSCRIPT_SECTION_SPECS]
    assert rendered.split("\n\n") == [f"## {name}" for name in expected_names]
    assert sorted(seen) == sorted((name, "sealed-context") for name in expected_names)


def test_section_specs_keep_literature_and_evidence_boundaries() -> None:
    instructions = {spec.key: spec.instruction for spec in MANUSCRIPT_SECTION_SPECS}
    assert "direct-comparator" in instructions["introduction"]
    assert "method-source key" in instructions["methods"]
    assert "{evidence:id}" in instructions["results"]
    assert "specific population" in instructions["discussion"]
