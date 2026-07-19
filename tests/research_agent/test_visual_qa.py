"""Visual QA deterministic checks plus optional VLM adapter."""

from __future__ import annotations

from pathlib import Path


def _write_plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [0, 1, 0])
    ax.set_xlabel("SOFA-2")
    ax.set_ylabel("Mortality rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def test_visual_qa_vlm_adapter_appends_findings(ra, tmp_path: Path):
    fig_path = tmp_path / "figure.png"
    _write_plot(fig_path)

    class _VisionLLM:
        name = "vision"

        def __init__(self):
            self.seen_paths = None

        def complete_with_images(self, *, prompt, image_paths, **kwargs):
            self.seen_paths = list(image_paths)
            return (
                '{"findings":[{"path":"figure.png","severity":"warning",'
                '"message":"Legend text may be too small.",'
                '"detail":{"panel":"main"}}]}'
            )

    from easyicu.research_agent.gates.visual_qa import VLMVisualQAAdapter, VisualQAAuditor

    llm = _VisionLLM()
    findings = VisualQAAuditor(
        vlm_adapter=VLMVisualQAAdapter(llm),
    ).audit(figure_paths=[fig_path])

    assert llm.seen_paths == [fig_path]
    assert any(f.validator == "vlm_visual_qa" for f in findings)
    vlm = [f for f in findings if f.validator == "vlm_visual_qa"][0]
    assert vlm.severity == "warning"
    assert vlm.detail["path"] == str(fig_path)


def test_pipeline_enables_vlm_visual_qa_when_client_is_configured(ra, tmp_path: Path):
    class _VisionLLM:
        name = "vision"

        def complete_with_images(self, *, prompt, image_paths, **kwargs):
            return '{"findings":[]}'

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=ra.MockLLMClient(),
        vlm_client=_VisionLLM(),
    )

    assert pipeline._enable_vlm_visual_qa is True


def test_pipeline_auto_enables_vlm_visual_qa_for_vision_capable_llm(ra, tmp_path: Path):
    class _VisionLLM:
        name = "vision"
        supports_vision = True

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return '{"findings":[]}'

        def complete_with_images(self, *, prompt, image_paths, **kwargs):
            return '{"findings":[]}'

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=_VisionLLM(),
    )

    assert pipeline._enable_vlm_visual_qa is True


def test_pipeline_auto_disables_vlm_visual_qa_for_text_only_llm(ra, tmp_path: Path):
    class _TextOnlyLLM:
        name = "text-only"
        supports_vision = False

        def complete(self, messages, *, max_tokens=2048, temperature=0.2):
            return '{"findings":[]}'

    pipeline = ra.ResearchAgentPipeline(
        workdir=tmp_path,
        llm=_TextOnlyLLM(),
    )

    assert pipeline._enable_vlm_visual_qa is False


def test_parse_vlm_visual_qa_response_tolerates_json_fence(ra, tmp_path: Path):
    fig_path = tmp_path / "a.png"
    fig_path.write_bytes(b"not a real image but path resolution is enough")
    raw = """```json
{"findings":[{"path":"a.png","severity":"error","message":"Blank panel."}]}
```"""

    from easyicu.research_agent.gates.visual_qa import parse_vlm_visual_qa_response

    findings = parse_vlm_visual_qa_response(raw, known_paths=[fig_path])
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["path"] == str(fig_path)


def test_visual_qa_flags_svg_text_overlap(ra, tmp_path: Path):
    fig_path = tmp_path / "overlap.svg"
    fig_path.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="panel_a">
            <text x="70" y="80" style="font-size: 16px; text-anchor: middle">Adjusted mortality</text>
          </g>
          <g id="panel_b">
            <text x="74" y="82" style="font-size: 16px; text-anchor: middle">Primary model</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    from easyicu.research_agent.gates.visual_qa import VisualQAAuditor

    findings = VisualQAAuditor(min_bytes=1).audit(figure_paths=[fig_path])

    assert any(f.severity == "error" and "overlapping text" in f.message for f in findings)


def test_visual_qa_downgrades_panel_label_title_overlap(ra, tmp_path: Path):
    fig_path = tmp_path / "panel_title.svg"
    fig_path.write_text(
        """
        <svg width="420pt" height="240pt" viewBox="0 0 420 240" xmlns="http://www.w3.org/2000/svg">
          <rect width="420" height="240" fill="white"/>
          <g id="panel_label">
            <text x="10" y="24" style="font-size: 18px">A</text>
          </g>
          <g id="title">
            <text x="11" y="24" style="font-size: 16px">Spearman Correlation Matrix: SOFA-2 Components vs Total SOFA-2</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    from easyicu.research_agent.gates.visual_qa import VisualQAAuditor

    findings = VisualQAAuditor(min_bytes=1).audit(figure_paths=[fig_path])

    assert any(f.severity == "warning" and "panel label close to a title" in f.message for f in findings)
    assert not any(f.severity == "error" and "overlapping text" in f.message for f in findings)


def test_visual_qa_flags_svg_cropped_text(ra, tmp_path: Path):
    fig_path = tmp_path / "cropped.svg"
    fig_path.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="ylabel">
            <text x="-12" y="80" transform="rotate(-90 -12 80)" style="font-size: 12px; text-anchor: middle">Adjusted predicted death (%)</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    from easyicu.research_agent.gates.visual_qa import VisualQAAuditor

    findings = VisualQAAuditor(min_bytes=1).audit(figure_paths=[fig_path])

    assert any("outside the canvas" in f.message for f in findings)


def test_visual_qa_svg_numeric_consistency_passes_when_value_is_present(ra, tmp_path: Path):
    fig_path = tmp_path / "numbers_ok.svg"
    fig_path.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="panel">
            <text x="40" y="40" style="font-size: 12px">OR 1.23</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    from easyicu.research_agent.gates.visual_qa import VisualQAAuditor

    findings = VisualQAAuditor(min_bytes=1).audit_with_expected(
        figure_paths=[fig_path],
        expected_numeric_by_path={str(fig_path): {"primary_or": 1.23}},
    )

    assert not any("numeric consistency" in f.message for f in findings)


def test_visual_qa_svg_numeric_consistency_warns_when_value_is_missing(ra, tmp_path: Path):
    fig_path = tmp_path / "numbers_bad.svg"
    fig_path.write_text(
        """
        <svg width="220pt" height="160pt" viewBox="0 0 220 160" xmlns="http://www.w3.org/2000/svg">
          <rect width="220" height="160" fill="white"/>
          <g id="panel">
            <text x="40" y="40" style="font-size: 12px">OR 0.87</text>
          </g>
        </svg>
        """.strip(),
        encoding="utf-8",
    )

    from easyicu.research_agent.gates.visual_qa import VisualQAAuditor

    findings = VisualQAAuditor(min_bytes=1).audit_with_expected(
        figure_paths=[fig_path],
        expected_numeric_by_path={str(fig_path): {"primary_or": 1.23}},
    )

    assert any("numeric consistency" in f.message for f in findings)
