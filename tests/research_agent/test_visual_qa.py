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

    from easyicu.research_agent.visual_qa import VLMVisualQAAdapter, VisualQAAuditor

    llm = _VisionLLM()
    findings = VisualQAAuditor(
        vlm_adapter=VLMVisualQAAdapter(llm),
    ).audit(figure_paths=[fig_path])

    assert llm.seen_paths == [fig_path]
    assert any(f.validator == "vlm_visual_qa" for f in findings)
    vlm = [f for f in findings if f.validator == "vlm_visual_qa"][0]
    assert vlm.severity == "warning"
    assert vlm.detail["path"] == str(fig_path)


def test_parse_vlm_visual_qa_response_tolerates_json_fence(ra, tmp_path: Path):
    fig_path = tmp_path / "a.png"
    fig_path.write_bytes(b"not a real image but path resolution is enough")
    raw = """```json
{"findings":[{"path":"a.png","severity":"error","message":"Blank panel."}]}
```"""

    from easyicu.research_agent.visual_qa import parse_vlm_visual_qa_response

    findings = parse_vlm_visual_qa_response(raw, known_paths=[fig_path])
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].detail["path"] == str(fig_path)

