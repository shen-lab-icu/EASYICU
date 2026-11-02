"""Synthetic EvidenceStore to normal preview owner and production JS renderer."""
import json
from pathlib import Path
import shutil

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.webserver.agent_runs import read_run_evidence_preview
from tests.support.node import run_node


@pytest.mark.parametrize("mode", ["llm", "deterministic_standard"])
def test_registered_synthetic_evidence_survives_normal_reader_path(tmp_path, mode):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    evidence = EvidenceStore(tmp_path)
    input_record = evidence.register_text(kind="log", description="synthetic input", text="value=3", filename="input.txt", evidence_id="input_synthetic")
    code = evidence.register_text(kind="code", description="synthetic code", text="print(3)", filename="analysis.py", evidence_id="code_synthetic", produced_by_step="step1", inputs=[input_record.evidence_id], producer="coder" if mode == "llm" else "standard_executor", generation_mode=mode)
    output = evidence.register_file(kind="table", description="synthetic result", source_path=_synthetic_output(tmp_path), evidence_id="output_synthetic", produced_by_step="step1", inputs=[input_record.evidence_id], script_evidence_id=code.evidence_id, producer="runner", generation_mode=mode)
    response = read_run_evidence_preview(str(tmp_path), output.evidence_id, output.sha256)
    assert response["ok"] is True
    payload = response["payload"]
    assert payload["generation_mode"] == mode
    assert {row["relation"] for row in payload["declared_lineage"]} == {"analysis_code", "input_data"}
    assert all(row["status"] == "registered" for row in payload["declared_lineage"])
    owner = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js"
    script = r"""
global.window = {EU_LANG:'zh', EU_HTML: {esc:v=>String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}};
require(process.argv[1]); require(process.argv[2]);
const html=window.EasyICU.guidedPi.require('evidencePreview').render(JSON.parse(process.argv[3]));
process.stdout.write(html);
"""
    result = run_node(node, script, str(owner / "screens-guided-pi-modules.js"), str(owner / "screens-guided-pi-evidence-preview.js"), json.dumps(payload), check=True)
    assert "记录的生成方式" in result.stdout
    assert ("模型生成代码" if mode == "llm" else "标准执行器") in result.stdout
    assert "code_synthetic" in result.stdout and "input_synthetic" in result.stdout
    assert "verified_tool" not in result.stdout and "composed_workflow" not in result.stdout


def _synthetic_output(run_dir):
    path = run_dir / "synthetic.csv"
    path.write_text("value\n3\n")
    return path
