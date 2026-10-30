"""The actual stop renderer explains safe metadata admission failures."""

import subprocess
from pathlib import Path


def test_capacity_stop_is_explained_in_chinese_and_english():
    owner = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static/js/screens-guided-pi-error-text.js"
    code = r"""
global.window = {EU_HTML: {esc: s => s}, EasyICU: {guidedPi: {declare: (_, owner) => {global.owner = owner;}}}};
require(process.argv[1]);
for (const lang of [0, 1]) {
  const text = owner.create({tr: (en, zh) => lang ? zh : en, staticPreview: false})
    .runFailureText('research_pipeline_metadata_provider_capacity_unavailable');
  if (!text.includes(lang ? '本次请求没有发出' : 'This request was not sent')) process.exit(1);
  if (!text.includes(lang ? '远端状态未知' : 'remote status may be unknown')) process.exit(2);
  const shared = owner.create({tr: (en, zh) => lang ? zh : en, staticPreview: false})
    .runFailureText('research_pipeline_metadata_shared_scope_unavailable');
  if (!shared.includes(lang ? '本次请求没有发出' : 'This request was not sent')) process.exit(5);
  if (!shared.includes(lang ? '实例登记' : 'enrollment')) process.exit(6);
  const incomplete = owner.create({tr: (en, zh) => lang ? zh : en, staticPreview: false})
    .runFailureText('research_agent_pipeline_failed_closed', {code: 'planned_products_missing'});
  if (!incomplete.includes(lang ? '分析尚未完成' : 'analysis is incomplete')) process.exit(3);
  if (!incomplete.includes(lang ? '已生成的结果仍可查看' : 'remain available to view')) process.exit(4);
}
"""
    subprocess.run(["node", "-e", code, str(owner)], check=True, capture_output=True, text=True)
