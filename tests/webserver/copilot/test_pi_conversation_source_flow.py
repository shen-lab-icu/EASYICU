"""Source selection and artifact reading do not manufacture research decisions."""
import json
import shutil
import subprocess

import pytest

from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    return json.loads(subprocess.check_output([node, "--eval", script], text=True))


def test_source_confirmation_continues_once_but_selection_only_opens_picker():
    result = run_js(f"""
      global.window = {{}};
      eval({_read('js/screens-guided-pi-data-binding.js')!r});
      let session = {{session_id:'s'}};
      const calls = [];
      const owner = window.EU_GUIDED_PI_DATA_BINDING.create({{
        session: () => session, busy: () => false, projectId: () => 'p',
        api: () => ({{authorizePiCopilotDataSource: async (id, request) => {{
          calls.push(request.action);
          return {{session: {{session_id:id, data_source_authorization: {{status:
            request.action === 'use_study_required_data' ? 'confirmed' : 'selection_in_progress'}}}}}};
        }}}}),
        setSession: value => session = value, rememberSession: () => {{}},
        setError: value => {{if(value) throw Error(value);}}, errorText: String,
        render: () => {{}}, loadWorkflow: async () => calls.push('workflow'),
        continueAfterDataSourceConfirmation: async () => calls.push('continue'),
      }});
      (async () => {{
        await owner.authorizeDataSource('use_study_required_data');
        await owner.authorizeDataSource('begin_local_selection');
        process.stdout.write(JSON.stringify(calls));
      }})();
    """)
    assert result == ['use_study_required_data', 'workflow', 'continue', 'begin_local_selection']


def test_resource_cards_preserve_coordinates_and_escape_labels():
    result = run_js(f"""
      global.window = {{EU_LANG:'zh'}};
      eval({_read('js/html-escape.js')!r});
      eval({_read('js/screens-guided-pi-resources.js')!r});
      const owner = window.EU_GUIDED_PI_RESOURCES.create({{esc:window.EU_HTML.esc}});
      const row = {{resources:[
        {{kind:'research_artifact', run_id:'run_safe', artifact:'result_tables.json',
          sha256:'abc', label:'打开结果表 <img src=x onerror=bad()>', media_type:'application/json'}},
        {{kind:'research_artifact', run_id:'run_safe', artifact:'evidence_ledger.json', label:'证据台账'}},
      ]}};
      const before = JSON.stringify(row);
      const html = owner.renderForMessage(row);
      process.stdout.write(JSON.stringify({{html, unchanged: before === JSON.stringify(row)}}));
    """)
    assert result['unchanged']
    html = result['html']
    assert 'gpi-file-card' in html
    assert 'data-gpi-resource-run="run_safe"' in html
    assert 'data-gpi-resource-digest="abc"' in html
    assert '<img' not in html
    assert '<details class="gpi-resource-technical">' in html
    assert '交互预览' in html
