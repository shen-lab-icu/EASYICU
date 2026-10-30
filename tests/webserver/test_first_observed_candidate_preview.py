from types import SimpleNamespace
import shutil
import json

import pytest

from easyicu.webserver import agent_pipeline_runs, first_observed_preparation, raw_preparation_candidate
from easyicu.webserver.pi_copilot import population_spec_review
from tests.webserver.test_sic_raw_preparation_grant import _case
from tests.webserver.copilot.pi_copilot_static_fixtures import _load_guided_pi_module_harness  # noqa: F401


def _setup(tmp_path, monkeypatch):
    result, _, _, checkpoint = _case(tmp_path, monkeypatch, approved=False)
    (tmp_path / "research_context.json").write_text(result.context.model_dump_json())
    pending = SimpleNamespace(run_id="run", run_dir=str(tmp_path), requests=(checkpoint.request,))
    monkeypatch.setattr(agent_pipeline_runs, "raw_preparation_review_owner", lambda *_: (pending, {"id": "study"}))
    authority = raw_preparation_candidate._plan_authority(pending)
    criterion = {"kind": "first_observed_icu_single_hospital_adult", "role": "include"}
    projection = {"plan_sha256": authority.plan_sha256, "criteria": [{"criterion": criterion}]}
    monkeypatch.setattr(population_spec_review, "reviewed_population_spec", lambda **_: projection)
    monkeypatch.setattr(raw_preparation_candidate.SourceIdentity, "from_record",
                        lambda *_: SimpleNamespace(database="eicu_demo"))
    scope = {"source_table": "patient", "source_columns": ["patientunitstayid", "age"],
             "population_definition": "adult_first_observed_icu_in_single_visible_hospitalization",
             "multiple_hospitalization_policy": "exclude_patient", "criterion": criterion}
    monkeypatch.setattr(first_observed_preparation, "proposed_first_observed_scope", lambda **_: scope)
    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not read patient data")
    monkeypatch.setattr(first_observed_preparation, "derive_from_verified_eicu_demo", forbidden)
    return projection, scope


def test_first_observed_preview_has_no_grant_or_checkpoint(tmp_path, monkeypatch):
    _, scope = _setup(tmp_path, monkeypatch)
    before = (tmp_path / "sic_raw_preparation_checkpoint.json").read_bytes()
    outcome = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    preview = outcome["first_observed_preparation_preview"]
    assert outcome["status"] == "blocked"
    assert preview["source_columns"] == scope["source_columns"]
    assert preview["not_lifetime_first_icu"] and not preview["preparation_authorized"]
    assert not preview["analysis_authorized"]
    assert not (tmp_path / "raw_preparation_checkpoint.json").exists()
    assert (tmp_path / "sic_raw_preparation_checkpoint.json").read_bytes() == before


def test_preview_fails_closed_on_plan_or_criterion_change(tmp_path, monkeypatch):
    projection, scope = _setup(tmp_path, monkeypatch)
    projection["plan_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="population_ambiguous"):
        raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    projection["plan_sha256"] = raw_preparation_candidate._plan_authority(
        agent_pipeline_runs.raw_preparation_review_owner("run", "study")[0]).plan_sha256
    scope["criterion"] = {"kind": "first_icu_stay", "role": "include"}
    with pytest.raises(ValueError, match="population_changed"):
        raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")


def test_preview_ui_reports_scope_and_has_no_approval_action():
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      const esc=v=>String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;');
      const workflow={{next_action_code:'plan_ready',
        raw_preparation_candidate_run_id:'run',raw_preparation_candidate_result:{{status:'blocked',
          first_observed_preparation_preview:{{population_definition:'adult_first_observed_icu_in_single_visible_hospitalization',
            preparation_authorized:false,analysis_authorized:false,source_columns:['age','<field>']}}}}}};
      const owner=window.EU_GUIDED_PI_CONFIRMATION.create({{tr:(en,zh)=>zh,esc,iconHtml:()=>'',resourceButton:()=>'',
        sessionIsStale:()=>false,busy:()=>false,workflow:()=>workflow,session:()=>({{archived_child_jobs:[]}})}});
      const visible=owner.workflowConfirmationHtml();
      workflow.next_action_code='starting';
      workflow.starting={{action_code:'prepare_analysis_data'}};
      const starting=owner.workflowConfirmationHtml();
      workflow.next_action_code='plan_ready';
      workflow.starting=null;
      const after=owner.workflowConfirmationHtml();
      delete workflow.raw_preparation_candidate_result;
      const refreshed=owner.workflowConfirmationHtml();
      process.stdout.write(JSON.stringify({{visible,starting,after,refreshed}}));
    """
    rendered = json.loads(run_node(node, script, check=True).stdout)
    html = rendered['visible']
    assert "首次可观察 ICU 数据准备预览" in html and "不能证明是终生首次" in html
    assert "暂不能准备或分析" in html and "&lt;field>" in html
    assert "data-gpi-preparation-review" not in html
    assert "正在准备" in rendered['starting']
    assert "首次可观察 ICU 数据准备预览" not in rendered['starting']
    assert "首次可观察 ICU 数据准备预览" in rendered['after']
    assert 'data-gpi-preparation-preview="run"' in rendered['refreshed']


def test_preview_click_forwards_complete_api_result():
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read

    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = 'global.window = {};\n' + _read('js/screens-guided-pi-events.js') + """
      let click;
      const host={querySelector:()=>null,addEventListener:(name,fn)=>{if(name==='click')click=fn;}};
      const state={host,session:{session_id:'session',binding:{study_context_id:'study'}},
        workflow:{raw_preparation_candidate_run_id:'run'}};
      const preview={source_table:'patient',source_columns:['patientunitstayid','age'],
        population_definition:'adult_first_observed_icu_in_single_visible_hospitalization',
        multiple_hospitalization_policy:'exclude_patient',not_lifetime_first_icu:true,
        preparation_authorized:false,analysis_authorized:false};
      const owner=window.EasyICU.guidedPi.require('events').create({state,projectId:()=> 'project',render:()=>{},
        api:()=>({previewRawPreparation:async()=>({status:'blocked',first_observed_preparation_preview:preview})}),
        loadWorkflow:async()=>{}});
      owner.wire();
      click({target:{closest:s=>s==='[data-gpi-preparation-preview]' ? {dataset:{gpiPreparationPreview:'run'}} : null}});
      setImmediate(()=>process.stdout.write(JSON.stringify({expected:preview,actual:state.workflow.raw_preparation_candidate_result})));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    assert result["actual"]["first_observed_preparation_preview"] == result["expected"]
