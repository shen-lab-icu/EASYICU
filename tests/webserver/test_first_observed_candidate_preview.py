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
    scope = {"adapter_id": "eicu-demo-first-observed-single-hospital-adult/1",
             "purpose": "prepare_only", "analysis_authorized": False, "read_outcome_columns": False,
             "not_lifetime_first_icu": True,
             "source_table": "patient", "source_columns": ["patientunitstayid", "age"],
             "population_definition": "adult_first_observed_icu_in_single_visible_hospitalization",
             "multiple_hospitalization_policy": "exclude_patient", "criterion": criterion}
    monkeypatch.setattr(first_observed_preparation, "proposed_first_observed_scope", lambda **_: scope)
    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not read patient data")
    monkeypatch.setattr(first_observed_preparation, "derive_from_verified_eicu_demo", forbidden)
    return projection, scope


def test_first_observed_request_refuses_another_preparation_checkpoint(tmp_path, monkeypatch):
    _, scope = _setup(tmp_path, monkeypatch)
    before = (tmp_path / "sic_raw_preparation_checkpoint.json").read_bytes()
    with pytest.raises(ValueError, match="pending_conflict"):
        raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert not (tmp_path / "raw_preparation_checkpoint.json").exists()
    assert (tmp_path / "sic_raw_preparation_checkpoint.json").read_bytes() == before


def test_first_observed_request_is_idempotent_and_does_not_grant_analysis(tmp_path, monkeypatch):
    from easyicu.webserver import durable_research
    _setup(tmp_path, monkeypatch)
    (tmp_path / "sic_raw_preparation_checkpoint.json").unlink()
    monkeypatch.setattr(durable_research, "current_service", lambda: None)
    first = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    second = raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    assert first == second and first["status"] == "pending"
    assert not first["analysis_authorized"] and not first["approvable"]
    assert first["reason_code"] == "raw_preparation_durable_service_unavailable"
    checkpoint = raw_preparation_candidate.RawPreparationCheckpoint.model_validate_json(
        (tmp_path / "raw_preparation_checkpoint.json").read_bytes())
    assert checkpoint.state == "pending" and checkpoint.decision is None
    assert checkpoint.request.payload["first_observed_icu_preparation_scope"]["read_outcome_columns"] is False


def test_first_observed_offer_requires_available_guarded_service(tmp_path, monkeypatch):
    from easyicu.webserver import durable_research, study_contexts
    _setup(tmp_path, monkeypatch)
    (tmp_path / "sic_raw_preparation_checkpoint.json").unlink()
    raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    monkeypatch.setattr(study_contexts, "scientific_configuration_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(durable_research, "current_service", lambda: None)
    with pytest.raises(ValueError, match="durable_service_unavailable"):
        raw_preparation_candidate.available_host_review(run_id="run", study_context_id="study")
    service = SimpleNamespace(manager=SimpleNamespace(_journal=object()),
                              dispatcher=SimpleNamespace(_launched=True),
                              _require_handoff_deployment_lease=lambda: None)
    monkeypatch.setattr(durable_research, "current_service", lambda: service)
    assert raw_preparation_candidate.create_candidate_for_review(run_id="run", study_context_id="study")["approvable"]
    choice, checkpoint, scope, root = raw_preparation_candidate.available_host_review(
        run_id="run", study_context_id="study")
    assert root == tmp_path and checkpoint.state == "pending"
    assert scope["read_outcome_columns"] is False and choice.checkpoint_sha256 == checkpoint.checkpoint_sha256


def test_first_observed_review_ui_explains_private_preparation_and_no_analysis():
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = f"""
      global.window = {{ EU_HTML: {{ esc: value => String(value ?? '') }} }};
      eval({_read('js/screens-guided-pi-confirmation.js')!r});
      const scope={{adapter_id:'eicu-demo-first-observed-single-hospital-adult/1',purpose:'prepare_only',
        analysis_authorized:false,not_lifetime_first_icu:true,read_outcome_columns:false,
        population_definition:'adult_first_observed_icu_in_single_visible_hospitalization',
        source_columns:['patientunitstayid','uniquepid','patienthealthsystemstayid','unitvisitnumber','unittype','age']}};
      const preview={{schema_version:'easyicu.candidate-preparation-preview/1',complete:false,preparation_authorized:false,
        analysis_authorized:false,step_inputs:[],context_windows:[],blockers:['candidate_dependency_closure_unverified'],
        population_needs:[{{criterion_id:'c1',role:'include',kind:'first_observed_icu_single_hospital_adult',concepts:[]}},
          {{criterion_id:'c2',role:'exclude',kind:'diagnosis_codes',concepts:[],diagnosis_system:'icd10',diagnosis_codes:['K92.2']}},
          {{criterion_id:'c3',role:'include',kind:'measurement',concepts:['creat'],measurement_summary:'first',operator:'<=',value:4}}]}};
      const workflow={{next_action_code:'plan_scientific_changes_required',raw_preparation:{{status:'awaiting_consumption_closure',scope,candidate_preparation_preview:preview}},
        host_decisions:{{raw_preparation_review:{{scope_sha256:'a'.repeat(64)}}}}}};
      const owner=window.EU_GUIDED_PI_CONFIRMATION.create({{tr:(en,zh)=>zh,esc:v=>String(v??''),iconHtml:()=>'',resourceButton:()=>'',
        sessionIsStale:()=>false,busy:()=>false,workflow:()=>workflow,session:()=>({{session_id:'session',archived_child_jobs:[]}})}});
      const pending=owner.workflowConfirmationHtml();
      const message=owner.workflowConfirmation().message;
      workflow.raw_preparation={{status:'first_observed_complete',analysis_authorized:false}};
      const completed=owner.workflowConfirmationHtml();
      workflow.raw_preparation={{status:'first_observed_unavailable',analysis_authorized:false,recovery_session_id:'session'}};
      const resumable=owner.workflowConfirmationHtml();
      workflow.raw_preparation.recovery_session_id='another';
      const otherSession=owner.workflowConfirmationHtml();
      process.stdout.write(JSON.stringify({{pending,message,completed,resumable,otherSession}}));
    """
    result = json.loads(run_node(node, script, check=True).stdout)
    html = result['pending']
    assert "data-gpi-preparation-review=\"approved\"" in html
    assert "不读取结局" in html and "后续科研数据绑定尚未接通" in html
    assert "不能证明是终生首次 ICU" in html
    assert "成年患者首次可核实 ICU" in html and "纳入" in html and "排除" in html
    assert "诊断编码" in html and "K92.2" in html and "测量值阈值" in html
    assert "尚未完整确认计划的全部依赖" in html and "批准私有坐标准备" in html
    assert "单独审阅的首次 ICU 准备仅生成私有坐标" in result['message']
    assert "私有首次 ICU 坐标已准备完成" in result['completed']
    assert "data-gpi-preparation-review" not in result['completed']
    assert "继续原准备任务" in result['resumable'] and "data-gpi-preparation-review=\"approved\"" in result['resumable']
    assert "data-gpi-preparation-review" not in result['otherSession']


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


def test_status_refresh_only_reloads_workflow():
    from tests.support.node import run_node
    from tests.webserver.copilot.pi_copilot_static_fixtures import _read
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = 'global.window = {};\n' + _read('js/screens-guided-pi-events.js') + """
      let click, reloads=0, renders=0;
      const host={querySelector:()=>null,addEventListener:(name,fn)=>{if(name==='click')click=fn;}};
      const state={host,session:{session_id:'session',binding:{study_context_id:'study'}},workflow:{}};
      const owner=window.EasyICU.guidedPi.require('events').create({state,projectId:()=> 'project',
        tr:(en,zh)=>zh,render:()=>{renders++},loadWorkflow:async()=>{reloads++}});
      owner.wire();
      click({target:{closest:s=>s==='[data-gpi-preparation-status-refresh]' ? {} : null}});
      setImmediate(()=>process.stdout.write(JSON.stringify({reloads,renders})));
    """
    assert json.loads(run_node(node, script, check=True).stdout) == {"reloads": 1, "renders": 1}
