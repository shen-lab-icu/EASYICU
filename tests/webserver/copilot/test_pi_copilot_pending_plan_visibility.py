"""A live candidate must be reviewed before unresolved setup is confirmed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot


STATIC = Path(__file__).parents[3] / "src/easyicu/webserver/static"


@pytest.mark.parametrize("matches", [True, False])
def test_candidate_review_or_regeneration_precedes_standalone_setup_choice(matches):
    study = {
        "id": "study-candidate", "revision": 2,
        "question": "How is the exposure associated with the outcome?",
        "data_source": {"database": "test", "path": "/test/source"},
    }
    digest = study_contexts.scientific_configuration_sha256(study) if matches else "a" * 64
    snapshot = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None,
        latest_run={
            "run_id": "candidate", "run_type": "full",
            "engine": "easyicu.research_agent.pipeline", "gate_status": "blocked",
            "run_status": "human_review_pending",
            "pending_review_reason_codes": ["plan_scientific_changes_required"],
            "scientific_configuration_sha256": digest,
            "artifact_names": ["agent_plan.json", "scientific_plan_review.json", "source_run_manifest.json"],
        },
        plan_review_authority={
            "run_id": "candidate", "resumable_here": True,
            "scientific_configuration_sha256": digest,
        },
    )
    assert "cohort_eligibility" in snapshot.missing_setup_fields
    assert snapshot.current_stage == "plan"
    assert snapshot.next_action_code == (
        "plan_scientific_changes_required" if matches else "plan_configuration_superseded"
    )
    assert snapshot.plan_execution_ready is False
    assert next(s for s in snapshot.stages if s.id == "analysis").status == "blocked"


@pytest.mark.parametrize("action,expected", [
    ("plan_scientific_changes_required", False),
    ("cohort_eligibility_confirmation_required", True),
])
def test_candidate_review_has_no_duplicate_preplan_cohort_card(action, expected):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    owner = (STATIC / "js/screens-guided-pi-cohort-eligibility.js").read_text()
    modules = (STATIC / "js/screens-guided-pi-modules.js").read_text()
    script = f"""
      global.window = {{ EU_LANG: 'zh' }};
      eval({json.dumps(modules)});
      eval({json.dumps(owner)});
      const host = {{
        tr: (en, zh) => zh || en, esc: x => String(x),
        session: () => ({{ cohort_eligibility_selection: {{
          present: true, stated: false, blocker_code: 'cohort_eligibility_confirmation_required',
          options: [{{ id: 'no_eligibility_filter' }}, {{ id: 'first_admission_only' }}],
        }} }}),
        workflow: () => ({{ next_action_code: {json.dumps(action)} }}),
        busy: () => false, sessionIsStale: () => false,
      }};
      const cohort = window.EasyICU.guidedPi.require('cohortEligibility');
      process.stdout.write(cohort.create(host).render());
    """
    result = subprocess.run([node, "--eval", script], capture_output=True, text=True, check=True, timeout=15)
    assert bool(result.stdout) is expected


def test_review_visibility_stays_in_its_widget_owner():
    owner = (STATIC / "js/screens-guided-pi-cohort-eligibility.js").read_text()
    assert "planNeedsThisDecision" in owner
    for name in ("app.js", "tweaks.js"):
        assert "planNeedsThisDecision" not in (STATIC / "js" / name).read_text()
