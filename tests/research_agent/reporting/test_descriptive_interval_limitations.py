"""The reviewed descriptive source ceiling survives the card and Writer path."""

import json
import shutil

import pytest

from easyicu.research_agent.authority.evidence_store import EvidenceEnforcementMode, EvidenceStore
from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
from easyicu.research_agent.reporting.manuscript_method_facts import place_manuscript_method_facts
from easyicu.research_agent.reporting.manuscript_post import bind_numeric_values
from easyicu.research_agent.reporting.plan_review_limitations import (
    PLAN_REVIEW_LIMITATION_UNMAPPED,
    audit_bound_plan_review_limitations,
    writer_plan_review_limitations,
)
from tests.research_agent.planning.descriptive_interval_fixtures import (
    descriptive_context,
    descriptive_plan,
)
from tests.support.node import run_node
from tests.webserver.copilot.pi_copilot_static_fixtures import _read


def review_store(tmp_path, review):
    path = tmp_path / "scientific_plan_review.json"
    path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    store = EvidenceStore(tmp_path, enforcement_mode=EvidenceEnforcementMode.STRICT)
    store.register_file(
        kind="log", source_path=path, description="Synthetic complete-plan review.",
        evidence_id="scientific_plan_review", producer="plan_scientific_review",
        generation_mode="deterministic_skill",
    )
    return store


@pytest.mark.parametrize("variance", ["model_based", "cluster_robust", "heteroskedasticity_robust"])
def test_the_disclosed_ceiling_is_bound_into_report_limitations(tmp_path, variance):
    context = descriptive_context(variance)
    review = build_plan_scientific_review(context=context, plan=descriptive_plan(context))
    store = review_store(tmp_path, review)
    limitations, failure = writer_plan_review_limitations(store)
    assert failure is None
    assert len(limitations) == 1
    limitation = limitations[0]
    assert limitation.code == "DESCRIPTIVE_INTERVALS_WITHHELD_WITHOUT_PATIENT_GROUPING"
    assert variance.replace("_", " ") in limitation.text
    assert "repeated ICU stays without verified patient grouping" in limitation.text
    assert "all stays were retained" in limitation.text
    assert "only counts and proportions" in limitation.text
    assert limitation.source_sha256 == store.get("scientific_plan_review").sha256
    scaffold, _ = place_manuscript_method_facts(
        "## Limitations\n\nThe source is observational.\n\n## Conclusion\n", limitations,
    )
    safe, removed = store.enforce_evidence_bound_scaffold(scaffold, per_step_records=[])
    assert not removed
    bound = store.bind_manuscript(safe, per_step_records=[])
    _, _, untraced = bind_numeric_values(bound, evidence=store, per_step_records=[])
    assert not untraced
    assert audit_bound_plan_review_limitations(bound, evidence=store, per_step_records=[]) is None
    assert audit_bound_plan_review_limitations(
        "## Limitations\n\nOmitted.\n", evidence=store, per_step_records=[],
    ) is not None


def test_the_review_card_shows_the_ceiling_and_declared_estimator():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    context = descriptive_context()
    review = build_plan_scientific_review(context=context, plan=descriptive_plan(context))
    script = f"""
      global.window = {{
        EU_HTML: {{
          esc: value => String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
          escAttr: value => String(value ?? ''),
        }},
        t: (en, zh) => en, icon: () => '',
      }};
      eval({json.dumps(_read('js/screens-agent-reader-vocab.js'))});
      eval({json.dumps(_read('js/screens-agent-render.js'))});
      process.stdout.write(window.AGENT_RENDER.artifactStructuredView(
        'scientific_plan_review.json', {review.model_dump_json()}
      ));
    """
    html = run_node(node, script, check=True).stdout
    assert "model based uncertainty estimator was withheld" in html
    assert "repeated ICU stays without verified patient grouping" in html
    assert "all stays were retained and only counts and proportions were reported" in html


def test_a_ceiling_finding_without_its_typed_reason_fails_closed(tmp_path):
    context = descriptive_context()
    review = build_plan_scientific_review(context=context, plan=descriptive_plan(context))
    review = review.model_copy(update={"findings": [
        finding.model_copy(update={"descriptive_interval_limitation": None})
        for finding in review.findings
    ]})
    limitations, failure = writer_plan_review_limitations(review_store(tmp_path, review))
    assert limitations == ()
    assert failure is not None
    assert failure.detail["reason_code"] == PLAN_REVIEW_LIMITATION_UNMAPPED
