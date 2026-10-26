"""A later addition depends on immutable earlier results, not just their IDs."""
from types import SimpleNamespace

import pytest

from tests.support import pytest_state  # noqa: F401
from tests.support.result_append import completed_source
from easyicu.webserver import research_append_run as owner, study_contexts
from easyicu.webserver.research_append_actions import append_options
from easyicu.webserver.research_append_history import resolve_result_source
from easyicu.webserver.research_append_source import read_json, seal_completed_source


@pytest.fixture
def chain(tmp_path, monkeypatch, request):
    wrapper, run, _, _, baseline = completed_source(tmp_path)
    source = seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
        baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})
    prior = owner.propose_append_run(source=source, kind=getattr(request, "param", "subgroup"), wrapper=wrapper.parent / "run_prior",
                                    study={"id": "study"}, registered_source_changed=False)
    owner.approve_append_run(prior.wrapper, decision="approved", reviewer="synthetic")
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    return source, prior


def test_completed_addition_remains_a_source_after_its_review_deadline(chain, monkeypatch):
    from datetime import datetime, timedelta
    from easyicu.research_agent.orchestration import human_review_checkpoint as checkpoints

    source, prior = chain
    before = {str(path): path.read_bytes() for root in (source.wrapper, prior.wrapper)
              for path in root.rglob("*") if path.is_file()}
    deadline = datetime.fromisoformat(prior.checkpoint.expires_at) + timedelta(days=1)
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return deadline.astimezone(tz)
    monkeypatch.setattr(checkpoints, "datetime", Later)
    service = SimpleNamespace(_research_run_row=lambda *args: {"project_dir": str(prior.wrapper), "study_id": "study"})
    offered = append_options(service, project_id="project", run_id=prior.checkpoint.run_id)
    assert "sensitivity" in {row["kind"] for row in offered["options"] if row["available"]}
    resolved, predecessor = resolve_result_source(prior.wrapper, run_id=prior.checkpoint.run_id, study_id="study")
    child = owner.propose_append_run(source=resolved, kind="sensitivity", wrapper=source.wrapper.parent / "run_later",
                                     study={"id": "study"}, registered_source_changed=False, predecessor=predecessor)
    owner.approve_append_run(child.wrapper, decision="approved", reviewer="synthetic")
    assert owner.load_append_run(child.wrapper).checkpoint.state == "completed"
    assert {str(path): path.read_bytes() for root in (source.wrapper, prior.wrapper)
            for path in root.rglob("*") if path.is_file()} == before


def test_unapproved_addition_still_obeys_its_review_deadline(chain, monkeypatch):
    from datetime import datetime, timedelta
    from easyicu.research_agent.orchestration import human_review_checkpoint as checkpoints

    source, _ = chain
    child = owner.propose_append_run(source=source, kind="sensitivity", wrapper=source.wrapper.parent / "run_pending",
                                     study={"id": "study"}, registered_source_changed=False)
    deadline = datetime.fromisoformat(child.checkpoint.expires_at) + timedelta(days=1)
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return deadline.astimezone(tz)
    monkeypatch.setattr(checkpoints, "datetime", Later)
    with pytest.raises(checkpoints.HumanReviewCheckpointExpired):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="synthetic")
    assert not (child.run_dir / "steps").exists()


@pytest.mark.parametrize("damage", ["result", "evidence", "seal"])
def test_changed_earlier_addition_cannot_offer_another_append(chain, damage):
    _, prior = chain
    if damage == "result":
        path = prior.wrapper / "result_tables.json"
    elif damage == "evidence":
        from easyicu.research_agent.authority.evidence_store import EvidenceStore
        record = next(r for r in EvidenceStore(prior.run_dir).records() if r.produced_by_step == "append_subgroup")
        path = prior.run_dir / record.relative_path
    else:
        path = prior.wrapper / "result_append_completion.json"
    if damage == "seal":
        seal = read_json(path)
        seal["sha256"] = "a" * 64
        import json
        path.write_text(json.dumps(seal))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    service = SimpleNamespace(_research_run_row=lambda *args: {"project_dir": str(prior.wrapper), "study_id": "study"})
    result = append_options(service, project_id="project", run_id=prior.checkpoint.run_id)
    assert not any(option["available"] for option in result["options"])


def test_prior_result_is_checked_again_at_approval(chain):
    source, prior = chain
    _, predecessor = resolve_result_source(prior.wrapper, run_id=prior.checkpoint.run_id, study_id="study")
    child = owner.propose_append_run(source=source, kind="sensitivity", wrapper=source.wrapper.parent / "run_next",
        study={"id": "study"}, registered_source_changed=False, predecessor=predecessor)
    path = prior.wrapper / "result_tables.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="earlier addition's result changed"):
        owner.approve_append_run(child.wrapper, decision="approved", reviewer="synthetic")
    assert not (child.run_dir / "steps").exists()
    assert not read_json(child.wrapper / "source_run_manifest.json")["plan_approval_allowed"]


def test_completed_append_cannot_be_claimed_by_another_study(chain):
    _, prior = chain
    with pytest.raises(ValueError, match="another study"):
        resolve_result_source(prior.wrapper, run_id=prior.checkpoint.run_id, study_id="different-study")


@pytest.mark.parametrize("chain", ["article"], indirect=True)
def test_later_addition_keeps_the_earlier_article_figure(chain):
    source, prior = chain
    before = {str(p): p.read_bytes() for p in prior.wrapper.rglob('*') if p.is_file()}
    old = read_json(prior.wrapper / "figure_gallery.json")["figures"]
    assert old and old[0]["data_url"]
    _, predecessor = resolve_result_source(prior.wrapper, run_id=prior.checkpoint.run_id, study_id="study")
    child = owner.propose_append_run(source=source, kind="subgroup", wrapper=source.wrapper.parent / "run_next",
        study={"id": "study"}, registered_source_changed=False, predecessor=predecessor)
    owner.approve_append_run(child.wrapper, decision="approved", reviewer="synthetic")
    figures = read_json(child.wrapper / "figure_gallery.json")["figures"]
    assert len(figures) == 1 and figures[0]["data_url"] == old[0]["data_url"]
    assert {str(p): p.read_bytes() for p in prior.wrapper.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize("chain", ["article"], indirect=True)
def test_changed_figure_inventory_blocks_the_next_addition(chain):
    _, prior = chain
    path = prior.run_dir / "figure_gallery.json"
    path.write_bytes(path.read_bytes() + b" ")
    service = SimpleNamespace(_research_run_row=lambda *args: {"project_dir": str(prior.wrapper), "study_id": "study"})
    offered = append_options(service, project_id="project", run_id=prior.checkpoint.run_id)
    assert not any(row["available"] for row in offered["options"])
