"""A combined review never borrows approval or egress from another database."""
from copy import deepcopy

import pytest

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, assess_study
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver.agent_review_recovery import _write
from easyicu.webserver.research_multi_contracts import (
    DatabaseTarget, EgressGrant, MultiDatabaseError, MultiDatabaseRequest, PreparedTarget, ReviewedRun,
)
from easyicu.webserver.research_multi_run import MultiDatabaseRun


class SyntheticOwner:
    def __init__(self):
        self.design_value = {"study": "synthetic", "outcome": "hospital_death", "population": "all_declared_rows"}
        self.prepared = {}
        self.started = []
        self.results = {}
        self.failures = {}
        self.hook = lambda _: None

    def design(self, request, job):
        return deepcopy(self.design_value)

    def verify_design(self, design):
        if design != self.design_value:
            raise MultiDatabaseError("multi_database_design_changed")

    def prepare(self, design, target, *, batch_id, job):
        decision = assess_study(research_capability_profile(target.database), StudyRequirements(
            family="association_study", outcome="hospital_death", concepts=("los_icu",)))
        digest = canonical_sha256(target.model_dump(mode="json"))
        target = target.model_copy(update={"source_id": "fixture_" + target.database})
        review = ReviewedRun(run_id="run_" + target.database, study_context_id="study_" + target.database,
            plan_sha256=digest, context_sha256=digest, scientific_review_sha256=digest,
            requests_sha256=digest, input_sha256=digest, configuration_sha256=digest)
        grant = EgressGrant(database=target.database, source_id=target.source_id, source_sha256=digest,
            provider_sha256="f" * 64, data_handling="on_device", clearance="provider_required",
            grants=("prepare_source", "review_plan", "execute_plan", "provider_egress"))
        value = PreparedTarget(target=target, decision=decision, source_sha256=digest,
            review=review if decision.status != "refusal" else None,
            egress=grant if decision.status != "refusal" else None,
            stop_code="study_capability_refused" if decision.status == "refusal" else None)
        self.prepared[target.database] = value
        return value

    def verify(self, design, target):
        if self.prepared[target.target.database] != target:
            raise MultiDatabaseError("multi_database_review_changed")

    def execute(self, target, *, reviewer, job, recovering):
        key = target.target.database
        if recovering and key in self.results:
            return self.results[key]
        self.started.append(key)
        self.hook(job)
        if job.is_cancel_requested():
            raise MultiDatabaseError("multi_database_cancelled")
        if key in self.failures:
            raise self.failures[key]
        result = dict(database=key, run_id=target.review.run_id, source_sha256=target.source_sha256,
            plan_sha256=target.review.plan_sha256, estimand_sha256="a" * 64, estimand_label="adjusted_association",
            unit="odds_ratio", population_n=24, estimate=1.5, ci_low=1.1, ci_high=2.0,
            evidence_id="synthetic_primary", evidence_sha256="b" * 64,
            substitution_codes=[row.reason_code for row in target.decision.substitutions])
        self.results[key] = result
        return result

    def result(self, target):
        return deepcopy(self.results[target.target.database])


@pytest.fixture
def batch(tmp_path):
    owner = SyntheticOwner()
    batch = MultiDatabaseRun(tmp_path / ("a" * 64), owner)
    batch.prepare(MultiDatabaseRequest(project_id="project", session_id="session", study_context_id="study",
        targets=tuple(DatabaseTarget(database=key) for key in ("miiv", "aumc", "hirid", "sic"))))
    return batch


def approve(batch):
    return batch.approve(card_sha256=batch.load()[1].digest, reviewer="synthetic-reviewer")


def test_same_study_request_has_an_executing_owner():
    from easyicu.webserver.research_multi_run import MultiDatabaseRun

    assert callable(MultiDatabaseRun.prepare)
    assert callable(MultiDatabaseRun.approve)
    assert callable(MultiDatabaseRun.execute)


def test_one_card_names_exact_adjusted_and_refused_targets(batch):
    card = batch.load()[1]
    assert [row.disposition for row in card.targets[:3]] == ["exact", "adjusted", "not_run"]
    assert card.targets[1].substitutions
    assert "icu_los_unavailable" in card.targets[2].reason_codes
    assert card.targets[2].review is card.targets[2].egress is None
    with pytest.raises(MultiDatabaseError, match="approval_required"):
        batch.execute()
    approve(batch)
    result = batch.execute()
    assert batch.owner.started == ["miiv", "aumc", "sic"]
    assert result["targets"]["hirid"]["state"] == "not_run"


@pytest.mark.parametrize("field", ["plan_sha256", "context_sha256", "scientific_review_sha256", "requests_sha256",
                                    "input_sha256", "configuration_sha256"])
def test_any_changed_per_run_binding_invalidates_combined_approval(batch, field):
    approve(batch)
    old = batch.owner.prepared["aumc"]
    batch.owner.prepared["aumc"] = old.model_copy(update={"review": old.review.model_copy(update={field: "c" * 64})})
    state = batch.execute()
    assert batch.owner.started == []
    assert state["targets"]["miiv"]["reason_code"] == "multi_database_review_changed"


@pytest.mark.parametrize("field,value", [("source_sha256", "d" * 64), ("provider_sha256", "e" * 64),
                                        ("data_handling", "unspecified"), ("clearance", "verified_public_or_synthetic")])
def test_changed_source_or_egress_needs_a_new_card(batch, field, value):
    old = batch.owner.prepared["aumc"]
    batch.owner.prepared["aumc"] = old.model_copy(update={"egress": old.egress.model_copy(update={field: value})})
    with pytest.raises(MultiDatabaseError, match="review_changed"):
        approve(batch)
    assert not (batch.root / "approval.json").exists()


def test_changed_design_cannot_use_an_old_approval(batch):
    approve(batch)
    batch.owner.design_value["outcome"] = "synthetic_other_outcome"
    with pytest.raises(MultiDatabaseError, match="design_changed"):
        batch.execute()
    assert batch.owner.started == []


def test_a_failure_preserves_its_code_and_other_results(batch):
    batch.owner.failures["aumc"] = MultiDatabaseError("synthetic_singular_model")
    approve(batch)
    result = batch.execute()
    assert result["targets"]["aumc"]["reason_code"] == "synthetic_singular_model"
    assert [result["targets"][key]["state"] for key in ("miiv", "sic")] == ["completed", "completed"]
    assert batch.owner.started == ["miiv", "aumc", "sic"]


def test_pending_cancellation_only_stops_that_database(batch):
    approve(batch)
    batch.cancel("aumc")
    result = batch.execute()
    assert batch.owner.started == ["miiv", "sic"]
    assert result["targets"]["aumc"]["reason_code"] == "multi_database_cancelled"


def test_mid_fit_cancellation_is_visible_to_the_single_run(batch):
    approve(batch)
    batch.owner.hook = lambda job: batch.cancel("aumc") if job.database == "aumc" else None
    result = batch.execute()
    assert result["targets"]["aumc"]["state"] == "cancelled"
    assert result["targets"]["sic"]["state"] == "completed"


def test_restart_reconciles_started_run_and_does_not_repeat_finished_fits(batch):
    approve(batch)
    def crash(job):
        if job.database == "aumc":
            raise KeyboardInterrupt("synthetic power loss")
    batch.owner.hook = crash
    with pytest.raises(KeyboardInterrupt):
        batch.execute()
    assert batch.state()["targets"]["miiv"]["state"] == "completed"
    assert batch.state()["targets"]["aumc"]["state"] == "running"
    batch.owner.hook = lambda _: None
    restarted = MultiDatabaseRun(batch.root, batch.owner)
    restarted.execute()
    assert batch.owner.started.count("miiv") == 1
    assert batch.owner.started.count("sic") == 1
    assert restarted.state()["targets"]["aumc"]["state"] == "completed"


def test_completed_run_after_crash_before_state_write_is_reconciled(batch):
    approve(batch)
    execute = batch.owner.execute
    def die_after_result(target, **kwargs):
        execute(target, **kwargs)
        raise KeyboardInterrupt("synthetic result persisted before power loss")
    batch.owner.execute = die_after_result
    with pytest.raises(KeyboardInterrupt):
        batch.execute()
    batch.owner.execute = execute
    MultiDatabaseRun(batch.root, batch.owner).execute()
    assert batch.owner.started == ["miiv", "aumc", "sic"]


def test_changed_card_and_changed_request_cannot_reuse_approval(batch):
    approve(batch)
    record, _, _ = batch.load()
    record["request"]["project_id"] = "other_project"
    _write(batch.root / "request.json", record)
    with pytest.raises(MultiDatabaseError, match="binding_changed"):
        batch.execute()


def test_wrong_approval_digest_never_starts_a_run(batch):
    with pytest.raises(MultiDatabaseError, match="approval_changed"):
        batch.approve(card_sha256="0" * 64, reviewer="synthetic-reviewer")
    assert batch.owner.started == []


def test_pending_original_plan_joins_the_same_card_once(tmp_path):
    owner = SyntheticOwner()
    original = owner.prepare({}, DatabaseTarget(database="miiv"), batch_id="x", job=None)
    owner.reference_target = lambda _: original
    batch = MultiDatabaseRun(tmp_path / ("a" * 64), owner)
    card = batch.prepare(MultiDatabaseRequest(project_id="p", session_id="s", study_context_id="study",
        targets=(DatabaseTarget(database="miiv"), DatabaseTarget(database="sic"))))
    assert [row.database for row in card.targets] == ["miiv", "sic"]
    approve(batch)
    batch.execute()
    assert owner.started == ["miiv", "sic"]


def test_pending_original_never_silently_replaces_a_different_selected_source(tmp_path):
    owner = SyntheticOwner()
    owner.reference_target = lambda _: owner.prepare({}, DatabaseTarget(database="miiv"), batch_id="x", job=None)
    batch = MultiDatabaseRun(tmp_path / ("a" * 64), owner)
    with pytest.raises(MultiDatabaseError, match="source_conflict"):
        batch.prepare(MultiDatabaseRequest(project_id="p", session_id="s", study_context_id="study",
            targets=(DatabaseTarget(database="miiv", source_id="different_source"),)))


def test_projection_failure_does_not_relabel_a_completed_single_run(batch):
    original = batch.owner.result
    def result(target):
        if target.target.database == "aumc":
            raise MultiDatabaseError("multi_database_primary_estimate_unavailable")
        return original(target)
    batch.owner.result = result
    approve(batch)
    state = batch.execute()
    assert state["targets"]["aumc"]["state"] == "completed"
    assert state["targets"]["aumc"]["reason_code"] == "multi_database_primary_estimate_unavailable"
    assert state["targets"]["sic"]["state"] == "completed"


def test_repeated_approval_and_execution_do_not_repeat_results(batch):
    approve(batch)
    batch.execute()
    approve(batch)
    batch.execute()
    assert batch.owner.started == ["miiv", "aumc", "sic"]


def test_combined_answer_has_its_own_deterministic_evidence(batch):
    import json
    from easyicu.webserver.research_multi_results import combined_answer
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.runtime_artifacts import verified_run_evidence_path

    approve(batch)
    batch.execute()
    first = combined_answer(batch)
    assert first["scope"] == "analysis_level"
    assert first["heterogeneity"]["description_code"] == "directions_agree"
    assert "pooled_estimate" not in str(first)
    assert combined_answer(batch) == first
    evidence = EvidenceStore(batch.root / "comparison").get(first["heterogeneity"]["evidence_id"])
    path = verified_run_evidence_path(batch.root / "comparison", evidence)
    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["card_sha256"] == first["card_sha256"]
    assert payload["inputs"] == [row["primary"] for row in first["targets"] if "primary" in row]
    assert payload["result"]["scope"] == "analysis_level"


def test_incompatible_or_unknown_units_do_not_produce_heterogeneity(batch):
    from easyicu.webserver.research_multi_results import combined_answer

    approve(batch)
    batch.execute()
    state = batch.state()
    for database, result in batch.owner.results.items():
        result["unit"] = "synthetic_unknown_unit"
        state["targets"][database]["result"] = result
    batch._save(state)
    assert combined_answer(batch)["heterogeneity"]["reason_code"] == "multi_database_estimates_incompatible"


def test_completed_reference_result_joins_a_single_target_comparison(batch):
    from easyicu.webserver.research_multi_results import combined_answer

    approve(batch)
    batch.execute()
    reference = {**batch.owner.results["miiv"], "database": "eicu", "run_id": "earlier",
                 "source_sha256": "e" * 64}
    batch.owner.reference_result = lambda _: reference
    result = combined_answer(batch)
    assert result["reference"]["primary"] == reference
    assert result["heterogeneity"]["n_databases"] == 4


def test_changed_primary_result_is_not_projected(batch):
    from easyicu.webserver.research_multi_results import combined_answer

    approve(batch)
    batch.execute()
    batch.owner.results["miiv"]["estimate"] = 1.8
    result = combined_answer(batch)
    assert result["targets"][0]["state"] == "completed"
    assert result["targets"][0]["primary_state"] == "unavailable"
    assert result["targets"][1]["state"] == "completed"


@pytest.mark.parametrize("field,value", [("estimand_sha256", "c" * 64), ("unit", "risk_ratio")])
def test_different_estimands_or_scales_stay_side_by_side_without_comparison(batch, field, value):
    from easyicu.webserver.research_multi_results import combined_answer

    approve(batch)
    batch.execute()
    batch.owner.results["aumc"][field] = value
    state = batch.state()
    state["targets"]["aumc"]["result"] = deepcopy(batch.owner.results["aumc"])
    batch._save(state)
    answer = combined_answer(batch)
    assert all(row["primary"] for row in answer["targets"] if row["state"] == "completed")
    assert answer["heterogeneity"] == {"reason_code": "multi_database_estimates_incompatible"}


def test_duplicate_database_request_cannot_silently_drop_a_source():
    with pytest.raises(ValueError, match="multi_database_duplicate_target"):
        MultiDatabaseRequest(project_id="p", session_id="s", study_context_id="study",
            targets=(DatabaseTarget(database="miiv", source_id="one"), DatabaseTarget(database="miiv", source_id="two")))


@pytest.mark.parametrize("change", ["database", "source", "decision", "refusal"])
def test_host_bindings_cannot_cross_targets(change):
    owner = SyntheticOwner()
    target = owner.prepare({}, DatabaseTarget(database="miiv"), batch_id="a" * 64, job=None)
    wire = target.model_dump(mode="json")
    if change == "database":
        wire["egress"]["database"] = "aumc"
    elif change == "source":
        wire["egress"]["source_id"] = "neighbour"
    elif change == "decision":
        wire["decision"]["database"] = "aumc"
    else:
        wire["decision"] = owner.prepare({}, DatabaseTarget(database="hirid"), batch_id="a" * 64, job=None).decision.model_dump(mode="json")
        wire["target"]["database"] = "hirid"
        wire["egress"]["database"] = "hirid"
    with pytest.raises(ValueError):
        PreparedTarget.model_validate(wire)
