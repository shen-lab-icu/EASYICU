"""Scripted desktop acceptance runs only on the generated eICU-shaped source."""

from __future__ import annotations

import json
import socket
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE
from easyicu.research_agent.providers.clients import ReplayLLMClient
from easyicu.research_agent.providers.protocol import ReplayCassette, ReplayError
from easyicu.synthetic_desktop_source import ANALYSIS_INPUTS, PROVENANCE_DATABASE, source_spec
from tools import windows_laptop_acceptance as acceptance

ROOT = Path(__file__).resolve().parents[2]
CASSETTE = ROOT / "tests/fixtures/desktop_m1/scripted_cassette.json"
DISCLOSURE = "desktop first-result acceptance used scripted model responses, not a live model"


def test_cassette_uses_the_reviewed_response_table():
    table = json.loads(CASSETTE.with_name("scripted_responses.json").read_text(encoding="utf-8"))
    cassette = ReplayCassette.read(CASSETTE)
    assert cassette.model.model_dump(mode="json", exclude_defaults=True) == table["model"]
    assert [entry.model_dump(mode="json") for entry in cassette.entries] == [
        {**item, "response": canonical_json(item["response"])} for item in table["entries"]
    ]


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    from easyicu.research_agent.providers import factory
    from easyicu.webserver import demo_source_storage as storage

    def forbidden(*args, **kwargs):
        pytest.fail("scripted acceptance attempted a network, download or provider call")

    socket_init = socket.socket.__init__

    def offline_socket(sock, family=-1, *args, **kwargs):
        if family in {-1, socket.AF_INET, socket.AF_INET6}:
            forbidden()
        socket_init(sock, family, *args, **kwargs)

    monkeypatch.setattr(storage, "download_archive", forbidden)
    monkeypatch.setattr(factory, "build_provider_client", forbidden)
    monkeypatch.setattr(socket.socket, "__init__", offline_socket)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def run_driver(tmp_path, monkeypatch):
    from easyicu.webserver import catalog, cohort_review, patient_drilldown, sources

    work = tmp_path / "EasyICU 数据"
    receipts = tmp_path / "receipts"
    monkeypatch.setattr(sources, "_CONFIG_PATH", work / "home" / "sources.json")

    class InProcessWebServer:
        """Keep the driver's read checks and exercise their owners in process."""

        def __init__(self, work_dir, *, desktop_backend=None):
            assert work_dir == work
            assert desktop_backend is None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def request(self, method, path, body):
            if path == "/api/catalog":
                return 200, catalog.build_catalog()
            owners = {
                "/api/cohort-review/summary": cohort_review.cohort_review_summary,
                "/api/patient-review/drilldown": patient_drilldown.patient_review_drilldown,
            }
            return 200, owners[path](body)

    monkeypatch.setattr(acceptance, "LocalWebServer", InProcessWebServer)

    def run(cassette=CASSETTE):
        if not cassette.is_relative_to(ROOT):
            # Keep the normal CLI repository-path check, with a temporary
            # checkout root for the deliberately incomplete cassette.
            monkeypatch.setattr(acceptance, "__file__", str(tmp_path / "tools/windows_laptop_acceptance.py"))
        root = ROOT if cassette.is_relative_to(ROOT) else tmp_path
        code = acceptance.main([
            "--desktop-first-result", "--synthetic-source", "--model-cassette", str(cassette.relative_to(root)),
            "--memory-pressure-gb", "0", "--work-dir", str(work), "--output-dir", str(receipts),
        ])
        receipt = json.loads((receipts / "acceptance.json").read_text(encoding="utf-8"))
        summary = (receipts / "summary.md").read_text(encoding="utf-8")
        assert receipt["model_mode"] == "scripted_replay"
        assert receipt["model_disclosure"] == DISCLOSURE
        assert DISCLOSURE in summary and "`model_mode=scripted_replay`" in summary
        assert receipt["failures"] == []
        return code, receipt

    return run


@pytest.mark.slow
@pytest.mark.integration
def test_exact_scripted_cassette_passes_through_the_driver(run_driver):
    cassette = ReplayCassette.read(CASSETTE)
    code, receipt = run_driver()
    assert code == 0 and receipt["status"] == "passed"
    assert len(receipt["datasets"]) == 1
    assert {name: phase["status"] for name, phase in receipt["datasets"][0]["phases"].items()} == {
        "download_extract": "passed", "convert": "passed", "export": "passed", "register": "passed",
    }
    assert {record["read"] for record in receipt["web"]["reads"]} == {"catalog", "dictionary", "cohort_1", "patient_1"}
    research = receipt["research"]
    assert research["source_validated"] is True and research["runner_kind"] == "host"
    assert research["provider_kind"] == "replay" and research["provider_calls"] == 0
    result = research["first_result"]
    assert result["reason_code"] == "public_demo_replay_result_verified"
    assert result["evidence_bound_result"] is True and result["verified_primary_artifacts"] > 0
    assert result["budget_met"] is True
    assert result["profile_ref"] == PUBLIC_DEMO_ACCEPTANCE.ref
    assert result["model_identity"] == cassette.model.model_dump(mode="json")
    assert result["cassette_sha256"] == cassette.cassette_sha256
    assert result["source"] == cassette.source.model_dump(mode="json")
    assert result["source"]["database"] == PROVENANCE_DATABASE
    assert result["source"]["archive_sha256"] == source_spec().archive_sha256
    assert result["reportable"] is result["publishable"] is False


@pytest.mark.slow
@pytest.mark.integration
def test_a_changed_cohort_cannot_reuse_the_scripted_approval(run_driver, monkeypatch):
    from easyicu.research_agent.cohort import materializer

    original = materializer._materialize_cohort_with_metadata

    def changed(**kwargs):
        frame, provenance, metadata = original(**kwargs)
        # Keep the marginal and joint counts identical while attaching age to
        # different invented stays. Exact approval must still refuse it.
        _, group = next(iter(frame.groupby(["sex", "death"])))
        first = group.index[0]
        second = group.index[group["age"] != group.loc[first, "age"]][0]
        frame.loc[[first, second], "age"] = frame.loc[[second, first], "age"].to_numpy()
        return frame, provenance, metadata

    monkeypatch.setattr(materializer, "_materialize_cohort_with_metadata", changed)
    code, receipt = run_driver()
    assert code == 2 and receipt["status"] == "blocked"
    result = receipt["research"]["first_result"]
    assert result["evidence_bound_result"] is False
    assert result["reason_code"] == "model_replay_review_missing"


@pytest.mark.slow
@pytest.mark.integration
def test_a_missing_scripted_request_stops_without_provider_fallback(run_driver, tmp_path):
    cassette = ReplayCassette.read(CASSETTE)
    missing = tmp_path / "missing.json"
    ReplayCassette.create(source=cassette.source, model=cassette.model, approvals=cassette.approvals).write(missing)
    code, receipt = run_driver(missing)
    assert code == 2 and receipt["status"] == "blocked"
    research = receipt["research"]
    result = research["first_result"]
    assert result["reason_code"] == "model_replay_request_missing"
    assert result["missing_request_sha256"] in {entry.request_sha256 for entry in cassette.entries}
    assert result["evidence_bound_result"] is False and research["provider_calls"] == 0


def test_scripted_cassette_cannot_be_consumed_in_a_live_lane(tmp_path, monkeypatch):
    from easyicu.webserver import demo_source_storage as storage

    cassette = ReplayCassette.read(CASSETTE)
    monkeypatch.delenv("EASYICU_DEPLOYMENT", raising=False)
    with pytest.raises(ReplayError, match="model_replay_scope_required"):
        ReplayLLMClient(cassette)
    with pytest.raises(ReplayError, match="model_replay_profile_forbidden"):
        with storage.public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                              profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
            pytest.fail("scripted acceptance scope opened outside desktop deployment")
    monkeypatch.setattr(storage, "egress_source_database", lambda path, database: database)
    with storage.public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                          profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref) as public:
        # Even a re-sealed cassette claiming an ordinary demo source cannot
        # give that lane a scripted provider.
        relabeled = ReplayCassette.create(source=public.source, model=cassette.model, entries=cassette.entries)
        with pytest.raises(ReplayError, match="model_replay_provider_mismatch"):
            ReplayLLMClient(relabeled)


def test_scripted_replay_refuses_a_live_provider_identity(tmp_path, monkeypatch):
    from easyicu.webserver import demo_source_storage as storage

    cassette = ReplayCassette.read(CASSETTE)
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setattr(storage, "_verified_synthetic_replay_source", lambda path, database: source_spec())
    monkeypatch.setattr(storage, "egress_source_database", lambda path, database: database)
    relabeled = ReplayCassette.create(
        source=cassette.source,
        model=cassette.model.model_copy(update={"provider": "openai"}),
        entries=cassette.entries,
        approvals=cassette.approvals,
    )
    with storage.public_demo_replay_scope(export_path=tmp_path, database="eicu_demo",
                                          profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        with pytest.raises(ReplayError, match="model_replay_provider_mismatch"):
            ReplayLLMClient(relabeled)


@pytest.fixture
def scripted_source(tmp_path, monkeypatch):
    from easyicu.webserver import demo_source_storage as storage

    export = tmp_path / "invented-stays"
    export.mkdir()
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    monkeypatch.setattr(storage, "_verified_synthetic_replay_source", lambda path, database: source_spec())
    monkeypatch.setattr(storage, "egress_source_database", lambda path, database: database)
    return {"export_path": export, "database": source_spec().database}


def test_blocked_replay_refuses_to_reuse_its_first_stay_coordinate(tmp_path, monkeypatch, scripted_source):
    from easyicu.research_agent import facade
    from easyicu.research_agent.acquisition import catalog
    from easyicu.research_agent.cohort import materializer
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    work = tmp_path / "interrupted-acceptance"
    options = {**scripted_source, **ANALYSIS_INPUTS, "work_dir": work}
    monkeypatch.setattr(materializer, "materialize_to_parquet", lambda output, **kwargs: {
        "parquet": output / "cohort.parquet",
    })
    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: SimpleNamespace(concepts=[]))
    first = facade.replay_public_demo(cassette_path=CASSETTE, synthetic_source=True, **options)
    assert first["reason_code"] == "model_replay_baseline_endpoint_missing"
    coordinate = work / "synthetic_first_stay.parquet"
    original = coordinate.read_bytes()
    receipt_path = work / "model_replay_receipt.json"
    original_receipt = receipt_path.read_bytes()
    assert not (work / "runs").exists()

    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        client = ReplayLLMClient(ReplayCassette.read(CASSETTE))
        with pytest.raises(ReplayError, match="model_replay_workdir_not_empty"):
            facade.run_public_demo(client=client, review=None, **options)
    assert receipt_path.read_bytes() == original_receipt

    second = facade.replay_public_demo(cassette_path=CASSETTE, synthetic_source=True, **options)
    assert second["status"] == "blocked" and second["reason_code"] == "model_replay_workdir_not_empty"
    assert receipt_path.read_bytes() == original_receipt
    assert second["model_mode"] == "scripted_replay"
    assert second["model_disclosure"] == DISCLOSURE
    assert coordinate.read_bytes() == original


@pytest.mark.parametrize("artifact", ["materialized", "runs"])
def test_replay_refuses_each_preexisting_output_directory(tmp_path, scripted_source, artifact):
    from easyicu.research_agent.facade import replay_public_demo

    work = tmp_path / "interrupted-analysis"
    (work / artifact).mkdir(parents=True)
    note = work / artifact / "invented-progress.txt"
    note.write_text("The synthetic export was interrupted.", encoding="utf-8")
    result = replay_public_demo(
        cassette_path=CASSETTE, work_dir=work, synthetic_source=True, **scripted_source, **ANALYSIS_INPUTS,
    )
    assert result["status"] == "blocked" and result["reason_code"] == "model_replay_workdir_not_empty"
    assert not (work / "synthetic_first_stay.parquet").exists()
    assert note.read_text(encoding="utf-8") == "The synthetic export was interrupted."


def test_replay_refuses_a_standalone_first_stay_coordinate(tmp_path, monkeypatch, scripted_source):
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.facade import replay_public_demo

    work = tmp_path / "coordinate-only"
    work.mkdir()
    coordinate = work / "synthetic_first_stay.parquet"
    original = b"unfinished coordinate for invented stays"
    coordinate.write_bytes(original)
    materialize = Mock(side_effect=AssertionError("the stale coordinate must stop materialization"))
    monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)

    result = replay_public_demo(
        cassette_path=CASSETTE, work_dir=work, synthetic_source=True, **scripted_source, **ANALYSIS_INPUTS,
    )
    assert result["reason_code"] == "model_replay_workdir_not_empty"
    assert coordinate.read_bytes() == original
    assert not (work / "materialized").exists()
    assert not (work / "model_replay_receipt.json").exists()
    materialize.assert_not_called()


@pytest.mark.parametrize(("field", "value"), [
    ("question", "Is admission weight related to the length of an invented hospital stay?"),
    ("primary_exposure", "admissionweight"),
    ("target_outcome", "dischargeweight"),
    ("covariates", ()),
    ("covariates", ("admissionweight",)),
])
def test_scripted_replay_refuses_other_analysis_inputs_before_materialization(
    tmp_path, monkeypatch, scripted_source, field, value,
):
    from easyicu.research_agent import facade
    from easyicu.research_agent.cohort import materializer
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    def materialize(*args, **kwargs):
        raise AssertionError("unexpected materialization for an unreviewed synthetic analysis")

    monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)
    direct_work = tmp_path / "direct-different-analysis"
    cassette_work = tmp_path / "cassette-different-analysis"
    options = {**scripted_source, **ANALYSIS_INPUTS, field: value}
    reason = "model_replay_scripted_inputs_mismatch"
    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        client = ReplayLLMClient(ReplayCassette.read(CASSETTE))
        with pytest.raises(ReplayError, match=reason):
            facade.run_public_demo(client=client, review=None, work_dir=direct_work, **options)
    result = facade.replay_public_demo(cassette_path=CASSETTE, synthetic_source=True, work_dir=cassette_work, **options)
    assert result["status"] == "blocked" and result["reason_code"] == reason
    for work in (direct_work, cassette_work):
        receipt = work / "model_replay_receipt.json"
        assert set(work.iterdir()) == {receipt, work / "materialized"}
        assert list((work / "materialized").iterdir()) == []
        assert json.loads(receipt.read_text(encoding="utf-8")) == result


def test_scripted_run_resolves_scope_through_a_registered_wrapper(tmp_path, monkeypatch, scripted_source):
    from easyicu.research_agent import facade
    from easyicu.research_agent.acquisition import catalog
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.providers.llm import FallbackLLMClient
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    work = tmp_path / "wrapped-acceptance"
    options = {**scripted_source, **ANALYSIS_INPUTS, "work_dir": work}
    monkeypatch.setattr(materializer, "materialize_to_parquet", lambda output, **kwargs: {
        "parquet": output / "cohort.parquet",
    })
    monkeypatch.setattr(catalog, "build_available_catalog", lambda path: SimpleNamespace(concepts=[]))
    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        client = FallbackLLMClient(ReplayLLMClient(ReplayCassette.read(CASSETTE)), name="invented transport wrapper")
        with pytest.raises(ReplayError, match="model_replay_source_mismatch"):
            facade.run_public_demo(client=client, review=None, **{**options, "export_path": tmp_path})
        assert not work.exists()
        with pytest.raises(ReplayError, match="model_replay_baseline_endpoint_missing"):
            facade.run_public_demo(client=client, review=None, **options)
        assert (work / "synthetic_first_stay.parquet").is_file()
    with pytest.raises(ReplayError, match="model_replay_scope_required"):
        facade.run_public_demo(client=client, review=None, **options)


@pytest.mark.parametrize("kind", ["live-unverified-export", "live-credentialed-database", "offline"])
def test_public_demo_run_refuses_non_replay_clients(tmp_path, monkeypatch, kind):
    from easyicu.research_agent import facade, pipeline
    from easyicu.research_agent.cohort import materializer
    from easyicu.research_agent.providers.clients import OpenAIClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver import demo_source_storage as storage

    verified = tmp_path / "verified-demo"
    verified.mkdir()
    export = tmp_path / "unverified-private-export" if kind == "live-unverified-export" else verified
    export.mkdir(exist_ok=True)
    database = "mimiciv" if kind == "live-credentialed-database" else "eicu_demo"
    work = tmp_path / "ordinary-acceptance"
    monkeypatch.setattr(storage, "_verified_public_demo", lambda path: (
        SimpleNamespace(database="eicu_demo") if Path(path) == verified else None
    ))
    materialize = Mock(side_effect=AssertionError("unexpected materialization of invented stays"))
    dispatch = Mock(side_effect=AssertionError("unexpected provider dispatch"))
    monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)
    monkeypatch.setattr(pipeline, "ResearchAgentPipeline", dispatch)
    live = kind.startswith("live-")
    with storage.public_demo_egress_scope(str(verified)) if live else nullcontext():
        client = (OpenAIClient(model="invented-model", api_key="unused", base_url="http://127.0.0.1:8000/v1")
                  if live else ScriptedMockLLMClient(["Count the invented squares."]))
        with pytest.raises(ReplayError, match="model_replay_client_required"):
            facade.run_public_demo(
                export_path=export, database=database, work_dir=work, client=client,
                question="Does admission weight vary with the duration of invented stays?",
                primary_exposure="admissionweight", target_outcome="dischargelength", covariates=(), review=None,
            )
    assert not work.exists()
    materialize.assert_not_called()
    dispatch.assert_not_called()
    if not live:
        assert client.calls == []


def test_scope_resolver_refuses_a_non_replay_primary(scripted_source):
    from easyicu.research_agent.providers.client_trust import require_credentialed_data_clearance
    from easyicu.research_agent.providers.llm import FallbackLLMClient
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    primary = ScriptedMockLLMClient(["Count the corners on an invented drawing."])
    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        replay = ReplayLLMClient(ReplayCassette.read(CASSETTE))
        client = FallbackLLMClient(primary, replay)
        with pytest.raises(ReplayError, match="model_replay_mixed_provider_forbidden"):
            require_credentialed_data_clearance(client, scripted_source["database"])
    assert primary.calls == []


def test_scope_resolver_refuses_different_replay_leaves_before_checking_active_scope(tmp_path, scripted_source):
    from easyicu.research_agent.providers.client_trust import require_credentialed_data_clearance
    from easyicu.research_agent.providers.llm import FallbackLLMClient
    from easyicu.research_agent.providers.protocol import ModelIdentity
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        scripted = ReplayLLMClient(ReplayCassette.read(CASSETTE))
    public_export = tmp_path / "second-invented-export"
    public_export.mkdir()
    with public_demo_replay_scope(export_path=public_export, database=scripted_source["database"],
                                  profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref) as scope:
        public = ReplayLLMClient(ReplayCassette.create(
            source=scope.source, model=ModelIdentity(provider="synthetic", model="invented polygon counter"),
        ))
    for leaves in ((public, scripted), (scripted, public)):
        with pytest.raises(ReplayError, match="model_replay_source_mismatch"):
            require_credentialed_data_clearance(FallbackLLMClient(*leaves), scripted_source["database"])


@pytest.mark.parametrize("entry", ["direct", "cassette"])
@pytest.mark.parametrize("shape", ["file", "broken-symlink"])
def test_replay_work_dir_must_be_a_directory(tmp_path, scripted_source, entry, shape):
    from easyicu.research_agent import facade
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    work = tmp_path / "invalid-run-location"
    missing = tmp_path / "missing-target"
    note = "This location belongs to another invented experiment."
    if shape == "file":
        work.write_text(note, encoding="utf-8")
    else:
        try:
            work.symlink_to(missing, target_is_directory=True)
        except (NotImplementedError, OSError):
            pytest.skip("creating directory symlinks is unavailable")

    options = {**scripted_source, **ANALYSIS_INPUTS, "work_dir": work}
    with pytest.raises(ReplayError, match="model_replay_workdir_not_directory"):
        if entry == "cassette":
            facade.replay_public_demo(cassette_path=CASSETTE, synthetic_source=True, **options)
        else:
            with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref,
                                          synthetic_source=True):
                client = ReplayLLMClient(ReplayCassette.read(CASSETTE))
                facade.run_public_demo(client=client, review=None, **options)
    if shape == "file":
        assert work.read_text(encoding="utf-8") == note
    else:
        assert work.readlink() == missing and not missing.exists()


def test_rejected_attempts_keep_a_completed_receipt(tmp_path, scripted_source):
    from easyicu.research_agent import facade
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    work = tmp_path / "completed-run"
    work.mkdir()
    receipt = work / "model_replay_receipt.json"
    original = b'{"status":"passed","note":"invented summary belongs to the original run"}\n'
    receipt.write_bytes(original)
    options = {**scripted_source, **ANALYSIS_INPUTS, "work_dir": work}

    second = facade.replay_public_demo(cassette_path=CASSETTE, synthetic_source=True, **{
        **options, "question": "Does the previous invented summary describe any outliers?",
    })
    assert receipt.read_bytes() == original
    assert second["reason_code"] == "model_replay_workdir_not_empty"
    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        client = ReplayLLMClient(ReplayCassette.read(CASSETTE))
        with pytest.raises(ReplayError, match="model_replay_workdir_not_empty"):
            facade.run_public_demo(client=client, review=None, **options)
    assert receipt.read_bytes() == original
    assert not (work / "materialized").exists()


def test_replay_contender_preserves_the_in_flight_receipt(tmp_path, monkeypatch, scripted_source):
    from easyicu.research_agent import facade
    from easyicu.research_agent.cohort import materializer

    work = tmp_path / "running-invented-study"
    receipt = work / "model_replay_receipt.json"
    options = dict(cassette_path=CASSETTE, work_dir=work, synthetic_source=True, **scripted_source, **ANALYSIS_INPUTS)

    def materialize(output, **kwargs):
        assert output.is_dir()
        original = receipt.read_bytes()
        assert "status" not in json.loads(original)
        contender = facade.replay_public_demo(**options)
        assert contender["reason_code"] == "model_replay_workdir_not_empty"
        assert receipt.read_bytes() == original
        raise ReplayError("invented_cohort_unavailable")

    monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)
    owner = facade.replay_public_demo(**options)
    assert owner["reason_code"] == "invented_cohort_unavailable"
    assert json.loads(receipt.read_bytes()) == owner


def test_direct_replay_persists_client_provenance_on_failure(tmp_path, monkeypatch, scripted_source):
    from easyicu.research_agent import facade
    from easyicu.research_agent.cohort import materializer
    from easyicu.webserver.demo_source_storage import public_demo_replay_scope

    work = tmp_path / "invented-cohort-replay"
    receipt = work / "model_replay_receipt.json"
    with public_demo_replay_scope(**scripted_source, profile_ref=PUBLIC_DEMO_ACCEPTANCE.ref, synthetic_source=True):
        client = ReplayLLMClient(ReplayCassette.read(CASSETTE))

        def materialize(output, **kwargs):
            assert json.loads(receipt.read_bytes()) == client.receipt()
            raise ReplayError("invented_stay_fixture_unavailable")

        monkeypatch.setattr(materializer, "materialize_to_parquet", materialize)
        with pytest.raises(ReplayError, match="invented_stay_fixture_unavailable"):
            facade.run_public_demo(client=client, work_dir=work, review=None, **scripted_source, **ANALYSIS_INPUTS)
    payload = json.loads(receipt.read_bytes())
    provenance = client.receipt()
    assert {key: payload[key] for key in provenance} == provenance
    assert payload["status"] == "blocked" and payload["reason_code"] == "invented_stay_fixture_unavailable"
