"""Scripted desktop acceptance runs only on the generated eICU-shaped source."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from easyicu.research_agent.canonical_json import canonical_json
from easyicu.research_agent.orchestration.profiles import PUBLIC_DEMO_ACCEPTANCE
from easyicu.research_agent.providers.clients import ReplayLLMClient
from easyicu.research_agent.providers.protocol import ReplayCassette, ReplayError
from easyicu.synthetic_desktop_source import PROVENANCE_DATABASE, source_spec
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
