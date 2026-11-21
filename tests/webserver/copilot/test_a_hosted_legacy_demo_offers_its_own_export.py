"""A demo sharing its full database's key is offered by its release (pilot2 P1, 54420219b).

The MIMIC-IV demo registers under ``miiv``, the full database's key. A hosted
card naming that demo had no offer to read, so it showed the administrator
hint although the demo's export was registered. The host now offers the
demo's own exports by their stored release identity, also when the full data
are registered, and binds only those; the full export is never a demo choice.
Only synthetic registry metadata and disposable StudyContexts are used here.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from easyicu.official_demo_sources import get_source
from easyicu.webserver import sources, study_contexts
from easyicu.webserver.pi_copilot.contracts import PiCopilotError, PiSessionRecord
from easyicu.webserver.pi_copilot.project_authority import ProjectAuthorityStore
from easyicu.webserver.pi_copilot.service import PiCopilotService
from tests.webserver.copilot.pi_copilot_contract_fixtures import FakeGateway

PROJECT = "project-legacy-demo-offer"
SESSION = "pi_legacy_demo_offer"
DEMO = get_source("mimic_iv_demo_v2_2")


def _row(row_id: str, *, stays: int, **identity) -> dict:
    return {
        "id": row_id, "database": "miiv", "ok": True,
        "path": f"/synthetic/private/{row_id}", "label": "internal-run-label",
        "generated": "2026-11-21T08:00:00Z", "modules": ["demographics", "outcome"],
        "summary": {"stays": stays}, **identity,
    }


FULL = _row("src_full", stays=90000)
RECEIPT = {"source_id": DEMO.id, "version": DEMO.version, "archive_sha256": DEMO.archive_sha256}
DEMO_ROW = _row("src_demo", stays=140, **RECEIPT)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    registry = {"active_path": None, "sources": []}
    monkeypatch.setattr(sources, "load_registry", lambda **_kwargs: copy.deepcopy(registry))
    service = PiCopilotService(
        store_path=tmp_path / "sessions.json", gateway=FakeGateway(),
        project_store=ProjectAuthorityStore(tmp_path / "projects.json"),
    )
    monkeypatch.setattr(service, "_latest_run_id", lambda *args, **kwargs: None)
    study = study_contexts.upsert_context({
        "id": "study-legacy-demo", "question": "Describe ICU stays in the MIMIC-IV demo.",
    }, active=True)
    service.project_store.bind(PROJECT, study["id"])
    service._save_record(PiSessionRecord(
        session_id=SESSION, project_id=PROJECT,
        binding=service._binding_for_context(study),
        data_source_authorization=service._new_session_data_authorization(study, agent_mode="research"),
    ))
    return service, registry


def _offer(service: PiCopilotService) -> dict:
    return service.get_registered_source_offer(SESSION, project_id=PROJECT, database="miiv")


def _use(service: PiCopilotService, source_id: str) -> dict:
    return service.authorize_data_source(
        SESSION, project_id=PROJECT, action="use_registered_source", database="miiv", source_id=source_id,
    )


def _demo_ids(offer: dict) -> set[str]:
    own = offer["official_demo"]["registered_offer"]
    return {row["source_id"] for row in ([own["recommended"]] if own["recommended"] else own["choices"])}


@pytest.mark.parametrize("with_full", (True, False))
def test_the_demo_export_is_offered_by_its_release_and_never_the_full_data(host, with_full):
    service, registry = host
    registry["sources"] = [*([FULL] if with_full else []), DEMO_ROW]

    offer = _offer(service)

    assert offer["official_demo"]["id"] == DEMO.id
    assert _demo_ids(offer) == {"src_demo"}
    assert offer["official_demo"]["registered_offer"]["recommended"]["stays"] == 140
    # The full database's own offer is unchanged: a demo never stands in for it.
    assert (offer["recommended"] or {}).get("source_id") == ("src_full" if with_full else None)


def test_the_demo_export_binds_as_the_researchers_choice(host):
    service, registry = host
    registry["sources"] = [FULL, DEMO_ROW]

    selected = _use(service, "src_demo")["session"]["data_source_authorization"]["source"]

    assert selected["database"] == "miiv"
    assert selected["stays"] == 140


@pytest.mark.parametrize("identity", (
    {"source_id": DEMO.id, "version": DEMO.version, "archive_sha256": "0" * 64},
    {"source_id": "another-release", "version": "9.9"},
    {},
))
def test_an_export_without_this_release_identity_is_never_offered_as_the_demo(host, identity):
    service, registry = host
    registry["sources"] = [FULL, _row("src_other", stays=140, **identity)]

    offer = _offer(service)

    assert _demo_ids(offer) == set()
    assert offer["recommended"]["source_id"] == "src_full"
    with pytest.raises(PiCopilotError) as error:
        _use(service, "src_other")
    assert error.value.code == "pi_data_source_not_offered"


def test_a_demo_with_its_own_key_keeps_its_own_offer(host):
    service, registry = host
    registry["sources"] = [{**FULL, "id": "src_eicu", "database": "eicu"}]

    offer = service.get_registered_source_offer(SESSION, project_id=PROJECT, database="eicu")

    assert offer["recommended"]["source_id"] == "src_eicu"
    assert "official_demo" not in offer
