"""MIMIC-III's official demo uses the same preparation and binding owners."""

from easyicu.databases.profiles import get_database_profile
from easyicu.webserver.demo_source_contracts import allowed_source_ids, build_catalog, get_source
from easyicu.webserver.pi_copilot.source_recommendation import official_demos_for_database


SOURCE_ID = "mimic_iii_demo_v1_4"


def test_mimic_iii_release_is_allowlisted_with_pinned_provenance():
    source = get_source(SOURCE_ID)
    assert SOURCE_ID in allowed_source_ids()
    assert source.database == "mimic_demo"
    assert source.version == get_database_profile("mimic_demo").reference_release == "1.4"
    assert source.title == "MIMIC-III Clinical Database Demo"
    assert source.archive_filename == "mimic-iii-clinical-database-demo-1.4.zip"
    assert source.size_bytes == 11_350_216
    assert source.size_label == "10.8 MB"
    assert source.archive_sha256 == "f7f973de40324e84377226fe54d472e137f5351ba31563e4cef13bc6716b9012"
    assert source.download_url == (
        "https://physionet.org/static/published-projects/mimiciii-demo/"
        "mimic-iii-clinical-database-demo-1.4.zip"
    )
    assert source.landing_page == "https://physionet.org/content/mimiciii-demo/1.4/"
    assert source.citation_url == "https://doi.org/10.13026/C2HM2Q"
    assert source.license_url == "https://physionet.org/content/mimiciii-demo/view-license/1.4/"
    assert source.attribution == (
        "Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. 2019."
    )
    assert source.mirror_url is None
    assert source.size_bytes < source.max_download_bytes <= 32 * 1024**2
    assert 95 * 1024**2 < source.max_uncompressed_bytes <= 256 * 1024**2

    row = next(row for row in build_catalog(lambda _: {})["sources"] if row["id"] == SOURCE_ID)
    assert row["scope"] == {"summary": "100 patients, 136 ICU stays", "patients": 100, "icu_stays": "136"}
    assert row["download"]["preferred_transport"] == "physionet"
    assert row["download"]["fallback_transport"] is None
    assert row["provenance"]["license"]["short_name"] == "ODbL 1.0"


def test_mimic_iii_research_choice_retains_demo_identity():
    for database in ("mimic", "mimic_demo"):
        choices = official_demos_for_database({"sources": []}, database)
        assert len(choices) == 1
        assert choices[0]["source_id"] == SOURCE_ID
        assert choices[0]["database"] == "mimic_demo"
        assert choices[0]["version"] == "1.4"
        assert choices[0]["research_scope"] == "demo_only"


def test_prepare_route_accepts_the_catalog_id_without_a_transport_client(monkeypatch):
    from types import SimpleNamespace
    from easyicu.webserver.routes import demo_sources as routes

    prepared = []
    monkeypatch.setattr(routes.demo_store, "make_prepare_runner", lambda source_id: prepared.append(source_id))
    monkeypatch.setattr(routes, "submit_job", lambda kind, runner: SimpleNamespace(id="demo-job", kind=kind, status="queued"))
    result = routes.jobs_demo_source_prepare({"source_id": SOURCE_ID})
    assert result == {"job_id": "demo-job", "kind": "demo-source-prepare", "status": "queued"}
    assert prepared == [SOURCE_ID]
