"""Primary ENTRY/PATH/RESUME gate; delta41s probes, synthetic tables only."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from easyicu import storage_seal_publication as publication
from easyicu.config import DataSourceConfig
from easyicu.datasource import ICUDataSource, _aumc_admissions_relation_sql
from easyicu.storage_layout import StorageIntegrityError, flat_table_scan


def _flat(root, table="sample", suffix="parquet", directory=False):
    path = root / table / "part.parquet" if directory else root / f"{table}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"itemid": [200], "valuenum": [71.0]})
    if suffix == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path


def _mark(install, refused=False, dangling=False):
    suffix = publication.REFUSAL_SUFFIX if refused else publication.PENDING_SUFFIX
    marker = install.parent / f".{install.name}{suffix}"
    if dangling:
        marker.symlink_to(install.parent / "missing_marker_target")
    else:
        marker.write_text("{}")
    return marker


def _source(root, database="miiv", table="sample", file="sample.parquet", **kwargs):
    config = DataSourceConfig(
        name=database, tables={table: {"files": [file]}}
    )
    return ICUDataSource(config, base_path=root, **kwargs)


def _read_spies(monkeypatch):
    spies = []
    for owner, names in (
        (pd, ("read_parquet", "read_csv")),
        (pq, ("read_table", "read_schema", "ParquetFile")),
        (duckdb, ("connect",)),
    ):
        for name in names:
            spy = Mock(wraps=getattr(owner, name))
            monkeypatch.setattr(owner, name, spy)
            spies.append(spy)
    return spies


def _at_start(factory, read, spies, reason=publication.UNPUBLISHED):
    started = []
    with pytest.raises(publication.SealPublicationError) as caught:
        source = factory()
        started.append(True)
        read(source)
    assert caught.value.reason == reason
    assert isinstance(caught.value, StorageIntegrityError)
    assert not started, "publication must refuse construction before a read is selected"
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize(
    "entry", ["csv", "hirid", "registered", "aumc_parquet", "aumc_csv"]
)
def test_entry_refuses_before_any_read_owner(tmp_path, monkeypatch, entry):
    root = tmp_path / "install"
    database = "miiv"
    registered = None
    table, file = "sample", "sample.parquet"
    if entry == "csv":
        _flat(root, suffix="csv")  # No conversion manifest.
        file = "sample.csv"

        def read(source):
            return source._load_raw_frame("sample", None)
    elif entry == "hirid":
        parts = _flat(root, table="observations", directory=True).parent
        database = "hirid"
        table, file = "observations", "observations/*.parquet"

        def read(source):
            return flat_table_scan(source, "observations", parts)
    elif entry == "registered":
        registered = {"sample": _flat(root)}

        # Registered Paths use scan_loader=None in the raw-frame owner.
        def read(source):
            return source._load_raw_frame("sample", None)
    else:
        suffix = "csv" if entry.endswith("csv") else "parquet"
        _flat(root, table="admissions", suffix=suffix)
        database = "aumc"
        table, file = "admissions", f"admissions.{suffix}"

        def read(source):
            expression = _aumc_admissions_relation_sql(source)
            with duckdb.connect() as connection:
                connection.execute("SELECT COUNT(*) FROM " + expression).fetchone()

    _mark(root)
    spies = _read_spies(monkeypatch)
    _at_start(
        lambda: _source(root, database, table, file, table_sources=registered),
        read,
        spies,
    )


def test_web_raw_cohort_reader_refuses_before_schema_or_table_read(
    tmp_path, monkeypatch
):
    from easyicu.webserver import dataio

    root = tmp_path / "install"
    _flat(root)
    _mark(root, refused=True)
    spies = _read_spies(monkeypatch)
    schema = Mock(wraps=dataio._read_columns)
    monkeypatch.setattr(dataio, "_read_columns", schema)
    _at_start(
        lambda: dataio._find_table_file(root, ["sample"]),
        lambda path: dataio._read_table_columns(path, ["valuenum"]),
        [*spies, schema],
        publication.REFUSED,
    )


@pytest.mark.parametrize(
    "layout", ["flat_farm", "icu", "middle", "file", "parent_link", "unused"]
)
def test_path_closure_covers_subroots_files_and_every_link_hop(
    tmp_path, monkeypatch, layout
):
    target = tmp_path / "target"
    table = _flat(target, directory=True)
    outer = tmp_path / "outer"
    outer.mkdir()
    marked = target
    refused = layout == "icu"
    if layout == "flat_farm":
        (outer / "sample").symlink_to(table.parent, target_is_directory=True)
    elif layout == "icu":
        (outer / "icu").symlink_to(target, target_is_directory=True)
    elif layout in {"middle", "file"}:
        middle = tmp_path / "middle"
        middle.mkdir()
        marked = middle
        source_target = table if layout == "file" else table.parent
        (middle / "link").symlink_to(source_target)
        (outer / "sample").symlink_to(Path("../middle/link"))
    elif layout == "parent_link":
        middle = tmp_path / "middle"
        middle.symlink_to(target, target_is_directory=True)
        marked = middle
        (outer / "sample").symlink_to(middle / "sample")
    else:
        _flat(outer)
        (outer / "unused_table").symlink_to(target, target_is_directory=True)
    _mark(marked, refused=refused, dangling=True)
    spies = _read_spies(monkeypatch)
    _at_start(
        lambda: _source(outer),
        lambda source: source._load_raw_frame("sample", None),
        spies,
        publication.REFUSED if refused else publication.UNPUBLISHED,
    )


def test_flat_file_link_refuses_the_actual_target_install_at_start(
    tmp_path, monkeypatch
):
    """delta45s FLAT-LINK: no virtual <table>_bucket can identify this file."""
    target = tmp_path / "target"
    table = _flat(target)
    farm = tmp_path / "farm"
    farm.mkdir()
    (farm / "sample.parquet").symlink_to(table)
    _mark(target, refused=True)
    spies = _read_spies(monkeypatch)
    _at_start(
        lambda: _source(farm),
        lambda source: source._load_raw_frame("sample", None),
        spies,
        publication.REFUSED,
    )


@pytest.mark.parametrize(
    "binding",
    [
        "derived_manifest", "linked_manifest", "cwd_manifest",
        "derived_files", "bucket_links", "cwd_bucket_links",
    ],
)
def test_provenance_targets_are_in_the_start_closure(tmp_path, monkeypatch, binding):
    target = tmp_path / "target"
    table = _flat(target)
    outer = tmp_path / "outer"
    _flat(outer)
    if binding in {"derived_manifest", "linked_manifest", "cwd_manifest"}:
        manifest = target / "conversion_manifest.json"
        manifest.write_text("{}")
        path = outer / "conversion_manifest.json"
        payload = {"derived_from": {"path": str(manifest), "sha256": "0" * 64}}
        if binding == "cwd_manifest":
            monkeypatch.chdir(tmp_path)
            payload["derived_from"]["path"] = "target/conversion_manifest.json"
        if binding == "linked_manifest":
            metadata = tmp_path / "metadata"
            metadata.mkdir()
            path.symlink_to(metadata / "provenance.json")
    elif binding == "derived_files":
        bucket = outer / "sample_bucket"
        bucket.mkdir()
        path = bucket / "_BUCKET_MANIFEST.json"
        payload = {"derived_from": {"table": "sample", "files": [{"path": str(table)}]}}
    else:
        path = outer / publication.IN_ROOT_RECEIPT
        payload = {"bucket_links": {"sample_bucket": str(target)}}
        if binding == "cwd_bucket_links":
            monkeypatch.chdir(tmp_path)
            payload["bucket_links"]["sample_bucket"] = "target"
    path.write_text(json.dumps(payload))
    _mark(target)
    spies = _read_spies(monkeypatch)
    _at_start(lambda: _source(outer), lambda source: source.load_table("sample"), spies)


@pytest.mark.parametrize("kind", ["files", "dataset", "registered", "register_later"])
def test_external_configured_paths_gate_even_without_base_path(
    tmp_path, monkeypatch, kind
):
    target = tmp_path / "target"
    table = _flat(target)
    table_config = {}
    registered = None
    if kind == "files":
        table_config = {"files": [str(table)]}
    elif kind == "dataset":
        table_config = {"dataset": {"path": str(target)}}
    elif kind == "registered":
        registered = {"sample": table}
    config = DataSourceConfig(name="miiv", tables={"sample": table_config})
    source = ICUDataSource(config) if kind == "register_later" else None
    _mark(target)
    spies = _read_spies(monkeypatch)
    if source is not None:
        with pytest.raises(publication.SealPublicationError):
            source.register_table_source("sample", table)
        assert "sample" not in source._table_sources
        assert all(spy.call_count == 0 for spy in spies)
    else:
        _at_start(
            lambda: ICUDataSource(config, table_sources=registered),
            lambda source: source._load_raw_frame("sample", None),
            spies,
        )


@pytest.mark.parametrize(
    "entry", ["extract", "cohort", "cached_api", "all_roots", "web_cohort"]
)
def test_run_entry_checks_unused_roots_before_cohort_or_worker(
    tmp_path, monkeypatch, entry
):
    from easyicu.api import concepts, extraction
    from easyicu.api import get_all_patient_ids
    from easyicu.webserver import dataio

    root = tmp_path / "install"
    _flat(root, table="icustays")
    unused = root / "unused"
    unused.mkdir()
    _mark(unused)
    reads = Mock(side_effect=AssertionError("read before start gate"))
    monkeypatch.setattr(extraction, "_get_all_patient_ids", reads)
    monkeypatch.setattr(concepts, "BaseICULoader", reads)
    spies = _read_spies(monkeypatch)
    if entry == "extract":

        def call():
            return extraction.extract_database(
                "miiv", root, modules=["vitals"], verbose=False
            )
    elif entry == "cohort":

        def call():
            return get_all_patient_ids(root, database="miiv")
    elif entry == "cached_api":

        def call():
            return concepts._get_global_loader("miiv", root)
    elif entry == "all_roots":
        healthy = tmp_path / "healthy"
        _flat(healthy)
        monkeypatch.setattr(extraction, "extract_database", reads)

        def call():
            return extraction.extract_all_databases(
                ["miiv", "aumc"],
                {"miiv": str(healthy), "aumc": str(root)},
                verbose=False,
            )
    else:
        api = Mock()
        api.get_all_patient_ids = reads

        def call():
            return dataio._resolve_export_cohort(str(root), "miiv", None, None, api)

    with pytest.raises(publication.SealPublicationError):
        call()
    assert reads.call_count == 0
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("default_path", [False, True])
def test_base_loader_gates_before_path_schema_or_readiness_checks(
    tmp_path, monkeypatch, default_path
):
    from easyicu.base import BaseICULoader

    root = tmp_path / "install"
    _flat(root, table="icustays")
    _mark(root)
    if default_path:
        monkeypatch.setattr(BaseICULoader, "_setup_data_path", lambda *args: root)
    readiness = Mock(side_effect=AssertionError("readiness before gate"))
    monkeypatch.setattr(BaseICULoader, "_ensure_data_ready", readiness)
    spies = _read_spies(monkeypatch)
    _at_start(
        lambda: BaseICULoader(
            database="miiv", data_path=None if default_path else root
        ),
        lambda loader: None,
        [*spies, readiness],
    )


def test_cached_api_default_loader_gates_its_bound_source(tmp_path, monkeypatch):
    from collections import OrderedDict
    from types import SimpleNamespace

    from easyicu.api import concepts

    root = tmp_path / "install"
    _flat(root)
    loader = SimpleNamespace(datasource=_source(root))
    key = ("miiv", None, None, frozenset())
    monkeypatch.setattr(concepts, "_loader_cache", OrderedDict([(key, loader)]))
    monkeypatch.setattr(concepts, "_global_loader", None)
    monkeypatch.setattr(concepts, "_loader_config", None)
    _mark(root, refused=True)
    spies = _read_spies(monkeypatch)
    _at_start(
        lambda: concepts._get_global_loader("miiv"),
        lambda loader: None,
        spies,
        publication.REFUSED,
    )


def _committed_ledger(tmp_path, monkeypatch):
    from easyicu.api import extraction

    root = tmp_path / "source"
    target = tmp_path / "target"
    table = _flat(target, directory=True)
    root.mkdir()
    (root / "sample").symlink_to(table.parent, target_is_directory=True)
    out = tmp_path / "output"
    out.mkdir()
    kwargs = {"database": "miiv", "data_path": str(root), "concepts": ["hr"]}
    # Isolate publication identity from the pre-existing code/data signatures.
    monkeypatch.setattr(extraction, "_stream_code_identity_sha256", lambda _: "code")
    monkeypatch.setattr(extraction, "_stream_source_identity_sha256", lambda _: "data")
    identity = extraction._stream_ledger_identity(
        "test_module",
        ["hr"],
        kwargs,
        "stay_id",
        extraction._stream_ids_digest("stay_id", [1]),
    )
    pd.DataFrame({"id": [1]}).to_parquet(out / ".test_module.stream_order.parquet")
    pd.DataFrame({"stay_id": [1], "charttime": [0.0], "hr": [71.0]}).to_parquet(
        out / "part.parquet"
    )
    ledger = {
        **identity,
        "initial_batch_size": 1,
        "current_batch_size": 1,
        "initial_planned_partition_count": 1,
        "patient_partition_strategy": "identifier_order",
        "batches": [
            {
                "stays": 1,
                "part": "part.parquet",
                "output_rows": 1,
                "batch_process_isolation": False,
                "loader_concepts": ["hr"],
            }
        ],
        "resume_events": [],
    }
    path = out / ".test_module.stream_ledger.json"
    path.write_text(json.dumps(ledger))
    return extraction, root, target, out, kwargs, path


@pytest.mark.parametrize("change", ["pending", "refused", "retarget", "legacy"])
def test_committed_resume_regates_and_refuses_changed_or_missing_start_digest(
    tmp_path,
    monkeypatch,
    change,
):
    extraction, root, target, out, kwargs, ledger_path = _committed_ledger(
        tmp_path, monkeypatch
    )
    if change in {"pending", "refused"}:
        _mark(target, refused=change == "refused")
    elif change == "retarget":
        replacement = _flat(tmp_path / "replacement", directory=True)
        (root / "sample").unlink()
        (root / "sample").symlink_to(replacement.parent, target_is_directory=True)
    else:
        ledger = json.loads(ledger_path.read_text())
        ledger.pop("seal_publication_closure_sha256", None)
        ledger_path.write_text(json.dumps(ledger))
    before = ledger_path.read_bytes()
    spies = _read_spies(monkeypatch)
    loader = Mock(side_effect=AssertionError("fully committed ledger must not reload"))
    monkeypatch.setattr(extraction, "_load_stream_module_batch", loader)
    with pytest.raises(publication.SealPublicationError) as caught:
        extraction._stream_module_batches_to_parquet(
            "test_module",
            ["hr"],
            kwargs,
            {"stay_id": [1]},
            1,
            str(out),
        )
    expected = {"pending": publication.UNPUBLISHED, "refused": publication.REFUSED}
    assert caught.value.reason == expected.get(change, publication.UNACCEPTED)
    assert ledger_path.read_bytes() == before
    assert (out / "part.parquet").is_file()
    assert not (out / "test_module.parquet").exists()
    assert loader.call_count == 0
    assert all(spy.call_count == 0 for spy in spies)


def test_committed_resume_with_original_closure_assembles_without_reloading(
    tmp_path, monkeypatch
):
    import easyicu.datasource as datasource

    extraction, root, target, out, kwargs, ledger_path = _committed_ledger(
        tmp_path, monkeypatch
    )
    ledger = json.loads(ledger_path.read_text())
    original = ledger["seal_publication_closure_sha256"]
    assert original == _source(root).seal_publication_closure["sha256"]
    gate = Mock(wraps=datasource.assert_roots_published)
    monkeypatch.setattr(datasource, "assert_roots_published", gate)
    loader = Mock(side_effect=AssertionError("committed batches must not reload"))
    monkeypatch.setattr(extraction, "_load_stream_module_batch", loader)
    info = extraction._stream_module_batches_to_parquet(
        "test_module", ["hr"], kwargs, {"stay_id": [1]}, 1, str(out)
    )
    assert gate.call_count == 1
    assert loader.call_count == 0
    assert Path(info["path"]).is_file()
    assert info["stream_resume_events"][0]["start_offset"] == 1
    assert not ledger_path.exists()


def test_cache_documents_accepted_install_is_immutable_and_late_refusal_is_nb(tmp_path):
    """B41-CACHE: late REFUSED is outside the seal flow; no cache fix here."""
    root = tmp_path / "install"
    _flat(root)
    source = _source(root)
    original = source._load_raw_frame("sample", None)
    identity = dict(source.seal_publication_closure)
    from easyicu.concept import ConceptDictionary, ConceptResolver

    resolver = ConceptResolver(ConceptDictionary({}))
    resolver._store_raw_concept_cache(
        "sample", None, original, aggregator=None, grid=None
    )
    pd.testing.assert_frame_equal(original, source._load_raw_frame("sample", None))
    pd.testing.assert_frame_equal(original, resolver.get_raw_concept("sample", source))
    assert source.seal_publication_closure == identity
    # Accepted installs never return to pending and are never changed in place.
    # REFUSED belongs only to NEW installs before acceptance. Simulate this
    # out-of-protocol event solely to document the existing read-time defense.
    _mark(root, refused=True)
    # Concept caches may still serve their immutable accepted input. This is
    # exactly the supervisor's NB boundary, not a cache invalidation fix.
    pd.testing.assert_frame_equal(original, resolver.get_raw_concept("sample", source))
    with pytest.raises(publication.SealPublicationError):
        source._load_raw_frame("sample", None)
    assert source.seal_publication_closure == identity


def test_closure_digest_is_sorted_reproducible_and_walk_opens_no_table(
    tmp_path, monkeypatch
):
    root = tmp_path / "install"
    table = _flat(root, directory=True)
    (root / "back").symlink_to(root, target_is_directory=True)
    (root / "one.parquet").symlink_to(table)
    (root / "two.parquet").symlink_to(table)
    markers = Mock(wraps=publication._marker)
    monkeypatch.setattr(publication, "_marker", markers)
    spies = _read_spies(monkeypatch)
    first = publication.assert_roots_published([root, table])
    # Dedupe marker probes by install, including shared flat FILE targets.
    probed = [(call.args[0], call.args[1]) for call in markers.call_args_list]
    assert len(probed) == len(set(probed))
    second = publication.assert_roots_published([table, root, root])
    directories = {root, table.parent, *root.parents}
    expected = json.dumps(
        [[str(p), False, False] for p in sorted(directories)], separators=(",", ":")
    )
    import hashlib

    assert first.sha256 == hashlib.sha256(expected.encode()).hexdigest()
    assert first == second
    assert first.marker_hits == 0
    assert first.link_hops == 3
    assert all(spy.call_count == 0 for spy in spies)


def test_linked_table_also_follows_its_parent_install_receipt(tmp_path, monkeypatch):
    root = tmp_path / "outer"
    root.mkdir()
    target = tmp_path / "target"
    table = _flat(target, directory=True)
    (root / "sample").symlink_to(table.parent, target_is_directory=True)
    refused = tmp_path / "refused"
    _flat(refused)
    (target / publication.IN_ROOT_RECEIPT).write_text(
        json.dumps({"bucket_links": {"unused_bucket": str(refused)}})
    )
    _mark(refused)
    spies = _read_spies(monkeypatch)
    _at_start(lambda: _source(root), lambda source: source.load_table("sample"), spies)


def test_native_manifest_carries_the_original_start_digest(tmp_path, monkeypatch):
    from easyicu.api import extraction

    monkeypatch.setattr(extraction, "EXTRACT_MODULES", {"demographics": ["age"]})
    pd.DataFrame({"stay_id": [1], "age": [63.0]}).to_parquet(
        tmp_path / "demographics.parquet",
        index=False,
    )
    closure = {
        "sha256": "a" * 64,
        "install_count": 20,
        "link_hops": 0,
        "marker_hits": 0,
    }
    extraction._publish_native_export_v2(
        database="miiv",
        data_path="/raw/source-must-not-be-read",
        output_dir=str(tmp_path),
        modules=["demographics"],
        max_patients=None,
        result={
            "modules": {"demographics": {"errors": []}},
            "seal_publication_closure": closure,
        },
    )
    manifest = json.loads((tmp_path / "_manifest.json").read_text())
    assert manifest["runtime_provenance"]["seal_publication_closure"] == closure


@pytest.mark.parametrize("refused", [False, True])
def test_missing_directory_checks_install_markers_before_returning_empty(
    tmp_path, refused
):
    from easyicu.storage_layout import enumerate_parquet_files

    install = tmp_path / "install"
    install.mkdir()
    _mark(install, refused=refused)
    with pytest.raises(publication.SealPublicationError) as caught:
        enumerate_parquet_files(install / "missing")
    assert caught.value.reason == (
        publication.REFUSED if refused else publication.UNPUBLISHED
    )


@pytest.mark.parametrize("subdir", ["icu", "hosp", "arbitrary/nested/tables"])
@pytest.mark.parametrize("refused", [False, True])
def test_nested_table_roots_refuse_the_true_install_before_construction_or_run(
    tmp_path, monkeypatch, subdir, refused
):
    from easyicu.api import extraction

    install = tmp_path / "install"
    root = install / subdir
    _flat(root, directory=True)
    (install / "conversion_manifest.json").write_text("{}")
    _mark(install, refused=refused)
    spies = _read_spies(monkeypatch)
    cohort = Mock(side_effect=AssertionError("cohort before gate"))
    monkeypatch.setattr(extraction, "_get_all_patient_ids", cohort)
    _at_start(
        lambda: _source(root), lambda source: source.load_table("sample"), spies,
        publication.REFUSED if refused else publication.UNPUBLISHED,
    )
    with pytest.raises(publication.SealPublicationError):
        extraction.extract_database("miiv", root, modules=["vitals"], verbose=False)
    assert cohort.call_count == 0
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("subdir", ["icu", "hosp", "arbitrary/nested/tables"])
def test_nested_read_time_checks_also_find_the_install_marker(tmp_path, subdir):
    from easyicu.storage_layout import enumerate_parquet_files, read_bucket_contract
    from tests.support.bucket_storage import write_contract

    install = tmp_path / "install"
    bucket = install / subdir / "sample_bucket"
    _flat(bucket, table="bucket_id=0", directory=True)
    write_contract(bucket, key="itemid")
    (install / publication.IN_ROOT_RECEIPT).write_text("{}")
    _mark(install)
    for call in (
        lambda: publication.assert_seal_published(bucket, formal=False),
        lambda: read_bucket_contract(bucket),
        lambda: enumerate_parquet_files(bucket),
    ):
        with pytest.raises(publication.SealPublicationError) as caught:
            call()
        assert caught.value.reason == publication.UNPUBLISHED


def test_scores_aux_reader_is_refused_at_datasource_construction(tmp_path, monkeypatch):
    from easyicu.scores.icu_mortality import _hirid_time

    install = tmp_path / "install"
    _flat(install, table="observations")
    _mark(install, refused=True)
    spies = _read_spies(monkeypatch)
    # If construction succeeded this auxiliary fallback could read the flat
    # observations file directly. The constructor must refuse first.
    _at_start(
        lambda: _source(install, "hirid", "observations", "observations.parquet"),
        lambda source: _hirid_time(install, pd.DataFrame(), source=source),
        spies, publication.REFUSED,
    )


def test_aumc_admission_time_owner_preserves_typed_integrity_error(tmp_path):
    from types import SimpleNamespace
    from easyicu.concept import ConceptDictionary, ConceptResolver

    expected = publication.SealPublicationError(publication.REFUSED, tmp_path)
    source = SimpleNamespace(load_table=Mock(side_effect=expected))
    resolver = ConceptResolver(ConceptDictionary({}))
    with pytest.raises(publication.SealPublicationError) as caught:
        resolver._aumc_admission_times(source, "admissionid")
    assert caught.value is expected
    assert source.load_table.call_count == 1


@pytest.mark.parametrize("entry", ["bootstrap_ids", "schema_detection"])
def test_bootstrap_readers_gate_before_any_schema_or_cohort_read(
    tmp_path, monkeypatch, entry
):
    from easyicu.databases.detection import peek_table_columns
    from easyicu.runtime.project_config import _load_ids_for_database

    root = tmp_path / "install"
    _flat(root, table="admissions")
    _mark(root)
    spies = _read_spies(monkeypatch)
    with pytest.raises(publication.SealPublicationError):
        if entry == "bootstrap_ids":
            _load_ids_for_database("aumc", root, 1)
        else:
            peek_table_columns(root, "admissions")
    assert all(spy.call_count == 0 for spy in spies)


def test_extract_gate_precedes_every_audited_read_owner(tmp_path, monkeypatch):
    from easyicu.api import extraction
    from easyicu.concept import ConceptResolver, eicu_io
    from easyicu.databases import detection
    from easyicu import datasource
    from easyicu.io import community_databases, parquet_reader
    from easyicu.io.data_env import SrcEnv
    from easyicu.runtime import project_config
    from easyicu.runtime.performance import TablePreloader
    from easyicu.scores import urine_output

    root = tmp_path / "install"
    _flat(root, table="icustays")
    _mark(root)
    spies = _read_spies(monkeypatch)
    for owner, name in (
        (datasource, "mimic_disk_columns"),
        (urine_output, "optional_availability_projection"),
        (eicu_io, "read_io_parquet"),
        (ICUDataSource, "_read_dataset"),
        (ConceptResolver, "_synthesise_los_column"),
        (datasource, "_aumc_admissions_relation_sql"),
        (datasource, "_sic_cases_relation_sql"),
        (community_databases, "ensure_community_stays"),
        (SrcEnv, "load_table"),
        (TablePreloader, "preload_for_patients"),
        (parquet_reader, "read_table_parquet"),
        (project_config, "_load_ids_for_database"),
        (detection, "peek_table_columns"),
        (extraction, "_get_all_patient_ids"),
    ):
        spy = Mock(side_effect=AssertionError(f"{name} before gate"))
        monkeypatch.setattr(owner, name, spy)
        spies.append(spy)
    with pytest.raises(publication.SealPublicationError):
        extraction.extract_database("miiv", root, modules=["vitals"], verbose=False)
    assert all(spy.call_count == 0 for spy in spies)


def _driver():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools/reextract_native_export_v2.py"
    spec = importlib.util.spec_from_file_location("slgate_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("resume", [False, True])
def test_driver_gates_before_cohort_or_completed_export_reuse(tmp_path, monkeypatch, resume):
    driver = _driver()
    target = tmp_path / "target"
    _flat(target)
    middle = tmp_path / "middle"
    middle.mkdir()
    (middle / "link").symlink_to(target, target_is_directory=True)
    farm = tmp_path / "farm"
    farm.symlink_to(middle / "link", target_is_directory=True)
    _mark(middle)
    out = tmp_path / "output"
    if resume:
        out.mkdir()
    args = driver._parse_args([
        "--output-root", str(out), "--databases", "miiv",
        "--data-path", f"miiv={farm}", *(["--resume"] if resume else []),
    ])
    monkeypatch.setattr(driver, "_git_identity", lambda: {})
    monkeypatch.setattr(driver, "_require_clean_identity", lambda _: None)
    monkeypatch.setattr(driver, "_load_psutil", lambda _: (None, {}))
    monkeypatch.setattr(driver, "_detect_effective_memory", lambda _: {})
    spies = _read_spies(monkeypatch)
    for name in ("_pending_databases", "_run_pending", "_load_resume_manifest"):
        spy = Mock(side_effect=AssertionError("driver read before gate"))
        monkeypatch.setattr(driver, name, spy)
        spies.append(spy)
    with pytest.raises(publication.SealPublicationError):
        driver.run(args)
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("change", ["marker", "retarget", "legacy"])
def test_driver_resume_regates_and_compares_original_closure(tmp_path, monkeypatch, change):
    driver = _driver()
    target = tmp_path / "target"
    table = _flat(target)
    root = tmp_path / "source"
    root.mkdir()
    link = root / "sample.parquet"
    link.symlink_to(table)
    paths = {"miiv": str(root)}
    args = driver._parse_args(["--output-root", str(tmp_path / "output")])
    identity = {"repository_root": str(driver.REPOSITORY_ROOT), "commit": "a" * 40}
    manifest = driver._new_run_manifest(
        databases=["miiv"], data_paths=paths, identity=identity,
        monitoring={}, memory={}, args=args,
    )
    # A base-compatible oracle records the start closure when the helper is
    # absent; the resumed helper itself still has to implement the gate.
    gate = getattr(driver, "_source_publication_closure", None)
    closure = gate(paths) if gate is not None else {"sha256": "original"}
    manifest["runtime_provenance"] = {"seal_publication_closure": closure}
    if change == "marker":
        _mark(target)
    elif change == "retarget":
        other = _flat(tmp_path / "other")
        link.unlink()
        link.symlink_to(other)
    else:
        manifest.pop("runtime_provenance")
    out = tmp_path / "output"
    out.mkdir()
    ledger = out / "run_manifest.json"
    ledger.write_text(json.dumps(manifest))
    before = ledger.read_bytes()
    spies = _read_spies(monkeypatch)
    with pytest.raises(publication.SealPublicationError):
        driver._load_resume_manifest(
            run_root=out, databases=["miiv"], data_paths=paths,
            identity=identity, resource_policy=args.resource_policy,
        )
    assert ledger.read_bytes() == before
    assert all(spy.call_count == 0 for spy in spies)


def test_driver_records_start_closure_and_accepts_an_unchanged_resume(tmp_path, monkeypatch):
    driver = _driver()
    source = tmp_path / "source"
    _flat(source)
    out = tmp_path / "output"
    args = driver._parse_args([
        "--output-root", str(out), "--databases", "miiv",
        "--data-path", f"miiv={source}",
    ])
    identity = {"repository_root": str(driver.REPOSITORY_ROOT), "commit": "a" * 40}
    monkeypatch.setattr(driver, "_git_identity", lambda: identity)
    monkeypatch.setattr(driver, "_require_clean_identity", lambda _: None)
    monkeypatch.setattr(driver, "_load_psutil", lambda _: (None, {"release_sealable": True}))
    monkeypatch.setattr(driver, "_detect_effective_memory", lambda _: {})
    # Model a completed database without running extraction or reading data.
    def completed(**kwargs):
        kwargs["manifest"]["sources"]["miiv"] = {"status": "complete"}
        return []

    monkeypatch.setattr(driver, "_pending_databases", completed)
    spies = _read_spies(monkeypatch)
    original = driver.run(args)
    closure = original["runtime_provenance"]["seal_publication_closure"]
    assert closure == driver._source_publication_closure({"miiv": str(source)})
    resumed = driver._load_resume_manifest(
        run_root=out, databases=["miiv"], data_paths={"miiv": str(source)},
        identity=identity, resource_policy=args.resource_policy,
    )
    assert resumed["runtime_provenance"]["seal_publication_closure"] == closure
    assert all(spy.call_count == 0 for spy in spies)


@pytest.mark.parametrize("entry", ["file", "table_dir", "subroot"])
def test_nested_link_walks_sibling_subroots_of_its_true_install(
    tmp_path, monkeypatch, entry
):
    """Entering icu/ must also visit hosp/ in the owner install's closure."""
    target = tmp_path / "target"
    table = _flat(target / "icu", directory=True)
    (target / "conversion_manifest.json").write_text("{}")
    refused = tmp_path / "refused"
    _flat(refused)
    hosp = target / "hosp"
    hosp.mkdir()
    (hosp / "unused").symlink_to(refused, target_is_directory=True)
    outer = tmp_path / "outer"
    outer.mkdir()
    if entry == "file":
        (outer / "sample.parquet").symlink_to(table)
    elif entry == "table_dir":
        (outer / "sample").symlink_to(table.parent, target_is_directory=True)
    else:
        outer = target / "icu"
    _mark(refused, refused=True)
    spies = _read_spies(monkeypatch)
    _at_start(lambda: _source(outer), lambda _: None, spies, publication.REFUSED)
