from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_refresher():
    path = ROOT / "scripts/releases/EX-A03_refresh_selected_modules.py"
    spec = importlib.util.spec_from_file_location("selected_module_refresh", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_module_refresh_is_limited_to_correctness_modules() -> None:
    refresher = _load_refresher()
    assert refresher._validate_modules(["outcome"]) == ("outcome",)
    assert refresher._validate_modules(["renal"]) == ("renal",)
    assert refresher._validate_modules(["respiratory"]) == ("respiratory",)
    assert refresher._validate_modules(["sofa1_score"]) == ("sofa1_score",)
    assert refresher._validate_modules(["sofa2_score"]) == ("sofa2_score",)
    assert refresher._validate_modules(["renal", "respiratory"]) == (
        "renal",
        "respiratory",
    )
    with pytest.raises(refresher.ModuleRefreshError, match="sofa2_score"):
        refresher._validate_modules(["vitals"])


def test_respiratory_refresh_expands_to_score_and_sepsis_dependencies() -> None:
    refresher = _load_refresher()
    assert refresher._expand_module_dependency_closure(["outcome"]) == ("outcome",)
    assert refresher._expand_module_dependency_closure(["respiratory"]) == (
        "respiratory",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert refresher._expand_module_dependency_closure(
        ["outcome", "renal", "respiratory"]
    ) == (
        "outcome",
        "respiratory",
        "renal",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert refresher._expand_module_dependency_closure(["sofa1_score"]) == (
        "sofa1_score",
        "sepsis3_sofa1",
    )
    assert refresher._expand_module_dependency_closure(["sofa2_score"]) == (
        "sofa2_score",
        "sepsis3_sofa2",
    )


def test_release_plan_is_database_by_module_under_fixed_8gib_contract() -> None:
    refresher = _load_refresher()
    manifest = {
        "sources": {
            "eicu": {"module_metrics": {"outcome": {"rows": 200_859}}}
        }
    }

    plan = refresher._build_refresh_resource_plan(
        manifest,
        requested_modules=["respiratory"],
        databases=["eicu"],
        memory_budget_mb=8 * 1024,
    )

    modules = plan["databases"]["eicu"]["modules"]
    assert plan["raw_database_reread"] is False
    assert plan["resource_execution_limits"] == {
        "resource_budget_mb": 8192.0,
        "modeled_total_memory_gb": pytest.approx(11.428571),
        "parallel_max_workers": 2,
        "arrow_threads": 2,
        "duckdb_threads": 2,
        "duckdb_memory_limit_mb": 2048,
        "resolver_cache_budget_mb": 512,
    }
    assert modules["respiratory"]["batch_size"] == 50_000
    assert modules["respiratory"]["planned_batches"] == 5
    assert modules["sofa1_score"]["batch_size"] == 67_000
    assert modules["sofa2_score"]["reason_code"] == (
        "invalidated_profile_memory_guard"
    )
    assert modules["sepsis3_sofa1"]["batch_size"] == 67_000
    assert modules["sepsis3_sofa2"]["reason_code"] == (
        "invalidated_profile_memory_guard"
    )
    assert modules["sofa2_score"]["measured_peak_rss_mb"] is None
    assert plan["formal_release_admissible"] is False
    assert plan["unmeasured_or_overridden_modules"] == {
        "eicu": ["sofa2_score", "sepsis3_sofa2"]
    }


def test_release_cli_blocks_unreviewed_fixed_batch_override() -> None:
    refresher = _load_refresher()
    base = [
        "--source-run-root",
        "/source",
        "--output-root",
        "/candidate",
        "--module",
        "respiratory",
    ]
    with pytest.raises(SystemExit):
        refresher.parse_args([*base, "--batch-size", "5000"])

    with pytest.raises(SystemExit):
        refresher.parse_args([*base, "--benchmark-only"])

    parsed = refresher.parse_args(
        [
            *base,
            "--batch-size",
            "5000",
            "--allow-resource-policy-override",
            "--resource-policy-override-reason",
            "controlled partition-invariance experiment",
        ]
    )
    assert parsed.batch_size == 5_000


def test_release_cli_accepts_explicit_benchmark_only_scope() -> None:
    refresher = _load_refresher()
    args = refresher.parse_args(
        [
            "--source-run-root",
            "/source",
            "--output-root",
            "/candidate",
            "--module",
            "sofa2_score",
            "--database",
            "eicu",
            "--batch-size",
            "30000",
            "--allow-resource-policy-override",
            "--resource-policy-override-reason",
            "bounded profiling",
            "--benchmark-only",
        ]
    )

    assert args.benchmark_only is True
    assert args.batch_size == 30_000


def test_plan_only_does_not_require_candidate_output_root() -> None:
    refresher = _load_refresher()
    parsed = refresher.parse_args(
        [
            "--source-run-root",
            "/source",
            "--plan-only",
            "--module",
            "respiratory",
            "--database",
            "eicu",
        ]
    )
    assert parsed.plan_only
    assert parsed.output_root is None


def test_selected_module_refresh_rejects_duplicate_data_path_overrides() -> None:
    refresher = _load_refresher()
    with pytest.raises(refresher.ModuleRefreshError, match="Duplicate"):
        refresher._parse_data_path_overrides(["miiv=/tmp/one", "miiv=/tmp/two"])


def test_selected_database_scope_is_nonempty_valid_and_canonical() -> None:
    refresher = _load_refresher()
    assert refresher._validate_databases(["mimic", "eicu", "mimic"]) == (
        "eicu",
        "mimic",
    )
    with pytest.raises(refresher.ModuleRefreshError, match="At least one"):
        refresher._validate_databases([])
    with pytest.raises(refresher.ModuleRefreshError, match="Unknown"):
        refresher._validate_databases(["not-a-database"])


def test_per_database_module_scope_preserves_minimal_refresh_boundaries() -> None:
    refresher = _load_refresher()
    parsed = refresher._parse_database_module_scopes(
        [
            "eicu=respiratory",
            "mimic=respiratory",
            "miiv=sofa1_score,sofa2_score",
        ]
    )

    databases, requested, closed = refresher._resolve_database_module_scope(
        modules=(),
        databases=(),
        database_module_scope=parsed,
    )

    assert databases == ("eicu", "mimic", "miiv")
    assert requested["eicu"] == ("respiratory",)
    assert closed["eicu"] == (
        "respiratory",
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert closed["miiv"] == (
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    )
    assert "respiratory" not in closed["miiv"]
    assert "sepsis_shared" not in closed["miiv"]


def test_per_database_release_plan_uses_each_database_closure() -> None:
    refresher = _load_refresher()
    manifest = {
        "sources": {
            database: {"module_metrics": {"outcome": {"rows": rows}}}
            for database, rows in {"eicu": 200_859, "miiv": 94_458}.items()
        }
    }

    plan = refresher._build_refresh_resource_plan(
        manifest,
        requested_modules=(),
        databases=(),
        memory_budget_mb=8 * 1024,
        database_module_scope={
            "eicu": ("respiratory",),
            "miiv": ("sofa1_score", "sofa2_score"),
        },
    )

    assert "respiratory" in plan["databases"]["eicu"]["modules"]
    assert "respiratory" not in plan["databases"]["miiv"]["modules"]
    assert set(plan["databases"]["miiv"]["modules"]) == {
        "sofa1_score",
        "sofa2_score",
        "sepsis3_sofa1",
        "sepsis3_sofa2",
    }
    assert plan["formal_release_admissible"] is False
    assert plan["unmeasured_or_overridden_modules"] == {
        "eicu": ["sofa2_score", "sepsis3_sofa2"]
    }


def test_measured_miiv_score_plan_is_formally_admissible() -> None:
    refresher = _load_refresher()
    manifest = {
        "sources": {
            "miiv": {"module_metrics": {"outcome": {"rows": 94_458}}}
        }
    }

    plan = refresher._build_refresh_resource_plan(
        manifest,
        requested_modules=("sofa1_score", "sofa2_score"),
        databases=("miiv",),
        memory_budget_mb=8 * 1024,
    )

    assert plan["formal_release_admissible"] is True
    assert plan["unmeasured_or_overridden_modules"] == {}


def test_unmeasured_aumc_score_pilot_is_not_a_release_standard() -> None:
    refresher = _load_refresher()
    manifest = {
        "sources": {
            "aumc": {"module_metrics": {"outcome": {"rows": 23_106}}}
        }
    }

    plan = refresher._build_refresh_resource_plan(
        manifest,
        requested_modules=("sofa2_score",),
        databases=("aumc",),
        memory_budget_mb=8 * 1024,
    )

    modules = plan["databases"]["aumc"]["modules"]
    assert modules["sofa2_score"]["batch_size"] == 5_000
    assert modules["sofa2_score"]["reason_code"] == (
        "unmeasured_profile_memory_guard"
    )
    assert plan["formal_release_admissible"] is False
    assert plan["unmeasured_or_overridden_modules"] == {
        "aumc": ["sofa2_score", "sepsis3_sofa2"]
    }


def test_data_path_resolution_checks_only_selected_databases(tmp_path: Path) -> None:
    refresher = _load_refresher()
    eicu = tmp_path / "eicu"
    mimic = tmp_path / "mimic"
    eicu.mkdir()
    mimic.mkdir()
    manifest = {
        "data_paths": {
            "eicu": str(eicu),
            "mimic": str(mimic),
            "miiv": str(tmp_path / "deliberately-missing-miiv"),
        }
    }

    assert refresher._resolve_data_paths(
        manifest, {}, ("eicu", "mimic")
    ) == {"eicu": str(eicu.resolve()), "mimic": str(mimic.resolve())}


def test_database_subset_resume_is_refused_without_transaction_receipt(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    with pytest.raises(refresher.ModuleRefreshError, match="fresh candidate"):
        refresher.refresh_candidate(
            source_run_root=tmp_path / "source",
            output_root=tmp_path / "candidate",
            modules=["respiratory"],
            data_path_overrides={},
            batch_size=1,
            resource_policy_override_reason="unit-test benchmark override",
            databases=["eicu", "mimic"],
            resume=True,
        )


def test_legacy_lineage_inference_requires_exact_six_database_receipts() -> None:
    refresher = _load_refresher()
    provenance = {
        "schema_version": refresher.LEGACY_SCHEMA_VERSION,
        "publication_easyicu_git_dirty": False,
        "refreshed_modules": ["renal"],
        "raw_database_reread": True,
        "raw_data_paths": {
            database: f"/raw/{database}" for database in refresher.DATABASES
        },
        "per_database_runtime": {
            database: {"modules": {"renal": {}}}
            for database in refresher.DATABASES
        },
    }
    assert refresher._database_refresh_scope(
        provenance, label="legacy"
    ) == {database: ["renal"] for database in refresher.DATABASES}

    del provenance["per_database_runtime"]["miiv"]["modules"]["renal"]
    with pytest.raises(refresher.ModuleRefreshError, match="miiv"):
        refresher._database_refresh_scope(provenance, label="legacy")


def test_targeted_refresh_extracts_only_selected_but_republishes_all_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    raw_eicu = tmp_path / "raw-eicu"
    raw_mimic = tmp_path / "raw-mimic"
    raw_eicu.mkdir()
    raw_mimic.mkdir()
    data_paths = {
        database: str(tmp_path / f"missing-{database}")
        for database in refresher.DATABASES
    }
    data_paths.update({"eicu": str(raw_eicu), "mimic": str(raw_mimic)})
    source_module_metrics = {
        database: {
            module: {
                "elapsed_seconds": 10.0,
                "peak_rss_mb": 20.0,
                "peak_working_set_mb": 30.0,
                "rows": 1,
                "parquet_bytes": 1,
                "parquet_sha256": "0" * 64,
            }
            for module in refresher.MODULES
        }
        for database in refresher.DATABASES
    }
    legacy_refresh = {
        "schema_version": refresher.LEGACY_SCHEMA_VERSION,
        "publication_easyicu_git_commit": "d" * 40,
        "publication_easyicu_git_dirty": False,
        "requested_modules": ["renal"],
        "refreshed_modules": ["renal"],
        "raw_database_reread": True,
        "raw_data_paths": dict(data_paths),
        "per_database_runtime": {
            database: {
                "database": database,
                "data_path": data_paths[database],
                "modules": {
                    "renal": {
                        "elapsed_seconds": 11.0,
                        "peak_rss_mb": 21.0,
                        "peak_working_set_mb": 31.0,
                    }
                },
            }
            for database in refresher.DATABASES
        },
    }
    source_manifest = {
        "data_paths": data_paths,
        "sources": {
            database: {"module_metrics": source_module_metrics[database]}
            for database in refresher.DATABASES
        },
        "module_refresh": legacy_refresh,
    }
    source.mkdir()
    (source / "run_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    (source / "module_refresh_provenance.json").write_text(
        json.dumps(legacy_refresh), encoding="utf-8"
    )
    (source / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "easyicu_commit": "d" * 40,
                "source_manifest_sha256": {
                    database: "c" * 64 for database in refresher.DATABASES
                },
            }
        ),
        encoding="utf-8",
    )
    (source / "database_extraction_timing.csv").write_text(
        "database,total_rows,total_parquet_bytes\n", encoding="utf-8"
    )
    for database in refresher.DATABASES:
        database_root = source / "exports" / database
        database_root.mkdir(parents=True)
        (database_root / "_manifest.json").write_text(
            json.dumps({"database": database, "unchanged": True}),
            encoding="utf-8",
        )

    extracted: list[str] = []
    republished: list[str] = []
    semantic_audited: list[str] = []

    monkeypatch.setattr(
        refresher.REPUBLICATION, "_require_clean_checkout", lambda: "a" * 40
    )
    monkeypatch.setattr(
        refresher.REPUBLICATION,
        "_validate_source",
        lambda _source: source_manifest,
    )
    monkeypatch.setattr(
        refresher.REPUBLICATION, "_sha256_file", lambda _path: "b" * 64
    )
    monkeypatch.setattr(
        refresher.REPUBLICATION,
        "_source_database_receipt",
        lambda _root: {
            "native_manifest_sha256": "c" * 64,
            "easyicu_git_commit": "d" * 40,
            "easyicu_git_dirty": False,
        },
    )
    monkeypatch.setattr(
        refresher.REPUBLICATION,
        "_rebind_extraction_timing_receipts",
        lambda *_args, **_kwargs: None,
    )

    selected_modules = refresher._expand_module_dependency_closure(["respiratory"])

    def fake_refresh_one_database(**kwargs):
        database = str(kwargs["database"])
        extracted.append(database)
        return {
            "database": database,
            "data_path": kwargs["data_path"],
            "num_patients": 1,
            "batch_size": 1,
            "resource_budget_mb": 8192.0,
            "resource_execution_limits": (
                refresher._resource_budget_execution_limits(8192.0)
            ),
            "total_elapsed_seconds": 1.0,
            "modules": {
                module: {
                    "elapsed_seconds": 1.0,
                    "peak_rss_mb": 2.0,
                    "peak_working_set_mb": 3.0,
                }
                for module in selected_modules
            },
        }

    def fake_republish_database(_root, *, database, **_kwargs):
        republished.append(database)
        return {
            "database": database,
            "runtime_provenance": {"easyicu_git_commit": "a" * 40},
            "source_extraction_provenance": {
                "publication_only": True,
                "raw_database_reread": False,
            },
            "module_timings_seconds": {
                module: 0.0 for module in refresher.MODULES
            },
            "module_peak_rss_mb": {
                module: 0.0 for module in refresher.MODULES
            },
            "module_peak_working_set_mb": {
                module: 0.0 for module in refresher.MODULES
            },
            "files": [
                {
                    "module": module,
                    "rows": 1,
                    "parquet_bytes": 2,
                    "parquet_sha256": "e" * 64,
                }
                for module in refresher.MODULES
            ],
        }

    monkeypatch.setattr(
        refresher, "_refresh_one_database", fake_refresh_one_database
    )
    monkeypatch.setattr(
        refresher.REPUBLICATION, "_republish_database", fake_republish_database
    )
    monkeypatch.setattr(
        refresher,
        "_validate_publication_only_database_semantics",
        lambda _source, candidate_root, **_kwargs: (
            semantic_audited.append(candidate_root.name)
            or {"status": "PASS", "modules": {}}
        ),
    )

    result = refresher.refresh_candidate(
        source_run_root=source,
        output_root=candidate,
        modules=["respiratory"],
        data_path_overrides={},
        batch_size=1,
        resource_policy_override_reason="unit-test benchmark override",
        databases=["mimic", "eicu"],
    )

    assert result == candidate
    assert extracted == ["eicu", "mimic"]
    assert republished == list(refresher.DATABASES)
    assert semantic_audited == ["aumc", "hirid", "miiv", "sic", "eicu", "mimic"]
    assert json.loads(
        (source / "exports" / "miiv" / "_manifest.json").read_text(
            encoding="utf-8"
        )
    ) == {"database": "miiv", "unchanged": True}
    candidate_miiv = json.loads(
        (candidate / "exports" / "miiv" / "_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert candidate_miiv["source_extraction_provenance"] == {
        "publication_only": True,
        "raw_database_reread": False,
        "refreshed_modules": [],
        "current_refreshed_modules": [],
        "inherited_refreshed_modules": ["renal"],
        "cumulative_refreshed_modules": ["renal"],
        "reused_modules": list(refresher.MODULES),
        "selected_module_refresh_scope": (
            "publication_only_for_six_database_commit_harmonization"
        ),
        "transformation": (
            "canonical native-v2 republication without raw-data reread; "
            "logical contents must match the source release"
        ),
    }
    provenance = json.loads(
        (candidate / "module_refresh_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["schema_version"].endswith("_v2")
    assert provenance["selected_databases"] == ["eicu", "mimic"]
    assert provenance["latest_refresh_databases"] == ["eicu", "mimic"]
    assert provenance["raw_database_reread_scope"] == "selected_databases_only"
    assert provenance["cumulative_raw_database_reread_scope"] == (
        "all_six_databases"
    )
    assert provenance["raw_data_paths"] == data_paths
    assert provenance["requested_modules"] == ["respiratory", "renal"]
    assert "renal" in provenance["refreshed_modules"]
    assert provenance["latest_requested_modules"] == ["respiratory"]
    assert provenance["inherited_refreshed_modules"] == ["renal"]
    assert provenance["per_database_runtime"]["eicu"][
        "resource_execution_limits"
    ] == refresher._resource_budget_execution_limits(8192.0)
    assert provenance["per_database_runtime"]["mimic"][
        "resource_execution_limits"
    ] == refresher._resource_budget_execution_limits(8192.0)
    assert provenance["per_database_refreshed_modules"]["miiv"] == ["renal"]
    assert provenance["per_database_refreshed_modules"]["eicu"] == [
        module
        for module in refresher.MODULES
        if module in set(selected_modules) or module == "renal"
    ]
    assert provenance["parent_module_refresh_provenance"] == {
        "path": str(source / "module_refresh_provenance.json"),
        "sha256": "b" * 64,
        "schema_version": refresher.LEGACY_SCHEMA_VERSION,
        "publication_easyicu_git_commit": "d" * 40,
    }
    assert set(provenance["publication_only_semantic_audit"]) == {
        "aumc",
        "hirid",
        "miiv",
        "sic",
    }
    assert set(provenance["reused_module_semantic_audit"]) == {"eicu", "mimic"}
    rebound_run_manifest = json.loads(
        (candidate / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert rebound_run_manifest["module_refresh"] == provenance
    assert rebound_run_manifest["sources"]["miiv"]["module_metrics"]["renal"][
        "elapsed_seconds"
    ] == 11.0
    assert rebound_run_manifest["sources"]["eicu"]["module_metrics"][
        "respiratory"
    ]["elapsed_seconds"] == 1.0
    for database in refresher.DATABASES:
        native = json.loads(
            (candidate / "exports" / database / "_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        source_record = rebound_run_manifest["sources"][database]
        assert source_record["native_manifest_sha256"] == "b" * 64
        files = {entry["module"]: entry for entry in native["files"]}
        for module in refresher.MODULES:
            metric = source_record["module_metrics"][module]
            assert metric["rows"] == files[module]["rows"]
            assert metric["parquet_bytes"] == files[module]["parquet_bytes"]
            assert metric["parquet_sha256"] == files[module]["parquet_sha256"]
            assert metric["elapsed_seconds"] == native[
                "module_timings_seconds"
            ][module]
            assert metric["peak_rss_mb"] == native["module_peak_rss_mb"][module]
            assert metric["peak_working_set_mb"] == native[
                "module_peak_working_set_mb"
            ][module]


def test_publication_only_semantic_audit_ignores_order_but_detects_values(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1, 2],
            "charttime": [0.0, 1.0, 0.0],
            "respiratory_rate": [12.0, None, 18.0],
            "state": ["a", "b", None],
        }
    )
    frame.to_parquet(source / "respiratory.parquet", index=False)
    frame.iloc[::-1].to_parquet(candidate / "respiratory.parquet", index=False)
    (candidate / "_manifest.json").write_text(
        json.dumps({"unavailable_concepts": []}), encoding="utf-8"
    )

    audit = refresher._validate_publication_only_database_semantics(
        source, candidate, modules=("respiratory",)
    )
    assert audit["status"] == "PASS"
    assert audit["modules"]["respiratory"]["rows"] == 3

    changed = frame.copy()
    changed.loc[0, "respiratory_rate"] = 99.0
    changed.to_parquet(candidate / "respiratory.parquet", index=False)
    with pytest.raises(refresher.ModuleRefreshError, match="changed logical"):
        refresher._validate_publication_only_database_semantics(
            source, candidate, modules=("respiratory",)
        )


def test_publication_only_semantic_audit_accepts_only_declared_null_extensions(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    frame = pd.DataFrame({"stay_id": [1, 2], "age": [60.0, 70.0]})
    frame.to_parquet(source / "demographics.parquet", index=False)

    extended = frame.copy()
    extended["icu_unit_type"] = pd.Series([None, None], dtype="string")
    extended.to_parquet(candidate / "demographics.parquet", index=False)
    (candidate / "_manifest.json").write_text(
        json.dumps(
            {
                "unavailable_concepts": [
                    {
                        "module": "demographics",
                        "concept": "icu_unit_type",
                        "reason": "producer_returned_no_physical_column",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = refresher._validate_publication_only_database_semantics(
        source, candidate, modules=("demographics",)
    )
    assert audit["modules"]["demographics"][
        "candidate_added_declared_all_null_columns"
    ] == ["icu_unit_type"]

    extended.loc[0, "icu_unit_type"] = "MICU"
    extended.to_parquet(candidate / "demographics.parquet", index=False)
    with pytest.raises(refresher.ModuleRefreshError, match="column with data"):
        refresher._validate_publication_only_database_semantics(
            source, candidate, modules=("demographics",)
        )

    extended["icu_unit_type"] = pd.Series([None, None], dtype="string")
    extended.to_parquet(candidate / "demographics.parquet", index=False)
    (candidate / "_manifest.json").write_text(
        json.dumps({"unavailable_concepts": []}), encoding="utf-8"
    )
    with pytest.raises(refresher.ModuleRefreshError, match="not declared"):
        refresher._validate_publication_only_database_semantics(
            source, candidate, modules=("demographics",)
        )


def test_new_candidate_never_reuses_source_module_just_because_schema_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only an explicit resume may reuse a completed selected-module export."""

    refresher = _load_refresher()
    candidate = tmp_path / "candidate"
    destination = candidate / "exports" / "hirid"
    destination.mkdir(parents=True)
    source_database = tmp_path / "source" / "hirid"
    source_database.mkdir(parents=True)
    (source_database / "outcome.parquet").write_bytes(b"sealed-outcome")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)

    def fake_extract_database(*args, **kwargs):
        calls.append(kwargs)
        staging = Path(kwargs["output_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "renal.parquet").write_bytes(b"parquet-placeholder")
        (staging / "renal.manifest.json").write_text(json.dumps({}))
        return {
            "num_patients": 1,
            "batch_size": 1,
            "total_elapsed": 1.0,
            "modules": {
                "renal": {
                    "errors": [],
                    "elapsed": 1.0,
                    "peak_rss_mb": 1.0,
                    "peak_working_set_mb": 1.0,
                }
            },
        }

    monkeypatch.setattr(refresher, "extract_database", fake_extract_database)

    refresher._refresh_one_database(
        database="hirid",
        data_path=str(tmp_path),
        source_database_root=source_database,
        candidate_root=candidate,
        modules=("renal",),
        batch_size=None,
        reuse_completed_export=False,
    )

    assert len(calls) == 1


def test_resume_never_treats_destination_schema_as_raw_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume also needs staged evidence or a fresh extraction."""

    refresher = _load_refresher()
    candidate = tmp_path / "candidate"
    destination = candidate / "exports" / "miiv"
    destination.mkdir(parents=True)
    source_database = tmp_path / "source" / "miiv"
    source_database.mkdir(parents=True)
    (source_database / "outcome.parquet").write_bytes(b"sealed-outcome")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)

    def fake_extract_database(*args, **kwargs):
        calls.append(kwargs)
        staging = Path(kwargs["output_dir"])
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "respiratory.parquet").write_bytes(b"parquet-placeholder")
        (staging / "respiratory.manifest.json").write_text(json.dumps({}))
        return {
            "num_patients": 1,
            "batch_size": 1,
            "total_elapsed": 1.0,
            "modules": {
                "respiratory": {
                    "errors": [],
                    "elapsed": 1.0,
                    "peak_rss_mb": 1.0,
                    "peak_working_set_mb": 1.0,
                }
            },
        }

    monkeypatch.setattr(refresher, "extract_database", fake_extract_database)
    refresher._refresh_one_database(
        database="miiv",
        data_path=str(tmp_path),
        source_database_root=source_database,
        candidate_root=candidate,
        modules=("respiratory",),
        batch_size=None,
        reuse_completed_export=True,
    )

    assert len(calls) == 1


def test_selected_longitudinal_refresh_stages_exact_outcome_dependency(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    outcome = source / "outcome.parquet"
    outcome.write_bytes(b"sealed-outcome-bytes")

    staged = refresher._stage_outcome_time_bound_dependency(
        source_database_root=source,
        staging_root=staging,
        modules=("respiratory",),
    )

    assert staged == staging / "outcome.parquet"
    assert staged.read_bytes() == outcome.read_bytes()
    assert not staged.samefile(outcome)
    assert refresher.REPUBLICATION._sha256_file(staged) == (
        refresher.REPUBLICATION._sha256_file(outcome)
    )
    assert refresher._stage_outcome_time_bound_dependency(
        source_database_root=source,
        staging_root=staging,
        modules=("outcome",),
    ) is None


def test_score_refresh_stages_exact_sepsis_dependency_without_refreshing_it(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    dependency = source / "sepsis_shared.parquet"
    dependency.write_bytes(b"sealed-sepsis-evidence")

    staged = refresher._stage_sepsis_shared_dependency(
        source_database_root=source,
        staging_root=staging,
        modules=("sofa1_score", "sepsis3_sofa1"),
    )

    assert staged == staging / "sepsis_shared.parquet"
    assert staged.read_bytes() == dependency.read_bytes()
    assert not staged.samefile(dependency)
    assert refresher.REPUBLICATION._sha256_file(staged) == (
        refresher.REPUBLICATION._sha256_file(dependency)
    )
    assert refresher._stage_sepsis_shared_dependency(
        source_database_root=source,
        staging_root=staging,
        modules=("sofa1_score",),
    ) is None


def test_selected_longitudinal_refresh_fails_before_raw_read_without_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresher = _load_refresher()
    candidate = tmp_path / "candidate"
    (candidate / "exports" / "eicu").mkdir(parents=True)
    source_database = tmp_path / "source" / "eicu"
    source_database.mkdir(parents=True)
    monkeypatch.setattr(
        refresher,
        "extract_database",
        lambda *args, **kwargs: pytest.fail("raw extraction must not start"),
    )

    with pytest.raises(
        refresher.ModuleRefreshError,
        match="sealed outcome time-bound dependency",
    ):
        refresher._refresh_one_database(
            database="eicu",
            data_path=str(tmp_path),
            source_database_root=source_database,
            candidate_root=candidate,
            modules=("sepsis3_sofa2",),
            batch_size=1,
            reuse_completed_export=False,
        )


def test_completed_producer_staging_is_native_published_without_reextracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source" / "eicu"
    candidate_root = tmp_path / "candidate"
    staging = candidate_root / ".module_refresh_staging" / "eicu"
    destination = candidate_root / "exports" / "eicu"
    source.mkdir(parents=True)
    staging.mkdir(parents=True)
    destination.mkdir(parents=True)
    pd.DataFrame({"stay_id": [1], "los_icu": [24.0]}).to_parquet(
        source / "outcome.parquet", index=False
    )
    pd.DataFrame(
        {"stay_id": [1], "charttime": [0.0], "susp_inf": [True]}
    ).to_parquet(source / "sepsis_shared.parquet", index=False)
    pd.DataFrame(
        {"stay_id": [1], "charttime": [1.0], "sep3_sofa2": [True]}
    ).to_parquet(staging / "sepsis3_sofa2.parquet", index=False)
    (staging / "sepsis3_sofa2.manifest.json").write_text(
        json.dumps(
            {
                "module": "sepsis3_sofa2",
                "saved": {"sep3_sofa2": {}},
                "errors": [],
                "elapsed_sec": 7.2,
                "peak_rss_mb": 12.0,
                "peak_working_set_mb": 10.0,
            }
        )
    )
    publish_calls: list[dict[str, object]] = []

    def fake_publish(**kwargs):
        publish_calls.append(kwargs)
        (Path(kwargs["output_dir"]) / "_manifest.json").write_text(
            json.dumps({"schema_version": "easyicu_native_export_v2"})
        )

    monkeypatch.setattr(refresher, "_publish_native_export_v2", fake_publish)
    monkeypatch.setattr(
        refresher,
        "extract_database",
        lambda *args, **kwargs: pytest.fail("completed staging was recomputed"),
    )

    result = refresher._refresh_one_database(
        database="eicu",
        data_path=str(tmp_path),
        source_database_root=source,
        candidate_root=candidate_root,
        modules=("sepsis3_sofa2",),
        batch_size=31_000,
        reuse_completed_export=False,
    )

    assert len(publish_calls) == 1
    assert publish_calls[0]["require_stay_time_bounds"] is True
    assert not staging.exists()
    assert (destination / "sepsis3_sofa2.parquet").is_file()
    assert not (destination / "outcome.parquet").exists()
    assert result["recovery_mode"] == "completed_staging_promoted"


def test_resume_reuses_only_complete_files_detached_from_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refresher = _load_refresher()
    source = tmp_path / "source" / "aumc"
    candidate_root = tmp_path / "candidate"
    candidate = candidate_root / "exports" / "aumc"
    source.mkdir(parents=True)
    candidate.mkdir(parents=True)
    for suffix in (".parquet", ".manifest.json"):
        (source / f"respiratory{suffix}").write_bytes(b"source")
        (candidate / f"respiratory{suffix}").write_bytes(b"refreshed")
    (candidate / "respiratory.manifest.json").write_text(
        json.dumps({"elapsed_sec": 1, "peak_rss_mb": 2, "peak_working_set_mb": 3})
    )
    monkeypatch.setattr(refresher, "_module_is_canonical_refresh", lambda *_: True)
    monkeypatch.setattr(
        refresher,
        "extract_database",
        lambda *args, **kwargs: pytest.fail(
            "detached completed files were re-extracted"
        ),
    )

    result = refresher._refresh_one_database(
        database="aumc",
        data_path=str(tmp_path),
        source_database_root=source,
        candidate_root=candidate_root,
        modules=("respiratory",),
        batch_size=None,
        reuse_completed_export=True,
    )

    assert result["recovery_mode"].startswith("explicit_resume")


def test_score_content_gate_rejects_all_null_sofa_totals(tmp_path: Path) -> None:
    refresher = _load_refresher()
    components = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 1.0],
            "sofa2": [None, None],
            **{component: [None, None] for component in components},
        }
    )
    for component in components:
        frame[f"{component}_observed"] = False
        frame[f"{component}_available"] = False
    frame["sofa2_observed"] = False
    frame["sofa2_available"] = False
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)

    with pytest.raises(refresher.ModuleRefreshError, match="0 non-null"):
        refresher._validate_refreshed_score_content(
            tmp_path, ("sofa2_score",), database="eicu"
        )


def test_score_content_gate_accepts_sic_sofa2_only_when_cns_is_structurally_unavailable(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    components = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "sofa2": [None],
            **{component: [0.0] for component in components},
        }
    )
    for component in components:
        frame[f"{component}_observed"] = component != "sofa2_cns"
        frame[f"{component}_available"] = component != "sofa2_cns"
    frame["sofa2_observed"] = False
    frame["sofa2_available"] = False
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)

    refresher._validate_refreshed_score_content(
        tmp_path,
        ("sofa2_score",),
        database="sic",
    )

    with pytest.raises(refresher.ModuleRefreshError, match="0 non-null"):
        refresher._validate_refreshed_score_content(
            tmp_path,
            ("sofa2_score",),
            database="eicu",
        )


def test_score_content_gate_streams_non_null_sofa_totals(tmp_path: Path) -> None:
    refresher = _load_refresher()
    components = [
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cardio",
        "sofa_cns",
        "sofa_renal",
    ]
    pd.DataFrame(
        {
            "stay_id": [1, 1],
            "charttime": [0.0, 1.0],
            "sofa": [0.0, 4.0],
            **{
                component: [0.0, 4.0 if component == "sofa_resp" else 0.0]
                for component in components
            },
        }
    ).to_parquet(tmp_path / "sofa1_score.parquet", index=False)

    refresher._validate_refreshed_score_content(
        tmp_path, ("sofa1_score",), database="miiv"
    )


def test_score_content_gate_rejects_independently_aggregated_sofa_total(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "sofa": [4.0],
            "sofa_resp": [4.0],
            "sofa_coag": [4.0],
            "sofa_liver": [0.0],
            "sofa_cardio": [0.0],
            "sofa_cns": [0.0],
            "sofa_renal": [0.0],
        }
    ).to_parquet(tmp_path / "sofa1_score.parquet", index=False)

    with pytest.raises(refresher.ModuleRefreshError, match="total/receipts"):
        refresher._validate_refreshed_score_content(
            tmp_path, ("sofa1_score",), database="miiv"
        )


def test_score_content_gate_checks_sofa2_aggregate_receipts(tmp_path: Path) -> None:
    refresher = _load_refresher()
    components = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "sofa2": [4.0],
            **{
                component: [4.0 if component == "sofa2_resp" else 0.0]
                for component in components
            },
        }
    )
    for component in components:
        frame[f"{component}_observed"] = True
        frame[f"{component}_available"] = True
    frame["sofa2_observed"] = True
    frame["sofa2_available"] = False
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)

    with pytest.raises(refresher.ModuleRefreshError, match="total/receipts"):
        refresher._validate_refreshed_score_content(
            tmp_path, ("sofa2_score",), database="miiv"
        )


def test_score_content_gate_accepts_primary_normal_value_imputation(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    components = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "sofa2": [4.0],
            **{
                component: [4.0 if component == "sofa2_resp" else None]
                for component in components
            },
        }
    )
    for component in components:
        available = component == "sofa2_resp"
        frame[f"{component}_observed"] = available
        frame[f"{component}_available"] = available
    frame["sofa2_observed"] = False
    frame["sofa2_available"] = False
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)

    refresher._validate_refreshed_score_content(
        tmp_path,
        ("sofa2_score",),
        database="miiv",
    )


def test_score_content_gate_rejects_nonzero_disclaimed_component(
    tmp_path: Path,
) -> None:
    refresher = _load_refresher()
    components = [
        "sofa2_resp",
        "sofa2_coag",
        "sofa2_liver",
        "sofa2_cardio",
        "sofa2_cns",
        "sofa2_renal",
    ]
    frame = pd.DataFrame(
        {
            "stay_id": [1],
            "charttime": [0.0],
            "sofa2": [0.0],
            **{
                component: [2.0 if component == "sofa2_resp" else 0.0]
                for component in components
            },
        }
    )
    for component in components:
        frame[f"{component}_observed"] = False
        frame[f"{component}_available"] = False
    frame["sofa2_observed"] = False
    frame["sofa2_available"] = False
    frame.to_parquet(tmp_path / "sofa2_score.parquet", index=False)

    with pytest.raises(refresher.ModuleRefreshError, match="non-zero organ score"):
        refresher._validate_refreshed_score_content(
            tmp_path,
            ("sofa2_score",),
            database="miiv",
        )
