"""Every tool or script that reads data either gates exports or says why not.

B-EXPORT-PARTIAL: an interrupted extraction leaves raw, unpublished module
files that look finished. Readers were gated round by round; this test stops
the regression mechanically. Each ``tools/`` / ``scripts/`` file (and each
packaged CLI entry point) that calls a data reader must either call a
publication gate itself, or be listed in ``READER_CONTRACTS`` as reading a
raw source or something that is not an export, with the reason. A listed file
may not declare an export-root option. New readers fail until classified.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

READER_CALLS = frozenset({
    "read_parquet", "read_schema", "ParquetFile", "read_table", "read_csv",
    "read_csv_auto", "read_excel", "read_metadata", "read_feather", "read_json",
    "dataset",
})
GATE_CALLS = frozenset({
    "require_published_export_package", "require_published_native_export",
    "require_verified_export_package",
    "require_no_in_progress_marker", "open_export_package", "index_export_package",
    "inspect_export_layout",
})
EXPORT_OPTION = re.compile(r"add_argument\(\s*['\"]--[\w-]*(?:export|full6|package)", re.I)

#: path -> (category, what it reads). Categories: raw-source, not-export.
READER_CONTRACTS: dict[str, tuple[str, str]] = {
    "tools/audit_full_concept_dictionary_structure.py": (
        "raw-source", "prepared source tables under --db-root"),
    "tools/audit_source_dictionary_coverage.py": (
        "raw-source", "source item dictionaries under --db-root"),
    "tools/build_demo_walkthrough_trace.py": (
        "not-export", "research run directory JSON receipts"),
    "tools/build_source_item_catalog.py": (
        "raw-source", "d_labitems / d_items source catalogs under --db"),
    "tools/duckdb_raw_parity_census.py": (
        "raw-source", "source tables under --data-path and its own census outputs"),
    "tools/import_canonical9_to_agent_projects.py": (
        "not-export", "benchmark run manifests and evidence files"),
    "tools/render_e2_presentation_figures.py": (
        "not-export", "analysis result CSVs"),
    "tools/replay_landmark_reporting_contract.py": (
        "not-export", "a digest-bound run cohort file and result CSVs"),
    "tools/rerender_demo_figure.py": ("not-export", "figure source tables (CSV)"),
    "tools/run_idea_mining_data_first_discovery.py": (
        "not-export", "a prepared wide cohort file built by the gated cohort builders"),
    "tools/run_research_agent_bench.py": (
        "not-export", "benchmark run step summaries and the run's own cohort file"),
    "tools/verify_figure_presentation.py": ("not-export", "figure source CSVs"),
    "tools/windows_laptop_acceptance.py": (
        "raw-source", "the prepared raw demo folder inventory"),
    "scripts/build_itemid_bucket_cache.py": (
        "raw-source", "flat source tables it buckets (the bucket builder)"),
    "scripts/complete_mined_ideas_demo.py": (
        "raw-source", "raw MIMIC-IV hosp tables"),
    "scripts/figures/QC-A03_main_figures.py": (
        "not-export", "QC-A01 / QC-A02 audit CSV outputs"),
    "scripts/r4_crossdb_sofa2_extract.py": (
        "raw-source", "raw id-table footers under EASYICU_DB_ROOT"),
    "scripts/r4_fig4_nature.py": ("not-export", "analysis result CSVs"),
    "scripts/r4_misimputation_analysis.py": (
        "not-export", "per-stay analysis outputs of r4_crossdb_sofa2_extract"),
    "scripts/r5_fig5_nature.py": ("not-export", "an analysis result CSV"),
}


def _entry_files():
    for base in ("tools", "scripts"):
        yield from sorted((REPO / base).rglob("*.py"))
    package = REPO / "src" / "easyicu"
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if (path.name == "__main__.py"
                or path.name in {"cli.py", "replication_cli.py", "install.py", "hosted_llm_server.py"}
                or relative.startswith("scripts/")):
            yield path


def _call_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                names.add(func.attr)
            elif isinstance(func, ast.Name):
                names.add(func.id)
    return names


def _reads_data(tree: ast.AST, calls: set[str]) -> bool:
    if calls & READER_CALLS:
        return True
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"glob", "rglob"} and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).endswith((".parquet", ".csv", ".csv.gz"))):
            return True
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and re.search(r"read_(parquet|csv)", node.value) and "SELECT" in node.value.upper()):
            return True
    return False


def _scan():
    readers = {}
    for path in _entry_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        calls = _call_names(tree)
        if _reads_data(tree, calls):
            readers[path.relative_to(REPO).as_posix()] = (bool(calls & GATE_CALLS), text)
    return readers


def test_every_data_reader_gates_exports_or_states_its_contract():
    readers = _scan()
    unclassified = sorted(
        path for path, (gated, _text) in readers.items()
        if not gated and path not in READER_CONTRACTS
    )
    assert not unclassified, (
        "These files read data without a publication gate and without a stated "
        "contract. Call require_published_export_package (or open the package "
        f"through intake), or classify them in READER_CONTRACTS: {unclassified}"
    )


def test_a_classified_reader_declares_no_export_root_option():
    readers = _scan()
    offending = sorted(
        path for path in READER_CONTRACTS
        if path in readers and EXPORT_OPTION.search(readers[path][1])
    )
    assert not offending, f"an export-root option needs the gate, not an allowlist entry: {offending}"


def test_the_contract_list_has_no_stale_entries():
    readers = _scan()
    stale = sorted(
        path for path in READER_CONTRACTS
        if path not in readers or readers[path][0]
    )
    assert not stale, f"remove entries that no longer read data or now call a gate: {stale}"
    assert {category for category, _ in READER_CONTRACTS.values()} <= {"raw-source", "not-export"}


# --- member integrity (r13exportsha) -----------------------------------------

INTEGRITY_CALLS = frozenset({"verify_member_integrity", "require_member_digest",
                             "verify_package_members", "_verify_export_member",
                             "formal_integrity_scope", "require_verified_export_package"})

#: Package owners that open export members for screens, intake or tools. Each
#: must compare the member with its manifest record (size + SHA-256).
INTEGRITY_OWNERS = (
    "src/easyicu/research_agent/intake/export_package.py",
    "src/easyicu/io/export_plausibility.py",
    "src/easyicu/webserver/dataio.py",
    "src/easyicu/webserver/agent_outputs.py",
    "src/easyicu/webserver/export_download.py",
    "src/easyicu/webserver/ideas/mining.py",
    "src/easyicu/webserver/patient_drilldown/coverage.py",
    "src/easyicu/webserver/catalog.py",
    "src/easyicu/research_agent/discovery/idea_mining_source_status.py",
)

#: Formal / release tools keep their own full (unmemoised) parquet_sha256 check.
FORMAL_SHA_CHECKS = {
    "scripts/releases/EX-A01_seal_full6_release.py": "parquet_sha256",
    "scripts/releases/EX-A03_refresh_selected_modules.py": "parquet_sha256",
    "tools/reextract_native_export_v2.py": "parquet_sha256",
    "src/easyicu/api/derivation_context.py": "parquet_sha256",
}

#: Formal consumers that verify members with ``formal=True`` (an unrecorded
#: manifest is refused) or run inside ``formal_integrity_scope``.
FORMAL_INTEGRITY_CONSUMERS = {
    "scripts/figures/QC-A01_cross_database_distributions.py":
        "documented publication-QC CLI (scripts/figures/README.md)",
    "scripts/figures/QC-A02_easyicu_cross_database_reliability_audit.py":
        "documented publication-QC CLI (scripts/figures/README.md)",
    "tools/materialize_e3_miiv_kdigo_gradient.py":
        "reproduces the Figure 2 canonical E3 benchmark input",
    "tools/run_research_agent_bench.py": "formal benchmark (Figure 4 / Case B)",
    "tools/audit_extraction_coverage.py":
        "developer coverage audit; gated formally, so no unverified pass",
    "tools/build_discovery_universe.py": "discovery inputs; every globbed member verified",
    "tools/run_idea_mining_longitudinal_discovery.py":
        "listed experimental (docs/research_agent_capability_inventory.md); "
        "gated anyway: each profiled module verified",
    "tools/run_idea_mining_real_discovery.py": "every globbed member verified",
    "tools/run_idea_mining_s6_validation_harness.py": "each consumed member verified",
    "scripts/mine_discovery_gpt54.py": "every globbed and consumed member verified",
    "src/easyicu/research_agent/discovery/idea_mining_source_status.py":
        "source-status profiles are evidence; package and each table verified",
}

#: Publication-gated tool readers that do not compare member digests, each
#: with why it is not on a formal, refresh, webserver or documented-user path
#: and the experimental/uncertified notice it prints in its own output.
#: New tools may not join without review; stale entries fail. Empty since
#: r13exportsha delta27e: every gated tool verifies members formally.
INTEGRITY_PENDING_TOOLS: dict[str, str] = {}


def test_member_owners_compare_each_member_with_its_manifest_record():
    missing = sorted(
        path for path in INTEGRITY_OWNERS
        if not (_call_names(ast.parse((REPO / path).read_text(encoding="utf-8")))
                & INTEGRITY_CALLS)
    )
    assert not missing, f"export member readers without the integrity check: {missing}"


def test_formal_tools_keep_their_own_full_digest_check():
    for path, needle in FORMAL_SHA_CHECKS.items():
        assert needle in (REPO / path).read_text(encoding="utf-8"), path


def test_every_gated_tool_reader_verifies_members_or_is_listed_pending():
    readers = _scan()
    unverified = {
        path for path, (gated, text) in readers.items()
        if gated and path not in FORMAL_SHA_CHECKS
        and not (_call_names(ast.parse(text)) & INTEGRITY_CALLS)
    }
    pending = set(INTEGRITY_PENDING_TOOLS)
    assert unverified <= pending, sorted(unverified - pending)
    stale = sorted(pending - unverified)
    assert not stale, f"now verified or gone; remove from INTEGRITY_PENDING_TOOLS: {stale}"


def test_formal_consumers_refuse_an_unrecorded_manifest():
    for path, why in FORMAL_INTEGRITY_CONSUMERS.items():
        text = (REPO / path).read_text(encoding="utf-8")
        assert ("formal=True" in text or "formal_integrity_scope" in text
                or "require_verified_export_package" in text), (path, why)
    assert all(INTEGRITY_PENDING_TOOLS.values())
