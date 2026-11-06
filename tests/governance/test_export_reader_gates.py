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
