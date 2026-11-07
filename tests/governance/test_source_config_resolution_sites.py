"""By-name source-configuration lookups are pinned to known sites (lead-r13 r13srccfg).

A function that already has a data source must read with that source's own
configuration.  Re-loading the configuration by database name there pairs a
packaged (registry) mapping with the caller's root, and an attached source or
a custom configuration then reads the wrong files without failing: the HiRID
observation helper did this for the death callbacks.

Every ``load_src_cfg(...)`` and ``load_data_sources().get(...)`` call in
``src/easyicu`` must be one of the sites below, each with the reason it may
resolve by name.  A new site fails this test until it is reviewed.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

_REGISTRY_SAME = (
    "database-identity entry without a source object; the modern loader binds the "
    "same active-registry configuration by name (base._init_datasource) and passes "
    "that object on, so the lookup returns the configuration the data was read with"
)

ALLOWED_SITES = {
    # Source owners: they select or build the source.
    "easyicu/datasource.py::storage_source": "the storage owner's database + root source",
    "easyicu/table/__init__.py::load_table": "load_table's own source selection",
    "easyicu/table/__init__.py::select_table_source": "load_table's own source selection",
    "easyicu/table/__init__.py::load_id_tbl": "public by-name API; its caller names the database",
    # Legacy loader: bound once per loader on the r13aumcwt lineage (8d90dac58).
    "easyicu/load_concepts.py::__init__": "legacy ConceptLoader construction from a name",
    "easyicu/load_concepts.py::_load_recursive_concept": "legacy loader; removed on the r13aumcwt lineage",
    "easyicu/load_concepts.py::_select_source": "legacy loader's single selection (r13aumcwt lineage)",
    # Publication and export metadata keyed by the database identity.
    "easyicu/api/extraction.py::_publish_native_export_v2": _REGISTRY_SAME,
    "easyicu/concept/export_metadata.py::_export_identity_and_time_coordinates": _REGISTRY_SAME,
    "easyicu/research_agent/cohort/materializer.py::_normalize_legacy_export_categorical": _REGISTRY_SAME,
    "easyicu/research_agent/cohort/materializer.py::_missing_outside_declared_source_factor": _REGISTRY_SAME,
    "easyicu/research_agent/cohort/materializer.py::_materialize_cohort_from_resolved_source": _REGISTRY_SAME,
    "easyicu/research_agent/intake/export_package.py::_load_column_metadata_sidecar": _REGISTRY_SAME,
    "easyicu/research_agent/intake/typed_cohort_source.py::typed_cohort_source_database": _REGISTRY_SAME,
    # Native database + root readers (no source object is passed in).
    "easyicu/scores/aki_profiles.py::rrt_source_receipt": _REGISTRY_SAME,
    "easyicu/scores/comorbidity.py::_build_datasource": _REGISTRY_SAME,
    # Web host flows over registered sources (registry configurations).
    "easyicu/webserver/dataio.py::runner": _REGISTRY_SAME,
    "easyicu/webserver/pi_copilot/source_concept_reference.py::source_concept_reference": _REGISTRY_SAME,
    "easyicu/webserver/raw_preparation_candidate.py::create_candidate_for_review": _REGISTRY_SAME,
    "easyicu/webserver/raw_preparation_dependencies.py::native_planning_metadata": _REGISTRY_SAME,
    "easyicu/webserver/source_partition_candidates.py::discover_source_partition_candidates": _REGISTRY_SAME,
    "easyicu/webserver/source_partition_candidates.py::candidate_from_bound_research_context": _REGISTRY_SAME,
    "easyicu/webserver/time_origin_source_authority.py::prepare_case_duration_metadata_plan": _REGISTRY_SAME,
}


def _call_name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _sites():
    found = []
    for path in sorted((SRC / "easyicu").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(SRC).as_posix()

        def visit(node, owner):
            for child in ast.iter_child_nodes(node):
                name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else owner
                if isinstance(child, ast.Call):
                    called = _call_name(child.func)
                    by_registry = (
                        called == "get"
                        and isinstance(child.func, ast.Attribute)
                        and isinstance(child.func.value, ast.Call)
                        and _call_name(child.func.value.func) == "load_data_sources"
                    )
                    if called == "load_src_cfg" or by_registry:
                        found.append((f"{rel}::{name}", child.lineno))
                visit(child, name)

        visit(tree, "<module>")
    return found


def test_by_name_configuration_lookups_are_reviewed_sites():
    unknown = [f"{site} (line {line})" for site, line in _sites() if site not in ALLOWED_SITES]
    assert not unknown, "by-name configuration lookup outside the reviewed sites:\n" + "\n".join(unknown)


def test_source_bound_callbacks_never_resolve_by_name():
    # The callback layer always has the caller's data source.
    for site, line in _sites():
        assert not site.startswith(("easyicu/concept/callback_apply.py", "easyicu/concept/__init__.py",
                                    "easyicu/scores/icu_mortality.py")), f"{site} (line {line})"
