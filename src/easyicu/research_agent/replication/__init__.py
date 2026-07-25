"""Replication subpackage.

Groups six modules that together implement
cross-database / paper-aware replication and reproducibility:

* :mod:`.discovery` — deterministic export discovery and the
  ``lactate / MAP / vasopressor`` cross-database driver.
* :mod:`.paper` — paper-profile parsing and replication-spec/report building.
* :mod:`.metrics` — provider- and pipeline-neutral metric comparison.
* :mod:`.notebook` — ``run.ipynb`` and ``requirements.lock.txt``
  provenance artefacts.
* :mod:`.envelope` — LLM reproducibility envelope (prompt/response
  hashes, environment snapshot, recording client).
* :mod:`.report` — cross-database run-summary and comparison rendering.

The submodules do not cross-import each other, so they are loaded lazily.
Public symbols remain available from this package without importing every
replication feature when a caller needs only one leaf module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Dict

_SYMBOL_MODULE: Dict[str, str] = {
    "LACTATE_MAP_VASO_EXPORT_GROUPS": ".discovery",
    "LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS": ".discovery",
    "ReplicationTarget": ".discovery",
    "discover_easyicu_exports": ".discovery",
    "export_lactate_map_vaso_concepts_from_easyicu": ".discovery",
    "run_lactate_map_vaso_replication": ".discovery",
    "shock_strata": ".discovery",
    "summarize_lactate_map_vaso_cohort": ".discovery",
    "ENVELOPE_SCHEMA_VERSION": ".envelope",
    "ReproCallRecord": ".envelope",
    "ReproEnvelope": ".envelope",
    "ReproRecordingClient": ".envelope",
    "build_environment_snapshot": ".envelope",
    "envelope_role_resolver": ".envelope",
    "sha256_messages": ".envelope",
    "sha256_text": ".envelope",
    "NotebookStep": ".notebook",
    "build_notebook": ".notebook",
    "build_requirements_lockfile": ".notebook",
    "write_notebook": ".notebook",
    "compare_metric_values": ".metrics",
    "build_paper_replication_spec": ".paper",
    "build_paper_result_ledger": ".paper",
    "canonical_outcome_name": ".paper",
    "collect_easyicu_metrics": ".paper",
    "compare_paper_to_easyicu": ".paper",
    "load_paper_source": ".paper",
    "map_text_to_easyicu_concept": ".paper",
    "parse_paper_profile": ".paper",
    "postprocess_paper_replication": ".paper",
    "render_deviation_report": ".paper",
    "render_replication_report": ".paper",
    "render_showcase_manuscript": ".paper",
    "write_claim_csv": ".paper",
    "write_fail_closed_paper_package": ".paper",
}

__all__ = [
    # discovery
    "LACTATE_MAP_VASO_EXPORT_GROUPS",
    "LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS",
    "ReplicationTarget",
    "discover_easyicu_exports",
    "export_lactate_map_vaso_concepts_from_easyicu",
    "run_lactate_map_vaso_replication",
    "shock_strata",
    "summarize_lactate_map_vaso_cohort",
    # envelope
    "ENVELOPE_SCHEMA_VERSION",
    "ReproCallRecord",
    "ReproEnvelope",
    "ReproRecordingClient",
    "build_environment_snapshot",
    "envelope_role_resolver",
    "sha256_messages",
    "sha256_text",
    # notebook
    "NotebookStep",
    "build_notebook",
    "build_requirements_lockfile",
    "write_notebook",
    # paper
    "build_paper_replication_spec",
    "build_paper_result_ledger",
    "canonical_outcome_name",
    "collect_easyicu_metrics",
    "compare_metric_values",
    "compare_paper_to_easyicu",
    "load_paper_source",
    "map_text_to_easyicu_concept",
    "parse_paper_profile",
    "postprocess_paper_replication",
    "render_deviation_report",
    "render_replication_report",
    "render_showcase_manuscript",
    "write_claim_csv",
    "write_fail_closed_paper_package",
]


def __getattr__(name: str):
    module_name = _SYMBOL_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
