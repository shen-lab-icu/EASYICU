"""Replication subpackage.

Groups four formerly-top-level modules that together implement
cross-database / paper-aware replication and reproducibility:

* :mod:`.discovery` — deterministic export discovery and the
  ``lactate / MAP / vasopressor`` cross-database driver.
* :mod:`.paper` — paper-profile parsing, spec building, and
  metric comparison.
* :mod:`.notebook` — ``run.ipynb`` and ``requirements.lock.txt``
  provenance artefacts.
* :mod:`.envelope` — LLM reproducibility envelope (prompt/response
  hashes, environment snapshot, recording client).

The submodules do not cross-import each other, so they can be loaded
independently. This ``__init__`` re-exports their public symbols so
callers can keep writing ``from easyicu.research_agent.replication
import X`` for any symbol previously exposed by the four old modules.
"""

from __future__ import annotations

from .discovery import (
    LACTATE_MAP_VASO_EXPORT_GROUPS,
    LACTATE_MAP_VASO_MINIMAL_EXPORT_GROUPS,
    ReplicationTarget,
    discover_easyicu_exports,
    export_lactate_map_vaso_concepts_from_easyicu,
    run_lactate_map_vaso_replication,
    shock_strata,
    summarize_lactate_map_vaso_cohort,
)
from .envelope import (
    ENVELOPE_SCHEMA_VERSION,
    ReproCallRecord,
    ReproEnvelope,
    ReproRecordingClient,
    build_environment_snapshot,
    envelope_role_resolver,
    sha256_messages,
    sha256_text,
)
from .notebook import (
    NotebookStep,
    build_notebook,
    build_requirements_lockfile,
    write_notebook,
)
from .paper import (
    build_paper_replication_spec,
    build_paper_result_ledger,
    canonical_outcome_name,
    collect_easyicu_metrics,
    compare_metric_values,
    compare_paper_to_easyicu,
    load_paper_source,
    map_text_to_easyicu_concept,
    parse_paper_profile,
    postprocess_paper_replication,
    render_deviation_report,
    render_replication_report,
    render_showcase_manuscript,
    write_claim_csv,
    write_fail_closed_paper_package,
)

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
