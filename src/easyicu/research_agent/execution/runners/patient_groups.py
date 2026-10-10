"""The patient grouping an executed step reads.

Owner
-----
This module answers one question for every runner that splits or resamples
ICU stays by patient: which grouping, if any, the run's authorities issue
for the step's cohort.  The planning owner answers first
(:func:`...planning.dependence_authority.context_patient_group_authority`:
a bound replacement row identity or a direct patient identifier).  Only when
it states none does the verified materialization ancestry of the step's
cohort answer: a parent authority whose replacement row identity derives the
patient as the prefix before ``:s``.  Nothing else -- a column's name or the
pattern of its values -- issues a grouping.  The static prediction owner and
every other runner that groups by patient read this one answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ...contracts.dependence import PlannedDependenceRequirement
from ...intake.materialized_metadata import (
    MaterializedCohortAuthority,
    MaterializedCohortAuthorityRef,
    VerifiedMaterializedCohortAuthority,
    load_verified_materialized_cohort_authority,
)
from ...planning.dependence_authority import context_patient_group_authority
from ...research_context.typed import ResearchContextAuthority
from .typed_input_binding import sha256_file


def step_patient_group_authority(
    *,
    context: ResearchContextAuthority,
    source_cohort: Path,
    run_dir: Path,
) -> PlannedDependenceRequirement | None:
    """The patient grouping the run's authorities issue for a step's cohort, or None.

    The planning owner answers first; only when it states none does the
    verified materialization ancestry of ``source_cohort`` answer.
    """

    direct = context_patient_group_authority(context)
    if direct is not None:
        return direct

    verified = load_verified_materialized_cohort_authority(Path(source_cohort))
    authority_root = Path(source_cohort).parent
    if verified is None:
        materialized_inputs = getattr(context, "materialized_inputs", None)
        context_cohort = getattr(materialized_inputs, "cohort", None)
        if context_cohort is not None:
            cohort_file = str(context_cohort.cohort_file)
            if Path(cohort_file).name != cohort_file:
                raise RuntimeError("typed context cohort file is not run-local")
            expected = MaterializedCohortAuthorityRef.from_dict(
                context_cohort.authority_ref
            )
            authority_root = Path(run_dir)
            verified = load_verified_materialized_cohort_authority(
                authority_root / cohort_file,
                expected_authority=expected,
            )
    return verified_ancestry_patient_group(verified, authority_root=authority_root)


def verified_ancestry_patient_group(
    verified: VerifiedMaterializedCohortAuthority | None,
    *,
    authority_root: Path,
) -> PlannedDependenceRequirement | None:
    """The grouping one verified authority ancestry issues, or None.

    Only a parent authority that binds the same cohort and row identity and
    derives its replacement row identity as the patient before ``:s``
    issues one.
    """

    if verified is None or verified.authority.parent_authority_sha256 is None:
        return None
    parent_sha256 = verified.authority.parent_authority_sha256
    parent_path = Path(authority_root) / (
        f"cohort_authority.sha256-{parent_sha256}.json"
    )
    if not parent_path.is_file() or sha256_file(parent_path) != parent_sha256:
        return None
    payload = json.loads(parent_path.read_text("utf-8"))
    if not isinstance(payload, Mapping):
        return None
    parent = MaterializedCohortAuthority.from_dict(payload)
    if (
        parent.cohort_sha256 != verified.authority.cohort_sha256
        or parent.row_identity_sha256 != verified.authority.row_identity_sha256
        or parent.identity_column != verified.authority.identity_column
    ):
        return None
    replacement = parent.producer_parameters.get("replacement_row_identity")
    if not isinstance(replacement, Mapping):
        return None
    derivation = replacement.get("patient_group_derivation")
    if not (
        replacement.get("output_identity_column") == parent.identity_column
        and isinstance(derivation, Mapping)
        and derivation.get("algorithm") == "prefix_before_:s"
        and derivation.get("delimiter") == ":s"
    ):
        return None
    return PlannedDependenceRequirement(
        group_source=parent.identity_column,
        group_derivation="prefix_before_delimiter",
        delimiter=":s",
    )


__all__ = ["step_patient_group_authority", "verified_ancestry_patient_group"]
