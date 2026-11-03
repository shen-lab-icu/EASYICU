"""A host-derived analysis cohort descends from the run's cohort by its authority.

The analysis cohort the host materialises from the run's input cohort is
registered without an input edge; its typed authority names the parent
authority instead.  Lineage readers follow that verified chain, never a scan
or a file name.
"""

from __future__ import annotations

from tests.support.host_cohort import typed_run as typed_run

from easyicu.research_agent.intake.materialized_metadata import (
    load_verified_materialized_cohort_authority,
    materialized_cohort_descends_from,
)


def _identities(run_dir):
    child = load_verified_materialized_cohort_authority(run_dir / "cohort_analysis.parquet")
    parent = load_verified_materialized_cohort_authority(run_dir / "cohort.parquet")
    assert child is not None and parent is not None
    return child, parent


def test_the_analysis_cohort_descends_from_the_run_cohort(typed_run):
    run_dir, _ = typed_run
    child, parent = _identities(run_dir)

    assert materialized_cohort_descends_from(
        run_dir,
        authority_sha256=child.reference.sha256,
        cohort_sha256=child.authority.cohort_sha256,
        ancestor_cohort_sha256=parent.authority.cohort_sha256,
    )
    # The parent's own authority reaches itself.
    assert materialized_cohort_descends_from(
        run_dir,
        authority_sha256=parent.reference.sha256,
        cohort_sha256=parent.authority.cohort_sha256,
        ancestor_cohort_sha256=parent.authority.cohort_sha256,
    )


def test_a_wrong_cohort_unrelated_ancestor_or_altered_parent_answers_no(typed_run):
    run_dir, _ = typed_run
    child, parent = _identities(run_dir)
    descends = dict(
        authority_sha256=child.reference.sha256,
        cohort_sha256=child.authority.cohort_sha256,
        ancestor_cohort_sha256=parent.authority.cohort_sha256,
    )

    assert not materialized_cohort_descends_from(run_dir, **{**descends, "cohort_sha256": "c" * 64})
    assert not materialized_cohort_descends_from(run_dir, **{**descends, "ancestor_cohort_sha256": "d" * 64})
    assert not materialized_cohort_descends_from(run_dir, **{**descends, "authority_sha256": "e" * 64})
    assert not materialized_cohort_descends_from(run_dir, **{**descends, "authority_sha256": "not-a-digest"})

    parent_file = run_dir / parent.reference.file
    parent_file.write_bytes(parent_file.read_bytes().replace(b"}", b" }", 1))
    assert not materialized_cohort_descends_from(run_dir, **descends)
