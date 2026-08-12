"""EvidenceStore: hashing, alias resolution, manuscript binding (T1.2).

These tests pin the behaviour the writer agent depends on. If the
alias system regresses, manuscripts immediately fill up with
``[evidence missing: …]`` markers — exactly what T1.2 fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_register_file_creates_index_and_hash(ra, tmp_path: Path):
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="table",
        description="a tiny csv",
        source_path=src,
    )
    assert rec.sha256 and len(rec.sha256) == 64
    assert (tmp_path / "evidence" / "evidence_index.json").exists()
    persisted = json.loads((tmp_path / "evidence" / "evidence_index.json").read_text())
    assert any(p["evidence_id"] == rec.evidence_id for p in persisted)


def test_alias_resolves_via_filename_stem(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="table",
        description="t1",
        source_path=src,
    )
    via_alias = store.get("table_one")
    assert via_alias is not None
    assert via_alias.evidence_id == rec.evidence_id


def test_unique_hash_suffixed_evidence_resolves_by_stable_prefix(ra, tmp_path: Path):
    src = tmp_path / "mortality_by_sofa2_stratum.png"
    src.write_text("fake image", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="figure",
        description="mortality figure",
        source_path=src,
        evidence_id="figure_mortality_by_sofa2_stratum_abc12345",
    )

    via_prefix = store.get("figure_mortality_by_sofa2_stratum")

    assert via_prefix is not None
    assert via_prefix.evidence_id == rec.evidence_id


def test_explicit_aliases(ra, tmp_path: Path):
    src = tmp_path / "step_summary.json"
    src.write_text('{"x":1}', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="statistic",
        description="summary",
        source_path=src,
        aliases=["outcome_rate", "outcome_incidence"],
    )
    for name in ("outcome_rate", "outcome_incidence"):
        got = store.get(name)
        assert got is not None and got.evidence_id == rec.evidence_id


def test_registration_can_defer_all_alias_publication(ra, tmp_path: Path):
    src = tmp_path / "step_summary.json"
    src.write_text('{"x":1}', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)

    rec = store.register_file(
        kind="statistic",
        description="Unsealed candidate summary.",
        source_path=src,
        produced_by_step="03_model",
        evidence_id="draft_summary_v1",
        aliases=["primary_association"],
        publish_aliases=False,
    )

    assert store.aliases() == {}
    assert store.get("primary_association") is None
    assert store.get("step_summary") is None
    assert store.get("draft_summary") is None
    assert store.get(rec.evidence_id) is not None


def test_publish_success_aliases_exposes_deferred_alias_surface(ra, tmp_path: Path):
    src = tmp_path / "step_summary.json"
    src.write_text('{"x":1}', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="statistic",
        description="Sealed summary.",
        source_path=src,
        produced_by_step="03_model",
        evidence_id="sealed_summary",
        aliases=["not_published_during_registration"],
        publish_aliases=False,
    )

    published = store.publish_success_aliases(
        rec.evidence_id,
        aliases=["primary_association"],
    )

    assert set(published) == {
        "primary_association",
        "step_summary",
        rec.evidence_id,
    }
    assert store.get("primary_association").evidence_id == rec.evidence_id
    assert store.get("step_summary").evidence_id == rec.evidence_id
    assert store.aliases()[rec.evidence_id] == rec.evidence_id
    assert store.get("not_published_during_registration") is None


def test_publish_success_aliases_replaces_only_same_step_owner(ra, tmp_path: Path):
    first_src = tmp_path / "first.json"
    same_step_src = tmp_path / "same_step.json"
    other_step_src = tmp_path / "other_step.json"
    first_src.write_text('{"x":1}', encoding="utf-8")
    same_step_src.write_text('{"x":2}', encoding="utf-8")
    other_step_src.write_text('{"x":3}', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(
        kind="statistic",
        description="First successful attempt.",
        source_path=first_src,
        produced_by_step="03_model",
        evidence_id="summary_attempt_1",
        aliases=["primary_association"],
    )
    same_step = store.register_file(
        kind="statistic",
        description="Later successful attempt.",
        source_path=same_step_src,
        produced_by_step="03_model",
        evidence_id="summary_attempt_2",
        publish_aliases=False,
    )
    other_step = store.register_file(
        kind="statistic",
        description="Different step.",
        source_path=other_step_src,
        produced_by_step="04_other",
        evidence_id="summary_other_step",
        publish_aliases=False,
    )

    same_step_published = store.publish_success_aliases(
        same_step.evidence_id,
        aliases=["primary_association"],
    )
    other_step_published = store.publish_success_aliases(
        other_step.evidence_id,
        aliases=["primary_association"],
    )

    assert first.evidence_id != same_step.evidence_id
    assert same_step_published["primary_association"] == same_step.evidence_id
    assert "primary_association" not in other_step_published
    assert store.get("primary_association").evidence_id == same_step.evidence_id


def test_step_success_alias_batch_rejects_stale_owner_without_partial_publish(
    ra, tmp_path: Path
):
    first_src = tmp_path / "first.json"
    candidate_a_src = tmp_path / "candidate_a.json"
    candidate_b_src = tmp_path / "candidate_b.json"
    first_src.write_text('{"x":1}', encoding="utf-8")
    candidate_a_src.write_text('{"x":2}', encoding="utf-8")
    candidate_b_src.write_text('{"x":3}', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(
        kind="statistic",
        description="Existing authority from another step.",
        source_path=first_src,
        produced_by_step="03_model",
        evidence_id="summary_first",
        aliases=["primary_association"],
    )
    candidate_a = store.register_file(
        kind="statistic",
        description="Candidate with a conflicting semantic alias.",
        source_path=candidate_a_src,
        produced_by_step="04_model",
        evidence_id="summary_candidate_a",
        publish_aliases=False,
    )
    candidate_b = store.register_file(
        kind="statistic",
        description="Candidate whose alias would otherwise be publishable.",
        source_path=candidate_b_src,
        produced_by_step="04_model",
        evidence_id="summary_candidate_b",
        publish_aliases=False,
    )

    with pytest.raises(ValueError, match="already owned"):
        store.publish_step_success_aliases(
            {
                candidate_b.evidence_id: ["clean_alias"],
                candidate_a.evidence_id: ["primary_association"],
            },
            step_id="04_model",
        )

    assert store.get("primary_association").evidence_id == first.evidence_id
    assert store.get("clean_alias") is None
    assert candidate_a.evidence_id not in store.aliases()
    assert candidate_b.evidence_id not in store.aliases()
    assert store.get(candidate_a.evidence_id).metadata["aliases_published"] is False
    assert store.get(candidate_b.evidence_id).metadata["aliases_published"] is False


def test_step_success_alias_batch_rejects_internal_explicit_alias_collision(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_text(
        kind="statistic",
        description="First candidate.",
        text='{"estimate": 1}',
        filename="first.json",
        produced_by_step="03_model",
        evidence_id="first_candidate",
        publish_aliases=False,
    )
    second = store.register_text(
        kind="statistic",
        description="Second candidate.",
        text='{"estimate": 2}',
        filename="second.json",
        produced_by_step="03_model",
        evidence_id="second_candidate",
        publish_aliases=False,
    )

    with pytest.raises(ValueError, match="batch alias 'primary_result'.*both"):
        store.publish_step_success_aliases(
            {
                first.evidence_id: ["primary_result", "first_only"],
                second.evidence_id: ["primary_result", "second_only"],
            },
            step_id="03_model",
        )

    assert store.aliases() == {}
    assert store.get("first_only") is None
    assert store.get("second_only") is None
    assert store.get(first.evidence_id).metadata["aliases_published"] is False
    assert store.get(second.evidence_id).metadata["aliases_published"] is False


def test_step_success_alias_batch_rejects_internal_basename_collision(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_text(
        kind="table",
        description="First candidate with shared source name.",
        text="term,estimate\na,1\n",
        filename="shared_results.csv",
        produced_by_step="03_model",
        evidence_id="first_shared_candidate",
        publish_aliases=False,
    )
    second = store.register_text(
        kind="table",
        description="Second candidate with shared source name.",
        text="term,estimate\nb,2\n",
        filename="shared_results.csv",
        produced_by_step="03_model",
        evidence_id="second_shared_candidate",
        publish_aliases=False,
    )

    with pytest.raises(ValueError, match="batch alias 'shared_results'.*both"):
        store.publish_step_success_aliases(
            {
                first.evidence_id: ["first_result"],
                second.evidence_id: ["second_result"],
            },
            step_id="03_model",
        )

    assert store.aliases() == {}
    assert store.get("first_result") is None
    assert store.get("second_result") is None
    assert store.get(first.evidence_id).metadata["aliases_published"] is False
    assert store.get(second.evidence_id).metadata["aliases_published"] is False


def test_step_success_alias_batch_allows_idempotent_alias_repetition(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    record = store.register_text(
        kind="statistic",
        description="One candidate with repeated declarations.",
        text='{"estimate": 1}',
        filename="primary_result.json",
        produced_by_step="03_model",
        evidence_id="primary_result_candidate",
        publish_aliases=False,
    )

    published = store.publish_step_success_aliases(
        {
            record.evidence_id: [
                "primary_result",
                "primary_result",
                record.evidence_id,
            ]
        },
        step_id="03_model",
    )

    assert published[record.evidence_id]["primary_result"] == record.evidence_id
    assert store.get("primary_result").evidence_id == record.evidence_id
    assert store.get(record.evidence_id).metadata["aliases_published"] is True


def test_retire_step_current_aliases_removes_only_explicit_same_step_authority(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_text(
        kind="statistic",
        description="First current product for the revalidated step.",
        text='{"estimate": 1}',
        filename="first.json",
        produced_by_step="03_model",
        evidence_id="first_current_product",
        aliases=["primary_association"],
    )
    second = store.register_text(
        kind="table",
        description="Second current product for the revalidated step.",
        text="term,estimate\nexposure,1\n",
        filename="second.csv",
        produced_by_step="03_model",
        evidence_id="second_current_product",
        aliases=["primary_results_table"],
    )
    historical = store.register_text(
        kind="log",
        description="Unretired history from the same step.",
        text="history",
        filename="history.txt",
        produced_by_step="03_model",
        evidence_id="same_step_history",
        aliases=["same_step_history_alias"],
    )

    retired = store.retire_step_current_aliases(
        [first.evidence_id, second.evidence_id],
        step_id="03_model",
    )

    assert retired["primary_association"] == first.evidence_id
    assert retired["primary_results_table"] == second.evidence_id
    assert store.get("primary_association") is None
    assert store.get("primary_results_table") is None
    assert store.get(first.evidence_id).metadata["aliases_published"] is False
    assert store.get(second.evidence_id).metadata["aliases_published"] is False
    assert store.get("same_step_history_alias").evidence_id == historical.evidence_id
    assert store.get(historical.evidence_id).metadata["aliases_published"] is True

    reloaded = ra.EvidenceStore(root=tmp_path)
    assert reloaded.get("primary_association") is None
    assert reloaded.get(first.evidence_id).metadata["aliases_published"] is False
    assert reloaded.get("same_step_history_alias").evidence_id == historical.evidence_id


def test_retire_step_current_aliases_preserves_cross_step_and_run_level_aliases(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    current = store.register_text(
        kind="statistic",
        description="Authority to retire.",
        text='{"estimate": 1}',
        filename="current.json",
        produced_by_step="03_model",
        evidence_id="current_product",
        aliases=["current_result"],
    )
    other_step = store.register_text(
        kind="statistic",
        description="Authority from a different step.",
        text='{"estimate": 2}',
        filename="other.json",
        produced_by_step="04_model",
        evidence_id="other_step_product",
        aliases=["other_step_result"],
    )
    run_level = store.register_text(
        kind="log",
        description="Run-level authority.",
        text="run authority",
        filename="run_authority.txt",
        evidence_id="run_level_product",
        aliases=["run_level_result"],
    )

    store.retire_step_current_aliases([current.evidence_id], step_id="03_model")

    assert store.get("current_result") is None
    assert store.get("other_step_result").evidence_id == other_step.evidence_id
    assert store.get("run_level_result").evidence_id == run_level.evidence_id
    assert store.get(other_step.evidence_id).metadata["aliases_published"] is True
    assert store.get(run_level.evidence_id).metadata["aliases_published"] is True


def test_retire_step_current_aliases_rejects_invalid_owner_without_partial_mutation(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    owned = store.register_text(
        kind="statistic",
        description="Valid member of the requested retirement batch.",
        text='{"estimate": 1}',
        filename="owned.json",
        produced_by_step="03_model",
        evidence_id="owned_product",
        aliases=["owned_result"],
    )
    foreign = store.register_text(
        kind="statistic",
        description="Invalid foreign member of the retirement batch.",
        text='{"estimate": 2}',
        filename="foreign.json",
        produced_by_step="04_model",
        evidence_id="foreign_product",
        aliases=["foreign_result"],
    )
    aliases_before = store.aliases()

    with pytest.raises(ValueError, match="owned by step '04_model'"):
        store.retire_step_current_aliases(
            [owned.evidence_id, foreign.evidence_id],
            step_id="03_model",
        )

    assert store.aliases() == aliases_before
    assert store.get(owned.evidence_id).metadata["aliases_published"] is True
    assert store.get(foreign.evidence_id).metadata["aliases_published"] is True


def test_retire_step_current_aliases_rolls_back_memory_when_save_fails(
    ra, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ra.EvidenceStore(root=tmp_path)
    record = store.register_text(
        kind="statistic",
        description="Authority whose retirement cannot be persisted.",
        text='{"estimate": 1}',
        filename="current.json",
        produced_by_step="03_model",
        evidence_id="current_product",
        aliases=["current_result"],
    )
    aliases_before = store.aliases()

    def fail_save() -> None:
        raise OSError("persistence unavailable")

    monkeypatch.setattr(store, "_save", fail_save)
    with pytest.raises(OSError, match="persistence unavailable"):
        store.retire_step_current_aliases(
            [record.evidence_id],
            step_id="03_model",
        )

    assert store.aliases() == aliases_before
    assert store.get(record.evidence_id).metadata["aliases_published"] is True


def test_first_write_wins_on_alias_collision(ra, tmp_path: Path):
    a = tmp_path / "table_one.csv"
    a.write_text("a\n1\n")
    b = tmp_path / "redo"
    b.mkdir()
    b = b / "table_one.csv"
    b.write_text("a\n2\n")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(kind="table", description="first", source_path=a)
    second = store.register_file(kind="table", description="second", source_path=b)
    # Alias still points at the first registration.
    assert store.get("table_one").evidence_id == first.evidence_id
    # And the second record exists under its hash-suffixed evidence_id.
    assert store.get(second.evidence_id) is not None


def test_register_file_new_id_preserves_existing_blob(ra, tmp_path: Path):
    first_src = tmp_path / "figure.png"
    first_src.write_text("old", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(
        kind="figure",
        description="first",
        source_path=first_src,
        evidence_id="publication_figure_png",
    )
    first_blob = tmp_path / first.relative_path
    first_sha = first.sha256

    second_src = tmp_path / "rerun" / "figure.png"
    second_src.parent.mkdir()
    second_src.write_text("new", encoding="utf-8")
    second = store.register_file(
        kind="figure",
        description="second",
        source_path=second_src,
        evidence_id="publication_figure_png",
        on_sha_change="new_id",
    )

    assert second.evidence_id == "publication_figure_png_v2"
    assert first_blob.read_text(encoding="utf-8") == "old"
    assert store.get(first.evidence_id).sha256 == first_sha
    assert (tmp_path / second.relative_path).read_text(encoding="utf-8") == "new"


def test_bind_manuscript_replaces_known_placeholders(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    store.register_file(kind="table", description="t1", source_path=src)
    bound = store.bind_manuscript(
        "Cohort: {evidence:table_one}. Missing piece: {evidence:does_not_exist}."
    )
    assert "table_one" in bound
    assert "[evidence missing: does_not_exist]" in bound
    # Ensure the resolved placeholder embeds the relative path + sha
    assert "sha256=" in bound


def test_bind_manuscript_accepts_double_brace_writer_placeholders(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    store.register_file(kind="table", description="t1", source_path=src)

    bound = store.bind_manuscript(
        "Cohort: {{evidence:table_one}}. Single: {evidence:table_one}."
    )

    assert bound.count("[table_one]") == 2
    assert "{[" not in bound
    assert "]}" not in bound
    assert "{{evidence:" not in bound
    assert "{evidence:" not in bound


def test_bind_manuscript_supports_comma_separated_placeholders(ra, tmp_path: Path):
    first = tmp_path / "cluster_characteristics.csv"
    second = tmp_path / "cluster_mortality.csv"
    first.write_text("cluster,n\n0,10\n", encoding="utf-8")
    second.write_text("cluster,mortality\n0,0.1\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    store.register_file(
        kind="table",
        description="cluster characteristics",
        source_path=first,
        aliases=["cluster_characteristics"],
    )
    store.register_file(
        kind="table",
        description="cluster mortality",
        source_path=second,
        aliases=["cluster_mortality"],
    )

    bound = store.bind_manuscript(
        "Tables: {evidence:cluster_characteristics, cluster_mortality}."
    )

    assert "cluster_characteristics" in bound
    assert "cluster_mortality" in bound
    assert "evidence missing" not in bound


def test_bind_manuscript_strips_repeated_evidence_prefix_in_comma_items(
    ra, tmp_path: Path
):
    first = tmp_path / "cluster_characteristics.csv"
    second = tmp_path / "cluster_mortality.csv"
    first.write_text("cluster,n\n0,10\n", encoding="utf-8")
    second.write_text("cluster,mortality\n0,0.1\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    store.register_file(
        kind="table",
        description="cluster characteristics",
        source_path=first,
        aliases=["cluster_characteristics"],
    )
    store.register_file(
        kind="table",
        description="cluster mortality",
        source_path=second,
        aliases=["cluster_mortality"],
    )

    bound = store.bind_manuscript(
        "Tables: {evidence:cluster_characteristics, evidence:cluster_mortality}."
    )

    assert "cluster_characteristics" in bound
    assert "cluster_mortality" in bound
    assert "evidence missing" not in bound


def test_register_text_alias_survives_evidence_id_with_double_underscore(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_text(
        kind="log",
        description="interpretation",
        text="A long-step interpretation.",
        filename="interpretation_04_model_fitting_reduced_variable.md",
    )

    assert "__" in rec.evidence_id
    assert store.get("interpretation_04_model_fitting_reduced_variable") is not None
    bound = store.bind_manuscript(
        "See {evidence:interpretation_04_model_fitting_reduced_variable}."
    )
    assert "evidence missing" not in bound


def test_resolvable_names_includes_aliases_and_ids(ra, tmp_path: Path):
    src = tmp_path / "missingness.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="table", description="m", source_path=src, aliases=["my_alias"]
    )
    names = set(store.resolvable_names())
    assert rec.evidence_id in names
    assert "missingness" in names
    assert "my_alias" in names


def test_aliases_are_persisted(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    store1 = ra.EvidenceStore(root=tmp_path)
    rec = store1.register_file(kind="table", description="t1", source_path=src)
    # New store instance should reload aliases from disk.
    store2 = ra.EvidenceStore(root=tmp_path)
    got = store2.get("table_one")
    assert got is not None and got.evidence_id == rec.evidence_id


def test_evidence_id_is_stable_for_same_content(ra, tmp_path: Path):
    src = tmp_path / "stable.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(kind="table", description="stable", source_path=src)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    same = other_dir / "stable.csv"
    same.write_text("a\n1\n", encoding="utf-8")
    second = store.register_file(
        kind="table", description="stable copy", source_path=same
    )
    assert first.evidence_id == second.evidence_id


def test_bind_manuscript_hides_warning_caveat_in_default_mode(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(kind="table", description="t1", source_path=src)
    store.update_record(
        rec.evidence_id,
        finding_severity="warning",
        finding_messages=["example warning"],
    )
    bound = store.bind_manuscript("See {evidence:table_one}.")
    assert "(warning: see manifest)" not in bound
    assert "<!-- warning: see manifest -->" in bound


def test_bind_manuscript_verbose_mode_keeps_warning_caveat_visible(ra, tmp_path: Path):
    src = tmp_path / "table_one.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(kind="table", description="t1", source_path=src)
    store.update_record(
        rec.evidence_id,
        finding_severity="warning",
        finding_messages=["example warning"],
    )
    bound = store.bind_manuscript("See {evidence:table_one}.", verbose=True)
    assert "(warning: see manifest)" in bound


def test_update_record_can_clear_resolved_finding_caveat(ra, tmp_path: Path):
    src = tmp_path / "result.json"
    src.write_text('{"estimate": 1.2}\n', encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(kind="statistic", description="result", source_path=src)
    store.update_record(
        rec.evidence_id,
        finding_severity="error",
        finding_messages=["transient validation error"],
    )
    store.update_record(
        rec.evidence_id,
        finding_severity=None,
        finding_messages=[],
    )

    reloaded = ra.EvidenceStore(root=tmp_path).get(rec.evidence_id)
    assert reloaded is not None
    assert reloaded.finding_severity is None
    assert reloaded.finding_messages == []
    assert "see manifest" not in store.bind_manuscript(
        f"See {{evidence:{rec.evidence_id}}}."
    )


def test_enforce_evidence_bound_scaffold_filters_unsupported_result_sentences(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    scaffold = (
        "# Results\n\n"
        "The cohort comprised 10 stays.\n"
        "This study describes baseline characteristics.\n"
        "Median age was 65 years {evidence:table_one}.\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert "The cohort comprised 10 stays." not in filtered
    assert "This study describes baseline characteristics." in filtered
    assert "Median age was 65 years {evidence:table_one}." in filtered
    assert removed == ["The cohort comprised 10 stays."]


def test_enforce_evidence_bound_scaffold_filters_bold_section_result_sentences(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    scaffold = (
        "**Background:** The relation between early SOFA-2 severity and ICU mortality "
        "remains sensitive to missingness and component-completeness artefacts.\n"
        "**Discussion:** Several mechanisms could be consistent with the observed "
        "association, although none can be separated definitively.\n"
        "**Results:** Median age was 65 years {evidence:table_one}.\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert "**Results:** Median age was 65 years {evidence:table_one}." in filtered
    assert "**Background:**" not in filtered
    assert "**Discussion:**" not in filtered
    assert any(
        "missingness and component-completeness artefacts" in item for item in removed
    )
    assert any("consistent with the observed association" in item for item in removed)


def test_enforce_evidence_bound_scaffold_does_not_exempt_list_or_quote_claims(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    scaffold = (
        "## Mortality results\n"
        "- **Results:** Mortality was lower in the intervention arm.\n"
        "> Mortality was higher after adjustment.\n"
        "- This section explains the prespecified study design.\n"
        "> Context for the analysis is described here.\n"
        "- **Results:** Mortality was lower {evidence:primary_result}.\n"
        "- **Results:**\n"
    )

    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)

    assert "## Mortality results" in filtered
    assert "Mortality was lower in the intervention arm" not in filtered
    assert "Mortality was higher after adjustment" not in filtered
    assert "- This section explains the prespecified study design." in filtered
    assert "> Context for the analysis is described here." in filtered
    assert "- **Results:** Mortality was lower {evidence:primary_result}." not in filtered
    assert "- **Results:**" in filtered
    assert removed == [
        "**Results:** Mortality was lower in the intervention arm.",
        "Mortality was higher after adjustment.",
        "**Results:** Mortality was lower {evidence:primary_result}.",
    ]


def test_enforce_evidence_bound_scaffold_audits_assertive_result_headings(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    scaffold = (
        "## Mortality was lower in the intervention arm.\n"
        "> ## Median age was 65 years.\n"
        "## Mean age was sixty-five years.\n"
        "## Mortality was lower {evidence:primary_result}\n"
    )

    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)

    assert "Mortality was lower in the intervention arm" not in filtered
    assert "Median age was 65 years" not in filtered
    assert "Mean age was sixty-five years" not in filtered
    assert "## Mortality was lower {evidence:primary_result}" not in filtered
    assert removed == [
        "Mortality was lower in the intervention arm.",
        "Median age was 65 years.",
        "Mean age was sixty-five years.",
        "Mortality was lower {evidence:primary_result}",
    ]


def test_enforce_evidence_bound_scaffold_preserves_structural_headings(
    ra, tmp_path: Path
):
    store = ra.EvidenceStore(root=tmp_path)
    scaffold = (
        "# Results\n"
        "## Mortality results\n"
        "> ## Methods and analysis\n"
        "## 2. Sensitivity analyses\n"
    )

    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)

    assert filtered == scaffold
    assert removed == []


@pytest.mark.parametrize(
    ("evidence_id", "filename"),
    [
        ("../../escaped", "artifact.txt"),
        ("/tmp/escaped", "artifact.txt"),
        ("safe_id", "../../escaped.txt"),
        ("safe_id", "/tmp/escaped.txt"),
    ],
)
def test_evidence_registration_rejects_path_escape(
    ra,
    tmp_path: Path,
    evidence_id: str,
    filename: str,
):
    store = ra.EvidenceStore(root=tmp_path / "run")

    with pytest.raises(ValueError, match="single safe path component"):
        store.register_text(
            kind="log",
            description="escape probe",
            text="probe",
            filename=filename,
            evidence_id=evidence_id,
        )

    assert not (tmp_path / "escaped.txt").exists()


def test_evidence_store_rejects_evidence_directory_symlink_escape(ra, tmp_path: Path):
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    run_dir.mkdir()
    outside.mkdir()
    (run_dir / "evidence").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link|escapes the store root"):
        ra.EvidenceStore(root=run_dir)


@pytest.mark.parametrize("operation", ["text", "file", "save"])
def test_evidence_store_revalidates_directory_after_initialisation(
    ra, tmp_path: Path, operation: str
):
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    store = ra.EvidenceStore(root=run_dir)
    store.dir.rmdir()
    store.dir.symlink_to(outside, target_is_directory=True)
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic link|escapes the store root"):
        if operation == "text":
            store.register_text(
                kind="log",
                description="replacement probe",
                text="must stay inside run root",
                filename="probe.txt",
                evidence_id="replacement_probe",
            )
        elif operation == "file":
            store.register_file(
                kind="log",
                description="replacement probe",
                source_path=source,
                evidence_id="replacement_probe",
            )
        else:
            store._save()

    assert list(outside.iterdir()) == []
