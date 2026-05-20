"""EvidenceStore: hashing, alias resolution, manuscript binding (T1.2).

These tests pin the behaviour the writer agent depends on. If the
alias system regresses, manuscripts immediately fill up with
``[evidence missing: …]`` markers — exactly what T1.2 fixed.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_register_file_creates_index_and_hash(ra, tmp_path: Path):
    src = tmp_path / "src.csv"
    src.write_text("a,b\n1,2\n", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="table", description="a tiny csv",
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
        kind="table", description="t1",
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
    src.write_text("{\"x\":1}", encoding="utf-8")
    store = ra.EvidenceStore(root=tmp_path)
    rec = store.register_file(
        kind="statistic", description="summary",
        source_path=src, aliases=["outcome_rate", "outcome_incidence"],
    )
    for name in ("outcome_rate", "outcome_incidence"):
        got = store.get(name)
        assert got is not None and got.evidence_id == rec.evidence_id


def test_first_write_wins_on_alias_collision(ra, tmp_path: Path):
    a = tmp_path / "table_one.csv"; a.write_text("a\n1\n")
    b = tmp_path / "redo" ; b.mkdir(); b = b / "table_one.csv"; b.write_text("a\n2\n")
    store = ra.EvidenceStore(root=tmp_path)
    first = store.register_file(kind="table", description="first", source_path=a)
    second = store.register_file(kind="table", description="second", source_path=b)
    # Alias still points at the first registration.
    assert store.get("table_one").evidence_id == first.evidence_id
    # And the second record exists under its hash-suffixed evidence_id.
    assert store.get(second.evidence_id) is not None


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


def test_bind_manuscript_strips_repeated_evidence_prefix_in_comma_items(ra, tmp_path: Path):
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


def test_register_text_alias_survives_evidence_id_with_double_underscore(ra, tmp_path: Path):
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
    rec = store.register_file(kind="table", description="m", source_path=src,
                              aliases=["my_alias"])
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
    second = store.register_file(kind="table", description="stable copy", source_path=same)
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


def test_enforce_evidence_bound_scaffold_filters_unsupported_result_sentences(ra, tmp_path: Path):
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
        "remains sensitive to missingness and score-zero artefacts.\n"
        "**Discussion:** Several mechanisms could be consistent with the observed "
        "association, although none can be separated definitively.\n"
        "**Results:** Median age was 65 years {evidence:table_one}.\n"
    )
    filtered, removed = store.enforce_evidence_bound_scaffold(scaffold)
    assert "**Results:** Median age was 65 years {evidence:table_one}." in filtered
    assert "**Background:**" not in filtered
    assert "**Discussion:**" not in filtered
    assert any("missingness and score-zero artefacts" in item for item in removed)
    assert any("consistent with the observed association" in item for item in removed)
