"""Search uses only synthetic host metadata, never result/extraction contents."""

import json
from pathlib import Path

import pytest

from tests.support.project_search import corpus as corpus, write_json


def search_index():
    from easyicu.webserver.project_search import SearchIndex

    return SearchIndex()


def test_english_chinese_and_message_targets(corpus):
    index = search_index()
    english = index.search("IMPROVED", scope="tasks")
    hit = english["hits"][0]
    assert hit["project_id"] == "orchard"
    assert hit["conversation_id"] == "c1"
    assert hit["message_id"] == "a1"
    assert hit["open"] == {"kind": "conversation", "project_id": "orchard",
                            "conversation_id": "c1", "message_id": "a1", "source": "pi"}
    assert index.search("生长差异")["hits"][0]["message_id"] == "u1"
    assert index.search("果园", scope="projects")["hits"][0]["project_id"] == "orchard"
    assert not index.search("privatehiddenword")["hits"]


def test_scope_files_plans_answers_and_no_result_rows(corpus):
    index = search_index()
    for query in ("growth_estimates.csv", "生长估计", "manuscript_scaffold.pdf"):
        hits = index.search(query, scope="files")["hits"]
        assert hits and all(hit["file_id"] and hit["open"]["run_id"] == "r1" for hit in hits)
    assert index.search("Seasonal", scope="tasks")["hits"][0]["kind"] == "plan"
    assert index.search("Canopy", scope="tasks")["hits"][0]["kind"] == "answer"
    assert not index.search("rowsecretword")["hits"]
    assert not index.search("Canopy", scope="files")["hits"]
    assert not index.search("growth", project="meadow")["hits"]


def test_updates_and_deletions_replace_cached_entries(corpus):
    state, _, run, transcript = corpus
    index = search_index()
    assert index.search("improved")["hits"]
    transcript.write_text(transcript.read_text().replace("improved", "declined"))
    assert not index.search("improved")["hits"]
    assert index.search("declined")["hits"]
    (run / "manuscript_scaffold.pdf").unlink()
    assert not index.search("manuscript_scaffold.pdf")["hits"]
    write_json(state / "pi_copilot_sessions.json", {"sessions": []})
    assert not index.search("declined")["hits"]
    write_json(state / "webserver_guided_drafts.json", {"drafts": []})
    assert not index.search("果园")["hits"]
    assert not index.search("Seasonal")["hits"]


def test_patient_files_are_never_opened_or_indexed(corpus, monkeypatch):
    _, projects, run, _ = corpus
    patient = run / "patient_visits.csv"
    patient.write_text("stay_id,secret\n1,neverreadme")
    extraction = projects / "orchard/extraction_rows.parquet"
    extraction.write_bytes(b"neverreadme")
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path not in {patient, extraction, run / "manuscript_scaffold.pdf"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    index = search_index()
    for query in ("patient_visits", "extraction_rows", "neverreadme", "not a real PDF"):
        assert not index.search(query)["hits"]


def test_utf16_offsets_select_original_text_after_an_emoji(corpus):
    hit = search_index().search("improved")["hits"][0]
    assert hit["offset_unit"] == "utf16"
    span = hit["matches"][0]
    encoded = hit["snippet"].encode("utf-16-le")
    assert encoded[span["start"] * 2:span["end"] * 2].decode("utf-16-le") == "improved"


def test_search_is_bounded_and_blank_queries_are_empty(corpus):
    from easyicu.webserver.project_search import SearchIndex, SearchLimits

    index = SearchIndex(limits=SearchLimits(documents=3, text_bytes=512, postings=50))
    result = index.search("growth", limit=1)
    assert len(result["hits"]) <= 1
    assert result["index_truncated"]
    assert index.stats["documents"] <= 3
    assert index.stats["text_bytes"] <= 512
    assert index.stats["postings"] <= 50
    assert not index.search("   ")["hits"]


def test_hosted_index_cannot_follow_other_instance_paths(corpus, tmp_path, monkeypatch):
    state, _, _, _ = corpus
    foreign = tmp_path / "other-user"
    foreign.mkdir()
    write_json(foreign / "guided_draft.json", {"id": "foreign", "title": "Foreignsecret"})
    write_json(state / "webserver_guided_drafts.json", {"drafts": [
        dict(id="foreign", title="Foreignsecret", project_dir=str(foreign)),
    ]})
    # Give this instance a narrower home, so the sibling is genuinely outside it.
    own = tmp_path / "own"
    own.mkdir()
    import shutil

    shutil.move(str(state), str(own / ".easyicu"))
    monkeypatch.setenv("EASYICU_HOME", str(own))
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(foreign))
    assert not search_index().search("Foreignsecret")["hits"]


def test_branch_tool_and_host_instructions_are_not_searchable(corpus):
    _, _, _, path = corpus
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    entries[1]["message"]["content"][0]["text"] += "\n\n[EASYICU_INTERNAL_RESPONSE_LANGUAGE_V1]\nlanguagehiddenword"
    entries.extend([
        {"type": "message", "id": "abandoned", "parentId": "a1",
         "message": {"role": "assistant", "content": "abandonedword"}},
        {"type": "message", "id": "tool", "parentId": "a1",
         "message": {"role": "toolResult", "content": "toolhiddenword"}},
        {"type": "message", "id": "active", "parentId": "tool",
         "message": {"role": "assistant", "content": "activebranchword"}},
    ])
    path.write_text("\n".join(json.dumps(e) for e in entries))
    index = search_index()
    for word in ("languagehiddenword", "abandonedword", "toolhiddenword"):
        assert not index.search(word)["hits"]
    assert index.search("activebranchword")["hits"][0]["message_id"] == "active"


def test_legacy_writer_updates_are_visible_and_message_ids_survive_trimming(corpus, monkeypatch):
    from easyicu.webserver import copilot_sessions as owner

    state, projects, _, _ = corpus
    monkeypatch.setattr(owner, "_CONFIG_DIR", state)
    monkeypatch.setattr(owner, "_CONFIG_PATH", state / "webserver_copilot_sessions.json")
    monkeypatch.setattr(owner, "_PROJECTS_ROOT", projects)
    session = owner.create_session({"scope": "page_guide"})["session"]
    sid = session["id"]
    row = owner._find_session(sid)
    row["messages"] = [dict(role="user", text="Earlier note", created_at="2026-01-01"),
                       dict(role="user", text="Stable woodland question", created_at="2026-01-02")]
    owner._upsert_session(row)
    index = search_index()
    before = index.search("woodland")["hits"][0]
    row["messages"] = row["messages"][1:]
    owner._upsert_session(row)
    after = index.search("woodland")["hits"][0]
    assert before["message_id"] == after["message_id"]
    assert after["open"]["message_index"] == 0
    assert after["conversation_id"] == sid


def test_guided_sessions_and_appended_run_projections(corpus):
    state, projects, run, _ = corpus
    write_json(projects / "orchard/guided_copilot_session.json", {
        "id": "guided1", "messages": [{"role": "assistant", "reply": {"en": "Sketch design", "zh": "生长方案"}}],
    })
    index = search_index()
    assert index.search("生长方案")["hits"][0]["conversation_id"] == "guided1"
    appended = run.with_name("run_append")
    write_json(appended / "run_context.json", {"run_id": "append-one", "question": "Additional season"})
    write_json(appended / "agent_plan.json", {"research_question": "Additional season"})
    assert any(hit["open"].get("run_id") == "append-one" for hit in index.search("Additional")["hits"])
    write_json(appended / "manuscript_draft.json", {"markdown_preview": "Append canopy answer"})
    assert index.search("Append canopy")["hits"][0]["kind"] == "answer"
    assert state.is_dir()


def test_symlinked_sources_never_enter_the_index(corpus):
    state, projects, run, transcript = corpus
    secret = projects / "orchard/secret.json"
    write_json(secret, {"title": "symlinksecret", "summary": "symlinksecret"})
    (run / "agent_plan.json").unlink()
    (run / "agent_plan.json").symlink_to(secret)
    original = transcript.with_suffix(".old")
    transcript.rename(original)
    transcript.symlink_to(original)
    index = search_index()
    assert not index.search("symlinksecret")["hits"]
    assert not index.search("improved")["hits"]
    write_json(state / "webserver_guided_drafts.json", {"drafts": [
        {"id": "linked", "title": "symlinksecret", "project_dir": str(projects / "linked")},
    ]})
    (projects / "linked").symlink_to(projects / "orchard", target_is_directory=True)
    assert not index.search("symlinksecret")["hits"]


def test_reusing_an_index_after_switching_homes_drops_the_previous_user(corpus, tmp_path, monkeypatch):
    index = search_index()
    assert index.search("improved")["hits"]
    second = tmp_path / "second-home"
    second.mkdir()
    monkeypatch.setenv("EASYICU_HOME", str(second))
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    assert not index.search("improved")["hits"]
    assert index.stats["documents"] == 0


def test_oversized_and_malformed_sources_fail_closed(corpus, monkeypatch):
    from easyicu.webserver import search_sources

    _, _, run, _ = corpus
    write_json(run / "agent_plan.json", {"title": "Bad shape", "steps": None})
    index = search_index()
    assert index.search("Bad shape")["hits"]
    monkeypatch.setattr(search_sources, "MAX_TRANSCRIPT_BYTES", 10)
    index = search_index()
    result = index.search("improved")
    assert not result["hits"]
    assert "source_too_large" in result["issues"]
    (run / "agent_plan.json").write_text("{broken")
    assert not index.search("Bad shape")["hits"]


def test_cached_projection_is_reused_until_its_source_changes(corpus, monkeypatch):
    from easyicu.webserver.search_sources import SearchCatalog

    index = search_index()
    assert index.search("improved")["hits"]
    calls = []
    original = SearchCatalog.load

    def loaded(self, source):
        calls.append(source.path.name)
        return original(self, source)

    monkeypatch.setattr(SearchCatalog, "load", loaded)
    index.search("improved")
    assert calls == []
    run = corpus[2]
    write_json(run / "agent_plan.json", {"title": "Revised season"})
    assert index.search("Revised season")["hits"]
    assert calls == ["agent_plan.json"]


def test_identifier_tables_are_withheld(corpus):
    run = corpus[2]
    write_json(run / "result_tables.json", {"tables": [
        {"name": "private_estimates.csv", "title": "Private table", "headers": ["subject_id"]},
    ]})
    assert not search_index().search("private")["hits"]


@pytest.mark.parametrize("kwargs", [{"q": "a" * 201}, {"q": "a", "scope": "all"},
                                   {"q": "a", "limit": 0}, {"q": "a", "limit": 51},
                                   {"q": "a", "project": "p" * 161}])
def test_invalid_query_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):
        search_index().search(**kwargs)


def test_catalog_has_global_discovery_budgets(corpus, monkeypatch):
    from easyicu.webserver import search_sources

    monkeypatch.setattr(search_sources, "MAX_CATALOG_DOCUMENTS", 2)
    catalog = search_sources.SearchCatalog()
    catalog.discover()
    assert len(catalog.documents) <= 2
    assert "catalog_limit" in catalog.issues


def test_cjk_ngrams_verify_adjacency_and_casefold_preserves_offsets():
    from easyicu.webserver.search_contracts import SearchDocument, tokens

    assert tokens("生长差异") >= {"生长", "长差", "差异"}
    doc = SearchDocument("p", "message", "", "🌿 Straße ＧＲＯＷＴＨ", {"message_id": "m"})
    for query, expected in (("STRASSE", "Straße"), ("growth", "ＧＲＯＷＴＨ")):
        hit = doc.hit(query)
        span = hit["matches"][0]
        assert hit["snippet"].encode("utf-16-le")[span["start"] * 2:span["end"] * 2].decode("utf-16-le") == expected


def test_route_owner_uses_the_local_index(corpus, monkeypatch):
    from easyicu.webserver.routes import search

    monkeypatch.setattr(search, "INDEX", search_index())
    assert search.get_search(q="improved", scope="tasks", project="orchard", limit=1)["hits"][0]["message_id"] == "a1"


@pytest.mark.parametrize("field,maximum", [("documents", 1), ("text_bytes", 45), ("postings", 3)])
def test_each_index_budget_limits_admission_independently(corpus, field, maximum):
    from easyicu.webserver.project_search import SearchIndex, SearchLimits

    index = SearchIndex(limits=SearchLimits(**{field: maximum}))
    result = index.search("growth")
    assert result["index_truncated"]
    assert index.stats[field] <= maximum


def test_hit_limit_reports_more_and_never_returns_host_paths(corpus):
    result = search_index().search("growth", limit=1)
    assert len(result["hits"]) == 1
    assert result["has_more"]
    assert str(corpus[0].parent) not in json.dumps(result)


def test_single_cjk_character_is_searchable(corpus):
    assert search_index().search("树")["hits"]


def test_desktop_external_registered_folder_and_legacy_run_folder(corpus, tmp_path):
    state, projects, _, _ = corpus
    external = tmp_path / "chosen-project"
    row = dict(id="chosen", title="Wildflower study", project_dir=str(external))
    write_json(external / "guided_draft.json", row)
    write_json(state / "webserver_guided_drafts.json", {"drafts": [row]})
    write_json(projects / "legacy-study/run_x/run_context.json", {"run_id": "legacy-run", "question": "Fern density"})
    index = search_index()
    assert index.search("Wildflower", scope="projects")["hits"][0]["project_id"] == "chosen"
    assert index.search("Fern")["hits"][0]["open"]["run_id"] == "legacy-run"


def test_cjk_bigrams_do_not_join_distant_fragments(corpus):
    transcript = corpus[3]
    transcript.write_text(transcript.read_text().replace("比较果树生长差异", "生长和长差以及差异"))
    assert not search_index().search("生长差异")["hits"]


def test_initial_project_question_is_searchable_before_any_conversation(corpus):
    state = corpus[0]
    path = state / "webserver_guided_drafts.json"
    payload = json.loads(path.read_text())
    payload["drafts"][0]["question"] = "How does rainfall change tree height?"
    write_json(path, payload)
    hit = search_index().search("rainfall", scope="tasks")["hits"][0]
    assert hit["open"] == {"kind": "project", "project_id": "orchard"}


def test_file_targets_resolve_with_the_existing_run_reference_owner(corpus):
    from easyicu.webserver.run_refs import resolve

    hit = search_index().search("manuscript_scaffold.pdf", scope="files")["hits"][0]
    assert resolve(hit["open"]["run_ref"]) == str(corpus[2])


def test_manuscript_display_title_follows_the_existing_file_guide(corpus):
    run = corpus[2]
    write_json(run / "quality_gate.json", {"gate": {"checks": [{"id": "manuscript_ready", "passed": True}]}})
    hits = search_index().search("manuscript_scaffold.pdf", scope="files")["hits"]
    assert hits[0]["title"] == "Manuscript PDF"
