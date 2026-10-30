"""Source review is path-free and scoped to a unique current registration."""
from easyicu.webserver.pi_copilot.workflow import _source_boundary_review


def _study(database="sic", path="/private/export"):
    return {"id": "study-a", "revision": 3,
            "data_source": {"database": database, "path": path}}


def _registry(database="sic", path="/private/export"):
    return {"sources": [{"id": "src-current", "ok": True, "database": database, "path": path}]}


def test_unique_registered_source_exposes_a_nonexecuting_path_free_fact():
    row = _source_boundary_review(_study(), _registry(), "study-a")
    assert row["status"] == "discussion_only_not_approved_not_executable"
    assert (row["study_context_id"], row["study_revision"], row["registration_id"]) == ("study-a", 3, "src-current")
    assert (row["start_column"], row["end_column"]) == ("ICUOffset", "TimeOfStay")
    assert "/private/" not in str(row)


def test_missing_ambiguous_stale_or_other_source_is_not_projected():
    assert _source_boundary_review(_study(), _registry(), "") is None
    assert _source_boundary_review(_study(), _registry(), "study-other") is None
    assert _source_boundary_review({**_study(), "revision": None}, _registry(), "study-a") is None
    assert _source_boundary_review(_study(), _registry(path="/private/other"), "study-a") is None
    assert _source_boundary_review(_study(), _registry(database="miiv"), "study-a") is None
    duplicate = _registry()
    duplicate["sources"].append(dict(duplicate["sources"][0], id="src-other"))
    assert _source_boundary_review(_study(), duplicate, "study-a") is None
    assert _source_boundary_review(_study("miiv"), _registry("miiv"), "study-a") is None
