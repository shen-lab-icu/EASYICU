import json

import pytest

from easyicu.webserver.release_notes import parse_release_notes, whats_new


def source(*versions):
    return json.dumps({"schema_version": "easyicu.release-notes/1", "entries": [
        {"version": version, "date": "2026-10-26", "changes": [
            {"code": "trust_views", "en": "Release note pending.", "zh": "Release note pending."}
        ]} for version in versions]})


def test_versions_compare_numerically_and_only_acknowledged_versions_are_seen():
    entries = parse_release_notes(source("1.10.0", "1.9.0", "1.2.0"))
    result = whats_new("1.2.0", entries=entries)
    assert [entry["version"] for entry in result["entries"]] == ["1.10.0", "1.9.0"]
    assert result["latest_version"] == "1.10.0" and result["show_after_update"]
    assert not whats_new("1.10.0", entries=entries)["show_after_update"]
    assert not whats_new("2.0.0", entries=entries)["entries"]
    assert len(whats_new(None, entries=entries)["entries"]) == 3


@pytest.mark.parametrize("raw", [source("1.0.0", "1.0.0"), source("not-a-version"),
    source("1.0.0").replace('"zh"', '"fr"'),
    source("1.0.0").replace("2026-10-26", "2026-02-30"),
    '{"schema_version":"unknown","entries":[]}'])
def test_invalid_checked_in_source_fails_closed(raw):
    with pytest.raises(ValueError, match="release_notes_invalid"):
        parse_release_notes(raw)


def test_invalid_last_seen_is_rejected_and_default_source_is_packaged():
    with pytest.raises(ValueError, match="last_seen_version_invalid"):
        whats_new("../private")
    result = whats_new(None)
    assert result["entries"] and result["latest_version"]
    assert set(result["entries"][0]["changes"][0]) == {"code", "zh", "en"}
