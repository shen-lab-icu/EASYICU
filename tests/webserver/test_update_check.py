"""The desktop new-version check: which release counts, and quiet failure."""

from datetime import datetime, timedelta, timezone
import json
import socket
import urllib.error

from fastapi import HTTPException
import pytest

from easyicu.webserver import update_check
from easyicu.webserver.routes import trust_help

PAGE = "https://github.com/shen-lab-icu/EASYICU/releases/tag/"
NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


def release(tag, *, draft=False, prerelease=True, url=None):
    return {"tag_name": tag, "draft": draft, "prerelease": prerelease, "html_url": url or PAGE + tag}


@pytest.fixture(autouse=True)
def desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    monkeypatch.setenv("EASYICU_DEPLOYMENT", "desktop")
    for name in ("EASYICU_HOSTED", update_check.FEED_ENV, update_check.SWITCH_ENV):
        monkeypatch.delenv(name, raising=False)
    update_check.reset_cache()
    yield
    update_check.reset_cache()


def feed(rows, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        return rows
    return fetch


def test_a_published_pre_release_is_an_update_and_other_tags_are_not():
    rows = [
        release("official-demo-data-v1", prerelease=False),  # GitHub's "Latest", not an app version
        release("v0.3.0", draft=True),  # never listed anonymously, but never trusted either
        release("v0.2.0"),
        release("v0.1.5"),
        release("v9.9.9", url="https://example.org/v9.9.9"),
        release("v0.2"),
    ]
    status = update_check.update_status(current="0.1.0", fetch=feed(rows))
    assert status["status"] == "update_available"
    assert status["latest_version"] == "0.2.0"
    assert status["release_url"] == PAGE + "v0.2.0"
    assert status["prerelease"] is True


@pytest.mark.parametrize(("current", "expected"), [("0.9.0", "update_available"), ("0.10.0", "current"), ("0.11.0", "current")])
def test_versions_compare_numerically(current, expected):
    status = update_check.update_status(current=current, fetch=feed([release("v0.2.0"), release("v0.10.0")]))
    assert (status["status"], status["latest_version"]) == (expected, "0.10.0")


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("offline"), socket.timeout("timed out"), TimeoutError(), ConnectionResetError(),
    urllib.error.HTTPError(update_check.DEFAULT_FEED, 403, "rate limited", {}, None),
    json.JSONDecodeError("bad", "x", 0), ValueError("feed_too_large"),
])
def test_an_offline_or_failed_fetch_is_quiet(failure):
    def fetch(url):
        raise failure
    status = update_check.update_status(current="0.1.0", fetch=fetch)
    assert status == {"schema_version": update_check.SCHEMA, "status": "unavailable", "current_version": "0.1.0"}


@pytest.mark.parametrize("reply", [None, {}, {"message": "Not Found"}, [None, "v0.2.0", {"tag_name": "v0.2.0"}]])
def test_a_malformed_reply_is_unavailable(reply):
    assert update_check.update_status(current="0.1.0", fetch=feed(reply))["status"] == "unavailable"


def test_an_unparseable_current_version_never_claims_an_update():
    assert update_check.update_status(current="0+unknown", fetch=feed([release("v0.2.0")]))["status"] == "unavailable"


def test_one_answered_request_per_process_and_a_failure_is_asked_again():
    calls = []

    def offline(url):
        calls.append(url)
        raise urllib.error.URLError("offline")
    assert update_check.update_status(current="0.1.0", fetch=offline)["status"] == "unavailable"
    for _ in range(3):
        status = update_check.update_status(current="0.1.0", fetch=feed([release("v0.2.0")], calls))
        assert status["status"] == "update_available"
    assert calls == [update_check.DEFAULT_FEED] * 2


@pytest.mark.parametrize("environment", [{"EASYICU_DEPLOYMENT": "server"}, {"EASYICU_HOSTED": "1"}])
def test_only_the_desktop_app_asks(monkeypatch, environment):
    if environment.get("EASYICU_DEPLOYMENT"):
        monkeypatch.setenv("EASYICU_DEPLOYMENT", environment["EASYICU_DEPLOYMENT"])
    if environment.get("EASYICU_HOSTED"):
        monkeypatch.setenv("EASYICU_HOSTED", "1")
    calls = []
    assert update_check.update_status(current="0.1.0", fetch=feed([], calls))["status"] == "not_desktop"
    assert calls == []
    with pytest.raises(PermissionError):
        update_check.snooze("0.2.0")


@pytest.mark.parametrize(("value", "allowed"), [
    ("https://mirror.hospital.example/easyicu/releases.json", True),
    ("http://127.0.0.1:9000/releases.json", True),
    ("http://localhost:9000/releases.json", True),
    ("http://mirror.hospital.example/releases.json", False),
    ("file:///etc/passwd", False),
    ("ftp://127.0.0.1/releases.json", False),
])
def test_a_feed_override_is_https_or_loopback(monkeypatch, value, allowed):
    monkeypatch.setenv(update_check.FEED_ENV, value)
    assert (update_check.feed_url() == value) is allowed
    calls = []
    status = update_check.update_status(current="0.1.0", fetch=feed([release("v0.2.0")], calls))
    assert status["status"] == ("update_available" if allowed else "disabled")
    assert calls == ([value] if allowed else [])


def test_the_switch_turns_the_check_off(monkeypatch):
    monkeypatch.setenv(update_check.SWITCH_ENV, "0")
    calls = []
    assert update_check.update_status(current="0.1.0", fetch=feed([release("v0.2.0")], calls))["status"] == "disabled"
    assert calls == []


def test_remind_later_lasts_a_week_for_that_version_only():
    fetch = feed([release("v0.2.0")])
    update_check.snooze("0.2.0", now=NOW)
    assert update_check.update_status(current="0.1.0", fetch=fetch, now=NOW + timedelta(days=6))["status"] == "snoozed"
    assert update_check.update_status(current="0.1.0", fetch=fetch, now=NOW + timedelta(days=8))["status"] == "update_available"
    update_check.reset_cache()
    newer = feed([release("v0.2.0"), release("v0.3.0")])
    assert update_check.update_status(current="0.1.0", fetch=newer, now=NOW + timedelta(days=1))["status"] == "update_available"
    with pytest.raises(ValueError):
        update_check.snooze("../0.2.0")


def test_a_damaged_snooze_record_is_ignored(tmp_path):
    (tmp_path / ".easyicu").mkdir()
    (tmp_path / ".easyicu" / "desktop_update_check.json").write_text("{not json")
    assert update_check.update_status(current="0.1.0", fetch=feed([release("v0.2.0")]))["status"] == "update_available"


def test_download_opens_only_the_checked_release_page():
    opened = []
    with pytest.raises(PermissionError):
        update_check.open_release_page(opener=opened.append)
    update_check.update_status(current="0.1.0", fetch=feed([release("v0.2.0")]))
    assert update_check.open_release_page(opener=lambda url: opened.append(url) or True)["opened"] is True
    assert opened == [PAGE + "v0.2.0"]

    def broken(url):
        raise OSError("no browser")
    assert update_check.open_release_page(opener=broken) == {
        "schema_version": update_check.SCHEMA, "opened": False, "release_url": PAGE + "v0.2.0"}


def test_routes_answer_quietly_and_refuse_bad_input(monkeypatch):
    def offline(url):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(update_check, "fetch_releases", offline)
    response = trust_help.get_update_check()
    assert json.loads(response.body)["status"] == "unavailable"
    assert response.headers["cache-control"] == "no-store"
    with pytest.raises(HTTPException) as caught:
        trust_help.snooze_update({"version": "latest"})
    assert caught.value.status_code == 400
    with pytest.raises(HTTPException) as caught:
        trust_help.open_update_page()
    assert caught.value.status_code == 409
    assert json.loads(trust_help.snooze_update({"version": "0.2.0"}).body)["status"] == "snoozed"
