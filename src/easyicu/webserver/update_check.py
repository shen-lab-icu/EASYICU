"""Whether a newer packaged EasyICU has been published; the desktop app only.

The source is GitHub's public releases list, not ``releases/latest``: v0.x
releases are pre-releases, which ``latest`` skips, and the repository's
"Latest" release is a demo-data bundle, not an app version. Drafts are not
listed to anonymous readers, so a version appears once the maintainer
publishes it. The request is one anonymous GET of public JSON: no cookies,
tokens, project, data or machine identifiers. It is not a model call, so the
provider gate does not apply. Any failure (offline, proxy, rate limit,
malformed reply) is ``unavailable``, which the UI renders as nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
import re
import threading
from typing import Any, Callable
from urllib.parse import urlsplit
import urllib.request
import webbrowser

from easyicu import state_paths
from easyicu.webserver.hosted import deployment_kind

SCHEMA = "easyicu.update-check/1"
REPOSITORY_RELEASES = "https://github.com/shen-lab-icu/EASYICU/releases/"
DEFAULT_FEED = "https://api.github.com/repos/shen-lab-icu/EASYICU/releases?per_page=30"
FEED_ENV = "EASYICU_UPDATE_FEED_URL"  # an IT mirror, or a local QA feed
SWITCH_ENV = "EASYICU_UPDATE_CHECK"  # "0" turns the check off
TIMEOUT_SECONDS = 4.0
MAX_BYTES = 2_000_000
SNOOZE = timedelta(days=7)
_TAG = re.compile(r"v((?:0|[1-9][0-9]{0,5})(?:\.(?:0|[1-9][0-9]{0,5})){2})")

_lock = threading.Lock()
_cached: dict[str, Any] | None = None


def parse_version(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "")
    if not re.fullmatch(r"(?:0|[1-9][0-9]{0,5})(?:\.(?:0|[1-9][0-9]{0,5})){2}", text):
        return None
    return tuple(map(int, text.split(".")))  # type: ignore[return-value]


def newest_release(releases: Any, *, link_prefix: str | None = REPOSITORY_RELEASES) -> dict[str, Any] | None:
    """The highest published ``vX.Y.Z`` app release; other tags are not app versions."""
    if not isinstance(releases, list):
        return None
    best = None
    for row in releases:
        if not isinstance(row, dict) or row.get("draft") is not False:
            continue
        match = _TAG.fullmatch(str(row.get("tag_name") or ""))
        link = str(row.get("html_url") or "")
        if not match or not link.startswith(link_prefix or "https://"):
            continue
        version = parse_version(match.group(1))
        if best is None or version > best[0]:
            best = (version, {"version": match.group(1), "release_url": link,
                              "prerelease": row.get("prerelease") is True})
    return best[1] if best else None


def compare(current: str, latest: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"schema_version": SCHEMA, "status": "unavailable", "current_version": current}
    mine = parse_version(current)
    if mine is None or not latest:
        return payload
    payload.update(latest_version=latest["version"], release_url=latest["release_url"],
                   prerelease=latest["prerelease"])
    payload["status"] = "update_available" if parse_version(latest["version"]) > mine else "current"
    return payload


def feed_url() -> str | None:
    """HTTPS, or HTTP only to a loopback host; anything else turns the check off."""
    raw = str(os.environ.get(FEED_ENV) or "").strip()
    if not raw:
        return DEFAULT_FEED
    parts = urlsplit(raw)
    if parts.scheme == "https" and parts.hostname:
        return raw
    try:
        loopback = parts.hostname is not None and (
            parts.hostname == "localhost" or ipaddress.ip_address(parts.hostname).is_loopback)
    except ValueError:
        loopback = False
    return raw if parts.scheme == "http" and loopback else None


def fetch_releases(url: str) -> Any:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "EasyICU-Desktop"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed or vetted URL
        body = response.read(MAX_BYTES + 1)
    if len(body) > MAX_BYTES:
        raise ValueError("feed_too_large")
    return json.loads(body)


def _snooze_path():
    return state_paths.state_root() / "desktop_update_check.json"


def _snoozed(version: str, now: datetime) -> bool:
    try:
        record = json.loads(_snooze_path().read_text(encoding="utf-8"))
        until = datetime.fromisoformat(record["until"])
        return record.get("version") == version and until.tzinfo is not None and now < until
    except (OSError, ValueError, KeyError, TypeError):
        return False


def update_status(*, current: str | None = None, fetch: Callable[[str], Any] | None = None,
                  now: datetime | None = None) -> dict[str, Any]:
    """One answered check per backend process; a failed one is asked again on the
    next page load. Snoozing is read on every call."""
    global _cached
    if current is None:
        from easyicu import __version__ as current
    if deployment_kind() != "desktop":
        return {"schema_version": SCHEMA, "status": "not_desktop", "current_version": current}
    url = feed_url()
    if os.environ.get(SWITCH_ENV) == "0" or url is None:
        return {"schema_version": SCHEMA, "status": "disabled", "current_version": current}
    with _lock:
        if _cached is None or _cached.get("current_version") != current:
            try:
                releases = (fetch or fetch_releases)(url)
            except Exception:  # offline, proxy, TLS, HTTP or JSON failure: stay quiet
                return compare(current, None)
            _cached = compare(current, newest_release(
                releases, link_prefix=REPOSITORY_RELEASES if url == DEFAULT_FEED else None))
        payload = dict(_cached)
    if payload["status"] == "update_available" and _snoozed(payload["latest_version"], now or datetime.now(timezone.utc)):
        payload["status"] = "snoozed"
    return payload


def snooze(version: str, *, now: datetime | None = None) -> dict[str, Any]:
    if parse_version(version) is None:
        raise ValueError("version_invalid")
    if deployment_kind() != "desktop":
        raise PermissionError("not_desktop")
    until = (now or datetime.now(timezone.utc)) + SNOOZE
    path = _snooze_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"version": version, "until": until.isoformat()}), encoding="utf-8")
    temporary.replace(path)
    return {"schema_version": SCHEMA, "status": "snoozed", "latest_version": version, "until": until.isoformat()}


def open_release_page(*, opener: Callable[[str], bool] = webbrowser.open) -> dict[str, Any]:
    """Open the checked release page in the system browser; never a client URL."""
    with _lock:
        cached = dict(_cached or {})
    if deployment_kind() != "desktop" or cached.get("status") != "update_available":
        raise PermissionError("no_update")
    url = cached["release_url"]
    try:
        opened = bool(opener(url))
    except Exception:
        opened = False
    return {"schema_version": SCHEMA, "opened": opened, "release_url": url}


def reset_cache() -> None:
    global _cached
    with _lock:
        _cached = None
