"""Every mounted route needs an explicit filesystem/network boundary review.

The fixture is an audit, not generated during tests. New routes/mounts fail
closed until their request fields and downstream owners have been classified.
Behavioral tests cover path-bearing JSON/query aliases before feature dispatch;
test_hosted_route_data_boundary covers indirect coordinates at their owners.
"""

import json
from pathlib import Path

import pytest
from starlette.routing import Mount

from easyicu.webserver import hosted
from easyicu.webserver.app import STATIC_DIR, app
from tests.support.hosted_http import stub_route

INVENTORY = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "hosted_route_inventory.json").read_text()
)


def registered_routes():
    for branch in app.routes:
        yield from (
            branch.effective_route_contexts()
            if hasattr(branch, "effective_route_contexts") else [branch]
        )


def test_every_route_and_mount_has_a_reviewed_boundary():
    observed = []
    for route in registered_routes():
        if isinstance(route, Mount):
            observed.append(("MOUNT", route.path or "/"))
            assert Path(route.app.directory) == STATIC_DIR
            assert not route.app.follow_symlink
        else:
            observed.extend((method, route.path) for method in route.methods)
            if hasattr(route, "dependant"):
                assert any(d.call is hosted.guard_request for d in route.dependant.dependencies), route.path
    reviewed = [(row["method"], row["route"]) for row in INVENTORY]
    assert len(reviewed) == len(set(reviewed)), "Duplicate audit entry"
    assert sorted(observed) == sorted(reviewed), "Classify new routes/mounts in hosted_route_inventory.json"
    for row in INVENTORY:
        assert row["reaches"] and row["protection"]
        if row["protection"] == "no path inputs":
            assert not row["fields"], row


def test_every_route_has_a_reviewed_write_destination():
    # Coverage equality above rejects new routes; even a route without path
    # arguments must declare its downstream writes before it can be reviewed.
    for row in INVENTORY:
        assert row.get("writes") in {"none", "instance home", "refused on hosted"}, row
        assert row.get("write_owner"), row


GUARDED_FIELDS = [
    (row["method"], row["route"], field, row.get("guarded_errors", {}).get(field, "path_outside_data_roots"))
    for row in INVENTORY
    for field in row.get("guarded_fields", [])
]


@pytest.mark.parametrize("method,route,field,error", GUARDED_FIELDS)
def test_inventory_path_fields_are_refused_before_dispatch(tmp_path, monkeypatch, method, route, field, error):
    root = tmp_path / "permitted"
    root.mkdir()
    outside = str(tmp_path / "outside")
    monkeypatch.setenv("EASYICU_HOSTED", "1")
    monkeypatch.setenv("EASYICU_DATA_ROOTS", str(root))
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path / "home"))
    calls = stub_route(monkeypatch, route)
    location, name = field.split(":", 1)
    value = [outside] if name.endswith("[]") else outside
    parts = name.removesuffix("[]").split(".")
    for part in reversed(parts):
        value = {part.removesuffix("[]"): [value] if part.endswith("[]") else value}
    from fastapi.testclient import TestClient

    response = TestClient(app).request(method, route, **{"params" if location == "query" else "json": value})
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == error
    assert calls == []
