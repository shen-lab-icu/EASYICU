"""Replace HTTP feature handlers while retaining routing and dependencies."""


def stub_route(monkeypatch, path):
    from easyicu.webserver.app import app

    calls = []

    def reached(**kwargs):
        calls.append(kwargs)
        return {"reached_owner": True}

    found = False
    for branch in app.routes:
        candidates = branch.effective_route_contexts() if hasattr(branch, "effective_route_contexts") else [branch]
        for route in candidates:
            if getattr(route, "path", None) == path and hasattr(route, "dependant"):
                monkeypatch.setattr(route.dependant, "call", reached)
                found = True
    assert found, path
    return calls
