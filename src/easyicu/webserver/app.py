"""EasyICU web server — FastAPI backend for the native UI.

Migration target (route C, see WEBAPP_MIGRATION_PLAN.md): the static frontend
under ``static/`` is the real product UI (vendored from the easyicu_ui design
repo and evolved here). Python lives behind ``/api/*`` instead of rendering DOM.

Runs locally to preserve the local-first contract (no data upload, local
filesystem access for data roots). This module is Stage 0+1: it serves the
frontend and the first real read-only endpoint, ``/api/catalog``.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from easyicu.webserver.host_security import AllowedHostsMiddleware
from easyicu.webserver.routes.agent import artifact_router as agent_artifact_router
from easyicu.webserver.routes.agent import control_router as agent_control_router
from easyicu.webserver.routes.copilot import router as copilot_router
from easyicu.webserver.routes.extraction import router as extraction_router
from easyicu.webserver.routes.guided import router as guided_router
from easyicu.webserver.routes.ideas import router as ideas_router
from easyicu.webserver.routes.jobs import lifecycle_router as job_lifecycle_router
from easyicu.webserver.routes.jobs import submission_router as job_submission_router
from easyicu.webserver.routes.local_data import router as local_data_router
from easyicu.webserver.routes.page_guide import router as page_guide_router
from easyicu.webserver.routes.reviews import router as reviews_router
from easyicu.webserver.routes.system import router as system_router
from easyicu.webserver.routes.workspaces import router as workspaces_router

STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(title="EasyICU", version="0.1.0")


def _web_allowed_hosts() -> list[str]:
    configured = [
        host.strip()
        for host in os.getenv("EASYICU_WEB_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    ]
    allow_any = os.getenv("EASYICU_WEB_ALLOW_ANY_HOST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if "*" in configured and not allow_any:
        configured = [host for host in configured if host != "*"]
    return configured or ["127.0.0.1", "localhost", "[::1]", "testserver"]


WEB_ALLOWED_HOSTS = _web_allowed_hosts()
app.add_middleware(AllowedHostsMiddleware, allowed_hosts=WEB_ALLOWED_HOSTS)


def _is_local_web_client(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    if peer in {"testclient", "testserver"}:
        return True
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return bool(
        address.version == 6 and address.ipv4_mapped and address.ipv4_mapped.is_loopback
    )


@app.middleware("http")
async def local_clients_only(request: Request, call_next):
    """Keep filesystem and job APIs local until the product has remote auth."""
    if not _is_local_web_client(request):
        return JSONResponse(
            status_code=403,
            content={"detail": "EasyICU WebApp accepts loopback clients only."},
        )
    return await call_next(request)


@app.middleware("http")
async def no_store_native_ui_assets(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/index.html"} or path.startswith(("/js/", "/css/")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


app.include_router(system_router)
app.include_router(local_data_router)
app.include_router(reviews_router)
app.include_router(extraction_router)


app.include_router(workspaces_router)
app.include_router(job_submission_router)
app.include_router(agent_control_router)


app.include_router(guided_router)


app.include_router(copilot_router)
app.include_router(page_guide_router)
app.include_router(ideas_router)
app.include_router(agent_artifact_router)


app.include_router(job_lifecycle_router)


# Static frontend last, mounted at root, with HTML serving so "/" -> index.html.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:  # console-script entry candidate
    import uvicorn

    uvicorn.run("easyicu.webserver.app:app", host="127.0.0.1", port=8502, reload=False)


if __name__ == "__main__":
    main()
