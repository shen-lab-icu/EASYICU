"""EasyICU native web application composition root.

This module owns host middleware, ordered route registration, and the static
frontend mount. Feature behavior belongs to domain modules under ``routes/``
and their backing services.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from easyicu.webserver.host_security import (
    PROXY_HEADERS,
    AllowedHostsMiddleware,
    resolve_allowed_hosts,
    trusts_proxy,
)
from easyicu.webserver.routes.agent import artifact_router as agent_artifact_router
from easyicu.webserver.routes.agent import control_router as agent_control_router
from easyicu.webserver.routes.copilot import router as copilot_router
from easyicu.webserver.routes.demo_sources import (
    catalog_router as demo_source_catalog_router,
)
from easyicu.webserver.routes.demo_sources import (
    submission_router as demo_source_submission_router,
)
from easyicu.webserver.routes.extraction import router as extraction_router
from easyicu.webserver.routes.extensions import router as extensions_router
from easyicu.webserver.routes.guided import router as guided_router
from easyicu.webserver.routes.ideas import router as ideas_router
from easyicu.webserver.routes.jobs import lifecycle_router as job_lifecycle_router
from easyicu.webserver.routes.jobs import submission_router as job_submission_router
from easyicu.webserver.routes.local_data import router as local_data_router
from easyicu.webserver.routes.page_guide import router as page_guide_router
from easyicu.webserver.routes.pi_copilot import router as pi_copilot_router
from easyicu.webserver.routes.reviews import router as reviews_router
from easyicu.webserver.routes.study_contexts import router as study_contexts_router
from easyicu.webserver.routes.system import router as system_router
from easyicu.webserver.routes.workspaces import router as workspaces_router

STATIC_DIR = Path(__file__).with_name("static")


def _package_version() -> str:
    """Single source of truth for the version reported by the web API.

    Hardcoding it here made the OpenAPI doc, error reports and run evidence
    disagree with the installed package (0.1.0 vs 1.0.0), which is exactly the
    kind of drift that wastes time during support triage.
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("easyicu")
    except PackageNotFoundError:  # pragma: no cover - source checkout w/o install
        return "0+unknown"


app = FastAPI(title="EasyICU", version=_package_version())


WEB_ALLOWED_HOSTS = resolve_allowed_hosts()
app.add_middleware(AllowedHostsMiddleware, allowed_hosts=WEB_ALLOWED_HOSTS)


def _is_proxied_request(request: Request) -> bool:
    return any(header in request.headers for header in PROXY_HEADERS)


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
    # A loopback peer is only evidence of a local user when nothing sits in
    # front. Put this behind nginx, caddy, an SSH forward or a desktop proxy
    # and every remote request arrives from 127.0.0.1 — the check above passes
    # for the whole internet, and these APIs read and write the filesystem.
    # The forwarding headers are what gives that away; a browser on this
    # machine does not send them.
    if _is_proxied_request(request) and not trusts_proxy():
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "This request was forwarded by a proxy. EasyICU WebApp has no "
                    "remote authentication, so a loopback peer behind a proxy is "
                    "not proof of a local user. Connect directly, or set "
                    "EASYICU_WEB_TRUST_PROXY=1 if the proxy in front of it "
                    "authenticates every request itself."
                )
            },
        )
    return await call_next(request)


@app.middleware("http")
async def native_ui_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )
    response.headers.setdefault(
        "Content-Security-Policy",
        "; ".join(
            (
                "default-src 'self'",
                "base-uri 'none'",
                "object-src 'none'",
                "frame-ancestors 'self'",
                "form-action 'self'",
                "script-src 'self'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: blob:",
                "font-src 'self'",
                "connect-src 'self'",
                "frame-src 'self' blob:",
            )
        ),
    )
    path = request.url.path
    if path in {"/", "/index.html"} or path.startswith(("/js/", "/css/")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    return response


# Registration order is a compatibility contract; keep the root static mount last.
app.include_router(system_router)
app.include_router(extensions_router)
app.include_router(local_data_router)
app.include_router(reviews_router)
app.include_router(extraction_router)
app.include_router(workspaces_router)
app.include_router(study_contexts_router)
app.include_router(demo_source_catalog_router)
app.include_router(job_submission_router)
app.include_router(demo_source_submission_router)
app.include_router(agent_control_router)
app.include_router(guided_router)
app.include_router(copilot_router)
app.include_router(pi_copilot_router)
app.include_router(page_guide_router)
app.include_router(ideas_router)
app.include_router(agent_artifact_router)
app.include_router(job_lifecycle_router)


# Static frontend last, mounted at root, with HTML serving so "/" -> index.html.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# No ``main()`` here on purpose. This module used to carry a second entry point
# that bound port 8502 while the real console script (``easyicu-webapp`` ->
# ``easyicu.webserver.__main__:main``) binds 8765, so the docs, the launcher and
# this file each named a different port. ``__main__.py`` owns host/port.
