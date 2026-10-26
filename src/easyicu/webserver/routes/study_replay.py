"""Download one run's shareable replay: aggregates only, checked by the exporter."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from easyicu.webserver import run_refs
from easyicu.webserver.hosted import require_instance_path
from easyicu.webserver.study_replay import ReplayExportError, build_study_replay

router = APIRouter()


@router.post("/api/agent-runs/download-replay")
def post_agent_run_download_replay(body: Dict[str, Any]) -> Response:
    """The replay of the run a history-issued run_ref names; never a client path."""
    path = run_refs.resolve(body.get("run_ref"))
    if path is None:
        raise HTTPException(status_code=404, detail={"ok": False, "error": "run_ref_unknown"})
    require_instance_path(path)
    try:
        bundle = build_study_replay(Path(path))
    except ReplayExportError as exc:
        raise HTTPException(status_code=409, detail={"ok": False, "error": exc.code}) from None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in bundle.items():
            # Fixed timestamps keep the archive byte-stable for the same run.
            archive.writestr(zipfile.ZipInfo(f"easyicu-replay/{name}", date_time=(1980, 1, 1, 0, 0, 0)), data,
                             compress_type=zipfile.ZIP_DEFLATED)
    return Response(content=buffer.getvalue(), media_type="application/zip",
                    headers={"Cache-Control": "no-store",
                             "Content-Disposition": 'attachment; filename="easyicu-replay.zip"'})
