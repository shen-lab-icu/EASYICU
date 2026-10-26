"""Search is scoped by the host's instance home, never by a client path/user id."""

from typing import Annotated, Literal

from fastapi import APIRouter, Query

from easyicu.webserver.project_search import INDEX
from easyicu.webserver.search_contracts import MAX_HITS, MAX_QUERY

router = APIRouter()


@router.get("/api/search")
def get_search(
    q: Annotated[str, Query(max_length=MAX_QUERY)] = "",
    scope: Literal["tasks", "files", "projects"] | None = None,
    project: Annotated[str | None, Query(max_length=160)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_HITS)] = 20,
) -> dict:
    return INDEX.search(q, scope=scope, project=project, limit=limit)
