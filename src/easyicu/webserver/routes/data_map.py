"""Read-only data-map HTTP adapter; no export coordinates are accepted."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from easyicu.webserver.data_map import DataMapError, build_data_map
from easyicu.webserver.data_map_contracts import DataMap

router = APIRouter()
Selector = Annotated[str | None, Query(max_length=160)]
Offset = Annotated[int, Query(ge=0, le=10000)]
Limit = Annotated[int, Query(ge=1, le=100)]


@router.get("/api/data-map", response_model=DataMap)
def get_data_map(
    database: Selector = None, concept: Selector = None, family: Selector = None,
    concept_offset: Offset = 0, concept_limit: Limit = 40,
    source_offset: Offset = 0, source_limit: Limit = 20,
) -> DataMap:
    try:
        return build_data_map(database=database, concept=concept, family=family,
                              concept_offset=concept_offset, concept_limit=concept_limit,
                              source_offset=source_offset, source_limit=source_limit)
    except DataMapError as exc:
        raise HTTPException(status_code=400, detail={"error": exc.code}) from None
