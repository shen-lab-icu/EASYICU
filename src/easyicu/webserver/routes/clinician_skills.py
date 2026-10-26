"""Read-only clinician skill projections and scalar planning calculations."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute

from easyicu.research_agent import concept_catalog
from easyicu.research_agent.skill_packages import PlanningInputError
from easyicu.webserver.skill_planning import PlanningRequest, calculate


class SkillInputRoute(APIRoute):
    """Keep rejected non-finite JSON inputs out of the 422 response body."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                detail = [
                    {key: error[key] for key in ("loc", "type", "msg")}
                    for error in exc.errors()
                ]
                raise HTTPException(422, detail=detail) from exc

        return handler


router = APIRouter(prefix="/api/capabilities/method-skills", route_class=SkillInputRoute)
MAX_CONCEPTS = 256


@router.get("/clinical-concept-definitions/projection")
def get_clinical_definitions(
    concepts: Annotated[list[str] | None, Query()] = None,
    database: str | None = None,
) -> dict:
    """Omit concepts for all registered composites; repeat it for a subset."""
    try:
        if concepts is not None and len(concepts) > MAX_CONCEPTS:
            raise concept_catalog.DefinitionProjectionError("definition.concepts_required")
        return concept_catalog.clinical_definitions(concepts, database=database)
    except concept_catalog.DefinitionProjectionError as exc:
        raise HTTPException(422, detail={"error": exc.reason_code}) from exc


@router.get("/six-database-cohort-extraction/projection")
def get_extraction_projection(database: str | None = None) -> dict:
    """Return packaged extraction metadata and an unsubmitted owner prefill."""
    try:
        return concept_catalog.extraction_projection(database)
    except concept_catalog.DefinitionProjectionError as exc:
        raise HTTPException(422, detail={"error": exc.reason_code}) from exc


@router.post("/sample-size-and-power/calculate")
def post_sample_size_calculation(body: PlanningRequest) -> dict:
    """Return one planning result with its method identity and claim ceiling."""
    try:
        return calculate(body)
    except PlanningInputError as exc:
        raise HTTPException(422, detail={"error": exc.reason_code}) from exc
