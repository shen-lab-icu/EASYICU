"""Attention transport shared by browser notifications and the desktop shell."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from easyicu.webserver.attention_projection import CLIENT_IDENTIFIER, MAX_SEQUENCE
from easyicu.webserver.attention_service import attention_for_service
from easyicu.webserver.attention_store import MAX_PAGE, AttentionUnavailable
from easyicu.webserver.pi_copilot import get_pi_copilot_service

router = APIRouter()


class AttentionAcknowledgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(pattern=f"^{CLIENT_IDENTIFIER}$")
    sequences: list[Annotated[StrictInt, Field(ge=1, le=MAX_SEQUENCE)]] = Field(min_length=1, max_length=MAX_PAGE)


def _feed():
    return attention_for_service(get_pi_copilot_service())


@router.get("/api/attention")
def attention_feed(
    response: Response,
    client_id: Annotated[str, Query(pattern=f"^{CLIENT_IDENTIFIER}$")],
    after: Annotated[int, Query(ge=0, le=MAX_SEQUENCE)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = MAX_PAGE,
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return _feed().page(client_id=client_id, after=after, limit=limit)
    except AttentionUnavailable as exc:
        raise HTTPException(503, detail={"error": "attention_store_unavailable"}) from exc
    except ValueError as exc:
        raise HTTPException(422, detail={"error": str(exc)}) from exc


@router.post("/api/attention/ack")
def acknowledge_attention(body: AttentionAcknowledgement, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return _feed().acknowledge(client_id=body.client_id, sequences=body.sequences)
    except AttentionUnavailable as exc:
        raise HTTPException(503, detail={"error": "attention_store_unavailable"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": str(exc)}) from exc
