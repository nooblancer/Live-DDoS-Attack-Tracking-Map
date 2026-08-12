"""Replay control routes for the historical attack replay engine.

Provides endpoints for:
- POST /api/replay/start: Begin streaming dataset flows at specified speed
- POST /api/replay/stop: Gracefully stop the replay engine
- GET /api/replay/status: Return current replay state
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from models.schemas import ReplayStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class ReplayStartRequest(BaseModel):
    """Optional request body for POST /api/replay/start."""

    speed_multiplier: int = Field(default=10, description="Speed factor (1, 10, 100, or 1000)")


def _get_replay_engine(request: Request):
    """Retrieve ReplayEngine from app state, or raise 503 if unavailable."""
    replay_engine = getattr(request.app.state, "replay_engine", None)
    if replay_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Replay engine not available",
        )
    return replay_engine


@router.post("/api/replay/start")
async def start_replay(
    request: Request,
    body: Optional[ReplayStartRequest] = None,
) -> dict:
    """Start the historical attack replay engine.

    Accepts an optional JSON body with speed_multiplier (default 10x).
    Returns the started status and configured speed.
    """
    replay_engine = _get_replay_engine(request)

    speed = body.speed_multiplier if body is not None else 10

    await replay_engine.start(speed_multiplier=speed)

    return {"status": "started", "speed": speed}


@router.post("/api/replay/stop")
async def stop_replay(request: Request) -> dict:
    """Stop the replay engine gracefully.

    Returns confirmation that the replay has been stopped.
    """
    replay_engine = _get_replay_engine(request)

    await replay_engine.stop()

    return {"status": "stopped"}


@router.get("/api/replay/status", response_model=ReplayStatusResponse)
async def get_replay_status(request: Request) -> ReplayStatusResponse:
    """Return current replay engine status.

    Includes state (running/stopped), speed multiplier, elapsed time,
    flows processed, loops completed, and current events per second.
    """
    replay_engine = _get_replay_engine(request)

    return replay_engine.get_status()
