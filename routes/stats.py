"""Statistics API routes for the v2 SOC Dashboard.

Provides endpoints for:
- GET /api/model-stats: model performance metrics and running totals
- GET /api/top-attackers: top 20 source IPs by attack event count
- GET /api/attack-types: per-attack-type classification counts
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from models.schemas import ModelStatsResponse, TopAttackerEntry

router = APIRouter()


def _get_stats_accumulator(request: Request):
    """Retrieve StatsAccumulator from app state, or None if unavailable."""
    return getattr(request.app.state, "stats_accumulator", None)


@router.get("/api/model-stats", response_model=ModelStatsResponse)
async def get_model_stats(request: Request):
    """Return current model statistics including metrics, totals, and throughput."""
    stats_accumulator = _get_stats_accumulator(request)
    if stats_accumulator is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Stats accumulator not available"},
        )
    return stats_accumulator.get_model_stats()


@router.get("/api/top-attackers", response_model=list[TopAttackerEntry])
async def get_top_attackers(request: Request):
    """Return top 20 source IPs ranked by attack event count."""
    stats_accumulator = _get_stats_accumulator(request)
    if stats_accumulator is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Stats accumulator not available"},
        )
    return stats_accumulator.get_top_attackers(limit=20)


@router.get("/api/attack-types")
async def get_attack_types(request: Request) -> dict[str, int]:
    """Return count per attack type for all classifications since last start."""
    stats_accumulator = _get_stats_accumulator(request)
    if stats_accumulator is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "Stats accumulator not available"},
        )
    return stats_accumulator.get_attack_types()
