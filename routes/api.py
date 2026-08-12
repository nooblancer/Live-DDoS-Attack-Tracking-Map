"""API data routes for the DDoS Attack Tracking Map dashboard.

Provides endpoints for:
- GET /api/attacks: all attack records for initial globe population
- GET /api/stats: dashboard statistics
- GET /api/timeline: hourly attack counts for last 24h
- GET /health: application health check
"""

from fastapi import APIRouter, Request

from models.schemas import DashboardStats, HealthResponse, TimelineBucket

router = APIRouter()


def _get_db(request: Request):
    """Retrieve DatabaseService from app state."""
    return request.app.state.db


def _get_model_service(request: Request):
    """Retrieve ModelService from app state."""
    return request.app.state.model_service


@router.get("/api/attacks")
async def get_attacks(request: Request) -> list[dict]:
    """Return all attack records for initial globe population."""
    db = _get_db(request)
    return await db.get_all_attacks()


@router.get("/api/stats", response_model=DashboardStats)
async def get_stats(request: Request) -> dict:
    """Return dashboard statistics."""
    db = _get_db(request)
    return await db.get_stats()


@router.get("/api/timeline", response_model=list[TimelineBucket])
async def get_timeline(request: Request) -> list[dict]:
    """Return hourly attack counts for last 24h."""
    db = _get_db(request)
    return await db.get_timeline()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> dict:
    """Return app status, model loaded state, DB record count."""
    model_service = _get_model_service(request)
    db = _get_db(request)

    # Consider either v1 binary model or v2 multi-class model as "loaded"
    multi_class = getattr(request.app.state, "multi_class_classifier", None)
    any_model_loaded = model_service.is_loaded or (
        multi_class is not None and multi_class.is_loaded
    )

    stats = await db.get_stats()
    total_records = stats.get("total_ips", 0)

    return {
        "status": "ok" if any_model_loaded else "degraded",
        "model_loaded": any_model_loaded,
        "database_records": total_records,
    }
