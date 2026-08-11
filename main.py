"""FastAPI application entry point.

Initializes the app, mounts static files, configures templates,
manages service lifecycle (startup/shutdown), registers routes,
and provides the dashboard root endpoint.
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from routes import api, events, predict
from services.database import DatabaseService
from services.event_bus import EventBus
from services.geolocation import GeolocationService
from services.model_service import ModelService
from services.threat_feed import ThreatFeedService

logger = logging.getLogger(__name__)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="DDoS Attack Tracking Map")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))

# Scheduler instance (created at module level, started/stopped in lifecycle events)
scheduler = AsyncIOScheduler()


@app.on_event("startup")
async def startup():
    """Initialize DB, load model (graceful if missing), start scheduler."""
    # 1. Initialize DatabaseService
    db = DatabaseService(db_path=config.DB_PATH)
    await db.initialize()
    app.state.db = db
    logger.info("Database initialized at %s", config.DB_PATH)

    # 2. Load ModelService (graceful if files missing)
    model_service = ModelService()
    try:
        model_service.load(
            model_path=config.MODEL_PATH,
            scaler_path=config.SCALER_PATH,
            metadata_path=config.METADATA_PATH,
        )
    except Exception as exc:
        logger.warning("Model loading failed (non-fatal): %s", exc)
    app.state.model_service = model_service

    # Set module-level model_service on predict route
    predict.model_service = model_service

    # 3. Create EventBus
    event_bus = EventBus()
    app.state.event_bus = event_bus

    # 4. Create GeolocationService
    geo_service = GeolocationService()
    app.state.geo_service = geo_service

    # 5. Create ThreatFeedService
    threat_feed = ThreatFeedService(
        geo_service=geo_service,
        db=db,
        event_bus=event_bus,
    )
    app.state.threat_feed = threat_feed

    # 6. Start APScheduler with threat feed refresh job
    scheduler.add_job(
        threat_feed.refresh,
        "interval",
        seconds=config.FEED_INTERVAL_SEC,
        id="threat_feed_refresh",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info(
        "Scheduler started: threat feed refresh every %d seconds",
        config.FEED_INTERVAL_SEC,
    )


@app.on_event("shutdown")
async def shutdown():
    """Stop scheduler, close HTTP clients, close DB."""
    # 1. Stop scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    # 2. Close ThreatFeedService HTTP client
    threat_feed: ThreatFeedService | None = getattr(app.state, "threat_feed", None)
    if threat_feed is not None:
        await threat_feed.close()
        logger.info("ThreatFeedService HTTP client closed")

    # 3. Close GeolocationService HTTP client
    geo_service: GeolocationService | None = getattr(app.state, "geo_service", None)
    if geo_service is not None:
        await geo_service.close()
        logger.info("GeolocationService HTTP client closed")

    # 4. Close DatabaseService
    db: DatabaseService | None = getattr(app.state, "db", None)
    if db is not None:
        await db.close()
        logger.info("Database connection closed")


# --- Register route modules ---
app.include_router(predict.router)
app.include_router(events.router)
app.include_router(api.router)


# --- Global exception handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions, log traceback, return 500 JSON."""
    logger.error(
        "Unhandled exception on %s %s:\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# --- Manual refresh trigger (dev convenience) ---
@app.post("/api/trigger-refresh")
async def trigger_refresh():
    """Manually trigger a threat feed refresh cycle."""
    threat_feed: ThreatFeedService = app.state.threat_feed
    await threat_feed.refresh()
    db: DatabaseService = app.state.db
    stats = await db.get_stats()
    return {"status": "refreshed", "stats": stats}


# --- Root dashboard route ---
@app.get("/")
async def dashboard(request: Request):
    """Serve the Globe.gl dashboard template."""
    return templates.TemplateResponse(name="index.html", request=request)
