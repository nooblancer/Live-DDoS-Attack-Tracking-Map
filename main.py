"""FastAPI application entry point.

Initializes the app, mounts static files, configures templates,
manages service lifecycle (startup/shutdown), registers routes,
and provides the dashboard root endpoint.
"""

import logging
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import config
from routes import api, events, predict, replay, stats
from services.database import DatabaseService
from services.event_bus import EventBus
from services.geo_pools import GeoCoordinatePools
from services.geolocation import GeolocationService
from services.model_service import ModelService
from services.multi_class_classifier import MultiClassClassifier
from services.replay_engine import ReplayEngine
from services.stats_accumulator import StatsAccumulator
from services.threat_aggregator import ThreatAggregator
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
    """Initialize DB, load models (graceful if missing), start scheduler with v2 services."""
    # 1. Initialize DatabaseService
    db = DatabaseService(db_path=config.DB_PATH)
    await db.initialize()
    app.state.db = db
    logger.info("Database initialized at %s", config.DB_PATH)

    # 2. Load ModelService (v1 binary — graceful if files missing)
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

    # 5. Create ThreatFeedService (v1)
    threat_feed = ThreatFeedService(
        geo_service=geo_service,
        db=db,
        event_bus=event_bus,
    )
    app.state.threat_feed = threat_feed

    # --- v2 service initialization ---

    # 6. Load MultiClassClassifier
    multi_class_classifier = MultiClassClassifier()
    multi_class_model_dir = config.BASE_DIR / "ml" / "models" / "multi_class"
    try:
        multi_class_classifier.load(
            model_path=str(multi_class_model_dir / "model.json"),
            scaler_path=str(multi_class_model_dir / "scaler.pkl"),
            metadata_path=str(multi_class_model_dir / "metadata.json"),
        )
    except Exception as exc:
        logger.warning("Multi-class classifier loading failed (non-fatal): %s", exc)
    app.state.multi_class_classifier = multi_class_classifier

    # 7. Create GeoCoordinatePools
    geo_pools = GeoCoordinatePools()
    app.state.geo_pools = geo_pools

    # 8. Create StatsAccumulator and restore persisted state
    stats_accumulator = StatsAccumulator(db=db)
    await stats_accumulator.restore()
    # Populate model metrics from multi-class classifier metadata if available
    if multi_class_classifier.is_loaded:
        raw_metrics = multi_class_classifier._metadata.get("metrics", {})
        # Map metadata keys to StatsAccumulator expected keys
        mapped_metrics = {
            "f1_score": raw_metrics.get("f1", raw_metrics.get("f1_score", 0.0)),
            "precision": raw_metrics.get("precision", 0.0),
            "recall": raw_metrics.get("recall", 0.0),
            "roc_auc": raw_metrics.get("roc_auc", 0.0),
        }
        stats_accumulator.set_model_metadata(
            metrics=mapped_metrics,
            top_features=multi_class_classifier.feature_importances[:10],
        )
    app.state.stats_accumulator = stats_accumulator

    # 9. Create ReplayEngine
    replay_engine = ReplayEngine(
        classifier=multi_class_classifier,
        event_bus=event_bus,
        stats=stats_accumulator,
        geo_pools=geo_pools,
    )
    app.state.replay_engine = replay_engine

    # 10. Create ThreatAggregator (v2 multi-source)
    threat_aggregator = ThreatAggregator(
        geo_service=geo_service,
        db=db,
        event_bus=event_bus,
        stats=stats_accumulator,
    )
    app.state.threat_aggregator = threat_aggregator

    # --- Scheduler setup ---

    # v1 threat feed refresh
    scheduler.add_job(
        threat_feed.refresh,
        "interval",
        seconds=config.FEED_INTERVAL_SEC,
        id="threat_feed_refresh",
        replace_existing=True,
    )

    # v2 ThreatAggregator.refresh_all() — configurable interval (default 1800s)
    threat_aggregator_interval = int(os.getenv("FIREHOL_INTERVAL", "1800"))
    scheduler.add_job(
        threat_aggregator.refresh_all,
        "interval",
        seconds=threat_aggregator_interval,
        id="threat_aggregator_refresh",
        replace_existing=True,
    )

    # v2 StatsAccumulator.persist() — every 60 seconds
    stats_persist_interval = int(os.getenv("STATS_PERSIST_INTERVAL", "60"))
    scheduler.add_job(
        stats_accumulator.persist,
        "interval",
        seconds=stats_persist_interval,
        id="stats_persist",
        replace_existing=True,
    )

    scheduler.start()
    app.state.scheduler = scheduler
    logger.info(
        "Scheduler started: v1 feed refresh every %ds, "
        "v2 threat aggregator every %ds, stats persist every %ds",
        config.FEED_INTERVAL_SEC,
        threat_aggregator_interval,
        stats_persist_interval,
    )


@app.on_event("shutdown")
async def shutdown():
    """Stop scheduler, close HTTP clients, persist stats, close DB."""
    # 1. Stop scheduler
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    # 2. Stop ReplayEngine if running
    replay_engine: ReplayEngine | None = getattr(app.state, "replay_engine", None)
    if replay_engine is not None:
        await replay_engine.stop()
        logger.info("ReplayEngine stopped")

    # 3. Close ThreatAggregator HTTP client
    threat_aggregator: ThreatAggregator | None = getattr(app.state, "threat_aggregator", None)
    if threat_aggregator is not None:
        await threat_aggregator.close()
        logger.info("ThreatAggregator HTTP client closed")

    # 4. Persist StatsAccumulator one final time
    stats_accumulator: StatsAccumulator | None = getattr(app.state, "stats_accumulator", None)
    if stats_accumulator is not None:
        await stats_accumulator.persist()
        logger.info("StatsAccumulator persisted on shutdown")

    # 5. Close ThreatFeedService HTTP client (v1)
    threat_feed: ThreatFeedService | None = getattr(app.state, "threat_feed", None)
    if threat_feed is not None:
        await threat_feed.close()
        logger.info("ThreatFeedService HTTP client closed")

    # 6. Close GeolocationService HTTP client
    geo_service: GeolocationService | None = getattr(app.state, "geo_service", None)
    if geo_service is not None:
        await geo_service.close()
        logger.info("GeolocationService HTTP client closed")

    # 7. Close DatabaseService
    db: DatabaseService | None = getattr(app.state, "db", None)
    if db is not None:
        await db.close()
        logger.info("Database connection closed")


# --- Register route modules ---
app.include_router(predict.router)
app.include_router(events.router)
app.include_router(api.router)
app.include_router(stats.router)
app.include_router(replay.router)


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


@app.get("/changelog")
async def changelog(request: Request):
    """Serve the changelog/blog page."""
    return templates.TemplateResponse(name="changelog.html", request=request)
