# Changelog

## [1.0.0] - 2026-08-12

### Added

**ML Pipeline**
- Data preprocessing pipeline (`ml/preprocess.py`): CSV loading, column cleaning, inf/NaN removal, deduplication, binary label encoding
- Feature selection and scaling (`ml/features.py`): 46-feature subset selection, StandardScaler fitting and persistence
- Model training CLI (`ml/train.py`): Random Forest (balanced) and XGBoost (scale_pos_weight) training, evaluation (precision/recall/F1/ROC-AUC), best-model selection by F1-score
- Sample data generator (`ml/sample_data.py`): 200-row CSV (100 Benign, 100 DDoS) for development

**Backend Services**
- Database service (`services/database.py`): async SQLite with upsert semantics, stats, timeline queries
- Event bus (`services/event_bus.py`): asyncio.Queue-based pub/sub for SSE fanout
- Geolocation service (`services/geolocation.py`): ip-api.com batch lookups with rate limiting (45 req/min)
- Model service (`services/model_service.py`): model/scaler loading, inference with graceful degradation
- Threat feed service (`services/threat_feed.py`): FireHOL blocklist parsing, deduplication, refresh cycle

**API Routes**
- `POST /predict`: network flow classification (returns Benign/DDoS + probability)
- `GET /events`: Server-Sent Events stream with 30s heartbeat
- `GET /api/attacks`: all attack records for initial globe population
- `GET /api/stats`: dashboard statistics (total IPs, countries, attacks/hour)
- `GET /api/timeline`: hourly attack counts (24h)
- `GET /health`: app status, model loaded state, DB record count

**Frontend Dashboard**
- Full-screen 3D globe (Globe.gl) with night-Earth texture and neon green atmosphere
- Real-time attack markers and animated arcs (source → target) with fade-out
- Statistics panel overlay (top-left)
- Live event feed (top-right, max 50 items)
- Timeline bar chart (bottom, 24h hourly buckets)
- Marker click tooltips (IP, country, city, ISP, last-seen)
- Auto-rotation when idle (10s timeout)
- Cyberpunk theme: pitch-black (#000000), neon green (#00FF41), JetBrains Mono

**Application Wiring**
- FastAPI app with startup/shutdown lifecycle (DB init, model load, APScheduler)
- APScheduler for periodic threat feed refresh (configurable interval)
- Global exception handler (500 JSON responses)
- Graceful degradation: dashboard works without trained model

**Testing**
- 142 tests total (unit + property-based)
- 15 correctness properties validated via Hypothesis (100 examples each)
- Covers: preprocessing, features, training, prediction, geolocation, database, event bus, SSE

**Configuration**
- `config.py` loading from `.env` with sensible defaults
- `.env.example` with documented configuration options

### Known Issues
- XGBoost requires `n_jobs=1` on Python 3.14 (deadlocks with parallel threading)
- FireHOL level1 blocklist is mostly CIDRs; individual IP yield is low — consider adding level2/level3
- Starlette 1.3.x changed `TemplateResponse` API — uses keyword args (`name=`, `request=`)
