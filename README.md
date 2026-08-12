# Live DDoS Attack Tracking Map

A multi-panel SOC War Room command center that visualizes DDoS attacks on a 3D globe in real-time. Powered by a 12-class XGBoost classifier trained on the CIC-DDoS2019 dataset and live threat intelligence from multiple feeds.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![Tests](https://img.shields.io/badge/tests-270%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-gray)

## Overview

The dashboard simultaneously streams two data channels:
- **Historical Replay** — CIC-DDoS2019 network flows classified in real-time by the ML model at configurable speeds (1x–1000x)
- **Live Threat Intelligence** — Aggregated from AbuseIPDB, FireHOL L1-3, Feodo Tracker, and Emerging Threats

Six panels update simultaneously: 3D globe with color-coded attack arcs, real-time stats grid, scrolling terminal attack log, top attackers table, 60-minute timeline chart, and model performance card.

## Features

- **12-class DDoS classification** — SYN Flood, UDP Flood, DNS Amplification, HTTP Flood, LDAP, NTP, MSSQL, NetBIOS, SSDP, TFTP, UDPLag, WebDDoS
- **3D Globe visualization** — Globe.gl with color-coded arcs, impact rings, hex-bin heat overlay, 200-arc FIFO cap
- **CRT terminal aesthetic** — Scanlines, phosphor glow, flicker, JetBrains Mono, neon green/cyan
- **Replay engine** — Streams historical flows at 10-50 events/sec with dataset looping
- **Multi-source threat intel** — 4 feeds with CIDR expansion, deduplication, graceful failure handling
- **Property-based testing** — 17 Hypothesis properties validating system correctness
- **Full backward compatibility** — All original v1 endpoints preserved

## Quick Start

```bash
# Clone and install
git clone https://github.com/nooblancer/Live-DDoS-Attack-Tracking-Map.git
cd Live-DDoS-Attack-Tracking-Map
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# Run
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — click the ▶ button to start the replay engine.

## Training the ML Model

The app ships with a pre-trained model on synthetic data. To train on real CIC-DDoS2019:

```bash
# Place CIC-DDoS2019 CSVs in:
#   data/raw/01-12/  (training day)
#   data/raw/03-11/  (test day)

# Train multi-class (12 attack types)
python -m ml.train_multi_class

# Or generate synthetic model for testing
python -m ml.train_multi_class --synthetic
```

The training script handles memory-efficient chunked loading (50k rows/file), trains XGBoost with 300 estimators, and saves artifacts to `ml/models/multi_class/`.

## Project Structure

```
├── main.py                          # FastAPI app, lifecycle, service init
├── config.py                        # Settings from .env
├── ml/
│   ├── train_multi_class.py         # 12-class XGBoost training pipeline
│   ├── train.py                     # Binary model training (v1)
│   ├── preprocess.py                # Data loading and cleaning
│   ├── features.py                  # Feature selection (46 features)
│   └── models/multi_class/          # Trained model artifacts
├── services/
│   ├── replay_engine.py             # Historical flow replay at speed
│   ├── multi_class_classifier.py    # 12-class XGBoost inference
│   ├── threat_aggregator.py         # Multi-source threat intel
│   ├── stats_accumulator.py         # Metrics tracking + persistence
│   ├── geo_pools.py                 # Geographic coordinate assignment
│   ├── event_bus.py                 # Bounded pub/sub for SSE
│   ├── database.py                  # SQLite async operations
│   ├── geolocation.py               # IP → coordinates resolution
│   └── model_service.py             # Binary model service (v1)
├── routes/
│   ├── replay.py                    # POST /api/replay/start|stop
│   ├── stats.py                     # GET /api/model-stats, top-attackers
│   ├── predict.py                   # POST /predict
│   ├── events.py                    # GET /events (SSE stream)
│   └── api.py                       # GET /health, /api/attacks, etc.
├── templates/
│   ├── index.html                   # SOC dashboard
│   └── changelog.html               # Terminal-style changelog
├── static/
│   ├── js/
│   │   ├── globe.js                 # Globe.gl + arc management
│   │   ├── dashboard.js             # SSE consumer + attack log
│   │   ├── panels.js                # Stats grid, timeline, top attackers
│   │   └── controls.js              # Replay controls, model card, legend
│   └── css/style.css                # CRT terminal theme
├── tests/                           # 270 tests (unit + property-based)
└── models/schemas.py                # Pydantic models + attack type defs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | SOC dashboard |
| GET | `/changelog` | Project changelog |
| GET | `/events` | SSE stream (enhanced attack events) |
| GET | `/health` | App status + model state |
| GET | `/api/model-stats` | Model metrics + throughput |
| GET | `/api/top-attackers` | Top 20 source IPs by count |
| GET | `/api/attack-types` | Per-type classification counts |
| GET | `/api/replay/status` | Replay state, speed, flows processed |
| POST | `/api/replay/start` | Start replay (optional speed_multiplier) |
| POST | `/api/replay/stop` | Stop replay |
| POST | `/predict` | Classify a network flow |
| GET | `/api/attacks` | All stored attack records |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/timeline` | Hourly attack counts (24h) |

## Running Tests

```bash
# Full suite (270 tests, ~3 minutes)
python -m pytest tests/ -q

# Property-based tests only (17 files)
python -m pytest tests/test_property_*.py -v

# v1 regression tests
python -m pytest tests/test_api.py tests/test_predict.py -q

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, uvicorn, aiosqlite, APScheduler, httpx |
| ML | XGBoost, scikit-learn, pandas, NumPy |
| Frontend | Globe.gl (Three.js), Chart.js, vanilla JS, SSE |
| Testing | pytest, Hypothesis (property-based), pytest-asyncio |
| Data | CIC-DDoS2019, FireHOL, AbuseIPDB, Feodo, Emerging Threats |

## Configuration

See `.env.example` for all settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `FEED_INTERVAL_SEC` | 300 | v1 threat feed refresh interval |
| `FIREHOL_INTERVAL` | 1800 | Threat aggregator refresh (seconds) |
| `ABUSEIPDB_API_KEY` | — | AbuseIPDB API key (optional) |
| `TARGET_LAT` / `TARGET_LON` | 39.04 / -77.49 | Defended network coordinates |
| `STATS_PERSIST_INTERVAL` | 60 | Stats save interval (seconds) |
| `DB_PATH` | `data/attacks.db` | SQLite database location |

## License

MIT
