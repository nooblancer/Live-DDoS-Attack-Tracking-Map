# Live DDoS Attack Tracking Map

Real-time 3D globe visualization of DDoS attack sources, powered by ML classification and live threat intelligence feeds.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![License](https://img.shields.io/badge/license-MIT-gray)

## What it does

- Trains a binary classifier (Random Forest / XGBoost) on the CIC-DDoS2019 dataset to detect DDoS traffic
- Ingests live malicious IPs from FireHOL blocklists
- Geolocates attack sources via ip-api.com
- Renders an interactive 3D globe (Globe.gl) with animated attack arcs and markers
- Streams updates in real time via Server-Sent Events
- Dark cyberpunk aesthetic — pitch-black background, neon green accents, monospace typography

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-username/Live-DDoS-Attack-Tracking-Map.git
cd Live-DDoS-Attack-Tracking-Map
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# Run the dashboard
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — the globe loads immediately. The threat feed starts pulling IPs on a 5-minute interval. Attack markers and arcs appear as data flows in.

## Training the ML Model (Optional)

The dashboard works without a trained model (threat feed visualization only). To enable the `/predict` endpoint:

```bash
# Place CIC-DDoS2019 CSVs in:
#   data/raw/01-12/  (training day)
#   data/raw/03-11/  (test day)

python -m ml.train
```

This trains both Random Forest and XGBoost, picks the best F1-score model, and saves artifacts to `models/`.

## Project Structure

```
├── main.py                  # FastAPI app, lifecycle, routes
├── config.py                # Settings from .env
├── ml/
│   ├── preprocess.py        # Data loading and cleaning
│   ├── features.py          # Feature selection and scaling
│   ├── train.py             # Model training CLI
│   └── sample_data.py       # Sample data generator
├── services/
│   ├── database.py          # SQLite async operations
│   ├── event_bus.py         # In-process pub/sub for SSE
│   ├── geolocation.py       # ip-api.com batch geolocation
│   ├── model_service.py     # Model loading and inference
│   └── threat_feed.py       # FireHOL blocklist fetching
├── routes/
│   ├── predict.py           # POST /predict
│   ├── events.py            # GET /events (SSE)
│   └── api.py               # GET /api/attacks, /api/stats, /api/timeline
├── templates/
│   └── index.html           # Dashboard template
├── static/
│   ├── js/globe.js          # Globe.gl + SSE client
│   └── css/style.css        # Cyberpunk theme
└── tests/                   # 142 tests (unit + property-based)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard (3D globe) |
| GET | `/api/attacks` | All attack records |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/timeline` | Hourly attack counts (24h) |
| GET | `/events` | SSE stream (live attack events) |
| GET | `/health` | App status, model state, DB count |
| POST | `/predict` | Classify network flow features |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Property-based tests only
pytest tests/ -v -k "props"

# With coverage
pytest tests/ --cov=. --cov-report=html
```

142 tests covering ML preprocessing, feature engineering, model evaluation, API endpoints, database operations, event bus, geolocation, and SSE delivery. Property-based tests use Hypothesis (100 examples each).

## Tech Stack

- **Backend**: FastAPI, uvicorn, aiosqlite, APScheduler
- **ML**: scikit-learn, XGBoost, pandas, NumPy
- **Frontend**: Globe.gl (Three.js), vanilla JS, Server-Sent Events
- **Testing**: pytest, Hypothesis (property-based testing)
- **Data**: CIC-DDoS2019 dataset, FireHOL blocklist-ipsets

## Configuration

See `.env.example` for all settings. Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `FEED_INTERVAL_SEC` | 300 | Threat feed refresh interval |
| `GEO_BATCH_SIZE` | 50 | IPs per geolocation batch |
| `DB_PATH` | `data/attacks.db` | SQLite database location |
| `MODEL_PATH` | `models/ddos_model.joblib` | Trained model path |

## License

MIT
