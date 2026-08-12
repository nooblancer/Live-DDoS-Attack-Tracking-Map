# Changelog

All notable changes to this project are documented here.

## [2.0.0] — 2025-09-01

### SOC War Room Command Center — Full Release

- **Multi-panel dashboard**: 6 simultaneous panels (3D globe, stats grid, attack log, top attackers, timeline chart, model card)
- **12-class XGBoost classifier**: Trained on real CIC-DDoS2019 dataset with cross-day evaluation (F1=0.63, Precision=0.91)
- **Dual data streams**: Historical dataset replay at configurable speeds + live threat intelligence from 4 sources
- **Enhanced Globe.gl**: Color-coded arcs by attack type, impact rings, hex-bin heat overlay, 200-arc FIFO cap
- **CRT terminal aesthetics**: Scanlines, phosphor glow, flicker animation, JetBrains Mono, neon green/cyan scheme
- **Replay engine**: Streams CIC-DDoS2019 flows at 1x/10x/100x/1000x with dataset looping
- **Threat aggregator**: AbuseIPDB, FireHOL L1-3, Feodo Tracker, Emerging Threats with CIDR expansion and deduplication
- **Stats accumulator**: Sliding-window throughput, per-IP tracking, SQLite persistence
- **Property-based testing**: 17 Hypothesis properties validating system correctness
- **270 tests total**: 142 original + 128 new, full backward compatibility
- **Changelog page**: Terminal-style `/changelog` route documenting project history

## [1.9.0] — 2025-08-25

### Frontend Dashboard Redesign

- Redesigned `index.html` as 6-panel CSS Grid layout
- CRT effects: scanlines, phosphor glow, flicker animation
- JetBrains Mono typeface with neon green/cyan color scheme
- Responsive breakpoint at 1024px with stacked layout
- Attack log terminal with 200-entry FIFO and blinking cursor
- Replay controls with start/stop buttons and speed selector

## [1.8.0] — 2025-08-18

### API Routes and Service Integration

- `POST /api/replay/start`, `POST /api/replay/stop`, `GET /api/replay/status`
- `GET /api/model-stats`, `GET /api/top-attackers`, `GET /api/attack-types`
- Service initialization in `main.py` with graceful lifecycle management
- APScheduler for periodic threat refresh and stats persistence
- All v1 endpoints preserved — zero breaking changes

## [1.7.0] — 2025-08-11

### Replay Engine and Enhanced EventBus

- `ReplayEngine`: Streams CIC-DDoS2019 flows at configurable speeds with rate control
- Dataset looping with `loops_completed` counter
- `EventBus`: Bounded queue with drop-oldest overflow policy
- Enhanced SSE events with 13 fields + backward-compatible legacy format

## [1.6.0] — 2025-08-04

### Threat Aggregator — Multi-Source Intelligence

- `ThreatAggregator`: Fetches from AbuseIPDB, FireHOL L1-3, Feodo Tracker, Emerging Threats
- CIDR expansion for /24+ ranges (samples 5-10 IPs per range)
- Deduplication retaining highest confidence per unique IP
- Graceful failure handling — skip failed sources, continue with rest

## [1.5.0] — 2025-07-28

### Stats Accumulator and Persistence

- `StatsAccumulator`: Sliding 60s window for predictions/sec
- Per-IP attacker tracking with top-20 ranked output
- Per-attack-type count breakdown
- SQLite persistence and restore across restarts

## [1.4.0] — 2025-07-14

### Multi-Class XGBoost Classifier

- Retrained from binary to 12-class on CIC-DDoS2019 (SYN, UDP, DNS, HTTP, LDAP, NTP, MSSQL, NetBIOS, SSDP, TFTP, UDPLag, WebDDoS)
- Returns confidence scores, classification time, top 3 features
- Graceful degraded mode when model files are missing

## [1.3.0] — 2025-06-30

### Geographic Coordinate Pools

- 32 countries across all continents with 3-5 coordinate pairs each
- Weighted selection favoring botnet-heavy regions
- Attack-type regional bias for realistic source distribution
- Configurable target coordinates via environment variables

## [1.0.0] — 2025-05-15

### Initial Release — Binary DDoS Detection Globe

- FastAPI backend with SQLite storage and SSE event streaming
- Binary XGBoost classifier (Benign vs DDoS) on CIC-DDoS2019
- Globe.gl 3D visualization with animated attack arcs
- Geolocation service for IP → coordinates resolution
- FireHOL Level 1 threat feed integration
- 142 passing tests with property-based testing (Hypothesis)

## [0.0.0] — 2025-03-01

### Project Inception

- Idea conceived: real-time DDoS attack visualization on a 3D globe
- Objective: showcase ML classification with live threat intelligence
- Tech stack selected: Python, FastAPI, XGBoost, Globe.gl, CIC-DDoS2019
- Repository initialized
