"""Tests for routes/replay.py — Replay control endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from models.schemas import ReplayStatusResponse
from routes.replay import router


@pytest_asyncio.fixture
async def app_with_replay_engine():
    """Create a FastAPI app with a mock replay engine on app.state."""
    app = FastAPI()
    app.include_router(router)

    # Create a mock ReplayEngine
    replay_engine = MagicMock()
    replay_engine.start = AsyncMock()
    replay_engine.stop = AsyncMock()
    replay_engine.get_status = MagicMock(
        return_value=ReplayStatusResponse(
            state="stopped",
            speed_multiplier=10,
            elapsed_seconds=0.0,
            flows_processed=0,
            loops_completed=0,
            events_per_second=0.0,
        )
    )

    app.state.replay_engine = replay_engine

    yield app, replay_engine


@pytest_asyncio.fixture
async def app_without_replay_engine():
    """Create a FastAPI app without a replay engine (simulates 503)."""
    app = FastAPI()
    app.include_router(router)
    # No replay_engine set on app.state
    yield app


@pytest_asyncio.fixture
async def client(app_with_replay_engine):
    """Provide an async test client with replay engine available."""
    app, _ = app_with_replay_engine
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def client_no_engine(app_without_replay_engine):
    """Provide an async test client without replay engine (503 scenario)."""
    app = app_without_replay_engine
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestStartReplay:
    """Tests for POST /api/replay/start."""

    @pytest.mark.asyncio
    async def test_start_with_default_speed(self, app_with_replay_engine):
        """Starts replay at default 10x speed when no body provided."""
        app, replay_engine = app_with_replay_engine
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/replay/start")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["speed"] == 10
        replay_engine.start.assert_awaited_once_with(speed_multiplier=10)

    @pytest.mark.asyncio
    async def test_start_with_custom_speed(self, app_with_replay_engine):
        """Starts replay at specified speed multiplier."""
        app, replay_engine = app_with_replay_engine
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/replay/start",
                json={"speed_multiplier": 100},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["speed"] == 100
        replay_engine.start.assert_awaited_once_with(speed_multiplier=100)

    @pytest.mark.asyncio
    async def test_start_returns_503_when_engine_unavailable(self, client_no_engine):
        """Returns 503 when replay engine is not initialized."""
        response = await client_no_engine.post("/api/replay/start")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


class TestStopReplay:
    """Tests for POST /api/replay/stop."""

    @pytest.mark.asyncio
    async def test_stop_replay(self, app_with_replay_engine):
        """Stops the replay engine gracefully."""
        app, replay_engine = app_with_replay_engine
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/replay/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        replay_engine.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_returns_503_when_engine_unavailable(self, client_no_engine):
        """Returns 503 when replay engine is not initialized."""
        response = await client_no_engine.post("/api/replay/stop")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


class TestGetReplayStatus:
    """Tests for GET /api/replay/status."""

    @pytest.mark.asyncio
    async def test_returns_status_when_stopped(self, client):
        """Returns stopped status with default values."""
        response = await client.get("/api/replay/status")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "stopped"
        assert data["speed_multiplier"] == 10
        assert data["elapsed_seconds"] == 0.0
        assert data["flows_processed"] == 0
        assert data["loops_completed"] == 0
        assert data["events_per_second"] == 0.0

    @pytest.mark.asyncio
    async def test_returns_status_when_running(self, app_with_replay_engine):
        """Returns running status with active metrics."""
        app, replay_engine = app_with_replay_engine

        replay_engine.get_status.return_value = ReplayStatusResponse(
            state="running",
            speed_multiplier=100,
            elapsed_seconds=45.5,
            flows_processed=1200,
            loops_completed=2,
            events_per_second=26.3,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/replay/status")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "running"
        assert data["speed_multiplier"] == 100
        assert data["elapsed_seconds"] == 45.5
        assert data["flows_processed"] == 1200
        assert data["loops_completed"] == 2
        assert data["events_per_second"] == 26.3

    @pytest.mark.asyncio
    async def test_status_returns_503_when_engine_unavailable(self, client_no_engine):
        """Returns 503 when replay engine is not initialized."""
        response = await client_no_engine.get("/api/replay/status")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
