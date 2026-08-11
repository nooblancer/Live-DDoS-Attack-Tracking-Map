"""Tests for routes/api.py — API data routes."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from models.schemas import AttackRecord
from routes.api import router
from services.database import DatabaseService
from services.model_service import ModelService


@pytest_asyncio.fixture
async def app_with_db():
    """Create a FastAPI app with test database and model service on app.state."""
    app = FastAPI()
    app.include_router(router)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_attacks.db")
        db = DatabaseService(db_path)
        await db.initialize()

        model_service = ModelService()
        # model_service.is_loaded defaults to False (no model loaded)

        app.state.db = db
        app.state.model_service = model_service

        yield app, db, model_service

        await db.close()


@pytest_asyncio.fixture
async def client(app_with_db):
    """Provide an async test client."""
    app, _, _ = app_with_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def populated_app(app_with_db):
    """Insert sample attack records into the database."""
    app, db, model_service = app_with_db

    records = [
        AttackRecord(
            ip_address="1.2.3.4",
            latitude=51.5,
            longitude=-0.1,
            country="United Kingdom",
            city="London",
            isp="BT",
            threat_source="firehol_level1",
            first_seen=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            last_seen=datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        ),
        AttackRecord(
            ip_address="5.6.7.8",
            latitude=48.8,
            longitude=2.3,
            country="France",
            city="Paris",
            isp="OVH",
            threat_source="firehol_level1",
            first_seen=datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            last_seen=datetime(2024, 1, 15, 13, 0, 0, tzinfo=timezone.utc),
        ),
    ]

    for record in records:
        await db.upsert_attack(record)

    return app, db, model_service


class TestGetAttacks:
    """Tests for GET /api/attacks."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_data(self, client):
        """Returns empty list when database has no records."""
        response = await client.get("/api/attacks")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_returns_all_attack_records(self, populated_app):
        """Returns all attack records from the database."""
        app, _, _ = populated_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/attacks")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        ips = {record["ip_address"] for record in data}
        assert ips == {"1.2.3.4", "5.6.7.8"}

    @pytest.mark.asyncio
    async def test_attack_record_has_expected_fields(self, populated_app):
        """Each record contains expected fields."""
        app, _, _ = populated_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/attacks")

        data = response.json()
        record = data[0]
        expected_fields = {
            "id", "ip_address", "latitude", "longitude",
            "country", "city", "isp", "threat_source",
            "first_seen", "last_seen",
        }
        assert expected_fields.issubset(set(record.keys()))


class TestGetStats:
    """Tests for GET /api/stats."""

    @pytest.mark.asyncio
    async def test_returns_zero_stats_when_empty(self, client):
        """Returns zero counts when database is empty."""
        response = await client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_ips"] == 0
        assert data["countries"] == 0
        assert data["attacks_last_hour"] == 0

    @pytest.mark.asyncio
    async def test_returns_correct_stats(self, populated_app):
        """Returns correct stats for populated database."""
        app, _, _ = populated_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["total_ips"] == 2
        assert data["countries"] == 2


class TestGetTimeline:
    """Tests for GET /api/timeline."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_data(self, client):
        """Returns empty list when no recent data exists."""
        response = await client.get("/api/timeline")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_returns_timeline_buckets(self, app_with_db):
        """Returns hourly buckets for recent attacks."""
        app, db, _ = app_with_db

        # Insert a record with a recent timestamp
        now = datetime.now(timezone.utc)
        record = AttackRecord(
            ip_address="10.0.0.1",
            latitude=40.7,
            longitude=-74.0,
            country="United States",
            city="New York",
            isp="Comcast",
            threat_source="firehol_level1",
            first_seen=now,
            last_seen=now,
        )
        await db.upsert_attack(record)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/timeline")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert "hour" in data[0]
        assert "count" in data[0]
        assert data[0]["count"] >= 1


class TestHealthCheck:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_degraded_when_model_not_loaded(self, client):
        """Returns degraded status when model is not loaded."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["database_records"] == 0

    @pytest.mark.asyncio
    async def test_health_ok_when_model_loaded(self, app_with_db):
        """Returns ok status when model is loaded."""
        app, _, model_service = app_with_db
        model_service.is_loaded = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_includes_record_count(self, populated_app):
        """Returns correct database record count."""
        app, _, _ = populated_app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["database_records"] == 2
