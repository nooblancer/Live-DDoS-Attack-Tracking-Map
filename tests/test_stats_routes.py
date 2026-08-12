"""Tests for routes/stats.py — Statistics API endpoints."""

from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from models.schemas import (
    AttackType,
    FeatureImportance,
    ModelStatsResponse,
    TopAttackerEntry,
)
from routes.stats import router


# ---------------------------------------------------------------------------
# Valid attack type keys (the 12 defined types)
# ---------------------------------------------------------------------------

VALID_ATTACK_TYPES = {at.value for at in AttackType}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_with_stats():
    """Create a FastAPI app with a mock stats_accumulator on app.state."""
    app = FastAPI()
    app.include_router(router)

    stats_accumulator = MagicMock()

    # Default return values for each endpoint
    stats_accumulator.get_model_stats.return_value = ModelStatsResponse(
        model_type="XGBoost",
        f1_score=0.97,
        precision=0.96,
        recall=0.98,
        roc_auc=0.99,
        total_predictions=5000,
        attacks_detected=4500,
        benign_classified=500,
        predictions_per_second=42.5,
        top_features=[
            FeatureImportance(name="flow_duration", importance=0.15),
            FeatureImportance(name="total_fwd_packets", importance=0.12),
        ],
        attack_type_breakdown={
            "SYN Flood": 1200,
            "UDP Flood": 800,
            "DNS Amplification": 600,
        },
    )

    stats_accumulator.get_top_attackers.return_value = [
        TopAttackerEntry(
            rank=1,
            ip_address="185.220.x.x",
            country="DE",
            attack_count=342,
            last_seen="2024-01-15T14:32:07Z",
        ),
        TopAttackerEntry(
            rank=2,
            ip_address="45.148.x.x",
            country="RU",
            attack_count=291,
            last_seen="2024-01-15T14:31:55Z",
        ),
        TopAttackerEntry(
            rank=3,
            ip_address="23.94.x.x",
            country="US",
            attack_count=150,
            last_seen="2024-01-15T14:30:12Z",
        ),
    ]

    stats_accumulator.get_attack_types.return_value = {
        "SYN Flood": 1200,
        "UDP Flood": 800,
        "DNS Amplification": 600,
        "HTTP Flood": 400,
        "LDAP": 200,
        "NTP": 180,
        "MSSQL": 150,
        "NetBIOS": 120,
        "SSDP": 100,
        "TFTP": 80,
        "UDPLag": 60,
        "WebDDoS": 40,
    }

    app.state.stats_accumulator = stats_accumulator

    yield app, stats_accumulator


@pytest_asyncio.fixture
async def app_without_stats():
    """Create a FastAPI app without stats_accumulator (simulates 503)."""
    app = FastAPI()
    app.include_router(router)
    # No stats_accumulator on app.state
    yield app


@pytest_asyncio.fixture
async def client(app_with_stats):
    """Provide an async test client with stats_accumulator available."""
    app, _ = app_with_stats
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def client_no_stats(app_without_stats):
    """Provide an async test client without stats_accumulator (503 scenario)."""
    app = app_without_stats
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests for GET /api/model-stats
# ---------------------------------------------------------------------------


class TestGetModelStats:
    """Tests for GET /api/model-stats."""

    @pytest.mark.asyncio
    async def test_returns_200_with_correct_schema(self, client):
        """Returns 200 with all ModelStatsResponse fields present."""
        response = await client.get("/api/model-stats")

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields from ModelStatsResponse schema
        assert data["model_type"] == "XGBoost"
        assert isinstance(data["f1_score"], float)
        assert isinstance(data["precision"], float)
        assert isinstance(data["recall"], float)
        assert isinstance(data["roc_auc"], float)
        assert isinstance(data["total_predictions"], int)
        assert isinstance(data["attacks_detected"], int)
        assert isinstance(data["benign_classified"], int)
        assert isinstance(data["predictions_per_second"], float)
        assert isinstance(data["top_features"], list)
        assert isinstance(data["attack_type_breakdown"], dict)

    @pytest.mark.asyncio
    async def test_top_features_have_name_and_importance(self, client):
        """Each top_features entry has name (str) and importance (float)."""
        response = await client.get("/api/model-stats")

        data = response.json()
        for feature in data["top_features"]:
            assert "name" in feature
            assert "importance" in feature
            assert isinstance(feature["name"], str)
            assert isinstance(feature["importance"], float)

    @pytest.mark.asyncio
    async def test_returns_503_when_stats_unavailable(self, client_no_stats):
        """Returns 503 when stats_accumulator is not initialized."""
        response = await client_no_stats.get("/api/model-stats")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests for GET /api/top-attackers
# ---------------------------------------------------------------------------


class TestGetTopAttackers:
    """Tests for GET /api/top-attackers."""

    @pytest.mark.asyncio
    async def test_returns_200_with_list_of_entries(self, client):
        """Returns 200 with a list of TopAttackerEntry objects."""
        response = await client.get("/api/top-attackers")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify each entry has correct schema
        for entry in data:
            assert "rank" in entry
            assert "ip_address" in entry
            assert "country" in entry
            assert "attack_count" in entry
            assert "last_seen" in entry
            assert isinstance(entry["rank"], int)
            assert isinstance(entry["ip_address"], str)
            assert isinstance(entry["attack_count"], int)
            assert isinstance(entry["last_seen"], str)

    @pytest.mark.asyncio
    async def test_returns_at_most_20_entries_sorted_descending(self, app_with_stats):
        """Returns at most 20 entries sorted by attack_count descending."""
        app, stats_accumulator = app_with_stats

        # Generate 25 entries to verify the limit is 20
        entries = [
            TopAttackerEntry(
                rank=i + 1,
                ip_address=f"10.0.{i}.x.x",
                country="US",
                attack_count=1000 - (i * 30),
                last_seen="2024-01-15T14:32:07Z",
            )
            for i in range(25)
        ]
        stats_accumulator.get_top_attackers.return_value = entries[:20]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/top-attackers")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 20

        # Verify descending sort by attack_count
        counts = [entry["attack_count"] for entry in data]
        assert counts == sorted(counts, reverse=True)

        # Verify the endpoint was called with limit=20
        stats_accumulator.get_top_attackers.assert_called_with(limit=20)

    @pytest.mark.asyncio
    async def test_returns_503_when_stats_unavailable(self, client_no_stats):
        """Returns 503 when stats_accumulator is not initialized."""
        response = await client_no_stats.get("/api/top-attackers")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests for GET /api/attack-types
# ---------------------------------------------------------------------------


class TestGetAttackTypes:
    """Tests for GET /api/attack-types."""

    @pytest.mark.asyncio
    async def test_returns_200_with_dict_of_string_to_int(self, client):
        """Returns 200 with a dictionary mapping attack type strings to int counts."""
        response = await client.get("/api/attack-types")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

        # Each key is a string, each value is an int
        for key, value in data.items():
            assert isinstance(key, str)
            assert isinstance(value, int)

    @pytest.mark.asyncio
    async def test_returns_valid_attack_type_keys(self, client):
        """All returned keys are from the 12 defined attack types."""
        response = await client.get("/api/attack-types")

        data = response.json()
        for key in data.keys():
            assert key in VALID_ATTACK_TYPES, (
                f"Unexpected attack type key: '{key}'. "
                f"Valid types: {VALID_ATTACK_TYPES}"
            )

    @pytest.mark.asyncio
    async def test_returns_503_when_stats_unavailable(self, client_no_stats):
        """Returns 503 when stats_accumulator is not initialized."""
        response = await client_no_stats.get("/api/attack-types")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
