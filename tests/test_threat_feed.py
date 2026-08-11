"""Tests for ThreatFeedService."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from services.database import DatabaseService
from services.event_bus import EventBus
from services.geolocation import GeolocationService
from services.threat_feed import ThreatFeedService


@pytest_asyncio.fixture
async def db_service():
    """Provide a DatabaseService connected to a temporary SQLite file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_attacks.db")
        service = DatabaseService(db_path)
        await service.initialize()
        yield service
        await service.close()


@pytest.fixture
def event_bus():
    """Provide a fresh EventBus."""
    return EventBus()


@pytest.fixture
def geo_service():
    """Provide a GeolocationService (will be mocked in tests)."""
    return GeolocationService(batch_size=50)


@pytest_asyncio.fixture
async def feed_service(geo_service, db_service, event_bus):
    """Provide a ThreatFeedService with real DB and EventBus."""
    service = ThreatFeedService(geo_service, db_service, event_bus)
    yield service
    await service.close()


SAMPLE_NETSET = """\
# FireHOL blocklist
# comment line
#
192.168.1.1
10.0.0.1
172.16.0.0/12
8.8.8.8
1.2.3.4/24
255.255.255.255
"""


@pytest.mark.asyncio
async def test_fetch_blocklist_parses_ips(feed_service):
    """fetch_blocklist should return individual IPs, skipping comments and CIDRs."""
    mock_request = httpx.Request("GET", "http://test")
    mock_response = httpx.Response(200, text=SAMPLE_NETSET, request=mock_request)

    with patch.object(
        feed_service, "_get_client", return_value=AsyncMock(get=AsyncMock(return_value=mock_response))
    ):
        ips = await feed_service.fetch_blocklist()

    assert ips == ["192.168.1.1", "10.0.0.1", "8.8.8.8", "255.255.255.255"]


@pytest.mark.asyncio
async def test_fetch_blocklist_skips_empty_lines(feed_service):
    """Empty lines should be ignored."""
    content = "\n\n192.168.1.1\n\n10.0.0.1\n\n"
    mock_request = httpx.Request("GET", "http://test")
    mock_response = httpx.Response(200, text=content, request=mock_request)

    with patch.object(
        feed_service, "_get_client", return_value=AsyncMock(get=AsyncMock(return_value=mock_response))
    ):
        ips = await feed_service.fetch_blocklist()

    assert ips == ["192.168.1.1", "10.0.0.1"]


@pytest.mark.asyncio
async def test_fetch_blocklist_http_error_returns_empty(feed_service):
    """HTTP errors should log and return empty list."""
    mock_response = httpx.Response(500, request=httpx.Request("GET", "http://test"))

    async def raise_for_status_get(*args, **kwargs):
        resp = mock_response
        resp.raise_for_status()
        return resp

    mock_client = AsyncMock()
    mock_client.get = raise_for_status_get

    with patch.object(feed_service, "_get_client", return_value=mock_client):
        ips = await feed_service.fetch_blocklist()

    assert ips == []


@pytest.mark.asyncio
async def test_fetch_blocklist_network_error_returns_empty(feed_service):
    """Network errors should log and return empty list."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch.object(feed_service, "_get_client", return_value=mock_client):
        ips = await feed_service.fetch_blocklist()

    assert ips == []


@pytest.mark.asyncio
async def test_refresh_deduplicates_against_db(db_service, event_bus, geo_service):
    """refresh() should only process IPs not already in the database."""
    from datetime import datetime, timezone
    from models.schemas import AttackRecord

    # Pre-insert an IP into the DB
    existing_record = AttackRecord(
        ip_address="192.168.1.1",
        latitude=0.0,
        longitude=0.0,
        country="Test",
        city="Test",
        isp="Test",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
    )
    await db_service.upsert_attack(existing_record)

    feed_service = ThreatFeedService(geo_service, db_service, event_bus)

    # Mock fetch_blocklist to return both existing and new IPs
    feed_service.fetch_blocklist = AsyncMock(return_value=["192.168.1.1", "10.0.0.1"])

    # Mock geolocate_batch to return geo info for the new IP only
    geo_service.geolocate_batch = AsyncMock(
        return_value=[
            {"ip": "10.0.0.1", "lat": 40.0, "lon": -74.0, "country": "US", "city": "NY", "isp": "ISP1"}
        ]
    )

    # Subscribe to event bus to verify publish
    queue = event_bus.subscribe()

    await feed_service.refresh()

    # Should only have geolocated the new IP (10.0.0.1), not 192.168.1.1
    geo_service.geolocate_batch.assert_called_once_with(["10.0.0.1"])

    # Event should have been published
    event = queue.get_nowait()
    assert event["ip_address"] == "10.0.0.1"
    assert event["latitude"] == 40.0
    assert event["longitude"] == -74.0

    await feed_service.close()


@pytest.mark.asyncio
async def test_refresh_empty_blocklist_does_nothing(db_service, event_bus, geo_service):
    """refresh() should return early if fetch_blocklist returns empty."""
    feed_service = ThreatFeedService(geo_service, db_service, event_bus)
    feed_service.fetch_blocklist = AsyncMock(return_value=[])
    geo_service.geolocate_batch = AsyncMock()

    await feed_service.refresh()

    geo_service.geolocate_batch.assert_not_called()

    await feed_service.close()


@pytest.mark.asyncio
async def test_refresh_stores_records_in_db(db_service, event_bus, geo_service):
    """refresh() should store geolocated IPs in the database."""
    feed_service = ThreatFeedService(geo_service, db_service, event_bus)
    feed_service.fetch_blocklist = AsyncMock(return_value=["1.1.1.1", "2.2.2.2"])

    geo_service.geolocate_batch = AsyncMock(
        return_value=[
            {"ip": "1.1.1.1", "lat": 10.0, "lon": 20.0, "country": "DE", "city": "Berlin", "isp": "ISP-A"},
            {"ip": "2.2.2.2", "lat": 30.0, "lon": 40.0, "country": "FR", "city": "Paris", "isp": "ISP-B"},
        ]
    )

    await feed_service.refresh()

    # Verify records are in DB
    records = await db_service.get_all_attacks()
    ips = {r["ip_address"] for r in records}
    assert "1.1.1.1" in ips
    assert "2.2.2.2" in ips

    await feed_service.close()


@pytest.mark.asyncio
async def test_refresh_handles_exception_gracefully(db_service, event_bus, geo_service):
    """refresh() should not crash on unexpected exceptions."""
    feed_service = ThreatFeedService(geo_service, db_service, event_bus)
    feed_service.fetch_blocklist = AsyncMock(side_effect=RuntimeError("Unexpected"))

    # Should not raise
    await feed_service.refresh()

    await feed_service.close()
