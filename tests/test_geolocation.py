"""Unit tests for the geolocation service."""

import asyncio

import httpx
import pytest
import pytest_asyncio

from services.geolocation import GeolocationService, IP_API_BATCH_URL, _RATE_LIMIT_DELAY


@pytest_asyncio.fixture
async def geo_service():
    """Provide a GeolocationService with small batch size for testing."""
    service = GeolocationService(batch_size=3)
    yield service
    await service.close()


class TestGeolocationServiceInit:
    """Tests for GeolocationService initialization."""

    def test_default_batch_size_from_config(self):
        """Batch size defaults to config.GEO_BATCH_SIZE."""
        import config

        service = GeolocationService()
        assert service.batch_size == config.GEO_BATCH_SIZE

    def test_custom_batch_size(self):
        """Custom batch size is respected."""
        service = GeolocationService(batch_size=25)
        assert service.batch_size == 25

    def test_client_initially_none(self):
        """HTTP client is None before first use."""
        service = GeolocationService()
        assert service._client is None


class TestGeolocateBatch:
    """Tests for geolocate_batch method."""

    @pytest.mark.asyncio
    async def test_empty_ip_list_returns_empty(self, geo_service):
        """Empty input returns empty result without making requests."""
        result = await geo_service.geolocate_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_geolocation(self, geo_service, httpx_mock):
        """Successful responses are parsed correctly."""
        mock_response = [
            {
                "status": "success",
                "query": "8.8.8.8",
                "lat": 37.751,
                "lon": -97.822,
                "country": "United States",
                "city": "Mountain View",
                "isp": "Google LLC",
            }
        ]
        httpx_mock.add_response(
            url=IP_API_BATCH_URL,
            method="POST",
            json=mock_response,
        )

        result = await geo_service.geolocate_batch(["8.8.8.8"])

        assert len(result) == 1
        assert result[0] == {
            "ip": "8.8.8.8",
            "lat": 37.751,
            "lon": -97.822,
            "country": "United States",
            "city": "Mountain View",
            "isp": "Google LLC",
        }

    @pytest.mark.asyncio
    async def test_failed_ips_are_skipped(self, geo_service, httpx_mock):
        """IPs with 'fail' status are excluded from results."""
        mock_response = [
            {
                "status": "success",
                "query": "8.8.8.8",
                "lat": 37.751,
                "lon": -97.822,
                "country": "United States",
                "city": "Mountain View",
                "isp": "Google LLC",
            },
            {
                "status": "fail",
                "query": "192.168.1.1",
                "message": "private range",
            },
        ]
        httpx_mock.add_response(
            url=IP_API_BATCH_URL,
            method="POST",
            json=mock_response,
        )

        result = await geo_service.geolocate_batch(["8.8.8.8", "192.168.1.1"])

        assert len(result) == 1
        assert result[0]["ip"] == "8.8.8.8"

    @pytest.mark.asyncio
    async def test_batching_splits_large_lists(self, geo_service, httpx_mock):
        """IP lists larger than batch_size are split into multiple batches."""
        # geo_service has batch_size=3, sending 5 IPs should create 2 batches
        batch1_response = [
            {"status": "success", "query": f"1.1.1.{i}", "lat": i, "lon": i, "country": "C", "city": "City", "isp": "ISP"}
            for i in range(3)
        ]
        batch2_response = [
            {"status": "success", "query": f"2.2.2.{i}", "lat": 10 + i, "lon": 10 + i, "country": "C", "city": "City", "isp": "ISP"}
            for i in range(2)
        ]
        httpx_mock.add_response(url=IP_API_BATCH_URL, method="POST", json=batch1_response)
        httpx_mock.add_response(url=IP_API_BATCH_URL, method="POST", json=batch2_response)

        ips = [f"1.1.1.{i}" for i in range(3)] + [f"2.2.2.{i}" for i in range(2)]
        result = await geo_service.geolocate_batch(ips)

        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_timeout_skips_batch(self, geo_service, httpx_mock):
        """Network timeout results in skipping that batch."""
        httpx_mock.add_exception(httpx.ReadTimeout("timeout"))

        result = await geo_service.geolocate_batch(["8.8.8.8"])

        assert result == []

    @pytest.mark.asyncio
    async def test_http_error_skips_batch(self, geo_service, httpx_mock):
        """HTTP 500 errors result in skipping that batch."""
        httpx_mock.add_response(
            url=IP_API_BATCH_URL,
            method="POST",
            status_code=500,
        )

        result = await geo_service.geolocate_batch(["8.8.8.8"])

        assert result == []

    @pytest.mark.asyncio
    async def test_all_fail_returns_empty(self, geo_service, httpx_mock):
        """If all IPs fail geolocation, result is empty."""
        mock_response = [
            {"status": "fail", "query": "10.0.0.1", "message": "private range"},
            {"status": "fail", "query": "172.16.0.1", "message": "private range"},
        ]
        httpx_mock.add_response(
            url=IP_API_BATCH_URL,
            method="POST",
            json=mock_response,
        )

        result = await geo_service.geolocate_batch(["10.0.0.1", "172.16.0.1"])

        assert result == []


class TestGeolocationServiceClose:
    """Tests for the close method."""

    @pytest.mark.asyncio
    async def test_close_without_use(self):
        """Closing without ever using the client doesn't raise."""
        service = GeolocationService()
        await service.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_after_use(self, httpx_mock):
        """Client is properly closed after use."""
        httpx_mock.add_response(
            url=IP_API_BATCH_URL,
            method="POST",
            json=[{"status": "success", "query": "8.8.8.8", "lat": 0, "lon": 0, "country": "X", "city": "Y", "isp": "Z"}],
        )

        service = GeolocationService(batch_size=50)
        await service.geolocate_batch(["8.8.8.8"])
        assert service._client is not None

        await service.close()
        assert service._client is None
