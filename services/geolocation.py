"""IP geolocation service using ip-api.com batch endpoint."""

import asyncio
import logging

import httpx

import config

logger = logging.getLogger(__name__)

IP_API_BATCH_URL = "http://ip-api.com/batch"

# ip-api.com free tier: 45 requests/minute → ~1.34s between requests
_RATE_LIMIT_DELAY = 60.0 / 45.0  # ~1.333 seconds


class GeolocationService:
    """Resolves IPs to coordinates via ip-api.com batch endpoint."""

    def __init__(self, batch_size: int | None = None):
        self.batch_size = batch_size if batch_size is not None else config.GEO_BATCH_SIZE
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def geolocate_batch(self, ips: list[str]) -> list[dict]:
        """
        Send batch POST to ip-api.com/batch.
        Respects 45 req/min limit by sleeping between batches.
        Returns list of {ip, lat, lon, country, city, isp} dicts.
        Skips IPs that return 'fail' status.
        """
        if not ips:
            return []

        results: list[dict] = []
        # Split IPs into batches respecting batch_size
        batches = [
            ips[i : i + self.batch_size]
            for i in range(0, len(ips), self.batch_size)
        ]

        client = await self._get_client()

        for batch_index, batch in enumerate(batches):
            # Rate limit: sleep between batches (skip delay before first batch)
            if batch_index > 0:
                await asyncio.sleep(_RATE_LIMIT_DELAY)

            try:
                response = await client.post(
                    IP_API_BATCH_URL,
                    json=batch,
                )

                if response.status_code == 429:
                    # Rate limited - back off with exponential delay and retry
                    delay = 2 ** batch_index * _RATE_LIMIT_DELAY
                    logger.warning(
                        "Rate limited by ip-api.com, backing off %.1fs", delay
                    )
                    await asyncio.sleep(delay)
                    # Retry this batch once
                    response = await client.post(
                        IP_API_BATCH_URL,
                        json=batch,
                    )
                    if response.status_code == 429:
                        logger.error(
                            "Still rate limited after retry, skipping batch of %d IPs",
                            len(batch),
                        )
                        continue

                response.raise_for_status()
                data = response.json()

                for entry in data:
                    if entry.get("status") == "fail":
                        logger.debug(
                            "Geolocation failed for IP %s: %s",
                            entry.get("query", "unknown"),
                            entry.get("message", "unknown reason"),
                        )
                        continue

                    results.append(
                        {
                            "ip": entry["query"],
                            "lat": entry["lat"],
                            "lon": entry["lon"],
                            "country": entry.get("country"),
                            "city": entry.get("city"),
                            "isp": entry.get("isp"),
                        }
                    )

            except httpx.TimeoutException:
                logger.error(
                    "Timeout on batch request (%d IPs), skipping batch",
                    len(batch),
                )
                continue
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "HTTP error %d on batch request, skipping batch",
                    exc.response.status_code,
                )
                continue
            except httpx.HTTPError as exc:
                logger.error(
                    "Network error on batch request: %s, skipping batch",
                    str(exc),
                )
                continue

        return results

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
