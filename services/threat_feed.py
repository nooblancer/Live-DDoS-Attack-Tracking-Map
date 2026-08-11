"""Threat feed service: fetches FireHOL blocklist, deduplicates, geolocates, and publishes."""

import logging
import re
from datetime import datetime, timezone

import httpx

import config
from models.schemas import AttackEvent, AttackRecord
from services.database import DatabaseService
from services.event_bus import EventBus
from services.geolocation import GeolocationService

logger = logging.getLogger(__name__)

# Regex to match a single IPv4 address (no CIDR suffix)
_IPV4_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


class ThreatFeedService:
    """Fetches and parses FireHOL blocklist IPs."""

    def __init__(
        self,
        geo_service: GeolocationService,
        db: DatabaseService,
        event_bus: EventBus,
    ):
        self._geo_service = geo_service
        self._db = db
        self._event_bus = event_bus
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def fetch_blocklist(self) -> list[str]:
        """GET the .netset file, parse lines into IP addresses (skip comments/CIDRs)."""
        client = await self._get_client()

        try:
            response = await client.get(config.FIREHOL_URL)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "HTTP error %d fetching blocklist, skipping this cycle",
                exc.response.status_code,
            )
            return []
        except httpx.HTTPError as exc:
            logger.warning(
                "Network error fetching blocklist: %s, skipping this cycle",
                str(exc),
            )
            return []

        ips: list[str] = []
        for line in response.text.splitlines():
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Skip CIDR ranges (contain /)
            if "/" in line:
                continue
            # Validate as IPv4
            if _IPV4_PATTERN.match(line):
                ips.append(line)

        return ips

    async def refresh(self) -> None:
        """Fetch → deduplicate against DB → geolocate new IPs → store → publish events."""
        try:
            # 1. Fetch blocklist IPs
            ips = await self.fetch_blocklist()
            if not ips:
                return

            # 2. Get existing IPs from DB
            existing_records = await self._db.get_all_attacks()
            existing_ips = {record["ip_address"] for record in existing_records}

            # 3. Filter to only new IPs
            new_ips = [ip for ip in ips if ip not in existing_ips]
            if not new_ips:
                logger.debug("No new IPs found in blocklist")
                return

            logger.info("Found %d new IPs to process", len(new_ips))

            # 4. Geolocate new IPs
            geo_results = await self._geo_service.geolocate_batch(new_ips)

            # 5. Store and publish each geolocated IP
            now = datetime.now(timezone.utc)
            for geo in geo_results:
                record = AttackRecord(
                    ip_address=geo["ip"],
                    latitude=geo["lat"],
                    longitude=geo["lon"],
                    country=geo.get("country"),
                    city=geo.get("city"),
                    isp=geo.get("isp"),
                    threat_source="firehol_level1",
                    first_seen=now,
                    last_seen=now,
                )

                # 5a. Store in DB
                await self._db.upsert_attack(record)

                # 5b. Publish event to SSE clients
                event = AttackEvent(
                    ip_address=record.ip_address,
                    latitude=record.latitude,
                    longitude=record.longitude,
                    country=record.country,
                    city=record.city,
                    isp=record.isp,
                    timestamp=now.isoformat(),
                )
                await self._event_bus.publish(event.model_dump())

            logger.info(
                "Refresh complete: %d new IPs geolocated and stored",
                len(geo_results),
            )

        except Exception as exc:
            # Catch-all: the service should never crash
            logger.error("Unexpected error during refresh: %s", str(exc), exc_info=True)

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
