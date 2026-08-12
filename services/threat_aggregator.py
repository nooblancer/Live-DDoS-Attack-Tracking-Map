"""Multi-source threat intelligence aggregation service.

Fetches, merges, deduplicates, and geolocates IPs from multiple threat
intelligence sources: AbuseIPDB, FireHOL (L1-L3), Feodo Tracker, and
Emerging Threats. Publishes live threat events to the EventBus.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

import httpx

from models.schemas import (
    AttackRecord,
    EnhancedAttackEvent,
    ThreatIP,
    confidence_to_severity,
)
from services.database import DatabaseService
from services.event_bus import EventBus
from services.geolocation import GeolocationService

if TYPE_CHECKING:
    from services.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env vars with defaults per design doc)
# ---------------------------------------------------------------------------

ABUSEIPDB_API_KEY: str = os.getenv("ABUSEIPDB_API_KEY", "")
ABUSEIPDB_INTERVAL: int = int(os.getenv("ABUSEIPDB_INTERVAL", "3600"))
FIREHOL_INTERVAL: int = int(os.getenv("FIREHOL_INTERVAL", "1800"))
FEODO_INTERVAL: int = int(os.getenv("FEODO_INTERVAL", "1800"))
EMERGING_THREATS_INTERVAL: int = int(os.getenv("EMERGING_THREATS_INTERVAL", "1800"))

# Source URLs
ABUSEIPDB_URL: str = "https://api.abuseipdb.com/api/v2/blacklist"
FIREHOL_URLS: list[str] = [
    "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset",
    "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level2.netset",
    "https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level3.netset",
]
FEODO_URL: str = "https://feodotracker.abuse.ch/downloads/ipblocklist.txt"
EMERGING_THREATS_URL: str = (
    "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt"
)

# CIDR expansion: sample size for /24 or smaller ranges
CIDR_SAMPLE_MIN: int = 5
CIDR_SAMPLE_MAX: int = 10

# HTTP timeout for feed requests
_REQUEST_TIMEOUT: float = 30.0


class StatsAccumulatorProtocol(Protocol):
    """Protocol for StatsAccumulator dependency (avoid circular import)."""

    async def record_prediction(self, result: object, source_ip: str) -> None: ...


def expand_cidr(cidr: str) -> list[str]:
    """Expand a CIDR range (/24 or smaller) to 5-10 sampled IPs.

    For ranges with prefix length >= 24, sample uniformly distributed hosts.
    For ranges larger than /24, return an empty list (they are skipped).
    """
    try:
        network = ipaddress.IPv4Network(cidr, strict=False)
    except (ipaddress.AddressValueError, ValueError):
        return []

    # Only expand /24 or smaller (prefix >= 24)
    if network.prefixlen < 24:
        return []

    hosts = list(network.hosts())
    if not hosts:
        return []

    # Determine sample size
    sample_size = min(random.randint(CIDR_SAMPLE_MIN, CIDR_SAMPLE_MAX), len(hosts))

    if len(hosts) <= sample_size:
        return [str(ip) for ip in hosts]

    # Uniformly distributed step sampling
    step = max(1, len(hosts) // sample_size)
    sampled = [str(hosts[i * step]) for i in range(sample_size) if i * step < len(hosts)]
    return sampled


class ThreatAggregator:
    """Fetches, merges, and deduplicates IPs from multiple threat intelligence sources."""

    def __init__(
        self,
        geo_service: GeolocationService,
        db: DatabaseService,
        event_bus: EventBus,
        stats: StatsAccumulatorProtocol | None = None,
    ):
        self._geo_service = geo_service
        self._db = db
        self._event_bus = event_bus
        self._stats = stats
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh_all(self) -> None:
        """Refresh all sources in parallel, merge, deduplicate, geolocate, publish.

        Never raises unhandled exceptions — individual source failures are
        logged and skipped.
        """
        try:
            all_ips: list[ThreatIP] = []

            # Fetch all sources, handling each failure independently
            sources = [
                ("abuseipdb", self.fetch_abuseipdb),
                ("firehol", self.fetch_firehol),
                ("feodo", self.fetch_feodo),
                ("emerging_threats", self.fetch_emerging_threats),
            ]

            for source_name, fetch_fn in sources:
                try:
                    ips = await fetch_fn()
                    all_ips.extend(ips)
                    logger.info(
                        "Fetched %d IPs from %s", len(ips), source_name
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to fetch from %s: %s", source_name, str(exc),
                        exc_info=True,
                    )
                    continue

            if not all_ips:
                logger.warning("No IPs collected from any source")
                return

            # Deduplicate
            unique_ips = self.deduplicate(all_ips)
            logger.info(
                "Deduplication: %d → %d unique IPs", len(all_ips), len(unique_ips)
            )

            # Get existing IPs from DB to only process new ones
            existing_records = await self._db.get_all_attacks()
            existing_ip_set = {r["ip_address"] for r in existing_records}

            new_threats = [t for t in unique_ips if t.ip_address not in existing_ip_set]
            if not new_threats:
                logger.debug("No new threat IPs to process")
                return

            logger.info("Processing %d new threat IPs", len(new_threats))

            # Geolocate new IPs
            ip_addresses = [t.ip_address for t in new_threats]
            geo_results = await self._geo_service.geolocate_batch(ip_addresses)

            # Build lookup for geo results
            geo_lookup: dict[str, dict] = {g["ip"]: g for g in geo_results}

            # Store and publish each geolocated IP
            now = datetime.now(timezone.utc)
            for threat in new_threats:
                geo = geo_lookup.get(threat.ip_address)
                if geo is None:
                    # Skip IPs that couldn't be geolocated
                    continue

                # Persist to database
                record = AttackRecord(
                    ip_address=threat.ip_address,
                    latitude=geo["lat"],
                    longitude=geo["lon"],
                    country=geo.get("country"),
                    city=geo.get("city"),
                    isp=geo.get("isp"),
                    threat_source=threat.source,
                    first_seen=now,
                    last_seen=now,
                )
                await self._db.upsert_attack(record)

                # Publish enhanced event to EventBus with source_channel="live"
                event = EnhancedAttackEvent(
                    ip_address=threat.ip_address,
                    latitude=geo["lat"],
                    longitude=geo["lon"],
                    country=geo.get("country"),
                    city=geo.get("city"),
                    isp=geo.get("isp"),
                    attack_type="Unknown",
                    confidence=threat.confidence,
                    severity=confidence_to_severity(threat.confidence),
                    classified_in_ms=0.0,
                    top_features=threat.tags[:3] if len(threat.tags) >= 3 else threat.tags + ["threat_intel"] * (3 - len(threat.tags)),
                    source_channel="live",
                    timestamp=now.isoformat(),
                )
                await self._event_bus.publish(event.model_dump())

            logger.info(
                "Refresh complete: %d new threat IPs geolocated and published",
                len(geo_results),
            )

        except Exception as exc:
            # Catch-all: the service should never crash
            logger.error(
                "Unexpected error during refresh_all: %s", str(exc), exc_info=True
            )

    # ------------------------------------------------------------------
    # Individual source fetch methods
    # ------------------------------------------------------------------

    async def fetch_abuseipdb(self) -> list[ThreatIP]:
        """Fetch top 10K blacklisted IPs from AbuseIPDB API.

        Requires ABUSEIPDB_API_KEY env var. Returns empty list if not configured.
        """
        if not ABUSEIPDB_API_KEY:
            logger.debug("AbuseIPDB API key not configured, skipping")
            return []

        client = await self._get_client()
        headers = {
            "Key": ABUSEIPDB_API_KEY,
            "Accept": "application/json",
        }
        params = {"confidenceMinimum": "50", "limit": "10000"}

        response = await client.get(
            ABUSEIPDB_URL, headers=headers, params=params
        )
        response.raise_for_status()

        data = response.json()
        results: list[ThreatIP] = []

        for entry in data.get("data", []):
            ip = entry.get("ipAddress", "")
            score = entry.get("abuseConfidenceScore", 0)
            results.append(
                ThreatIP(
                    ip_address=ip,
                    source="abuseipdb",
                    confidence=score / 100.0,  # AbuseIPDB uses 0-100 scale
                    tags=["blacklisted"],
                    first_seen=None,
                )
            )

        return results

    async def fetch_firehol(self) -> list[ThreatIP]:
        """Fetch FireHOL L1-L3 with CIDR expansion for /24 or smaller ranges."""
        client = await self._get_client()
        results: list[ThreatIP] = []

        for level_idx, url in enumerate(FIREHOL_URLS, start=1):
            try:
                response = await client.get(url)
                response.raise_for_status()

                for line in response.text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    if "/" in line:
                        # CIDR range — expand if /24 or smaller
                        expanded = expand_cidr(line)
                        for ip in expanded:
                            results.append(
                                ThreatIP(
                                    ip_address=ip,
                                    source="firehol",
                                    confidence=0.8,
                                    tags=[f"firehol_level{level_idx}"],
                                    first_seen=None,
                                )
                            )
                    else:
                        # Single IP
                        results.append(
                            ThreatIP(
                                ip_address=line,
                                source="firehol",
                                confidence=0.8,
                                tags=[f"firehol_level{level_idx}"],
                                first_seen=None,
                            )
                        )

            except Exception as exc:
                logger.warning(
                    "Failed to fetch FireHOL level %d: %s", level_idx, str(exc)
                )
                continue

        return results

    async def fetch_feodo(self) -> list[ThreatIP]:
        """Fetch Feodo Tracker botnet C2 IPs."""
        client = await self._get_client()

        response = await client.get(FEODO_URL)
        response.raise_for_status()

        results: list[ThreatIP] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Feodo format: one IP per line (may have trailing comments)
            ip = line.split()[0] if line.split() else ""
            # Basic IPv4 validation
            try:
                ipaddress.IPv4Address(ip)
            except (ipaddress.AddressValueError, ValueError):
                continue

            results.append(
                ThreatIP(
                    ip_address=ip,
                    source="feodo",
                    confidence=0.9,
                    tags=["botnet_c2"],
                    first_seen=None,
                )
            )

        return results

    async def fetch_emerging_threats(self) -> list[ThreatIP]:
        """Fetch Emerging Threats compromised IP list."""
        client = await self._get_client()

        response = await client.get(EMERGING_THREATS_URL)
        response.raise_for_status()

        results: list[ThreatIP] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # ET format: one IP per line
            ip = line.split()[0] if line.split() else ""
            try:
                ipaddress.IPv4Address(ip)
            except (ipaddress.AddressValueError, ValueError):
                continue

            results.append(
                ThreatIP(
                    ip_address=ip,
                    source="emerging_threats",
                    confidence=0.75,
                    tags=["compromised"],
                    first_seen=None,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self, ips: list[ThreatIP]) -> list[ThreatIP]:
        """Merge across sources, retain highest confidence per unique IP.

        When the same IP appears from multiple sources, the entry with the
        highest confidence score is kept. Tags are merged from all sources.
        """
        best: dict[str, ThreatIP] = {}

        for threat in ips:
            existing = best.get(threat.ip_address)
            if existing is None:
                # First occurrence — clone to avoid mutating original
                best[threat.ip_address] = ThreatIP(
                    ip_address=threat.ip_address,
                    source=threat.source,
                    confidence=threat.confidence,
                    tags=list(threat.tags),
                    first_seen=threat.first_seen,
                )
            else:
                # Merge: keep highest confidence, merge tags
                if threat.confidence > existing.confidence:
                    existing.confidence = threat.confidence
                    existing.source = threat.source
                # Merge tags without duplicates
                for tag in threat.tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)

        return list(best.values())

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close HTTP clients."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
