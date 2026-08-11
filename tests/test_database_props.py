"""Property-based tests for services/database.py."""

# Feature: ddos-attack-tracking-map, Property 14: Database upsert idempotence

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import given, HealthCheck, settings
from hypothesis import strategies as st

from models.schemas import AttackRecord
from services.database import DatabaseService


# --- Strategies ---

# Generate valid IPv4 addresses
ipv4_strategy = st.tuples(
    st.integers(min_value=1, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=1, max_value=255),
).map(lambda parts: f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}")

# Generate UTC-aware datetimes within a reasonable range
timestamp_strategy = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2030, 12, 31),
    timezones=st.just(timezone.utc),
)

# Generate a list of at least 2 distinct timestamps for multiple upserts
timestamps_list_strategy = st.lists(
    timestamp_strategy,
    min_size=2,
    max_size=10,
).filter(lambda ts_list: len(set(ts_list)) >= 2)


# --- Property Tests ---


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    ip=ipv4_strategy,
    timestamps=timestamps_list_strategy,
)
def test_upsert_idempotence_single_row_per_ip(
    ip: str,
    timestamps: list[datetime],
):
    """Property 14: Database upsert idempotence.

    For any IP address inserted multiple times with different timestamps,
    the attacks table shall contain exactly one row for that IP with
    last_seen equal to the most recent timestamp provided.

    **Validates: Requirements 7.4**
    """

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test_attacks.db")
            db_service = DatabaseService(db_path)
            await db_service.initialize()
            try:
                # Sort timestamps so inserts are in chronological order (ascending).
                sorted_timestamps = sorted(timestamps)

                # Insert the same IP multiple times with different timestamps
                for ts in sorted_timestamps:
                    record = AttackRecord(
                        ip_address=ip,
                        latitude=51.5,
                        longitude=-0.1,
                        country="UK",
                        city="London",
                        isp="TestISP",
                        threat_source="firehol_level1",
                        first_seen=ts,
                        last_seen=ts,
                    )
                    await db_service.upsert_attack(record)

                # Query all attacks and filter for our IP
                all_attacks = await db_service.get_all_attacks()
                matching = [a for a in all_attacks if a["ip_address"] == ip]

                # Property: exactly one row per IP
                assert len(matching) == 1, (
                    f"Expected exactly 1 row for IP {ip}, got {len(matching)}"
                )

                # Property: last_seen equals the most recent (max) timestamp provided
                expected_last_seen = max(sorted_timestamps).isoformat()
                assert matching[0]["last_seen"] == expected_last_seen, (
                    f"Expected last_seen={expected_last_seen}, "
                    f"got last_seen={matching[0]['last_seen']}"
                )
            finally:
                await db_service.close()

    asyncio.run(_run())
