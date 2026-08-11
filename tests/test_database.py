"""Unit tests for services/database.py."""

from datetime import datetime, timedelta, timezone

import pytest

from models.schemas import AttackRecord
from services.database import DatabaseService


@pytest.mark.asyncio
async def test_initialize_creates_table(db_service: DatabaseService):
    """Verify that initialize() creates the attacks table."""
    cursor = await db_service._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='attacks';"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "attacks"


@pytest.mark.asyncio
async def test_initialize_creates_indexes(db_service: DatabaseService):
    """Verify that initialize() creates the expected indexes."""
    cursor = await db_service._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index';"
    )
    rows = await cursor.fetchall()
    index_names = [r[0] for r in rows]
    assert "idx_attacks_last_seen" in index_names
    assert "idx_attacks_country" in index_names


@pytest.mark.asyncio
async def test_upsert_insert_new_attack(db_service: DatabaseService):
    """Verify inserting a new attack record works."""
    now = datetime.now(timezone.utc)
    record = AttackRecord(
        ip_address="192.168.1.1",
        latitude=51.5,
        longitude=-0.1,
        country="UK",
        city="London",
        isp="BT",
        first_seen=now,
        last_seen=now,
    )
    await db_service.upsert_attack(record)

    attacks = await db_service.get_all_attacks()
    assert len(attacks) == 1
    assert attacks[0]["ip_address"] == "192.168.1.1"
    assert attacks[0]["country"] == "UK"


@pytest.mark.asyncio
async def test_upsert_updates_last_seen_on_conflict(db_service: DatabaseService):
    """Verify that upserting an existing IP updates last_seen only."""
    first_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    second_time = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)

    record1 = AttackRecord(
        ip_address="10.0.0.1",
        latitude=40.7,
        longitude=-74.0,
        country="US",
        city="New York",
        isp="Comcast",
        first_seen=first_time,
        last_seen=first_time,
    )
    record2 = AttackRecord(
        ip_address="10.0.0.1",
        latitude=40.7,
        longitude=-74.0,
        country="US",
        city="New York",
        isp="Comcast",
        first_seen=second_time,
        last_seen=second_time,
    )

    await db_service.upsert_attack(record1)
    await db_service.upsert_attack(record2)

    attacks = await db_service.get_all_attacks()
    assert len(attacks) == 1
    # last_seen should be updated to the second timestamp
    assert attacks[0]["last_seen"] == second_time.isoformat()
    # first_seen should remain the original
    assert attacks[0]["first_seen"] == first_time.isoformat()


@pytest.mark.asyncio
async def test_get_all_attacks_empty(db_service: DatabaseService):
    """Verify get_all_attacks returns empty list for empty DB."""
    attacks = await db_service.get_all_attacks()
    assert attacks == []


@pytest.mark.asyncio
async def test_get_all_attacks_returns_all_fields(db_service: DatabaseService):
    """Verify get_all_attacks includes all expected fields."""
    now = datetime.now(timezone.utc)
    record = AttackRecord(
        ip_address="1.2.3.4",
        latitude=35.6,
        longitude=139.7,
        country="Japan",
        city="Tokyo",
        isp="NTT",
        threat_source="firehol_level1",
        first_seen=now,
        last_seen=now,
    )
    await db_service.upsert_attack(record)

    attacks = await db_service.get_all_attacks()
    assert len(attacks) == 1
    attack = attacks[0]
    expected_keys = {
        "id", "ip_address", "latitude", "longitude",
        "country", "city", "isp", "threat_source",
        "first_seen", "last_seen",
    }
    assert set(attack.keys()) == expected_keys


@pytest.mark.asyncio
async def test_get_stats_empty_db(db_service: DatabaseService):
    """Verify stats on empty database."""
    stats = await db_service.get_stats()
    assert stats["total_ips"] == 0
    assert stats["countries"] == 0
    assert stats["attacks_last_hour"] == 0
    assert stats["latest_timestamp"] is None


@pytest.mark.asyncio
async def test_get_stats_with_data(db_service: DatabaseService):
    """Verify stats correctly count IPs, countries, and recent attacks."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)

    records = [
        AttackRecord(
            ip_address="1.1.1.1", latitude=51.5, longitude=-0.1,
            country="UK", city="London", isp="ISP1",
            first_seen=now, last_seen=now,
        ),
        AttackRecord(
            ip_address="2.2.2.2", latitude=48.8, longitude=2.3,
            country="France", city="Paris", isp="ISP2",
            first_seen=now, last_seen=now,
        ),
        AttackRecord(
            ip_address="3.3.3.3", latitude=40.7, longitude=-74.0,
            country="US", city="NYC", isp="ISP3",
            first_seen=old_time, last_seen=old_time,
        ),
    ]
    for r in records:
        await db_service.upsert_attack(r)

    stats = await db_service.get_stats()
    assert stats["total_ips"] == 3
    assert stats["countries"] == 3
    assert stats["attacks_last_hour"] == 2  # only 2 within last hour
    assert stats["latest_timestamp"] is not None


@pytest.mark.asyncio
async def test_get_timeline_empty_db(db_service: DatabaseService):
    """Verify timeline returns empty list for empty DB."""
    timeline = await db_service.get_timeline()
    assert timeline == []


@pytest.mark.asyncio
async def test_get_timeline_groups_by_hour(db_service: DatabaseService):
    """Verify timeline groups attacks into hourly buckets."""
    now = datetime.now(timezone.utc).replace(minute=30, second=0, microsecond=0)

    records = [
        AttackRecord(
            ip_address="1.1.1.1", latitude=51.5, longitude=-0.1,
            country="UK", city="London", isp="ISP1",
            first_seen=now, last_seen=now,
        ),
        AttackRecord(
            ip_address="2.2.2.2", latitude=48.8, longitude=2.3,
            country="France", city="Paris", isp="ISP2",
            first_seen=now, last_seen=now,
        ),
        AttackRecord(
            ip_address="3.3.3.3", latitude=40.7, longitude=-74.0,
            country="US", city="NYC", isp="ISP3",
            first_seen=now - timedelta(hours=1),
            last_seen=now - timedelta(hours=1),
        ),
    ]
    for r in records:
        await db_service.upsert_attack(r)

    timeline = await db_service.get_timeline()
    assert len(timeline) == 2  # 2 distinct hours
    # Each bucket has 'hour' and 'count'
    for bucket in timeline:
        assert "hour" in bucket
        assert "count" in bucket
        assert bucket["count"] >= 1


@pytest.mark.asyncio
async def test_close_sets_conn_to_none(db_service: DatabaseService):
    """Verify close() properly sets _conn to None."""
    assert db_service._conn is not None
    await db_service.close()
    assert db_service._conn is None
