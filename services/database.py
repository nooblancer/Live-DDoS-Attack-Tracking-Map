"""Async SQLite wrapper for attack records.

Provides CRUD operations for the attacks table using aiosqlite.
"""

from datetime import datetime, timedelta, timezone

import aiosqlite

from models.schemas import AttackRecord

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ip_address TEXT NOT NULL UNIQUE,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    country TEXT,
    city TEXT,
    isp TEXT,
    threat_source TEXT DEFAULT 'firehol_level1',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
"""

_CREATE_INDEX_LAST_SEEN_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_attacks_last_seen ON attacks(last_seen);"
)
_CREATE_INDEX_COUNTRY_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_attacks_country ON attacks(country);"
)

_UPSERT_SQL = """
INSERT INTO attacks (ip_address, latitude, longitude, country, city, isp, threat_source, first_seen, last_seen)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(ip_address) DO UPDATE SET last_seen = excluded.last_seen;
"""

_SELECT_ALL_SQL = """
SELECT id, ip_address, latitude, longitude, country, city, isp, threat_source, first_seen, last_seen
FROM attacks;
"""

_STATS_SQL = """
SELECT
    COUNT(*) AS total_ips,
    COUNT(DISTINCT country) AS countries,
    SUM(CASE WHEN last_seen >= ? THEN 1 ELSE 0 END) AS attacks_last_hour,
    MAX(last_seen) AS latest_timestamp
FROM attacks;
"""

_TIMELINE_SQL = """
SELECT
    strftime('%Y-%m-%dT%H:00:00Z', last_seen) AS hour,
    COUNT(*) AS count
FROM attacks
WHERE last_seen >= ?
GROUP BY hour
ORDER BY hour;
"""


class DatabaseService:
    """Async SQLite wrapper for attack records."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection, create attacks table if not exists."""
        self._conn = await aiosqlite.connect(self.db_path)
        # Enable WAL mode for better concurrent read performance
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute(_CREATE_TABLE_SQL)
        await self._conn.execute(_CREATE_INDEX_LAST_SEEN_SQL)
        await self._conn.execute(_CREATE_INDEX_COUNTRY_SQL)
        await self._conn.commit()

    async def upsert_attack(self, record: AttackRecord) -> None:
        """INSERT or UPDATE last_seen for existing IP (ON CONFLICT)."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        await self._conn.execute(
            _UPSERT_SQL,
            (
                record.ip_address,
                record.latitude,
                record.longitude,
                record.country,
                record.city,
                record.isp,
                record.threat_source,
                record.first_seen.isoformat(),
                record.last_seen.isoformat(),
            ),
        )
        await self._conn.commit()

    async def get_all_attacks(self) -> list[dict]:
        """Return all attack records for initial dashboard load."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        self._conn.row_factory = aiosqlite.Row
        cursor = await self._conn.execute(_SELECT_ALL_SQL)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_stats(self) -> dict:
        """Return total_ips, countries, attacks_last_hour, latest_timestamp."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        cursor = await self._conn.execute(_STATS_SQL, (one_hour_ago,))
        row = await cursor.fetchone()

        if row is None:
            return {
                "total_ips": 0,
                "countries": 0,
                "attacks_last_hour": 0,
                "latest_timestamp": None,
            }

        return {
            "total_ips": row[0] or 0,
            "countries": row[1] or 0,
            "attacks_last_hour": row[2] or 0,
            "latest_timestamp": row[3],
        }

    async def get_timeline(self, hours: int = 24) -> list[dict]:
        """Return hourly attack counts for the last N hours."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cursor = await self._conn.execute(_TIMELINE_SQL, (cutoff,))
        rows = await cursor.fetchall()
        return [{"hour": row[0], "count": row[1]} for row in rows]

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
