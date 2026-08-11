"""Shared test fixtures."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from services.database import DatabaseService


@pytest_asyncio.fixture
async def db_service():
    """Provide a DatabaseService connected to a temporary SQLite file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_attacks.db")
        service = DatabaseService(db_path)
        await service.initialize()
        yield service
        await service.close()
