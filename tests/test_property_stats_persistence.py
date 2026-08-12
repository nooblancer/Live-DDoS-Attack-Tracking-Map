"""Property-based tests for stats accumulator persistence round-trip."""

# Feature: v2-attack-tracking-overhaul, Property 7: Stats Persistence Round-Trip

import asyncio
import tempfile
from pathlib import Path

from hypothesis import given, HealthCheck, settings
from hypothesis import strategies as st

from models.schemas import AttackType
from services.database import DatabaseService
from services.stats_accumulator import StatsAccumulator


# --- Strategies ---

# All 12 valid attack type labels
ATTACK_TYPE_VALUES = [at.value for at in AttackType]

# Strategy for non-negative integers (stats counters)
non_negative_int = st.integers(min_value=0, max_value=10_000_000)

# Strategy for attack_type_breakdown: dict of attack type → non-negative int count
# Uses a subset of the 12 attack types with non-negative counts
attack_type_breakdown_strategy = st.dictionaries(
    keys=st.sampled_from(ATTACK_TYPE_VALUES),
    values=st.integers(min_value=0, max_value=1_000_000),
    min_size=0,
    max_size=12,
)

# Strategy for full stats state
stats_state_strategy = st.fixed_dictionaries(
    {
        "total_predictions": non_negative_int,
        "attacks_detected": non_negative_int,
        "benign_classified": non_negative_int,
        "attack_type_breakdown": attack_type_breakdown_strategy,
    }
)


# --- Property Tests ---


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(state=stats_state_strategy)
def test_stats_persist_restore_round_trip(state: dict):
    """Property 7: Stats Persistence Round-Trip.

    For any valid combination of total_predictions (non-negative int),
    attacks_detected (non-negative int), benign_classified (non-negative int),
    and attack_type_breakdown (dict of attack type → non-negative int count),
    calling persist() then restore() on a fresh StatsAccumulator produces
    identical values.

    **Validates: Requirements 5.3**
    """

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test_stats.db")
            db_service = DatabaseService(db_path)
            await db_service.initialize()
            try:
                # Create accumulator and set state
                accumulator = StatsAccumulator(db_service)
                accumulator.total_predictions = state["total_predictions"]
                accumulator.attacks_detected = state["attacks_detected"]
                accumulator.benign_classified = state["benign_classified"]
                accumulator._attack_type_counts = dict(state["attack_type_breakdown"])

                # Persist to SQLite
                await accumulator.persist()

                # Create a fresh accumulator (simulating restart) and restore
                restored = StatsAccumulator(db_service)
                await restored.restore()

                # Verify round-trip equivalence
                assert restored.total_predictions == state["total_predictions"], (
                    f"total_predictions mismatch: expected {state['total_predictions']}, "
                    f"got {restored.total_predictions}"
                )
                assert restored.attacks_detected == state["attacks_detected"], (
                    f"attacks_detected mismatch: expected {state['attacks_detected']}, "
                    f"got {restored.attacks_detected}"
                )
                assert restored.benign_classified == state["benign_classified"], (
                    f"benign_classified mismatch: expected {state['benign_classified']}, "
                    f"got {restored.benign_classified}"
                )
                assert restored._attack_type_counts == state["attack_type_breakdown"], (
                    f"attack_type_breakdown mismatch: "
                    f"expected {state['attack_type_breakdown']}, "
                    f"got {restored._attack_type_counts}"
                )
            finally:
                await db_service.close()

    asyncio.run(_run())


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    total_predictions=non_negative_int,
    attacks_detected=non_negative_int,
    benign_classified=non_negative_int,
)
def test_stats_round_trip_with_empty_breakdown(
    total_predictions: int,
    attacks_detected: int,
    benign_classified: int,
):
    """Property 7: Round-trip works with empty attack_type_breakdown.

    For any valid counters with an empty attack type breakdown dict,
    persisting and restoring SHALL produce an equivalent state.

    **Validates: Requirements 5.3**
    """

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test_stats_empty.db")
            db_service = DatabaseService(db_path)
            await db_service.initialize()
            try:
                accumulator = StatsAccumulator(db_service)
                accumulator.total_predictions = total_predictions
                accumulator.attacks_detected = attacks_detected
                accumulator.benign_classified = benign_classified
                accumulator._attack_type_counts = {}

                await accumulator.persist()

                restored = StatsAccumulator(db_service)
                await restored.restore()

                assert restored.total_predictions == total_predictions
                assert restored.attacks_detected == attacks_detected
                assert restored.benign_classified == benign_classified
                assert restored._attack_type_counts == {}
            finally:
                await db_service.close()

    asyncio.run(_run())


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    counts=st.lists(
        st.integers(min_value=0, max_value=1_000_000),
        min_size=12,
        max_size=12,
    )
)
def test_stats_round_trip_with_all_12_attack_types(counts: list[int]):
    """Property 7: Round-trip works with all 12 attack types having counts.

    When all 12 attack types have associated counts, persisting and restoring
    SHALL produce an equivalent state with all 12 entries preserved.

    **Validates: Requirements 5.3**
    """

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test_stats_full.db")
            db_service = DatabaseService(db_path)
            await db_service.initialize()
            try:
                # Build a full breakdown with all 12 types
                full_breakdown = {
                    ATTACK_TYPE_VALUES[i]: counts[i] for i in range(12)
                }

                accumulator = StatsAccumulator(db_service)
                accumulator.total_predictions = sum(counts)
                accumulator.attacks_detected = sum(counts)
                accumulator.benign_classified = 0
                accumulator._attack_type_counts = dict(full_breakdown)

                await accumulator.persist()

                restored = StatsAccumulator(db_service)
                await restored.restore()

                assert restored.total_predictions == sum(counts)
                assert restored.attacks_detected == sum(counts)
                assert restored.benign_classified == 0
                assert restored._attack_type_counts == full_breakdown
            finally:
                await db_service.close()

    asyncio.run(_run())


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(state=stats_state_strategy)
def test_stats_survive_across_different_instances(state: dict):
    """Property 7: Values survive across different StatsAccumulator instances (simulating restart).

    For any valid stats state, persisting with one StatsAccumulator instance,
    closing the database, reopening it, and restoring with a new instance
    SHALL produce identical values.

    **Validates: Requirements 5.3**
    """

    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "test_stats_restart.db")

            # Phase 1: Create DB, set state, persist, close
            db_service_1 = DatabaseService(db_path)
            await db_service_1.initialize()
            accumulator_1 = StatsAccumulator(db_service_1)
            accumulator_1.total_predictions = state["total_predictions"]
            accumulator_1.attacks_detected = state["attacks_detected"]
            accumulator_1.benign_classified = state["benign_classified"]
            accumulator_1._attack_type_counts = dict(state["attack_type_breakdown"])
            await accumulator_1.persist()
            await db_service_1.close()

            # Phase 2: Reopen DB with a new service (simulating app restart)
            db_service_2 = DatabaseService(db_path)
            await db_service_2.initialize()
            try:
                accumulator_2 = StatsAccumulator(db_service_2)
                await accumulator_2.restore()

                assert accumulator_2.total_predictions == state["total_predictions"], (
                    f"total_predictions mismatch after restart: "
                    f"expected {state['total_predictions']}, got {accumulator_2.total_predictions}"
                )
                assert accumulator_2.attacks_detected == state["attacks_detected"], (
                    f"attacks_detected mismatch after restart: "
                    f"expected {state['attacks_detected']}, got {accumulator_2.attacks_detected}"
                )
                assert accumulator_2.benign_classified == state["benign_classified"], (
                    f"benign_classified mismatch after restart: "
                    f"expected {state['benign_classified']}, got {accumulator_2.benign_classified}"
                )
                assert accumulator_2._attack_type_counts == state["attack_type_breakdown"], (
                    f"attack_type_breakdown mismatch after restart: "
                    f"expected {state['attack_type_breakdown']}, "
                    f"got {accumulator_2._attack_type_counts}"
                )
            finally:
                await db_service_2.close()

    asyncio.run(_run())
