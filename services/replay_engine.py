"""Replay engine for streaming CIC-DDoS2019 flows at accelerated speeds.

Loads preprocessed flows from a CSV/Parquet file and streams them to the
MultiClassClassifier at a configurable rate, publishing enhanced attack
events to the EventBus for SSE delivery to dashboard clients.
"""

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import SELECTED_FEATURES
from models.schemas import (
    EnhancedAttackEvent,
    ReplayStatusResponse,
    confidence_to_severity,
)
from services.event_bus import EventBus
from services.geo_pools import GeoCoordinatePools
from services.multi_class_classifier import MultiClassClassifier
from services.stats_accumulator import StatsAccumulator

logger = logging.getLogger(__name__)

# Default dataset path (configurable via environment variable)
_DEFAULT_DATASET_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "sample_data.csv"
)


def _generate_synthetic_flows(n: int = 200) -> pd.DataFrame:
    """Generate synthetic flow data for when no dataset file is available.

    Creates a DataFrame with SELECTED_FEATURES columns and realistic
    DDoS-like traffic patterns for demonstration purposes.

    Parameters
    ----------
    n : int
        Number of synthetic flows to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching SELECTED_FEATURES.
    """
    rng = np.random.default_rng(seed=42)
    data = {}

    for feature in SELECTED_FEATURES:
        if "Flag" in feature or "Flags" in feature:
            data[feature] = rng.choice([0.0, 1.0], size=n).astype(float)
        elif "Duration" in feature or "IAT" in feature or "Total" in feature:
            data[feature] = rng.uniform(0, 10_000_000, size=n).astype(float)
        elif "Bytes/s" in feature or "Packets/s" in feature:
            data[feature] = rng.uniform(100, 1_000_000, size=n).astype(float)
        elif "Length" in feature or "Size" in feature or "Packet" in feature:
            data[feature] = rng.uniform(0, 1500, size=n).astype(float)
        elif "Win" in feature or "win" in feature:
            data[feature] = rng.integers(0, 65535, size=n).astype(float)
        elif "Ratio" in feature:
            data[feature] = rng.uniform(0, 10, size=n).astype(float)
        else:
            data[feature] = rng.uniform(0, 10000, size=n).astype(float)

    return pd.DataFrame(data)


def _generate_random_ip() -> str:
    """Generate a random non-reserved IP address for replay events."""
    first_octet = random.randint(1, 223)
    # Avoid 10.x.x.x, 127.x.x.x, 172.16-31.x.x, 192.168.x.x
    while first_octet in (10, 127):
        first_octet = random.randint(1, 223)
    return f"{first_octet}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


class ReplayEngine:
    """Streams CIC-DDoS2019 flows at accelerated speeds for ML classification.

    Loads flows from a dataset file (CSV or Parquet), classifies each via
    MultiClassClassifier, assigns geo coordinates from GeoCoordinatePools,
    records stats via StatsAccumulator, and publishes enhanced events to the
    EventBus with source_channel="replay".

    Rate control uses inter-event delays based on original timestamp deltas
    divided by speed_multiplier, capped by max_events_per_second.
    """

    def __init__(
        self,
        classifier: MultiClassClassifier,
        event_bus: EventBus,
        stats: StatsAccumulator,
        geo_pools: GeoCoordinatePools,
        dataset_path: str | None = None,
        max_events_per_second: int = 50,
    ) -> None:
        self._classifier = classifier
        self._event_bus = event_bus
        self._stats = stats
        self._geo_pools = geo_pools
        self._dataset_path = dataset_path or os.getenv(
            "REPLAY_DATASET_PATH", _DEFAULT_DATASET_PATH
        )
        self._max_events_per_second = max_events_per_second

        # State
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._speed_multiplier: int = 10
        self._start_time: float | None = None
        self._flows_processed: int = 0
        self._loops_completed: int = 0
        self._current_eps: float = 0.0

        # Dataset
        self._flows: pd.DataFrame | None = None
        self._dataset_loaded: bool = False
        self._dataset_error: str | None = None

    def _load_dataset(self) -> None:
        """Load the dataset from disk or generate synthetic flows as fallback.

        Supports CSV and Parquet formats. If the file doesn't exist or fails
        to load, falls back to synthetic flow generation.
        """
        dataset_path = Path(self._dataset_path)

        try:
            if dataset_path.exists():
                if dataset_path.suffix == ".parquet":
                    df = pd.read_parquet(dataset_path)
                else:
                    df = pd.read_csv(dataset_path, low_memory=False)

                # Strip column name whitespace
                df.columns = df.columns.str.strip()

                # Filter to only SELECTED_FEATURES that exist in the dataset
                available_features = [f for f in SELECTED_FEATURES if f in df.columns]
                if not available_features:
                    logger.warning(
                        "Dataset at %s has no matching features. Generating synthetic flows.",
                        dataset_path,
                    )
                    self._flows = _generate_synthetic_flows()
                else:
                    self._flows = df[available_features].copy()
                    # Fill missing features with zeros
                    for feature in SELECTED_FEATURES:
                        if feature not in self._flows.columns:
                            self._flows[feature] = 0.0

                logger.info(
                    "Dataset loaded: %d flows from %s", len(self._flows), dataset_path
                )
            else:
                logger.warning(
                    "Dataset file not found at %s. Generating synthetic flows.",
                    dataset_path,
                )
                self._flows = _generate_synthetic_flows()

            # Replace NaN/Inf values with 0.0
            self._flows = self._flows.replace([np.inf, -np.inf], 0.0)
            self._flows = self._flows.fillna(0.0)

            self._dataset_loaded = True
            self._dataset_error = None

        except Exception as exc:
            logger.error("Failed to load dataset: %s", exc)
            self._dataset_error = str(exc)
            self._dataset_loaded = False
            # Fallback to synthetic
            self._flows = _generate_synthetic_flows()
            self._dataset_loaded = True

    async def start(self, speed_multiplier: int = 10) -> None:
        """Begin streaming flows. Spawns an asyncio task.

        Parameters
        ----------
        speed_multiplier : int
            Speed factor for replay (1, 10, 100, or 1000).
            Higher values = faster replay.
        """
        if self._running:
            logger.warning("Replay engine already running.")
            return

        self._speed_multiplier = speed_multiplier

        # Load dataset if not yet loaded
        if not self._dataset_loaded:
            self._load_dataset()

        if self._flows is None or self._flows.empty:
            logger.error("No flows available for replay.")
            self._dataset_error = "No flows available"
            return

        self._running = True
        self._start_time = time.time()
        self._flows_processed = 0
        self._loops_completed = 0
        self._current_eps = 0.0

        self._task = asyncio.create_task(self._stream_loop())
        logger.info(
            "Replay engine started: speed=%dx, flows=%d, max_eps=%d",
            speed_multiplier,
            len(self._flows),
            self._max_events_per_second,
        )

    async def stop(self) -> None:
        """Stop the streaming task gracefully."""
        if not self._running:
            logger.debug("Replay engine is not running; stop is a no-op.")
            return

        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info(
            "Replay engine stopped. Processed %d flows, %d loops completed.",
            self._flows_processed,
            self._loops_completed,
        )

    def get_status(self) -> ReplayStatusResponse:
        """Return current replay state.

        Returns
        -------
        ReplayStatusResponse
            Current state including running/stopped, speed, elapsed time,
            flows processed, loops completed, and events per second.
        """
        elapsed = 0.0
        if self._start_time is not None:
            if self._running:
                elapsed = time.time() - self._start_time
            else:
                elapsed = time.time() - self._start_time

        return ReplayStatusResponse(
            state="running" if self._running else "stopped",
            speed_multiplier=self._speed_multiplier,
            elapsed_seconds=round(elapsed, 2),
            flows_processed=self._flows_processed,
            loops_completed=self._loops_completed,
            events_per_second=round(self._current_eps, 2),
        )

    @property
    def is_running(self) -> bool:
        """Whether the replay engine is currently streaming flows."""
        return self._running

    async def _stream_loop(self) -> None:
        """Main streaming loop that iterates through dataset flows.

        Computes inter-event delay from original timestamp deltas divided by
        speed_multiplier, capped by max_events_per_second. When the dataset
        is exhausted, loops back to the beginning.
        """
        # Minimum inter-event delay based on max_events_per_second
        min_delay = 1.0 / self._max_events_per_second if self._max_events_per_second > 0 else 0.0

        # Default inter-event delay when no timestamp delta is available
        # Simulate ~100ms between flows in original dataset
        default_original_delta = 0.1  # 100ms

        flow_index = 0
        total_flows = len(self._flows)

        # Sliding window for EPS calculation
        recent_timestamps: list[float] = []
        eps_window = 5.0  # 5 second window for EPS calculation

        try:
            while self._running:
                # Get current flow
                row = self._flows.iloc[flow_index]

                # Build feature dictionary
                features = {
                    feature: float(row[feature])
                    for feature in SELECTED_FEATURES
                    if feature in row.index
                }

                # Classify the flow
                result = self._classifier.predict(features)

                # Assign geo coordinates based on attack type
                lat, lon = self._geo_pools.get_source_coords(result.attack_type)

                # Generate a source IP for this flow
                source_ip = _generate_random_ip()

                # Record in stats accumulator
                await self._stats.record_prediction(
                    result, source_ip=source_ip, country=None
                )

                # Build enhanced event
                event = EnhancedAttackEvent(
                    ip_address=source_ip,
                    latitude=lat,
                    longitude=lon,
                    country=None,
                    city=None,
                    isp=None,
                    attack_type=result.attack_type,
                    confidence=result.confidence,
                    severity=confidence_to_severity(result.confidence),
                    classified_in_ms=result.classified_in_ms,
                    top_features=result.top_features,
                    source_channel="replay",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                # Publish to EventBus
                await self._event_bus.publish(event.model_dump())

                self._flows_processed += 1

                # Update EPS tracking
                now = time.time()
                recent_timestamps.append(now)
                # Prune old timestamps
                cutoff = now - eps_window
                recent_timestamps = [t for t in recent_timestamps if t >= cutoff]
                if len(recent_timestamps) > 1:
                    elapsed_window = recent_timestamps[-1] - recent_timestamps[0]
                    if elapsed_window > 0:
                        self._current_eps = (len(recent_timestamps) - 1) / elapsed_window
                    else:
                        self._current_eps = 0.0
                else:
                    self._current_eps = 0.0

                # Advance index; loop if at end
                flow_index += 1
                if flow_index >= total_flows:
                    flow_index = 0
                    self._loops_completed += 1
                    logger.debug(
                        "Dataset loop completed. Total loops: %d",
                        self._loops_completed,
                    )

                # Compute inter-event delay
                # Use original_timestamp_delta / speed_multiplier, capped by min_delay
                delay = default_original_delta / self._speed_multiplier
                delay = max(delay, min_delay)

                await asyncio.sleep(delay)

        except asyncio.CancelledError:
            logger.debug("Replay stream loop cancelled.")
            raise
        except Exception as exc:
            logger.error("Replay stream loop error: %s", exc)
            self._running = False
