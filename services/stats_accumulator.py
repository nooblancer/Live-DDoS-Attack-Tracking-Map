"""Stats accumulator service for tracking model predictions and attack metrics.

Maintains running totals, sliding-window throughput, per-IP attacker counts,
and per-attack-type breakdowns. Persists cumulative stats to SQLite for
restart recovery.
"""

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone

from models.schemas import (
    ClassificationResult,
    FeatureImportance,
    ModelStatsResponse,
    TopAttackerEntry,
)
from services.database import DatabaseService

logger = logging.getLogger(__name__)

# Sliding window duration for predictions_per_second calculation
_WINDOW_SECONDS = 60.0

# Keys used in the SQLite stats table
_KEY_TOTAL_PREDICTIONS = "total_predictions"
_KEY_ATTACKS_DETECTED = "attacks_detected"
_KEY_BENIGN_CLASSIFIED = "benign_classified"
_KEY_ATTACK_TYPE_BREAKDOWN = "attack_type_breakdown"


class StatsAccumulator:
    """Tracks prediction counts, throughput, and attack type distribution.

    Uses a sliding 60-second window of timestamps (collections.deque) to
    compute predictions_per_second. Tracks per-IP event counts for top
    attackers and per-attack-type counts for distribution reporting.
    """

    def __init__(self, db: DatabaseService) -> None:
        self._db = db

        # Running counters
        self.total_predictions: int = 0
        self.attacks_detected: int = 0
        self.benign_classified: int = 0

        # Attack type breakdown: type_name → count
        self._attack_type_counts: dict[str, int] = {}

        # Sliding window of timestamps for throughput calculation
        self._prediction_timestamps: deque[float] = deque()

        # Per-IP tracking: ip → {count, country, last_seen}
        self._attacker_counts: dict[str, dict] = {}

        # Model metadata (populated externally or via defaults)
        self._model_metadata: dict[str, float] = {
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "roc_auc": 0.0,
        }
        self._top_features: list[FeatureImportance] = []

    def set_model_metadata(
        self,
        metrics: dict[str, float],
        top_features: list[FeatureImportance] | None = None,
    ) -> None:
        """Set model performance metrics (from metadata.json or model evaluation).

        Parameters
        ----------
        metrics : dict[str, float]
            Keys should include f1_score, precision, recall, roc_auc.
        top_features : list[FeatureImportance] | None
            Top features by importance from the trained model.
        """
        self._model_metadata.update(metrics)
        if top_features is not None:
            self._top_features = top_features

    async def record_prediction(
        self, result: ClassificationResult, source_ip: str, country: str | None = None
    ) -> None:
        """Record a prediction, update counters and per-IP tracking.

        Parameters
        ----------
        result : ClassificationResult
            The classification output for a single flow.
        source_ip : str
            The source IP address associated with this prediction.
        country : str | None
            The country of the source IP (optional).
        """
        now = time.time()

        # Update running counters
        self.total_predictions += 1

        if result.attack_type == "Benign" or result.attack_type == "UNKNOWN":
            self.benign_classified += 1
        else:
            self.attacks_detected += 1

        # Update attack type breakdown
        if result.attack_type != "Benign" and result.attack_type != "UNKNOWN":
            self._attack_type_counts[result.attack_type] = (
                self._attack_type_counts.get(result.attack_type, 0) + 1
            )

        # Record timestamp in the sliding window
        self._prediction_timestamps.append(now)

        # Prune expired timestamps from the window
        self._prune_window(now)

        # Update per-IP counts
        iso_now = datetime.now(timezone.utc).isoformat()
        if source_ip in self._attacker_counts:
            self._attacker_counts[source_ip]["count"] += 1
            self._attacker_counts[source_ip]["last_seen"] = iso_now
            if country is not None:
                self._attacker_counts[source_ip]["country"] = country
        else:
            self._attacker_counts[source_ip] = {
                "count": 1,
                "country": country,
                "last_seen": iso_now,
            }

    def _prune_window(self, now: float) -> None:
        """Remove timestamps older than the sliding window from the deque."""
        cutoff = now - _WINDOW_SECONDS
        while self._prediction_timestamps and self._prediction_timestamps[0] < cutoff:
            self._prediction_timestamps.popleft()

    def get_model_stats(self) -> ModelStatsResponse:
        """Return current model statistics.

        Returns
        -------
        ModelStatsResponse
            Contains model type, metrics, running totals, throughput,
            top features, and attack type breakdown.
        """
        # Prune the window to get accurate throughput
        now = time.time()
        self._prune_window(now)

        predictions_per_second = len(self._prediction_timestamps) / _WINDOW_SECONDS

        return ModelStatsResponse(
            model_type="XGBoost",
            f1_score=self._model_metadata.get("f1_score", 0.0),
            precision=self._model_metadata.get("precision", 0.0),
            recall=self._model_metadata.get("recall", 0.0),
            roc_auc=self._model_metadata.get("roc_auc", 0.0),
            total_predictions=self.total_predictions,
            attacks_detected=self.attacks_detected,
            benign_classified=self.benign_classified,
            predictions_per_second=round(predictions_per_second, 2),
            top_features=self._top_features[:10],
            attack_type_breakdown=dict(self._attack_type_counts),
        )

    def get_top_attackers(self, limit: int = 20) -> list[TopAttackerEntry]:
        """Return top N source IPs by event count.

        IP addresses are partially masked (last two octets replaced with "x.x").

        Parameters
        ----------
        limit : int
            Maximum number of entries to return (default 20).

        Returns
        -------
        list[TopAttackerEntry]
            Sorted descending by attack_count.
        """
        # Sort by count descending
        sorted_attackers = sorted(
            self._attacker_counts.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )

        entries: list[TopAttackerEntry] = []
        for rank, (ip, data) in enumerate(sorted_attackers[:limit], start=1):
            masked_ip = self._mask_ip(ip)
            entries.append(
                TopAttackerEntry(
                    rank=rank,
                    ip_address=masked_ip,
                    country=data.get("country"),
                    attack_count=data["count"],
                    last_seen=data["last_seen"],
                )
            )

        return entries

    def get_attack_types(self) -> dict[str, int]:
        """Return count per attack type.

        Returns
        -------
        dict[str, int]
            Mapping of attack type name to count of classifications.
        """
        return dict(self._attack_type_counts)

    async def persist(self) -> None:
        """Persist cumulative stats to SQLite for restart recovery.

        Stores total_predictions, attacks_detected, benign_classified,
        and attack_type_breakdown as key-value pairs in the stats table.
        """
        if self._db._conn is None:
            logger.warning("Database not initialized; skipping stats persist.")
            return

        iso_now = datetime.now(timezone.utc).isoformat()

        try:
            stats_to_persist = {
                _KEY_TOTAL_PREDICTIONS: str(self.total_predictions),
                _KEY_ATTACKS_DETECTED: str(self.attacks_detected),
                _KEY_BENIGN_CLASSIFIED: str(self.benign_classified),
                _KEY_ATTACK_TYPE_BREAKDOWN: json.dumps(self._attack_type_counts),
            }

            for key, value in stats_to_persist.items():
                await self._db._conn.execute(
                    """INSERT INTO stats (key, value, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at;""",
                    (key, value, iso_now),
                )

            await self._db._conn.commit()
            logger.debug("Stats persisted to SQLite.")

        except Exception as exc:
            logger.error("Failed to persist stats: %s", exc)

    async def restore(self) -> None:
        """Restore stats from SQLite on startup.

        Loads total_predictions, attacks_detected, benign_classified,
        and attack_type_breakdown from the stats table.
        """
        if self._db._conn is None:
            logger.warning("Database not initialized; skipping stats restore.")
            return

        try:
            cursor = await self._db._conn.execute("SELECT key, value FROM stats;")
            rows = await cursor.fetchall()

            for row in rows:
                key, value = row[0], row[1]
                if key == _KEY_TOTAL_PREDICTIONS:
                    self.total_predictions = int(value)
                elif key == _KEY_ATTACKS_DETECTED:
                    self.attacks_detected = int(value)
                elif key == _KEY_BENIGN_CLASSIFIED:
                    self.benign_classified = int(value)
                elif key == _KEY_ATTACK_TYPE_BREAKDOWN:
                    self._attack_type_counts = json.loads(value)

            logger.info(
                "Stats restored: total=%d, attacks=%d, benign=%d, types=%d",
                self.total_predictions,
                self.attacks_detected,
                self.benign_classified,
                len(self._attack_type_counts),
            )

        except Exception as exc:
            logger.error("Failed to restore stats: %s", exc)

    @staticmethod
    def _mask_ip(ip: str) -> str:
        """Partially mask an IP address (replace last two octets with 'x.x').

        Parameters
        ----------
        ip : str
            Full IP address, e.g. "185.220.101.34".

        Returns
        -------
        str
            Masked IP, e.g. "185.220.x.x".
        """
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.x.x"
        # Return as-is for non-IPv4 or unusual formats
        return ip
