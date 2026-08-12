"""Timeline bucketing utility for grouping timestamped events by minute.

Provides the `bucket_events` function used by the v2 timeline area chart
panel to aggregate attack events into 1-minute buckets over a 60-minute window.
"""

from datetime import datetime


def bucket_events(
    timestamps: list[datetime], window_minutes: int = 60
) -> dict[str, int]:
    """Group timestamped events into minute-level buckets.

    Each event is assigned to a bucket corresponding to its minute (truncated
    to the start of the minute). The bucket key is the minute-level ISO 8601
    string (e.g., "2024-01-15T14:32:00").

    Parameters
    ----------
    timestamps : list[datetime]
        A list of datetime objects representing event timestamps.
        All timestamps should fall within a `window_minutes` minute window.
    window_minutes : int
        The maximum window size in minutes (default 60). The result will
        contain at most this many buckets.

    Returns
    -------
    dict[str, int]
        A dictionary mapping minute-level ISO strings to event counts,
        sorted chronologically by key.
    """
    buckets: dict[str, int] = {}

    for ts in timestamps:
        # Truncate to minute: zero out seconds and microseconds
        minute_key = ts.replace(second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
        buckets[minute_key] = buckets.get(minute_key, 0) + 1

    # Return sorted by key (chronological since ISO strings sort lexicographically)
    return dict(sorted(buckets.items()))
