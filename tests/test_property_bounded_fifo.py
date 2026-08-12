"""Property-based tests for bounded FIFO eviction."""

# Feature: v2-attack-tracking-overhaul, Property 12: Bounded FIFO Eviction

from collections import deque

from hypothesis import given, settings
from hypothesis import strategies as st


# --- BoundedFIFO Implementation ---


class BoundedFIFO:
    """A bounded FIFO collection backed by collections.deque(maxlen=capacity).

    Represents the frontend's arc buffer (200 arcs) and attack log buffer
    (200 entries). When items are added beyond capacity, the oldest item
    is automatically evicted.
    """

    def __init__(self, capacity: int = 200):
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self._capacity = capacity
        self._deque: deque = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._deque)

    def append(self, item) -> None:
        """Add an item. If at capacity, the oldest item is evicted."""
        self._deque.append(item)

    def peek_oldest(self):
        """Return the oldest (first) item without removing it."""
        if not self._deque:
            raise IndexError("peek from empty BoundedFIFO")
        return self._deque[0]

    def items(self) -> list:
        """Return all items in insertion order (oldest first)."""
        return list(self._deque)

    def __getitem__(self, index):
        return self._deque[index]

    def __iter__(self):
        return iter(self._deque)


# --- Strategies ---

# Capacity between 1 and 200 (the spec uses 200, but we test with varied capacities)
capacity_strategy = st.integers(min_value=1, max_value=200)

# Items can be any hashable value; integers are simple and sufficient
item_strategy = st.integers(min_value=0, max_value=10000)

# Sequences of items to add to the FIFO
items_sequence_strategy = st.lists(item_strategy, min_size=0, max_size=500)


# --- Property Tests ---


@settings(max_examples=200)
@given(capacity=capacity_strategy, items=items_sequence_strategy)
def test_bounded_fifo_size_never_exceeds_capacity(capacity: int, items: list[int]):
    """Property 12: Collection size never exceeds capacity.

    For any sequence of N items added to a bounded collection of capacity C,
    the collection size SHALL never exceed C.

    **Validates: Requirements 7.4, 8.3**
    """
    fifo = BoundedFIFO(capacity=capacity)

    for item in items:
        fifo.append(item)
        assert len(fifo) <= capacity, (
            f"Collection size {len(fifo)} exceeds capacity {capacity} "
            f"after adding item {item}"
        )


@settings(max_examples=200)
@given(capacity=capacity_strategy, items=items_sequence_strategy)
def test_bounded_fifo_evicts_oldest_when_full(capacity: int, items: list[int]):
    """Property 12: When collection is full and a new item is added, the oldest is evicted.

    When the collection is full and a new item is added, the removed item SHALL
    be the oldest (first inserted) item.

    **Validates: Requirements 7.4, 8.3**
    """
    fifo = BoundedFIFO(capacity=capacity)

    for i, item in enumerate(items):
        was_full = len(fifo) == capacity
        oldest_before = fifo.peek_oldest() if was_full else None
        items_before = fifo.items() if was_full else None

        fifo.append(item)

        if was_full:
            # The oldest item should no longer be the first item
            # (unless the new item equals the old oldest, in that case
            # the second-oldest becomes the new oldest)
            new_items = fifo.items()
            # The expected result is items_before[1:] + [item]
            expected = items_before[1:] + [item]
            assert new_items == expected, (
                f"After eviction at step {i}: expected {expected}, "
                f"got {new_items}. Oldest was {oldest_before}, new item was {item}"
            )


@settings(max_examples=200)
@given(capacity=capacity_strategy, items=items_sequence_strategy)
def test_bounded_fifo_contains_last_c_items_after_overflow(
    capacity: int, items: list[int]
):
    """Property 12: After adding N > C items, collection contains last C items in order.

    After adding N > C items, the collection SHALL contain exactly the last C
    items in insertion order.

    **Validates: Requirements 7.4, 8.3**
    """
    fifo = BoundedFIFO(capacity=capacity)

    for item in items:
        fifo.append(item)

    n = len(items)
    if n > capacity:
        expected_items = items[-capacity:]
    else:
        expected_items = items

    assert fifo.items() == expected_items, (
        f"After adding {n} items to capacity-{capacity} FIFO: "
        f"expected {expected_items}, got {fifo.items()}"
    )


@settings(max_examples=200)
@given(capacity=capacity_strategy, items=items_sequence_strategy)
def test_bounded_fifo_maintains_fifo_order(capacity: int, items: list[int]):
    """Property 12: Collection maintains FIFO order at all times.

    The collection SHALL maintain FIFO order at all times — items are always
    ordered from oldest (index 0) to newest (last index).

    **Validates: Requirements 7.4, 8.3**
    """
    fifo = BoundedFIFO(capacity=capacity)

    # Track what we expect to be in the FIFO using a reference deque
    reference = deque(maxlen=capacity)

    for item in items:
        fifo.append(item)
        reference.append(item)

        # At every step, the FIFO contents should match the reference
        assert fifo.items() == list(reference), (
            f"FIFO order mismatch: expected {list(reference)}, got {fifo.items()}"
        )
