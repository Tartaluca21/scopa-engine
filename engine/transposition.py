"""Transposition Table (Phase 2): hash-state -> evaluation cache.

Stores evaluations of already-explored states to avoid re-exploring identical
branches. Key = Zobrist hash. Entries carry a node type (EXACT / LOWER / UPPER)
for future alpha-beta pruning, and the table is size-bounded with FIFO eviction
to prevent unbounded memory growth during deep rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class NodeType(IntEnum):
    """Alpha-beta bound classification of a stored evaluation."""

    EXACT = 0  # exact value
    LOWER = 1  # fail-high: value is a lower bound
    UPPER = 2  # fail-low: value is an upper bound


@dataclass(slots=True)
class TTEntry:
    """Table entry: evaluation of a state at a given search depth."""

    zhash: int
    value: float
    depth: int
    node_type: NodeType


DEFAULT_CAPACITY = 1 << 20  # ~1M entries


class TranspositionTable:
    """In-memory cache indexed by Zobrist hash, bounded by `capacity`."""

    _table: dict[int, TTEntry]
    capacity: int

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._table = {}
        self.capacity = capacity

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, zhash: int) -> bool:
        return zhash in self._table

    def get(self, zhash: int) -> TTEntry | None:
        """Entry for `zhash`, or None if never seen."""
        return self._table.get(zhash)

    def store(
        self,
        zhash: int,
        value: float,
        depth: int,
        node_type: NodeType = NodeType.EXACT,
    ) -> None:
        """Insert/update the entry, keeping the deeper result; evict FIFO if full."""
        prev = self._table.get(zhash)
        if prev is not None and depth < prev.depth:
            return
        if prev is None and len(self._table) >= self.capacity:
            self._table.pop(next(iter(self._table)))  # evict oldest insertion
        self._table[zhash] = TTEntry(zhash, value, depth, node_type)

    def clear(self) -> None:
        self._table.clear()
