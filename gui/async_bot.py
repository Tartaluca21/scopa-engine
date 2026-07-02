"""Background PIMC worker: run the bot's search off the main (render) thread.

`SearchAgent.select` only reads the engine (it clones internally for search), so
running it in a daemon thread is safe as long as the main thread is the sole
writer. The computed move is handed back via `take()` and executed on the main
thread; the worker never mutates the live engine.
"""

from __future__ import annotations

import threading
from typing import Protocol

from engine.core import ScopaEngine
from search.agent import SearchAgent

Move = tuple[int, list[int]]


class Bot(Protocol):
    """The move-worker interface the controller drives (see `AsyncBot`).

    Structural, so a synchronous test double works in place of the real thread.
    """

    @property
    def thinking(self) -> bool: ...

    @property
    def ready(self) -> bool: ...

    def start(self, engine: ScopaEngine, player: int) -> None: ...

    def take(self) -> Move: ...


class AsyncBot:
    """Runs a `SearchAgent`'s move search in a daemon thread."""

    agent: SearchAgent
    _thread: threading.Thread | None
    _result: Move | None
    _done: threading.Event

    def __init__(self, agent: SearchAgent) -> None:
        self.agent = agent
        self._thread = None
        self._result = None
        self._done = threading.Event()

    @property
    def thinking(self) -> bool:
        """True while a search is running and its result not yet taken."""
        return self._thread is not None and not self._done.is_set()

    @property
    def ready(self) -> bool:
        """True once a search has finished and a move is waiting in `take()`."""
        return self._thread is not None and self._done.is_set()

    def start(self, engine: ScopaEngine, player: int) -> None:
        """Spawn the search thread for `player` (no-op if one is active)."""
        if self._thread is not None:
            return
        self._done.clear()
        self._result = None
        self._thread = threading.Thread(target=self._run, args=(engine, player), daemon=True)
        self._thread.start()

    def _run(self, engine: ScopaEngine, player: int) -> None:
        self._result = self.agent.select(engine, player)
        self._done.set()

    def take(self) -> Move:
        """Join the finished thread and return its move, resetting the worker."""
        if self._thread is None or self._result is None:
            raise RuntimeError("no completed search to take")
        self._thread.join()
        move = self._result
        self._thread = None
        self._result = None
        self._done.clear()
        return move
