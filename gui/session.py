"""Session orchestration for the GUI: logging, matches, and deal advancement.

Wraps a single-deal `GameController` with the cross-deal concerns from the shared
`session` module: it logs every finished deal via `finalize_deal`, accumulates a
`MatchSession` in match mode, and decides whether the next deal continues the
match or starts a fresh session. All logging reuses the CLI schema, so GUI games
appear in `stats.py`, `match_stats.py`, and the decision dataset unchanged.

The wrapped `GameController` instance is stable across deals (only its engine is
replaced), so the render loop can hold a fixed reference to `controller`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gamelog import MATCH_LOG_PATH, DealRecord, MatchRecord
from gui.game import GUI_MAX_DEPTH, GUI_N_WORLDS, BotFactory, GameController
from session import MatchSession, SessionConfig, finalize_deal, new_match_id, pimc_bot_name


def gui_config(target: float | None = None, record_moves: bool = False) -> SessionConfig:
    """A `SessionConfig` naming the GUI's PIMC opponent for the logs."""
    return SessionConfig(
        target=target,
        record_moves=record_moves,
        bot_name=pimc_bot_name(GUI_N_WORLDS, GUI_MAX_DEPTH),
    )


class SessionController:
    """Drives a whole session (single deal or match) over a `GameController`."""

    config: SessionConfig
    controller: GameController
    match: MatchSession | None
    last_deal: DealRecord | None
    last_match: MatchRecord | None
    _deal_logged: bool
    _deal_log_path: Path | None
    _match_log_path: Path

    def __init__(
        self,
        config: SessionConfig | None = None,
        rng: np.random.Generator | None = None,
        *,
        bot_factory: BotFactory | None = None,
        deal_log_path: Path | None = None,
        match_log_path: Path = MATCH_LOG_PATH,
    ) -> None:
        self.config = config if config is not None else gui_config()
        self.controller = GameController(
            rng, record_moves=self.config.record_moves, bot_factory=bot_factory
        )
        self._deal_log_path = deal_log_path
        self._match_log_path = match_log_path
        self.match = None
        self.last_deal = None
        self.last_match = None
        self.start()

    # --- session lifecycle ----------------------------------------------

    def start(self) -> None:
        """Begin a fresh session: reset match bookkeeping, then deal the first."""
        if self.config.target is not None:
            self.match = MatchSession(
                target=self.config.target,
                bot_name=self.config.bot_name,
                match_id=new_match_id(),
            )
        else:
            self.match = None
        self.last_deal = None
        self.last_match = None
        self.controller.new_deal()
        self._deal_logged = False

    def advance(self) -> None:
        """Continue after a finished deal: next deal in a live match, else restart."""
        if self.match is not None and not self.match.is_decided():
            self.controller.new_deal()
            self._deal_logged = False
        else:
            self.start()

    def tick(self, dt: float) -> None:
        """Step the deal, then log it the moment it finishes."""
        self.controller.tick(dt)
        if self.controller.is_over():
            self._finish_deal()

    def _finish_deal(self) -> None:
        """Log the finished deal once, folding it into the match if any."""
        if self._deal_logged:
            return
        self._deal_logged = True
        record = finalize_deal(
            self.controller.engine,
            deal_id=self.controller.deal_id,
            bot_name=self.config.bot_name,
            moves=self.controller.moves,
            path=self._deal_log_path,
        )
        self.last_deal = record
        if self.match is not None:
            self.match.add_deal(record)
            if self.match.is_decided():
                self.last_match = self.match.finish(self._match_log_path)

    # --- queries for the renderer ---------------------------------------

    def is_over(self) -> bool:
        return self.controller.is_over()

    def awaiting_next_deal(self) -> bool:
        """True at deal-over while a match still needs more deals to decide."""
        return self.is_over() and self.match is not None and not self.match.is_decided()

    def match_status_line(self) -> str | None:
        """Overlay caption for the running/finished match; `None` in deal mode."""
        m = self.match
        if m is None:
            return None
        lead = f"You {m.human_total:g} - Bot {m.bot_total:g}"
        if not m.is_decided():
            return f"Match to {m.target:g}   {lead}"
        who = "You win the match!" if m.human_total > m.bot_total else "Bot wins the match!"
        return f"Match to {m.target:g}   {lead}   {who}"
