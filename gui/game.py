"""Game controller: owns the live engine, the human seat, and the async PIMC bot.

Keeps game rules out of the render loop and scopes itself to a *single deal*: the
match/logging orchestration that spans deals lives in `gui.session`. The human is
player 0, the trained champion `SearchAgent` is player 1, run off-thread via
`AsyncBot`. Single-option captures resolve immediately; ambiguous ones open a
`PendingMove` the human resolves by clicking table cards. When move recording is
on, each decision is snapshotted (pre-move) into `moves` via `decision_record`,
ready for `finalize_deal` to persist.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from botconfig import DEFAULT_MAX_DEPTH, DEFAULT_N_WORLDS, DEFAULT_WEIGHTS
from capture import decision_record
from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine
from gui.async_bot import AsyncBot, Bot
from gui.moves import CaptureAnim, PendingMove
from gui.scoreboard import Scoreboard, build_scoreboard
from search.agent import SearchAgent
from session import History

HUMAN = 0
BOT = 1

# The GUI opponent is the single deployed default bot (see `botconfig`); these
# aliases keep the config visible here and name it identically for the logs.
GUI_N_WORLDS = DEFAULT_N_WORLDS
GUI_MAX_DEPTH = DEFAULT_MAX_DEPTH

BotFactory = Callable[[np.random.Generator], Bot]


def _default_bot(rng: np.random.Generator) -> Bot:
    """Build the real off-thread champion worker for a fresh deal."""
    return AsyncBot(
        SearchAgent(DEFAULT_WEIGHTS, rng, n_worlds=GUI_N_WORLDS, max_depth=GUI_MAX_DEPTH)
    )


class GameController:
    """Drives one Scopa deal: human moves, off-thread bot replies, deal refills.

    `record_moves` toggles per-decision capture into `moves`. Deal identity lives
    in `deal_id`; `new_deal()` starts the next round on the same instance so the
    render loop keeps a stable controller reference across deals.
    """

    rng: np.random.Generator
    record_moves: bool
    engine: ScopaEngine
    bot: Bot
    pending: PendingMove | None
    capture_anim: CaptureAnim | None
    moves: History
    deal_id: int
    _turn: int
    _cached_scoreboard: Scoreboard | None
    _bot_factory: BotFactory

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        *,
        record_moves: bool = False,
        bot_factory: BotFactory | None = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.record_moves = record_moves
        self._bot_factory = bot_factory if bot_factory is not None else _default_bot
        self.new_deal()

    # --- deal lifecycle --------------------------------------------------

    def new_deal(self) -> None:
        """Deal a fresh round (reproducible from `deal_id`) and a new bot worker."""
        seed = np.random.SeedSequence()
        self.deal_id = int(seed.entropy)  # type: ignore[arg-type]
        self.engine = ScopaEngine()
        self.engine.deal_round(np.random.default_rng(seed))
        self.bot = self._bot_factory(self.rng)
        self.pending = None
        self.capture_anim = None
        self._cached_scoreboard = None
        self.moves = [] if self.record_moves else None
        self._turn = 0

    def reset(self) -> None:
        """Alias for `new_deal` (kept for external callers)."""
        self.new_deal()

    def _hands_empty(self) -> bool:
        return self.engine.count(Zone.MANO_P1) == 0 and self.engine.count(Zone.MANO_P2) == 0

    def maybe_deal(self) -> None:
        """Refill both hands from the deck when they run out mid-game."""
        if not self.engine.is_game_over() and self._hands_empty():
            self.engine.deal_round(self.rng)

    def is_over(self) -> bool:
        return self.engine.is_game_over()

    def is_human_turn(self) -> bool:
        """True if it is the human's move and the deal is still running."""
        return not self.engine.is_game_over() and self.engine.current_player == HUMAN

    @property
    def bot_thinking(self) -> bool:
        return self.bot.thinking

    def tick(self, dt: float) -> None:
        """Per-frame step: capture staging, deals, bot start/collect lifecycle."""
        if self.capture_anim is not None:
            self._advance_capture(dt)
            return
        self.maybe_deal()
        if self.is_over() or self.pending is not None:
            return
        if self.engine.current_player == BOT:
            if self.bot.ready:
                card, capture = self.bot.take()
                self._play(card, capture)
            elif not self.bot.thinking:
                self.bot.start(self.engine, BOT)

    # --- move execution (with staged capture animation) -----------------

    def _record(self, card: int, capture: list[int]) -> None:
        """Snapshot a decision (pre-move) into the moves history, if recording."""
        if self.moves is None:
            return
        player = int(self.engine.current_player)
        label = "human" if player == HUMAN else "bot"
        self.moves.append(decision_record(self.engine, label, player, self._turn, card, capture))
        self._turn += 1

    def _play(self, card: int, capture: list[int]) -> None:
        """Run a non-capturing play at once; stage a capture for animation."""
        self._record(card, capture)
        if capture:
            self.capture_anim = CaptureAnim(card, list(capture), int(self.engine.current_player))
        else:
            self.engine.execute_move(card, [])
            self.maybe_deal()

    def _advance_capture(self, dt: float) -> None:
        """Hold the staged capture until its slide-and-pause window elapses."""
        anim = self.capture_anim
        if anim is None:
            return
        anim.elapsed += dt
        if anim.done:
            self.engine.execute_move(anim.card, anim.capture)
            self.capture_anim = None
            self.maybe_deal()

    # --- human interaction ----------------------------------------------

    def begin_human_move(self, card: int) -> None:
        """Resolve a clicked hand card: play now, or open a capture selection."""
        if not self.is_human_turn() or self.pending is not None or self.capture_anim is not None:
            return
        if card not in (int(c) for c in self.engine.cards_in(HAND_ZONES[HUMAN])):
            return
        options = [[int(c) for c in opt] for opt in self.engine.captures_for(card)]
        if len(options) <= 1:
            self._play(card, options[0] if options else [])
        else:
            self.pending = PendingMove(card, options)

    def toggle_table_card(self, idx: int) -> None:
        """Toggle a candidate table card; auto-commit on an exact option match."""
        if self.pending is None or idx not in self.pending.candidates():
            return
        if idx in self.pending.selected:
            self.pending.selected.discard(idx)
        else:
            self.pending.selected.add(idx)
        option = self.pending.matched_option()
        if option is not None:
            card = self.pending.card
            self.pending = None
            self._play(card, option)

    def cancel_pending(self) -> None:
        """Abandon an in-progress capture selection."""
        self.pending = None

    def selection_highlight(self) -> tuple[set[int], set[int]]:
        """(candidate cards, selected cards) for the renderer; empty if idle."""
        if self.pending is None:
            return set(), set()
        return self.pending.candidates(), set(self.pending.selected)

    # --- result ----------------------------------------------------------

    def scoreboard(self) -> Scoreboard:
        """Final breakdown for the overlay, built once and cached at game-over."""
        if self._cached_scoreboard is not None:
            return self._cached_scoreboard
        board = build_scoreboard(self.engine)
        if self.is_over():
            self._cached_scoreboard = board
        return board
