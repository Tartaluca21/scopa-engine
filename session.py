"""Non-UI deal/match session logic shared by the CLI (`play.py`) and the GUI.

The interactive loops differ (CLI blocks on stdin; the GUI is an async frame
loop), so the *game loop* itself is not shared. What is shared lives here: the
UI-free pieces that turn a finished engine into a logged `DealRecord`, the
tie-aware match-termination rule, and an incremental match accumulator. Both
front ends produce the *same* JSONL schema, so `stats.py`, `match_stats.py`, and
`scripts/build_decision_dataset.py` read GUI- and CLI-played games identically.

Nothing here mutates the live engine: `finalize_deal` sweeps a clone, matching
`gui.scoreboard.build_scoreboard`, so scoring never depends on call order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from capture import Decision
from engine.core import ScopaEngine
from engine.features import Winner, deal_breakdown
from gamelog import (
    MATCH_LOG_PATH,
    DealRecord,
    MatchRecord,
    log_deal,
    log_match,
)

History = list[Decision] | None

# Player 0 is the human, player 1 is the bot; map breakdown labels accordingly.
_WINNER_NAMES = {"p0": "human", "p1": "bot", "none": "none"}


def named_winner(winner: Winner) -> str:
    """Remap a player-0/1 breakdown label to the human-vs-bot vocabulary."""
    return _WINNER_NAMES[winner]


def pimc_bot_name(n_worlds: int, max_depth: int) -> str:
    """Stable identifier of a PIMC configuration for later analysis."""
    return f"PIMC(n_worlds={n_worlds},max_depth={max_depth})"


def new_match_id() -> int:
    """A fresh, effectively unique match identifier."""
    return int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]


@dataclass(slots=True, frozen=True)
class SessionConfig:
    """What to play and how to log it, chosen once at session start.

    `target` is the match goal, or `None` for a single logged deal.
    `record_moves` toggles the optional per-decision `moves` history.
    """

    target: float | None = None
    record_moves: bool = False
    bot_name: str = "PIMC"


def finalize_deal(
    engine: ScopaEngine,
    *,
    deal_id: int | None,
    bot_name: str,
    moves: History = None,
    path: Path | None = None,
) -> DealRecord:
    """Sweep a clone, score the deal, and append one `DealRecord` (human view).

    Operates on a clone so the live engine is never mutated (the GUI keeps
    rendering it under the end-game overlay). `path` overrides the deal log for
    tests; `None` uses `gamelog`'s default.
    """
    final = engine.clone()
    final.end_of_deal_sweep()
    bd = deal_breakdown(final)
    kwargs: dict[str, object] = {} if path is None else {"path": path}
    return log_deal(
        bd.p0_score,
        bd.p1_score,
        deal_id=deal_id,
        human_scope=bd.p0_scope,
        bot_scope=bd.p1_scope,
        settebello_winner=named_winner(bd.settebello_winner),
        denari_winner=named_winner(bd.denari_winner),
        primiera_winner=named_winner(bd.primiera_winner),
        cards_winner=named_winner(bd.cards_winner),
        bot_name=bot_name,
        moves=moves,
        **kwargs,  # type: ignore[arg-type]
    )


def match_decided(human: float, bot: float, target: float) -> bool:
    """True once a side has reached `target` *and* the scores are not tied.

    A tie at or above the target is undecided: the match plays on until one side
    pulls ahead, so a completed match always has a decisive winner. This is the
    single source of truth for the rule, shared by `play.run_match` (as its stop
    condition) and `MatchSession` (the GUI's event-driven accumulator).
    """
    return (human >= target or bot >= target) and human != bot


@dataclass(slots=True)
class MatchSession:
    """Incremental match accumulator: feed finished deals, then log the match.

    The GUI cannot use `play.run_match`'s pull loop (deals arrive across frames),
    so it accumulates here instead. Same rule (`match_decided`), same `log_match`
    schema, so GUI matches appear in `match_stats.py` beside CLI ones.
    """

    target: float
    bot_name: str
    match_id: int
    human_total: float = 0.0
    bot_total: float = 0.0
    deal_ids: list[int] = field(default_factory=list)

    def add_deal(self, record: DealRecord) -> None:
        """Fold one finished deal into the running match totals."""
        self.human_total += record.human
        self.bot_total += record.bot
        if record.deal_id is not None:
            self.deal_ids.append(record.deal_id)

    def is_decided(self) -> bool:
        """True once the accumulated totals settle the match."""
        return match_decided(self.human_total, self.bot_total, self.target)

    def finish(self, path: Path = MATCH_LOG_PATH) -> MatchRecord:
        """Append the finished match and return its record."""
        return log_match(
            match_id=self.match_id,
            target_score=self.target,
            human_match_score=self.human_total,
            bot_match_score=self.bot_total,
            deal_ids=self.deal_ids,
            bot_name=self.bot_name,
            path=path,
        )
