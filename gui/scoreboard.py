"""End-of-game scoring breakdown for the overlay.

Pulls the definitive Scopa match statistics (cards, denari, primiera,
settebello, scope) for both players and the authoritative point totals. Kept
pygame-free so the numbers are testable without a display; `score_deal` stays
the single source of truth for who wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.cards import PRESE_ZONES
from engine.core import ScopaEngine
from engine.heuristic import capture_features, score_deal


@dataclass(frozen=True, slots=True)
class PlayerStats:
    """One player's captured-pile statistics at the end of a deal."""

    cards: int
    denari: int
    primiera: int
    settebello: bool
    scope: int


@dataclass(frozen=True, slots=True)
class Scoreboard:
    """Both players' stats, point totals, and the winner declaration."""

    human: PlayerStats
    bot: PlayerStats
    human_points: float
    bot_points: float
    winner: str


def _stats(engine: ScopaEngine, player: int) -> PlayerStats:
    f = capture_features(engine.cards_in(PRESE_ZONES[player]))
    return PlayerStats(
        cards=f.captures,
        denari=f.denari,
        primiera=f.primiera,
        settebello=bool(f.settebello),
        scope=int(engine.scopa_counts[player]),
    )


def build_scoreboard(engine: ScopaEngine) -> Scoreboard:
    """Sweep a clone and assemble the final scoreboard for both players."""
    final = engine.clone()
    final.end_of_deal_sweep()
    you, bot = score_deal(final)
    winner = "You Win!" if you > bot else "Bot Wins!" if bot > you else "It's a Tie!"
    return Scoreboard(_stats(final, 0), _stats(final, 1), you, bot, winner)


def scoreboard_rows(board: Scoreboard) -> list[tuple[str, str, str]]:
    """(label, human, bot) rows for the overlay, ending with the point totals."""
    h, b = board.human, board.bot
    return [
        ("Cards", str(h.cards), str(b.cards)),
        ("Denari", str(h.denari), str(b.denari)),
        ("Primiera", str(h.primiera), str(b.primiera)),
        ("Settebello", "Yes" if h.settebello else "No", "Yes" if b.settebello else "No"),
        ("Scope", str(h.scope), str(b.scope)),
        ("Points", f"{board.human_points:g}", f"{board.bot_points:g}"),
    ]
