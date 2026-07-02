"""Exact endgame solver for deck-empty Scopa states.

Once the deck (talon) is exhausted, the remaining play is a perfect-information
subgame: both hands, the table, and both capture piles are fully known. This
module solves such a state EXACTLY with memoized negamax -- no heuristic, no
depth cap -- returning the true final score margin under optimal play by both
sides.

Assumptions (the caller MUST guarantee; `solve_endgame` checks the first):
  * The deck is empty (`Zone.MAZZO` count == 0). No more cards are dealt, so the
    game tree is finite and shallow -- each side holds at most 3 cards, i.e. <= 6
    plies remain.
  * Both hands are known. This always holds inside a determinized search world,
    which is the only place the solver is invoked.
  * The state is reachable by legal play, so hand sizes stay balanced and every
    non-terminal node has a legal move for the side to move.

Return value: a *margin* in final deal points (carte + denari + settebello +
primiera + scope), from the perspective of the side to move:
    +v  -> the side to move finishes v points ahead under optimal play,
    -v  -> v points behind,
     0  -> the deal is split.
It is computed via the engine's own `execute_move` / `end_of_deal_sweep` /
`score_deal`, so the mandatory-capture rule, the last-play Scopa exception, and
the final table sweep to the last capturer all match live play exactly.
"""

from __future__ import annotations

import math

from engine.cards import HAND_ZONES, Zone
from engine.core import ScopaEngine
from engine.heuristic import score_deal

# Memo key: the Zobrist hash captures zone occupancy, side to move, and scopa
# counts, but NOT `last_capturer` -- which the end-of-deal sweep depends on, so
# two board-identical states with different last capturers have different values.
# Keying on both avoids conflating them (the plain zhash-keyed TranspositionTable
# cannot represent this, so the solver keeps its own memo).
MemoKey = tuple[int, int]


def _terminal_margin(engine: ScopaEngine) -> float:
    """Final score margin from the side-to-move POV (sweep, then score)."""
    final = engine.clone()
    final.end_of_deal_sweep()
    p0, p1 = score_deal(final)
    return (p0 - p1) if engine.current_player == 0 else (p1 - p0)


def _endgame_moves(engine: ScopaEngine, player: int) -> list[tuple[int, list[int]]]:
    """Every legal (card, capture_set) for `player`; [] capture means lay it."""
    moves: list[tuple[int, list[int]]] = []
    for card in engine.cards_in(HAND_ZONES[player]):
        c = int(card)
        options = engine.captures_for(c)
        if options:
            moves.extend((c, [int(x) for x in opt]) for opt in options)
        else:
            moves.append((c, []))
    return moves


def solve_endgame(engine: ScopaEngine, memo: dict[MemoKey, float] | None = None) -> float:
    """Exact negamax margin (side-to-move POV) for a deck-empty state.

    See the module docstring for the assumptions and the meaning of the value.
    `memo` (optional) is reused across calls to share solved subgames; pass the
    same dict for a whole search. Raises `ValueError` if the deck is not empty.
    """
    if engine.count(Zone.MAZZO) != 0:
        raise ValueError("endgame solver requires an empty deck")
    return _solve(engine, memo if memo is not None else {})


def _solve(engine: ScopaEngine, memo: dict[MemoKey, float]) -> float:
    key: MemoKey = (engine.zhash, engine.last_capturer)
    cached = memo.get(key)
    if cached is not None:
        return cached
    if engine.is_game_over():
        value = _terminal_margin(engine)
    else:
        best = -math.inf
        player = engine.current_player
        for card, cap in _endgame_moves(engine, player):
            child = engine.clone()
            child.execute_move(card, cap)
            score = -_solve(child, memo)
            if score > best:
                best = score
        if best == -math.inf:  # non-terminal with no legal move: illegal state
            raise ValueError("no legal move in a non-terminal endgame (illegal state)")
        value = best
    memo[key] = value
    return value
