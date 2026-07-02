"""Display-free tests for the end-game scoring breakdown (gui.scoreboard).

Each test hand-builds an end-of-deal engine state by placing captured cards
directly into the PRESE zones, so the scoreboard can be verified without
launching the Pygame GUI. `score_deal` stays the authority for the winner text.
"""

from __future__ import annotations

import numpy as np

from engine.cards import PRESE_ZONES, Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.heuristic import score_deal
from gui.scoreboard import build_scoreboard

SUITS = (Suit.DENARI, Suit.COPPE, Suit.BASTONI, Suit.SPADE)


def _suit_cards(suit: Suit) -> list[int]:
    """All ten card indices of a suit (values 1..10)."""
    return [card_index(suit, v) for v in range(1, 11)]


def _end_state(
    human: list[int],
    bot: list[int],
    scope_human: int = 0,
    scope_bot: int = 0,
) -> ScopaEngine:
    """Build a finished deal: deck/hands empty, captures placed in PRESE zones."""
    eng = ScopaEngine()
    eng.state[:] = 0
    eng.state[PRESE_ZONES[0], human] = 1
    eng.state[PRESE_ZONES[1], bot] = 1
    eng.scopa_counts = np.array([scope_human, scope_bot], dtype=np.int64)
    eng.current_player = 0
    eng.last_capturer = 0
    eng.rehash()
    return eng


def _winner(eng: ScopaEngine) -> str:
    final = eng.clone()
    final.end_of_deal_sweep()
    you, bot = score_deal(final)
    return "You Win!" if you > bot else "Bot Wins!" if bot > you else "It's a Tie!"


def test_cards_and_denari_counts() -> None:
    human = _suit_cards(Suit.DENARI)  # all 10 denari
    bot = _suit_cards(Suit.COPPE) + _suit_cards(Suit.BASTONI) + _suit_cards(Suit.SPADE)
    board = build_scoreboard(_end_state(human, bot))
    assert board.human.cards == 10
    assert board.bot.cards == 30
    assert board.human.denari == 10
    assert board.bot.denari == 0


def test_settebello_owned_by_human() -> None:
    human = [card_index(Suit.DENARI, 7), card_index(Suit.COPPE, 3)]
    bot = [card_index(Suit.BASTONI, 5)]
    board = build_scoreboard(_end_state(human, bot))
    assert board.human.settebello is True
    assert board.bot.settebello is False


def test_settebello_owned_by_bot() -> None:
    human = [card_index(Suit.COPPE, 3)]
    bot = [card_index(Suit.DENARI, 7), card_index(Suit.BASTONI, 5)]
    board = build_scoreboard(_end_state(human, bot))
    assert board.human.settebello is False
    assert board.bot.settebello is True


def test_primiera_calculation_and_winner() -> None:
    # Human holds every 7 (21 pts/suit -> 84, the max); bot holds every 6 (18 -> 72).
    human = [card_index(s, 7) for s in SUITS]
    bot = [card_index(s, 6) for s in SUITS]
    board = build_scoreboard(_end_state(human, bot))
    assert board.human.primiera == 84
    assert board.bot.primiera == 72
    assert board.human.primiera > board.bot.primiera


def test_scope_count_matches_execution() -> None:
    # Real play: table holds a lone 3 of Coppe; player 0 plays the 3 of Denari,
    # capturing it and clearing the table -> one scopa (deck still non-empty).
    eng = ScopaEngine()
    eng.state[:] = 0
    eng.state[Zone.TAVOLO, card_index(Suit.COPPE, 3)] = 1
    eng.state[Zone.MANO_P1, card_index(Suit.DENARI, 3)] = 1
    eng.state[Zone.MAZZO, card_index(Suit.SPADE, 10)] = 1  # keep it from being last play
    eng.current_player = 0
    eng.rehash()
    scopa = eng.execute_move(card_index(Suit.DENARI, 3), [card_index(Suit.COPPE, 3)])
    assert scopa is True
    assert int(eng.scopa_counts[0]) == 1
    board = build_scoreboard(eng)
    assert board.human.scope == 1
    assert board.bot.scope == 0


def test_winner_text_matches_score_deal_human() -> None:
    # Human: denari (10) + settebello + 2 scope; bot: only the cards majority.
    human = _suit_cards(Suit.DENARI)
    bot = _suit_cards(Suit.COPPE) + _suit_cards(Suit.BASTONI) + _suit_cards(Suit.SPADE)
    eng = _end_state(human, bot, scope_human=2, scope_bot=0)
    board = build_scoreboard(eng)
    you, opp = score_deal(eng.clone())
    assert (board.human_points, board.bot_points) == (you, opp)
    assert board.winner == _winner(eng) == "You Win!"


def test_winner_text_matches_score_deal_bot() -> None:
    # Bot takes cards, denari, settebello, and primiera -> bot wins.
    human = [card_index(Suit.COPPE, 1)]
    bot = _suit_cards(Suit.DENARI) + [card_index(Suit.COPPE, 7)]
    eng = _end_state(human, bot)
    board = build_scoreboard(eng)
    assert board.winner == _winner(eng) == "Bot Wins!"


def test_winner_text_tie() -> None:
    # Every category ties and the settebello (7D) is in neither pile -> 0-0 tie.
    # Cards: 3 each. Denari: 1 each. Primiera: 15+14+13 = 42 each.
    human = [card_index(Suit.COPPE, 5), card_index(Suit.BASTONI, 4), card_index(Suit.DENARI, 3)]
    bot = [card_index(Suit.SPADE, 5), card_index(Suit.DENARI, 4), card_index(Suit.COPPE, 3)]
    eng = _end_state(human, bot)
    board = build_scoreboard(eng)
    assert board.human.cards == board.bot.cards == 3
    assert board.human.denari == board.bot.denari == 1
    assert board.human.primiera == board.bot.primiera == 42
    assert (board.human_points, board.bot_points) == (0.0, 0.0)
    assert board.winner == _winner(eng) == "It's a Tie!"
