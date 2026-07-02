"""Interactive human-vs-bot Scopa CLI.

The human is player 0, the PIMC bot is player 1. Cards are shown with their
Italian names (the game's domain language); all code stays English. Run with:

    python play.py                 # one deal (default)
    python play.py --match-to 11   # deals until someone reaches 11 points
    python play.py --match-to 21   # ... or 21
    python play.py --record-moves  # also capture each decision into the log

Every deal is logged via `gamelog` regardless of mode; a match additionally
logs one match-level record. With `--record-moves`, each deal log also stores a
per-decision `moves` history for `scripts/build_decision_dataset.py`. See
`stats.py` and `match_stats.py` for reports.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from botconfig import default_pimc_config
from capture import Decision, decision_record
from engine.cards import Zone
from engine.core import ScopaEngine
from gamelog import DealRecord, MatchRecord, log_match
from search.pimc import PimcConfig, pimc_decide
from session import finalize_deal, match_decided, pimc_bot_name
from ui import describe_move, enumerate_moves, read_choice, show_state

History = list[Decision] | None

HUMAN = 0
BOT = 1


def human_turn(engine: ScopaEngine, turn: int = 0, history: History = None) -> None:
    """Show the legal moves, read the human's pick, and apply it."""
    show_state(engine, HUMAN)
    moves = enumerate_moves(engine, HUMAN)
    print("\nYour moves:")
    for i, (card, cap) in enumerate(moves, start=1):
        print(f"  {i}. {describe_move(card, cap)}")
    card, cap = moves[read_choice(len(moves))]
    if history is not None:
        history.append(decision_record(engine, "human", HUMAN, turn, card, cap))
    scopa = engine.execute_move(card, cap)
    print(f"You: {describe_move(card, cap)}" + ("  -- SCOPA!" if scopa else ""))


def bot_turn(
    engine: ScopaEngine,
    cfg: PimcConfig,
    rng: np.random.Generator,
    turn: int = 0,
    history: History = None,
) -> None:
    """Let the PIMC bot decide quickly, then apply and announce its move."""
    print("\nBot is thinking...")
    card, cap = pimc_decide(engine, BOT, cfg, rng)
    if history is not None:
        history.append(decision_record(engine, "bot", BOT, turn, card, cap))
    scopa = engine.execute_move(card, cap)
    print(f"Bot: {describe_move(card, cap)}" + ("  -- SCOPA!" if scopa else ""))


def _hands_empty(engine: ScopaEngine) -> bool:
    return engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0


def bot_name(cfg: PimcConfig) -> str:
    """Stable identifier of the bot's configuration for later analysis."""
    return pimc_bot_name(cfg.n_worlds, cfg.search.max_depth)


def show_final(
    engine: ScopaEngine, cfg: PimcConfig, deal_id: int, moves: History = None
) -> DealRecord:
    """Print the final score and log the rich deal record (via `session`)."""
    record = finalize_deal(engine, deal_id=deal_id, bot_name=bot_name(cfg), moves=moves)
    you, bot = record.human, record.bot
    print("\n=== Deal over ===")
    print(f"You: {you:g}    Bot: {bot:g}")
    if you > bot:
        print("You win the deal!")
    elif bot > you:
        print("Bot wins the deal!")
    else:
        print("The deal is a tie.")
    return record


def run_deal(
    cfg: PimcConfig, rng: np.random.Generator, deal_id: int, record_moves: bool = False
) -> DealRecord:
    """Play one full interactive deal to the end and return its logged record."""
    engine = ScopaEngine()
    engine.deal_round(rng)
    history: History = [] if record_moves else None
    turn = 0
    while not engine.is_game_over():
        if _hands_empty(engine):
            engine.deal_round(rng)
            print("\n-- New cards dealt --")
            continue
        if engine.current_player == HUMAN:
            human_turn(engine, turn, history)
        else:
            bot_turn(engine, cfg, rng, turn, history)
        turn += 1
    return show_final(engine, cfg, deal_id, history)


def _interactive_deal(cfg: PimcConfig, record_moves: bool = False) -> DealRecord:
    """Seed a fresh deal (reproducible from its deal_id) and play it out."""
    seed_seq = np.random.SeedSequence()
    deal_id = int(seed_seq.entropy)
    return run_deal(cfg, np.random.default_rng(seed_seq), deal_id, record_moves)


def run_match(target: float, deal_provider: Callable[[], DealRecord]) -> list[DealRecord]:
    """Pull deals from `deal_provider` until a side leads with at least `target`.

    A match ends only when someone has reached the target *and* the scores are
    not tied; a tie at or above the target plays on until one side pulls ahead,
    so a completed match always has a decisive winner. Pure control flow: deal
    production (interactive or canned) is injected, so this is unit-testable
    without stdin. Returns the deals played, in order.
    """
    records: list[DealRecord] = []
    human = bot = 0.0
    while not match_decided(human, bot, target):
        record = deal_provider()
        records.append(record)
        human += record.human
        bot += record.bot
        print(f"\n-- Match score: you {human:g}  bot {bot:g}  (target {target:g}) --")
    return records


def play_match(target: float, cfg: PimcConfig, record_moves: bool = False) -> MatchRecord:
    """Play a full match to `target`, logging each deal and the final match."""
    match_id = int(np.random.SeedSequence().entropy)
    print(f"=== Scopa match to {target:g}: you (player 1) vs the PIMC bot ===")
    records = run_match(target, lambda: _interactive_deal(cfg, record_moves))
    human = sum(r.human for r in records)
    bot = sum(r.bot for r in records)
    deal_ids = [r.deal_id for r in records if r.deal_id is not None]
    match = log_match(
        match_id=match_id,
        target_score=target,
        human_match_score=human,
        bot_match_score=bot,
        deal_ids=deal_ids,
        bot_name=bot_name(cfg),
    )
    verdict = {
        "human": "You win the match!",
        "bot": "Bot wins the match!",
        "tie": "The match is a tie.",
    }
    print("\n=== Match over ===")
    print(f"Final: you {human:g}    Bot: {bot:g}    ({match.n_deals} deals)")
    print(verdict[match.winner])
    print("(match logged -- run `python match_stats.py` for your match record)")
    return match


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options: `--match-to N` (one-deal if omitted) and `--record-moves`."""
    parser = argparse.ArgumentParser(description="Play Scopa against the PIMC bot.")
    parser.add_argument(
        "--match-to",
        type=int,
        default=None,
        metavar="N",
        help="play deals until you or the bot reach N points (e.g. 11 or 21); "
        "omit for a single deal",
    )
    parser.add_argument(
        "--record-moves",
        action="store_true",
        help="capture each decision into the deal log's `moves` field "
        "(for building an opponent-modeling dataset)",
    )
    args = parser.parse_args(argv)
    if args.match_to is not None and args.match_to <= 0:
        parser.error("--match-to must be a positive integer")
    return args


def main() -> None:
    opts = parse_args()
    bot_cfg = default_pimc_config()  # the single deployed default bot (CLI == GUI)
    if opts.match_to is None:
        print("=== Scopa: you (player 1) vs the PIMC bot ===")
        _interactive_deal(bot_cfg, opts.record_moves)
        print("(result logged -- run `python stats.py` for your record vs the bot)")
    else:
        play_match(float(opts.match_to), bot_cfg, opts.record_moves)


if __name__ == "__main__":
    main()
