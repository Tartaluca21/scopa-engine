"""Benchmark and decision-parity harness for the deployed default bot.

Runs deterministic self-play deals with the `botconfig` default (PIMC 12x5,
uniform weights, fixed TT key) and reports throughput. Because every deal is
driven by a fixed per-deal seed, the sequence of chosen moves is fully
reproducible: `--parity` prints a hash of that sequence so any optimization can
be proven decision-preserving by comparing the hash before and after.

    python scripts/benchmark.py                 # timing (default 8 deals)
    python scripts/benchmark.py -n 16           # more deals
    python scripts/benchmark.py --profile        # cProfile hot-path breakdown
    python scripts/benchmark.py --parity         # deterministic decision digest

Timing uses `time.perf_counter` (wall) and `time.process_time` (CPU) around the
search only; dealing and scoring are excluded so ms/move reflects the bot.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import pstats
import time
from dataclasses import dataclass

import numpy as np

from botconfig import default_pimc_config
from engine.cards import Zone
from engine.core import ScopaEngine
from engine.heuristic import score_deal
from search.pimc import pimc_decide

BASE_SEED = 1000


@dataclass(slots=True)
class Result:
    """Aggregated benchmark outcome across all played deals."""

    deals: int
    moves: int
    wall: float
    cpu: float
    digest: str

    @property
    def ms_per_move(self) -> float:
        return 1000.0 * self.wall / self.moves

    @property
    def deals_per_sec(self) -> float:
        return self.deals / self.wall

    def report(self) -> str:
        return (
            f"deals        : {self.deals}\n"
            f"moves        : {self.moves}\n"
            f"wall time    : {self.wall:.3f} s\n"
            f"cpu time     : {self.cpu:.3f} s\n"
            f"ms/move      : {self.ms_per_move:.3f}\n"
            f"deals/sec    : {self.deals_per_sec:.2f}\n"
            f"decision hash: {self.digest}"
        )


def play_deal(seed: int, hasher: hashlib._Hash) -> tuple[int, float, float]:
    """Self-play one deal with the default bot; fold each move into `hasher`.

    Returns `(n_moves, wall_seconds, cpu_seconds)` measured around the search
    calls only. The RNG is seeded per deal so the move sequence is deterministic.
    """
    rng = np.random.default_rng(seed)
    cfg = default_pimc_config()
    engine = ScopaEngine()
    engine.deal_round(rng)
    moves = 0
    wall = 0.0
    cpu = 0.0
    while not engine.is_game_over():
        if engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0:
            engine.deal_round(rng)
            continue
        player = engine.current_player
        w0, c0 = time.perf_counter(), time.process_time()
        card, cap = pimc_decide(engine, player, cfg, rng)
        wall += time.perf_counter() - w0
        cpu += time.process_time() - c0
        hasher.update(f"{seed}:{moves}:{player}:{card}:{sorted(cap)}".encode())
        engine.execute_move(card, cap)
        moves += 1
    engine.end_of_deal_sweep()
    score_deal(engine)
    return moves, wall, cpu


def run(n_deals: int) -> Result:
    """Play `n_deals` deterministic self-play deals and aggregate the metrics."""
    hasher = hashlib.sha256()
    total_moves = 0
    total_wall = 0.0
    total_cpu = 0.0
    for i in range(n_deals):
        m, w, c = play_deal(BASE_SEED + i, hasher)
        total_moves += m
        total_wall += w
        total_cpu += c
    return Result(n_deals, total_moves, total_wall, total_cpu, hasher.hexdigest()[:16])


def main() -> None:
    parser = argparse.ArgumentParser(description="Default-bot benchmark / parity harness.")
    parser.add_argument("-n", "--deals", type=int, default=8, help="number of deals")
    parser.add_argument("--profile", action="store_true", help="print cProfile breakdown")
    parser.add_argument("--parity", action="store_true", help="print only the decision hash")
    args = parser.parse_args()

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        result = run(args.deals)
        pr.disable()
        stats = pstats.Stats(pr).sort_stats("tottime")
        stats.print_stats(25)

    result = run(args.deals)
    if args.parity:
        print(result.digest)
    else:
        print(result.report())


if __name__ == "__main__":
    main()
