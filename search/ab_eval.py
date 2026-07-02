"""Paired self-play A/B evaluation for PIMC bot changes.

Compares a `baseline` PimcConfig against a `challenger` over `n_pairs` deals.
Each pair reuses ONE freshly seeded deck twice with the seats swapped, so both
bots play both sides of the identical deal -- this cancels deck luck and dealer
advantage, the low-variance estimator the README's Empirical Findings rely on.

The dealing RNG is kept separate from the agents' sampling RNGs, so the whole
card sequence of a deal is a pure function of its seed regardless of how either
bot plays: the two swapped games in a pair see byte-for-byte the same deck.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from engine.cards import Zone
from engine.core import ScopaEngine
from engine.features import DealBreakdown, deal_breakdown
from search.pimc import PimcConfig, pimc_decide

# One eval job: (baseline_cfg, challenger_cfg, deal_seed, agent_seed_base).
ABJob = tuple[PimcConfig, PimcConfig, int, int]
_COMPONENTS = ("settebello", "denari", "primiera", "cards")


@dataclass(slots=True, frozen=True)
class GameOutcome:
    """One finished deal, relabelled from p0/p1 to baseline/challenger."""

    base_score: float
    chal_score: float
    base_scope: int
    chal_scope: int
    settebello: str  # "baseline" | "challenger" | "none"
    denari: str
    primiera: str
    cards: str


def _play_deal(
    p0_cfg: PimcConfig, p1_cfg: PimcConfig, deal_seed: int, seed0: int, seed1: int
) -> DealBreakdown:
    """Play one full deal; p0/p1 use their own configs and sampling RNGs."""
    engine = ScopaEngine()
    deal_rng = np.random.default_rng(deal_seed)
    rngs = (np.random.default_rng(seed0), np.random.default_rng(seed1))
    cfgs = (p0_cfg, p1_cfg)
    engine.deal_round(deal_rng)
    while not engine.is_game_over():
        if engine.count(Zone.MANO_P1) == 0 and engine.count(Zone.MANO_P2) == 0:
            engine.deal_round(deal_rng)
            continue
        player = engine.current_player
        card, capture = pimc_decide(engine, player, cfgs[player], rngs[player])
        engine.execute_move(card, capture)
    engine.end_of_deal_sweep()
    return deal_breakdown(engine)


def _relabel(winner: str, base_is_p0: bool) -> str:
    """Map a p0/p1/none component winner to baseline/challenger/none."""
    if winner == "none":
        return "none"
    base_seat = "p0" if base_is_p0 else "p1"
    return "baseline" if winner == base_seat else "challenger"


def _outcome(bd: DealBreakdown, base_is_p0: bool) -> GameOutcome:
    """Relabel a p0/p1 breakdown by which seat the baseline occupied."""
    if base_is_p0:
        base_s, chal_s, base_sc, chal_sc = bd.p0_score, bd.p1_score, bd.p0_scope, bd.p1_scope
    else:
        base_s, chal_s, base_sc, chal_sc = bd.p1_score, bd.p0_score, bd.p1_scope, bd.p0_scope
    return GameOutcome(
        base_score=base_s,
        chal_score=chal_s,
        base_scope=base_sc,
        chal_scope=chal_sc,
        settebello=_relabel(bd.settebello_winner, base_is_p0),
        denari=_relabel(bd.denari_winner, base_is_p0),
        primiera=_relabel(bd.primiera_winner, base_is_p0),
        cards=_relabel(bd.cards_winner, base_is_p0),
    )


def _paired_trial(job: ABJob) -> tuple[GameOutcome, GameOutcome]:
    """Play one deal twice with seats swapped (process-pool worker)."""
    base_cfg, chal_cfg, deal_seed, a = job
    game_a = _play_deal(base_cfg, chal_cfg, deal_seed, a, a + 1)  # baseline = p0
    game_b = _play_deal(chal_cfg, base_cfg, deal_seed, a + 2, a + 3)  # baseline = p1
    return _outcome(game_a, True), _outcome(game_b, False)


@dataclass(slots=True)
class ABResult:
    """Aggregate of a paired A/B run, from the challenger's perspective."""

    n_pairs: int
    baseline_label: str
    challenger_label: str
    games: list[GameOutcome]

    @property
    def n_games(self) -> int:
        return len(self.games)

    def _margins(self) -> list[float]:
        return [g.chal_score - g.base_score for g in self.games]

    def _pair_means(self) -> list[float]:
        m = self._margins()
        return [(m[2 * i] + m[2 * i + 1]) / 2.0 for i in range(self.n_pairs)]

    def summary(self) -> dict[str, float]:
        """Challenger-vs-baseline headline numbers (margin in points/deal)."""
        pair_means = self._pair_means()
        mean = float(np.mean(pair_means))
        # Paired SE across pairs; the swap cancels deck luck and seat advantage.
        sd = float(np.std(pair_means, ddof=1)) if self.n_pairs > 1 else 0.0
        se = sd / math.sqrt(self.n_pairs) if self.n_pairs else 0.0
        wins = sum(g.chal_score > g.base_score for g in self.games)
        ties = sum(g.chal_score == g.base_score for g in self.games)
        return {
            "margin": mean,
            "ci95": 1.96 * se,
            "win_rate": (wins + 0.5 * ties) / self.n_games if self.n_games else 0.0,
            "chal_avg": float(np.mean([g.chal_score for g in self.games])),
            "base_avg": float(np.mean([g.base_score for g in self.games])),
            "chal_scope_avg": float(np.mean([g.chal_scope for g in self.games])),
            "base_scope_avg": float(np.mean([g.base_scope for g in self.games])),
        }

    def _component_tally(self) -> dict[str, dict[str, int]]:
        tally = {c: {"baseline": 0, "challenger": 0, "none": 0} for c in _COMPONENTS}
        for g in self.games:
            for c in _COMPONENTS:
                tally[c][getattr(g, c)] += 1
        return tally

    def to_markdown(self) -> str:
        """Render a clear Markdown report of the paired A/B run."""
        s = self.summary()
        verdict = "challenger better" if s["margin"] > 0 else "baseline better"
        if abs(s["margin"]) <= s["ci95"]:
            verdict = "no significant difference (95% CI spans 0)"
        lines = [
            "# Paired A/B self-play",
            "",
            f"- Baseline   : {self.baseline_label}",
            f"- Challenger : {self.challenger_label}",
            f"- Pairs      : {self.n_pairs}  ({self.n_games} games, seats swapped)",
            "",
            "## Headline (challenger − baseline)",
            "",
            f"- Margin        : **{s['margin']:+.3f} ± {s['ci95']:.3f}** pts/deal (95% CI)",
            f"- Verdict       : **{verdict}**",
            f"- Win-rate      : {s['win_rate']:.1%}  (challenger, ties as half)",
            f"- Avg score     : challenger {s['chal_avg']:.2f} vs baseline {s['base_avg']:.2f}",
            f"- Avg scope     : challenger {s['chal_scope_avg']:.2f} "
            f"vs baseline {s['base_scope_avg']:.2f}",
            "",
            f"## Component wins (of {self.n_games} games)",
            "",
            "| component | challenger | baseline | split |",
            "|---|---|---|---|",
        ]
        tally = self._component_tally()
        for c in _COMPONENTS:
            t = tally[c]
            lines.append(f"| {c} | {t['challenger']} | {t['baseline']} | {t['none']} |")
        return "\n".join(lines)


def run_ab(
    baseline: PimcConfig,
    challenger: PimcConfig,
    n_pairs: int,
    seed: int = 0,
    workers: int = 4,
    baseline_label: str = "baseline",
    challenger_label: str = "challenger",
) -> ABResult:
    """Run `n_pairs` paired deals and aggregate the result.

    Identical results for any `workers` (each pair is independently seeded);
    set `workers <= 1` to run sequentially.
    """
    jobs: list[ABJob] = [
        (baseline, challenger, seed + i, (seed + i) * 4 + 1) for i in range(n_pairs)
    ]
    if workers <= 1:
        results = [_paired_trial(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_paired_trial, jobs))
    games: list[GameOutcome] = []
    for game_a, game_b in results:
        games.append(game_a)
        games.append(game_b)
    return ABResult(n_pairs, baseline_label, challenger_label, games)
