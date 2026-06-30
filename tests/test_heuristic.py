"""Phase 3 tests: heuristic evaluation, exposure-aware selection, GA, self-play."""

from __future__ import annotations

import numpy as np

from engine.cards import Suit, Zone, card_index
from engine.core import ScopaEngine
from engine.genetic import BotPopulation, GeneticConfig, round_robin_fitness
from engine.heuristic import (
    HeuristicBot,
    Weights,
    capture_features,
    evaluate,
    score_deal,
    simulate_match,
)


def _capture(eng: ScopaEngine, suit: Suit, value: int, zone: Zone) -> None:
    eng.move(card_index(suit, value), Zone.MAZZO, zone)


def test_weights_vector_roundtrip() -> None:
    w = Weights(0.5, 0.0, 2.0, 0.25, 1.0)
    assert Weights.from_vector(w.to_vector()) == w
    assert w.to_vector().shape == (5,)


def test_random_weights_are_non_negative() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert np.all(Weights.random(rng).to_vector() >= 0.0)


def test_capture_features_counts_and_primiera() -> None:
    captured = np.array(
        [card_index(Suit.DENARI, 7), card_index(Suit.COPPE, 6)], dtype=np.intp
    )
    f = capture_features(captured)
    assert (f.captures, f.denari, f.settebello, f.primiera) == (2, 1, 1, 21 + 18)


def test_evaluate_rewards_settebello() -> None:
    weights = Weights(captures=0, denari=0, settebello=1, primiera=0, scope=0)
    eng = ScopaEngine()
    _capture(eng, Suit.DENARI, 7, Zone.PRESE_P1)
    assert evaluate(eng, 0, weights) == 1.0
    assert evaluate(eng, 1, weights) == 0.0


def test_bot_prefers_capturing_card() -> None:
    eng = ScopaEngine()
    _capture(eng, Suit.DENARI, 7, Zone.TAVOLO)
    taker = card_index(Suit.SPADE, 7)
    dud = card_index(Suit.SPADE, 2)
    eng.move(taker, Zone.MAZZO, Zone.MANO_P1)
    eng.move(dud, Zone.MAZZO, Zone.MANO_P1)
    card, capture = HeuristicBot(Weights()).select(eng, 0)
    assert card == taker
    assert capture == [card_index(Suit.DENARI, 7)]


def test_exposure_penalty_prefers_safer_table() -> None:
    # Hand has two non-capturing cards; laying the 10 leaves fewer <=10 subsets.
    eng = ScopaEngine()
    _capture(eng, Suit.COPPE, 9, Zone.TAVOLO)  # table has a 9 (no captures here)
    low = card_index(Suit.SPADE, 1)  # laying 1 -> table {9,1}, subset 9+1=10 exposed
    high = card_index(Suit.SPADE, 10)  # laying 10 -> table {9,10}, no <=10 combo of two
    eng.move(low, Zone.MAZZO, Zone.MANO_P1)
    eng.move(high, Zone.MAZZO, Zone.MANO_P1)
    weights = Weights(captures=1, denari=0, settebello=0, primiera=0, scope=1)
    assert HeuristicBot(weights).choose_move(eng, 0) == high


def test_population_evolve_preserves_size_and_elite() -> None:
    rng = np.random.default_rng(1)
    cfg = GeneticConfig(population_size=10, elite_frac=0.2, mutation_sigma=0.05)
    pop = BotPopulation(cfg, rng)
    best = pop.genomes[3]
    fitness = [0.0] * 10
    fitness[3] = 100.0
    pop.evolve(fitness, rng)
    assert len(pop) == 10
    assert best in pop.genomes


def test_score_deal_ties_award_nothing() -> None:
    eng = ScopaEngine()  # empty piles: every category tied
    assert score_deal(eng) == (0.0, 0.0)


def test_simulate_match_plays_full_deal() -> None:
    rng = np.random.default_rng(2)
    a = HeuristicBot(Weights())
    b = HeuristicBot(Weights(captures=2.0))
    sa, sb = simulate_match(a, b, rng)
    assert np.isfinite(sa) and np.isfinite(sb)
    # settebello is always awarded to exactly one side (all 40 cards captured)
    assert sa + sb >= 1.0


def test_round_robin_fitness_zero_sum() -> None:
    rng = np.random.default_rng(3)
    pop = BotPopulation(GeneticConfig(population_size=4), rng)
    fitness = round_robin_fitness(pop, rng)
    assert len(fitness) == 4
    assert abs(sum(fitness)) < 1e-9  # symmetric scoring sums to zero
