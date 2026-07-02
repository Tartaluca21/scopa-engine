"""Tests for the paired self-play A/B harness."""

from __future__ import annotations

from engine.features import DealBreakdown, Weights
from search.ab_eval import ABResult, GameOutcome, _outcome, _relabel, run_ab
from search.alphabeta import SearchConfig
from search.pimc import PimcConfig


def _tiny(n_worlds: int = 2, max_depth: int = 2, weights: Weights | None = None) -> PimcConfig:
    return PimcConfig(
        n_worlds=n_worlds, search=SearchConfig(max_depth=max_depth, weights=weights or Weights())
    )


def test_relabel_by_seat() -> None:
    assert _relabel("p0", base_is_p0=True) == "baseline"
    assert _relabel("p1", base_is_p0=True) == "challenger"
    assert _relabel("p0", base_is_p0=False) == "challenger"
    assert _relabel("p1", base_is_p0=False) == "baseline"
    assert _relabel("none", base_is_p0=True) == "none"


def test_outcome_swaps_scores_when_baseline_is_p1() -> None:
    bd = DealBreakdown(
        p0_score=7.0,
        p1_score=4.0,
        p0_scope=2,
        p1_scope=1,
        settebello_winner="p0",
        denari_winner="p1",
        primiera_winner="none",
        cards_winner="p0",
    )
    out = _outcome(bd, base_is_p0=False)  # baseline sat in p1
    assert out.base_score == 4.0 and out.chal_score == 7.0
    assert out.base_scope == 1 and out.chal_scope == 2
    assert out.settebello == "challenger" and out.denari == "baseline"
    assert out.primiera == "none" and out.cards == "challenger"


def test_run_ab_is_reproducible_and_paired() -> None:
    base, chal = _tiny(), _tiny()
    r1 = run_ab(base, chal, n_pairs=3, seed=7, workers=1)
    r2 = run_ab(base, chal, n_pairs=3, seed=7, workers=1)
    assert r1.n_pairs == 3 and r1.n_games == 6  # two swapped games per pair
    assert r1.summary() == r2.summary()  # deterministic given the seed


def test_summary_and_markdown_fields() -> None:
    base, chal = _tiny(), _tiny()
    result = run_ab(base, chal, n_pairs=2, seed=1, workers=1)
    s = result.summary()
    assert 0.0 <= s["win_rate"] <= 1.0
    assert s["ci95"] >= 0.0
    md = result.to_markdown()
    assert "# Paired A/B self-play" in md
    assert "Margin" in md and "pts/deal" in md
    for component in ("settebello", "denari", "primiera", "cards"):
        assert component in md


def test_component_tally_totals_match_game_count() -> None:
    games = [
        GameOutcome(6.0, 4.0, 1, 0, "baseline", "challenger", "none", "baseline"),
        GameOutcome(3.0, 8.0, 0, 2, "challenger", "challenger", "baseline", "none"),
    ]
    result = ABResult(1, "b", "c", games)
    tally = result._component_tally()
    for component in ("settebello", "denari", "primiera", "cards"):
        assert sum(tally[component].values()) == result.n_games
