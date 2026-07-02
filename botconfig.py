"""Single source of truth for the deployed default bot configuration.

Both front ends -- the terminal CLI (`play.py`) and the Pygame GUI (`gui.game`)
-- build their opponent from here, so they always play the *same* bot and log the
*same* `bot_name`. The deployed engine is PIMC at `n_worlds=12, max_depth=5`
(`~=10 ms` per move) with uniform leaf weights `DEFAULT_WEIGHTS`; the README's
Empirical Findings show this small config already plays near the classical
ceiling (extra depth/breadth does not help, and deeper search can hurt).

Kept dependency-light (engine + search + session only, no GUI imports) so the CLI
never pulls in pygame.
"""

from __future__ import annotations

from engine.heuristic import Weights
from search.alphabeta import SearchConfig
from search.pimc import PimcConfig
from session import pimc_bot_name

# Deployed PIMC budget: sampled worlds x alpha-beta depth cap.
DEFAULT_N_WORLDS = 12
DEFAULT_MAX_DEPTH = 5

# Uniform leaf weights (all 1.0). A paired A/B (N=450 deals over 3 seeds, seats
# swapped) showed uniform beats the old evolved genome by +0.298 ± 0.141 pts/deal
# (95% CI [+0.157, +0.438], significant): the genome over-prized scope and bled
# cards/primiera. Since PIMC's alpha-beta usually scores terminal states exactly,
# the leaf weights are near-washed out, and the neutral uniform shape wins.
DEFAULT_WEIGHTS = Weights()


def default_pimc_config() -> PimcConfig:
    """The deployed PIMC configuration used by both the CLI and the GUI."""
    return PimcConfig(
        n_worlds=DEFAULT_N_WORLDS,
        search=SearchConfig(max_depth=DEFAULT_MAX_DEPTH, weights=DEFAULT_WEIGHTS),
    )


def default_bot_name() -> str:
    """Stable log identifier of the deployed default bot (CLI and GUI share it)."""
    return pimc_bot_name(DEFAULT_N_WORLDS, DEFAULT_MAX_DEPTH)
