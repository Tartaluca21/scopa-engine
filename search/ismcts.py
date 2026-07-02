"""Information Set Monte Carlo Tree Search for Scopa (Phase 6).

ISMCTS handles hidden information by re-determinizing the opponent's hand at the
start of every iteration (biased by the `BeliefSystem` posterior) and treating
tree nodes as *public* information sets rather than perfect states. Because the
world differs per iteration, a node's untried moves cannot be cached: they are
recomputed from the current determinized world on each visit.

Speed: the only clone happens inside `determinize` at the root of each iteration;
that single world is then mutated in place while walking down the tree and during
the rollout. The rollout is heuristic-guided (see `search.rollout`) so
Cards/Denari/Primiera/Settebello reach selection instead of being drowned in
random-playout noise. Rewards are `score_deal` point margins, backpropagated
from the perspective of the player who made each move (UCB1).
"""

from __future__ import annotations

import math
import time

import numpy as np

from cognitive.belief import BeliefSystem
from engine.core import ScopaEngine
from engine.heuristic import HeuristicBot, Weights
from search.alphabeta import Move, legal_moves
from search.determinize import determinize
from search.rollout import maybe_deal, simulate

DEFAULT_C = math.sqrt(2.0)
# Progressive-bias defaults: strength of the exposure-aware heuristic prior and
# the softmax temperature that shapes it. `prior_weight = 0` reproduces plain
# UCB1 with uniform-random expansion (the pre-prior behaviour).
#
# `2.0` was chosen empirically: at a 160-iteration budget it lifted agreement
# with a 2500-iteration oracle from 65.8% to 71.1% at ~0.4% iteration cost, and
# the bias decays as 1/(1+n) so it provably vanishes with visits (asymptotically
# identical to UCB1). Paired self-play at this budget was non-negative but not
# statistically significant (+0.16 +/- 0.25 pts/deal), so this is enabled for
# its convergence efficiency and proven safety, not a demonstrated win rate.
DEFAULT_PRIOR_WEIGHT = 2.0
DEFAULT_PRIOR_TEMP = 1.0
# Normalizes a signed `score_deal` margin into (0, 1) for UCB exploitation.
# Tuned to the ~[-5, 5] span of the four majority points plus a Scopa or two, so
# the strategic gradient (worth up to 4 points) dominates a single immediate
# Scopa instead of being flattened by the old, far larger divisor.
_REWARD_NORM = 5.0

MoveKey = tuple[int, tuple[int, ...]]


def _move_key(move: Move) -> MoveKey:
    """Hashable public identity of a (card, capture_set) move."""
    return move[0], tuple(sorted(move[1]))


def _move_from_key(key: MoveKey) -> Move:
    """Reconstruct an executable move; capture order is irrelevant to the engine."""
    return key[0], list(key[1])


class ISMCTSNode:
    """A public information-set node: children keyed by public move identity."""

    __slots__ = ("player_just_moved", "visits", "reward", "children", "priors")

    def __init__(self, player_just_moved: int) -> None:
        self.player_just_moved = player_just_moved
        self.visits = 0
        self.reward = 0.0
        self.children: dict[MoveKey, ISMCTSNode] = {}
        # Softmax heuristic prior over this node's moves, filled once on the
        # first expansion (None until then). See `_ensure_priors`.
        self.priors: dict[MoveKey, float] | None = None

    def ucb_select(self, keys: list[MoveKey], c: float, prior_weight: float) -> MoveKey:
        """Pick the child maximizing UCB1 plus a decaying heuristic prior.

        value = w/n + c*sqrt(ln N / n) + prior_weight * P(a) / (1 + n)

        The prior term is progressive bias (Chaslot et al.): it steers early,
        low-visit exploration toward heuristic-sensible moves and vanishes as a
        child accrues visits, so it never overrides converged UCB statistics.
        `prior_weight = 0` recovers plain UCB1.
        """
        log_n = math.log(self.visits)
        priors = self.priors
        best_key = keys[0]
        best_val = -math.inf
        for k in keys:
            child = self.children[k]
            val = child.reward / child.visits + c * math.sqrt(log_n / child.visits)
            if prior_weight and priors is not None:
                val += prior_weight * priors.get(k, 0.0) / (1 + child.visits)
            if val > best_val:
                best_val, best_key = val, k
        return best_key


def _reward(margin: float, player: int) -> float:
    """Map a player-0 `score_deal` margin to a (0, 1) reward for `player`."""
    signed = margin if player == 0 else -margin
    return 0.5 + 0.5 * max(-1.0, min(1.0, signed / _REWARD_NORM))


def _ensure_priors(
    node: ISMCTSNode, world: ScopaEngine, player: int, weights: Weights, temp: float
) -> None:
    """Fill `node.priors` once with a softmax over the heuristic move scores.

    The prior is the exposure-aware `HeuristicBot` valuation of `player`'s legal
    moves in the current (determinized) world, frozen on first expansion so the
    heuristic runs at most once per node rather than once per visit.
    """
    if node.priors is not None:
        return
    scored = HeuristicBot(weights).move_scores(world, player)
    if not scored:
        node.priors = {}
        return
    top = max(s for _, _, s in scored)
    exp = [(math.exp((s - top) / temp), _move_key((card, cap))) for card, cap, s in scored]
    total = sum(e for e, _ in exp)
    node.priors = {k: e / total for e, k in exp}


def _run_iteration(
    root: ISMCTSNode,
    world: ScopaEngine,
    rng: np.random.Generator,
    c: float,
    weights: Weights,
    prior_weight: float = 0.0,
    prior_temp: float = DEFAULT_PRIOR_TEMP,
) -> None:
    """One ISMCTS iteration: select, expand, simulate, backpropagate.

    With `prior_weight > 0` the heuristic prior both biases UCB selection and
    orders expansion (best untried move first); otherwise expansion is uniform
    random and selection is plain UCB1.
    """
    use_prior = prior_weight > 0.0
    node = root
    path = [root]
    # SELECTION + single EXPANSION, mutating the determinized world in place.
    while not world.is_game_over():
        maybe_deal(world, rng)
        if world.is_game_over():
            break
        player = world.current_player
        moves = legal_moves(world, player)
        keyed = [(_move_key(m), m) for m in moves]
        untried = [(k, m) for k, m in keyed if k not in node.children]
        if untried:
            if use_prior:
                _ensure_priors(node, world, player, weights, prior_temp)
                priors = node.priors or {}
                key, move = max(untried, key=lambda km: priors.get(km[0], 0.0))
            else:
                key, move = untried[int(rng.integers(len(untried)))]
            world.execute_move(move[0], move[1])
            child = ISMCTSNode(player_just_moved=player)
            node.children[key] = child
            node = child
            path.append(node)
            break
        key = node.ucb_select([k for k, _ in keyed], c, prior_weight)
        world.execute_move(*_move_from_key(key))
        node = node.children[key]
        path.append(node)
    # SIMULATION from the leaf's world, then BACKPROPAGATION along the path.
    margin = simulate(world, weights, rng)
    for visited in path:
        visited.visits += 1
        visited.reward += _reward(margin, visited.player_just_moved)


def ismcts_decide(
    state: ScopaEngine,
    belief: BeliefSystem | None = None,
    max_time_ms: int = 500,
    max_iter: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    c: float = DEFAULT_C,
    weights: Weights | None = None,
    prior_weight: float = DEFAULT_PRIOR_WEIGHT,
    prior_temp: float = DEFAULT_PRIOR_TEMP,
) -> Move:
    """Return the best move for the side to move via time/iteration-bounded ISMCTS.

    The original `state` is never mutated: each iteration works on a fresh
    determinized clone. `belief` biases the opponent-hand sampling; when omitted
    a uniform belief is built from the current state. `weights` tune the
    heuristic rollout policy and default to the unit genome. `prior_weight`
    scales the exposure-aware heuristic prior (progressive bias); `0` is plain
    UCB1 with random expansion.
    """
    player = state.current_player
    root_moves = legal_moves(state, player)
    if not root_moves:
        raise ValueError("no legal move available")
    if len(root_moves) == 1:
        return root_moves[0]

    if rng is None:
        rng = np.random.default_rng()
    if belief is None:
        belief = BeliefSystem(bot_player=player)
        belief.update_on_deal(state)
    if weights is None:
        weights = Weights()
    posterior = belief.get_opponent_hand_probabilities(as_array=True)

    root = ISMCTSNode(player_just_moved=1 - player)
    deadline = time.monotonic() + max_time_ms / 1000.0
    iters = 0
    while (max_iter is None or iters < max_iter) and time.monotonic() < deadline:
        world = determinize(state, player, rng, posterior)
        _run_iteration(root, world, rng, c, weights, prior_weight, prior_temp)
        iters += 1

    if not root.children:  # never got an iteration in: fall back to a legal move
        return root_moves[0]
    best_key = max(
        root.children,
        key=lambda k: (root.children[k].visits, root.children[k].reward),
    )
    return _move_from_key(best_key)
