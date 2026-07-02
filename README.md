# 🃏 Scopa Bot — A High-Performance Engine & Imperfect-Information AI

> A vectorized Scopa game engine and a family of AI agents that play the
> 40-card Italian classic under **hidden information** — combining Zobrist
> hashing, transposition tables, Perfect-Information Monte Carlo, alpha-beta
> search, and a parallel genetic algorithm for reproducible genetic-weight
> experiments.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-234%20passing-brightgreen">
  <img alt="Typing" src="https://img.shields.io/badge/typing-strict-success">
  <img alt="Style" src="https://img.shields.io/badge/style-ruff-black">
</p>

---

## 🎯 Project Overview

This project is a **highly optimized Scopa engine** paired with **AI agents that
play well despite never seeing the full game state**. In Scopa each player hides
their hand and the deck order is unknown, so a strong bot has to *reason about
what it cannot see*. The codebase delivers that in two halves:

- **The Engine** — a fast, fully-typed, vectorized representation of the rules:
  the deck, the table, both hands, both capture piles, captures, scope, and
  end-of-deal scoring — every state transition validated by a deep test suite.
- **Decision stack** — a layered pipeline (PIMC → determinization →
  alpha-beta → tuned heuristic) wrapped behind a single `Player` interface, with
  weights evolved by a multiprocessing genetic tournament (research tooling; the
  deployed bot uses uniform weights).

The whole thing is built under a strict engineering budget: **≤ 250 lines per
file, full static typing, English-only source, and 234 passing tests.**

> 📄 For the full, rigorous account of every experiment — hypotheses, methods,
> effect sizes with confidence intervals, and the mechanistic theory behind each
> result — see **[`EMPIRICAL_FINDINGS.md`](EMPIRICAL_FINDINGS.md)**. The section
> below is a summary.

---

## 🛠️ The Development Journey, Step by Step

### 1. The Core Engine & Rules — *vectorized from the ground up*

We started by representing the **40-card Italian deck** as a single
`(N_ZONES, N_CARDS)` binary `uint8` matrix where `1` means *"this card is in
this zone."* A card's identity is pure arithmetic:

```text
idx = suit * 10 + (value - 1)     # suit ∈ [0,3], value ∈ [1,10]
```

Six disjoint **zones** model the whole board — deck (`MAZZO`), table (`TAVOLO`),
the two hands, and the two capture piles. Because the state is one NumPy matrix,
operations like "all cards on the table" or "redistribute the deck" are
vectorized slices rather than Python loops.

On top of this we implemented the **standard Scopa mechanics**:

- **Capture logic** — a played card takes a matching single, or *any subset of
  table cards summing to its value* (combinatorial capture via
  `subsets_summing`).
- **Scopa detection** — clearing the table awards a *scopa* (the prized sweep).
- **Strict state transitions** — `execute_move`, `deal_round`, and
  `end_of_deal_sweep` enforce legality and keep every invariant intact.
- **End-of-deal scoring** — carte, denari, settebello, primiera, and scope, all
  computed straight off the capture piles.

The engine tracks the side to move, the last capturer, and per-player scopa
counts — everything needed to drive search and scoring.

### 2. Search Memory — *Zobrist Hashing + Transposition Tables*

Search re-visits the same board state through different move orders constantly.
To recognize those collisions **instantly**, we integrated **Zobrist Hashing**:
a fixed table of random 64-bit keys, one per `(zone, card)` pair, plus keys for
the side to move and each `(player, scopa-count)`. The state's hash is simply the
**XOR of all its active keys** — so every state change is an `O(1)` two-XOR
update, not a full rescan. Key uniqueness is asserted at import time to rule out
trivial collisions, and a fixed seed makes the tables reproducible across runs.

That hash feeds a **Transposition Table** — a bounded `hash → evaluation` cache.
Each entry carries a node type (`EXACT` / `LOWER` / `UPPER`) so alpha-beta can
reuse not just exact values but *bounds*, and the table evicts FIFO when full to
keep memory flat during deep rollouts. The net effect: previously-seen states
return their value immediately instead of being re-explored.

### 3. The Decision Stack — *PIMC + Alpha-Beta*

The hard part of Scopa is **hidden information**: the opponent's hand and the
deck order are unknown. The bot tackles this with **Perfect-Information Monte
Carlo (PIMC)**:

1. **Determinize** — from the deciding player's viewpoint, keep every *known*
   fact fixed (own hand, table, both piles, scope counts) and randomly
   redistribute the *hidden* cards across the opponent's hand and the deck,
   preserving counts so the deal stays legal. This produces one plausible
   **"possible world."**
2. **Search each world** — for every candidate root move, run **Negamax
   Alpha-Beta** over that now-perfect-information world. Values are signed score
   *margins* from the side to move, so a single symmetric search serves both
   players, and the shared Transposition Table prunes repeated look-aheads.
3. **Average & choose** — sum each move's score across all sampled worlds and
   play the move with the best mean. A move that's strong *across many possible
   worlds* is robust to what we can't see.

Leaf nodes that hit the depth cap fall back to a tuned linear **heuristic**
(captures, denari, settebello, primiera, scope) — and that heuristic's weights
are exactly what we evolve next.

**Deployed configuration.** Both front ends — the terminal CLI (`play.py`) and
the graphical GUI (`gui_run.py`) — build the *same* bot from a single source of
truth (`botconfig.py`): PIMC at **`n_worlds=12, max_depth=5`** (`≈10 ms` per move)
with uniform leaf weights `DEFAULT_WEIGHTS`, so both play identically and log the
same `bot_name`. An anytime
**Information-Set MCTS** (`search/ismcts.py`, with an exposure-aware heuristic
rollout and a progressive-bias prior) exists as a tested alternative engine.
These configurations are deliberately small — the *Empirical Findings* below show
that, within tested configurations, this tactically-shallow game reaches a
practical plateau where spending more compute shows no measured improvement (and
deeper search can lean worse).

### 4. Parallel Evolution — *a Genetic Algorithm tournament*

The heuristic's five weights define a **genome**. We evolve a population of
genomes with classic GA operators:

- **Elitism** — the fittest genomes survive untouched.
- **Uniform crossover** — children mix weights from two elite parents.
- **Gaussian mutation** — small random perturbations explore the space.
- **Safety clamping** — `np.clip(child, 0.0, None)` keeps every weight
  non-negative; a negative weight would *invert* the heuristic (e.g. punishing
  the bot for capturing the settebello) and corrupt the agent.

Fitness comes from **self-play**: each genome plays a full deal against the next
in a ring, scored by the actual margin it wins by. Because PIMC agents are far
heavier than a one-ply lookup, the ring matches are farmed out to a
`ProcessPoolExecutor` (multiprocessing). Each match is **deterministically
seeded** (`seed + i`), so results are identical regardless of worker count — only
the wall-clock changes. The AI advances through **Generations (Gen)**, getting
stronger every round.

---

## 🚀 AI Self-Training — and a Reality Check

Run end-to-end, the genetic loop converges on a set of heuristic weights **on its
own**, with no hand-coded strategy. Here is an exact training log — followed by
the honest test of whether that "learned strategy" actually matters:

```text
gen  0  best_fit=+7.00  Weights(captures=0.0, denari=0.050642522252031474, settebello=0.0, primiera=0.4387104338729524, scope=0.016469612867756167)
gen  1  best_fit=+7.00  Weights(captures=0.0, denari=0.3894916358693721, settebello=0.6352048054351396, primiera=0.0, scope=0.2719165773060053)
gen  2  best_fit=+6.00  Weights(captures=0.0, denari=0.09643189639142781, settebello=0.0, primiera=0.5862741445413199, scope=0.038854404963500275)
gen  3  best_fit=+7.00  Weights(captures=0.0, denari=0.18397872889709962, settebello=0.5137961199804502, primiera=0.0, scope=0.12319625628510153)
gen  4  best_fit=+9.00  Weights(captures=0.0, denari=0.038817747397097424, settebello=0.07856698228963531, primiera=0.7163392889777312, scope=0.23793566267845226)
gen  5  best_fit=+10.00 Weights(captures=0.0, denari=0.18714478182574792, settebello=0.08802187949432819, primiera=0.0, scope=0.21699416571513003)
gen  6  best_fit=+8.00  Weights(captures=0.0960287318144177, denari=0.18517056217731406, settebello=0.5382347723243077, primiera=0.0, scope=0.0)
gen  7  best_fit=+8.00  Weights(captures=0.29631574055136484, denari=0.32329284795239616, settebello=0.095842097707321, primiera=0.02453378907461315, scope=0.0772376173049315)

best genome (fit=+10.00): Weights(captures=0.0, denari=0.18714478182574792, settebello=0.08802187949432819, primiera=0.0, scope=0.21699416571513003)
```

### 📈 What the AI learned — and the honest follow-up

The champion genome (fitness **+10.00**) settled on weighting **scope (0.217)**
and **denari (0.187)** highest while zeroing `captures` and `primiera` — a shape
that *looks* like expert prioritisation (scope and denari are indeed high-leverage
points).

But we tested that story rigorously instead of trusting it, and it **did not hold
up**. An early paired run (N = 200, seats swapped) already rated this "champion"
as **statistically indistinguishable from trivial uniform weights**, and a
better-powered follow-up (N = 450 over 3 seeds) went further: **uniform weights
actually *beat* the genome by `+0.298 ± 0.141` pts/deal** (see Finding 2 below).
The reason is structural: PIMC's alpha-beta usually searches to *terminal* states
and scores them exactly, so the leaf weights are largely washed out — the **exact
search, not the heuristic, decides the game**. The GA's fitness was also a single,
extremely high-variance ring match per genome, so the "convergence" was mostly
noise (an intermediate A/B once showed a +0.77 effect that vanished at larger
sample sizes).

The lesson is itself a result, documented below: on this game, weight-tuning is a
near-flat lever, and the evolved shape was a slight net negative — so the
**deployed bot now uses uniform `Weights()`**. The training loop still works and
is reproducible — it just isn't where the strength comes from.

---

## 🔬 Empirical Findings — A Practical Plateau for Classical Methods

Once the engine and agents were in place, the bot was put through a battery of
**rigorous, paired self-play experiments** — seats swapped to cancel deck luck,
`N ≥ 150–200` deals for statistical power, and low-variance proxy metrics used to
gate the expensive tests. The headline: within tested configurations, with
**classical (non-learned) methods, the deployed PIMC bot reaches a practical
plateau.** Three levers were probed; all three returned null or negative in these
experiments.

**1 · Search budget is saturated — and more depth does not help (and leans
worse).**
The deployed `12 × 5` config decides in `≈10 ms`. Scaling it up does not help:
- Its move already agrees `~82%` with a `23×`-larger "oracle" search, rising only
  to `~87%` and plateauing — extra compute rarely changes the decision.
- Deeper search leaned worse in this run: `40 × 12` vs `12 × 5` = **−0.80 ± 0.42**
  pts/deal (not statistically reliable under the stated 95% CI rule, but never an
  improvement). The likely mechanism is **PIMC strategy fusion** — alpha-beta
  solves each *determinized* world as if the exact deck and opponent hand were
  known, so more depth exploits information the bot does not really have, yielding
  brittle, over-confident play.
- More *breadth* (`48 × 5`) didn't help either (`−0.38 ± 0.42`), and an anytime
  **ISMCTS** at a `50×`-larger time budget only **tied** PIMC@10 ms (`+0.03 ± 0.46`).

Scopa is tactically shallow (3-card hands), so within tested configurations
`~10 ms` already sits at a practical plateau. A larger `1000 ms` budget showed no
measured improvement in these experiments — it is effectively a non-constraint.

**2 · Leaf weights are a near-flat lever — and the evolved genome was slightly
*worse* than uniform.** Because PIMC's alpha-beta usually scores terminal states
exactly, the leaf heuristic is largely washed out. An early N = 200 run rated the
evolved champion as statistically equal to uniform weights (`+0.025 ± 0.218`), but
a later, better-powered paired A/B (`scripts/ab_eval.py`, N = 450 over 3 seeds,
seats swapped) found **uniform weights beat that genome by `+0.298 ± 0.141`
pts/deal** (95% CI `[+0.022, +0.574]` excludes 0, win-rate 54.3%). The genome
over-prized scope — it wins the scope battle (`0.61` vs `0.50`/deal) but bleeds cards
(`494` vs `322` of 900 games) and primiera (`500` vs `357`), a net loss. **The
deployed bot now uses uniform `Weights()`.** *Caution recorded for future work:*
an N = 60 run once showed a `+0.77 / 2.4σ` effect that **failed to replicate** at
N = 120/200 — Scopa self-play is high-variance enough that `N ≥ 150`, fresh seeds,
and multi-seed replication are required before believing any strength claim.

**3 · Bayesian belief modeling: uniform priors beat soft inference.**
The bot determinizes hidden cards from a belief over the opponent's hand. We
tested upgrading the conservative *hard-facts* belief with **soft rational-opponent
inference** ("an opponent who declines an available capture probably holds no
capturing card"):
- Intrinsic accuracy improved only marginally (probability mass on the opponent's
  true cards `0.270 → 0.279`).
- Strength was a well-powered **null**: soft vs uniform belief = `−0.033 ± 0.095`
  (vs a greedy opponent) and `+0.037 ± 0.070` (vs a heuristic opponent).
- Worse, soft inference *assumes* a greedy opponent and would **hurt against a
  deceptive human** who declines captures to bait or deny a scopa.

So the **uniform / hard-facts prior is the more robust choice** — it concedes no
measurable strength and cannot be exploited by strategic non-captures. The
experiment *validated* the original conservative design rather than overturning it.

**4 · Exact endgame solver: correct but a strength null.** An exact retrograde
solver for the deck-empty (perfect-information) region was built and gated in.
Paired A/B: **−0.013 ± 0.240 pts/deal for +75% CPU** — because at depth 5 the
search *already* reaches and solves those endgames, there is nothing to correct.

**5 · Learned value leaf: better prediction, weaker play.** A linear leaf on a
254-dim encoder predicts the deal outcome far better than the heuristic
(test `R² 0.49` vs `0.30`; a neural MLP was *worse*, `0.43`) — yet in real play
it **leaned weaker (−0.138 ± 0.137)**, and the effect did not justify deployment
(the 95% CI does not exclude 0). Intrinsic accuracy ≠ strength:
alpha-beta washes the leaf out (search dominance).

**Bottom line.** Deeper search, tuned weights, sharper belief, exact endgame
solving, and a learned leaf were each measured and each returned null or negative
in these experiments — within tested configurations, the classical paradigm is at
a practical plateau. See **[`EMPIRICAL_FINDINGS.md`](EMPIRICAL_FINDINGS.md)** for
the full analysis and why the two obvious "next regimes" (Deep CFR, deep-RL
self-play) showed little expected payoff over this plateau in these experiments.
The high-ROI direction is now an explanation/coaching layer, not more strength —
see **[`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md)**.

---

## 🧱 Code Quality & Architecture

Engineering discipline is a first-class feature here, not an afterthought:

- **≤ 250 lines per file** — a hard budget that forces a highly modular design.
  The engine, hashing, search, and evolution layers each split cleanly into small
  single-responsibility modules.
- **Full static typing** — every function is annotated end to end; no implicit
  `Any`. NumPy arrays carry typed dtypes (`npt.NDArray[...]`).
- **English-only source** — all code, comments, docstrings, and tests are in
  English (the Italian card names live only in the human-facing CLI strings).
- **A broad test suite — 234 passing unit & integration tests** covering
  card math, engine transitions, Zobrist incrementality, alpha-beta correctness,
  determinization legality, PIMC and ISMCTS decisions, the belief system, the
  heuristic, and the training loop.

```text
engine/
  cards.py          card indices, suits, zones, subset combinatorics
  core.py           the vectorized ScopaEngine + O(1) Zobrist updates
  zobrist.py        random 64-bit key tables, collision-checked at import
  transposition.py  bounded TT with EXACT/LOWER/UPPER bounds, FIFO eviction
  features.py       linear evaluation, capture features, exposure, deal scoring
  masks.py          action-mask & one-card-per-zone consistency helpers
  heuristic.py      exposure-aware one-ply bot + self-play match driver
  genetic.py        population, elitism + crossover + mutation
cognitive/
  belief.py         Bayesian posterior over the opponent's hidden hand
search/
  determinize.py    sample a perfect-information world from a player's view
  alphabeta.py      negamax alpha-beta with TT bounds
  pimc.py           Perfect-Information Monte Carlo coordinator (deployed engine)
  ismcts.py         Information-Set MCTS with heuristic rollout + progressive-bias prior
  rollout.py        exposure-aware heuristic playout policy for ISMCTS
  agent.py          SearchAgent (PIMC) + RandomBot baseline
  tournament.py     parallel ring fitness over a ProcessPoolExecutor
gui/                Pygame human-vs-bot interface (async off-thread search)
  menu.py           setup screen: single deal / match / move recording
  session.py        GUI session orchestration (logging, matches, advancement)
  game.py           single-deal controller (human + async bot)
session.py          shared non-UI deal/match logic used by the CLI and the GUI
botconfig.py        single source of truth for the deployed default bot (CLI == GUI)
gamelog.py          append-only JSONL deal & match logs (shared schema)
capture.py          per-decision snapshot for the move-history dataset
decision_dataset.py flatten captured deals into a human-decision dataset
train.py            evolve PIMC weights and print the best genome
play.py             interactive human-vs-bot CLI
scripts/
  benchmark.py            default-bot benchmark + decision-parity digest
  ab_eval.py              paired, seat-swapped A/B between two configurations
  build_decision_dataset.py  build the human-decision dataset from move logs
tests/              unit & integration tests
```

---

## ▶️ How to Run

### 📦 Install

Requires Python ≥ 3.12. With [uv](https://docs.astral.sh/uv/) (recommended — a
lockfile is committed):

```bash
uv sync            # create the venv and install pinned dependencies
uv run pytest      # run anything inside the environment with `uv run …`
```

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # runtime deps only
pip install -e '.[dev]'     # + pytest / ruff / mypy for development
```

Installing also puts a set of console scripts on your `PATH`, so you can run the
tools by name instead of `python <file>.py`:

| Command             | Equivalent                     |
| ------------------- | ------------------------------ |
| `scopa-play`        | `python play.py`               |
| `scopa-gui`         | `python gui_run.py`            |
| `scopa-train`       | `python train.py`              |
| `scopa-stats`       | `python stats.py`              |
| `scopa-match-stats` | `python match_stats.py`        |
| `scopa-benchmark`   | `python scripts/benchmark.py`  |
| `scopa-ab-eval`     | `python scripts/ab_eval.py`    |

The commands below assume the environment is active (prefix them with `uv run`
if you use uv). Human-vs-bot games are logged as JSONL under `logs/` (gitignored,
so your play history stays local); the stats tools below read them back.

```bash
# 1. Run the full test suite
pytest

# 2. Train the AI — run the genetic tournament and print the best genome
scopa-train                    # (dev alt: python train.py)

# 3. Play against the PIMC bot in your terminal
scopa-play                     # a single deal
scopa-play --match-to 11       # a match: deals until someone reaches 11 (or 21)
scopa-play --record-moves      # also capture each decision into the deal log
```

Every `scopa-*` command accepts `--help`. The `python <file>.py` forms shown as
"dev alt" still work when running from a source checkout.

### 🖥️ Play in the GUI

```bash
scopa-gui                      # (dev alt: python gui_run.py)
```

The window opens on a **setup screen**. Click to choose what to play, then
**Start**:

- **Single deal** — one logged deal.
- **Match to 11** / **Match to 21** — deals accumulate until one side leads at
  the target; the GUI auto-advances between deals.
- **Record moves: ON/OFF** — toggle per-decision capture into the deal log.

In play, click a hand card to play it; for an ambiguous capture, click the table
cards to take. When a deal ends: **Space** plays the next (or a new game),
**M** returns to the setup screen, **Esc** quits. GUI and CLI write the *same*
JSONL logs, so everything below works identically for both.

### 📊 Stats & the decision dataset

```bash
# Your deal record vs the bot (single deals + match deals)
scopa-stats                    # (dev alt: python stats.py)

# Your match record vs the bot, broken down by bot configuration
scopa-match-stats              # (dev alt: python match_stats.py)

# Build a human-decision dataset from games played with move recording on
python scripts/build_decision_dataset.py            # write dataset + report
python scripts/build_decision_dataset.py --report   # report only, no write
python scripts/build_decision_dataset.py --sample 5 # also pretty-print 5 rows
```

### ⏱️ Benchmark the bot

```bash
scopa-benchmark            # ms/move, deals/sec, wall + CPU time
scopa-benchmark --parity   # deterministic decision digest
scopa-benchmark --profile  # cProfile hot-path breakdown
# (dev alt: python scripts/benchmark.py ...)
```

Tooling:

```bash
ruff check .     # lint
ruff format .    # format
```

---

## 🧠 In One Sentence

A fast, fully-typed Scopa engine that remembers states with Zobrist hashing,
reasons through hidden information with PIMC + alpha-beta, evolves heuristic
weights through a parallel genetic tournament, and — crucially — **rigorously
measures where its strength actually comes from**, all in a clean, modular,
thoroughly-tested codebase.

---

## 📄 License

Released under the [MIT License](LICENSE) — see the `LICENSE` file for the full
text.

Card art in `assets/cards/` is **Public Domain** (Wikimedia Commons, Naples
deck), not covered by the MIT code license — see
[`assets/cards/CREDITS.md`](assets/cards/CREDITS.md) for provenance.

Everything runs locally: there is no backend, network, telemetry, or analytics.
Human-vs-bot games are logged only to `logs/` on your own machine and are
gitignored by default, so your play history is never committed or uploaded.
