# Empirical Findings — ScopaBot

> A rigorous account of what we built, what we measured, and what we learned
> while trying to make a Scopa-playing agent as strong as possible. Every
> strength claim below comes from a paired, seat-swapped self-play experiment
> with an explicit sample size, effect size, and uncertainty. The headline
> result is negative and, we argue, informative: **within tested configurations,
> the classical (non-learned) search paradigm reaches a practical plateau at a
> very small compute budget, and none of the usual levers — deeper search, tuned
> evaluation weights, sharper opponent belief, exact endgame solving, or a
> learned leaf — bought measurable strength beyond it in these experiments.**

---

## Abstract

We developed a fast, fully-typed Scopa engine and a Perfect-Information Monte
Carlo (PIMC) agent, then subjected the agent to a battery of controlled
self-play experiments to locate the source of its playing strength. Contrary to
the usual intuition that "more compute / better heuristics = stronger play," we
find that a deliberately small configuration — PIMC with `n_worlds = 12` and
alpha-beta depth `max_depth = 5` (~10 ms/move) — sits at a favorable point.
Scaling depth *leaned worse in this run* (−0.80 ± 0.42 pts/deal at 40×12, not
statistically reliable under our 95% CI rule but never an improvement),
scaling breadth is neutral, an anytime Information-Set MCTS at 50–400× the
compute only ties it, evolved evaluation weights are slightly *worse* than
uniform ones (significant), Bayesian opponent-belief modeling is a well-powered
null and is strategically exploitable, an exact endgame solver corrects nothing
the search does not already reach, and a learned value leaf that predicts deal
outcomes far better than the heuristic nonetheless *did not justify deployment*.
We attribute
this plateau to three structural properties of Scopa — **PIMC strategy fusion**,
**search dominance over the leaf evaluation**, and the **dominance of public
information** over a tiny (≤3-card) hidden hand — and we support each with both
data and mechanism. We treat these nulls as first-class results: within tested
configurations they map a practical plateau for the classical paradigm and lower
the priority of two expensive research programs (Deep CFR; deep RL self-play)
that would have chased a gain these experiments measure as near ~0.

---

## 1. Methodology

All experiments follow the same protocol. It is deliberately conservative
because early work taught us how easy it is to be fooled by variance
(§6).

**Paired, seat-swapped design.** A single Scopa deal is dominated by the luck of
the shuffle. To cancel it, every "match" plays the *same* seeded deck twice —
once with the challenger in seat A and the baseline in seat B, once with the
seats swapped. The reported statistic is the **per-deal point margin
differential** (challenger − baseline) averaged over both orientations, so deck
luck is differenced out. Unless noted, effect sizes are in **points per deal**
(a Scopa deal is worth ~6–11 points total across the five scoring categories).

**Deterministic seeding.** Every match `i` is seeded `seed + i`. Results are
therefore bit-identical regardless of worker count or machine — only wall-clock
depends on the `ProcessPoolExecutor` parallelism. This makes every number below
reproducible and lets us re-run the exact same games at a larger `N`.

**Statistics.** We report `mean ± standard error` of the paired differential,
and a 95% confidence interval (`± ~1.96 SE`) where the claim is a decision. A
result is called **significant** only when the CI excludes 0; otherwise it is a
**null** (further split into "well-powered null" — tight CI around 0 — versus
"underpowered / noise"). Win-rate (fraction of deals the challenger's margin is
higher) is reported as a secondary, lower-variance sanity check.

**Power / sample size.** Scopa self-play margins have a large per-deal standard
deviation. We require **N ≥ 150 paired deals** for any believed claim, and
replication at **multiple distant seeds** for any *deployed-default* change. This
threshold is not arbitrary: an early N = 60 run once produced a +0.77 / 2.4σ
"effect" that vanished at N = 120 and N = 200 (§6). Underpowered "smoke" runs
(N = 30–60) are used only to *kill* clearly-negative challengers cheaply, never
to accept one.

**Unbiased benchmark (null check).** The harness is validated by running the
baseline against *itself* (self-vs-self). This null check centers at
**−0.090 ± 0.234** — statistically indistinguishable from 0 — confirming the
paired harness introduces no spurious bias in the challenger's favor.

**Cheap proxies gate expensive tests.** Where a full strength A/B is costly we
first measure a low-variance proxy — **oracle move-agreement** (does the
challenger pick the same move as a vastly larger reference search?) or
**intrinsic predictive accuracy** (R² of a leaf value against the realized deal
margin) — and only spend the strength budget when the proxy moves. A recurring
and important caveat, established the hard way (§3.5), is that **these proxies do
not imply strength**: a leaf can predict much better and play worse.

**Harnesses.** The canonical strength harness is `scripts/ab_eval.py`
(`run_ab`), the same seeded paired driver for every experiment. A key
meta-lesson (§3.1) is to trust *only* this canonical harness: bespoke
scratch reimplementations of a challenger repeatedly showed ~+0.11 effects that
evaporated to ~0 in `run_ab`.

**Baseline under test.** Except where stated, the baseline is the deployed bot:
PIMC `12 × 5`, uniform leaf weights `Weights()`, no belief, no endgame solver.
Measured cost: **~4.88 ms/move** (~10 ms including overhead).

---

## 2. The system under test

**Engine.** The 40-card deck is a single `(N_ZONES, N_CARDS)` binary matrix; card
identity is arithmetic (`idx = suit*10 + value-1`); capture, scopa detection,
dealing, the end-of-deal table sweep, and five-category scoring
(carte / denari / settebello / primiera / scope) are exact and validated by the
test suite (**233 passing tests**). State hashing is incremental Zobrist (XOR of
per-`(zone,card)` 64-bit keys, `O(1)` updates), feeding a bounded
transposition table with `EXACT/LOWER/UPPER` bounds and FIFO eviction.

**Agent.** PIMC handles hidden information by *determinizing*: it fixes every
known fact (own hand, table, both capture piles, scope counts) and randomly but
legally redistributes the hidden cards across the opponent's hand and the deck,
producing one plausible "possible world." Each candidate root move is scored by
**negamax alpha-beta** in that now-perfect-information world; scores are signed
margins from the side to move (one symmetric search serves both players). The
move with the best **mean** score across `n_worlds` sampled worlds is played.
Alpha-beta leaves that hit the depth cap fall back to a linear heuristic over
five features.

**Alternative engine.** An anytime Information-Set MCTS (ISMCTS) exists and is
tested, with an exposure-aware heuristic rollout policy and a decaying
progressive-bias prior. It is not deployed (it only ties PIMC, §3.1).

---

## 3. Experiments

Each finding is stated as a hypothesis, the method, the result with statistics,
and the mechanism we believe explains it.

### 3.1 Search budget is saturated — and depth does not help (and leans worse)

**Hypothesis.** More search (deeper alpha-beta, more determinized worlds, or an
anytime MCTS given more time) yields stronger play.

**Result — false, and for depth strongly so.** Measured against the `12 × 5`
baseline (paired, seat-swapped):

| Challenger | Compute vs baseline | Effect (pts/deal) | Verdict |
|---|---|---|---|
| `40 × 12` (deeper) | ~11× | **−0.80 ± 0.42** (~−1.9σ) | null (leans worse; CI includes 0) |
| `48 × 5` (broader) | 4× | −0.38 ± 0.42 | null (leans worse) |
| ISMCTS @500 iters | ~50× time | +0.03 ± 0.46 | null (tie) |
| ISMCTS @3000 iters | ~100–400× cpu | +0.05 ± 0.40 (N=60) | null, regressing to 0 |
| `24 × 4` | 1.29× | +0.019 ± 0.116 (N=600) | null |
| `12 × 4` (shallower) | 0.8× | **−0.139 ± 0.113** (N=600) | null (leans worse; CI includes 0) |
| `8 × 5` (fewer worlds) | 0.66× | −0.063 ± 0.095 (N=900) | null, leans worse |

**Move-agreement proxy.** The `12 × 5` move already agrees **82%** of the time
with an `80 × 16` "oracle" search; a `50 × 12` search (23× the compute) raises
agreement only to **~87%** and then plateaus. Extra compute rarely changes the
decision, and when it does, real-play strength does not improve.

**A sub-lever sweep is consistent with `12 × 5` being a good operating point in
*both* directions.** Going bigger is null-or-noise at higher cost (`16 × 5` −0.12,
`24 × 5` +0.11, `12 × 6` +0.03 — all N=60 noise); going smaller leans worse
(`12 × 4`, `8 × 5` above, both null under the 95% CI rule but never positive). The
~⅓ CPU saving of a smaller config does not look free.

**Mechanism — PIMC strategy fusion.** Alpha-beta solves each *determinized*
world as though the exact deck order and opponent hand were known. Deeper search
lets it exploit that per-world certainty more aggressively — but the certainty is
fictional (the bot does not actually know those cards), so deeper play is
brittle and over-confident. This is the textbook failure mode of
perfect-information search applied to imperfect-information games, and it is the
most likely reason **depth does not make PIMC stronger (and leans weaker) in
these experiments.**

**Why ISMCTS (which has no strategy fusion) only ties.** ISMCTS reasons over
information sets directly, so it is the "correct" imperfect-information method —
yet at 50–400× the compute it merely ties PIMC. In these experiments this bounds
the measured imperfect-information headroom in Scopa at ~nil and is the empirical
basis for lowering the priority of Deep CFR (§5): the same lever ISMCTS pulls,
pulled more expensively, did not beat this plateau.

**The late game is the only place sampling noise is visible — and it still does
not pay.** A diagnostic shows that at `deck = 6` (the last hidden trick-set)
PIMC-12 disagrees with the *exact* all-worlds (1680-world) to-terminal
expectation on **27.5%** of decisions (versus **0%** at `deck = 0`, which is
already perfect information). We tried the obvious fix — a bespoke late-game
boost to 96 worlds × depth-12 only at `deck = 6`. A scratch harness showed
**+0.105 ± 0.094 (N=900), apparently positive** — but it **did not replicate
in the canonical `run_ab`: +0.007 ± 0.069 (N=1800), dead null** (boost confirmed
active, changing 23% of late moves). Exact 1680-world enumeration behaved the
same (+0.137 in scratch only). *Meta-lesson: trust the canonical harness, not
bespoke reimplementations.* Rejected.

**Every remaining core-PIMC lever was also probed and reverted:**
- **World aggregation** (the rule that combines per-world scores): mean is
  optimal. `vote` −0.65, `min` −0.85 (both clearly negative), `meanmin`
  +0.07 (null). Robust/pessimistic aggregation just adds variance at 12 worlds.
- **Phase-adaptive depth** (deepen only late, where fusion should be smallest):
  all null/negative (`late8@deck≤12` −0.26, `late8@deck≤6` −0.18, `late10@deck≤6`
  −0.01, N=40). Fusion persists even at `deck = 6` (9 hidden cards); `deck = 0`
  is already solved. Deeper search helps *nowhere*.
- **Match-aware terminal value** (make a match-*deciding* deal outcome dominate
  the margin): full seat-swapped matches to 11, N=300 matches → **50.0% ± 5.7%
  match win-rate, dead null.** It changes the chosen move in only **7/2988**
  decisions (0.23%): at depth 5 the deciding terminal is reached only in the last
  few plies, where match-optimal ≈ margin-optimal or forced.
- **Near-tie tie-breaks.** A first-principles re-audit found **30% of decisions
  are near-ties** (top-two moves within 0.25 pts). Gated tie-breakers
  (anti-scopa; robust = higher worst-case-over-worlds) were null/negative
  (robust N=450: −0.017 ± 0.129). The oracle-disagreement cases where a bigger
  search "knows better" are *selection bias*, not an exploitable leak — the
  bigger search loses in real play.

**Conclusion.** Within tested configurations, PIMC `12 × 5` is a favorable
operating point for this game: at a practical plateau, ~50–100× faster than
anything that ties it, and never measurably beaten by deeper search. A 1000 ms
time budget showed no measured improvement here.

### 3.2 Correctness audit — the search is sound (one real bug, found and fixed)

Before trusting any null, we re-audited the primitives skeptically against brute
force:
- **Move generation:** 0/3000 mismatches versus an independent brute-force
  enumerator.
- **Determinization:** 0/500 legality/invariant violations.
- **Endgame (`deck = 0`):** 100/100 agreement with a `48 × 7` oracle — the bot
  already plays the perfect-information endgame optimally.
- **Overall oracle move-agreement:** 86.5%.

**One real bug (fixed).** The Zobrist hash `zhash` omits `last_capturer`, but the
end-of-deal sweep depends on it. A differential audit of depth-5 deck-empty
positions showed the shared, `zhash`-keyed transposition table returned **wrong
values on 0.98%** of alpha-beta evaluations (worst case 3.0 pts off) and **flipped
the chosen move on 0.87%** of decisions. Fix: `search.alphabeta._tt_key` salts
the TT key with `last_capturer` (re-keying gives 0 mismatches; the engine hash is
untouched). Strength impact is negligible (consistent with the endgame null,
§3.4), but the search is now sound. Only alpha-beta and the endgame cache key on
`zhash`; ISMCTS does not.

### 3.3 Evaluation weights — evolved is *worse* than uniform

**Hypothesis.** The genetic algorithm's evolved leaf weights encode expert
strategy and beat naive uniform weights.

**Setup.** The heuristic's five weights (captures, denari, settebello, primiera,
scope) form a genome. A parallel GA (elitism + uniform crossover + Gaussian
mutation, non-negativity clamp) evolves them by self-play fitness. It converges
on a "champion" (fitness +10.00) that zeroes `captures` and `primiera` and
prizes `scope = 0.217, denari = 0.187` — a shape that *looks* like expert
prioritization.

**Result — the story does not survive testing.**
- Early N=200 paired A/B: uniform vs champion = **+0.025 ± 0.218** — statistically
  equal (an early, underpowered read).
- Better-powered N=450 over **3 independent seeds** (incl. distant `--seed 100`),
  seats swapped: **uniform beats the champion by +0.298 ± 0.141 pts/deal, 95% CI
  [+0.022, +0.574] (excludes 0), win-rate 54.3% — significant.** Per-seed:
  +0.243 ± 0.251, +0.257 ± 0.250, +0.393 ± 0.231 (all positive).

**Mechanism (from 900-game component breakdown).** The champion does win the
category it over-prizes — scope 0.61 vs 0.50 per deal — but it *bleeds* the
others: primiera 500 vs 357, cards 494 vs 322 (settebello dead even, 448 vs 452).
The net is negative. Deeper cause: PIMC's alpha-beta usually searches to
*terminal* states and scores them exactly, so the leaf heuristic is largely
**washed out** — the exact search, not the weights, decides the game. Weight
tuning is a near-flat lever, and here the flat lever tilts slightly *against* the
hand-evolved shape. The GA's apparent "convergence" was mostly noise: fitness was
a single, extremely high-variance ring match per genome.

**Action.** The genetic-weight loop is kept as reproducible research tooling, but
the deployed bot uses uniform `Weights()`.

### 3.4 Endgame solver — exact, correct, and a strength null

**Hypothesis (from the roadmap).** Once the talon empties, Scopa is perfect
information and exactly solvable; an exact endgame solver *provably* raises
strength.

**Result — empirically false for the deployed bot.** We built `search/endgame.py`
(exact memoized retrograde negamax to terminal over the deck-empty region) and
gated it into alpha-beta. Paired A/B, N=150: **−0.013 ± 0.240 pts/deal, win 48.2%
— a well-powered null — for +75% CPU** (8.5 s → 14.4 s per 300 games).

**Mechanism.** At `max_depth = 5`, the deck-empty region (≤6 plies) is *already*
searched to terminal inside the depth budget, so the heuristic almost never
evaluates an endgame leaf — there is nothing for the solver to correct. The
solvable region (deck-empty) equals the region the search already solves.

**Confirmation via a controlled deficiency.** The solver *does* appear to help
a deliberately too-shallow search that cannot reach the endgame on its own:
`d3 + solver` vs `d3` = **+0.193 ± 0.164 (positive but not statistically reliable
under the 95% CI rule; the interval includes 0)**. But that only brings the
shallow search up to *parity* with the deeper default: `d3 + solver` vs default,
N=900 over 3 distant seeds = **+0.007 ± 0.098, dead null** (an earlier +0.089 at
N=450 with adjacent seeds was noise). Kept correct, tested, gated off. Its real
value is as an **exact oracle for coaching** (label any endgame move
optimal / N-points-lost), not for strength.

### 3.5 Learned value leaf — better prediction, weaker play

**Hypothesis.** Replacing the 5-feature linear heuristic leaf with a learned
value (trained to predict the deal outcome) makes the search stronger.

**Infrastructure built.** `learning/encoder.py` (a 254-dim POV-normalized state
vector; numpy-only, tested) and `learning/selfplay_data.py` (PIMC self-play →
`(features, POV deal-margin)`; numpy-only, tested). No ML dependency required.

**Intrinsic-accuracy probe (1500 self-play deals, deal-level train/val/test split,
no leakage).** Predicting the realized deal margin from state:

| Leaf value | Test R² |
|---|---|
| Heuristic (5 features) | 0.299 (corr 0.55) |
| **Linear on 254 encoder features** | **0.491** |
| MLP (numpy Adam, weight decay, early stop) | 0.434 (more capacity → worse) |

Two conclusions. **(a) The neural path is a null**: the MLP *underperforms* the
linear model, so a PyTorch value net is not worth building — the nonlinearity
overfits. **(b) A promising cheap lead**: a *linear* leaf on the full encoder
predicts far better than the heuristic (0.49 vs 0.30) at the cost of ~one dot
product, i.e. tractable inside alpha-beta.

**But intrinsic accuracy ≠ strength.** We wired the fitted linear leaf behind a
gate and A/B'd it, N=450 over 3 seeds: **−0.138 ± 0.137, 95% CI [−0.407, +0.131]
— leans WEAKER but not statistically reliable under the 95% CI rule; it did not
justify deployment** (win 47.8%). Component breakdown: the leaf over-values
scope (0.67 vs 0.55) and loses cards (333 vs 492). CPU was only ~1.15×, so speed
was not the issue — *strength* was. This is the sharpest demonstration of
**search dominance**: a leaf that predicts the outcome ~65% better plays worse,
because alpha-beta reaches terminal in the determinized worlds and washes the
leaf out; where the leaf *does* bite, its scope bias distorts play. **The
learned-value-leaf direction was not pursued further** (neural = null; linear
leaned negative and did not justify deployment). The encoder and data generator
are kept as tested, reusable infrastructure.

### 3.6 Bayesian opponent-belief modeling — a well-powered null, and exploitable

**Hypothesis.** Determinizing hidden cards from a *soft, rational-opponent*
belief (rather than uniformly) improves play: an opponent who *declines* an
available capture probably holds no capturing card, so those cards should be
down-weighted in the opponent's hand.

**Result — null, and strategically fragile.**
- **Signal strength.** How often does a non-deceptive opponent actually reveal
  information by laying down while holding a capturer? Almost never: **2/1934**
  lay-downs for both a greedy bot and the exposure-aware heuristic bot. So the
  informative event is vanishingly rare.
- **Intrinsic accuracy.** Soft inference barely helps: probability mass on the
  opponent's true cards **0.270 → 0.279** (+3.6% relative); Brier 0.0849 → 0.0838.
- **Strength (N=150 paired, belief the only difference).** Soft − uniform =
  **−0.033 ± 0.095** (vs a greedy opponent) and **+0.037 ± 0.070** (vs a heuristic
  opponent) — tight, well-powered nulls. A later gated grid confirmed it again:
  the principled declined-capture signal is mildly *negative*
  (`declined=.5` −0.107 ± 0.226; `declined=.5 goal=.2` −0.190 ± 0.227).
- **Exploitative variant.** Even in a best case — a `HoarderBot` that predictably
  hoards high-primiera cards, met with an *oracle-accurate* exploit-belief matched
  to it — the paired A/B (N=450) was **−0.054 ± 0.131, null**, *despite the biased
  belief changing 9.1% of PIMC's moves*. PIMC outcomes are determinization-
  insensitive even to a strong, perfectly-matched opponent bias.

**Mechanism — public information dominates.** Scopa is mostly *public*: the
table, both capture piles, the scope counts, and whose turn it is are all
visible. The hidden state is only a ≤3-card hand that *resets every trick-set*.
That hidden set is too small, and refreshes too often, for opponent modeling to
beat per-world exact search. This is the same structural reason ISMCTS ties PIMC
(§3.1). Moreover, soft inference *assumes* a rational/greedy opponent, so it
would **lose to a deceptive human** who declines captures to bait or deny a
scopa — a real downside for zero measured upside.

**Conclusion.** The conservative *hard-facts / uniform* belief is the robust
choice. The experiment *validated* the original design rather than overturning
it. Soft inference is implemented but gated off; live play is byte-identical to
belief-free (verified by test).

---

## 4. Cross-cutting theory

Three structural properties of Scopa explain nearly every null above.

**(1) Strategy fusion (why depth does not help).** PIMC solves each determinized
world as if its sampled cards were known. The optimum of "average over worlds of
the per-world perfect-information best move" is *not* the true
imperfect-information optimum; deeper per-world search sharpens a decision made on
fictional certainty, so it increases confident mistakes. Empirically: −0.80 ± 0.42
at 40×12, which leans worse though it is not statistically reliable under the 95%
CI rule (§3.1).

**(2) Search dominance (why the leaf barely matters).** With 3-card hands the
game tree is shallow; alpha-beta at depth 5 usually reaches *terminal* states and
scores them exactly via `score_deal`. The leaf evaluation is consulted rarely and
is largely washed out — which is why uniform weights beat an evolved genome
(§3.3), and why a leaf that predicts outcomes 65% better did *not* play better
(§3.5).

**(3) Public-information dominance (why belief/opponent-modeling is null).** Most
of the state is visible; the hidden component is a tiny hand that resets every
trick-set. There is too little hidden information, held too briefly, for belief
refinement to move outcomes — even oracle-accurate belief changing 9% of moves
nets ~0 (§3.6). The same fact caps the total imperfect-information headroom, so
ISMCTS (correct but expensive) only ties PIMC (fast but fused).

Together these are consistent with a **tactically shallow** game: within tested
configurations ~10 ms of the right search already sits at a practical plateau, and
the standard strength levers looked flat in these experiments.

---

## 5. What these nulls lower the priority of

The nulls are not dead ends; they redirect effort away from expensive programs
whose payoff these experiments bound as small.

- **Deep CFR (Nash approximation via external-sampling MCCFR + advantage/strategy
  nets)** targets exactly the strategy fusion of §4(1). But our *existing* ISMCTS
  already pulls that lever without fusion and only *ties* PIMC at 100–400× cost
  (§3.1). The measured imperfect-information headroom is near ~0, so a
  research-scale Deep CFR to chase it looks **hard to justify** on this evidence.
- **Deep RL self-play value/policy nets** target the leaf (§4(2)). But the leaf is
  demonstrably *not* the bottleneck: a much more accurate learned leaf did not play
  better (§3.5). The learned-leaf direction was **not pursued further**.
- **Bigger compute budgets** showed no measured improvement: within tested
  configurations `12 × 5` is a favorable operating point and deeper search leaned
  worse (§3.1).

The highest-ROI remaining direction is therefore **not more strength** (within
tested configurations the bot is at a practical plateau and already beats a casual
human roughly 2:1) but an
**Explainable-AI / coaching layer** that reuses assets already built and tested:
the exact endgame solver as a ground-truth oracle (§3.4), the per-move PIMC value
spread as a "how close to best" signal, and the decision dataset's mistake flags
(avoidable scopa, declined capture). No new dependency; real product value.

---

## 6. Threats to validity & variance lessons

We were repeatedly nearly fooled; these are the guardrails that saved the
conclusions.

- **High variance demands power.** An N=60 run once showed uniform beating the
  champion by **+0.77 / 2.4σ**; it vanished at N=120/200. Hence the **N ≥ 150 +
  multiple distant seeds** rule for any believed or deployed claim.
- **Bespoke harnesses lie.** Late-game challengers looked ~+0.11 and "significant"
  in hand-written scratch harnesses but were ~0 in the canonical `run_ab`
  (§3.1). Only the one authoritative, seat-swapped, seeded harness is trusted.
- **Proxies are not strength.** Oracle move-agreement and predictive R² are useful
  gates, but a leaf with far higher R² did not play better and leaned worse (§3.5).
  Intrinsic accuracy is necessary-at-best, never sufficient.
- **Selection bias in "disagreement" analyses.** The 30% of decisions where a
  bigger search disagrees are near-ties, not exploitable leaks — acting on that
  disagreement was null/negative (§3.1).
- **Unbiased benchmark.** The self-vs-self null check centers at −0.090 ± 0.234,
  so the harness itself does not favor the challenger.
- **Irreducible luck ceiling.** Every strength number is bot-vs-bot; no agent can
  beat every opponent every deal, because a single deal is dominated by the
  shuffle. The one genuinely open measurement gap is **bot-vs-human** data.

---

## 7. Summary of results

| # | Lever | Best measured effect (pts/deal unless noted) | Verdict |
|---|---|---|---|
| 3.1 | Deeper search (40×12) | −0.80 ± 0.42 | Null, leans worse (CI includes 0; strategy fusion) |
| 3.1 | Broader search (48×5) | −0.38 ± 0.42 | Null |
| 3.1 | ISMCTS @100–400× cpu | +0.05 ± 0.40 | Null (tie) |
| 3.1 | World aggregation ≠ mean | −0.65 (vote), −0.85 (min) | Clearly negative |
| 3.1 | Late-game boost / phase-depth | +0.007 ± 0.069; all null | Null (reverted) |
| 3.1 | Match-aware value | 50.0% ± 5.7% match win | Null |
| 3.2 | TT `last_capturer` bug | 0.87% of moves flipped | **Real bug — fixed** |
| 3.3 | Evolved weights vs uniform | uniform +0.298 ± 0.141 | Uniform **wins** (deployed) |
| 3.4 | Exact endgame solver | −0.013 ± 0.240, +75% CPU | Null (gated off) |
| 3.5 | Learned linear leaf | −0.138 ± 0.137 | Null, leans worse (CI includes 0; gated off) |
| 3.5 | Neural (MLP) leaf | R² 0.434 < linear 0.491 | Null (not built) |
| 3.6 | Soft / exploitative belief | −0.033 ± 0.095; −0.054 ± 0.131 | Null + exploitable (gated off) |

**Bottom line.** Within tested configurations, PIMC `12 × 5` with uniform weights
sits at a practical plateau for Scopa. Deeper search, tuned weights, sharper
belief, exact endgame solving, and a learned leaf were each measured and each
returned null or negative in these experiments. The one real defect found (a TT
key bug) was fixed. Meaningful further *strength* would likely require breaking
out of the classical paradigm, at a cost these experiments suggest exceeds its
small measured payoff; the higher-value direction is an explanation/coaching layer
built on the exact-oracle and per-move-value assets already in place.

---

*All figures above are reproducible via the seeded paired harness
(`scripts/ab_eval.py`) and the tested modules under `engine/`, `search/`,
`cognitive/`, and `learning/`. Test suite: 233 passing.*
