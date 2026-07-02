# Future Work

This is the forward-looking companion to
**[`EMPIRICAL_FINDINGS.md`](../EMPIRICAL_FINDINGS.md)**, which is the canonical,
evidence-backed record of what was tried and measured. Everything below is
framed by that evidence: several "obvious" next steps were built and measured,
and turned out to be strength nulls — they are listed here as *de-risked* so
nobody re-opens them expecting a gain.

The deployed bot is PIMC `n_worlds=12, max_depth=5` with uniform leaf weights.
Paired, seat-swapped, seeded A/B testing (`scripts/ab_eval.py`) shows this small
configuration is already at the classical ceiling for Scopa.

## Strength levers — closed (measured null or negative)

These were implemented and tested; do not re-open them expecting more strength.
See the linked sections of the findings for effect sizes and confidence intervals.

- **Bigger / deeper search** — `12 × 5` is Pareto-optimal; deeper search is
  *actively worse* (strategy fusion). *(Findings §3.1)*
- **Tuned leaf weights** — the evolved genome lost to uniform weights, which are
  now the default. *(Findings §3.3)*
- **Exact endgame solver** — built, correct, and tested, but a strength null at
  the deployed depth (the search already reaches the deck-empty region). Kept,
  gated off. *(Findings §3.4)*
- **Learned value leaf** (linear or neural, trained on self-play) — a more
  accurate leaf played *weaker*; the leaf is not the bottleneck. Direction
  closed. *(Findings §3.5)*
- **Sharper / exploitative opponent belief** — a well-powered null, and
  exploitable; the conservative hard-facts belief is retained. *(Findings §3.6)*
- **Deep CFR** (external-sampling MCCFR + advantage/strategy nets) — de-risked
  without building it: the existing ISMCTS already targets the same
  imperfect-information ceiling and only *ties* PIMC at 100–400× the cost, so the
  ceiling is ~0. Not justified. *(Findings §5)*

## Open directions (higher value than raw strength)

1. **Explanation / coaching layer (recommended next).** The bot is near the
   algorithmic ceiling, so extra strength has low product value; explaining and
   coaching does not. Most of the needed pieces already exist and are tested:
   - the exact endgame solver (`search/endgame.py`) as a ground-truth oracle —
     label any deck-empty move optimal vs. how many points it cost;
   - the per-move PIMC value spread (already computed in `pimc_decide`) as a
     cheap "how close was your move to the bot's best" signal for the mid-game;
   - the decision dataset's mistake flags (`avoidable_scopa`,
     `left_table_scopable`, declined captures) as a ready mistake taxonomy.

   Deliverable: a post-deal review that flags avoidable scopas, declined
   captures, and endgame mistakes with the exact points lost, plus optional live
   "why" annotations in the GUI. No new dependencies; reuses tested modules.

2. **More bot-vs-human data.** Every strength number is bot-vs-bot. The one
   genuinely open *measurement* gap is play against real people; the logging and
   analysis pipeline for this already exists (`play.py --record-moves`,
   `stats.py`, `match_stats.py`, `scripts/build_decision_dataset.py`).

## Reusable infrastructure already in place

Kept as tested, dependency-light building blocks for the directions above:

- `learning/encoder.py` — POV-normalized fixed-length feature vector from an
  engine state (numpy only).
- `learning/selfplay_data.py` — seeded, parallel self-play dataset generator.
- `scripts/ab_eval.py` — the authoritative paired, seat-swapped, seeded A/B
  harness. Any strength claim must clear this at `N ≥ 150` over multiple seeds
  before it ships (Scopa self-play is high-variance — see Findings §6).
