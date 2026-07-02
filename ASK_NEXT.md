# What to ask next — prioritized

Ordered by value × readiness. Each item has a ready-to-paste ask, why it matters,
and rough effort. Context lives in the memory files + README "Empirical Findings".

## 1. Measure the bot vs humans (the #1 open gap)
> "Add game logging to play.py and a script that reports my win-rate and average
> margin vs the bot over the deals I play."

- **Why:** every strength claim so far is bot-vs-bot. We have *zero* human data —
  we literally don't know how good it is against a person.
- **Unblocks:** judging whether any future upgrade actually matters.
- **Effort:** small. **Do this first.**

## 2. Endgame tablebase — Fase 7 (surest real gain)
> "Implement the exact endgame solver (retrograde analysis) and wire it into the
> PIMC/ISMCTS leaf evaluation once the deck is empty."

- **Why:** once the talon empties it's perfect information → solvable *exactly*.
  Provable strength gain, and it sidesteps PIMC strategy fusion entirely.
- **Effort:** medium. Highest confidence of a real improvement.

## 3. Safe exploitative opponent modeling — Fase 8, done right
> "Build adaptive opponent modeling with the safe-exploitation design: estimate
> the opponent's tendencies online, confidence-gate every deviation, and use a
> restricted Nash response so we never lose to a deceptive opponent. Validate on
> a panel of greedy/passive/deceptive/strong bots."

- **Why:** the earlier version was discarded for *assuming* a greedy opponent.
  The redesign estimates + hedges, so it exploits weak play without an exposed flank.
- **Needs:** ideally real human logs (from item 1) to model actual human mistakes.
- **Effort:** medium–large.

## 4. Neural value/policy network + self-play — Fase 9
> "Scope and prototype a learned value network trained by self-play, and slot its
> value head into ISMCTS leaves and its policy head into the progressive-bias prior."

- **Why:** the path to breaking the early/mid-game classical ceiling.
- **Needs:** an ML dependency (PyTorch, not yet in pyproject), a training entrypoint,
  and the paired-self-play gauntlet to prove it beats PIMC at N≥150.
- **Caveat:** the game is shallow — gains may be modest. **Effort:** large.

## 5. Deep CFR — Fase 10 (principled fix for strategy fusion)
> "Design a Deep CFR agent (advantage/strategy nets, external-sampling MCCFR) and
> compare it head-to-head against the PIMC baseline."

- **Why:** approximates the true imperfect-information optimum (Nash).
- **Effort:** large. Do after 4.

## Optional / lower priority
- **XAI (Fase 12):** "Make the GUI explain why the bot chose each move." Nice-to-have.
- **Ship ISMCTS to the GUI:** it only *ties* PIMC in testing, so skip unless you
  want the anytime engine for other reasons.

---
**Reminders for any strength claim:** Scopa self-play is high-variance — require
N ≥ 150 paired deals and fresh seeds (an N=60 result once vanished at N=200).
And the whole thing is capped by the deal's luck: no agent can beat every human
every game.
