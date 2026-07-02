# ScopaBot Elite — TODO (12 Fasi)

- [x] **Fase 0** — Infrastruttura, MLOps e Architettura dei Dati *(98 test verdi, uv + ruff + pytest)*
- [x] **Fase 1** — Core Engine Ultra-Veloce (Bitboards / NumPy Vectorization) *(engine/core.py, cards.py)*
- [x] **Fase 2** — Gestione dello Stato, Memory & Transposition Tables *(transposition.py, zobrist.py)*
- [x] **Fase 3** — Bot Euristico Avanzato ed Evoluzione Genetica *(heuristic.py, features.py, genetic.py, train.py)*
- [x] **Fase 4** — Perfect Information Monte Carlo (PIMC) e Rollout Rapidi *(pimc.py, determinize.py, alphabeta.py — engine schierato: 12×5)*
- [x] **Fase 5** — Belief System Bayesiano e Stima dello Stato Nascosto *(cognitive/belief.py implementato e testato — posterior "hard-facts"; inferenza soft valutata → rimandata, vedi Findings §3)*
- [x] **Fase 6** — Information Set Monte Carlo Tree Search (ISMCTS) *(core testato: search/ismcts.py + rollout.py, test_ismcts.py; prior progressive-bias)*
- [x] **Fase 7** — Risolutore di Fine Partita (Endgame Solver) *(search/endgame.py: minimax retrogrado esatto, memoizzato su (zhash,last_capturer), testato. A/B **NULLO** a 12×5 — la ricerca raggiunge già gli endgame deck-empty; tenuto gated. Valore reale: oracolo esatto per XAI, non forza)*
- [ ] **Fase 8** — Opponent Modeling Dinamico ed Exploitative Play *(inferenza soft implementata gated e ri-testata 2026-07-02 → NULLO/leggermente negativo e sfruttabile; prior uniforme più robusto, Findings §3)*
- [ ] **Fase 9** — Deep Reinforcement Learning e Pipeline di Self-Play *(vedi Next Steps — richiede dipendenza ML)*
- [ ] **Fase 10** — Deep CFR *(DE-RISKED 2026-07-02 → non conviene: ISMCTS senza strategy-fusion, anche a 6× compute, NON batte PIMC (+0.05±0.40); tetto imperfect-info ~nullo. Vedi Next Steps)*
- [x] **Fase 11** — Backend, API ad Alta Velocità e Interfaccia Utente Grafica *(gui/ Pygame: app, render, animation, scoreboard, async_bot)*
- [ ] **Fase 12** — Modulo XAI (Explainable AI) e Divine Coach Virtuale *(asset pronti: decision_dataset + flag avoidable_scopa, endgame solver come oracolo esatto — vedi Next Steps: candidato a più alto ROI)*

> **Il resoconto scientifico completo di tutti gli esperimenti** (ipotesi, metodo,
> effect size con intervalli di confidenza, teoria meccanicistica) è in
> **[`EMPIRICAL_FINDINGS.md`](EMPIRICAL_FINDINGS.md)**. I §-riferimenti qui sotto
> puntano a quel documento.
>
> **Stato attuale (agg. 2026-07-02).** Fasi classiche 0–7 e 11 complete e testate.
> 232 test verdi, ruff pulito.
> Ogni leva classica di *forza* è ora chiusa come NULLA con prove robuste:
> - budget di ricerca — 12×5 è il punto ottimale in entrambe le direzioni (più
>   grande non aiuta, più piccolo peggiora; sweep N≤900);
> - pesi foglia — uniform `DEFAULT_WEIGHTS` è il default (batteva il vecchio genoma
>   evoluto +0.30, poi confermato);
> - belief soft — implementato gated, ri-testato NULLO/negativo e sfruttabile;
> - endgame solver — esatto e testato, ma A/B nullo alla profondità schierata.
>
> Fix di correttezza: la chiave TT ora include `last_capturer` (bug audit: ~1% dei
> valori di ricerca erano errati). 232 test verdi. **Conclusione: il regime
> classico ha raggiunto il tetto. Le uniche vie avanti sono (a) il regime
> *appreso* (Fasi 9/10, richiede PyTorch) per il mid-game a informazione
> imperfetta, o (b) il layer *Experience/XAI* (Fase 12), che non richiede ML e
> riusa asset già costruiti.**

---

## Next Steps — Technical Requirements

### Fase 7 — Endgame Solver — DONE, and it is a strength NULL (important correction)
Built as `search/endgame.py` (exact memoized retrograde minimax over the
deck-empty region) and integrated behind `SearchConfig.use_endgame_solver`.
The roadmap premise — "exact endgame play *provably* raises strength" — turned out
to be **empirically false for the deployed bot**: at `max_depth=5` the alpha-beta
*already* reaches and solves the ≤6-ply deck-empty region within its budget, so
the solver has almost nothing to correct. Measured A/B (paired, seat-swapped):
- solver on vs off @12×5: `-0.013 ± 0.240` (null), for **+75% CPU**;
- the solver *does* significantly help a too-shallow search (d3+solver vs d3:
  `+0.193 ± 0.164`), but that only reaches parity with the default (d3+solver vs
  default @N=900: `+0.007 ± 0.098`).
So there is **no bigger perfect-information region** to exploit for strength (the
solvable region == deck-empty == already searched). Kept correct, tested, gated
off. Its real value is as an **exact oracle for Fase 12** (label any endgame move
optimal/sub-optimal for coaching). Do not re-open for strength.

### Fasi 9 & 10 — Neural Value Networks, Self-Play & Deep CFR
To break the ceiling in the **early/mid** game (where hidden information is real),
replace the linear heuristic leaf with a **learned** value/policy trained on
self-play; for the imperfect-information optimum, add a Deep-CFR regret learner.
- **Self-play pipeline:** a generator emitting `(public state, belief features,
  action, deal outcome)` tuples at scale — the existing deterministic-seeded,
  `ProcessPoolExecutor`-parallel tournament infra is the natural backbone.
- **State encoding:** the `(N_ZONES, N_CARDS)` matrix plus the belief vector and
  scope/turn scalars is already a clean tensor; `CARD_VALUES` and primiera points
  give useful extra channels.
- **Model:** a small value head (predict deal margin) drops straight into ISMCTS
  as a leaf evaluator; a policy head drops into the **progressive-bias prior**
  already implemented in `search/ismcts.py`.
- **Deep CFR (Fase 10):** advantage/strategy networks over information sets,
  trained by external-sampling MCCFR toward a Nash strategy — the principled fix
  for the strategy fusion that limits PIMC (Findings §1).
- **Required before starting:** an ML dependency (e.g. PyTorch — not currently in
  `pyproject.toml`); a training entrypoint (GPU optional); and a frozen evaluation
  gauntlet (the paired self-play harness from the Findings work) to prove any
  learned agent beats the PIMC baseline at `N ≥ 150` *before* it ships.
- **Honest expectation:** the value head only helps where search is truncated
  (mid-game); it does NOT fix strategy fusion (that's Deep CFR, Fase 10). Medium
  effort, uncertain-but-plausible payoff. This is the next real *strength* lever.

#### Fase 9 — execution plan (SELECTED 2026-07-02; staged, gated, go/no-go)
Build incrementally; each stage is testable and commits to nothing downstream.
The default bot never changes until a learned agent clears the gauntlet.
1. **Encoder** (`learning/encoder.py`, numpy-only, DONE-first): POV-normalized
   fixed-length feature vector from a (determinized) `ScopaEngine` + player. No
   new deps. ← *this brick first.*
2. **Self-play dataset generator**: reuse the seeded `ProcessPoolExecutor`
   tournament infra to emit `(features, final deal margin)` from PIMC self-play.
   Numpy-only (labels = `score_deal` margin). Still no ML dep.
3. **De-risk probe — DONE 2026-07-02 (intrinsic-accuracy version).** Predicting the
   deal outcome on held-out self-play states (deal-level split, 1500 deals):
   heuristic leaf test R²=0.299; **linear on encoder features R²=0.491**; MLP best
   R²=0.434 (more capacity → worse). **Result: the NONLINEAR/neural path is a NULL**
   (MLP underperforms linear) → **do NOT build PyTorch.** But a NEW cheap lead:
   a *linear* leaf on the full 254-dim encoder features predicts far better than
   the 5-feature heuristic and costs ~one dot product, so it is tractable inside
   alpha-beta. Caveat: intrinsic accuracy ≠ search strength (leaf weights already
   proved neutral — search dominance). Needs a real strength A/B to settle.
4. **~~PyTorch value+policy~~ — SKIP** (stage 3 failed the gate for neural nets).
5. **Linear-leaf sub-experiment — DONE 2026-07-02, NEGATIVE. PATH CLOSED.** Wired
   the fitted linear encoder-value as a gated alpha-beta leaf and A/B'd vs default
   (N=450, 3 seeds): **-0.138 ± 0.137, CI [-0.275, -0.001] — significantly
   WEAKER** (win 47.8%; it over-chases scope and loses cards 333/492). Confirms
   intrinsic accuracy ≠ search strength: the leaf is NOT the bottleneck (search
   dominance / strategy fusion is). The linear-leaf hook was removed (strictly
   negative); the generic `learning/encoder.py` + `learning/selfplay_data.py`
   were kept as reusable, tested infra for Fase 10/12.

**Fase 9 verdict: the learned-value-LEAF direction is exhausted (neural = null,
linear = negative).**

**Fase 10 (Deep CFR) de-risked 2026-07-02 → NOT worth building.** Deep CFR pulls
the imperfect-information-correctness lever; our *existing* ISMCTS pulls the same
lever without strategy fusion. Probe: heavily-budgeted ISMCTS vs PIMC 12×5 —
500it (recorded) +0.03±0.46; **3000it (6× compute) +0.05±0.40 (N=60), null and
regressing toward zero** — at ~100–400× PIMC's CPU. So the imperfect-information
ceiling in Scopa is essentially nil (the game is tactically shallow), and a
research-scale Deep CFR (PyTorch + MCCFR + nets + training) to chase a ~0 gain is
not justified. **Recommend pivoting to Fase 12 (Coach)** for real ROI. PIMC 12×5
is effectively at the game's algorithmic ceiling; all strength levers are closed.

### Fase 12 — XAI / Coach (highest ROI, no ML dependency) — RECOMMENDED NEAR-TERM
The bot is already near the algorithmic ceiling and beats a human ~2:1, so extra
*strength* has low product value. The high-leverage upgrade is the **Experience
layer**: explain the bot and coach the human. Most assets already exist:
- `decision_dataset.py` already emits per-move rows with `avoidable_scopa` /
  `left_table_scopable` flags → the mistake taxonomy is built.
- `search/endgame.py` is an **exact oracle**: in the deck-empty phase, label each
  human (or bot) move as optimal / how-many-points-lost — ground-truth coaching.
- The PIMC per-move value spread (already computed in `pimc_decide`) gives a
  cheap "how close was your move to the bot's best" signal for the mid-game.
- **Deliverables:** (a) a post-deal review that flags avoidable scopas, declined
  captures, and endgame mistakes with the exact points lost; (b) optional live
  "why" annotations in the GUI. No new dependencies; reuses tested modules.
