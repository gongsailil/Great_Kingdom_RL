# Great Kingdom AlphaZero Research

*Unofficial independent research implementation.*

## Overview

This portfolio project independently implements Great Kingdom mechanics and
uses them to study self-play reinforcement learning, policy/value networks, and
PUCT Monte Carlo tree search. It grew from a PPO prototype into a
self-contained AlphaZero-style V5 system with a rules-exact engine, batched
search, replay stabilization, and direct human evaluation.

The final repository contains the frozen V5 implementation. Earlier systems,
diagnostics, and raw reports remain recoverable from annotated Git tags.

## Why this project

Human play revealed that the original PPO baseline had both strategic limits
and rule mismatches. After auditing the mechanics, the project moved to an
82-action Rules V2 engine and AlphaZero-style search. Automatic evaluation was
not treated as sufficient: V4 improved in internal arenas but was rejected
after human games exposed incoherent openings and repeated forced-loss states.
V5 therefore redesigned representation, throughput, calibration, and model
promotion as one integrated system.

## Final V5 architecture

- 12-plane current-player encoder: stones, neutral castle, PASS/inventory
  state, absolute color, territory, group liberties, and scoring margin.
- 128-channel, 6-block residual trunk with 1,820,674 parameters.
- 82-action policy head and calibrated binary value-logit head.
- Rules-derived liberty and score auxiliary heads (not reward shaping).
- Batched PUCT search with exact one-ply win and safe-defense filtering.
- Online D4 augmentation and duplicate-aware replay aggregation.
- Frozen BEST model, trained CANDIDATE, deterministic paired arena, and a 55%
  promotion gate.
- Held-out scalar value calibration and board-only territory-rich starts.

## Results

V5 completed 31 cycles, 20 promotions, and 11 candidate rejections. It
generated 7,936 self-play games and 219,336 samples, filled a 200,000-position
replay, and produced zero illegal-action violations. Batched inference measured
about 15.82× the V4 sequential throughput.

In the fixed paired regression arena, V5 scored 100-0 against V4 iteration 50,
100-0 against V3 iteration 50, and 100-0 against V2 iteration 375, winning all
50 games as each color in every matchup.

**These are deterministic internal regression matches and should not be
interpreted as a human-strength benchmark.** Human evaluation found V5 clearly
better than V4, while also confirming that human-level strategic play was not
fully solved. See [detailed results](docs/RESULTS.md).

## Known limitations

- D4 policy symmetry consistency did not improve as expected.
- Historical automatic opponents are not a substitute for strong human
  evaluation.
- When already losing by immediate double-PASS scoring, the pure win/loss
  objective can favor moves that reduce territory but prolong the game.
- The project makes no claim that Great Kingdom is solved or that V5 is
  superhuman.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Human versus human under Rules V2:

```bash
python play_human_v2.py
```

Human versus a local V5 checkpoint:

```bash
python play_vs_alphazero_v5.py \
  --checkpoint runs/alphazero_v5/strategy_20260908/latest.pt \
  --human-player blue \
  --mcts-simulations 256
```

Use `--human-player red` for an AI Blue opening. Press `P` to pass and `R` to
restart. Checkpoint and run artifacts are intentionally not distributed in Git.

Validate a local checkpoint without historical model dependencies:

```bash
python evaluate_alphazero_v5.py \
  --checkpoint runs/alphazero_v5/strategy_20260908/latest.pt
```

The frozen training entry point remains available for reproducibility. A new
run needs a compatible nine-plane replay once via `--territory-replay`; only
valid board states are extracted, and policy/outcome labels are ignored.

## Repository structure

```text
great_kingdom_v2.py          Rules V2 state and transitions
gk_env_v2.py                 Gymnasium adapter and legal-action mask
game_ui.py                   Shared Pygame board renderer
play_human_v2.py             Human-versus-human interface
play_vs_alphazero_v5.py      V5 human interface and local game logging
train_alphazero_v5.py        Frozen BEST/CANDIDATE training entry point
evaluate_alphazero_v5.py     V5 checkpoint and strategic validation
alphazero_v5/                Encoder, network, replay, search, and training
docs/                        Rules, history, and result summaries
reports/alphazero_v5_strategy/  Curated machine-readable V5 results
tests/                       Public Rules V2, V5, and UI regressions
```

## Research history

The PPO-to-V5 progression and recovery tags are summarized in
[docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md). Historical code should be
examined by checking out the corresponding annotated tag rather than mixing it
with V5.

## Intellectual property and disclaimer

Great Kingdom is a board game designed by Lee Sedol and published by Korea
Boardgames Co., Ltd. This repository is an unofficial, independent educational
and research implementation and is not affiliated with or endorsed by Lee
Sedol or Korea Boardgames.

The repository contains independently written software and does not distribute
official rulebook text, artwork, logos, product photography, or other
proprietary assets. Game names, trademarks, and third-party intellectual
property belong to their respective owners. See [NOTICE.md](NOTICE.md) for the
scope of the software license and third-party rights.

## License

Original source code is available under the [MIT License](LICENSE). That
license does not grant rights to the underlying game or third-party materials.
