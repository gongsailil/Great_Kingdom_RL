# AlphaZero V5 human acceptance

Run five games as each color only after the automatic V5 evaluation is complete:

```bash
python play_vs_alphazero_v5.py \
  --checkpoint runs/alphazero_v5/strategy_20260908/latest.pt \
  --human-player blue \
  --mcts-simulations 256

python play_vs_alphazero_v5.py \
  --checkpoint runs/alphazero_v5/strategy_20260908/latest.pt \
  --human-player red \
  --mcts-simulations 256
```

Logs are written under `human_games/v5_strategy/` and remain local. Review all
ten games for opening coherence, shape/heaviness, unexplained edge moves,
territory creation or disruption, entry into forced-loss positions, and
rational PASS/endgame behavior. Human moves and outcomes are evaluation
evidence only and must not be inserted into replay or used as training labels.
