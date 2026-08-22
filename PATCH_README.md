# Great Kingdom RL — Masked PPO Minimal E2E Patch

Purpose: test one hypothesis only:

> Can PPO learn to beat a random opponent without human game records when impossible actions are masked correctly?

## Rule semantics in this patch

- Occupied cell: impossible -> masked
- Territory-forbidden cell: impossible -> masked
- Suicide: selectable -> NOT masked -> immediate loss
- Capture: immediate win
- Normal move: reward 0
- Win/Loss: +1 / -1

## Files

- `great_kingdom.py`: separates move outcomes and implements suicide as an immediate loss
- `gk_env.py`: action mask + random Blue opponent + sparse reward
- `train_masked_minimal.py`: one finite 200k-step MaskablePPO run
- `evaluate_masked.py`: win rate / suicide rate / mask violations / first-move distribution
- `requirements-minimal.txt`: needed Python packages
- `test_rules_minimal.py`: core rule sanity tests

## Suggested local workflow

```bash
git checkout master
git pull
git checkout -b feat/masked-ppo-minimal-e2e-20260822

# Copy the patch files into the repository root.
python -m pip install -r requirements-minimal.txt

python test_rules_minimal.py
python train_masked_minimal.py

python evaluate_masked.py models/MaskablePPO_CNN/masked_ppo_200000.zip --episodes 500
```

## First experiment success criteria

Do not add self-play, MCTS, reward shaping, or hyperparameter sweeps yet.

Record:
1. win rate vs random opponent
2. agent suicide rate
3. mask violations (must be 0)
4. first-move distribution
5. mean game length

The first question is only whether learning occurs at all.
