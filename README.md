# Great Kingdom RL

## Rules V2 + AlphaZero/MCTS

This repository implements the audited 9×9 Great Kingdom rules and explores a
shared policy-value network guided by PUCT MCTS. PPO V1 is retained only as a
historical baseline; it does not implement the current rules.

Current research status:

- Exact Rules V2 engine with player-dependent 82-action legality masks.
- Minimal and resumable AlphaZero self-play/training pipelines.
- A completed 375-iteration V2 run and deterministic milestone evaluation.
- MCTS search-budget, policy/value-guidance, and PUCT diagnostics.
- A completed V3 pilot adding current/opponent territory planes.
- Primary unresolved issue: one-ply defensive threat recognition and
  search-learning stability. More search improves coverage, but learned-value
  guidance can still starve immediate terminal wins.

## Rules V2

- 9×9 board, Blue first, 40 castles per player, one central neutral castle.
- Actions `0..80` place a castle; action `81` is PASS.
- Opponent territory is blocked; own territory remains playable.
- Pure suicide is illegal, while a simultaneous capture has priority.
- Capturing any opposing group wins immediately.
- Two consecutive passes trigger territory scoring.
- Blue wins when `Blue territory >= Red territory + 2`; otherwise Red wins.
- No draw and no ko rule.

See [RULES_AUDIT.md](RULES_AUDIT.md) for evidence and the V1/V2 comparison.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimal.txt
```

Install the appropriate CUDA-enabled PyTorch build separately when GPU use is
required. Model checkpoints and run state are local artifacts excluded from
Git.

## Play

Human versus human under Rules V2:

```bash
python play_human_v2.py
```

Human versus an AlphaZero V2 checkpoint:

```bash
python play_vs_alphazero_v2.py \
  --checkpoint runs/alphazero_v2/main_20260830/latest.pt \
  --human-player blue \
  --mcts-simulations 64
```

Use `--human-player red` for an AI Blue opening. Press `P` to pass and `R` to
restart. V1 PPO checkpoints are incompatible with the 82-action Rules V2 UI.

## Active entrypoints

Training runners:

```bash
# Resumable V2 runner; omit --hours for unlimited iteration-boundary execution.
python train_alphazero_v2.py --run-dir runs/alphazero_v2/<run_id>

# Bounded territory-representation pilot runner.
python train_alphazero_v3.py \
  --run-dir runs/alphazero_v3/<run_id> \
  --max-iterations 50
```

Evaluation and diagnostics:

```bash
python evaluate_alphazero_v2.py
python run_alphazero_v2_mcts_ablation.py
python run_alphazero_search_guidance_audit.py
python run_alphazero_puct_ablation.py
```

These scripts use Rules V2 transitions and legal masks. Evaluation runs with
root noise disabled and deterministic maximum-visit action selection.

## Active structure

```text
great_kingdom_v2.py              Rules V2 source of truth
gk_env_v2.py                     Gymnasium wrapper and legal action mask
game_ui.py                       Shared Rules V2 Pygame renderer
play_human_v2.py                 Human versus human
play_vs_alphazero_v2.py          Human versus AlphaZero V2
alphazero_v2/                    Encoder, network, MCTS, self-play, training
alphazero_v3/                    Territory encoder and diagnostic extensions
reports/                         Curated AlphaZero experiment reports
docs/                            Experiment chronology and project history
legacy/202601_ui/                Historical UI reference only
```

## Curated reports

- [Minimal AlphaZero V2 E2E](reports/alphazero_v2_minimal_e2e_20260830/summary.txt)
- [V2 milestone evaluation](reports/alphazero_v2_evaluation_20260901/summary.txt)
- [MCTS simulation-budget ablation](reports/alphazero_v2_mcts_ablation_20260901/summary.txt)
- [V3 territory pilot](reports/alphazero_v3_territory_pilot_20260901/summary.txt)
- [Policy/value guidance audit](reports/alphazero_search_guidance_audit_20260902/summary.txt)
- [PUCT exploration ablation](reports/alphazero_puct_ablation_20260902/summary.txt)

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the chronological index.

## Milestone tags

| Tag | Commit | Meaning |
|---|---|---|
| `ppo-v1-final` | `31074b2` | Final historical PPO V1 baseline |
| `rules-v2` | `ad7d194` | Audited Rules V2 engine |
| `alphazero-v2-minimal` | `e4a6203` | Minimal AlphaZero E2E |
| `alphazero-v2-training` | `cb40bd3` | Resumable V2 training runner |
| `alphazero-v2-evaluation` | `9361ecf` | V2 milestone evaluation |
| `alphazero-v2-mcts-ablation` | `bea5d60` | MCTS budget diagnosis |
| `alphazero-v3-territory-pilot` | `8eb9c19` | Nine-plane territory pilot |
| `alphazero-search-guidance-audit` | `711d1e5` | Policy/value guidance separation |
| `alphazero-puct-ablation` | `01db29b` | PUCT exploration diagnosis |

## Historical PPO V1

PPO V1 used an incompatible 81-action engine and rules that differed from the
physical game, including selectable suicide and no PASS action. Its current
role is historical context only. See [docs/history/ppo_v1.md](docs/history/ppo_v1.md),
or check out the `ppo-v1-final` tag to recover its exact code, tests, reports,
and checkpoint documentation.
