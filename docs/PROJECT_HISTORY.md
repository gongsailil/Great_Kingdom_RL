# Project History

This repository began as a PPO self-play prototype and ended as a frozen,
self-contained AlphaZero-style research system. Historical source, tests, and
reports remain recoverable from annotated Git tags rather than being carried
in the public working tree.

| Stage | Milestone | What changed |
|---|---|---|
| PPO V1 | `ppo-v1-final` | Established the first action-masked self-play baseline through Red14/Blue13. Human play exposed weak territory and defense, while the engine also differed from the physical rules. |
| Rules V2 | `rules-v2` | Rebuilt the game authority around placement/PASS, inventories, asymmetric territory scoring, pure-suicide rejection, and capture-priority semantics. |
| AlphaZero V2 | `alphazero-v2-minimal`, `alphazero-v2-training`, `alphazero-v2-evaluation` | Connected policy/value inference, PUCT, self-play, replay training, resume-safe checkpoints, and milestone evaluation. A 375-iteration run improved internally but remained capture-heavy. |
| Search audit | `alphazero-v2-mcts-ablation`, `alphazero-search-guidance-audit`, `alphazero-puct-ablation` | Larger search recovered some defense, but neither search budget nor PUCT scaling alone solved learned-value starvation and PASS behavior. |
| AlphaZero V3 | `alphazero-v3-territory-pilot` | Added explicit territory planes. Score endings appeared, but tactical generalization regressed during the 50-iteration pilot. |
| AlphaZero V4 | `alphazero-v4-partial` | Added calibrated binary value learning, duplicate-aware replay, an early-greedy temperature schedule, and exact one-ply tactical filtering. Automatic play improved; direct human testing still found incoherent shape and repeated forced-loss entries, with the human winning all five evaluation games. |
| AlphaZero V5 | current | Reworked representation, capacity, batching, augmentation, frozen-BEST candidate promotion, calibration, and territory-rich starts. The final BEST reached 20 promotions and decisively passed the deterministic historical regression arena. |
| Freeze | 2026-09-08 | Human evaluation found V5 clearly better than V4, but not a solved or reliably human-level player. Research was frozen with the remaining endgame and strategic limitations documented. |

The original V5 territory-rich start pool was derived from valid board states
in a historical V4 replay. Only board state was reused: no historical policy or
outcome label entered V5 training. The public loader for this provenance is
self-contained in `alphazero_v5/legacy_state_loader.py`.
