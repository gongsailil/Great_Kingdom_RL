# 📘 Great Kingdom RL — Reinforcement Learning Agent for a Custom Strategy Game

This repository contains **Great Kingdom**, a custom 9×9 strategic board game, along with a **Reinforcement Learning (RL) agent** trained using **PPO + a custom CNN** in parallel environments.

It includes:

- A full Pygame-based board game implementation  
- A Gymnasium-compatible environment  
- A custom CNN feature extractor for board states  
- A parallel PPO training pipeline using Stable-Baselines3  
- An interactive UI for playing against the trained AI

---

## 🏰 Game Overview

Great Kingdom is a deterministic, turn-based strategy game played on a **9×9 grid**.

Two players, **Blue (1)** and **Red (2)**, take turns placing stones on the board:

- The board starts with a **neutral tile** in the center (value `3`).
- Placement rules and captures are based on **liberties** (similar to the game of Go).
- The game tracks:
  - Legal/illegal moves  
  - Captures  
  - Territory  
  - Game-over conditions  
  - Final scoring with **komi** (default: 3.0 for Red)

The game ends when:

- A capture immediately determines the winner, or  
- No legal moves remain, or  
- The board is full and final territory scoring is applied.

The final winner is decided by comparing **Blue territory** vs **Red territory + komi**.

---

## 📂 Project Structure

```text
Great_Kingdom_RL/
│
├── great_kingdom.py          # Core game logic + base Pygame UI
├── gk_env.py                 # Gymnasium environment wrapper
├── custom_cnn.py             # Custom CNN feature extractor for PPO
├── train_cnn_parallel.py     # Parallel PPO training script
├── play_rl.py                # UI for playing against the trained AI
│
└── models/
    └── PPO_CNN/              # Trained PPO model checkpoints (auto-saved)
