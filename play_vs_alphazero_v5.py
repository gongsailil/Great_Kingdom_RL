"""Play Rules V2 against an AlphaZero V5 BEST checkpoint."""

import argparse
from pathlib import Path

from game_ui import GreatKingdomRenderer
from great_kingdom_v2 import player_name
from alphazero_v4.acceptance import action_record
from play_vs_alphazero_v4 import HumanVsAlphaZeroV4Controller, HumanVsAlphaZeroV4UI

from alphazero_v5.evaluation import load_v5_checkpoint, select_v5_with_diagnostics


DEFAULT_CHECKPOINT = Path("runs/alphazero_v5/strategy_20260908/latest.pt")
DEFAULT_LOG_DIR = Path("human_games/v5_strategy")


class HumanVsAlphaZeroV5Controller(HumanVsAlphaZeroV4Controller):
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, selector=select_v5_with_diagnostics):
        super().__init__(human_player, checkpoint, mcts_simulations, log_dir, selector)

    def _new_log(self):
        super()._new_log()
        self.game_log.update(
            architecture="AlphaZero V5 strategy",
            checkpoint_cycle=self.checkpoint.cycle,
            best_version=self.checkpoint.best_version,
            calibration_temperature=self.checkpoint.calibration_temperature,
        )
        self._save()

    def play_ai_move(self):
        if not self.is_ai_turn:
            raise RuntimeError("not the AI turn")
        selected = self.selector(
            self.checkpoint, self.logic, self.mcts_simulations, top_k=5)
        result = self._play_logged(selected["action"], "AI", selected)
        self.last_ai_action = selected["action"]
        self.last_tactical_mode = selected["tactical_mode"]
        print(
            f"V5 {self.last_tactical_mode}: {action_record(self.last_ai_action)} | "
            f"best v{self.checkpoint.best_version} cycle {self.checkpoint.cycle} | {self.log_path}",
            flush=True,
        )
        return self.last_ai_action, result


class HumanVsAlphaZeroV5UI(HumanVsAlphaZeroV4UI):
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, renderer=None, selector=select_v5_with_diagnostics):
        self.controller = HumanVsAlphaZeroV5Controller(
            human_player, checkpoint, mcts_simulations, log_dir, selector)
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs AlphaZero V5")
        self.info_message = self._start_message()

    def _identity_text(self):
        controller = self.controller
        return (
            f"Human {player_name(controller.human_player)} | "
            f"V5 best v{controller.checkpoint.best_version} "
            f"cycle {controller.checkpoint.cycle} | MCTS {controller.mcts_simulations}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--human-player", choices=("blue", "red"), default="blue")
    parser.add_argument("--mcts-simulations", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args(argv)
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = load_v5_checkpoint(args.checkpoint, args.device)
    HumanVsAlphaZeroV5UI(
        args.human_player, checkpoint, args.mcts_simulations, args.log_dir).run()


if __name__ == "__main__":
    main()
