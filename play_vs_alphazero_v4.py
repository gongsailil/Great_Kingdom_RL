"""Human vs fixed AlphaZero V4, with production tactics and local game logs."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from alphazero_v2.training_runner import _atomic_json_save
from alphazero_v4.acceptance import (
    LEGAL_RESULTS, action_record, assert_v4_tactical_choice, file_digest,
    position_record, select_v4_with_diagnostics,
)
from alphazero_v4.evaluation import load_v4_evaluation_checkpoint
from alphazero_v4.tactical import solve_tactical_root
from game_ui import GreatKingdomRenderer
from great_kingdom_v2 import GreatKingdomLogicV2, PASS_ACTION, player_name
from play_vs_alphazero_v2 import HumanVsAlphaZeroV2Controller, HumanVsAlphaZeroV2UI


DEFAULT_CHECKPOINT = Path("runs/alphazero_v4/stability_20260903/latest.pt")
DEFAULT_LOG_DIR = Path("human_games/v4_20260906")


class HumanVsAlphaZeroV4Controller(HumanVsAlphaZeroV2Controller):
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, selector=select_v4_with_diagnostics):
        super().__init__(human_player, checkpoint, mcts_simulations)
        self.selector = selector
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_sha256 = file_digest(checkpoint.path)
        self.last_tactical_mode = None
        self._new_log()

    def _new_log(self):
        # Exclusive creation reserves the next local name without overwriting games.
        index = 1
        while True:
            path = self.log_dir / f"game_{index:03d}.json"
            try:
                with path.open("x", encoding="utf-8") as handle:
                    handle.write("{}\n")
                break
            except FileExistsError:
                index += 1
        self.log_path = path
        self.game_log = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(self.checkpoint.path), "checkpoint_iteration": self.checkpoint.iteration,
            "checkpoint_sha256": self.checkpoint_sha256,
            "human_player": self.human_player, "ai_player": self.ai_player,
            "mcts_simulations": self.mcts_simulations, "c_puct": 1.5,
            "root_noise": False, "temperature": 0.0, "tactical_solver": True,
            "status": "active", "moves": [], "game_length": 0,
            "winner": None, "terminal_reason": None, "score_blue": None, "score_red": None,
        }
        self._save()

    def _save(self):
        _atomic_json_save(self.game_log, self.log_path)

    def close_log(self, status="interrupted"):
        if self.game_log["status"] == "active":
            self.game_log["status"] = status
            self._save()

    def restart(self):
        self.close_log("restarted")
        self.logic = GreatKingdomLogicV2()
        self.last_ai_action = None
        self.last_tactical_mode = None
        self._new_log()

    def _play_logged(self, action, actor, selection=None):
        before = position_record(self.logic)
        # Human annotations describe the same exact root solver, without search.
        mode = selection["tactical_mode"] if selection else solve_tactical_root(self.logic).mode
        result = self.logic.apply_action(action)
        if result not in LEGAL_RESULTS:
            if actor == "AI":
                raise RuntimeError(f"V4 selected illegal action {action}: {result.name}")
            return result
        if actor == "AI":
            assert_v4_tactical_choice({**selection, "root_player": before["player"]}, self.logic)
        self.game_log["moves"].append({
            "ply": len(self.game_log["moves"]), "actor": actor,
            **before, **action_record(action), "tactical_mode": mode,
            "ai_root_top5": selection["top_actions"][:5] if selection else None,
            "mcts_executed": selection["mcts_executed"] if selection else False,
            "result": result.name,
        })
        self.game_log["game_length"] = len(self.game_log["moves"])
        if self.logic.game_over:
            self.game_log.update(status="completed", winner=self.logic.winner,
                                 terminal_reason=result.name, score_blue=self.logic.score_blue,
                                 score_red=self.logic.score_red)
        self._save()
        return result

    def play_human_move(self, x, y):
        if not self.is_human_turn:
            raise RuntimeError("not the human turn")
        if not self.logic.is_on_board(x, y):
            return self.logic.place_stone_detailed(x, y)
        return self._play_logged(y * 9 + x, "Human")

    def play_human_pass(self):
        if not self.is_human_turn:
            raise RuntimeError("not the human turn")
        return self._play_logged(PASS_ACTION, "Human")

    def play_ai_move(self):
        if not self.is_ai_turn:
            raise RuntimeError("not the AI turn")
        selected = self.selector(self.checkpoint, self.logic, self.mcts_simulations, top_k=5)
        result = self._play_logged(selected["action"], "AI", selected)
        self.last_ai_action = selected["action"]
        self.last_tactical_mode = selected["tactical_mode"]
        print(f"V4 {self.last_tactical_mode}: {action_record(self.last_ai_action)} | {self.log_path}", flush=True)
        return self.last_ai_action, result


class HumanVsAlphaZeroV4UI(HumanVsAlphaZeroV2UI):
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, renderer=None, selector=select_v4_with_diagnostics):
        self.controller = HumanVsAlphaZeroV4Controller(
            human_player, checkpoint, mcts_simulations, log_dir, selector)
        self.renderer = renderer or GreatKingdomRenderer("Great Kingdom - Human vs AlphaZero V4")
        self.info_message = self._start_message()

    def _identity_text(self):
        c = self.controller
        return f"Human {player_name(c.human_player)} | V4 iter{c.checkpoint.iteration} | MCTS{c.mcts_simulations}"

    def run(self, max_frames=None):
        print(f"Local game log: {self.controller.log_path}", flush=True)
        try:
            return super().run(max_frames=max_frames)
        finally:
            self.controller.close_log()


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
    checkpoint = load_v4_evaluation_checkpoint(args.checkpoint, args.device, expected_iteration=50)
    HumanVsAlphaZeroV4UI(args.human_player, checkpoint, args.mcts_simulations, args.log_dir).run()


if __name__ == "__main__":
    main()
