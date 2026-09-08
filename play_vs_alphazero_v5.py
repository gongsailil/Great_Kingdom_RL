"""Play Great Kingdom Rules V2 against a frozen AlphaZero V5 checkpoint."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pygame

from alphazero_v5.common import atomic_json_save
from alphazero_v5.evaluation import load_v5_checkpoint, select_v5_with_diagnostics
from alphazero_v5.tactical import immediate_winning_actions, solve_tactical_root
from game_ui import COLOR_BLUE_TOP, COLOR_RED_TOP, GreatKingdomRenderer
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE, RED, BOARD_SIZE, PASS_ACTION, GreatKingdomLogicV2, MoveResultV2,
    player_name,
)


DEFAULT_CHECKPOINT = Path("runs/alphazero_v5/strategy_20260908/latest.pt")
DEFAULT_LOG_DIR = Path("human_games/v5_strategy")
LEGAL_RESULTS = (
    MoveResultV2.NORMAL, MoveResultV2.CAPTURE_WIN,
    MoveResultV2.PASS, MoveResultV2.PASS_SCORE_END,
)
IMPOSSIBLE_MESSAGES = {
    MoveResultV2.IMPOSSIBLE_OCCUPIED: "Cannot play: occupied square.",
    MoveResultV2.IMPOSSIBLE_OPPONENT_TERRITORY:
        "Cannot play inside the opponent's territory.",
    MoveResultV2.IMPOSSIBLE_SUICIDE: "Cannot play: pure suicide is illegal.",
    MoveResultV2.IMPOSSIBLE_NO_CASTLES: "No castles remain; press P to pass.",
    MoveResultV2.IMPOSSIBLE_OUT_OF_BOUNDS: "Cannot play outside the board.",
}


def player_number(player):
    if player in ("blue", BLUE):
        return BLUE
    if player in ("red", RED):
        return RED
    raise ValueError("human player must be 'blue' or 'red'")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_record(action):
    return {
        "action": int(action), "is_pass": action == PASS_ACTION,
        "coordinate": (
            None if action == PASS_ACTION
            else [action % BOARD_SIZE, action // BOARD_SIZE]
        ),
    }


def position_record(logic):
    counts = logic.territory_counts()
    return {
        "player": logic.turn,
        "territory_blue": counts[BLUE], "territory_red": counts[RED],
        "current_player_territory": counts[logic.turn],
        "opponent_territory": counts[3 - logic.turn],
        "consecutive_passes": logic.consecutive_passes,
    }


class HumanVsAlphaZeroV5Controller:
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, selector=select_v5_with_diagnostics):
        if int(mcts_simulations) <= 0:
            raise ValueError("MCTS simulations must be positive")
        self.human_player = player_number(human_player)
        self.ai_player = 3 - self.human_player
        self.checkpoint = checkpoint
        self.mcts_simulations = int(mcts_simulations)
        self.selector = selector
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_sha256 = file_digest(checkpoint.path)
        self.logic = GreatKingdomLogicV2()
        self.last_ai_action = None
        self.last_tactical_mode = None
        self._new_log()

    @property
    def is_human_turn(self):
        return not self.logic.game_over and self.logic.turn == self.human_player

    @property
    def is_ai_turn(self):
        return not self.logic.game_over and self.logic.turn == self.ai_player

    def _new_log(self):
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
            "architecture": "AlphaZero V5 strategy",
            "checkpoint": str(self.checkpoint.path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_cycle": self.checkpoint.cycle,
            "best_version": self.checkpoint.best_version,
            "calibration_temperature": self.checkpoint.calibration_temperature,
            "human_player": self.human_player, "ai_player": self.ai_player,
            "mcts_simulations": self.mcts_simulations, "c_puct": 1.5,
            "root_noise": False, "temperature": 0.0,
            "tactical_solver": True, "status": "active", "moves": [],
            "game_length": 0, "winner": None, "terminal_reason": None,
            "score_blue": None, "score_red": None,
        }
        self._save()

    def _save(self):
        atomic_json_save(self.game_log, self.log_path)

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
        root_player = self.logic.turn
        tactical = solve_tactical_root(self.logic)
        mode = selection["tactical_mode"] if selection else tactical.mode
        if actor == "AI":
            mask = action_mask_for_logic(self.logic, root_player)
            if not 0 <= action < len(mask) or not mask[action]:
                raise RuntimeError(f"V5 selected illegal action {action}")
        result = self.logic.apply_action(action)
        if result not in LEGAL_RESULTS:
            if actor == "AI":
                raise RuntimeError(f"V5 selected illegal action {action}: {result.name}")
            return result
        if actor == "AI" and mode == "IMMEDIATE_WIN":
            if not self.logic.game_over or self.logic.winner != root_player:
                raise RuntimeError("exact immediate-win layer failed")
        if actor == "AI" and mode == "SAFE_DEFENSE":
            if self.logic.game_over and self.logic.winner != root_player:
                raise RuntimeError("exact safe-defense layer failed")
            if not self.logic.game_over and immediate_winning_actions(
                self.logic, self.logic.turn
            ):
                raise RuntimeError("exact safe-defense layer left an immediate loss")
        self.game_log["moves"].append({
            "ply": len(self.game_log["moves"]), "actor": actor,
            **before, **action_record(action), "tactical_mode": mode,
            "ai_root_top5": selection["top_actions"][:5] if selection else None,
            "mcts_executed": selection["mcts_executed"] if selection else False,
            "result": result.name,
        })
        self.game_log["game_length"] = len(self.game_log["moves"])
        if self.logic.game_over:
            self.game_log.update(
                status="completed", winner=self.logic.winner,
                terminal_reason=result.name, score_blue=self.logic.score_blue,
                score_red=self.logic.score_red,
            )
        self._save()
        return result

    def play_human_move(self, x, y):
        if not self.is_human_turn:
            raise RuntimeError("not the human turn")
        if not self.logic.is_on_board(x, y):
            return self.logic.place_stone_detailed(x, y)
        return self._play_logged(y * BOARD_SIZE + x, "Human")

    def play_human_pass(self):
        if not self.is_human_turn:
            raise RuntimeError("not the human turn")
        return self._play_logged(PASS_ACTION, "Human")

    def play_ai_move(self):
        if not self.is_ai_turn:
            raise RuntimeError("not the AI turn")
        selected = self.selector(
            self.checkpoint, self.logic, self.mcts_simulations, top_k=5
        )
        result = self._play_logged(selected["action"], "AI", selected)
        self.last_ai_action = selected["action"]
        self.last_tactical_mode = selected["tactical_mode"]
        print(
            f"V5 {self.last_tactical_mode}: {action_record(self.last_ai_action)} | "
            f"best v{self.checkpoint.best_version} cycle {self.checkpoint.cycle} | "
            f"{self.log_path}", flush=True,
        )
        return self.last_ai_action, result


class HumanVsAlphaZeroV5UI:
    def __init__(self, human_player, checkpoint, mcts_simulations=256,
                 log_dir=DEFAULT_LOG_DIR, renderer=None,
                 selector=select_v5_with_diagnostics):
        self.controller = HumanVsAlphaZeroV5Controller(
            human_player, checkpoint, mcts_simulations, log_dir, selector
        )
        self.renderer = renderer or GreatKingdomRenderer(
            "Great Kingdom - Human vs AlphaZero V5"
        )
        self.info_message = self._start_message()

    def _identity_text(self):
        controller = self.controller
        return (
            f"Human {player_name(controller.human_player)} | "
            f"V5 best v{controller.checkpoint.best_version} "
            f"cycle {controller.checkpoint.cycle} | "
            f"MCTS {controller.mcts_simulations}"
        )

    def _start_message(self):
        opener = "AI opens." if self.controller.ai_player == BLUE else "Human opens."
        return f"{self._identity_text()} | {opener} P = PASS"

    def _terminal_message(self, result):
        logic = self.controller.logic
        score = (
            "" if logic.score_blue is None
            else f" | territory B {logic.score_blue} R {logic.score_red}"
        )
        return f"Winner {player_name(logic.winner)} | {result.name}{score}"

    def restart(self):
        self.controller.restart()
        self.info_message = self._start_message()

    def play_human_at(self, x, y):
        result = self.controller.play_human_move(x, y)
        if result in IMPOSSIBLE_MESSAGES:
            self.info_message = IMPOSSIBLE_MESSAGES[result]
        elif self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        else:
            self.info_message = f"{self._identity_text()} | AI thinking..."
        return result

    def play_human_pass(self):
        result = self.controller.play_human_pass()
        if self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        else:
            self.info_message = f"{self._identity_text()} | Human passed; AI thinking..."
        return result

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.restart()
            elif event.key == pygame.K_p and self.controller.is_human_turn:
                self.play_human_pass()
        elif (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
            and self.controller.is_human_turn
        ):
            coordinate = self.renderer.board_coordinate(event.pos)
            if coordinate is not None:
                self.play_human_at(*coordinate)
        return True

    def draw(self):
        logic = self.controller.logic
        if logic.game_over:
            headline = headline_color = ghost_player = None
        else:
            role = "AI" if self.controller.is_ai_turn else "Human"
            headline = (
                f"{role} {player_name(logic.turn)} turn | "
                f"B {logic.castles_remaining[BLUE]} "
                f"R {logic.castles_remaining[RED]} | "
                f"passes {logic.consecutive_passes}"
            )
            headline_color = COLOR_BLUE_TOP if logic.turn == BLUE else COLOR_RED_TOP
            ghost_player = (
                self.controller.human_player if self.controller.is_human_turn else None
            )
        self.renderer.draw_frame(
            logic, self.info_message, headline=headline,
            headline_color=headline_color, ghost_player=ghost_player,
        )

    def _set_ai_message(self, action, result):
        if self.controller.logic.game_over:
            self.info_message = self._terminal_message(result)
        elif action == PASS_ACTION:
            self.info_message = f"{self._identity_text()} | AI last: PASS | Your turn"
        else:
            self.info_message = (
                f"{self._identity_text()} | AI last: "
                f"({action % BOARD_SIZE}, {action // BOARD_SIZE}) | Your turn"
            )

    def run(self, max_frames=None):
        print(f"Local game log: {self.controller.log_path}", flush=True)
        running, frame_count = True, 0
        try:
            while running and (max_frames is None or frame_count < max_frames):
                self.renderer.clock.tick(30)
                for event in pygame.event.get():
                    running = self.handle_event(event)
                    if not running:
                        break
                self.draw()
                if running and self.controller.is_ai_turn:
                    pygame.event.pump()
                    action, result = self.controller.play_ai_move()
                    self._set_ai_message(action, result)
                frame_count += 1
        finally:
            self.controller.close_log()
            self.renderer.close()


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
        args.human_player, checkpoint, args.mcts_simulations, args.log_dir
    ).run()


if __name__ == "__main__":
    main()
