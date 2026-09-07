"""Acceptance protocol/UI regression, using temporary untrained test fixtures."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import torch

from alphazero_v2.evaluate import (
    apply_opening, capture_tactical_position, defense_tactical_position,
    winning_pass_tactical_position,
)
from alphazero_v3.temperature_audit import network_state_digest
from alphazero_v4.acceptance import (
    action_record, classify_acceptance, generate_acceptance_openings,
    json_digest, paired_bootstrap, play_acceptance_game,
    select_v4_with_diagnostics, strategic_positions, summarize_matchup,
    validate_acceptance_openings,
)
from alphazero_v4.config import V4Config
from alphazero_v4.evaluation import load_v4_evaluation_checkpoint
from alphazero_v4.network import PolicyValueLogitNetwork
from alphazero_v4.self_play import select_root_action
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import BLUE, RED, PASS_ACTION, GreatKingdomLogicV2, MoveResultV2
from play_vs_alphazero_v4 import HumanVsAlphaZeroV4Controller, HumanVsAlphaZeroV4UI, parse_args


def temporary_checkpoint(directory):
    path = Path(directory) / "iteration_000050.pt"
    config = V4Config(channels=8, residual_blocks=1, mcts_simulations=2)
    network = PolicyValueLogitNetwork(channels=8, residual_blocks=1)
    torch.save({"architecture": "alphazero_v4_raw_value_logit", "iteration": 50,
                "config": config.to_dict(), "network_state_dict": network.state_dict()}, path)
    return load_v4_evaluation_checkpoint(path, "cpu", expected_iteration=50)


def test_checkpoint_load_consistency_and_no_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = temporary_checkpoint(tmp)
        assert checkpoint.iteration == 50
        try:
            load_v4_evaluation_checkpoint(checkpoint.path, "cpu", expected_iteration=49)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong iteration accepted")
        before = network_state_digest(checkpoint.network)
        for logic in (GreatKingdomLogicV2(), capture_tactical_position(),
                      defense_tactical_position(), winning_pass_tactical_position()):
            state_before = copy.deepcopy(vars(logic))
            selected = select_v4_with_diagnostics(checkpoint, logic, 2)
            repeat = select_v4_with_diagnostics(checkpoint, logic, 2)
            production = select_root_action(checkpoint.network, logic, checkpoint.config,
                                           checkpoint.device, np.random.default_rng(0),
                                           ply=8, add_root_noise=False, temperature_override=0)
            assert selected == repeat
            assert selected["action"] == production.action
            assert selected["tactical_mode"] == production.tactical.mode
            assert action_mask_for_logic(logic, logic.turn)[selected["action"]]
            assert vars(logic) == state_before
            if selected["tactical_mode"] == "IMMEDIATE_WIN":
                assert not selected["mcts_executed"]
                assert selected["top_actions"][0]["q_value_root_player"] == 1
                assert selected["top_actions"][0]["visit_count"] is None
        assert network_state_digest(checkpoint.network) == before


def test_openings_deterministic_unique_reused_and_paired():
    suite = generate_acceptance_openings()
    assert suite == generate_acceptance_openings()
    assert len(suite["openings"]) == 50
    assert {len(o["actions"]) for o in suite["openings"]} == set(range(2, 7))
    assert validate_acceptance_openings(suite)
    before = json_digest(suite)
    opponent = SimpleNamespace(config={"mcts_simulations": 64})

    def v4_pass(*args):
        return {**action_record(PASS_ACTION), "tactical_mode": "NORMAL"}

    def historical_pass(*args):
        assert args[-1] == 64
        return PASS_ACTION

    all_results = []
    for name in ("v3_iter50", "v2_iter375"):
        games = [play_acceptance_game(None, opponent, opening, color, name,
                                     v4_pass, historical_pass)
                 for opening in suite["openings"][:2] for color in (BLUE, RED)]
        assert [g["v4_color"] for g in games] == [BLUE, RED, BLUE, RED]
        assert all(g["terminal_reason"] == "PASS_SCORE_END" for g in games)
        assert all(g["illegal_violations"] == 0 for g in games)
        result = summarize_matchup(games)
        assert result["v4_wins"] == 2
        assert result["paired_bootstrap"]["difference_ci95"] == [0, 0]
        all_results.append(games)
    assert [g["opening_actions"] for g in all_results[0]] == [g["opening_actions"] for g in all_results[1]]
    assert json_digest(suite) == before
    broken = copy.deepcopy(suite)
    broken["openings"][0]["actions"][0] = PASS_ACTION
    try:
        validate_acceptance_openings(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("PASS opening accepted")
    try:
        paired_bootstrap(all_results[0][:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired bootstrap accepted")


def test_illegal_selection_fails():
    opponent = SimpleNamespace(config={"mcts_simulations": 64})
    def illegal(*args):
        return {"action": 40, "tactical_mode": "NORMAL"}
    opening = {"opening_id": 0, "actions": [0, 80], "resulting_turn": BLUE}
    try:
        play_acceptance_game(None, opponent, opening, BLUE, "v2_iter375", illegal)
    except RuntimeError as error:
        assert "illegal" in str(error)
    else:
        raise AssertionError("illegal action accepted")


def test_strategic_fixture_meanings():
    states = {name: logic for name, logic, exact in strategic_positions()}
    assert len(states) == 10
    for name, logic in states.items():
        assert not logic.game_over
        if "exact_winning_pass" in name:
            child = logic.copy()
            assert child.apply_action(PASS_ACTION) == MoveResultV2.PASS_SCORE_END
            assert child.winner == logic.turn
    blue_needs = states["E_blue_needs_more_territory"]
    assert blue_needs.territory_counts() == {BLUE: 1, RED: 0}
    assert blue_needs.turn == BLUE
    assert states["F_red_scoring_advantage"].turn == RED
    cost = states["G_own_territory_cost"]
    before = cost.count_territory(BLUE)
    assert cost.get_territory_owner(0, 0) == BLUE
    assert cost.apply_action(0) == MoveResultV2.NORMAL
    assert cost.count_territory(BLUE) == before - 1


def test_human_initialization_first_move_logging_pass_restart_and_dummy():
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = temporary_checkpoint(tmp)
        before = network_state_digest(checkpoint.network)
        blue = HumanVsAlphaZeroV4Controller("blue", checkpoint, 2, Path(tmp) / "logs")
        assert blue.is_human_turn and not blue.is_ai_turn
        assert blue.play_human_move(4, 4) == MoveResultV2.IMPOSSIBLE_OCCUPIED
        assert blue.game_log["moves"] == []
        assert blue.play_human_pass() == MoveResultV2.PASS
        action, result = blue.play_ai_move()
        assert action == PASS_ACTION and result == MoveResultV2.PASS_SCORE_END
        assert blue.logic.winner == RED
        old_path = blue.log_path
        payload = json.loads(old_path.read_text())
        assert payload["status"] == "completed" and payload["game_length"] == 2
        assert [m["actor"] for m in payload["moves"]] == ["Human", "AI"]
        assert payload["moves"][1]["tactical_mode"] == "IMMEDIATE_WIN"
        assert payload["moves"][1]["ai_root_top5"][0]["action"] == PASS_ACTION
        assert payload["score_blue"] == payload["score_red"] == 0
        blue.restart()
        assert blue.log_path != old_path and json.loads(old_path.read_text()) == payload
        active_path = blue.log_path
        blue.play_human_move(0, 0)
        blue.restart()
        assert json.loads(active_path.read_text())["status"] == "restarted"
        blue.close_log()
        assert json.loads(blue.log_path.read_text())["status"] == "interrupted"
        # Real two-simulation first-move search on the dummy window, no full game.
        red_ui = HumanVsAlphaZeroV4UI("red", checkpoint, 2, Path(tmp) / "logs")
        assert red_ui.controller.is_ai_turn
        red_ui.run(max_frames=1)
        red_log = json.loads(red_ui.controller.log_path.read_text())
        assert red_log["moves"][0]["player"] == BLUE
        assert red_log["moves"][0]["actor"] == "AI"
        assert red_ui.controller.is_human_turn
        blue_ui = HumanVsAlphaZeroV4UI("blue", checkpoint, 2, Path(tmp) / "logs")
        blue_ui.run(max_frames=1)
        assert blue_ui.controller.game_log["moves"] == []
        assert network_state_digest(checkpoint.network) == before
        assert parse_args([]).mcts_simulations == 256


def test_conservative_acceptance_policy():
    exact = [{"exact_assertion": True, "success": True}]
    def metric(rate, ci, blue, red):
        return {"illegal_violations": 0, "tactical_failures": 0,
                "paired_bootstrap": {"v4_win_rate": rate, "difference_ci95": ci},
                "v4_by_color": {"1": {"wins": blue, "games": 50}, "2": {"wins": red, "games": 50}}}
    assert classify_acceptance({"v3_iter50": metric(.65, [.1, .5], 35, 30)}, exact)[0] == "AUTO_PASS"
    assert classify_acceptance({"v3_iter50": metric(.52, [-.1, .2], 28, 24)}, exact)[0] == "AUTO_INCONCLUSIVE"
    assert classify_acceptance({"v3_iter50": metric(.3, [-.6, -.2], 20, 10)}, exact)[0] == "AUTO_FAIL"
    assert classify_acceptance({"v3_iter50": metric(.52, [-.1, .2], 48, 4)}, exact)[0] == "AUTO_FAIL"


def main():
    torch.set_num_threads(1)
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("AlphaZero V4 acceptance/UI tests: PASS")


if __name__ == "__main__":
    main()
