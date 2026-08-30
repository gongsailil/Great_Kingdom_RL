import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from alphazero_v2.config import AlphaZeroConfig
from alphazero_v2.encoder import ENCODED_SHAPE, encode_state
from alphazero_v2.mcts import (
    MCTS,
    Node,
    backup,
    masked_policy,
    terminal_value,
    visit_count_policy,
)
from alphazero_v2.network import PolicyValueNetwork
from alphazero_v2.self_play import TrainingExample, play_self_play_game
from alphazero_v2.train import load_checkpoint, save_checkpoint, train_on_examples
from gk_env_v2 import action_mask_for_logic
from great_kingdom_v2 import (
    BLUE,
    BOARD_SIZE,
    NEUTRAL,
    NUM_ACTIONS,
    PASS_ACTION,
    RED,
    GreatKingdomLogicV2,
    MoveResultV2,
)


class BiasedNetwork(nn.Module):
    def __init__(self, preferred_action=None, value=0.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.preferred_action = preferred_action
        self.fixed_value = float(value)

    def forward(self, inputs):
        logits = torch.zeros(
            (inputs.shape[0], NUM_ACTIONS),
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        if self.preferred_action is not None:
            logits[:, self.preferred_action] = 12.0 + self.anchor * 0.0
        values = torch.full(
            (inputs.shape[0],),
            self.fixed_value,
            dtype=inputs.dtype,
            device=inputs.device,
        ) + self.anchor * 0.0
        return logits, values


def edge_blue_territory_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    for x, y in ((2, 0), (2, 1), (2, 2), (0, 2), (1, 2)):
        logic.board[y][x] = BLUE
    logic.board[8][8] = RED
    return logic


def pure_suicide_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    for x, y in ((1, 0), (0, 1), (2, 1), (0, 2), (2, 2), (1, 3)):
        logic.board[y][x] = RED
    logic.board[2][1] = BLUE
    return logic


def capture_priority_position():
    logic = GreatKingdomLogicV2()
    logic.board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    logic.board[4][4] = NEUTRAL
    logic.board[0][1] = RED
    for x, y in ((0, 1), (1, 1), (2, 0)):
        logic.board[y][x] = BLUE
    for x, y in ((0, 2), (2, 1), (1, 2)):
        logic.board[y][x] = RED
    return logic


def test_encoder_shape_and_current_player_semantics():
    logic = GreatKingdomLogicV2()
    logic.board[0][0] = BLUE
    logic.board[0][1] = RED
    logic.turn = RED
    logic.consecutive_passes = 1
    logic.castles_remaining[RED] = 20
    logic.castles_remaining[BLUE] = 10

    encoded = encode_state(logic)

    assert encoded.shape == ENCODED_SHAPE == (7, 9, 9)
    assert encoded.dtype == np.float32
    assert encoded[:, 0, 1].tolist()[:3] == [1.0, 0.0, 0.0]
    assert encoded[:, 0, 0].tolist()[:3] == [0.0, 1.0, 0.0]
    assert encoded[2, 4, 4] == 1.0
    assert np.all(encoded[3] == 0.5)
    assert np.all(encoded[4] == 0.5)
    assert np.all(encoded[5] == 0.25)
    assert np.all(encoded[6] == 0.0)  # Absolute Red-to-play marker.


def test_network_policy_value_shapes_and_range():
    network = PolicyValueNetwork(channels=16, residual_blocks=1)
    inputs = torch.zeros((2, *ENCODED_SHAPE), dtype=torch.float32)
    logits, values = network(inputs)
    assert logits.shape == (2, 82)
    assert values.shape == (2,)
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)


def test_masked_policy_has_exact_zero_on_illegal_actions():
    logic = GreatKingdomLogicV2()
    legal_mask = action_mask_for_logic(logic, logic.turn)
    probabilities = masked_policy(np.zeros(NUM_ACTIONS), legal_mask)
    assert probabilities.shape == (82,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[~legal_mask] == 0.0)
    assert probabilities[PASS_ACTION] > 0.0


def test_mcts_root_uses_rules_v2_mask_and_pass_child():
    config = AlphaZeroConfig(
        channels=16,
        residual_blocks=1,
        mcts_simulations=2,
    )
    logic = GreatKingdomLogicV2()
    root = MCTS(BiasedNetwork(), config, "cpu").run(logic)
    assert 4 + 4 * BOARD_SIZE not in root.children
    assert PASS_ACTION in root.children
    policy = visit_count_policy(root, temperature=1.0)
    legal_mask = action_mask_for_logic(logic, BLUE)
    assert np.all(policy[~legal_mask] == 0.0)


def test_territory_and_suicide_legality_are_inherited_by_mcts():
    config = AlphaZeroConfig(mcts_simulations=1)
    search = MCTS(BiasedNetwork(), config, "cpu")

    territory = edge_blue_territory_position()
    own_action = 0
    blue_root = search.run(territory)
    assert own_action in blue_root.children
    territory.turn = RED
    red_root = search.run(territory)
    assert own_action not in red_root.children

    suicide = pure_suicide_position()
    suicide_action = 1 + BOARD_SIZE
    suicide_root = search.run(suicide)
    assert suicide_action not in suicide_root.children


def test_capture_terminal_value_and_mcts_backup_sign():
    logic = capture_priority_position()
    capture_action = 0
    assert logic.classify_placement(BLUE, 0, 0) == MoveResultV2.CAPTURE_WIN
    transitioned = logic.copy()
    assert transitioned.apply_action(capture_action) == MoveResultV2.CAPTURE_WIN
    assert terminal_value(transitioned, BLUE) == 1.0
    assert terminal_value(transitioned, RED) == -1.0

    config = AlphaZeroConfig(mcts_simulations=4)
    root = MCTS(
        BiasedNetwork(preferred_action=capture_action),
        config,
        "cpu",
    ).run(logic)
    capture_child = root.children[capture_action]
    assert capture_child.visit_count > 0
    assert capture_child.to_play == BLUE
    assert np.isclose(capture_child.value(), 1.0)

    blue_node = Node(1.0, BLUE)
    red_node = Node(1.0, RED)
    backup([blue_node, red_node], value=-1.0, value_player=RED)
    assert blue_node.value() == 1.0
    assert red_node.value() == -1.0


def test_pass_pass_scoring_path_is_searchable():
    config = AlphaZeroConfig(mcts_simulations=4)
    root = MCTS(
        BiasedNetwork(preferred_action=PASS_ACTION),
        config,
        "cpu",
    ).run(GreatKingdomLogicV2())
    first_pass = root.children[PASS_ACTION]
    assert first_pass.visit_count > 0
    assert PASS_ACTION in first_pass.children
    assert first_pass.children[PASS_ACTION].visit_count > 0

    logic = GreatKingdomLogicV2()
    assert logic.apply_action(PASS_ACTION) == MoveResultV2.PASS
    assert logic.apply_action(PASS_ACTION) == MoveResultV2.PASS_SCORE_END
    assert logic.game_over and logic.winner == RED


def test_biased_pass_self_play_builds_state_pi_z():
    config = AlphaZeroConfig(
        mcts_simulations=4,
        self_play_games=1,
        max_game_moves=10,
        dirichlet_fraction=0.0,
    )
    examples, stats = play_self_play_game(
        BiasedNetwork(preferred_action=PASS_ACTION),
        config,
        "cpu",
        np.random.default_rng(7),
    )
    assert stats["terminal_reason"] == "PASS_SCORE_END"
    assert stats["pass_usage"] == 2
    assert stats["illegal_probability_violations"] == 0
    assert len(examples) == 2
    for example in examples:
        assert example.state.shape == (7, 9, 9)
        assert example.policy.shape == (82,)
        assert np.isclose(example.policy.sum(), 1.0)
        assert example.value in (-1.0, 1.0)


def test_tiny_training_and_checkpoint_round_trip():
    config = AlphaZeroConfig(
        channels=8,
        residual_blocks=1,
        training_epochs=2,
        batch_size=2,
    )
    logic = GreatKingdomLogicV2()
    state = encode_state(logic)
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    policy[PASS_ACTION] = 1.0
    examples = [
        TrainingExample(state.copy(), policy.copy(), 1.0, BLUE),
        TrainingExample(state.copy(), policy.copy(), -1.0, RED),
    ]
    network = PolicyValueNetwork(config.channels, config.residual_blocks)
    optimizer, before, after = train_on_examples(
        network,
        examples,
        config,
        "cpu",
    )
    assert np.isfinite(before["total"])
    assert np.isfinite(after["total"])

    network.eval()
    inputs = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        expected = network(inputs)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        checkpoint = Path(temp_dir_name) / "minimal.pt"
        metadata = {"test": True}
        save_checkpoint(checkpoint, network, optimizer, config, metadata)
        loaded, _, loaded_config, loaded_metadata = load_checkpoint(
            checkpoint,
            "cpu",
        )
        loaded.eval()
        with torch.no_grad():
            actual = loaded(inputs)
        assert loaded_config.to_dict() == config.to_dict()
        assert loaded_metadata == metadata
        assert torch.allclose(expected[0], actual[0])
        assert torch.allclose(expected[1], actual[1])


if __name__ == "__main__":
    test_encoder_shape_and_current_player_semantics()
    test_network_policy_value_shapes_and_range()
    test_masked_policy_has_exact_zero_on_illegal_actions()
    test_mcts_root_uses_rules_v2_mask_and_pass_child()
    test_territory_and_suicide_legality_are_inherited_by_mcts()
    test_capture_terminal_value_and_mcts_backup_sign()
    test_pass_pass_scoring_path_is_searchable()
    test_biased_pass_self_play_builds_state_pi_z()
    test_tiny_training_and_checkpoint_round_trip()
    print("AlphaZero V2 tests: PASS")
