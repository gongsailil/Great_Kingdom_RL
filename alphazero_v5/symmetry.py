"""Eight exact D4 transforms for 9x9 state and 82-action policy tensors."""

import numpy as np

from great_kingdom_v2 import BOARD_SIZE, NUM_ACTIONS, PASS_ACTION


NUM_TRANSFORMS = 8


def _validate_transform(transform):
    transform = int(transform)
    if not 0 <= transform < NUM_TRANSFORMS:
        raise ValueError("D4 transform must be in [0,7]")
    return transform


def transform_spatial(values, transform):
    transform = _validate_transform(transform)
    result = np.asarray(values)
    if result.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("spatial value must end in a 9x9 board")
    if transform >= 4:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(np.rot90(result, transform % 4, axes=(-2, -1)))


def _action_maps():
    maps = []
    for transform in range(NUM_TRANSFORMS):
        mapping = np.empty(NUM_ACTIONS, dtype=np.int64)
        for action in range(PASS_ACTION):
            marker = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
            marker[action // BOARD_SIZE, action % BOARD_SIZE] = 1
            ty, tx = np.argwhere(transform_spatial(marker, transform))[0]
            mapping[action] = int(ty * BOARD_SIZE + tx)
        mapping[PASS_ACTION] = PASS_ACTION
        maps.append(mapping)
    return tuple(maps)


ACTION_MAPS = _action_maps()
INVERSE_TRANSFORMS = tuple(
    next(
        candidate for candidate in range(NUM_TRANSFORMS)
        if np.array_equal(ACTION_MAPS[candidate][ACTION_MAPS[transform]], np.arange(NUM_ACTIONS))
    )
    for transform in range(NUM_TRANSFORMS)
)


def transform_action(action, transform):
    action = int(action)
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError("action outside the 82-action space")
    return int(ACTION_MAPS[_validate_transform(transform)][action])


def transform_policy(policy, transform):
    policy = np.asarray(policy)
    if policy.shape != (NUM_ACTIONS,):
        raise ValueError("policy must have shape (82,)")
    transformed = np.empty_like(policy)
    transformed[ACTION_MAPS[_validate_transform(transform)]] = policy
    return transformed


def transform_state(state, transform):
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[-2:] != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError("state must have shape (planes,9,9)")
    return transform_spatial(state, transform)


def inverse_transform(transform):
    return INVERSE_TRANSFORMS[_validate_transform(transform)]


def inverse_policy(policy, transform):
    return transform_policy(policy, inverse_transform(transform))


def inverse_state(state, transform):
    return transform_state(state, inverse_transform(transform))
