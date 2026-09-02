"""Territory-aware AlphaZero V3 pilot built on unchanged Rules V2."""

from .config import TerritoryPilotConfig
from .encoder import ENCODED_SHAPE, NUM_PLANES, encode_state

__all__ = [
    "ENCODED_SHAPE",
    "NUM_PLANES",
    "TerritoryPilotConfig",
    "encode_state",
]
