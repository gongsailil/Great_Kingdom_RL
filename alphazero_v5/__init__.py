"""AlphaZero V5 strategy training on the unchanged Great Kingdom Rules V2."""

from .config import V5Config
from .encoder import ENCODED_SHAPE, NUM_PLANES, encode_state
from .network import PolicyValueAuxNetwork

__all__ = ["V5Config", "ENCODED_SHAPE", "NUM_PLANES", "encode_state", "PolicyValueAuxNetwork"]
