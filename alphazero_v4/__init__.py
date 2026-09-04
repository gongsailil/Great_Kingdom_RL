"""AlphaZero V4 stability architecture on unchanged Great Kingdom Rules V2."""

from .config import V4Config
from .network import PolicyValueLogitNetwork

__all__ = ["PolicyValueLogitNetwork", "V4Config"]
