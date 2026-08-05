"""Render palette shared by producers and pixel-only learners.

This module intentionally contains no simulator class, action table, physical
state, or transition function.  It is safe to include in the isolated learner
source bundle.
"""

from __future__ import annotations

import numpy as np


PALETTE = {
    "outside": np.asarray((1, 1, 6), dtype=np.uint8),
    "field": np.asarray((2, 6, 16), dtype=np.uint8),
    "line": np.asarray((7, 63, 156), dtype=np.uint8),
    "wall": np.asarray((21, 188, 228), dtype=np.uint8),
    "goal": np.asarray((255, 186, 50), dtype=np.uint8),
    "player": np.asarray((255, 91, 26), dtype=np.uint8),
    "player_core": np.asarray((255, 225, 115), dtype=np.uint8),
    "puck": np.asarray((8, 122, 199), dtype=np.uint8),
    "puck_core": np.asarray((183, 244, 255), dtype=np.uint8),
}


__all__ = ["PALETTE"]
