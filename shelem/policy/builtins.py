from __future__ import annotations

import random

import numpy as np

from shelem.policy.base import ShelemPolicy


class RandomPolicy(ShelemPolicy):
    """Uniform-random over legal actions."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: dict, mask: np.ndarray) -> int:
        legal = mask.nonzero()[0].tolist()
        return self._rng.choice(legal)
