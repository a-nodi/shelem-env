from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ShelemPolicy(ABC):
    """Abstract base for all Shelem policies.

    Subclass this and implement ``act()`` to plug any model into
    ``watch()``, ``record()``, or the AEC env directly.

    To save per-step debug information (logits, value estimates, etc.)
    during ``record()``, also override ``act_with_info()``.
    """

    @abstractmethod
    def act(self, obs: dict, mask: np.ndarray) -> int:
        """Return a legal action index.

        Args:
            obs:  Observation dict from ``env.observe(agent)``.
            mask: ``np.ndarray`` shape ``(74,)`` dtype ``int8``;
                  1 = legal, 0 = illegal.

        Returns:
            Integer action in ``[0, 73]``.
        """

    def act_with_info(self, obs: dict, mask: np.ndarray) -> tuple[int, dict]:
        """Return ``(action, debug_info)`` for a single step.

        Override this to attach arbitrary debug data to each recorded step.
        The ``debug_info`` dict is stored in ``Episode.debug_info`` and
        serialised to JSON when ``save_episode()`` is called.

        Numpy arrays in the returned dict are automatically converted to
        lists during serialisation.

        Default implementation delegates to ``act()`` and returns ``{}``.

        Example::

            def act_with_info(self, obs, mask):
                flat   = flatten_obs(obs)
                logits = self.model(flat)
                value  = self.value_head(flat)
                action = int(np.argmax(logits * mask))
                return action, {
                    "logits": logits,          # np.ndarray → saved as list
                    "value":  float(value),
                    "entropy": float(-np.sum(
                        softmax(logits) * np.log(softmax(logits) + 1e-8)
                    )),
                }
        """
        return self.act(obs, mask), {}

    def __call__(self, obs: dict, mask: np.ndarray) -> int:
        return self.act(obs, mask)
