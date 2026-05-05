from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from shelem.policy.base import ShelemPolicy
from shelem.policy.builtins import RandomPolicy

# Flat obs dimension:
#   hand(52) + played_cards(52) + current_trick(4)
#   + scalars(6) + bid_history(80) + tricks_won(2) + points_won(2) = 198
OBS_DIM = 198


def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten the observation dict to a 1-D float32 array (shape: (198,)).

    Use this when your model expects a flat vector input (e.g. SB3, ONNX).
    """
    return np.concatenate([
        obs["hand"].astype(np.float32),
        obs["played_cards"].astype(np.float32),
        obs["current_trick"].astype(np.float32),
        np.array([
            obs["trick_leader"],
            obs["trump_suit"],
            obs["play_mode"],
            obs["phase"],
            obs["declarer"],
            obs["bid"],
        ], dtype=np.float32),
        obs["bid_history"].flatten().astype(np.float32),
        obs["tricks_won"].astype(np.float32),
        obs["points_won"].astype(np.float32),
    ])


# ---------------------------------------------------------------------------
# Concrete wrappers
# ---------------------------------------------------------------------------

class _CallablePolicy(ShelemPolicy):
    """Wraps any callable ``(obs, mask) -> int`` as a ShelemPolicy."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    def act(self, obs: dict, mask: np.ndarray) -> int:
        return int(self._fn(obs, mask))


class SB3Policy(ShelemPolicy):
    """Wraps a Stable-Baselines3 model (.zip).

    The model must have been trained on a flat obs vector of size 198.
    Illegal actions are re-masked to the first legal action as a safety net.

    Requires: ``pip install stable-baselines3``
    """

    def __init__(self, model_path: str | Path, algorithm: str = "auto") -> None:
        try:
            import stable_baselines3 as sb3
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required for SB3Policy.\n"
                "Install it with: pip install stable-baselines3"
            )
        path = str(model_path)
        if algorithm == "auto":
            # try common algorithms in order
            for algo_name in ("PPO", "A2C", "DQN", "SAC", "TD3"):
                cls = getattr(sb3, algo_name, None)
                if cls is None:
                    continue
                try:
                    self._model = cls.load(path)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(
                    f"Could not auto-detect SB3 algorithm for '{path}'. "
                    "Pass algorithm='PPO' (or whichever you used) explicitly."
                )
        else:
            cls = getattr(sb3, algorithm)
            self._model = cls.load(path)

    def act(self, obs: dict, mask: np.ndarray) -> int:
        flat = flatten_obs(obs)
        action, _ = self._model.predict(flat, deterministic=True)
        action = int(action)
        if not mask[action]:
            action = int(mask.nonzero()[0][0])
        return action


class OnnxPolicy(ShelemPolicy):
    """Wraps an ONNX model (.onnx).

    Expects a single input of shape ``(1, 198)`` float32 and
    a single output of shape ``(1, 74)`` (action logits).

    Requires: ``pip install onnxruntime``
    """

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for OnnxPolicy.\n"
                "Install it with: pip install onnxruntime"
            )
        self._session = ort.InferenceSession(str(model_path))
        self._input_name = self._session.get_inputs()[0].name

    def act(self, obs: dict, mask: np.ndarray) -> int:
        flat = flatten_obs(obs)[None].astype(np.float32)
        logits = self._session.run(None, {self._input_name: flat})[0][0]
        logits[mask == 0] = -1e9
        return int(np.argmax(logits))


class TorchPolicy(ShelemPolicy):
    """Wraps a PyTorch model (.pt / .pth).

    Expects ``model.forward(x)`` where x is ``(1, 198)`` float32 tensor.
    Output must be ``(1, 74)`` logits.

    Requires: ``pip install torch``
    """

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for TorchPolicy.\n"
                "Install it with: pip install torch"
            )
        self._torch = torch
        self._device = device
        self._model = torch.load(str(model_path), map_location=device)
        self._model.eval()

    def act(self, obs: dict, mask: np.ndarray) -> int:
        flat = flatten_obs(obs)
        x = self._torch.tensor(
            flat, dtype=self._torch.float32, device=self._device
        ).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(x)[0].cpu().numpy()
        logits[mask == 0] = -1e9
        return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_policy(source=None, **kwargs) -> ShelemPolicy:
    """Load a Shelem policy from various sources.

    Args:
        source: One of the following:
            ``None`` or ``"random"``  → :class:`RandomPolicy`
            :class:`ShelemPolicy`     → returned as-is
            ``callable(obs, mask)``   → wrapped in a thin adapter
            path ending in ``.zip``   → :class:`SB3Policy`
            path ending in ``.onnx``  → :class:`OnnxPolicy`
            path ending in ``.pt`` / ``.pth`` → :class:`TorchPolicy`
        **kwargs: Forwarded to the policy constructor.

    Examples::

        policy = load_policy()                          # random
        policy = load_policy("models/ppo_shelem.zip")  # SB3
        policy = load_policy("models/actor.onnx")      # ONNX
        policy = load_policy("models/net.pt", device="cuda")  # PyTorch

        # Custom callable
        policy = load_policy(lambda obs, mask: mask.nonzero()[0][0])
    """
    if source is None or source == "random":
        return RandomPolicy(**kwargs)

    if isinstance(source, ShelemPolicy):
        return source

    if callable(source):
        return _CallablePolicy(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".zip":
        return SB3Policy(path, **kwargs)
    if suffix == ".onnx":
        return OnnxPolicy(path, **kwargs)
    if suffix in (".pt", ".pth"):
        return TorchPolicy(path, **kwargs)

    raise ValueError(
        f"Unknown model format '{suffix}'. "
        "Supported: .zip (SB3), .onnx (ONNX), .pt/.pth (PyTorch)"
    )
