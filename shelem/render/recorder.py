from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv
from shelem.policy.builtins import RandomPolicy
from shelem.render.terminal import watch

if TYPE_CHECKING:
    from shelem.policy.base import ShelemPolicy


@dataclass
class Episode:
    """Full record of one game episode.

    ``actions`` is the only field needed to replay: given the same
    ``seed`` and ``config``, the env is deterministic and will produce
    the identical sequence of states.

    ``debug_info`` holds one dict per step (same length as ``actions``),
    populated when the policy implements ``act_with_info()``.
    Numpy arrays inside the dicts are converted to plain lists on save.
    """

    actions: list[tuple[str, int]]        # (agent_name, action_index)
    seed: int | None
    config: dict                           # ShelemConfig serialised as dict
    final_scores: list[int] = field(default_factory=lambda: [0, 0])
    winner: int = -1                       # 0 or 1 (team index)
    total_steps: int = 0
    debug_info: list[dict] = field(default_factory=list)  # per-step model output


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record(
    policy: ShelemPolicy | None = None,
    policies: dict[str, ShelemPolicy] | None = None,
    config: ShelemConfig | None = None,
    seed: int | None = None,
) -> Episode:
    """Play a full game silently and return the recorded :class:`Episode`.

    Args:
        policy:   Single policy used for **all** agents.
        policies: Per-agent override, e.g.
                  ``{"player_0": my_model, "player_2": my_model}``.
                  Takes priority over ``policy`` when the agent key exists.
        config:   ``ShelemConfig`` instance.  ``None`` = ``default.yaml``.
        seed:     RNG seed for reproducibility.

    Example::

        from shelem.policy import load_policy
        from shelem.render.recorder import record, save_episode

        policy = load_policy("models/ppo.zip")
        ep = record(policy, seed=42)
        save_episode(ep, "replays/game_001.json")
    """
    cfg = config or ShelemConfig.from_yaml()
    env = ShelemAECEnv(config=cfg)
    env.reset(seed=seed)

    _default = policy or RandomPolicy(seed=seed)
    actions: list[tuple[str, int]] = []
    debug_info: list[dict] = []

    while env.agents:
        agent = env.agent_selection

        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
            continue

        obs  = env.observe(agent)
        mask = obs["action_mask"]

        p = (policies or {}).get(agent, _default)
        action, info = p.act_with_info(obs, mask)
        action = int(action)

        actions.append((agent, action))
        debug_info.append(info)
        env.step(action)

    state = env._raw.state
    scores = state.scores[:]
    winner = 0 if scores[0] >= cfg.game_threshold else 1

    return Episode(
        actions=actions,
        seed=seed,
        config=_cfg_to_dict(cfg),
        final_scores=scores,
        winner=winner,
        total_steps=len(actions),
        debug_info=debug_info,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(
    episode: Episode,
    delay: float = 0.5,
    show_all_hands: bool = True,
) -> None:
    """Re-run a recorded episode in the terminal renderer.

    The env is reset with the same seed so the dealing is identical,
    then the saved actions are fed back in order.

    Args:
        episode:        A previously recorded :class:`Episode`.
        delay:          Seconds to pause between steps.
        show_all_hands: Show all 4 players' cards (default ``True``).

    Example::

        from shelem.render.recorder import load_episode, replay

        ep = load_episode("replays/game_001.json")
        replay(ep, delay=0.3)
    """
    cfg = ShelemConfig(**episode.config)
    action_iter = iter(episode.actions)

    def _replay_policy(obs, mask):
        try:
            _, action = next(action_iter)
            return action
        except StopIteration:
            return int(mask.nonzero()[0][0])

    watch(
        policy=_replay_policy,
        config=cfg,
        seed=episode.seed,
        delay=delay,
        show_all_hands=show_all_hands,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_episode(episode: Episode, path: str | Path) -> None:
    """Save an episode to a JSON file.

    Args:
        episode: The episode to save.
        path:    Destination path (parent directories created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "seed":         episode.seed,
        "config":       episode.config,
        "final_scores": episode.final_scores,
        "winner":       episode.winner,
        "total_steps":  episode.total_steps,
        "actions":      [[a, c] for a, c in episode.actions],
        "debug_info":   [_json_safe(info) for info in episode.debug_info],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_episode(path: str | Path) -> Episode:
    """Load an episode from a JSON file.

    Args:
        path: Path to a file previously created by :func:`save_episode`.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Episode(
        actions=[(a, int(c)) for a, c in data["actions"]],
        seed=data["seed"],
        config=data["config"],
        final_scores=data["final_scores"],
        winner=data["winner"],
        total_steps=data["total_steps"],
        debug_info=data.get("debug_info", []),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _cfg_to_dict(cfg) -> dict:
    return {
        "num_players":         cfg.num_players,
        "hand_size":           cfg.hand_size,
        "zamin_size":          cfg.zamin_size,
        "zamin_discard_count": cfg.zamin_discard_count,
        "card_points":         cfg.card_points,
        "min_bid":             cfg.min_bid,
        "max_bid":             cfg.max_bid,
        "bid_increment":       cfg.bid_increment,
        "shelem_bonus":        cfg.shelem_bonus,
        "game_threshold":      cfg.game_threshold,
    }
