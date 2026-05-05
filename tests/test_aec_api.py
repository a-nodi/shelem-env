"""PettingZoo AEC API conformance tests."""
from __future__ import annotations

import pytest
from pettingzoo.test import api_test

from shelem.env.shelem_aec import ShelemAECEnv
from shelem.config import ShelemConfig


def test_aec_api_conformance():
    env = ShelemAECEnv(ShelemConfig.from_yaml())
    api_test(env, num_cycles=100, verbose_progress=False)


def test_reset_gives_active_agents():
    env = ShelemAECEnv()
    env.reset(seed=0)
    assert len(env.agents) == 4
    assert env.agent_selection in env.agents


def test_observe_returns_dict_with_action_mask():
    env = ShelemAECEnv()
    env.reset(seed=0)
    obs = env.observe(env.agent_selection)
    assert "action_mask" in obs
    assert obs["action_mask"].shape[0] == 74


def test_step_advances_agent():
    env = ShelemAECEnv()
    env.reset(seed=1)
    first = env.agent_selection
    legal = env.action_mask(first).nonzero()[0]
    env.step(int(legal[0]))
    # bidding: one action always moves to the next eligible player
    assert env.agent_selection != first, (
        f"agent_selection should advance after a bidding step, got {env.agent_selection}"
    )


def test_full_episode_terminates():
    import random
    rng = random.Random(99)
    # low threshold so random play can reach it in reasonable steps
    cfg = ShelemConfig.from_yaml()
    cfg_low = ShelemConfig(
        num_players=cfg.num_players,
        hand_size=cfg.hand_size,
        zamin_size=cfg.zamin_size,
        zamin_discard_count=cfg.zamin_discard_count,
        card_points=cfg.card_points,
        min_bid=cfg.min_bid,
        max_bid=cfg.max_bid,
        bid_increment=cfg.bid_increment,
        shelem_bonus=cfg.shelem_bonus,
        game_threshold=50,          # terminate quickly (positive or negative)
    )
    env = ShelemAECEnv(config=cfg_low)
    env.reset(seed=99)

    max_steps = 50_000
    for _ in range(max_steps):
        if not env.agents:
            break
        agent = env.agent_selection
        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
            continue
        mask = env.action_mask(agent)
        legal = mask.nonzero()[0].tolist()
        action = rng.choice(legal)
        env.step(action)

    assert all(env.terminations.values()) or not env.agents
