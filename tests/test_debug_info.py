import random

import numpy as np
from shelem.config import ShelemConfig
from shelem.policy import ShelemPolicy
from shelem.render.recorder import load_episode, record, save_episode


class DebugPolicy(ShelemPolicy):
    """Random policy that also returns debug info."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, obs, mask):
        return self._rng.choice(mask.nonzero()[0].tolist())

    def act_with_info(self, obs, mask):
        action = self.act(obs, mask)
        return action, {
            "logits": np.zeros(74, dtype=np.float32),
            "value": 0.5,
            "phase": int(obs["phase"]),
        }


def _fast_cfg():
    cfg = ShelemConfig.from_yaml()
    return ShelemConfig(
        num_players=cfg.num_players, hand_size=cfg.hand_size,
        zamin_size=cfg.zamin_size, zamin_discard_count=cfg.zamin_discard_count,
        card_points=cfg.card_points, min_bid=cfg.min_bid, max_bid=cfg.max_bid,
        bid_increment=cfg.bid_increment, shelem_bonus=cfg.shelem_bonus,
        game_threshold=50,
    )


def test_debug_info_length():
    ep = record(policy=DebugPolicy(), config=_fast_cfg(), seed=1)
    assert len(ep.debug_info) == ep.total_steps


def test_debug_info_keys():
    ep = record(policy=DebugPolicy(), config=_fast_cfg(), seed=2)
    assert "logits" in ep.debug_info[0]
    assert "value" in ep.debug_info[0]
    assert "phase" in ep.debug_info[0]


def test_debug_info_json_roundtrip(tmp_path):
    ep = record(policy=DebugPolicy(), config=_fast_cfg(), seed=3)
    path = tmp_path / "ep.json"
    save_episode(ep, path)
    ep2 = load_episode(path)
    assert len(ep2.debug_info) == ep.total_steps
    # numpy arrays converted to list on save
    assert isinstance(ep2.debug_info[0]["logits"], list)
    assert len(ep2.debug_info[0]["logits"]) == 74
    assert ep2.debug_info[0]["value"] == 0.5


def test_no_debug_info_by_default():
    from shelem.policy import RandomPolicy
    ep = record(policy=RandomPolicy(seed=0), config=_fast_cfg(), seed=0)
    assert len(ep.debug_info) == ep.total_steps
    assert ep.debug_info[0] == {}
