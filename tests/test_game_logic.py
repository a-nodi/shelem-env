"""Tests for core game logic: deal, bidding, Zamin exchange, play."""
from __future__ import annotations

import pytest

from shelem.config import ShelemConfig
from shelem.env.raw_env import RawEnv
from shelem.game.state import PhaseEnum
from shelem.spaces.action import ACTION_PASS, bid_to_action


@pytest.fixture
def raw():
    env = RawEnv(ShelemConfig.from_yaml())
    env.reset(seed=42)
    return env


def test_deal_hand_sizes(raw):
    state = raw.state
    assert all(len(state.hands[p]) == 12 for p in range(4))
    assert len(state.zamin) == 4


def test_total_cards_dealt(raw):
    state = raw.state
    total = sum(len(h) for h in state.hands.values()) + len(state.zamin)
    assert total == 52


def test_initial_phase_is_bidding(raw):
    assert raw.state.phase == PhaseEnum.BIDDING


def test_bidding_pass_all_voids_hand(raw):
    for _ in range(4):
        raw.step(ACTION_PASS)
    # after void: re-dealt, back in BIDDING
    assert raw.state.phase == PhaseEnum.BIDDING
    assert not raw.state.void_hand  # reset after re-deal


def test_bidding_single_winner(raw):
    cfg = raw.state.config
    raw.step(bid_to_action(cfg.min_bid, cfg))
    for _ in range(3):
        raw.step(ACTION_PASS)
    assert raw.state.phase == PhaseEnum.ZAMIN_EXCHANGE
    assert raw.state.declarer is not None
    assert raw.state.current_bid == cfg.min_bid


def test_zamin_exchange_gives_hakem_16_cards(raw):
    cfg = raw.state.config
    raw.step(bid_to_action(cfg.min_bid, cfg))
    for _ in range(3):
        raw.step(ACTION_PASS)

    hakem = raw.state.declarer
    assert len(raw.state.hands[hakem]) == 16


def test_zamin_exchange_leaves_12_cards(raw):
    cfg = raw.state.config
    raw.step(bid_to_action(cfg.min_bid, cfg))
    for _ in range(3):
        raw.step(ACTION_PASS)

    hakem = raw.state.declarer
    for _ in range(4):
        card_action = list(raw.state.hands[hakem])[0].index
        raw.step(card_action)

    assert raw.state.phase == PhaseEnum.TRUMP_DECLARATION
    assert len(raw.state.hands[hakem]) == 12


def test_full_hand_reaches_scoring(raw):
    """Play a complete hand with random-legal moves and reach SCORING."""
    import random
    rng = random.Random(0)
    cfg = raw.state.config

    # bidding
    raw.step(bid_to_action(cfg.min_bid, cfg))
    for _ in range(3):
        raw.step(ACTION_PASS)

    # zamin exchange
    hakem = raw.state.declarer
    for _ in range(cfg.zamin_discard_count):
        action = rng.choice(raw.legal_actions())
        raw.step(action)

    # trump declaration
    action = raw.legal_actions()[0]
    raw.step(action)

    prev_scores = raw.state.scores[:]

    # play all tricks
    while raw.state.phase == PhaseEnum.PLAY:
        action = rng.choice(raw.legal_actions())
        raw.step(action)

    # after hand: either SCORING (game over) or BIDDING (auto-redealt, scores updated)
    assert raw.state.phase in (PhaseEnum.SCORING, PhaseEnum.BIDDING)
    assert raw.state.scores != prev_scores, "scores must change after a completed hand"
