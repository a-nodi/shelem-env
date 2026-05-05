"""Tests for action mask correctness."""
from __future__ import annotations

import pytest

from shelem.config import ShelemConfig
from shelem.env.raw_env import RawEnv
from shelem.game.state import PhaseEnum
from shelem.spaces.action import (
    ACTION_MODE_OFFSET,
    ACTION_PASS,
    ACTION_SPACE_SIZE,
    BID_OFFSET,
    bid_to_action,
)
from shelem.utils.action_mask import compute_action_mask


@pytest.fixture
def raw():
    env = RawEnv(ShelemConfig.from_yaml())
    env.reset(seed=7)
    return env


def test_bidding_mask_includes_pass(raw):
    player = raw.state.current_agent
    mask = compute_action_mask(raw.state, player)
    assert mask[ACTION_PASS]


def test_bidding_mask_only_higher_bids(raw):
    cfg = raw.state.config

    # bid 100 first so current_bid > 0; now both branches of the loop are reachable
    raw.step(bid_to_action(100, cfg))

    state = raw.state
    player = state.current_agent
    mask = compute_action_mask(state, player)

    for i, val in enumerate(cfg.bid_values):
        if val > state.current_bid:          # > 100 → legal
            assert mask[BID_OFFSET + i], f"bid {val} should be legal"
        else:                                 # ≤ 100 → masked
            assert not mask[BID_OFFSET + i], f"bid {val} should not be legal"


def test_play_mask_must_follow():
    """If player has the led suit, only that suit is legal."""
    from shelem.game.card import Card, Rank, Suit
    from shelem.game.state import GameState, PhaseEnum

    cfg = ShelemConfig.from_yaml()
    state = GameState(config=cfg)
    state.phase = PhaseEnum.PLAY
    state.trump_suit = Suit.SPADES

    # player 0 has 2♣ and 3♥; led suit is ♣
    c1 = Card(Rank.TWO, Suit.CLUBS)
    c2 = Card(Rank.THREE, Suit.HEARTS)
    state.hands = {0: {c1, c2}}
    state.current_trick = [(1, Card(Rank.FOUR, Suit.CLUBS))]  # led ♣

    mask = compute_action_mask(state, 0)
    assert mask[c1.index]       # 2♣ legal (led suit)
    assert not mask[c2.index]   # 3♥ not legal (has ♣)


def test_play_mask_void_suit_all_legal():
    """If player has no led suit, any card is legal."""
    from shelem.game.card import Card, Rank, Suit
    from shelem.game.state import GameState, PhaseEnum

    cfg = ShelemConfig.from_yaml()
    state = GameState(config=cfg)
    state.phase = PhaseEnum.PLAY

    c1 = Card(Rank.TWO, Suit.HEARTS)
    c2 = Card(Rank.THREE, Suit.SPADES)
    state.hands = {0: {c1, c2}}
    state.current_trick = [(1, Card(Rank.FOUR, Suit.CLUBS))]  # led ♣

    mask = compute_action_mask(state, 0)
    assert mask[c1.index]
    assert mask[c2.index]


def test_trump_declaration_mask(raw):
    cfg = raw.state.config
    raw.step(bid_to_action(cfg.min_bid, cfg))
    for _ in range(3):
        raw.step(ACTION_PASS)
    for _ in range(cfg.zamin_discard_count):
        raw.step(raw.legal_actions()[0])

    assert raw.state.phase == PhaseEnum.TRUMP_DECLARATION
    mask = compute_action_mask(raw.state, raw.state.declarer)
    for m in range(4):
        assert mask[ACTION_MODE_OFFSET + m]
