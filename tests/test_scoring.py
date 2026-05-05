"""Tests for scoring logic (§7.2 of shelem_rules.md)."""
from __future__ import annotations

import pytest

from shelem.config import ShelemConfig
from shelem.game.scoring import compute_hand_result
from shelem.game.state import GameState, PhaseEnum


def _make_state(
    *,
    declarer: int,
    bid: int,
    tricks_won: list[int],
    points_won: list[int],
    config: ShelemConfig | None = None,
) -> GameState:
    cfg = config or ShelemConfig.from_yaml()
    state = GameState(config=cfg)
    state.declarer = declarer
    state.current_bid = bid
    state.tricks_won = tricks_won[:]
    state.points_won = points_won[:]
    return state


def test_declarer_win():
    # declarer (team 0) earned 100 >= bid 85
    state = _make_state(declarer=0, bid=85, tricks_won=[7, 6], points_won=[100, 65])
    d0, d1 = compute_hand_result(state)
    assert d0 == 100
    assert d1 == 65


def test_declarer_fail():
    # earned 84 < bid 85; earned >= defenders (84 >= 81) → plain fail
    # trick pts: team0=6×5=30, team1=7×5=35 → card pts 54+46=100; total 84+81=165 ✓
    state = _make_state(declarer=0, bid=85, tricks_won=[6, 7], points_won=[84, 81])
    d0, d1 = compute_hand_result(state)
    assert d0 == -85
    assert d1 == 81


def test_declarer_doubled():
    # earned 60 < bid 85 AND earned < defenders (60 < 105) → doubled
    # trick pts: team0=5×5=25, team1=8×5=40 → card pts 35+65=100; total 60+105=165 ✓
    state = _make_state(declarer=0, bid=85, tricks_won=[5, 8], points_won=[60, 105])
    d0, d1 = compute_hand_result(state)
    assert d0 == -(2 * 85)
    assert d1 == 105


def test_shelem_bonus():
    cfg = ShelemConfig.from_yaml()
    # tricks_per_hand + 1 = 13 tricks total for declarer
    state = _make_state(
        declarer=0,
        bid=100,
        tricks_won=[13, 0],
        points_won=[165, 0],
        config=cfg,
    )
    d0, d1 = compute_hand_result(state)
    assert d0 == cfg.shelem_bonus
    assert d1 == 0


def test_declarer_team1_fail():
    # declarer is player 1 (team 1)
    # trick pts: team0=7×5=35, team1=6×5=30 → card pts 45+55=100; total 80+85=165 ✓
    # team1 earned 85 < bid 90 → fail; 85 >= 80 → plain fail
    state = _make_state(declarer=1, bid=90, tricks_won=[7, 6], points_won=[80, 85])
    d0, d1 = compute_hand_result(state)
    assert d1 == -90
    assert d0 == 80
