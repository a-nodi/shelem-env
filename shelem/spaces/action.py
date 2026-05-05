from __future__ import annotations

from typing import TYPE_CHECKING

from shelem.game.card import PlayMode

if TYPE_CHECKING:
    from shelem.config import ShelemConfig

# --- action index layout (Discrete(74)) ---
# 0–51   card index  (play / zamin discard selection)
# 52     PASS        (bidding only)
# 53–69  bid values  85, 90, … 165  (up to 17 values; range driven by config)
# 70–73  play mode   NORMAL, NARES, ACE_NARES, SARRES

ACTION_PASS: int = 52
BID_OFFSET: int = 53
ACTION_MODE_OFFSET: int = 70
ACTION_SPACE_SIZE: int = 74  # 52 cards + PASS + 17 bids + 4 modes


def bid_to_action(bid_value: int, config: ShelemConfig) -> int:
    idx = config.bid_values.index(bid_value)
    return BID_OFFSET + idx


def action_to_bid(action: int, config: ShelemConfig) -> int:
    return config.bid_values[action - BID_OFFSET]


def action_to_play_mode(action: int) -> PlayMode:
    return PlayMode(action - ACTION_MODE_OFFSET)


def is_card_action(action: int) -> bool:
    return 0 <= action <= 51


def is_bid_action(action: int) -> bool:
    return BID_OFFSET <= action < ACTION_MODE_OFFSET


def is_mode_action(action: int) -> bool:
    return ACTION_MODE_OFFSET <= action < ACTION_SPACE_SIZE
