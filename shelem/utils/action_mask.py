from __future__ import annotations

import numpy as np

from shelem.game.card import PlayMode
from shelem.game.state import GameState, PhaseEnum
from shelem.spaces.action import (
    ACTION_MODE_OFFSET,
    ACTION_PASS,
    ACTION_SPACE_SIZE,
    BID_OFFSET,
    bid_to_action,
)


def compute_action_mask(state: GameState, player: int) -> np.ndarray:
    """Return bool array shape (ACTION_SPACE_SIZE,) with True for legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

    if state.phase == PhaseEnum.BIDDING:
        mask[ACTION_PASS] = True
        if not state.passed[player]:
            for val in state.config.bid_values:
                if val > state.current_bid:
                    mask[bid_to_action(val, state.config)] = True

    elif state.phase == PhaseEnum.ZAMIN_EXCHANGE:
        if player == state.declarer:
            for card in state.hands[player]:
                mask[card.index] = True

    elif state.phase == PhaseEnum.TRUMP_DECLARATION:
        if player == state.declarer:
            for m in range(len(PlayMode)):
                mask[ACTION_MODE_OFFSET + m] = True

    elif state.phase == PhaseEnum.PLAY:
        hand = state.hands[player]
        if not state.current_trick:
            legal = list(hand)
        else:
            led_suit = state.current_trick[0][1].suit
            same_suit = [c for c in hand if c.suit == led_suit]
            legal = same_suit if same_suit else list(hand)
        for card in legal:
            mask[card.index] = True

    return mask
