from __future__ import annotations

import numpy as np
from gymnasium import spaces

from shelem.game.state import GameState
from shelem.spaces.action import ACTION_SPACE_SIZE
from shelem.utils.action_mask import compute_action_mask


def build_observation_space() -> spaces.Dict:
    return spaces.Dict({
        "hand":           spaces.MultiBinary(52),
        "played_cards":   spaces.MultiBinary(52),
        "current_trick":  spaces.Box(-1, 51, shape=(4,), dtype=np.int8),
        "trick_leader":   spaces.Discrete(4),
        "trump_suit":     spaces.Discrete(5),        # 0=unknown, 1–4
        "play_mode":      spaces.Discrete(4),
        "phase":          spaces.Discrete(5),
        "declarer":       spaces.Discrete(5),        # 0=unknown, 1–4
        "bid":            spaces.Discrete(166),      # 0=not yet set, max bid=165
        "bid_history":    spaces.Box(0, 165, shape=(4, 20), dtype=np.int16),
        "zamin_taken":    spaces.Discrete(2),
        "tricks_won":     spaces.Box(0, 13,  shape=(2,), dtype=np.int8),
        "points_won":     spaces.Box(0, 200, shape=(2,), dtype=np.int16),
        "action_mask":    spaces.MultiBinary(ACTION_SPACE_SIZE),
    })


class ObservationBuilder:
    @staticmethod
    def build(state: GameState, player: int) -> dict:
        hand_vec = np.zeros(52, dtype=np.int8)
        for card in state.hands.get(player, []):
            hand_vec[card.index] = 1

        played_vec = np.zeros(52, dtype=np.int8)
        # cards in current trick are visible
        trick_vec = np.full(4, -1, dtype=np.int8)
        for i, (_, card) in enumerate(state.current_trick):
            trick_vec[i] = card.index
            played_vec[card.index] = 1

        bid_hist = np.zeros((4, 20), dtype=np.int16)
        for p, history in enumerate(state.bid_history):
            for r, val in enumerate(history[:20]):
                bid_hist[p, r] = val if val is not None else 0

        return {
            "hand":          hand_vec,
            "played_cards":  played_vec,
            "current_trick": trick_vec,
            "trick_leader":  state.trick_leader,
            "trump_suit":    (state.trump_suit.value + 1) if state.trump_suit is not None else 0,
            "play_mode":     int(state.play_mode),
            "phase":         int(state.phase),
            "declarer":      (state.declarer + 1) if state.declarer is not None else 0,
            "bid":           state.current_bid,
            "bid_history":   bid_hist,
            "zamin_taken":   int(state.zamin_taken),
            "tricks_won":    np.array(state.tricks_won, dtype=np.int8),
            "points_won":    np.array(state.points_won, dtype=np.int16),
            "action_mask":   compute_action_mask(state, player),
        }
