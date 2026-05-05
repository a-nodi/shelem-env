from __future__ import annotations

from shelem.game.phases.base import Phase
from shelem.game.state import GameState
from shelem.spaces.action import ACTION_PASS, bid_to_action, action_to_bid, BID_OFFSET


class BiddingPhase(Phase):
    def legal_actions(self, state: GameState, player: int) -> list[int]:
        actions = [ACTION_PASS]
        if not state.passed[player]:
            for val in state.config.bid_values:
                if val > state.current_bid:
                    actions.append(bid_to_action(val, state.config))
        return actions

    def apply(self, state: GameState, player: int, action: int) -> None:
        if action == ACTION_PASS:
            state.passed[player] = True
            state.consecutive_passes += 1
            state.bid_history[player].append(None)
        else:
            bid_val = action_to_bid(action, state.config)
            state.current_bid = bid_val
            state.declarer = player
            state.any_bid_made = True
            state.consecutive_passes = 0
            state.bid_history[player].append(bid_val)

        # advance to next eligible bidder counter-clockwise
        state.current_agent = _next_bidder(state, player)

    def is_terminal(self, state: GameState) -> bool:
        if all(state.passed):
            state.void_hand = True
            return True
        # bidding ends after 3 consecutive passes following a bid
        if state.any_bid_made and state.consecutive_passes >= 3:
            return True
        # only one active bidder left
        if state.any_bid_made and len(state.active_bidders()) == 1:
            state.declarer = state.active_bidders()[0]
            return True
        return False


def _next_bidder(state: GameState, current: int) -> int:
    """Counter-clockwise next eligible (non-passed) bidder."""
    n = state.config.num_players
    for i in range(1, n + 1):
        candidate = (current + i) % n
        if not state.passed[candidate]:
            return candidate
    return current  # all passed; terminal check will catch this
