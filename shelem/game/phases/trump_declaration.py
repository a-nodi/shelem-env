from __future__ import annotations

from shelem.game.card import PlayMode
from shelem.game.phases.base import Phase
from shelem.game.state import GameState
from shelem.spaces.action import ACTION_MODE_OFFSET


class TrumpDeclarationPhase(Phase):
    def legal_actions(self, state: GameState, player: int) -> list[int]:
        # actions 70–73 map to PlayMode 0–3
        return [ACTION_MODE_OFFSET + m for m in range(len(PlayMode))]

    def apply(self, state: GameState, player: int, action: int) -> None:
        mode = PlayMode(action - ACTION_MODE_OFFSET)
        state.play_mode = mode
        # trump_suit stays None for non-NORMAL modes and will be set on first lead
        state.current_agent = state.declarer  # Hâkem leads first trick

    def is_terminal(self, state: GameState) -> bool:
        # terminal immediately after the single declaration action
        return state.play_mode is not None and state.phase.value >= 2

    @staticmethod
    def setup(state: GameState) -> None:
        """Called on phase entry: credit zamin discard points, set Hâkem as actor."""
        assert state.declarer is not None
        team = state.config.team_of(state.declarer)
        # deduct zamin cards from hand and add their points to declarer team (§4.3)
        for card in state.zamin_discards:
            state.hands[state.declarer].discard(card)
            state.points_won[team] += card.points(state.config.card_points)
        state.current_agent = state.declarer
        state.trick_leader = state.declarer
