from __future__ import annotations

from shelem.game.card import INDEX_TO_CARD
from shelem.game.phases.base import Phase
from shelem.game.state import GameState


class ZaminExchangePhase(Phase):
    def legal_actions(self, state: GameState, player: int) -> list[int]:
        return [card.index for card in state.hands[player]]

    def apply(self, state: GameState, player: int, action: int) -> None:
        card = INDEX_TO_CARD[action]
        state.hands[player].discard(card)
        state.zamin_discards.add(card)
        state.zamin_selected_count += 1
        # current_agent stays the same (Hâkem selects all 4 cards)

    def is_terminal(self, state: GameState) -> bool:
        return state.zamin_selected_count >= state.config.zamin_discard_count

    @staticmethod
    def setup(state: GameState) -> None:
        """Called on phase entry: give Hâkem all zamin cards."""
        assert state.declarer is not None
        state.hands[state.declarer].update(state.zamin)
        state.zamin_taken = True
        state.current_agent = state.declarer

        # credit zamin trick points to declarer team (§4.3)
        team = state.config.team_of(state.declarer)
        state.tricks_won[team] += 1
        state.points_won[team] += sum(
            c.points(state.config.card_points) for c in state.zamin
        )
