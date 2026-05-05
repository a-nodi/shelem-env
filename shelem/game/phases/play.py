from __future__ import annotations

from shelem.game.card import INDEX_TO_CARD, PlayMode, Rank, Suit
from shelem.game.phases.base import Phase
from shelem.game.state import GameState


class PlayPhase(Phase):
    def legal_actions(self, state: GameState, player: int) -> list[int]:
        hand = state.hands[player]

        # first lead of the hand: any card (§11 rule 3)
        if not state.current_trick and state.completed_tricks == 0:
            return [c.index for c in hand]

        # leader may play any card
        if not state.current_trick:
            return [c.index for c in hand]

        led_suit = state.current_trick[0][1].suit
        same_suit = [c for c in hand if c.suit == led_suit]
        legal = same_suit if same_suit else list(hand)  # must-follow (§6.2)
        return [c.index for c in legal]

    def apply(self, state: GameState, player: int, action: int) -> None:
        card = INDEX_TO_CARD[action]
        state.hands[player].discard(card)
        state.current_trick.append((player, card))

        # set trump implicitly on Hâkem's first lead in NORMAL mode (§5)
        if (
            state.play_mode == PlayMode.NORMAL
            and state.trump_suit is None
            and player == state.declarer
            and state.completed_tricks == 0
            and len(state.current_trick) == 1
        ):
            state.trump_suit = card.suit

        if len(state.current_trick) == state.config.num_players:
            _resolve_trick(state)
        else:
            state.current_agent = (player + 1) % state.config.num_players

    def is_terminal(self, state: GameState) -> bool:
        return state.completed_tricks >= state.config.tricks_per_hand


def _resolve_trick(state: GameState) -> None:
    winner = _trick_winner(state.current_trick, state)
    team = state.config.team_of(winner)
    state.tricks_won[team] += 1
    state.points_won[team] += sum(
        c.points(state.config.card_points) for _, c in state.current_trick
    )
    # 5 points per trick (§7.1)
    state.points_won[team] += 5

    state.completed_tricks += 1
    state.current_trick = []
    state.trick_leader = winner
    state.current_agent = winner


def _trick_winner(trick: list[tuple[int, Card]], state: GameState) -> int:
    led_suit = trick[0][1].suit
    mode = state.play_mode

    if mode == PlayMode.NORMAL:
        trump = state.trump_suit
        trump_cards = [(p, c) for p, c in trick if trump and c.suit == trump]
        if trump_cards:
            winner_p, _ = max(trump_cards, key=lambda x: x[1].rank.value)
        else:
            led_cards = [(p, c) for p, c in trick if c.suit == led_suit]
            winner_p, _ = max(led_cards, key=lambda x: x[1].rank.value)

    elif mode == PlayMode.NARES:
        # reversed hierarchy: 2 highest, A lowest (§5.1)
        led_cards = [(p, c) for p, c in trick if c.suit == led_suit]
        winner_p, _ = min(led_cards, key=lambda x: x[1].rank.value)

    elif mode == PlayMode.ACE_NARES:
        # A > 2 > 3 > … > K (§5.1)
        led_cards = [(p, c) for p, c in trick if c.suit == led_suit]
        winner_p, _ = max(led_cards, key=lambda x: _ace_nares_strength(x[1].rank))

    else:  # SARRES: normal A-high, no trump (§5.1)
        led_cards = [(p, c) for p, c in trick if c.suit == led_suit]
        winner_p, _ = max(led_cards, key=lambda x: x[1].rank.value)

    return winner_p


def _ace_nares_strength(rank: Rank) -> int:
    if rank == Rank.ACE:
        return 14
    # 2→12, 3→11, …, K→1
    return 13 - rank.value
