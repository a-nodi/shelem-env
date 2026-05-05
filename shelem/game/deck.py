from __future__ import annotations

import random as _random

from shelem.game.card import ALL_CARDS, Card, Rank, Suit


class Deck:
    def __init__(self, exclude: set[Card] | None = None) -> None:
        self._cards: list[Card] = [
            c for c in ALL_CARDS if exclude is None or c not in exclude
        ]

    def shuffle(self, rng: _random.Random | None = None) -> None:
        (rng or _random).shuffle(self._cards)

    def deal(self, n: int) -> list[Card]:
        if n > len(self._cards):
            raise ValueError(f"Cannot deal {n} from {len(self._cards)} remaining")
        dealt, self._cards = self._cards[:n], self._cards[n:]
        return dealt

    def __len__(self) -> int:
        return len(self._cards)


def make_deck(num_players: int) -> Deck:
    """Return a deck appropriate for the player count (§9.2: 3-player removes 2♣)."""
    if num_players == 3:
        return Deck(exclude={Card(rank=Rank.TWO, suit=Suit.CLUBS)})
    return Deck()
