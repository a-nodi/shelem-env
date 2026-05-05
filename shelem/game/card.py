from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def label(self) -> str:
        special = {10: "10", 11: "J", 12: "Q", 13: "K", 14: "A"}
        return special.get(self.value, str(self.value))


class PlayMode(IntEnum):
    NORMAL = 0     # first lead sets trump (§5)
    NARES = 1      # reversed hierarchy, no trump (§5.1)
    ACE_NARES = 2  # A highest then 2>3>…>K, no trump (§5.1)
    SARRES = 3     # A-high, no trump (§5.1)


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def index(self) -> int:
        """0–51 action-space index (suit-major: 13 cards per suit)."""
        return self.suit.value * 13 + (self.rank.value - 2)

    def points(self, card_points: dict[str, int]) -> int:
        return card_points.get(self.rank.label, 0)

    def __repr__(self) -> str:
        symbols = {Suit.CLUBS: "♣", Suit.DIAMONDS: "♦", Suit.HEARTS: "♥", Suit.SPADES: "♠"}
        return f"{self.rank.label}{symbols[self.suit]}"


ALL_CARDS: list[Card] = [
    Card(rank=Rank(r + 2), suit=Suit(s))
    for s in range(4)
    for r in range(13)
]

INDEX_TO_CARD: dict[int, Card] = {c.index: c for c in ALL_CARDS}
