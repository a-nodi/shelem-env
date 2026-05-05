from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

from shelem.game.card import Card, PlayMode, Suit

if TYPE_CHECKING:
    from shelem.config import ShelemConfig


class PhaseEnum(IntEnum):
    BIDDING = 0
    ZAMIN_EXCHANGE = 1
    TRUMP_DECLARATION = 2
    PLAY = 3
    SCORING = 4


@dataclass
class GameState:
    config: ShelemConfig

    # --- deal ---
    hands: dict[int, set[Card]] = field(default_factory=dict)
    zamin: frozenset[Card] = field(default_factory=frozenset)
    dealer: int = 0

    # --- current actor ---
    current_agent: int = 0

    # --- phase ---
    phase: PhaseEnum = PhaseEnum.BIDDING

    # --- bidding ---
    bid_history: list[list[int | None]] = field(
        default_factory=lambda: [[], [], [], []]
    )
    current_bid: int = 0
    passed: list[bool] = field(default_factory=lambda: [False, False, False, False])
    any_bid_made: bool = False
    consecutive_passes: int = 0
    declarer: int | None = None

    # --- zamin exchange ---
    zamin_discards: set[Card] = field(default_factory=set)
    zamin_taken: bool = False
    zamin_selected_count: int = 0

    # --- trump declaration ---
    play_mode: PlayMode = PlayMode.NORMAL
    trump_suit: Suit | None = None

    # --- play ---
    current_trick: list[tuple[int, Card]] = field(default_factory=list)
    trick_leader: int = 0
    tricks_won: list[int] = field(default_factory=lambda: [0, 0])
    points_won: list[int] = field(default_factory=lambda: [0, 0])
    completed_tricks: int = 0

    # --- game ---
    scores: list[int] = field(default_factory=lambda: [0, 0])
    hand_over: bool = False
    game_over: bool = False
    void_hand: bool = False  # True when all 4 passed without a bid

    def active_bidders(self) -> list[int]:
        return [p for p in range(self.config.num_players) if not self.passed[p]]
