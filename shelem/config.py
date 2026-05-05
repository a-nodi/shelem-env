from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "default.yaml"


@dataclass
class ShelemConfig:
    num_players: int
    hand_size: int
    zamin_size: int
    zamin_discard_count: int
    card_points: dict[str, int]
    min_bid: int
    max_bid: int
    bid_increment: int
    shelem_bonus: int
    game_threshold: int

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> ShelemConfig:
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    @property
    def bid_values(self) -> list[int]:
        return list(range(self.min_bid, self.max_bid + 1, self.bid_increment))

    @property
    def tricks_per_hand(self) -> int:
        return self.hand_size

    def team_of(self, player: int) -> int:
        """Team 0 = players {0, 2}, Team 1 = players {1, 3}."""
        return player % 2
