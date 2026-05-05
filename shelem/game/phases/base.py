from __future__ import annotations

from abc import ABC, abstractmethod

from shelem.game.state import GameState


class Phase(ABC):
    @abstractmethod
    def legal_actions(self, state: GameState, player: int) -> list[int]:
        """Return list of legal action indices for *player* in *state*."""

    @abstractmethod
    def apply(self, state: GameState, player: int, action: int) -> None:
        """Mutate *state* by applying *action* for *player*."""

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Return True when this phase is done and a transition should occur."""
