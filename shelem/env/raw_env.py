from __future__ import annotations

import copy
import random as _random

from shelem.config import ShelemConfig
from shelem.game.deck import make_deck
from shelem.game.phases.base import Phase
from shelem.game.phases.bidding import BiddingPhase
from shelem.game.phases.play import PlayPhase
from shelem.game.phases.trump_declaration import TrumpDeclarationPhase
from shelem.game.phases.zamin_exchange import ZaminExchangePhase
from shelem.game.scoring import apply_hand_result
from shelem.game.state import GameState, PhaseEnum
from shelem.utils.action_mask import compute_action_mask


class RawEnv:
    """
    Pure game logic with no PettingZoo dependency.
    Safe to copy.deepcopy() for MCTS rollouts.
    """

    _PHASE_HANDLERS: dict[PhaseEnum, Phase] = {
        PhaseEnum.BIDDING:           BiddingPhase(),
        PhaseEnum.ZAMIN_EXCHANGE:    ZaminExchangePhase(),
        PhaseEnum.TRUMP_DECLARATION: TrumpDeclarationPhase(),
        PhaseEnum.PLAY:              PlayPhase(),
    }

    def __init__(self, config: ShelemConfig | None = None) -> None:
        self.config = config or ShelemConfig.from_yaml()
        self.state: GameState | None = None
        self._rng = _random.Random()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, dealer: int = 0) -> GameState:
        if seed is not None:
            self._rng.seed(seed)

        state = self._deal(dealer)
        self.state = state
        return state

    def step(self, action: int) -> None:
        assert self.state is not None, "call reset() first"
        state = self.state
        player = state.current_agent
        handler = self._PHASE_HANDLERS[state.phase]

        handler.apply(state, player, action)

        if handler.is_terminal(state):
            self._transition(state)

    def legal_actions(self, player: int | None = None) -> list[int]:
        assert self.state is not None
        p = player if player is not None else self.state.current_agent
        return self._PHASE_HANDLERS[self.state.phase].legal_actions(self.state, p)

    def action_mask(self, player: int | None = None) -> list[bool]:
        assert self.state is not None
        p = player if player is not None else self.state.current_agent
        return compute_action_mask(self.state, p).tolist()

    def clone(self) -> RawEnv:
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deal(self, dealer: int) -> GameState:
        deck = make_deck(self.config.num_players)
        deck.shuffle(self._rng)

        hands: dict[int, set] = {}
        for i in range(self.config.num_players):
            hands[i] = set(deck.deal(self.config.hand_size))
        zamin = frozenset(deck.deal(self.config.zamin_size))

        first_bidder = (dealer + 1) % self.config.num_players

        # reset per-player bid_history lists
        bid_history: list[list] = [[] for _ in range(self.config.num_players)]
        passed = [False] * self.config.num_players

        state = GameState(
            config=self.config,
            hands=hands,
            zamin=zamin,
            dealer=dealer,
            current_agent=first_bidder,
            phase=PhaseEnum.BIDDING,
            bid_history=bid_history,
            passed=passed,
        )
        return state

    def _transition(self, state: GameState) -> None:
        current = state.phase

        if current == PhaseEnum.BIDDING:
            if state.void_hand:
                # re-deal with next dealer (§12 edge case)
                next_dealer = (state.dealer + 1) % self.config.num_players
                prev_scores = state.scores[:]
                prev_log = state.score_log[:]
                self.state = self._deal(next_dealer)
                self.state.scores = prev_scores
                self.state.score_log = prev_log
            else:
                state.phase = PhaseEnum.ZAMIN_EXCHANGE
                ZaminExchangePhase.setup(state)

        elif current == PhaseEnum.ZAMIN_EXCHANGE:
            state.phase = PhaseEnum.TRUMP_DECLARATION
            TrumpDeclarationPhase.setup(state)

        elif current == PhaseEnum.TRUMP_DECLARATION:
            state.phase = PhaseEnum.PLAY
            state.current_agent = state.declarer  # type: ignore[assignment]

        elif current == PhaseEnum.PLAY:
            apply_hand_result(state)
            if state.game_over:
                state.phase = PhaseEnum.SCORING
            else:
                # auto-deal next hand; SCORING phase is never "resting state"
                next_dealer = (state.dealer + 1) % self.config.num_players
                prev_scores = state.scores[:]
                prev_log = state.score_log[:]
                self.state = self._deal(next_dealer)
                self.state.scores = prev_scores
                self.state.score_log = prev_log
