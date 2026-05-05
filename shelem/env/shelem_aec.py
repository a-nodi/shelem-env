from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector

from shelem.config import ShelemConfig
from shelem.env.raw_env import RawEnv
from shelem.game.state import PhaseEnum
from shelem.spaces.action import ACTION_SPACE_SIZE
from shelem.spaces.observation import ObservationBuilder, build_observation_space


class ShelemAECEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "shelem_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        config: ShelemConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config or ShelemConfig.from_yaml()
        self.render_mode = render_mode

        self.possible_agents = [
            f"player_{i}" for i in range(self._config.num_players)
        ]

        _obs_space = build_observation_space()
        _act_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.observation_spaces = {a: _obs_space for a in self.possible_agents}
        self.action_spaces = {a: _act_space for a in self.possible_agents}

        self._raw: RawEnv = RawEnv(self._config)

    # ------------------------------------------------------------------
    # Required AEC methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        self._raw.reset(seed=seed)
        state = self._raw.state

        self.agents = self.possible_agents[:]
        self._AgentSelector = AgentSelector(self.possible_agents)
        self.agent_selection = self.possible_agents[state.current_agent]

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def step(self, action: int) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        self.rewards = {a: 0.0 for a in self.agents}
        # AEC convention: clear acting agent's cumulative reward at step start
        self._cumulative_rewards[self.agent_selection] = 0

        prev_scores = self._raw.state.scores[:]

        self._raw.step(action)
        state = self._raw.state  # may be new object after void-hand / auto-redeal

        # rewards: non-zero whenever scores change (i.e. a hand just ended)
        new_scores = state.scores
        if new_scores != prev_scores:
            for i, agent in enumerate(self.possible_agents):
                team = self._config.team_of(i)
                self.rewards[agent] = float(new_scores[team] - prev_scores[team])

        # update agent_selection to whoever acts next
        if state.game_over:
            self.terminations = {a: True for a in self.agents}
        else:
            self.agent_selection = self.possible_agents[state.current_agent]

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def observe(self, agent: str) -> dict:
        player = self._agent_id(agent)
        assert self._raw.state is not None
        return ObservationBuilder.build(self._raw.state, player)

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def action_mask(self, agent: str) -> np.ndarray:
        from shelem.utils.action_mask import compute_action_mask
        assert self._raw.state is not None
        return compute_action_mask(self._raw.state, self._agent_id(agent))

    def state(self) -> np.ndarray:
        """Flat global observable state for centralised critics."""
        assert self._raw.state is not None
        s = self._raw.state
        parts = [
            np.array(s.tricks_won, dtype=np.float32),
            np.array(s.points_won, dtype=np.float32),
            np.array(s.scores, dtype=np.float32),
            np.array([int(s.phase), s.current_bid, int(s.play_mode)], dtype=np.float32),
        ]
        return np.concatenate(parts)

    def render(self) -> None:
        if self.render_mode != "human":
            return
        s = self._raw.state
        print(
            f"Phase={s.phase.name}  "
            f"Bid={s.current_bid}  "
            f"Declarer={s.declarer}  "
            f"Trump={s.trump_suit}  "
            f"Tricks={s.tricks_won}  "
            f"Scores={s.scores}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _agent_id(self, agent: str) -> int:
        return self.possible_agents.index(agent)


# ------------------------------------------------------------------
# Factory (standard PettingZoo entry point)
# ------------------------------------------------------------------

def env(**kwargs) -> ShelemAECEnv:
    raw = ShelemAECEnv(**kwargs)
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw
