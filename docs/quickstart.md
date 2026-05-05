# Shelem-Env: Quick Start Guide

Shelem is a 4-player Iranian trick-taking card game. This library exposes it as a
[PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) reinforcement learning environment.

---

## Installation

```bash
pip install -e .          # standard rules
pip install -e ".[torch]" # + PyTorch & Stable-Baselines3
pip install -e ".[dev]"   # + pytest
```

Requires Python ≥ 3.9.

---

## 5-Minute Example

```python
from shelem.env.shelem_aec import ShelemAECEnv

env = ShelemAECEnv()
env.reset(seed=42)

while env.agents:                            # agents list empties when game ends
    agent = env.agent_selection

    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)                       # required by PettingZoo AEC
        continue

    obs    = env.observe(agent)              # dict — see Observation section
    mask   = obs["action_mask"]              # np.ndarray shape (74,), dtype int8
    legal  = mask.nonzero()[0].tolist()      # list of legal action indices

    action = legal[0]                        # pick first legal action (replace with policy)
    env.step(action)

print("Final rewards:", env.rewards)
```

---

## Agents and Teams

| Agent string  | Player index | Team |
|---------------|--------------|------|
| `"player_0"`  | 0            | 0    |
| `"player_1"`  | 1            | 1    |
| `"player_2"`  | 2            | 0    |
| `"player_3"`  | 3            | 1    |

Teams are fixed: players 0 & 2 vs players 1 & 3. Rewards are shared within a team.

---

## Action Space — `Discrete(74)`

Every step takes a single integer action. The 74 slots are:

| Range   | Meaning                                      |
|---------|----------------------------------------------|
| 0 – 51  | Card play / Zamin discard (card index)       |
| 52      | `PASS` (bidding only)                        |
| 53 – 69 | Bid values: 85, 90, 95 … 165 (step 5)       |
| 70 – 73 | Play-mode declaration: NORMAL / NARES / ACE_NARES / SARRES |

**Always use the action mask** — illegal actions raise an error or produce
undefined behaviour.

```python
from shelem.spaces.action import (
    ACTION_PASS,        # 52
    BID_OFFSET,         # 53
    ACTION_MODE_OFFSET, # 70
    ACTION_SPACE_SIZE,  # 74
    bid_to_action,      # bid_to_action(100, cfg) → int
    action_to_bid,      # action_to_bid(action, cfg) → int
    action_to_play_mode # action_to_play_mode(action) → PlayMode
)

cfg = env._config
bid_action = bid_to_action(100, cfg)  # action index for a bid of 100
```

### Card index layout

Cards are indexed `suit * 13 + (rank - 2)`:

| Suit     | Index range |
|----------|-------------|
| Clubs    | 0 – 12      |
| Diamonds | 13 – 25     |
| Hearts   | 26 – 38     |
| Spades   | 39 – 51     |

Within each suit: index 0 = 2, index 1 = 3, …, index 12 = Ace.

```python
from shelem.game.card import Card, Rank, Suit, INDEX_TO_CARD

card = Card(Rank.ACE, Suit.SPADES)
print(card.index)          # 51
print(INDEX_TO_CARD[0])    # 2♣
```

---

## Observation Space — `Dict`

`env.observe(agent)` returns a dictionary:

| Key             | Shape / Type              | Description                                                    |
|-----------------|---------------------------|----------------------------------------------------------------|
| `hand`          | `MultiBinary(52)`         | Cards in *this* agent's hand (1 = held)                       |
| `played_cards`  | `MultiBinary(52)`         | Cards visible in the current trick                            |
| `current_trick` | `Box(-1,51, shape=(4,))`  | Card indices played so far this trick; -1 = slot empty        |
| `trick_leader`  | `Discrete(4)`             | Player index who led this trick                               |
| `trump_suit`    | `Discrete(5)`             | 0 = unknown, 1–4 = Clubs/Diamonds/Hearts/Spades               |
| `play_mode`     | `Discrete(4)`             | 0=NORMAL, 1=NARES, 2=ACE_NARES, 3=SARRES                     |
| `phase`         | `Discrete(5)`             | 0=BIDDING, 1=ZAMIN_EXCHANGE, 2=TRUMP_DECLARATION, 3=PLAY, 4=SCORING |
| `declarer`      | `Discrete(5)`             | 0 = not yet determined, 1–4 = player index + 1               |
| `bid`           | `Discrete(186)`           | Current winning bid (0 = none yet)                            |
| `bid_history`   | `Box(0,185, shape=(4,20))`| Per-player bid history; 0 = no bid in that slot               |
| `zamin_taken`   | `Discrete(2)`             | 1 if Hâkem has picked up Zamin cards                         |
| `tricks_won`    | `Box(0,13, shape=(2,))`   | Tricks won by [team 0, team 1]                                |
| `points_won`    | `Box(0,185, shape=(2,))`  | Card + trick points by [team 0, team 1]                       |
| `action_mask`   | `MultiBinary(74)`         | 1 = legal action, dtype `int8`                                |

Opponent hands and Zamin contents are hidden (zero-filled).

---

## Game Phases

The game progresses through these phases in order:

```
BIDDING → ZAMIN_EXCHANGE → TRUMP_DECLARATION → PLAY
    ↑                                             │
    └─────────────── auto re-deal ───────────────┘
                     (until game_over)
```

| Phase              | Who acts          | Valid actions              |
|--------------------|-------------------|----------------------------|
| BIDDING            | All players round-robin | Bid (53–69) or PASS (52) |
| ZAMIN_EXCHANGE     | Hâkem only        | Card index (0–51) × 4 discards |
| TRUMP_DECLARATION  | Hâkem only        | Play mode (70–73)          |
| PLAY               | Current trick leader and followers | Card index (0–51) |
| SCORING            | — (terminal)      | None — game over           |

**Void hand**: if all 4 players pass during BIDDING, hands are re-dealt automatically
(scores are preserved).

---

## Rewards

Rewards are **sparse** — only non-zero when a hand ends:

- **Win**: declarer team earned ≥ bid → `+earned` for declarers, `+earned` for defenders
- **Fail**: earned < bid, earned ≥ defenders → `-bid` for declarers, `+earned` for defenders
- **Doubled**: earned < bid, earned < defenders → `-(2 × bid)` for declarers, `+earned` for defenders
- **Shelem**: declarer team won all 13 tricks → `+shelem_bonus` (default 250)

`env.rewards[agent]` is the per-agent reward from the last step.
`env._cumulative_rewards[agent]` accumulates since the agent last acted.

---

## Configuration

Load a built-in config or pass your own:

```python
from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv

# default rules (configs/default.yaml)
env = ShelemAECEnv()

# built-in variants
cfg = ShelemConfig.from_yaml("configs/ace15.yaml")      # Ace = 15 pts
cfg = ShelemConfig.from_yaml("configs/kqj_variant.yaml") # K=4, Q=3, J=2
cfg = ShelemConfig.from_yaml("configs/three_player.yaml")

# custom config in code
cfg = ShelemConfig(
    num_players=4,
    hand_size=12,
    zamin_size=4,
    zamin_discard_count=4,
    card_points={"A": 10, "10": 10, "5": 5, "K": 0, "Q": 0, "J": 0},
    min_bid=85,
    max_bid=165,
    bid_increment=5,
    shelem_bonus=250,
    game_threshold=505,
)
env = ShelemAECEnv(config=cfg)
```

| Parameter           | Default | Description                                    |
|---------------------|---------|------------------------------------------------|
| `num_players`       | 4       | Number of players                              |
| `hand_size`         | 12      | Cards dealt per player                         |
| `zamin_size`        | 4       | Zamin (kitty) cards                            |
| `zamin_discard_count` | 4     | Cards Hâkem discards after picking up Zamin    |
| `card_points`       | see above | Points per rank label                        |
| `min_bid`           | 85      | Minimum opening bid                            |
| `max_bid`           | 165     | Maximum bid                                    |
| `bid_increment`     | 5       | Bid step size                                  |
| `shelem_bonus`      | 250     | Bonus for winning all 13 tricks                |
| `game_threshold`    | 505     | Score at which a team wins the game            |

---

## MCTS / Game-Tree Rollouts (no PettingZoo overhead)

`RawEnv` is a thin, PettingZoo-free wrapper that is safe to deep-copy:

```python
from shelem.env.raw_env import RawEnv
from shelem.config import ShelemConfig

env = RawEnv(ShelemConfig.from_yaml())
env.reset(seed=0)

# clone for rollout (no PettingZoo, no reward bookkeeping)
snapshot = env.clone()   # copy.deepcopy under the hood

legal = snapshot.legal_actions()   # list[int]
snapshot.step(legal[0])

state = snapshot.state             # GameState dataclass
print(state.phase, state.scores)
```

`RawEnv.step()` accepts the same action integers as the AEC env.

---

## Running Tests

```bash
pytest                 # all 23 tests
pytest -v              # verbose output
pytest tests/test_scoring.py   # single module
```
