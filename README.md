# shelem-env

[PettingZoo](https://pettingzoo.farama.org/) AEC environment for **Shelem (شلم)**, the Iranian trick-taking card game. Built for multi-agent reinforcement learning research.

---

## Installation

```bash
git clone https://github.com/a-nodi/shelem-env.git
cd shelem-env
pip install -e .

# optional extras
pip install -e ".[render]"   # terminal renderer (rich)
pip install -e ".[torch]"    # PyTorch + Stable-Baselines3
pip install -e ".[dev]"      # pytest
```

---

## Quick Start

```python
from shelem.env.shelem_aec import ShelemAECEnv

env = ShelemAECEnv()
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    mask = obs["action_mask"]
    action = env.action_space(agent).sample(mask)
    env.step(action)
```

### RawEnv (MCTS / tree search)

`RawEnv` exposes pure game logic without any PettingZoo overhead. It supports `deepcopy`, making it suitable for MCTS rollouts.

```python
from shelem.env.raw_env import RawEnv

env = RawEnv()
env.reset(seed=0)

while env.state.phase.name != "SCORING":
    actions = env.legal_actions()
    env.step(actions[0])
```

---

## Game Overview

Shelem is a traditional Iranian trick-taking card game.

| Item | Detail |
|------|--------|
| Players | 4 (2 teams of 2) |
| Deck | Standard 52 cards |
| Card points | A=11, Q=10, 10=5, rest=0 |
| Minimum bid | 100 |
| Win condition | Cumulative score ≥ 660 |

**Game flow**

```
Deal → Bidding → Zamin Exchange → Trump Declaration → Play → Scoring
  └── (all pass) re-deal ──┘
```

For full rules see [docs/en/rules.md](docs/en/rules.md).

---

## Action Space

`Discrete(74)`

| Index | Meaning |
|-------|---------|
| 0 – 51 | Play a card / discard to Zamin (card index) |
| 52 | PASS (forfeit bid) |
| 53 – 69 | Declare bid (100, 105, … 165) |
| 70 – 73 | Declare play mode (Normal / Nares / Ace-Nares / Sarres) |

Actions illegal in the current phase are masked via `obs["action_mask"]`.

---

## Observation Space

`Dict` — each player can only observe their own hand.

| Key | Type | Description |
|-----|------|-------------|
| `hand` | `MultiBinary(52)` | Cards in the agent's hand |
| `played_cards` | `MultiBinary(52)` | Cards visible in the current trick |
| `current_trick` | `Box(4,)` | Card indices in this trick (-1 = not yet played) |
| `trick_leader` | `Discrete(4)` | Player index who led this trick |
| `trump_suit` | `Discrete(5)` | Trump suit (0 = not yet determined) |
| `play_mode` | `Discrete(4)` | Normal / Nares / Ace-Nares / Sarres |
| `phase` | `Discrete(5)` | Current game phase |
| `declarer` | `Discrete(5)` | Hâkem player (0 = not yet decided) |
| `bid` | `Discrete(166)` | Current highest bid |
| `bid_history` | `Box(4, 20)` | Per-player bidding history |
| `zamin_taken` | `Discrete(2)` | Whether Hâkem has taken the Zamin |
| `tricks_won` | `Box(2,)` | Tricks won per team |
| `points_won` | `Box(2,)` | Card points won per team |
| `action_mask` | `MultiBinary(74)` | Legal action mask |

---

## Configuration

Edit `configs/default.yaml` or construct a `ShelemConfig` directly to change the rules.

```python
from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv

cfg = ShelemConfig.from_yaml("configs/default.yaml")
env = ShelemAECEnv(config=cfg)
```

Bundled config files:

| File | Description |
|------|-------------|
| `configs/default.yaml` | Standard Shelem rules |
| `configs/ace15.yaml` | Ace worth 15 points variant |
| `configs/kqj_variant.yaml` | K/Q/J scoring variant |
| `configs/three_player.yaml` | 3-player variant |

---

## Built-in Policies

```python
from shelem.policy.builtins import RandomPolicy, GreedyPolicy

policy = RandomPolicy(seed=0)
action = policy.act(obs, legal_actions)
```

---

## Testing

```bash
pytest
```

---

## Documentation

| | English | 한국어 |
|-|---------|--------|
| Game Rules | [docs/en/rules.md](docs/en/rules.md) | [docs/ko/rules.md](docs/ko/rules.md) |
| Quick Start | [docs/en/quickstart.md](docs/en/quickstart.md) | [docs/ko/quickstart.md](docs/ko/quickstart.md) |
| Visualization | [docs/en/visualization.md](docs/en/visualization.md) | [docs/ko/visualization.md](docs/ko/visualization.md) |
