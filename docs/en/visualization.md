# Visualization & Model Integration Guide

Watch a trained model play in real time in the terminal, or record episodes
to JSON for later replay.

---

## Installation

```bash
pip install -e ".[render]"   # includes rich
```

---

## Quick Start

```bash
# Watch a random agent in real time
python examples/watch.py

# Fixed seed, slow pace
python examples/watch.py --seed 42 --delay 1.0

# Load a trained model (extension auto-detected)
python examples/watch.py --model models/ppo_shelem.zip
python examples/watch.py --model models/actor.onnx
python examples/watch.py --model models/net.pt

# Record an episode then replay it immediately
python examples/watch.py --model models/ppo.zip --record replays/game_001.json

# Replay a saved episode
python examples/watch.py --replay replays/game_001.json --delay 0.3
```

---

## Screen Layout

```
┌────── SHELEM  │  Acting: player_2  │  Goal: 660 pts ──────┐
│ Phase: PLAY   Trump: ♠   Declarer: player_0   Bid: 100    │
│                                                            │
│ Current Trick  P1: 7♠    P2: K♠                           │
│                                                            │
│ Hands                                                      │
│   P0    A♠ K♥ Q♥ J♥ 10♦ 9♦ 8♣ 7♣ 5♣ 4♣ 3♣ 2♣            │
│   P1    [12 cards]                                         │
│   P2 ►  A♥ Q♠ J♣ 10♣ 9♠ 8♠ 6♠ 5♠ 4♠ 3♠ 2♠ A♦            │
│   P3    [12 cards]                                         │
│                                                            │
│          Team 0  (P0+P2)    Team 1  (P1+P3)               │
│  Score        120                45                        │
│  Tricks         6                 3                        │
│  Points        80                45                        │
│                                                            │
│  >> player_1: play 7♠                                      │
└────────────────────────────────────────────────────────────┘
```

| Area | Description |
|------|-------------|
| `Phase / Trump / Declarer / Bid` | Current game state |
| `Current Trick` | Cards played in this trick |
| `Hands` | All 4 hands (`►` = current acting player) |
| `Score / Tricks / Points` | Per-team cumulative score, tricks, card points |
| `>> agent: action` | Most recent action |

Use `--hide-hands` to show only the acting player's hand.

---

## Python API

### Live Watching — `watch()`

```python
from shelem.render.terminal import watch

watch(
    policy=None,          # None = random. ShelemPolicy or callable
    config=None,          # ShelemConfig. None = default.yaml
    seed=42,              # RNG seed
    delay=0.5,            # seconds between steps
    show_all_hands=True,  # False = show only acting player's hand
)
```

---

## Model Loader — `load_policy()`

```python
from shelem.policy import load_policy

policy = load_policy(source, **kwargs)
```

The appropriate Policy class is returned automatically based on `source`:

| `source` value | Return type | Required package |
|----------------|-------------|------------------|
| `None` or `"random"` | `RandomPolicy` | — |
| `ShelemPolicy` instance | returned as-is | — |
| `callable(obs, mask) → int` | wrapped in adapter | — |
| `"path/to/model.zip"` | `SB3Policy` | `stable-baselines3` |
| `"path/to/model.onnx"` | `OnnxPolicy` | `onnxruntime` |
| `"path/to/model.pt"` | `TorchPolicy` | `torch` |

### Examples

```python
from shelem.policy import load_policy

# random
policy = load_policy()

# Stable-Baselines3 model
policy = load_policy("models/ppo_shelem.zip")
policy = load_policy("models/ppo_shelem.zip", algorithm="PPO")

# ONNX
policy = load_policy("models/actor.onnx")

# PyTorch
policy = load_policy("models/net.pt", device="cuda")

# plain function
policy = load_policy(lambda obs, mask: int(mask.nonzero()[0][0]))
```

---

## Custom Model Integration

### Option 1 — callable function

```python
import numpy as np
from shelem.policy import load_policy, flatten_obs

def my_policy(obs: dict, mask: np.ndarray) -> int:
    flat = flatten_obs(obs)   # converts to (198,) float32 vector
    logits = my_model(flat)
    logits[mask == 0] = -1e9  # mask illegal actions
    return int(np.argmax(logits))

policy = load_policy(my_policy)
```

### Option 2 — `ShelemPolicy` subclass (recommended)

```python
import numpy as np
from shelem.policy import ShelemPolicy, flatten_obs

class MyPolicy(ShelemPolicy):
    def __init__(self, model_path: str):
        self.model = load_my_model(model_path)

    def act(self, obs: dict, mask: np.ndarray) -> int:
        flat = flatten_obs(obs)
        logits = self.model.predict(flat)
        logits[mask == 0] = -1e9
        return int(np.argmax(logits))

policy = MyPolicy("models/my_model.bin")
```

### `flatten_obs()` — observation dict → flat vector

Use when your model expects a flat vector input.

```python
from shelem.policy import flatten_obs

flat = flatten_obs(obs)  # shape: (198,), dtype: float32
```

**Vector layout (198 dimensions total):**

| Field | Dims | Description |
|-------|------|-------------|
| `hand` | 52 | Own hand (0/1) |
| `played_cards` | 52 | Cards visible in the current trick |
| `current_trick` | 4 | Card indices in the trick (-1 = empty slot) |
| 6 scalars | 6 | trick_leader, trump_suit, play_mode, phase, declarer, bid |
| `bid_history` | 80 | Per-player bid history (4×20) |
| `tricks_won` | 2 | Per-team trick count |
| `points_won` | 2 | Per-team card points |

### SB3 integration example

```python
from stable_baselines3 import PPO
from shelem.policy import ShelemPolicy, flatten_obs
import numpy as np

class SB3Policy(ShelemPolicy):
    def __init__(self, path: str):
        self.model = PPO.load(path)

    def act(self, obs: dict, mask: np.ndarray) -> int:
        flat = flatten_obs(obs)
        action, _ = self.model.predict(flat, deterministic=True)
        action = int(action)
        # safety: fall back to first legal action if model outputs an illegal one
        if not mask[action]:
            action = int(mask.nonzero()[0][0])
        return action
```

> `SB3Policy` is also returned by `load_policy("model.zip")`.

### Multi-agent (different models per team)

```python
from shelem.policy import load_policy
from shelem.render.recorder import record

my_model  = load_policy("models/my_ppo.zip")
opp_model = load_policy("models/opponent.zip")

policies = {
    "player_0": my_model,    # Team 0
    "player_2": my_model,    # Team 0 (shared model)
    "player_1": opp_model,   # Team 1
    "player_3": opp_model,   # Team 1
}

ep = record(policies=policies, seed=42)
```

---

## Episode Recording & Replay

Given the same seed and action sequence, games are fully deterministic.

### Recording — `record()`

```python
from shelem.policy import load_policy
from shelem.render.recorder import record, save_episode

policy = load_policy("models/ppo.zip")
ep = record(policy=policy, seed=42)

print(ep.total_steps)    # total steps
print(ep.final_scores)   # [team0_score, team1_score]
print(ep.winner)         # 0 or 1

save_episode(ep, "replays/game_001.json")
```

### Replay — `replay()`

```python
from shelem.render.recorder import load_episode, replay

ep = load_episode("replays/game_001.json")
replay(ep, delay=0.5)
replay(ep, delay=0.0, show_all_hands=False)  # fast, hide hands
```

### JSON Format

Structure of the file produced by `save_episode()`:

```json
{
  "seed": 42,
  "config": {
    "num_players": 4,
    "game_threshold": 660,
    ...
  },
  "final_scores": [120, -85],
  "winner": 0,
  "total_steps": 147,
  "actions": [
    ["player_1", 57],
    ["player_2", 52],
    ...
  ]
}
```

---

## Full CLI Options

```
python examples/watch.py [options]

Options:
  --model PATH      Model file (.zip / .onnx / .pt/.pth)
  --seed INT        RNG seed
  --delay FLOAT     Seconds between steps (default 0.5)
  --config PATH     YAML config file path
  --record PATH     Save episode as JSON then replay it
  --replay PATH     Replay a saved episode
  --hide-hands      Hide opponents' hands (show only acting player's hand)
```

---

## Full Workflow Example

```python
from shelem.policy import load_policy
from shelem.render.terminal import watch
from shelem.render.recorder import record, save_episode, load_episode, replay

# 1. Load model
policy = load_policy("models/ppo_shelem.zip")

# 2. Watch live
watch(policy=policy, seed=42, delay=0.5)

# 3. Batch record (no rendering)
for i in range(10):
    ep = record(policy=policy, seed=i)
    save_episode(ep, f"replays/game_{i:03d}.json")
    print(f"game {i}: winner=Team {ep.winner}, scores={ep.final_scores}")

# 4. Replay a specific episode
ep = load_episode("replays/game_003.json")
replay(ep, delay=0.3)
```

---

## Human vs Model — `play_vs_model()`

Play against a trained model interactively.
On your turn, select an action by number; the model plays automatically on its turn.

### CLI

```bash
# Play against a random agent
python examples/play.py

# Play against a trained model
python examples/play.py --model models/ppo_shelem.zip
python examples/play.py --model models/actor.onnx

# Choose your player position (default: player_0)
python examples/play.py --human player_1

# Control both positions on your team (Team 0 = player_0 + player_2)
python examples/play.py --human player_0 player_2

# Fix seed, adjust model speed
python examples/play.py --seed 42 --model-delay 0.8

# Use a variant config
python examples/play.py --config configs/kqj_variant.yaml
```

### Screen Example

```
──────────── YOUR TURN  player_0  (Team 0) ────────────
Phase: PLAY   Trump: ♠   Declarer: player_0   Bid: 100

         Team 0  (P0+P2)    Team 1  (P1+P3)
Score         120                 45
Tricks          6                  3
Points         80                 45

Current Trick:  P1: 7♠    P2: K♠

Your hand (6 cards):  A♠  Q♥  J♥  10♦  9♦  2♣

Led suit: ♠  (you must follow suit if you have it)
Legal cards:
  [0] A♠

Enter number [0]: _
```

Model turns are shown as a single line:

```
  player_1 (model): bid 100
  player_2 (model): play A♦
  player_3 (model): pass
```

### Python API

```python
from shelem.policy import load_policy
from shelem.render.human import play_vs_model

model = load_policy("models/ppo_shelem.zip")

play_vs_model(
    policy=model,                        # model (None = random opponent)
    human_agents=["player_0"],           # positions you control
    config=None,                         # ShelemConfig. None = default.yaml
    seed=42,                             # RNG seed
    model_delay=0.4,                     # seconds to pause after model action
)
```

### Input per Phase

**Bidding**
```
Legal bids:
  [0] PASS    [1] 100    [2] 105    [3] 110  ...

Enter number: _
```

**Zamin Exchange** — discard 4 cards one at a time
```
Discard a card  (0/4 discarded so far)
  [0] A♠   [1] K♠   [2] Q♠  ...

Enter number: _
```

**Trump Declaration**
```
Declare play mode:
  [0] NORMAL     — first card sets trump, A > K > … > 2
  [1] NARES      — reversed order: 2 > 3 > … > A, no trump
  [2] ACE_NARES  — A highest, then 2 > 3 > … > K, no trump
  [3] SARRES     — standard A-high, no trump

Enter number: _
```

**Play** — only legal cards shown
```
Led suit: ♠  (you must follow suit if you have it)
Legal cards:
  [0] 3♠   [1] 7♠   [2] A♠

Enter number: _
```

### Full CLI Options

```
python examples/play.py [options]

Options:
  --model PATH        Model file (.zip / .onnx / .pt/.pth). Omit for random opponent.
  --human AGENT ...   Player positions to control (default: player_0)
  --seed INT          RNG seed
  --model-delay FLOAT Pause after model action in seconds (default 0.4)
  --config PATH       YAML config file path
```
