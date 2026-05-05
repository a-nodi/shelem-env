# 시각화 & 모델 연동 가이드

터미널에서 학습된 모델이 게임을 플레이하는 것을 실시간으로 보거나,
에피소드를 JSON으로 기록·저장·재생할 수 있습니다.

---

## 설치

```bash
pip install -e ".[render]"   # rich 포함
```

---

## 빠른 시작

```bash
# random agent 실시간 관전
python examples/watch.py

# seed 고정, 느린 속도
python examples/watch.py --seed 42 --delay 1.0

# 학습된 모델 로드 (확장자 자동 감지)
python examples/watch.py --model models/ppo_shelem.zip
python examples/watch.py --model models/actor.onnx
python examples/watch.py --model models/net.pt

# 에피소드 기록 후 바로 재생
python examples/watch.py --model models/ppo.zip --record replays/game_001.json

# 저장된 에피소드 재생
python examples/watch.py --replay replays/game_001.json --delay 0.3
```

---

## 화면 구성

```
┌────── SHELEM  │  Acting: player_2  │  Goal: 505 pts ──────┐
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

| 영역 | 설명 |
|------|------|
| `Phase / Trump / Declarer / Bid` | 현재 게임 상태 |
| `Current Trick` | 이번 트릭에 낸 카드들 |
| `Hands` | 4명의 손패 (`►` = 현재 액션 플레이어) |
| `Score / Tricks / Points` | 팀별 누적 점수·트릭·카드점수 |
| `>> agent: action` | 직전 액션 |

`--hide-hands` 옵션 사용 시 현재 액션 플레이어의 패만 공개됩니다.

---

## Python API

### 실시간 관전 — `watch()`

```python
from shelem.render.terminal import watch

watch(
    policy=None,          # None = random. ShelemPolicy 또는 callable
    config=None,          # ShelemConfig. None = default.yaml
    seed=42,              # RNG seed
    delay=0.5,            # 스텝 간 대기 시간 (초)
    show_all_hands=True,  # False = 액션 플레이어 패만 공개
)
```

---

## 모델 로더 — `load_policy()`

```python
from shelem.policy import load_policy

policy = load_policy(source, **kwargs)
```

`source`에 따라 자동으로 적합한 Policy 클래스를 반환합니다.

| `source` 값 | 반환 타입 | 필요 패키지 |
|-------------|-----------|-------------|
| `None` 또는 `"random"` | `RandomPolicy` | — |
| `ShelemPolicy` 인스턴스 | 그대로 반환 | — |
| `callable(obs, mask)→int` | 어댑터로 래핑 | — |
| `"path/to/model.zip"` | `SB3Policy` | `stable-baselines3` |
| `"path/to/model.onnx"` | `OnnxPolicy` | `onnxruntime` |
| `"path/to/model.pt"` | `TorchPolicy` | `torch` |

### 예시

```python
from shelem.policy import load_policy

# random
policy = load_policy()

# SB3 모델
policy = load_policy("models/ppo_shelem.zip")
policy = load_policy("models/ppo_shelem.zip", algorithm="PPO")  # 알고리즘 명시

# ONNX
policy = load_policy("models/actor.onnx")

# PyTorch
policy = load_policy("models/net.pt", device="cuda")

# 간단한 함수로 직접 전달
policy = load_policy(lambda obs, mask: int(mask.nonzero()[0][0]))
```

---

## 커스텀 모델 연동

### 방법 1 — callable 함수

```python
import numpy as np
from shelem.policy import load_policy, flatten_obs

def my_policy(obs: dict, mask: np.ndarray) -> int:
    flat = flatten_obs(obs)   # (198,) float32 벡터로 변환
    logits = my_model(flat)   # 모델 추론
    logits[mask == 0] = -1e9  # 불법 액션 마스킹
    return int(np.argmax(logits))

policy = load_policy(my_policy)
```

### 방법 2 — `ShelemPolicy` 서브클래스 (권장)

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

모델이 flat 벡터 입력을 기대할 때 사용합니다.

```python
from shelem.policy import flatten_obs

flat = flatten_obs(obs)  # shape: (198,), dtype: float32
```

**벡터 구성 (총 198차원):**

| 필드 | 차원 | 설명 |
|------|------|------|
| `hand` | 52 | 내 손패 (0/1) |
| `played_cards` | 52 | 이번 트릭에 공개된 카드 |
| `current_trick` | 4 | 트릭 내 카드 인덱스 (-1=빈 슬롯) |
| 스칼라 6개 | 6 | trick_leader, trump_suit, play_mode, phase, declarer, bid |
| `bid_history` | 80 | 플레이어별 비딩 기록 (4×20) |
| `tricks_won` | 2 | 팀별 트릭 수 |
| `points_won` | 2 | 팀별 카드 점수 |

### SB3 연동 예시

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
        # 안전장치: 모델이 불법 액션을 출력할 경우 첫 번째 합법 액션으로 대체
        if not mask[action]:
            action = int(mask.nonzero()[0][0])
        return action
```

> SB3Policy는 `load_policy("model.zip")`으로도 동일하게 로드됩니다.

### 멀티-에이전트 (팀별 다른 모델)

```python
from shelem.policy import load_policy
from shelem.render.recorder import record

my_model   = load_policy("models/my_ppo.zip")
opp_model  = load_policy("models/opponent.zip")

policies = {
    "player_0": my_model,    # Team 0
    "player_2": my_model,    # Team 0 (같은 모델 공유)
    "player_1": opp_model,   # Team 1
    "player_3": opp_model,   # Team 1
}

ep = record(policies=policies, seed=42)
```

---

## 에피소드 기록 & 재생

같은 seed와 action 시퀀스가 있으면 게임이 결정론적으로 완전히 재현됩니다.

### 기록 — `record()`

```python
from shelem.policy import load_policy
from shelem.render.recorder import record, save_episode

policy = load_policy("models/ppo.zip")
ep = record(policy=policy, seed=42)

print(ep.total_steps)    # 총 스텝 수
print(ep.final_scores)   # [team0_score, team1_score]
print(ep.winner)         # 0 또는 1

save_episode(ep, "replays/game_001.json")
```

### 재생 — `replay()`

```python
from shelem.render.recorder import load_episode, replay

ep = load_episode("replays/game_001.json")
replay(ep, delay=0.5)
replay(ep, delay=0.0, show_all_hands=False)  # 빠른 재생, 패 숨김
```

### JSON 형식

`save_episode()`가 생성하는 파일 구조:

```json
{
  "seed": 42,
  "config": {
    "num_players": 4,
    "game_threshold": 505,
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

## CLI 옵션 전체

```
python examples/watch.py [옵션]

옵션:
  --model PATH      모델 파일 경로 (.zip / .onnx / .pt/.pth)
  --seed INT        RNG seed
  --delay FLOAT     스텝 간 대기 시간 (초, 기본값 0.5)
  --config PATH     YAML 설정 파일 경로
  --record PATH     에피소드를 JSON으로 저장하고 바로 재생
  --replay PATH     저장된 에피소드 재생
  --hide-hands      상대 패 숨기기 (현재 플레이어 패만 공개)
```

---

## 전체 워크플로우 예시

```python
from shelem.policy import load_policy
from shelem.render.terminal import watch
from shelem.render.recorder import record, save_episode, load_episode, replay
from shelem.config import ShelemConfig

# 1. 모델 로드
policy = load_policy("models/ppo_shelem.zip")

# 2. 실시간 관전
watch(policy=policy, seed=42, delay=0.5)

# 3. 배치 기록 (시각화 없이)
for i in range(10):
    ep = record(policy=policy, seed=i)
    save_episode(ep, f"replays/game_{i:03d}.json")
    print(f"game {i}: winner=Team {ep.winner}, scores={ep.final_scores}")

# 4. 원하는 에피소드만 replay
ep = load_episode("replays/game_003.json")
replay(ep, delay=0.3)
```

---

## 사람 vs 모델 대전 — `play_vs_model()`

학습된 모델을 상대로 직접 플레이할 수 있습니다.
사람 턴에는 번호로 행동을 선택하고, 모델 턴은 자동으로 진행됩니다.

### CLI

```bash
# random 상대와 대전
python examples/play.py

# 학습된 모델과 대전
python examples/play.py --model models/ppo_shelem.zip
python examples/play.py --model models/actor.onnx
python examples/play.py --model models/net.pt

# 다른 플레이어 포지션으로 플레이 (기본값: player_0)
python examples/play.py --human player_1

# 같은 팀 두 포지션 모두 직접 제어 (Team 0 = player_0 + player_2)
python examples/play.py --human player_0 player_2

# seed 고정, 모델 속도 조절
python examples/play.py --seed 42 --model-delay 0.8

# 변형 룰 사용
python examples/play.py --config configs/kqj_variant.yaml
```

### 화면 예시

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

모델 턴은 한 줄로 표시됩니다:

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
    policy=model,                        # 모델. None = random 상대
    human_agents=["player_0"],           # 내가 제어할 포지션
    config=None,                         # ShelemConfig. None = default.yaml
    seed=42,                             # RNG seed
    model_delay=0.4,                     # 모델 턴 후 대기 시간 (초)
)
```

### 각 페이즈별 입력 방식

**비딩 (BIDDING)**
```
Legal bids:
  [0] PASS    [1] 85    [2] 90    [3] 95  ...

Enter number: _
```

**자민 교환 (ZAMIN_EXCHANGE)** — 4장 순서대로 버리기
```
Discard a card  (0/4 discarded so far)
  [0] A♠   [1] K♠   [2] Q♠  ...

Enter number: _
```

**트럼프 선언 (TRUMP_DECLARATION)**
```
Declare play mode:
  [0] NORMAL     — first card sets trump, A > K > … > 2
  [1] NARES      — reversed order: 2 > 3 > … > A, no trump
  [2] ACE_NARES  — A highest, then 2 > 3 > … > K, no trump
  [3] SARRES     — standard A-high, no trump

Enter number: _
```

**플레이 (PLAY)** — 합법적인 카드만 표시됨
```
Led suit: ♠  (you must follow suit if you have it)
Legal cards:
  [0] 3♠   [1] 7♠   [2] A♠

Enter number: _
```

### CLI 옵션 전체

```
python examples/play.py [옵션]

옵션:
  --model PATH        모델 파일 (.zip / .onnx / .pt/.pth). 없으면 random 상대
  --human AGENT ...   내가 제어할 플레이어 이름 (기본값: player_0)
  --seed INT          RNG seed
  --model-delay FLOAT 모델 액션 후 대기 시간 (초, 기본값 0.4)
  --config PATH       YAML 설정 파일 경로
```
