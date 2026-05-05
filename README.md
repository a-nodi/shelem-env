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

`RawEnv`는 PettingZoo 의존성 없이 순수 게임 로직만 제공합니다. `deepcopy`가 가능해 MCTS 롤아웃에 적합합니다.

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

Shelem은 이란의 전통 트릭테이킹 카드 게임입니다.

| 항목 | 내용 |
|------|------|
| 인원 | 4명 (2인 × 2팀) |
| 덱 | 표준 52장 |
| 카드 점수 | A=11, Q=10, 10=5, 나머지=0 |
| 최소 비드 | 100점 |
| 승리 조건 | 누적 점수 660점 이상 |

**게임 흐름**

```
딜 → 비딩 → 자민 교환 → 트럼프 선언 → 플레이 → 채점
 └── (전원 패스 시) 재딜 ──┘
```

자세한 규칙은 [docs/rules.md](docs/rules.md)를 참고하세요.

---

## Action Space

`Discrete(74)`

| 인덱스 | 의미 |
|--------|------|
| 0 – 51 | 카드 내기 / 자민 버리기 (카드 인덱스) |
| 52 | PASS (비딩 포기) |
| 53 – 69 | 비드 선언 (100, 105, … 165) |
| 70 – 73 | 플레이 모드 선택 (Normal / Nares / Ace-Nares / Sarres) |

현재 단계에서 유효하지 않은 액션은 `obs["action_mask"]`로 마스킹됩니다.

---

## Observation Space

`Dict` — 각 플레이어는 자신의 패만 볼 수 있습니다.

| 키 | 타입 | 설명 |
|----|------|------|
| `hand` | `MultiBinary(52)` | 자신이 보유한 카드 |
| `played_cards` | `MultiBinary(52)` | 현재 트릭에서 공개된 카드 |
| `current_trick` | `Box(4,)` | 이번 트릭 카드 인덱스 (-1=미출) |
| `trick_leader` | `Discrete(4)` | 현재 트릭 리더 |
| `trump_suit` | `Discrete(5)` | 트럼프 수트 (0=미확정) |
| `play_mode` | `Discrete(4)` | Normal/Nares/Ace-Nares/Sarres |
| `phase` | `Discrete(5)` | 현재 게임 단계 |
| `declarer` | `Discrete(5)` | Hâkem 플레이어 (0=미결정) |
| `bid` | `Discrete(166)` | 현재 최고 비드 |
| `bid_history` | `Box(4, 20)` | 플레이어별 비딩 기록 |
| `zamin_taken` | `Discrete(2)` | 자민 수령 여부 |
| `tricks_won` | `Box(2,)` | 팀별 획득 트릭 수 |
| `points_won` | `Box(2,)` | 팀별 획득 점수 |
| `action_mask` | `MultiBinary(74)` | 유효 액션 마스크 |

---

## Configuration

`configs/default.yaml`을 수정하거나 `ShelemConfig`를 직접 생성해 규칙을 변경할 수 있습니다.

```python
from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv

cfg = ShelemConfig.from_yaml("configs/default.yaml")
env = ShelemAECEnv(config=cfg)
```

제공되는 설정 파일:

| 파일 | 설명 |
|------|------|
| `configs/default.yaml` | 표준 셸렘 룰 |
| `configs/ace15.yaml` | 에이스 15점 변형 |
| `configs/kqj_variant.yaml` | K/Q/J 점수 추가 변형 |
| `configs/three_player.yaml` | 3인 변형 |

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
