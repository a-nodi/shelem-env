# Shelem-Env: 빠른 시작 가이드

Shelem은 이란의 4인 트릭테이킹 카드 게임입니다. 이 라이브러리는
[PettingZoo AEC](https://pettingzoo.farama.org/api/aec/) 강화학습 환경으로 제공됩니다.

---

## 설치

```bash
pip install -e .          # 표준 룰
pip install -e ".[torch]" # + PyTorch & Stable-Baselines3
pip install -e ".[dev]"   # + pytest
```

Python ≥ 3.9 필요.

---

## 5분 예제

```python
from shelem.env.shelem_aec import ShelemAECEnv

env = ShelemAECEnv()
env.reset(seed=42)

while env.agents:                            # 게임 종료 시 agents 리스트가 비워짐
    agent = env.agent_selection

    if env.terminations[agent] or env.truncations[agent]:
        env.step(None)                       # PettingZoo AEC 필수 처리
        continue

    obs    = env.observe(agent)              # dict — Observation 섹션 참고
    mask   = obs["action_mask"]              # np.ndarray shape (74,), dtype int8
    legal  = mask.nonzero()[0].tolist()      # 합법 액션 인덱스 리스트

    action = legal[0]                        # 첫 번째 합법 액션 (실제 정책으로 교체)
    env.step(action)

print("최종 보상:", env.rewards)
```

---

## 에이전트와 팀

| 에이전트 문자열 | 플레이어 인덱스 | 팀 |
|-----------------|-----------------|-----|
| `"player_0"`    | 0               | 0   |
| `"player_1"`    | 1               | 1   |
| `"player_2"`    | 2               | 0   |
| `"player_3"`    | 3               | 1   |

팀은 고정: 플레이어 0·2 (팀 0) vs 플레이어 1·3 (팀 1). 보상은 팀 내에서 공유됩니다.

---

## 액션 공간 — `Discrete(74)`

매 스텝에서 정수 액션 하나를 전달합니다. 74개 슬롯의 의미:

| 범위    | 의미                                                          |
|---------|---------------------------------------------------------------|
| 0 – 51  | 카드 내기 / 자민 버리기 (카드 인덱스)                        |
| 52      | `PASS` (비딩 포기)                                           |
| 53 – 69 | 비드 선언: 100, 105, … 165 (5점 단위)                        |
| 70 – 73 | 플레이 모드 선언: NORMAL / NARES / ACE_NARES / SARRES        |

**항상 action mask를 사용하세요** — 불법 액션은 오류를 발생시킵니다.

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
bid_action = bid_to_action(100, cfg)  # 비드 100에 해당하는 액션 인덱스
```

### 카드 인덱스 구조

카드 인덱스 = `수트 * 13 + (랭크 - 2)`:

| 수트     | 인덱스 범위 |
|----------|-------------|
| Clubs    | 0 – 12      |
| Diamonds | 13 – 25     |
| Hearts   | 26 – 38     |
| Spades   | 39 – 51     |

각 수트 내: 인덱스 0 = 2, 인덱스 1 = 3, …, 인덱스 12 = Ace.

```python
from shelem.game.card import Card, Rank, Suit, INDEX_TO_CARD

card = Card(Rank.ACE, Suit.SPADES)
print(card.index)          # 51
print(INDEX_TO_CARD[0])    # 2♣
```

---

## 관측 공간 — `Dict`

`env.observe(agent)`가 반환하는 딕셔너리:

| 키              | 형태 / 타입                | 설명                                                           |
|-----------------|----------------------------|----------------------------------------------------------------|
| `hand`          | `MultiBinary(52)`          | 해당 에이전트의 손패 (1 = 보유)                               |
| `played_cards`  | `MultiBinary(52)`          | 현재 트릭에 공개된 카드                                       |
| `current_trick` | `Box(-1,51, shape=(4,))`   | 이번 트릭의 카드 인덱스; -1 = 슬롯 비어 있음                 |
| `trick_leader`  | `Discrete(4)`              | 현재 트릭을 리드한 플레이어 인덱스                            |
| `trump_suit`    | `Discrete(5)`              | 0 = 미확정, 1–4 = Clubs/Diamonds/Hearts/Spades               |
| `play_mode`     | `Discrete(4)`              | 0=NORMAL, 1=NARES, 2=ACE_NARES, 3=SARRES                     |
| `phase`         | `Discrete(5)`              | 0=BIDDING, 1=ZAMIN_EXCHANGE, 2=TRUMP_DECLARATION, 3=PLAY, 4=SCORING |
| `declarer`      | `Discrete(5)`              | 0 = 미결정, 1–4 = 플레이어 인덱스 + 1                       |
| `bid`           | `Discrete(166)`            | 현재 최고 비드 (0 = 아직 없음)                               |
| `bid_history`   | `Box(0,165, shape=(4,20))` | 플레이어별 비딩 기록; 0 = 해당 슬롯 비드 없음               |
| `zamin_taken`   | `Discrete(2)`              | 1이면 Hâkem이 자민을 수령함                                  |
| `tricks_won`    | `Box(0,13, shape=(2,))`    | 팀별 획득 트릭 수 [팀0, 팀1]                                 |
| `points_won`    | `Box(0,200, shape=(2,))`   | 팀별 카드+트릭 점수 [팀0, 팀1]                               |
| `action_mask`   | `MultiBinary(74)`          | 1 = 합법 액션, dtype `int8`                                  |

상대방 손패와 자민 내용은 숨겨져 있습니다 (0으로 채워짐).

---

## 게임 단계

게임은 다음 순서로 진행됩니다:

```
BIDDING → ZAMIN_EXCHANGE → TRUMP_DECLARATION → PLAY
    ↑                                             │
    └─────────────── 자동 재딜 ──────────────────┘
                     (game_over까지)
```

| 단계                  | 행동 플레이어       | 유효 액션                        |
|-----------------------|---------------------|----------------------------------|
| BIDDING               | 전원 순환           | 비드 (53–69) 또는 PASS (52)      |
| ZAMIN_EXCHANGE        | Hâkem만             | 카드 인덱스 (0–51) × 4회 버리기  |
| TRUMP_DECLARATION     | Hâkem만             | 플레이 모드 (70–73)              |
| PLAY                  | 현재 리더 및 팔로워 | 카드 인덱스 (0–51)               |
| SCORING               | — (종료)            | 없음                             |

**Void hand**: 비딩에서 4명 모두 패스하면 자동 재딜됩니다 (점수 유지).

---

## 보상

보상은 **희소(sparse)** 합니다 — 핸드가 끝날 때만 0이 아닙니다:

- **Win**: 선언팀 득점 ≥ 비드 → 선언팀 `+득점`, 수비팀 `+득점`
- **Fail**: 득점 < 비드, 선언팀 ≥ 수비팀 → 선언팀 `-bid`, 수비팀 `+득점`
- **Double**: 득점 < 비드, 선언팀 < 수비팀 → 선언팀 `-(2×bid)`, 수비팀 `+득점`
- **Shelem**: 선언팀이 13트릭 전부 획득 → 선언팀 `+shelem_bonus` (기본 250)

`env.rewards[agent]` — 마지막 스텝의 에이전트별 보상.
`env._cumulative_rewards[agent]` — 에이전트가 마지막으로 행동한 이후 누적 보상.

---

## 설정

내장 설정을 로드하거나 직접 생성합니다:

```python
from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv

# 기본 룰 (configs/default.yaml)
env = ShelemAECEnv()

# 내장 변형 룰
cfg = ShelemConfig.from_yaml("configs/ace15.yaml")       # 에이스 = 15점
cfg = ShelemConfig.from_yaml("configs/kqj_variant.yaml") # K=4, Q=3, J=2
cfg = ShelemConfig.from_yaml("configs/three_player.yaml")

# 코드로 직접 설정
cfg = ShelemConfig(
    num_players=4,
    hand_size=12,
    zamin_size=4,
    zamin_discard_count=4,
    card_points={"A": 11, "Q": 10, "10": 5},
    min_bid=100,
    max_bid=165,
    bid_increment=5,
    shelem_bonus=250,
    game_threshold=660,
)
env = ShelemAECEnv(config=cfg)
```

| 파라미터              | 기본값  | 설명                                      |
|-----------------------|---------|-------------------------------------------|
| `num_players`         | 4       | 플레이어 수                               |
| `hand_size`           | 12      | 플레이어당 카드 수                        |
| `zamin_size`          | 4       | 자민(kitty) 카드 수                       |
| `zamin_discard_count` | 4       | Hâkem이 버리는 카드 수                    |
| `card_points`         | 위 참고 | 랭크별 점수                               |
| `min_bid`             | 100     | 최소 비드                                 |
| `max_bid`             | 165     | 최대 비드                                 |
| `bid_increment`       | 5       | 비드 단위                                 |
| `shelem_bonus`        | 250     | 13트릭 전부 획득 시 보너스                |
| `game_threshold`      | 660     | 게임 승리 임계 점수                       |

---

## MCTS / 게임 트리 롤아웃 (PettingZoo 오버헤드 없음)

`RawEnv`는 PettingZoo 없이 사용할 수 있는 순수 게임 로직 래퍼로, deepcopy가 가능합니다:

```python
from shelem.env.raw_env import RawEnv
from shelem.config import ShelemConfig

env = RawEnv(ShelemConfig.from_yaml())
env.reset(seed=0)

# 롤아웃을 위한 복사 (PettingZoo, 보상 계산 없음)
snapshot = env.clone()   # 내부적으로 copy.deepcopy 사용

legal = snapshot.legal_actions()   # list[int]
snapshot.step(legal[0])

state = snapshot.state             # GameState 데이터클래스
print(state.phase, state.scores)
```

`RawEnv.step()`은 AEC 환경과 동일한 액션 정수를 받습니다.

---

## 테스트 실행

```bash
pytest                           # 전체 테스트
pytest -v                        # 상세 출력
pytest tests/test_scoring.py    # 모듈 단위 실행
```
