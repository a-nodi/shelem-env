# Shelem (شلم) — Game Rules

Shelem is a traditional Iranian trick-taking card game played by 4 players in two teams.

---

## Overview

| Item | Detail |
|------|--------|
| Players | 4 (2 teams of 2) |
| Deck | Standard 52 cards |
| Win condition | Cumulative score ≥ 660 points |
| Play direction | Counter-clockwise |

**Teams** — Seated partners form one team.

```
        Player 1 (North)
            |
Player 2 --+-- Player 4    Team B = Player 2 + Player 4
 (West)     |   (East)
        Player 3 (South)

Team A = Player 1 (North) + Player 3 (South)
```

---

## Card Points

| Card | Points |
|------|--------|
| A (Ace) | 11 |
| Q (Queen) | 10 |
| 10 | 5 |
| K / J / 9 ~ 2 | 0 |

- **Total card points**: 104 (A×4×11 + Q×4×10 + 10×4×5)
- **Trick points**: 5 points per trick × 13 tricks = 65
- **Total per hand**: **169 points**

---

## Game Flow

```
Deal → Bidding → Zamin Exchange → Trump Declaration → Play → Scoring
  └── (all pass) re-deal ──┘
```

---

## Phase 1: Deal

1. Shuffle 52 cards; the player to the dealer's left cuts.
2. Deal counter-clockwise, starting from the player to the dealer's right — **12 cards** each.
3. Place the remaining **4 cards** face-down in the center. These are the **Zamin (زمین)**.
4. The dealer rotates counter-clockwise each hand.

---

## Phase 2: Bidding

### Rules

- Starts with the player to the dealer's right; proceeds counter-clockwise.
- A bid declares the **minimum score your team aims to earn** this hand.
- Minimum bid: **100**, in increments of 5, up to **165**.
- Each player must either bid **higher than the current highest bid** or **pass**.
- A player who passes cannot re-enter bidding.
- Bidding ends when **3 consecutive players pass** after the highest bid.
- If **all 4 players pass**, the hand is void and the next dealer re-deals.

### Result

- The highest bidder becomes **Hâkem (حاکم, the declarer)**.
- Hâkem and their partner form the **declaring team**; the other two are the **defending team**.

---

## Phase 3: Zamin Exchange

1. Hâkem picks up all 4 Zamin cards, holding **16 cards** total.
2. Hâkem discards **4 cards face-down** (not revealed to opponents).
3. The discarded 4 cards are:
   - Counted as the **declaring team's first trick** (+5 trick points)
   - Any A, Q, or 10 among them also **awards card points to the declaring team**

---

## Phase 4: Trump Declaration

Hâkem selects a play mode:

| Mode | Description |
|------|-------------|
| **Normal** | The suit of Hâkem's first card becomes **trump**. Standard A > K > Q > J > 10 … 2 ranking. |
| **Nares (نارِس)** | **Reversed** ranking: 2 is highest, A is lowest. No trump. |
| **Ace-Nares** | Ace remains highest, then A > 2 > 3 > … > K. No trump. |
| **Sarres (سارِس)** | Standard A-high ranking but **no trump**. |

In Normal mode, trump is revealed the moment Hâkem plays their first card.

---

## Phase 5: Play

### Trick Mechanics

- One trick = each of the 4 players plays 1 card (counter-clockwise).
- 12 tricks per hand (13 including the Zamin trick).
- Hâkem always leads the first trick.

### Must-Follow Rule

1. The leader may play any card.
2. Subsequent players **must follow the led suit** if they hold a card of that suit.
3. If they have no card of the led suit, they may play **any card** (including trump).

### Trick Winner

| Situation | Winner |
|-----------|--------|
| One or more trump cards played | Highest trump card |
| No trump cards played | Highest card of the led suit |

The trick winner leads the next trick.

---

### Play Examples

> **Setup**: Trump = ♠ (Spades), Hâkem = Player 0 (declaring team: 0·2 / defending: 1·3)

#### Example 1 — Must-follow (all players hold the led suit)

```
Player 0 (Hâkem): ♥7  ← leads. Led suit = Hearts
Player 1:         ♥K  ← has Hearts → must follow
Player 2:         ♥3  ← has Hearts → must follow
Player 3:         ♥J  ← has Hearts → must follow

No trump → highest led suit: ♥K
→ Player 1 wins, leads next trick
```

#### Example 2 — No led suit, trump played

```
Player 1 (leader): ♦A  ← leads. Led suit = Diamonds
Player 2:          ♦9  ← has Diamonds → follows
Player 3:          ♠5  ← no Diamonds → plays trump (♠)
Player 0:          ♦Q  ← has Diamonds → follows

Trump (♠) played → highest trump: ♠5 (only one)
→ Player 3 wins (♠5 beats ♦A)
```

#### Example 3 — Multiple trump cards

```
Player 3 (leader): ♣8  ← leads. Led suit = Clubs
Player 0:          ♠J  ← no Clubs → plays trump
Player 1:          ♣2  ← has Clubs → follows
Player 2:          ♠A  ← no Clubs → plays higher trump

Two trumps (♠J, ♠A) → highest trump: ♠A
→ Player 2 wins
```

#### Example 4 — Off-suit, non-trump cards cannot win

```
Player 2 (leader): ♥Q  ← leads. Led suit = Hearts
Player 3:          ♦A  ← no Hearts → discards Diamond (cannot win)
Player 0:          ♥5  ← has Hearts → follows
Player 1:          ♥K  ← has Hearts → follows

No trump → highest led suit (♥): ♥K
→ Player 1 wins (♦A is irrelevant)
```

---

## Phase 6: Scoring

### Points Calculation

```
Team score = sum of card points in won tricks
           + tricks won × 5
           + (declaring team only) Zamin card points
           + (declaring team only) Zamin trick points (+5)
```

### Hand Outcome

**Declaring team earns ≥ bid → Win**
```
Declaring team score += declaring team earned
Defending team score += defending team earned
```

**Declaring team earns < bid → Fail**

| Condition | Declaring team | Defending team |
|-----------|----------------|----------------|
| Declaring ≥ Defending | **-bid** | +defending earned |
| Declaring < Defending ("Double") | **-(2 × bid)** | +defending earned |

**Shelem — all 13 tricks won by declaring team**
```
Declaring team score += 250 (Shelem bonus)
Defending team score += 0
```

### Game End

- A team wins when their cumulative score reaches **≥ 660** at the end of a hand.
- If both teams reach 660 in the same hand, the higher score wins.
- If tied, play continues.
- **Negative scores are possible** — repeated failures push scores below zero.

---

### Scoring Examples

> **Setup**: Declaring team = Player 0·2, Defending team = Player 1·3, Bid = 100

#### How Points Are Calculated

```
Declaring team earned = (card points in won tricks)
                      + (tricks won × 5)
                      + (Zamin card points)
                      + 5  ← Zamin trick (always goes to declaring team)

Defending team earned = (card points in won tricks)
                      + (tricks won × 5)
```

**Example calculation:**
```
Declaring team won: 8 tricks
  → Card points: ♥A(11) + ♠Q(10) + ♦10(5) + others(0) = 26
  → Trick points: 8 × 5 = 40
Zamin cards: ♣10(5) + ♥2(0) + ♠3(0) + ♦4(0) = 5
Zamin trick: +5

Declaring team earned = 26 + 40 + 5 + 5 = 76   ← below bid (100)
Defending team earned = 169 - 76 = 93
```

---

#### Example 1 — Win (bid met)

```
Bid: 100
Declaring team earned: 110  (≥ bid)
Defending team earned:  59

Result:
  Declaring team score += 110
  Defending team score +=  59
```

Both teams receive their actual earned points when the bid is met.

#### Example 2 — Fail (bid missed, declaring ≥ defending)

```
Bid: 100
Declaring team earned:  80  (< bid)
Defending team earned:  75  ← declaring earned more

Condition: declaring(80) ≥ defending(75) → plain fail
Result:
  Declaring team score -= 100  (= 1 × bid)
  Defending team score +=  75
```

Missed the bid but earned more than the defenders → penalty of one bid (-bid).

#### Example 3 — Double (bid missed, declaring < defending)

```
Bid: 100
Declaring team earned:  60  (< bid)
Defending team earned: 105  ← defending earned more

Condition: declaring(60) < defending(105) → double fail
Result:
  Declaring team score -= 200  (= 2 × bid)
  Defending team score += 105
```

Missed the bid and earned less than the defenders → penalty of two bids (-2×bid).

#### Example 4 — Shelem (all 13 tricks including Zamin)

```
Bid: 100
Declaring team wins all 13 tricks (including Zamin)
Defending team earned: 0

Result:
  Declaring team score += 250  (fixed Shelem bonus)
  Defending team score +=   0
```

The Shelem bonus (250) is fixed regardless of bid value or earned points.

---

#### Summary Table

| Bid | Declaring earned | Defending earned | Outcome | Declaring | Defending |
|-----|------------------|------------------|---------|-----------|-----------|
| 100 | 110 |  59 | Win    | +110 | +59  |
| 100 |  80 |  75 | Fail   | -100 | +75  |
| 100 |  60 | 105 | Double | -200 | +105 |
| 100 | 169 |   0 | Shelem | +250 | +0   |

---

## Glossary

| Term | Persian | Description |
|------|---------|-------------|
| Zamin | زمین | The 4 face-down kitty cards set aside after the deal |
| Hâkem / Declarer | حاکم | The player who wins the auction |
| Hokm / Trump | حکم | The trump suit, determined by Hâkem's first card |
| Shelem | شلم | Winning all tricks (grand slam) |
| Hol | سوراخ | Hâkem's weak suit (few or no cards) |
| Khâli Kardan | خالی کردن | Declaring team fails to meet their bid |
| Chagh Kardan | چاق کردن | Loading a trick with high-value cards (A, Q, 10) |
| Arreh Keshi | ارّه‌کشی | Partner leads high cards in sequence to exhaust opponents' suit |

---

## Variants

### Ace-15

- Ace worth **15 points** instead of 11.
- Total per hand = 185 points.
- Maximum bid = 185.
- Config: `configs/ace15.yaml`

### K/Q/J Points

- K=4, Q=3, J=2 added.
- Config: `configs/kqj_variant.yaml`

### 3-Player Shelem

- Remove the 2 of Clubs (51 cards).
- 15 cards each, 6-card Zamin.
- Min bid 95, Shelem bonus 350, game threshold 650.
- Config: `configs/three_player.yaml`

---

## Environment Mapping

| Game Concept | Implementation |
|--------------|----------------|
| Bid | actions 53–69 (`bid_to_action(value, cfg)`) |
| Pass | action 52 (`ACTION_PASS`) |
| Zamin discard | actions 0–51 (card index), repeated 4 times |
| Trump / mode declaration | actions 70–73 (`ACTION_MODE_OFFSET + PlayMode`) |
| Card play | actions 0–51 (card index) |
| Must-follow | enforced by `compute_action_mask()` |
| Scoring | `compute_hand_result()` in `shelem/game/scoring.py` |
