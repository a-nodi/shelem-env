from __future__ import annotations

from shelem.game.state import GameState


def compute_hand_result(state: GameState) -> tuple[int, int]:
    """
    Return (delta_team0, delta_team1) to add to accumulated scores.
    Implements §7.2 of shelem_rules.md.
    """
    assert state.declarer is not None
    declarer_team = state.config.team_of(state.declarer)
    defender_team = 1 - declarer_team

    declarer_earned = state.points_won[declarer_team]
    defender_earned = state.points_won[defender_team]
    bid = state.current_bid

    # Shelem: declarer won all tricks including zamin (§7.2)
    # zamin counts as trick 0, so total = tricks_per_hand + 1
    total_tricks = state.config.tricks_per_hand + 1
    if state.tricks_won[declarer_team] == total_tricks:
        deltas = [0, 0]
        deltas[declarer_team] = state.config.shelem_bonus
        return deltas[0], deltas[1]

    deltas = [0, 0]

    if declarer_earned >= bid:
        # success (§7.2 Win)
        deltas[declarer_team] = declarer_earned
        deltas[defender_team] = defender_earned
    elif declarer_earned < defender_earned:
        # doubled (§7.2 Doubled)
        deltas[declarer_team] = -(2 * bid)
        deltas[defender_team] = defender_earned
    else:
        # plain fail (§7.2 Fail)
        deltas[declarer_team] = -bid
        deltas[defender_team] = defender_earned

    return deltas[0], deltas[1]


def apply_hand_result(state: GameState) -> None:
    """Mutate state.scores and set hand_over / game_over flags."""
    d0, d1 = compute_hand_result(state)
    state.scores[0] += d0
    state.scores[1] += d1
    state.hand_over = True

    # game ends when a team reaches/exceeds threshold, or drops below -threshold (§7.3)
    t = state.config.game_threshold
    if state.scores[0] >= t or state.scores[1] >= t:
        # if both reach threshold, higher score wins; if equal, continue
        if state.scores[0] != state.scores[1]:
            state.game_over = True
    elif state.scores[0] <= -t or state.scores[1] <= -t:
        # bankrupt: a team deep in debt loses
        state.game_over = True
