#!/usr/bin/env python3
"""
Minimax agent (alpha-beta pruning) for Shelem with terminal visualization.

All 4 players are controlled by the same minimax agent operating on the full
game state (perfect-information search via RawEnv.clone()).

Usage:
    python examples/minimax_watch.py
    python examples/minimax_watch.py --seed 7 --delay 0.3 --depth 6
    python examples/minimax_watch.py --hide-hands
"""
from __future__ import annotations

import argparse
import time

from shelem.config import ShelemConfig
from shelem.env.raw_env import RawEnv
from shelem.env.shelem_aec import ShelemAECEnv
from shelem.game.card import INDEX_TO_CARD
from shelem.game.state import PhaseEnum
from shelem.render.terminal import _build_panel, console
from shelem.spaces.action import ACTION_PASS, ACTION_MODE_OFFSET, BID_OFFSET
from rich.live import Live


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(state) -> float:
    """Heuristic: current-hand card-point differential from Team 0's view."""
    return float(state.points_won[0] - state.points_won[1])


# ---------------------------------------------------------------------------
# Alpha-beta minimax (PLAY phase only)
# ---------------------------------------------------------------------------

def _minimax(env: RawEnv, depth: int, alpha: float, beta: float) -> float:
    state = env.state
    if depth == 0 or state.phase != PhaseEnum.PLAY:
        return _evaluate(state)

    legal = env.legal_actions()
    if not legal:
        return _evaluate(state)

    maximizing = state.current_agent % 2 == 0  # team 0 maximizes

    if maximizing:
        best = float("-inf")
        for action in legal:
            child = env.clone()
            child.step(action)
            v = _minimax(child, depth - 1, alpha, beta)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best
    else:
        best = float("inf")
        for action in legal:
            child = env.clone()
            child.step(action)
            v = _minimax(child, depth - 1, alpha, beta)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break
        return best


def _best_play_action(env: RawEnv, depth: int) -> int:
    state = env.state
    maximizing = state.current_agent % 2 == 0
    legal = env.legal_actions()

    best_action = legal[0]
    best_val = float("-inf") if maximizing else float("inf")

    for action in legal:
        child = env.clone()
        child.step(action)
        v = _minimax(child, depth - 1, float("-inf"), float("inf"))
        if maximizing and v > best_val:
            best_val = v
            best_action = action
        elif not maximizing and v < best_val:
            best_val = v
            best_action = action

    return best_action


# ---------------------------------------------------------------------------
# Heuristics for non-PLAY phases
# ---------------------------------------------------------------------------

_ZERO_POINT_RANKS = {2, 3, 4, 5, 6, 7, 8, 9, 11, 13}  # ranks with 0 card points


def _heuristic_action(env: RawEnv) -> int:
    state = env.state
    legal = env.legal_actions()
    phase = state.phase

    if phase == PhaseEnum.BIDDING:
        hand = state.hands[state.current_agent]
        point_cards = sum(1 for c in hand if c.rank.value in (14, 12, 10))
        pass_acts = [a for a in legal if a == ACTION_PASS]
        bid_acts = sorted(a for a in legal if BID_OFFSET <= a < ACTION_MODE_OFFSET)
        if point_cards >= 4 and bid_acts:
            return bid_acts[0]
        return pass_acts[0] if pass_acts else legal[0]

    if phase == PhaseEnum.ZAMIN_EXCHANGE:
        for action in legal:
            if INDEX_TO_CARD[action].rank.value in _ZERO_POINT_RANKS:
                return action
        return legal[0]

    # TRUMP_DECLARATION: choose NORMAL (first option)
    return legal[0]


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def run(seed: int | None = None, delay: float = 0.5, depth: int = 4,
        show_all_hands: bool = True) -> None:
    cfg = ShelemConfig.from_yaml()
    env = ShelemAECEnv(config=cfg)
    env.reset(seed=seed)
    raw: RawEnv = env._raw

    last_agent: str = ""
    last_action: int | None = None

    with Live(console=console, refresh_per_second=10, screen=False) as live:
        def refresh() -> None:
            live.update(_build_panel(env, last_agent, last_action, show_all_hands))

        refresh()
        time.sleep(delay)

        while env.agents:
            agent = env.agent_selection

            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
                refresh()
                continue

            state = raw.state
            if state.phase == PhaseEnum.PLAY:
                action = _best_play_action(raw, depth)
            else:
                action = _heuristic_action(raw)

            last_agent = agent
            last_action = action
            env.step(action)
            refresh()
            time.sleep(delay)

    state = raw.state
    console.print()
    console.print("[bold green]--- Game Over ---[/bold green]")
    console.print(
        f"  Team 0: [bold]{state.scores[0]}[/bold]   "
        f"Team 1: [bold]{state.scores[1]}[/bold]"
    )
    winner = 0 if state.scores[0] >= cfg.game_threshold else 1
    console.print(f"  Winner: [bold yellow]Team {winner}[/bold yellow]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a Minimax agent play Shelem")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds between steps (default 0.5)")
    parser.add_argument("--depth",      type=int,   default=4,
                        help="Alpha-beta search depth in individual card plays (default 4 = ~1 trick ahead)")
    parser.add_argument("--hide-hands", action="store_true",
                        help="Show only the acting player's hand")
    args = parser.parse_args()

    run(
        seed=args.seed,
        delay=args.delay,
        depth=args.depth,
        show_all_hands=not args.hide_hands,
    )
