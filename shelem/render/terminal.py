from __future__ import annotations

import io
import sys
import time
from typing import Callable

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv
from shelem.game.card import INDEX_TO_CARD, PlayMode
from shelem.game.state import PhaseEnum
from shelem.spaces.action import (
    ACTION_MODE_OFFSET,
    ACTION_PASS,
    BID_OFFSET,
    action_to_bid,
    is_card_action,
)

Policy = Callable[[dict, np.ndarray], int]

_SUIT_SYM = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
_SUIT_COLOR = {0: "white", 1: "red", 2: "red", 3: "white"}
_PHASE_LABEL = {
    PhaseEnum.BIDDING:           "BIDDING",
    PhaseEnum.ZAMIN_EXCHANGE:    "ZAMIN EXCHANGE",
    PhaseEnum.TRUMP_DECLARATION: "TRUMP DECLARATION",
    PhaseEnum.PLAY:              "PLAY",
    PhaseEnum.SCORING:           "SCORING",
}

def _make_console() -> Console:
    # On Windows the default codepage (cp949) can't encode many Unicode chars.
    # Redirect output through a UTF-8 wrapper when stdout has an underlying buffer.
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        file = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        return Console(file=file, force_terminal=True)
    return Console()

console = _make_console()


def _card_str(card) -> str:
    return f"{card.rank.label}{_SUIT_SYM[card.suit.value]}"


def _card_text(card) -> Text:
    t = Text()
    t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
    return t


def _action_label(action: int | None, cfg: ShelemConfig) -> str:
    if action is None:
        return "(dead step)"
    if action == ACTION_PASS:
        return "PASS"
    if is_card_action(action):
        return f"play {INDEX_TO_CARD[action]}"
    if BID_OFFSET <= action < ACTION_MODE_OFFSET:
        return f"bid {action_to_bid(action, cfg)}"
    return f"declare {PlayMode(action - ACTION_MODE_OFFSET).name}"


def _build_panel(
    env: ShelemAECEnv,
    last_agent: str,
    last_action: int | None,
    show_all_hands: bool,
) -> Panel:
    state = env._raw.state
    cfg = env._config
    current = env.agent_selection if env.agents else "-"

    # ── header row ──────────────────────────────────────────────────────
    trump_sym = _SUIT_SYM[state.trump_suit.value] if state.trump_suit else "?"
    trump_color = _SUIT_COLOR[state.trump_suit.value] if state.trump_suit else "dim"
    declarer_str = f"player_{state.declarer}" if state.declarer is not None else "-"

    header = Text()
    header.append("Phase: ", style="bold")
    header.append(_PHASE_LABEL.get(state.phase, str(state.phase)), style="bold cyan")
    header.append("   Trump: ", style="bold")
    header.append(trump_sym, style=f"bold {trump_color}")
    header.append("   Declarer: ", style="bold")
    header.append(declarer_str, style="bold magenta")
    header.append("   Bid: ", style="bold")
    header.append(str(state.current_bid) if state.current_bid else "-")

    # ── current trick ────────────────────────────────────────────────────
    trick_label = Text("Current Trick  ", style="bold underline")
    trick_body = Text()
    if state.current_trick:
        for i, (p, card) in enumerate(state.current_trick):
            if i:
                trick_body.append("    ")
            trick_body.append(f"P{p}: ")
            trick_body.append_text(_card_text(card))
    else:
        trick_body.append("(empty)", style="dim")

    trick_line = Text()
    trick_line.append_text(trick_label)
    trick_line.append_text(trick_body)

    # ── hands ────────────────────────────────────────────────────────────
    hand_section = Text("Hands\n", style="bold underline")
    for p in range(cfg.num_players):
        agent = f"player_{p}"
        is_active = agent == current
        row = Text()
        marker = "►" if is_active else " "
        row.append(
            f"  P{p} {marker}  ",
            style="bold yellow" if is_active else "",
        )

        hand = sorted(
            state.hands.get(p, []),
            key=lambda c: (c.suit.value, c.rank.value),
        )

        if show_all_hands or is_active:
            for i, card in enumerate(hand):
                if i:
                    row.append(" ")
                row.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
        else:
            row.append(f"[{len(hand)} cards]", style="dim")

        hand_section.append_text(row)
        hand_section.append("\n")

    # ── score table ───────────────────────────────────────────────────────
    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 3))
    tbl.add_column("", style="bold", min_width=8)
    tbl.add_column("Team 0  (P0+P2)", justify="center", min_width=16)
    tbl.add_column("Team 1  (P1+P3)", justify="center", min_width=16)
    tbl.add_row("Score",  str(state.scores[0]),      str(state.scores[1]))
    tbl.add_row("Tricks", str(state.tricks_won[0]),  str(state.tricks_won[1]))
    tbl.add_row("Points", str(state.points_won[0]),  str(state.points_won[1]))

    # ── last action ───────────────────────────────────────────────────────
    last_line = Text(style="dim")
    if last_action is not None:
        last_line.append(f"  >> {last_agent}: {_action_label(last_action, cfg)}")

    content = Group(
        header,
        Text(""),
        trick_line,
        Text(""),
        hand_section,
        tbl,
        Text(""),
        last_line,
    )

    title = Text()
    title.append("SHELEM", style="bold green")
    title.append("  │  Acting: ")
    title.append(current, style="bold yellow")
    title.append(f"  │  Goal: {cfg.game_threshold} pts")

    return Panel(content, title=title, border_style="green", expand=False)


def watch(
    policy: Policy | None = None,
    config: ShelemConfig | None = None,
    seed: int | None = None,
    delay: float = 0.5,
    show_all_hands: bool = True,
) -> None:
    """
    Watch a policy play a full game of Shelem in the terminal.

    Args:
        policy:         Callable (obs: dict, mask: np.ndarray) -> int.
                        None = uniform random over legal actions.
        config:         ShelemConfig instance. None = default.yaml.
        seed:           RNG seed for reproducibility.
        delay:          Seconds to pause between steps (default 0.5).
        show_all_hands: Show all 4 players' cards (default True).
                        Set False to see only what the acting agent sees.
    """
    import random as _rnd

    cfg = config or ShelemConfig.from_yaml()
    env = ShelemAECEnv(config=cfg)
    env.reset(seed=seed)
    rng = _rnd.Random(seed)

    last_agent: str = ""
    last_action: int | None = None

    with Live(console=console, refresh_per_second=10, screen=False) as live:
        def _refresh() -> None:
            live.update(_build_panel(env, last_agent, last_action, show_all_hands))

        _refresh()
        time.sleep(delay)

        while env.agents:
            agent = env.agent_selection

            if env.terminations.get(agent) or env.truncations.get(agent):
                env.step(None)
                _refresh()
                continue

            obs  = env.observe(agent)
            mask = obs["action_mask"]

            if policy is not None:
                action = int(policy(obs, mask))
            else:
                action = int(rng.choice(mask.nonzero()[0].tolist()))

            last_agent  = agent
            last_action = action

            env.step(action)
            _refresh()
            time.sleep(delay)

    # final summary (outside Live so it persists)
    state = env._raw.state
    console.print()
    console.print("[bold green]--- Game Over ---[/bold green]")
    console.print(
        f"  Team 0: [bold]{state.scores[0]}[/bold]   "
        f"Team 1: [bold]{state.scores[1]}[/bold]"
    )
    winner = 0 if state.scores[0] >= cfg.game_threshold else 1
    console.print(f"  Winner: [bold yellow]Team {winner}[/bold yellow]")
