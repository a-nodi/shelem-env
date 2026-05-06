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


def _hand_text(state, player: int, show: bool, per_row: int = 4, max_rows: int = 0) -> Text:
    hand = sorted(state.hands.get(player, []), key=lambda c: (c.suit.value, c.rank.value))
    t = Text()
    if not show:
        t.append(f"({len(hand)} cards)", style="dim")
        for _ in range(max(0, max_rows - 1)):
            t.append("\n")
        return t
    rows = [hand[i:i + per_row] for i in range(0, len(hand), per_row)]
    total = max(len(rows), max_rows)
    for r in range(total):
        if r > 0:
            t.append("\n")
        if r < len(rows):
            for c, card in enumerate(rows[r]):
                if c > 0:
                    t.append(" ")
                t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
    return t


def _bid_label(state, player: int) -> tuple[str, str]:
    """Return (label_text, style) for a player's bidding status."""
    if state.passed[player]:
        return "PASS", "dim"
    bids = [b for b in state.bid_history[player] if b]
    if bids:
        return f"bid {max(bids)}", "bold green"
    return "—", "dim"


def _player_panel(state, player: int, show: bool, per_row: int, active: bool) -> Panel:
    team = player % 2
    # Always 2 display chars so the panel width never changes on turn switch
    marker = " ◄" if active else "  "
    title = Text()
    title.append(f"P{player} · Team {team}{marker}", style="bold yellow" if active else "bold")

    # During bidding, pad label to fixed width so title length stays constant
    if state.phase == PhaseEnum.BIDDING:
        label, style = _bid_label(state, player)
        title.append(f"  [{label:<7}]", style=style)

    # Fixed height: always render ceil(hand_size / per_row) rows
    max_rows = (state.config.hand_size + per_row - 1) // per_row

    border = "yellow" if active else ("blue" if team == 0 else "white")
    return Panel(_hand_text(state, player, show, per_row, max_rows),
                 title=title, border_style=border, padding=(0, 1))


def _center_panel(state, cfg) -> Panel:
    """Center panel: same dimensions as player panels for a square cross layout.

    Width  = len("Zamin Exchange") + 4 = 18  (title-driven, matches player panel width)
    Height = ceil(hand_size / 2) content lines  (matches player panel row count with per_row=2)
    """
    _cw    = len("Zamin Exchange") + 4          # 18 for all configs
    _lines = (cfg.hand_size + 1) // 2           # 6 (default) / 8 (3-player)

    if state.phase == PhaseEnum.BIDDING:
        t = Text()
        t.append("Bid  ", style="bold")
        t.append(str(state.current_bid) if state.current_bid else "—", style="bold cyan")
        t.append("\n")
        for p in range(cfg.num_players):
            label, style = _bid_label(state, p)
            t.append(f"P{p}: ")
            t.append(label, style=style)
            t.append("\n")
        for _ in range(_lines - 1 - cfg.num_players):
            t.append("\n")
        return Panel(t, title="Bidding", border_style="cyan", padding=(0, 1), width=_cw)

    if state.phase == PhaseEnum.ZAMIN_EXCHANGE:
        zamin_cards = sorted(state.zamin, key=lambda c: (c.suit.value, c.rank.value))
        discarded   = sorted(state.zamin_discards, key=lambda c: (c.suit.value, c.rank.value))
        remaining   = cfg.zamin_discard_count - state.zamin_selected_count
        zamin_rows  = (len(zamin_cards) + 1) // 2
        disc_rows   = (cfg.zamin_discard_count + 1) // 2

        t = Text()
        t.append("Zamin\n", style="bold")
        for i in range(0, len(zamin_cards), 2):
            t.append("  ")
            for card in zamin_cards[i : i + 2]:
                t.append_text(_card_text(card))
                t.append(" ")
            t.append("\n")
        t.append("Discard", style="bold")
        if remaining > 0:
            t.append(f"  {remaining}", style="dim")
        t.append("\n")
        for i in range(0, disc_rows * 2, 2):
            row = discarded[i : i + 2]
            t.append("  ")
            if row:
                for card in row:
                    t.append_text(_card_text(card))
                    t.append(" ")
            else:
                t.append("—", style="dim")
            t.append("\n")
        for _ in range(_lines - 1 - zamin_rows - 1 - disc_rows):
            t.append("\n")
        return Panel(t, title="Zamin Exchange", border_style="magenta", padding=(0, 1), width=_cw)

    # PLAY and other phases: show current trick or last completed trick
    is_last      = not state.current_trick and bool(state.last_completed_trick)
    display_trick = state.current_trick or state.last_completed_trick

    players_played = {p: c for p, c in display_trick}
    t = Text()
    t.append("last " if is_last else "     ", style="dim")
    t.append("\n")
    for p in range(cfg.num_players):
        card = players_played.get(p)
        if card is not None:
            t.append(f"P{p}: ")
            t.append_text(_card_text(card))
        else:
            t.append(f"P{p}: ", style="dim")
            t.append("—", style="dim")
        t.append("\n")
    for _ in range(_lines - 1 - cfg.num_players):
        t.append("\n")
    return Panel(t, title="Trick", border_style="green", padding=(0, 1), width=_cw)


def _build_panel(
    env: ShelemAECEnv,
    last_agent: str,
    last_action: int | None,
    show_all_hands: bool,
) -> Panel:
    state = env._raw.state
    cfg = env._config
    current = env.agent_selection if env.agents else "-"

    def active(p: int) -> bool:
        return f"player_{p}" == current

    def show(p: int) -> bool:
        return show_all_hands or active(p)

    # ── header ───────────────────────────────────────────────────────────
    trump_sym   = _SUIT_SYM[state.trump_suit.value] if state.trump_suit else "?"
    trump_color = _SUIT_COLOR[state.trump_suit.value] if state.trump_suit else "dim"
    decl_str    = f"P{state.declarer}" if state.declarer is not None else "-"

    header = Text()
    header.append("Phase: ", style="bold")
    header.append(_PHASE_LABEL.get(state.phase, str(state.phase)), style="bold cyan")
    header.append("   Trump: ", style="bold")
    header.append(trump_sym, style=f"bold {trump_color}")
    header.append("   Declarer: ", style="bold")
    header.append(decl_str, style="bold magenta")
    header.append("   Bid: ", style="bold")
    header.append(str(state.current_bid) if state.current_bid else "-")

    # ── cross layout (all player panels same width) ──────────────────────
    #
    #       [ P2 (North) ]
    #  [P3]  [Center]  [P1]
    #       [ P0 (South) ]
    #
    mid = Table(show_header=False, box=None, padding=(0, 0))
    mid.add_column("west",   no_wrap=True)
    mid.add_column("center", no_wrap=True)
    mid.add_column("east",   no_wrap=True)
    mid.add_row(
        "",
        _player_panel(state, 2, show(2), per_row=2, active=active(2)),
        "",
    )
    mid.add_row(
        _player_panel(state, 3, show(3), per_row=2, active=active(3)),
        _center_panel(state, cfg),
        _player_panel(state, 1, show(1), per_row=2, active=active(1)),
    )
    mid.add_row(
        "",
        _player_panel(state, 0, show(0), per_row=2, active=active(0)),
        "",
    )

    # ── score history + current hand ─────────────────────────────────────
    _OUTCOME_STYLE = {
        "WIN":    "green",
        "FAIL":   "red",
        "DOUBLE": "bold red",
        "SHELEM": "bold yellow",
    }

    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    tbl.add_column("#",      justify="right",  style="dim", min_width=2)
    tbl.add_column("Bid",    justify="right",               min_width=3)
    tbl.add_column("Result", justify="left",                min_width=17)
    tbl.add_column("T0 Δ",   justify="right",               min_width=5)
    tbl.add_column("T1 Δ",   justify="right",               min_width=5)
    tbl.add_column("→ T0",   justify="right",               min_width=5)
    tbl.add_column("→ T1",   justify="right",               min_width=5)

    for entry in state.score_log:
        d0, d1 = entry["delta"]
        s0, s1 = entry["scores"]
        outcome = entry["outcome"]
        st = _OUTCOME_STYLE.get(outcome, "")
        result = f"T{entry['declarer_team']} {outcome}"
        tbl.add_row(
            str(entry["hand"]),
            str(entry["bid"]),
            f"[{st}]{result}[/{st}]" if st else result,
            f"[{st}]{d0:+}[/{st}]" if st else f"{d0:+}",
            f"[{st}]{d1:+}[/{st}]" if st else f"{d1:+}",
            str(s0),
            str(s1),
        )

    # current hand in-progress — fixed-width format keeps the table width stable
    tw = state.tricks_won
    pw = state.points_won
    result_now = f"T:{tw[0]:>2}/{tw[1]:<2} P:{pw[0]:>3}/{pw[1]:<3}"
    tbl.add_row(
        "now",
        str(state.current_bid) if state.current_bid else "-",
        f"[dim]{result_now}[/dim]",
        "",
        "",
        f"[bold]{state.scores[0]}[/bold]",
        f"[bold]{state.scores[1]}[/bold]",
    )

    # ── last action ───────────────────────────────────────────────────────
    last_line = Text(style="dim")
    if last_action is not None:
        last_line.append(f"  >> {last_agent}: {_action_label(last_action, cfg)}")

    content = Group(
        header,
        Text(""),
        mid,
        Text(""),
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
