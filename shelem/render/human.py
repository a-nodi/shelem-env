from __future__ import annotations

import io
import sys
import time

import numpy as np
from rich.console import Console
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from shelem.config import ShelemConfig
from shelem.env.shelem_aec import ShelemAECEnv
from shelem.game.card import INDEX_TO_CARD, PlayMode
from shelem.game.state import PhaseEnum
from shelem.policy.builtins import RandomPolicy
from shelem.spaces.action import (
    ACTION_MODE_OFFSET,
    ACTION_PASS,
    BID_OFFSET,
    action_to_bid,
    is_card_action,
)

_SUIT_SYM = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
_SUIT_COLOR = {0: "white", 1: "red", 2: "red", 3: "white"}
_PHASE_LABEL = {
    PhaseEnum.BIDDING:           "BIDDING",
    PhaseEnum.ZAMIN_EXCHANGE:    "ZAMIN EXCHANGE",
    PhaseEnum.TRUMP_DECLARATION: "TRUMP DECLARATION",
    PhaseEnum.PLAY:              "PLAY",
}
_MODE_LABELS = {
    0: "NORMAL     — first card sets trump, A > K > … > 2",
    1: "NARES      — reversed order: 2 > 3 > … > A, no trump",
    2: "ACE_NARES  — A highest, then 2 > 3 > … > K, no trump",
    3: "SARRES     — standard A-high, no trump",
}


def _make_console() -> Console:
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        file = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        return Console(file=file, force_terminal=True)
    return Console()


console = _make_console()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _card_str(card) -> str:
    return f"{card.rank.label}{_SUIT_SYM[card.suit.value]}"


def _card_text(card) -> Text:
    t = Text()
    t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
    return t


def _action_label(action: int, cfg: ShelemConfig) -> str:
    if action == ACTION_PASS:
        return "PASS"
    if is_card_action(action):
        return _card_str(INDEX_TO_CARD[action])
    if BID_OFFSET <= action < ACTION_MODE_OFFSET:
        return f"bid {action_to_bid(action, cfg)}"
    return f"declare {PlayMode(action - ACTION_MODE_OFFSET).name}"


# ---------------------------------------------------------------------------
# State display
# ---------------------------------------------------------------------------

def _print_header(state, player: int, cfg: ShelemConfig) -> None:
    trump = _SUIT_SYM[state.trump_suit.value] if state.trump_suit else "?"
    declarer = f"player_{state.declarer}" if state.declarer is not None else "-"
    phase = _PHASE_LABEL.get(state.phase, str(state.phase))

    line = Text()
    line.append("Phase: ", style="bold")
    line.append(phase, style="bold cyan")
    line.append("   Trump: ", style="bold")
    line.append(trump)
    line.append("   Declarer: ", style="bold")
    line.append(declarer, style="magenta")
    line.append("   Bid: ", style="bold")
    line.append(str(state.current_bid) if state.current_bid else "-")
    console.print(line)


def _print_scores(state) -> None:
    tbl = Table(show_header=True, header_style="bold", box=None, padding=(0, 3))
    tbl.add_column("", style="bold", min_width=8)
    tbl.add_column("Team 0  (P0+P2)", justify="center", min_width=14)
    tbl.add_column("Team 1  (P1+P3)", justify="center", min_width=14)
    tbl.add_row("Score",  str(state.scores[0]),     str(state.scores[1]))
    tbl.add_row("Tricks", str(state.tricks_won[0]), str(state.tricks_won[1]))
    tbl.add_row("Points", str(state.points_won[0]), str(state.points_won[1]))
    console.print(tbl)


def _print_trick(state) -> None:
    if not state.current_trick:
        console.print("[dim]Current Trick: (empty)[/dim]")
        return
    t = Text("Current Trick:  ")
    for i, (p, card) in enumerate(state.current_trick):
        if i:
            t.append("    ")
        t.append(f"P{p}: ")
        t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
    console.print(t)


def _print_hand(hand: list, label: str = "Your hand") -> None:
    hand_sorted = sorted(hand, key=lambda c: (c.suit.value, c.rank.value))
    t = Text(f"{label} ({len(hand_sorted)} cards):  ")
    for i, card in enumerate(hand_sorted):
        if i:
            t.append("  ")
        t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
    console.print(t)


def _print_state_for_human(state, player: int, cfg: ShelemConfig) -> None:
    console.print()
    _print_header(state, player, cfg)
    console.print()
    _print_scores(state)
    console.print()
    _print_trick(state)
    console.print()
    _print_hand(list(state.hands.get(player, [])))


# ---------------------------------------------------------------------------
# Human input
# ---------------------------------------------------------------------------

def _prompt_bidding(legal: list[int], cfg: ShelemConfig) -> int:
    options: list[tuple[str, int]] = []
    for a in legal:
        if a == ACTION_PASS:
            options.append(("PASS", a))
        else:
            options.append((f"bid {action_to_bid(a, cfg)}", a))

    console.print("[bold]Legal bids:[/bold]")
    cols = 6
    for row_start in range(0, len(options), cols):
        row = options[row_start:row_start + cols]
        parts = [f"  [{row_start + i}] {label:<10}" for i, (label, _) in enumerate(row)]
        console.print("".join(parts))

    choices = [str(i) for i in range(len(options))]
    idx = int(Prompt.ask("[bold yellow]Enter number[/bold yellow]",
                         choices=choices, console=console))
    return options[idx][1]


def _prompt_zamin(legal: list[int], state, player: int) -> int:
    hand = sorted(state.hands.get(player, []),
                  key=lambda c: (c.suit.value, c.rank.value))
    legal_set = set(legal)
    options: list[tuple[str, int]] = []

    for card in hand:
        if card.index in legal_set:
            options.append((_card_str(card), card.index))

    n_done = state.zamin_selected_count
    console.print(
        f"[bold]Discard a card  "
        f"({n_done}/4 discarded so far)[/bold]"
    )
    cols = 8
    for row_start in range(0, len(options), cols):
        row = options[row_start:row_start + cols]
        t = Text()
        for i, (label, _) in enumerate(row):
            card = INDEX_TO_CARD[options[row_start + i][1]]
            t.append(f"  [{row_start + i}] ")
            t.append(label, style=_SUIT_COLOR[card.suit.value])
        console.print(t)

    choices = [str(i) for i in range(len(options))]
    idx = int(Prompt.ask("[bold yellow]Enter number[/bold yellow]",
                         choices=choices, console=console))
    return options[idx][1]


def _prompt_trump(legal: list[int]) -> int:
    console.print("[bold]Declare play mode:[/bold]")
    for i, a in enumerate(legal):
        mode_idx = a - ACTION_MODE_OFFSET
        console.print(f"  [{i}] {_MODE_LABELS[mode_idx]}")

    choices = [str(i) for i in range(len(legal))]
    idx = int(Prompt.ask("[bold yellow]Enter number[/bold yellow]",
                         choices=choices, console=console))
    return legal[idx]


def _prompt_play(legal: list[int], state) -> int:
    options = []
    for a in legal:
        card = INDEX_TO_CARD[a]
        options.append((card, a))
    options.sort(key=lambda x: (x[0].suit.value, x[0].rank.value))

    # show must-follow hint
    if state.current_trick:
        led_suit = state.current_trick[0][1].suit
        console.print(
            f"[dim]Led suit: "
            f"{_SUIT_SYM[led_suit.value]}  "
            f"(you must follow suit if you have it)[/dim]"
        )

    console.print("[bold]Legal cards:[/bold]")
    cols = 8
    for row_start in range(0, len(options), cols):
        row = options[row_start:row_start + cols]
        t = Text()
        for i, (card, _) in enumerate(row):
            t.append(f"  [{row_start + i}] ")
            t.append(_card_str(card), style=_SUIT_COLOR[card.suit.value])
        console.print(t)

    choices = [str(i) for i in range(len(options))]
    idx = int(Prompt.ask("[bold yellow]Enter number[/bold yellow]",
                         choices=choices, console=console))
    return options[idx][1]


def _human_act(state, player: int, legal: list[int], cfg: ShelemConfig) -> int:
    if state.phase == PhaseEnum.BIDDING:
        return _prompt_bidding(legal, cfg)
    if state.phase == PhaseEnum.ZAMIN_EXCHANGE:
        return _prompt_zamin(legal, state, player)
    if state.phase == PhaseEnum.TRUMP_DECLARATION:
        return _prompt_trump(legal)
    return _prompt_play(legal, state)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def play_vs_model(
    policy=None,
    human_agents: list[str] | None = None,
    config: ShelemConfig | None = None,
    seed: int | None = None,
    model_delay: float = 0.4,
) -> None:
    """Play interactively against a model in the terminal.

    Args:
        policy:       Model policy for non-human agents.
                      ``None`` = random opponent.
        human_agents: Agent names controlled by the human.
                      Default: ``["player_0"]``.
        config:       ``ShelemConfig`` instance. ``None`` = ``default.yaml``.
        seed:         RNG seed.
        model_delay:  Seconds to pause after each model action so the
                      human can follow along (default 0.4).

    Example::

        from shelem.policy import load_policy
        from shelem.render.human import play_vs_model

        model = load_policy("models/ppo.zip")
        play_vs_model(policy=model)
    """
    cfg = config or ShelemConfig.from_yaml()
    env = ShelemAECEnv(config=cfg)
    env.reset(seed=seed)

    _human = set(human_agents or ["player_0"])
    _model = policy or RandomPolicy(seed=seed)

    human_team = {
        env.possible_agents.index(a) % 2
        for a in _human
        if a in env.possible_agents
    }
    human_team_str = " & ".join(
        f"Team {t} (P{t}+P{t+2})" for t in sorted(human_team)
    )

    console.print()
    console.print(Rule("[bold green]SHELEM[/bold green]"))
    console.print(
        f"  You control: [bold yellow]{', '.join(sorted(_human))}[/bold yellow]  "
        f"({human_team_str})"
    )
    console.print(f"  Goal: [bold]{cfg.game_threshold}[/bold] points")
    console.print(Rule())

    prev_trick_count = [0]   # track trick completion for separator

    while env.agents:
        agent = env.agent_selection

        if env.terminations.get(agent) or env.truncations.get(agent):
            env.step(None)
            continue

        state = env._raw.state
        player = env._agent_id(agent)
        obs = env.observe(agent)
        mask = obs["action_mask"]
        legal = mask.nonzero()[0].tolist()

        is_human = agent in _human

        # ── trick boundary separator ─────────────────────────────────────
        total_tricks = state.tricks_won[0] + state.tricks_won[1]
        if total_tricks > prev_trick_count[0]:
            console.print(Rule(f"[dim]trick {total_tricks}[/dim]", style="dim"))
            prev_trick_count[0] = total_tricks

        if is_human:
            # ── human turn ───────────────────────────────────────────────
            console.print(Rule(
                f"[bold yellow]YOUR TURN[/bold yellow]  {agent}  "
                f"(Team {cfg.team_of(player)})",
                style="yellow",
            ))
            _print_state_for_human(state, player, cfg)
            console.print()
            action = _human_act(state, player, legal, cfg)
            console.print(
                f"  [bold green]You played:[/bold green] "
                f"{_action_label(action, cfg)}"
            )
        else:
            # ── model turn ───────────────────────────────────────────────
            action = int(_model.act(obs, mask))
            label = _action_label(action, cfg)
            console.print(
                f"  [dim]{agent} (model):[/dim] "
                f"[cyan]{label}[/cyan]"
            )
            time.sleep(model_delay)

        env.step(action)

    # ── final result ─────────────────────────────────────────────────────
    state = env._raw.state
    console.print()
    console.print(Rule("[bold green]Game Over[/bold green]"))
    console.print(
        f"  Team 0: [bold]{state.scores[0]}[/bold]   "
        f"Team 1: [bold]{state.scores[1]}[/bold]"
    )
    winner = 0 if state.scores[0] >= cfg.game_threshold else 1
    you_win = winner in human_team
    if you_win:
        console.print("[bold green]  You win![/bold green]")
    else:
        console.print("[bold red]  Model wins.[/bold red]")
    console.print(Rule())
