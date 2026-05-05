"""
Watch a Shelem policy play in the terminal.

Quick start
-----------
    # random agent
    python examples/watch.py

    # fixed seed, slow playback
    python examples/watch.py --seed 42 --delay 1.0

    # load a trained model (auto-detects .zip / .onnx / .pt)
    python examples/watch.py --model models/ppo_shelem.zip

    # record & replay
    python examples/watch.py --record replays/game_001.json
    python examples/watch.py --replay replays/game_001.json

    # hide opponent hands (see only what each agent sees)
    python examples/watch.py --hide-hands

    # use a variant config
    python examples/watch.py --config configs/kqj_variant.yaml

Plug in your own model
----------------------
Edit the ``custom_policy`` function below.  It receives the observation
dict and the action mask, and must return a legal action index (int).

The simplest integration — override ``custom_policy``:

    import numpy as np
    from shelem.policy import flatten_obs   # obs dict → (198,) float32

    def custom_policy(obs, mask):
        flat = flatten_obs(obs)             # flatten for your model
        logits = my_model(flat)             # your inference here
        logits[mask == 0] = -1e9            # mask illegal actions
        return int(np.argmax(logits))

Or subclass ShelemPolicy for a clean interface:

    from shelem.policy import ShelemPolicy

    class MyPolicy(ShelemPolicy):
        def act(self, obs, mask):
            ...
            return action

    custom_policy = MyPolicy()
"""
from __future__ import annotations

import argparse

from shelem.config import ShelemConfig
from shelem.policy import load_policy
from shelem.render.recorder import load_episode, record, replay, save_episode
from shelem.render.terminal import watch


# ---------------------------------------------------------------------------
# Your model goes here
# ---------------------------------------------------------------------------
# Replace None with a path string or a callable to use your own model:
#
#   custom_policy = "models/ppo_shelem.zip"         # SB3 (.zip)
#   custom_policy = "models/actor.onnx"             # ONNX
#   custom_policy = "models/net.pt"                 # PyTorch
#   custom_policy = lambda obs, mask: mask.nonzero()[0][0]  # custom fn
#
custom_policy = None   # None → random agent
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a Shelem policy play")
    p.add_argument("--model",      type=str,   default=None,
                   help="Path to model file (.zip / .onnx / .pt/.pth)")
    p.add_argument("--seed",       type=int,   default=None)
    p.add_argument("--delay",      type=float, default=0.5,
                   help="Seconds between steps (default 0.5)")
    p.add_argument("--config",     type=str,   default=None,
                   help="Path to YAML config file")
    p.add_argument("--record",     type=str,   default=None,
                   help="Record the episode and save to this JSON path")
    p.add_argument("--replay",     type=str,   default=None,
                   help="Replay a previously recorded episode from this JSON path")
    p.add_argument("--hide-hands", action="store_true",
                   help="Show only the acting agent's cards")
    return p.parse_args()


def main() -> None:
    args = _parse()

    # ── replay mode ─────────────────────────────────────────────────────
    if args.replay:
        ep = load_episode(args.replay)
        print(f"Replaying: {args.replay}  "
              f"(seed={ep.seed}, {ep.total_steps} steps, "
              f"winner=Team {ep.winner})")
        replay(ep, delay=args.delay, show_all_hands=not args.hide_hands)
        return

    # ── resolve policy ───────────────────────────────────────────────────
    source = args.model or custom_policy
    policy = load_policy(source)

    cfg = ShelemConfig.from_yaml(args.config) if args.config else None

    # ── record + watch, or just watch ────────────────────────────────────
    if args.record:
        print(f"Recording episode → {args.record}")
        ep = record(policy=policy, config=cfg, seed=args.seed)
        save_episode(ep, args.record)
        print(f"Saved ({ep.total_steps} steps, winner=Team {ep.winner})")
        print("Replaying...")
        replay(ep, delay=args.delay, show_all_hands=not args.hide_hands)
    else:
        watch(
            policy=policy,
            config=cfg,
            seed=args.seed,
            delay=args.delay,
            show_all_hands=not args.hide_hands,
        )


if __name__ == "__main__":
    main()
