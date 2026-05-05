"""
Play Shelem interactively against a model (or random opponent).

Usage
-----
    # vs random opponent
    python examples/play.py

    # vs trained model
    python examples/play.py --model models/ppo_shelem.zip
    python examples/play.py --model models/actor.onnx
    python examples/play.py --model models/net.pt

    # control a different player (default: player_0)
    python examples/play.py --human player_1

    # control your full team (player_0 + player_2 = Team 0)
    python examples/play.py --human player_0 player_2

    # fixed seed, slower model
    python examples/play.py --seed 42 --model-delay 0.8

    # use a variant config
    python examples/play.py --config configs/kqj_variant.yaml
"""
from __future__ import annotations

import argparse

from shelem.config import ShelemConfig
from shelem.policy import load_policy
from shelem.render.human import play_vs_model


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Shelem vs a model")
    p.add_argument("--model",       type=str,   default=None,
                   help="Model path (.zip / .onnx / .pt). None = random opponent")
    p.add_argument("--human",       type=str,   nargs="+",
                   default=["player_0"],
                   help="Agent names you control (default: player_0)")
    p.add_argument("--seed",        type=int,   default=None)
    p.add_argument("--model-delay", type=float, default=0.4,
                   help="Seconds to pause after model actions (default 0.4)")
    p.add_argument("--config",      type=str,   default=None,
                   help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    policy = load_policy(args.model)
    cfg    = ShelemConfig.from_yaml(args.config) if args.config else None

    play_vs_model(
        policy=policy,
        human_agents=args.human,
        config=cfg,
        seed=args.seed,
        model_delay=args.model_delay,
    )
