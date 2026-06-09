from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import NTrailerConfig
from .env import NTrailerReverseEnv
from .mppi import MPPIController, rollout_mppi
from .path import Path2D
from .policy import load_policy
from .train_bc import train_and_save
from .utils import write_json


def default_path() -> Path2D:
    return Path2D.from_waypoints([(0, 0), (-3, 0.3), (-6, 1.3), (-9, -0.6), (-12, 0.0)])


def build_config(args: argparse.Namespace) -> NTrailerConfig:
    return NTrailerConfig(n_trailers=args.trailers, seed=args.seed)


def cmd_simulate(args: argparse.Namespace) -> None:
    path = Path2D.from_json(args.path) if args.path else default_path()
    config = build_config(args)
    env = NTrailerReverseEnv(config, path, seed=args.seed)
    controller = MPPIController(env, horizon=args.horizon, samples=args.samples, seed=args.seed)
    trace = rollout_mppi(env, controller, steps=args.steps)
    summary = {
        "config": {"n_trailers": config.n_trailers, "dt": config.dt},
        "path": path.to_json_dict(),
        "trace": trace,
        "final_progress": trace["progress"][-1] if trace["progress"] else 0.0,
        "final_cross_track": trace["cross_track"][-1] if trace["cross_track"] else 0.0,
        "jackknife": any(trace["jackknife"]),
    }
    write_json(args.out, summary)
    print(f"wrote {args.out}")


def cmd_train_bc(args: argparse.Namespace) -> None:
    path = Path2D.from_json(args.path) if args.path else default_path()
    config = build_config(args)
    meta = train_and_save(
        config,
        path,
        args.out,
        episodes=args.episodes,
        steps=args.steps,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"wrote {args.out}")
    print({k: v for k, v in meta.items() if k != "losses"})
    print(f"final_loss={meta['losses'][-1]:.6f}")


def cmd_eval_policy(args: argparse.Namespace) -> None:
    path = Path2D.from_json(args.path) if args.path else default_path()
    config = build_config(args)
    env = NTrailerReverseEnv(config, path, seed=args.seed)
    policy, meta = load_policy(args.policy)
    obs = env.reset(noise=True)
    trace = {"state": [], "action": [], "reward": [], "cross_track": [], "heading_error": [], "progress": [], "jackknife": []}
    for _ in range(args.steps):
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        trace["state"].append(info["raw_state"].tolist())
        trace["action"].append(float(action[0]))
        trace["reward"].append(float(reward))
        trace["cross_track"].append(float(info["cross_track"]))
        trace["heading_error"].append(float(info["heading_error"]))
        trace["progress"].append(float(info["progress"]))
        trace["jackknife"].append(bool(info["jackknife"]))
        if done:
            break
    write_json(args.out, {"metadata": meta, "path": path.to_json_dict(), "trace": trace})
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spherical Gaussian N-trailer reversal baseline")
    sub = parser.add_subparsers(required=True)

    sim = sub.add_parser("simulate")
    sim.add_argument("--trailers", type=int, default=3)
    sim.add_argument("--steps", type=int, default=250)
    sim.add_argument("--horizon", type=int, default=20)
    sim.add_argument("--samples", type=int, default=256)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--path", type=str, default=None)
    sim.add_argument("--out", type=str, default="runs/simulate.json")
    sim.set_defaults(func=cmd_simulate)

    train = sub.add_parser("train-bc")
    train.add_argument("--trailers", type=int, default=3)
    train.add_argument("--episodes", type=int, default=12)
    train.add_argument("--steps", type=int, default=180)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--path", type=str, default=None)
    train.add_argument("--out", type=str, default="runs/policy.pt")
    train.set_defaults(func=cmd_train_bc)

    ev = sub.add_parser("eval-policy")
    ev.add_argument("--trailers", type=int, default=3)
    ev.add_argument("--steps", type=int, default=250)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--path", type=str, default=None)
    ev.add_argument("--policy", type=str, required=True)
    ev.add_argument("--out", type=str, default="runs/eval.json")
    ev.set_defaults(func=cmd_eval_policy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
