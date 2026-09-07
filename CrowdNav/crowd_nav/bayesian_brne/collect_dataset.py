"""Collect SM-BRNE fitting data via the closed-loop interaction protocol.

Single responsibility (guide.md section 3 / 5.4): run episodes with the
interaction_protocol.py behavior FSM and one of the four Order-6 robot
controllers (goal_directed/orca/original_brne/scripted_probe), save via
data_io.py. Does NOT fit -- that happens in tools/fit_bayesian_brne.py +
mode_model.py.

``profile_name`` distinguishes smoke (pipeline regression only) / pilot
(Order 7 coverage-gate only) / formal (Order 8, the only data
fit_bayesian_brne.py may read) runs -- guide.md 10.4/Order 7 are explicit
that smoke and pilot data must never enter formal fitting; that boundary is
enforced by fit_bayesian_brne.py checking this field, not by this script
refusing to run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from crowd_nav.bayesian_brne import data_io
from crowd_nav.bayesian_brne.interaction_protocol import (
    _BRNE_DEFAULT_ROOT,
    SCENARIO_TABLE,
    allocate_controller_type,
    make_scenario,
    run_episode,
)
from crowd_nav.bayesian_brne.schemas import ROBOT_STATE_FIELDS

STEP = 4
IMPLEMENTED = True


def collect(
    split: str,
    scenario: str,
    episodes: int,
    seed: int,
    profile_name: str,
    output_dir: str,
    horizon_steps: int = 40,
    dt: float = 0.25,
    controller: Optional[str] = None,
    brne_root: str = _BRNE_DEFAULT_ROOT,
) -> List[str]:
    """Collect ``episodes`` episodes into ``output_dir/split/scenario/``.

    ``controller=None`` (the default) uses Order 6.1's deterministic equal
    allocation (``allocate_controller_type(episode_index)``); pass an
    explicit controller name only for single-controller diagnostic runs.
    """
    if scenario not in SCENARIO_TABLE:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {sorted(SCENARIO_TABLE)}")

    saved_paths = []
    out_dir = Path(output_dir)
    for ep in range(episodes):
        episode_seed = seed * 100000 + ep
        rng = np.random.default_rng(episode_seed)
        env = make_scenario(scenario, split, rng, dt)
        ep_controller = controller if controller is not None else allocate_controller_type(ep)
        episode = run_episode(
            env, horizon_steps=horizon_steps, controller=ep_controller, scenario=scenario,
            profile=profile_name, split=split, profile_name=profile_name, brne_root=brne_root,
        )
        episode["suite_seed"] = seed
        episode["episode_seed"] = episode_seed

        path = out_dir / split / scenario / f"ep{ep:05d}_seed{episode_seed}.npz"
        data_io.save_episode(str(path), episode, source_robot_state_layout=ROBOT_STATE_FIELDS)
        saved_paths.append(str(path))
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "validation", "test_nominal", "test_heldout_interactive"])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="alias for --profile-name smoke (back-compat)")
    parser.add_argument("--profile-name", default=None, help="smoke | pilot | formal")
    parser.add_argument("--controller", default=None, choices=["goal_directed", "orca", "original_brne", "scripted_probe"])
    parser.add_argument("--output-dir", default="runs/bayesian_brne/data")
    parser.add_argument("--horizon-steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--brne-root", default=_BRNE_DEFAULT_ROOT)
    args = parser.parse_args()

    profile_name = args.profile_name or ("smoke" if args.smoke else None)
    if profile_name is None:
        parser.error("either --smoke or --profile-name {smoke,pilot,formal} is required")

    paths = collect(
        split=args.split, scenario=args.scenario, episodes=args.episodes, seed=args.seed,
        profile_name=profile_name, output_dir=args.output_dir, horizon_steps=args.horizon_steps,
        dt=args.dt, controller=args.controller, brne_root=args.brne_root,
    )
    print(f"[collect_dataset] wrote {len(paths)} episodes under {args.output_dir}/{args.split}/{args.scenario}/")
    for p in paths[:3]:
        print(f"  {p}")
    if len(paths) > 3:
        print(f"  ... and {len(paths) - 3} more")


if __name__ == "__main__":
    main()
