#!/usr/bin/env python3
"""GPU-only teacher-equivalence check (Round 12).

Verifies ``BeliefMDPFeatureEngine.teacher_scores()`` agrees with Mamba's own
production-validated one-step lookahead (``MambaRLPolicy.predict_sarl_style``,
the same code path ``test.py`` uses) on real states drawn from real rollouts
-- not just the first frame of an episode, where an empty-history edge case
could hide a padding bug, and not a synthetic/mocked state, which would not
have caught the specific bug this check exists for.

Round 12 found ``teacher_scores()`` was dropping the *current* frame from
the sequence fed to Mamba (an earlier version excluded it, reasoning that
each candidate action's own next-step token made it redundant -- it does
not; next_token is one step *ahead* of the current frame, not a substitute
for it). Confirmed on 60 real states to change the teacher's top-1 action
on 36/60 of them (60%). This check would have caught that: it captures the
exact token tensor each function feeds to ``mamba.forward_value`` (via a
temporary monkeypatch, no changes to either function) and requires them to
match near-exactly, plus requires the two functions' top-1 action choice to
agree on every single checked state.

Not a substitute for ``selftest.py`` or ``--smoke``: this requires CUDA,
mamba_ssm, and CrowdSim, so it cannot run on the local (3060) dev machine --
run it on the 4090. Exits nonzero on any failure.

    python3 -u belief_mdp/test_teacher_equivalence.py
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CROWD_NAV_DIR = THIS_DIR.parent
REPO_ROOT = CROWD_NAV_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crowd_nav.belief_mdp.runtime import (  # noqa: E402
    BeliefMDPFeatureEngine,
    FullCrowdNavigationEnvironment,
    sort_humans_by_ttc,
)
from crowd_nav.belief_space_rl.runtime import build_frozen_mamba, merged_policy_config  # noqa: E402
from crowd_sim.envs.utils.state import JointState  # noqa: E402

DEFAULT_POLICY_CONFIG = "configs/policy_bayesian_fullcrowd_tail.config"
DEFAULT_BASE_ENV_CONFIG = "configs/env_belief_mdp.config"
DEFAULT_BASE_CHECKPOINT = "runs/mamba_vl/rl_model_ep10000_T24.pth"
DEFAULT_GDBN_PARAMS = "runs/bayesian_distributional/gdbn_params_cv_residual_k3"

# Multiple scenarios and >=1 full reset each -- exercises both the "history
# still building up" phase (first few steps of an episode, where the earlier
# padding logic differs most) and the steady-state (seq_len already full)
# phase, across more than one initial layout.
SCENARIOS = ["baseline_circle"] * 6
STEPS_PER_SCENARIO = 60
MIN_STATES_REQUIRED = 100
INPUT_TOKEN_ATOL = 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_config', default=DEFAULT_POLICY_CONFIG)
    parser.add_argument('--base_env_config', default=DEFAULT_BASE_ENV_CONFIG)
    parser.add_argument('--env_config', default=DEFAULT_BASE_ENV_CONFIG)
    parser.add_argument('--base_checkpoint', default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument('--gdbn_params', default=DEFAULT_GDBN_PARAMS)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from crowd_nav.belief_mdp.protocol import teacher_fingerprint
    # A failed attempt must not leave an earlier passing receipt usable.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'passed': False, 'status': 'started'}) + '\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit(
            "This check requires CUDA (mamba_ssm has no CPU path) -- run it on the 4090, "
            "not the local dev machine."
        )

    config = merged_policy_config(args.policy_config, args.base_env_config)
    mamba = build_frozen_mamba(config, args.base_checkpoint, device)
    mamba.set_phase("test")
    mamba.capture_lookahead_scores = True
    mamba.test_action_smoothing = 0.0  # isolate pure lookahead ranking from test-time action smoothing
    mamba.build_action_space(1.0)

    engine = BeliefMDPFeatureEngine(
        mamba, args.gdbn_params, device,
        belief_mode="recursive", K=3, n_particles=50, num_humans=5, seed=2407,
    )

    captured = []
    original_forward_value = mamba.forward_value

    def capturing_forward_value(tensor, *args, **kwargs):
        result = original_forward_value(tensor, *args, **kwargs)
        captured.append((tensor.detach().clone(), result.detach().clone()))
        return result

    mamba.forward_value = capturing_forward_value

    checked = 0
    mismatched_tensor = 0
    mismatched_top1 = 0
    mismatched_value = 0
    mismatched_score = 0
    max_tensor_diff = 0.0

    try:
        for scenario_index, scenario in enumerate(SCENARIOS):
            environment = FullCrowdNavigationEnvironment(
                args.env_config, scenario, robot_visible=False
            )
            engine.reset()
            mamba.reset_episode_stats()
            robot, humans = environment.reset(
                seed=9_000_000 + scenario_index, profile="nominal", test_case=scenario_index,
            )
            engine.encode(robot, humans)  # seed engine.history with the first frame

            for _ in range(STEPS_PER_SCENARIO):
                robot = environment.robot
                humans = list(environment.env.humans)
                top5 = sort_humans_by_ttc(robot, humans)
                state = JointState(
                    robot.get_full_state(), [h.get_observable_state() for h in top5[:5]],
                )
                if mamba.reach_destination(state):
                    break

                captured.clear()
                teacher_score = engine.teacher_scores(robot, humans)
                if len(captured) != 1:
                    raise SystemExit(
                        f"FAILED: expected exactly one forward_value call from teacher_scores(), "
                        f"got {len(captured)}"
                    )
                tensor_a, value_a = captured[0]

                captured.clear()
                action = mamba.predict_sarl_style(state)
                if len(captured) != 1:
                    raise SystemExit(
                        f"FAILED: expected exactly one forward_value call from predict_sarl_style(), "
                        f"got {len(captured)}"
                    )
                tensor_b, value_b = captured[0]
                production_scores = mamba._last_lookahead_scores.detach().cpu().numpy()
                if not np.allclose(teacher_score, production_scores, atol=1e-4, rtol=1e-6):
                    mismatched_score += 1
                if (value_a.shape != value_b.shape or not torch.allclose(
                        value_a, value_b, atol=1e-5, rtol=1e-5)):
                    mismatched_value += 1

                checked += 1
                if tensor_a.shape != tensor_b.shape:
                    mismatched_tensor += 1
                    print(f"[FAIL] state {checked}: input-token shape mismatch {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}")
                else:
                    diff = float((tensor_a - tensor_b).abs().max())
                    max_tensor_diff = max(max_tensor_diff, diff)
                    if diff > INPUT_TOKEN_ATOL:
                        mismatched_tensor += 1
                        print(f"[FAIL] state {checked}: input-token max abs diff {diff:.6f} (want < {INPUT_TOKEN_ATOL})")

                top1_a = int(np.argmax(teacher_score))
                distances = np.sum((engine.actions - np.array([action.vx, action.vy])) ** 2, axis=-1)
                top1_b = int(np.argmin(distances))
                if top1_a != top1_b or float(distances[top1_b]) > 1e-8:
                    mismatched_top1 += 1
                    print(
                        f"[FAIL] state {checked}: top-1 action mismatch -- teacher_scores()={top1_a}, "
                        f"predict_sarl_style()={top1_b} (nearest-grid-point distance={float(distances[top1_b]):.2e})"
                    )

                # Advance the real trajectory with the teacher's own choice so
                # later states are genuinely reachable, not synthetic.
                result = environment.step(action)
                if result.done:
                    break
                engine.encode(environment.robot, list(environment.env.humans))
    finally:
        mamba.forward_value = original_forward_value

    print(f"\n[TEACHER-EQUIVALENCE] checked {checked} real states across {len(SCENARIOS)} scenarios")
    print(f"[TEACHER-EQUIVALENCE] max input-token abs diff: {max_tensor_diff:.8f} (want < {INPUT_TOKEN_ATOL})")
    print(f"[TEACHER-EQUIVALENCE] input-token mismatches: {mismatched_tensor}/{checked}")
    print(f"[TEACHER-EQUIVALENCE] top-1 action mismatches: {mismatched_top1}/{checked}")

    if checked < MIN_STATES_REQUIRED:
        raise SystemExit(
            f"FAILED: only checked {checked} real states, need >= {MIN_STATES_REQUIRED} "
            "(increase SCENARIOS/STEPS_PER_SCENARIO)"
        )
    if mismatched_tensor > 0 or mismatched_top1 > 0 or mismatched_value > 0 or mismatched_score > 0:
        raise SystemExit(
            f"FAILED: {mismatched_tensor} input-token mismatches and {mismatched_top1} top-1 action "
            f"mismatches out of {checked} real states -- teacher_scores() disagrees with Mamba's own "
            f"production lookahead; value mismatches={mismatched_value}, score mismatches={mismatched_score}."
        )
    output.write_text(json.dumps(dict(
        passed=True, states=checked, tensor_mismatches=mismatched_tensor,
        value_mismatches=mismatched_value, top1_mismatches=mismatched_top1,
        score_mismatches=mismatched_score,
        max_tensor_diff=max_tensor_diff, fingerprint=teacher_fingerprint(args),
        scenarios=SCENARIOS, robot_visible=False), indent=2) + '\n')
    print("[TEACHER-EQUIVALENCE] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
