"""BayesianBRNEPolicy: ties belief_tracker -> robot_sampler ->
trajectory_sampler -> equilibrium_loop -> first control (Order F5,
2026-08-03 -- guide.md section 4.6).

Single public policy class, exactly the three methods guide.md specifies:

    configure(config, artifact)
    reset(episode_seed)
    predict(observation: schemas.PolicyObservation) -> ActionXY

``predict()`` follows guide.md 4.6's FIXED order every call:

    1. Validate schema/track-id/timestep.
    2. Update belief ONCE using the action ACTUALLY EXECUTED last step
       (never a candidate under consideration this step -- see
       ``belief_tracker.py``'s own docstring on why that distinction
       matters).
    3. Generate robot candidate trajectories (``robot_sampler``).
    4. Run the outer action-conditioned-prior/BRNE fixed-point loop
       (``equilibrium_loop``).
    5. Take the first-step control from the final robot posterior weights
       (``brne_adapter.weighted_first_control``).
    6. Clip ONLY by dynamics limits (never by any other diagnostic score).
    7. Save diagnostics -- they are recorded for inspection, never fed back
       into the action itself.

No Mamba/SARL/LSTM/RL checkpoint is ever loaded here; the whole prediction
is BRNE solved online from the AR-HMM artifact's own parameters.

Scope note (deferred to Order F6, not a silent omission): CrowdNav's
standard ``JointState``/``ObservableState`` classes carry no persistent
per-human track ID, which ``BeliefBank`` requires (guide.md 5.6). This
class operates on ``schemas.PolicyObservation``/``TrackObservation``
(which DO carry track_id) -- bridging CrowdNav's raw environment interface
into that schema, and maintaining track identity across steps, is Order
F6's "CrowdSim中策略所需的track id/last action传递接口" work, not this
file's. Similarly, a human track that is not present in the CURRENT
``observation.humans`` (occluded/out of range this step) does not get a
predicted trajectory fed into this step's BRNE equilibrium -- its belief
is still correctly advanced by ``BeliefBank`` (Bayes prior only), but until
it reappears it does not participate in the robot's collision-avoidance
reasoning for that step. This is a real, stated scope limitation, not
something to discover later by surprise.
"""

from __future__ import annotations

from typing import Dict, Optional
import time

import numpy as np

from crowd_nav.bayesian_brne.action_conditioned_arhmm import ARHMMArtifact
from crowd_nav.bayesian_brne.belief_tracker import BeliefBank
from crowd_nav.bayesian_brne.brne_adapter import BRNESolver, weighted_first_control
from crowd_nav.bayesian_brne.config import PlannerConfig
from crowd_nav.bayesian_brne.equilibrium_loop import run_outer_equilibrium_loop
from crowd_nav.bayesian_brne.robot_sampler import sample_robot_candidates
from crowd_nav.bayesian_brne.schemas import PolicyObservation, TrackObservation
from crowd_nav.bayesian_brne.trajectory_sampler import (
    DEFAULT_MAX_ACCELERATION, DEFAULT_MAX_SPEED, _clip_delta_v_and_speed,
)

STEP = 6
IMPLEMENTED = True


class PolicyValidationError(ValueError):
    """Raised by ``predict()`` when the input observation fails schema/
    identity/timestep checks -- fail closed rather than silently predicting
    from malformed input."""


# Compatibility name only: there is one planner schema, defined in config.py.
# New callers should import PlannerConfig directly. Keeping this alias avoids
# a needless API break for the existing policy factory and selftests.
BayesianBRNEPolicyConfig = PlannerConfig


class BayesianBRNEPolicy:
    def __init__(self):
        self.config: Optional[BayesianBRNEPolicyConfig] = None
        self.artifact: Optional[ARHMMArtifact] = None
        self.belief_bank: Optional[BeliefBank] = None
        self.solver: Optional[BRNESolver] = None
        self._episode_seed: Optional[int] = None
        self._step_index: int = 0
        self._last_executed_action = (0.0, 0.0)
        self._last_timestamp: Optional[float] = None
        self.last_diagnostics: Optional[dict] = None

    def configure(self, config: BayesianBRNEPolicyConfig, artifact: ARHMMArtifact) -> None:
        self.config = config
        self.artifact = artifact
        self.belief_bank = BeliefBank(artifact, max_missed_steps=config.max_missed_steps, seed=config.belief_seed)
        solver_kwargs = {"solver_mode": config.solver_mode, "brne_root": config.brne_root}
        if config.solver_mode == "stable_clearance":
            solver_kwargs.update(
                safe_distance=config.safe_distance,
                cost_sigma=config.cost_sigma,
                cost_scale=config.cost_scale,
            )
        self.solver = BRNESolver(**solver_kwargs)

    def reset(self, episode_seed: int) -> None:
        if self.belief_bank is None:
            raise PolicyValidationError("reset() called before configure()")
        self.belief_bank.reset(episode_seed)
        self._episode_seed = episode_seed
        self._step_index = 0
        self._last_executed_action = (0.0, 0.0)
        self._last_timestamp = None
        self.last_diagnostics = None

    def _validate(self, observation: PolicyObservation) -> None:
        if self.belief_bank is None:
            raise PolicyValidationError("predict() called before configure()")
        if self._episode_seed is None:
            raise PolicyValidationError("predict() called before reset()")
        track_ids = [h.track_id for h in observation.humans]
        if len(track_ids) != len(set(track_ids)):
            raise PolicyValidationError(f"duplicate track_id in observation.humans: {track_ids}")
        if not np.isfinite(observation.timestamp):
            raise PolicyValidationError(f"timestamp must be finite, got {observation.timestamp}")
        if observation.time_step <= 0:
            raise PolicyValidationError(f"time_step must be > 0, got {observation.time_step}")
        if self._last_timestamp is not None:
            global_steps = (observation.timestamp - self._last_timestamp) / observation.time_step
            if abs(global_steps - 1.0) > 1e-6:
                raise PolicyValidationError(
                    f"timestamp must advance by exactly one time_step; previous={self._last_timestamp}, "
                    f"current={observation.timestamp}, step={observation.time_step}, gap/dt={global_steps}"
                )
        for h in observation.humans:
            if abs(h.timestamp - observation.timestamp) > 1e-6:
                raise PolicyValidationError(
                    f"human track_id={h.track_id} timestamp={h.timestamp} does not match "
                    f"observation timestamp={observation.timestamp}"
                )
        for h in observation.humans:
            if not np.isfinite([h.px, h.py, h.vx, h.vy, h.radius]).all():
                raise PolicyValidationError(f"non-finite human observation for track_id={h.track_id}")
        if not np.isfinite([
            observation.robot_px, observation.robot_py, observation.robot_vx, observation.robot_vy,
            observation.robot_gx, observation.robot_gy, observation.robot_v_pref,
        ]).all():
            raise PolicyValidationError("non-finite robot observation")
        # Order R1: artifact.dt is the AR-HMM's own calibrated timestep --
        # Q_k's noise scale and the A/B/C/d dynamics were all fit assuming
        # exactly this much real time elapses per transition. Silently
        # running with a different env time_step would score/roll out
        # under a systematically wrong noise/dynamics assumption.
        if abs(observation.time_step - self.artifact.dt) > 1e-9:
            raise PolicyValidationError(
                f"observation.time_step={observation.time_step} does not match artifact.dt={self.artifact.dt} "
                "-- the AR-HMM's dynamics/noise were calibrated for a specific dt; running at a different "
                "one would silently misuse the model"
            )

    def predict(self, observation: PolicyObservation):
        from crowd_sim.envs.utils.action import ActionXY

        predict_start = time.perf_counter()
        self._validate(observation)

        # Step 2: update belief ONCE using the action ACTUALLY EXECUTED last
        # step (self._last_executed_action -- never a candidate this step's
        # own outer loop is currently evaluating).
        belief_start = time.perf_counter()
        self.belief_bank.update(
            observation.humans, robot_action=self._last_executed_action,
            robot_px=observation.robot_px, robot_py=observation.robot_py,
            robot_vx=observation.robot_vx, robot_vy=observation.robot_vy,
            current_timestamp=observation.timestamp,
        )
        belief_update_ms = (time.perf_counter() - belief_start) * 1000.0

        # Step 3: robot candidate trajectories, seeded deterministically from
        # (episode_seed, step_index) so a rerun with the same seed/episode
        # reproduces byte-identically. Order R1: robot_sampler now returns
        # FOUR separate arrays (actions / PRE-action state positions+
        # velocities / POST-action future positions) -- never one array
        # standing in for two different roles.
        robot_sampling_start = time.perf_counter()
        step_rng = np.random.default_rng([self._episode_seed, self._step_index])
        robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions = sample_robot_candidates(
            robot_px=observation.robot_px, robot_py=observation.robot_py,
            robot_vx=observation.robot_vx, robot_vy=observation.robot_vy,
            goal_gx=observation.robot_gx, goal_gy=observation.robot_gy,
            horizon=self.config.horizon_steps, num_candidates=self.config.num_samples,
            dt=observation.time_step, rng=step_rng, v_pref=observation.robot_v_pref,
            max_speed=self.config.max_speed, max_acceleration=self.config.max_acceleration,
            brne_root=self.config.brne_root,
        )
        robot_sampling_ms = (time.perf_counter() - robot_sampling_start) * 1000.0

        # Step 4: outer action-conditioned-prior / BRNE fixed-point loop --
        # only CURRENTLY-OBSERVED tracks participate this step (see module
        # docstring's scope note on occluded tracks).
        track_ids = [h.track_id for h in observation.humans]
        posteriors_by_track: Dict[int, np.ndarray] = {}
        human_state0_by_track: Dict[int, np.ndarray] = {}
        radii_by_track: Dict[int, float] = {}
        for h in observation.humans:
            posteriors_by_track[h.track_id] = self.belief_bank.predictive_mode_distribution(h.track_id)
            human_state0_by_track[h.track_id] = np.array([h.px, h.py, h.vx, h.vy], dtype=np.float64)
            radii_by_track[h.track_id] = h.radius

        if track_ids:
            outer_start = time.perf_counter()
            outer_result = run_outer_equilibrium_loop(
                posteriors_by_track, human_state0_by_track, self.artifact,
                robot_actions, robot_state_positions, robot_state_velocities, robot_future_positions,
                radii_by_track,
                robot_radius=observation.robot_radius, horizon=self.config.horizon_steps, solver=self.solver,
                sampling_mode=self.config.sampling_mode,
                max_outer_iterations=self.config.max_outer_iterations,
                weight_change_threshold=self.config.outer_tolerance,
                outer_damping=self.config.outer_damping,
                oscillation_tolerance=self.config.oscillation_tolerance,
                equilibrium_iterations=self.config.equilibrium_iterations,
                seed=int(step_rng.integers(0, 2**31 - 1)),
                brne_root=self.config.brne_root,
            )
            outer_loop_ms = (time.perf_counter() - outer_start) * 1000.0
            # Step 5: first-step control from the final robot posterior weights.
            action_selection_start = time.perf_counter()
            control = weighted_first_control(outer_result.robot_weights, robot_actions)
            action_selection_ms = (time.perf_counter() - action_selection_start) * 1000.0
            diagnostics = {
                "outer_iterations": outer_result.outer_iterations,
                "converged": outer_result.converged,
                "status": outer_result.status,
                "max_weight_residual": outer_result.max_weight_residual,
                "final_consistency_residual": outer_result.final_consistency_residual,
                "outer_residual_history": outer_result.outer_residual_history,
                "iteration_diagnostics": outer_result.iteration_diagnostics,
                "final_equilibrium": {
                    "iterations": outer_result.final_equilibrium.iterations,
                    "converged": outer_result.final_equilibrium.converged,
                    "max_weight_change": outer_result.final_equilibrium.max_weight_change,
                    "elapsed_ms": outer_result.final_equilibrium.elapsed_ms,
                    "numeric_fallback_used": outer_result.final_equilibrium.numeric_fallback_used,
                },
                "robot_weights": outer_result.robot_weights,
                "human_weights_by_track": outer_result.human_weights_by_track,
                "timings_ms": {
                    "belief_update": belief_update_ms,
                    "robot_sampling": robot_sampling_ms,
                    "outer_loop": outer_loop_ms,
                    "action_selection": action_selection_ms,
                },
            }
        else:
            # No tracked humans this step: fall back to the goal-directed
            # nominal candidate (candidate 0 -- Order R3, not this order,
            # tracks the separate known gap that candidate 0 currently still
            # carries the same GP noise as every other candidate).
            control = robot_actions[0, 0]
            diagnostics = {"outer_iterations": 0, "converged": True, "status": "no_tracks",
                            "max_weight_residual": 0.0, "final_consistency_residual": 0.0,
                            "outer_residual_history": [], "iteration_diagnostics": [],
                            "robot_weights": None, "human_weights_by_track": {},
                            "timings_ms": {
                                "belief_update": belief_update_ms,
                                "robot_sampling": robot_sampling_ms,
                                "outer_loop": 0.0,
                                "action_selection": 0.0,
                            }}

        # Step 6: clip ONLY by dynamics limits (same helper trajectory_sampler
        # uses, for one consistent definition of "physically realistic").
        v_prev = np.array([observation.robot_vx, observation.robot_vy])
        control = _clip_delta_v_and_speed(
            v_prev, np.asarray(control) - v_prev, observation.time_step,
            self.config.max_speed, self.config.max_acceleration,
        )

        # Step 7: save diagnostics; they do NOT feed back into the action above.
        self.last_diagnostics = diagnostics
        self.last_diagnostics["timings_ms"]["predict_total"] = (time.perf_counter() - predict_start) * 1000.0
        self._last_executed_action = (float(control[0]), float(control[1]))
        self._last_timestamp = observation.timestamp
        self._step_index += 1
        return ActionXY(vx=float(control[0]), vy=float(control[1]))
