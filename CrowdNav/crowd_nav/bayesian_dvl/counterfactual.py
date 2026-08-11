"""Multi-step counterfactual rollout for candidate-action ranking
(guide.md "R4-3 冻结契约", 2026-08-10).

Given the belief tracker at a real decision, this module answers "if the
robot committed to candidate action a and then held that same velocity
for ROLLOUT_HORIZON steps, how would the next 8 steps plausibly unfold
under the current posterior over human intent?" -- used to RANK
candidates for later counterfactual-supervision purposes (guide.md R4-4),
NOT to make the real decision (that remains the single-step Q(s,b,a)
scoring path in policy.py, unchanged by this module).

Frozen rules (guide.md R4-3, do not deviate without re-freezing there):

1. Horizon is exactly 8 steps, dt = artifact.dt.
2. The candidate action is a constant-velocity primitive held for all 8
   steps -- no re-planning mid-rollout, no swapping which continuation
   rule is used.
3. Per world sample, each human's latent mode is drawn ONCE (step 0)
   from the current belief/posterior_source, then evolved for the
   remaining 7 steps strictly via artifact.transition_matrix -- never
   re-sampled independently from the original belief at every step.
4. All 80 candidate actions share the SAME human world samples (mode
   sequence + Student-t noise + resulting motion) -- robot action does
   not affect human transition dynamics (guide.md 4.5), so this sampling
   happens exactly once per decision, not once per candidate.
5. Candidates are ranked by collision_prob ascending first (never
   negotiable -- guide.md R4-4R2's non-inferiority requirement), then,
   WITHIN a clearance_tolerance-wide band of that safety tier's best
   clearance, by expected_progress descending then control_cost
   ascending; candidates outside that band keep strict
   lower_tail_clearance-descending priority (guide.md R4-4R3 fix,
   2026-08-10: the original rule had zero tolerance on clearance, so a
   noise-scale clearance edge could veto a much larger real progress
   advantage -- see ``rank_counterfactual_candidates``'s docstring for
   the real audit example this was found from).
6. Terminal absorption: once a world's rollout terminates (collision or
   success) at step t, steps t+1..8 are frozen -- no further geometry
   changes, no further progress accumulation.
7. A CounterfactualRecord may only be built from what is knowable AT the
   real decision instant: the current belief, the frozen artifact, and
   the action table. It must never read the real future, any latent
   ground-truth behavior label, or heldout profile parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation
from crowd_nav.bayesian_dvl.policy import compute_action_features_array
from crowd_nav.bayesian_dvl.rollout import (
    POSTERIOR_SOURCES, RolloutError, _integrate_holonomic, _mode_probabilities, _stable_seed,
)
from crowd_nav.bayesian_dvl.transition import EVENT_COLLISION, RewardConfig, step as transition_step
from crowd_nav.bayesian_dvl.world_model import N_MODES, SBKHMMArtifact, sample_sign_truncated_predictive

ROLLOUT_HORIZON = 8


@dataclass(frozen=True)
class HumanMultiStepTrajectory:
    """One world sample's 8-step trajectory for one human track.

    ``modes[t]`` is the latent mode ACTIVE while generating step t's
    motion (mode 0 = the one drawn at decision time; modes 1..7 are each
    evolved from the previous one via ``artifact.transition_matrix``).
    ``step_velocities[t]`` is the CONSTANT velocity that reproduces
    ``_integrate_holonomic``'s trapezoidal-average position update for
    step t exactly when fed through ``transition.propagate_human`` --
    i.e. ``(pos_after_t - pos_before_t) / dt`` -- so this trajectory
    plugs directly into the same ``transition.step()`` machinery the
    one-step decision/reward path already uses, instead of a second,
    only-approximately-equivalent geometry.
    """

    track_id: int
    modes: Tuple[Optional[int], ...]
    step_velocities: Tuple[Tuple[float, float], ...]


def sample_human_multi_step_worlds(
    artifact: SBKHMMArtifact,
    track_beliefs: Dict[int, np.ndarray],
    track_positions: Dict[int, np.ndarray],
    track_speeds: Dict[int, float],
    track_headings: Dict[int, float],
    n_samples: int,
    dt: float,
    max_human_speed: float,
    source: str,
    seed: Tuple[int, int, int],
    horizon: int = ROLLOUT_HORIZON,
    shuffled_pool_by_track: Optional[Dict[int, Sequence[np.ndarray]]] = None,
) -> Dict[int, Tuple[HumanMultiStepTrajectory, ...]]:
    """guide.md R4-3 points 3-4: ONE shared draw per (track, world
    sample), reused identically by every candidate action's rollout.

    ``moment_mean`` has no discrete mode to evolve (guide.md's rollout.py
    docstring: it is a single averaged Gaussian proxy, not a mode
    trajectory) -- it repeats the SAME averaged (a_parallel, omega) every
    step rather than evolving via the transition matrix, since there is
    no mode index to transition between.

    R4-3R fix (2026-08-10, guide.md "R4-3R -- 行人顺序会改变采样结果"):
    each track gets its OWN independent RNG seeded from
    ``_stable_seed(*seed, track_id)``, not one RNG shared (and consumed
    in dict-iteration order) across every track. A single shared RNG
    made the sampled trajectory for EVERY track depend on which order
    ``track_beliefs.items()`` happened to iterate in -- verified by
    directly reproducing it: swapping two tracks' insertion order changed
    BOTH tracks' sampled futures under an identical seed. Per-track
    seeding makes each track's result depend only on its own track_id,
    never on how many OTHER tracks exist or what order they're stored in.
    """
    if source not in POSTERIOR_SOURCES:
        raise RolloutError(f"unknown posterior source {source!r}, must be one of {POSTERIOR_SOURCES}")
    if horizon < 1:
        raise RolloutError(f"horizon must be >= 1, got {horizon}")
    if n_samples <= 0:
        raise RolloutError(f"n_samples must be positive, got {n_samples}")
    if dt != artifact.dt:
        raise RolloutError(f"dt={dt} does not match artifact.dt={artifact.dt} -- must use the artifact's own dt")

    results: Dict[int, Tuple[HumanMultiStepTrajectory, ...]] = {}

    for track_id, belief in track_beliefs.items():
        rng = np.random.default_rng(_stable_seed(seed[0], seed[1], seed[2], track_id))
        position = np.asarray(track_positions[track_id], dtype=np.float64)
        speed0 = track_speeds[track_id]
        heading0 = track_headings[track_id]
        trajectories = []

        for _ in range(n_samples):
            if source == "moment_mean":
                sampled_mode: Optional[int] = None
            else:
                shuffled_pool = (shuffled_pool_by_track or {}).get(track_id)
                mode_probs = _mode_probabilities(belief, source, rng, shuffled_pool)
                mode_probs = mode_probs / mode_probs.sum()
                sampled_mode = int(rng.choice(N_MODES, p=mode_probs))

            cur_pos = position.copy()
            cur_speed, cur_heading, cur_mode = speed0, heading0, sampled_mode
            modes = []
            step_velocities = []
            for t in range(horizon):
                if source == "moment_mean":
                    a_parallel = float(belief @ artifact.emission_mean[:, 0])
                    omega = float(belief @ artifact.emission_mean[:, 1])
                else:
                    residual = sample_sign_truncated_predictive(rng, artifact.niw(cur_mode), cur_mode)
                    a_parallel, omega = float(residual[0]), float(residual[1])

                new_speed, new_heading, new_pos = _integrate_holonomic(
                    cur_speed, cur_heading, a_parallel, omega, cur_pos, dt, max_human_speed,
                )
                step_vx = float((new_pos[0] - cur_pos[0]) / dt)
                step_vy = float((new_pos[1] - cur_pos[1]) / dt)
                modes.append(cur_mode)
                step_velocities.append((step_vx, step_vy))

                cur_pos, cur_speed, cur_heading = new_pos, new_speed, new_heading
                if t < horizon - 1 and source != "moment_mean":
                    cur_mode = int(rng.choice(N_MODES, p=artifact.transition_matrix[cur_mode]))

            trajectories.append(HumanMultiStepTrajectory(
                track_id=track_id, modes=tuple(modes), step_velocities=tuple(step_velocities),
            ))
        results[track_id] = tuple(trajectories)

    return results


@dataclass(frozen=True)
class CounterfactualCandidateResult:
    action_index: int
    collision_prob: float
    lower_tail_clearance: float
    expected_progress: float
    control_cost: float


def _lower_tail_mean(values: Sequence[float], alpha: float) -> float:
    """Mean of the worst ``alpha`` fraction of ``values`` (ascending =
    worst-first for a clearance-like quantity) -- guide.md R4-3 point 5:
    reuses the SAME ``cvar_alpha`` convention already used for the
    decision-time lower-tail IQN CVaR, rather than introducing a second
    threshold."""
    if not values:
        raise ValueError("_lower_tail_mean requires at least one value")
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * len(sorted_values))))
    return float(np.mean(sorted_values[:k]))


def evaluate_counterfactual_candidates(
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    human_worlds: Dict[int, Tuple[HumanMultiStepTrajectory, ...]],
    n_samples: int,
    action_table: Sequence[Tuple[float, float]],
    reward_config: RewardConfig,
    dt: float,
    time_limit: float,
    global_time: float,
    cvar_alpha: float,
    horizon: int = ROLLOUT_HORIZON,
) -> Tuple[CounterfactualCandidateResult, ...]:
    """guide.md R4-3 points 2/5/6: for each candidate action, hold it
    constant for ``horizon`` steps against the SAME (already-sampled)
    human worlds, aggregate collision/clearance/progress across worlds
    with terminal absorption, and pair with the action's own control
    cost (reuses ``policy.compute_action_features_array``'s turn_cost,
    a pure function of the action and the robot's CURRENT velocity --
    no world-sampling needed).

    R4-3R fix (2026-08-10, guide.md "R4-3R -- 成功步进度计算错误"): the
    step ON WHICH termination occurs is now counted toward progress --
    freezing starts from the NEXT step, not this one. The previous
    version dropped the terminating step's progress unconditionally,
    which made a fast action that reaches the goal in ONE step score
    ``expected_progress=0`` while a slow action that merely creeps for
    all 8 steps without ever finishing scored strictly higher --
    reproduced directly: a 1-step-to-goal action showed 0.0 progress
    against a never-arrives creeping action's 0.2, an inverted
    preference. Collision and success are both real, already-executed
    geometry up to and including the terminating step; only what comes
    AFTER should be frozen.
    """
    if n_samples <= 0:
        raise RolloutError(f"n_samples must be positive, got {n_samples}")
    if not (0.0 < cvar_alpha <= 1.0):
        raise RolloutError(f"cvar_alpha must be in (0, 1], got {cvar_alpha}")
    track_ids = [h.track_id for h in humans]
    missing = [tid for tid in track_ids if tid not in human_worlds]
    if missing:
        raise RolloutError(f"human_worlds is missing track_id(s) {missing} present in humans")
    for tid in track_ids:
        if len(human_worlds[tid]) != n_samples:
            raise RolloutError(f"track {tid} has {len(human_worlds[tid])} world samples, expected n_samples={n_samples}")
        for traj in human_worlds[tid]:
            if len(traj.step_velocities) != horizon:
                raise RolloutError(f"track {tid} trajectory has {len(traj.step_velocities)} steps, expected horizon={horizon}")

    action_array = np.asarray(action_table, dtype=np.float64)
    control_costs = compute_action_features_array(robot, action_array)[:, 4]  # turn_cost column
    results = []

    for action_idx, (avx, avy) in enumerate(action_table):
        n_collisions = 0
        world_min_clearances = []
        world_progresses = []

        for w in range(n_samples):
            cur_robot = robot
            cur_humans = humans
            cur_global_time = global_time
            terminal = False
            min_clearance = float("inf")
            progress = 0.0

            for t in range(horizon):
                if terminal:
                    break
                human_actions = [human_worlds[tid][w].step_velocities[t] for tid in track_ids]
                step_result = transition_step(
                    cur_robot, cur_humans, avx, avy, human_actions,
                    dt, time_limit, cur_global_time, reward_config,
                )
                min_clearance = min(min_clearance, step_result.dmin)
                prev_dist = float(np.hypot(cur_robot.px - cur_robot.gx, cur_robot.py - cur_robot.gy))
                next_dist = float(np.hypot(step_result.next_robot.px - step_result.next_robot.gx, step_result.next_robot.py - step_result.next_robot.gy))
                progress += prev_dist - next_dist
                if step_result.terminated or step_result.truncated:
                    terminal = True
                    if step_result.terminated and step_result.event == EVENT_COLLISION:
                        n_collisions += 1
                    # guide.md R4-3R: absorption freezes STARTING NEXT
                    # step -- this step's progress (computed just above)
                    # is real, already-executed geometry and is kept.
                cur_robot = step_result.next_robot
                cur_humans = step_result.next_humans
                cur_global_time += dt

            world_min_clearances.append(min_clearance if np.isfinite(min_clearance) else 0.0)
            world_progresses.append(progress)

        results.append(CounterfactualCandidateResult(
            action_index=action_idx,
            collision_prob=n_collisions / n_samples,
            lower_tail_clearance=_lower_tail_mean(world_min_clearances, cvar_alpha),
            expected_progress=float(np.mean(world_progresses)),
            control_cost=float(control_costs[action_idx]),
        ))

    return tuple(results)


def rank_counterfactual_candidates(
    results: Sequence[CounterfactualCandidateResult], clearance_tolerance: float,
) -> Tuple[int, ...]:
    """Returns action indices sorted best-first.

    guide.md R4-4R3 fix (2026-08-10, "根源修复：字典序tie-break缺少
    clearance容忍度"): the original rule was a STRICT lexicographic
    order -- (collision_prob asc, clearance desc, progress desc,
    control_cost asc) -- with zero tolerance on the clearance
    comparison. Real R4-4R2 audit data (episode_seed=980030,
    decision_seed=2) showed this makes the rule pick a candidate with
    0.04m more clearance (1.970m vs 1.929m -- both already ~2m from any
    human, nowhere near the 0.20m discomfort distance, i.e. noise-scale
    under genuine belief sampling variance) over a candidate with 2.4x
    the progress (1.256 vs 0.522) and lower control cost. Belief
    tracking a real multi-modal distribution makes ``lower_tail_clearance``
    estimates noisier than a single-mode/CV assumption's, so this bug
    penalizes genuinely-informative belief MORE, not less -- a real
    mechanism, not a hypothetical one.

    Fix: within each EXACT collision_prob tier (safety itself keeps zero
    tolerance -- guide.md's non-inferiority requirement is not
    negotiable), candidates within ``clearance_tolerance`` of that
    tier's own best clearance are treated as safety-tied and ranked by
    progress (then control cost) instead of by the raw clearance gap;
    candidates further below the tier's best clearance keep the old
    strict clearance-first ordering (they are NOT safety-tied with the
    best, so clearance should still dominate for them).
    ``clearance_tolerance`` is REQUIRED (no silent default) so every
    caller consciously supplies a value -- guide.md requires it come
    from ``oracle_regret.clearance_equivalence_tolerance(...)``, the
    SAME data-derived (grid spacing * dt * horizon) tolerance already
    used by the R4-4R2 safety-optimal layer, never tuned ad hoc.
    """
    if clearance_tolerance < 0:
        raise ValueError(f"clearance_tolerance must be non-negative, got {clearance_tolerance}")
    if not results:
        raise ValueError("rank_counterfactual_candidates requires at least one result")

    by_collision: Dict[float, list] = {}
    for r in results:
        by_collision.setdefault(r.collision_prob, []).append(r)
    tier_best_clearance = {
        prob: max(r.lower_tail_clearance for r in group) for prob, group in by_collision.items()
    }

    def _key(r: CounterfactualCandidateResult) -> Tuple[float, int, float, float]:
        near_tier_best = r.lower_tail_clearance >= tier_best_clearance[r.collision_prob] - clearance_tolerance
        if near_tier_best:
            return (r.collision_prob, 0, -r.expected_progress, r.control_cost)
        return (r.collision_prob, 1, -r.lower_tail_clearance, -r.expected_progress)

    return tuple(r.action_index for r in sorted(results, key=_key))


COUNTERFACTUAL_CONTRACT_V1 = "bdvl_counterfactual_8step_constant_velocity_v1"


@dataclass(frozen=True)
class CounterfactualRecord:
    """guide.md R4-3 point 7: everything needed to audit/reproduce one
    decision's counterfactual ranking, and NOTHING that could only be
    known after the fact. ``decision_seed`` is the real
    ``BDVLPolicy.next_seed_key()``-derived per-decision counter (same
    mechanism as guide.md R4-2R-2), not a re-derived approximation.

    R4-3R fix (2026-08-10, guide.md "R4-3R -- CounterfactualRecord增加
    contract version、action-grid hash、posterior source、horizon、
    sample数和alpha"): these extra provenance fields let a FUTURE reader
    of a stored record verify exactly which contract/action-grid/budget
    produced it, matching the same fail-closed philosophy checkpoints
    and registries already use elsewhere in this codebase -- a record's
    numbers are meaningless without knowing which grid the action
    indices refer to and which sampling budget/alpha they were computed
    under.
    """

    contract_version: str
    suite_seed: int
    episode_seed: int
    decision_seed: int
    artifact_sha256: str
    action_grid_hash: str
    posterior_source: str
    horizon: int
    n_world_samples: int
    cvar_alpha: float
    candidate_results: Tuple[CounterfactualCandidateResult, ...]
    ranked_action_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if set(self.ranked_action_indices) != {r.action_index for r in self.candidate_results}:
            raise ValueError("ranked_action_indices must be a permutation of candidate_results' action indices")


def build_counterfactual_record(
    suite_seed: int,
    episode_seed: int,
    decision_seed: int,
    artifact: SBKHMMArtifact,
    action_grid_hash: str,
    robot: RobotObservation,
    humans: Sequence[HumanObservation],
    track_beliefs: Dict[int, np.ndarray],
    track_positions: Dict[int, np.ndarray],
    track_speeds: Dict[int, float],
    track_headings: Dict[int, float],
    action_table: Sequence[Tuple[float, float]],
    reward_config: RewardConfig,
    dt: float,
    time_limit: float,
    global_time: float,
    max_human_speed: float,
    n_world_samples: int,
    cvar_alpha: float,
    clearance_tolerance: float,
    posterior_source: str = "full",
) -> CounterfactualRecord:
    """The one entry point R4-4's data collection is expected to call.
    Builds the shared human worlds ONCE (guide.md R4-3 point 4), scores
    every candidate against them, and packages the frozen-schema record.

    ``action_grid_hash`` is REQUIRED and must come from the caller's own
    already-loaded registry (e.g. ``registry["action_grid_hash"]``),
    never recomputed here -- this module only receives a raw
    ``action_table``, not the ``ActionGridSpec`` that produced it, so
    independently re-deriving a hash here would risk silently drifting
    from whatever the caller actually validated.

    ``clearance_tolerance`` (guide.md R4-4R3) is REQUIRED and must come
    from ``oracle_regret.clearance_equivalence_tolerance(action_table,
    dt, horizon)`` -- see ``rank_counterfactual_candidates``'s docstring
    for why a tolerant tie-break is necessary here.
    """
    seed_key = (suite_seed, episode_seed, decision_seed)
    human_worlds = sample_human_multi_step_worlds(
        artifact=artifact, track_beliefs=track_beliefs, track_positions=track_positions,
        track_speeds=track_speeds, track_headings=track_headings, n_samples=n_world_samples,
        dt=dt, max_human_speed=max_human_speed, source=posterior_source, seed=seed_key,
    )
    candidate_results = evaluate_counterfactual_candidates(
        robot=robot, humans=humans, human_worlds=human_worlds, n_samples=n_world_samples,
        action_table=action_table, reward_config=reward_config, dt=dt, time_limit=time_limit,
        global_time=global_time, cvar_alpha=cvar_alpha,
    )
    ranked = rank_counterfactual_candidates(candidate_results, clearance_tolerance)
    return CounterfactualRecord(
        contract_version=COUNTERFACTUAL_CONTRACT_V1,
        suite_seed=suite_seed, episode_seed=episode_seed, decision_seed=decision_seed,
        artifact_sha256=artifact.content_sha256(), action_grid_hash=action_grid_hash,
        posterior_source=posterior_source, horizon=ROLLOUT_HORIZON, n_world_samples=n_world_samples,
        cvar_alpha=cvar_alpha, candidate_results=candidate_results, ranked_action_indices=ranked,
    )
