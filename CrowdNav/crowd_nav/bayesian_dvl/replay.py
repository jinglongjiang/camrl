"""Typed demo/online replay buffers (guide.md 6.2/8.1, A8).

Two buffers, never more (guide.md 6.1: "只有两类buffer，不引入DAgger五路
结构"):

    demo    reservoir sample of ORCA-demonstration transitions (T0)
    online  ring buffer of transitions collected during online RL (T1)

Sampling draws a fixed ratio (default 20% demo / 80% online, guide.md
11's ``demo_sample_ratio``); if one buffer is temporarily empty or too
small, its share is reallocated to the other rather than erroring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.bayesian_dvl.contracts import HumanObservation, RobotObservation


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class Transition:
    """One decision step. ``human_features``/``human_mask`` are the
    padded [MAX_HUMANS, D] / [MAX_HUMANS] arrays already suitable for
    SetEncoder; belief/next_belief are per-track dicts flattened into a
    fixed-size array by the caller (this module stores whatever array
    shape the caller provides, it does not interpret it)."""

    robot_features: np.ndarray
    human_features: np.ndarray
    human_mask: np.ndarray
    belief: np.ndarray
    action_index: int
    reward: float
    done: bool
    next_robot_features: np.ndarray
    next_human_features: np.ndarray
    next_human_mask: np.ndarray
    next_belief: np.ndarray
    artifact_sha256: str
    gamma_pow_n: float = 1.0
    episode_seed: int = 0
    step_index: int = 0


@dataclass(frozen=True)
class MCReturnSample:
    """One (state, action, full-trajectory return) sample for R2-3's
    Monte Carlo training path (guide.md R2-3, 2026-08-07).

    Deliberately NOT a ``Transition``: there is no ``next_*``/``done``/
    ``gamma_pow_n`` bootstrap machinery here at all -- ``target_return``
    IS the complete discounted return from this state to the episode's
    end, already computed by ``trainer.build_mc_return_samples``. Naming
    this distinctly (rather than overloading ``Transition.reward`` with
    a full return and faking ``done=True``) keeps the two training paths
    from being silently interchangeable by a future caller.

    R4-2 fix (2026-08-10, guide.md "R4-1R-4 -- 冻结 R4-2 的精确数学目标"):
    v1 stored only PRE-FLATTENED ``robot_features``/``human_features``/
    ``human_mask`` (built once at collection time from the CURRENT,
    pre-action state) and regressed those directly onto ``target_return``
    -- correct for a state-only V(s,b), but for Q(s,b,a) the MC-loss
    target must be built from the SAME candidate builder deployment uses
    (``policy._vectorized_candidate_batch``, restricted to the one
    ``executed_action_index``), which needs the RAW ``robot``/``humans``/
    ``belief_tracker_snapshot`` this dataclass discarded, not a lossy
    pre-encoded array. ``executed_action_features`` is filled in at
    collection time (real ``RobotObservation`` still available then) via
    ``policy.compute_action_features_array``, so ``build_mc_return_samples``
    never has to reconstruct it from anything else. ``posterior_seed_key``
    lets the MC-loss target reuse bit-identical tau/world-sampling to a
    live decision at this state (see ``policy._vectorized_candidate_batch``'s
    ``action_indices`` parameter). ``source_role`` mirrors
    ``ranking.RankingDemoSample``'s field of the same name.
    """

    robot: RobotObservation
    humans: Tuple[HumanObservation, ...]
    global_time: float
    belief_tracker_snapshot: Dict[int, object]
    executed_action_index: int
    executed_action_features: np.ndarray
    posterior_seed_key: Tuple[int, int, int]
    target_return: float
    artifact_sha256: str
    episode_seed: int
    step_index: int
    outcome: str = ""  # "success" | "collision" | "timeout", diagnostic-only (R2-5)
    source_role: str = "online"


class ReservoirBuffer:
    """Fixed-capacity reservoir sample -- uniform over everything ever
    added, not just a sliding window (appropriate for the one-time T0
    demonstration set, guide.md's "5,000个成功ORCA demonstration
    episode，数据集共享给所有训练消融")."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ReplayError("capacity must be positive")
        self.capacity = capacity
        self._items: List[Transition] = []
        self._n_seen = 0
        self._rng = random.Random(0)

    def add(self, transition: Transition) -> None:
        self._n_seen += 1
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            idx = self._rng.randint(0, self._n_seen - 1)
            if idx < self.capacity:
                self._items[idx] = transition

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, n: int, rng: random.Random) -> List[Transition]:
        if not self._items:
            return []
        return [rng.choice(self._items) for _ in range(n)]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "items": list(self._items),
            "n_seen": self._n_seen,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ReplayError("reservoir capacity mismatch during resume")
        self._items = list(state["items"])
        self._n_seen = int(state["n_seen"])
        self._rng.setstate(state["rng_state"])


class RingBuffer:
    """Fixed-capacity FIFO ring buffer for online transitions."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ReplayError("capacity must be positive")
        self.capacity = capacity
        self._items: List[Optional[Transition]] = [None] * capacity
        self._write_idx = 0
        self._size = 0

    def add(self, transition: Transition) -> None:
        self._items[self._write_idx] = transition
        self._write_idx = (self._write_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, n: int, rng: random.Random) -> List[Transition]:
        if self._size == 0:
            return []
        return [self._items[rng.randrange(self._size)] for _ in range(n)]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "items": list(self._items),
            "write_idx": self._write_idx,
            "size": self._size,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ReplayError("ring capacity mismatch during resume")
        items = list(state["items"])
        if len(items) != self.capacity:
            raise ReplayError("ring item storage length mismatch during resume")
        self._items = items
        self._write_idx = int(state["write_idx"])
        self._size = int(state["size"])


# R4-2R-1 fix (2026-08-10, guide.md "R4-2R-1 -- 训练契约与checkpoint必须
# 升版"): R4-2 changed BOTH RankingDemoSample's and MCReturnSample's
# field sets (executed_action_index/executed_action_features/
# posterior_seed_key/source_role added, MCReturnSample's flattened
# feature arrays replaced with raw robot/humans/belief snapshot) -- a
# pickled R4-1-era replay buffer's items are instances of the OLD
# dataclass shape and must never be silently resumed into code expecting
# the new one (a dataclass field mismatch on unpickle is not guaranteed
# to raise cleanly, and even if it does, the error would point nowhere
# near the real cause).
REPLAY_SCHEMA_V2_R4_2 = "bdvl_replay_executed_action_v2"


class DemoOnlineReplay:
    """Owns one ``ReservoirBuffer`` (demo) and one ``RingBuffer``
    (online); ``sample_batch`` mixes them at ``demo_ratio`` with
    graceful reallocation when either is empty (A8 acceptance:
    "某buffer暂空时正确重分配")."""

    def __init__(self, demo_capacity: int, online_capacity: int, demo_ratio: float = 0.20):
        if not (0.0 <= demo_ratio <= 1.0):
            raise ReplayError(f"demo_ratio must be in [0,1], got {demo_ratio}")
        self.demo = ReservoirBuffer(demo_capacity)
        self.online = RingBuffer(online_capacity)
        self.demo_ratio = demo_ratio

    def state_dict(self) -> dict:
        return {
            "replay_schema_version": REPLAY_SCHEMA_V2_R4_2,
            "demo_ratio": self.demo_ratio,
            "demo": self.demo.state_dict(),
            "online": self.online.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("replay_schema_version") != REPLAY_SCHEMA_V2_R4_2:
            raise ReplayError(
                f"replay state schema is {state.get('replay_schema_version')!r}, current code requires "
                f"{REPLAY_SCHEMA_V2_R4_2!r} (guide.md R4-2R-1) -- this replay state predates the R4-2 "
                "executed-action data model and cannot be resumed"
            )
        if abs(float(state["demo_ratio"]) - self.demo_ratio) > 1e-12:
            raise ReplayError("demo ratio mismatch during resume")
        self.demo.load_state_dict(state["demo"])
        self.online.load_state_dict(state["online"])

    def sample_batch(self, batch_size: int, rng: Optional[random.Random] = None) -> Tuple[List[Transition], int, int]:
        """Returns (transitions, n_demo_drawn, n_online_drawn)."""
        rng = rng or random.Random()
        n_demo_target = round(batch_size * self.demo_ratio)
        n_online_target = batch_size - n_demo_target

        if len(self.demo) == 0:
            n_online_target = batch_size
            n_demo_target = 0
        if len(self.online) == 0:
            n_demo_target = batch_size
            n_online_target = 0

        demo_samples = self.demo.sample(n_demo_target, rng)
        online_samples = self.online.sample(n_online_target, rng)

        # If a buffer had items but not enough to satisfy its own quota
        # gracefully (sample() with replacement always returns exactly
        # what's asked when nonempty), the only true shortfall is total
        # emptiness, already handled above. Re-top-up here defensively
        # in case a future caller adds a "sample without replacement" mode.
        deficit = batch_size - (len(demo_samples) + len(online_samples))
        if deficit > 0 and len(self.online) > 0:
            online_samples += self.online.sample(deficit, rng)
        elif deficit > 0 and len(self.demo) > 0:
            demo_samples += self.demo.sample(deficit, rng)

        return demo_samples + online_samples, len(demo_samples), len(online_samples)
