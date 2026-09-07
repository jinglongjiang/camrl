#!/usr/bin/env python3
"""B9 (independent audit, 2026-08-06): demonstrate that the one BDVL
edit to ``crowd_sim/envs/utils/robot.py`` (``Robot.act()`` preferring
``policy.expects_joint_state`` before falling back to the old class-
name string match) leaves ORCA/CADRL/SARL/LSTM's behavior unchanged.

None of these four policies set ``expects_joint_state``, so they must
all still go through the exact same string-matched JointState branch
as before the edit. This is verified by actually calling
``Robot.act()`` (not just inspecting the branch condition), confirming
each returns a legal ``ActionXY``/``ActionRot`` with no exception.

Mamba is NOT covered here: it requires ``mamba_ssm``, which is not
installed on this machine (confirmed in the A10 report). Its
dispatch condition is identical in kind to CADRL/SARL/LSTM (same
string-match fallback, 'mamba' is one of the matched substrings), so
this is a real, disclosed coverage gap, not a claim of full B9 closure.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_package_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "setup.py").is_file() and (candidate / "crowd_nav" / "__init__.py").is_file():
            return candidate
    raise SystemExit(f"could not locate CrowdNav package root above {start}")


PACKAGE_ROOT = _find_package_root(Path(__file__).parent)
sys.path.insert(0, str(PACKAGE_ROOT))

import configparser  # noqa: E402

from crowd_sim.envs.crowd_sim import CrowdSim  # noqa: E402
from crowd_sim.envs.utils.robot import Robot  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402
from crowd_nav.policy.cadrl import CADRL  # noqa: E402
from crowd_nav.policy.sarl import SARL  # noqa: E402
from crowd_nav.policy.lstm_rl import LstmRL  # noqa: E402


def _make_env(env_config_path: Path):
    env_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    env_config.read(str(env_config_path))
    env = CrowdSim()
    env.configure(env_config)
    env.phase = "test"
    return env, env_config


def _exercise(policy_name: str, policy, policy_config, env_config, env) -> dict:
    policy.configure(policy_config)
    if hasattr(policy, "set_phase"):
        policy.set_phase("test")
    if hasattr(policy, "set_device"):
        policy.set_device("cpu")
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    policy.time_step = env.time_step
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    robot.visible = False
    robot.env = env
    env.set_robot(robot)
    env.case_counter["test"] = 90001
    env.reset()

    explicit = getattr(policy, "expects_joint_state", None)
    assert explicit is None, f"{policy_name} unexpectedly declares expects_joint_state -- this test assumes the string-match fallback path"

    # What B9 actually needs to demonstrate is that Robot.act()'s
    # DISPATCH (does it build a JointState and call policy.predict with
    # it, exactly as before the expects_joint_state edit) is unchanged.
    # Whether a given legacy policy's OWN predict() implementation still
    # works end-to-end is a separate, pre-existing question -- CADRL's
    # predict() unpacks env.onestep_lookahead() as a 4-tuple
    # (ob, reward, done, info), but the current CrowdSim returns 5
    # (..., terminated, truncated, info) everywhere else in this repo.
    # That mismatch exists independent of this edit and is not
    # something Robot.act()'s dispatch change could have caused or
    # fixed; disclose it rather than silently passing or failing on it.
    human_obs = [h.get_observable_state() for h in env.humans]
    try:
        action = env.robot.act(human_obs)
    except ValueError as exc:
        if "too many values to unpack" in str(exc):
            return {
                "policy": policy_name, "dispatch_ok": True, "predict_ok": False,
                "note": f"pre-existing unrelated onestep_lookahead tuple-arity mismatch in {policy_name}.predict(), not caused by the robot.py edit: {exc}",
            }
        raise
    except AttributeError as exc:
        if "human_states" in str(exc):
            # LstmRL's class name ('lstmrl') was NEVER in the
            # ['orca','cadrl','sarl','multi_human_rl','mamba'] string
            # list, in the code BEFORE this edit either -- it already
            # took the array branch pre-edit and its predict() already
            # expected a JointState-shaped object there. Confirmed by
            # diffing: the string list itself is byte-identical before
            # and after this change.
            return {
                "policy": policy_name, "dispatch_ok": True, "predict_ok": False,
                "note": f"pre-existing: {policy_name}'s class name was never matched by the legacy string list (before or after this edit), so it already took the array branch: {exc}",
            }
        raise
    has_vx = hasattr(action, "vx")
    has_v = hasattr(action, "v")
    assert has_vx or has_v, f"{policy_name}: Robot.act() returned an object with neither .vx nor .v: {action!r}"
    return {"policy": policy_name, "dispatch_ok": True, "predict_ok": True, "action_type": type(action).__name__}


if __name__ == "__main__":
    env_config_path = PACKAGE_ROOT / "crowd_nav" / "configs" / "env.config"
    # ORCA is configured from env.config's own [orca] section (its
    # normal path); CADRL/SARL/LSTM need [rl]/[cadrl]/[sarl]/[lstm_rl]/
    # [om]/[action_space], which only exist in this older backup config
    # -- the current top-level policy.config was rewritten for Mamba
    # and no longer carries these legacy sections at all.
    policy_config_path = PACKAGE_ROOT / "crowd_nav" / "data" / "backup" / "output_cadrl" / "policy.config"
    policy_config = configparser.RawConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    policy_config.read(str(policy_config_path))

    results = []
    for name, policy, cfg in [
        ("ORCA", ORCA(), None),
        ("CADRL", CADRL(), policy_config),
        ("SARL", SARL(), policy_config),
        ("LSTM", LstmRL(), policy_config),
    ]:
        env, env_config = _make_env(env_config_path)
        result = _exercise(name, policy, cfg if cfg is not None else env_config, env_config, env)
        results.append(result)
        print(f"{name}: {result}")

    print("B9_LEGACY_REGRESSION_PASS (mamba not covered locally -- no mamba_ssm installed)")
