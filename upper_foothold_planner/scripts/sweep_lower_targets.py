"""Measure the frozen lower controller under explicit upper-level targets."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.config import AttrDict
from upper_planner.contracts import FootholdActionBounds
from upper_planner.env import FootholdEnv, make_sim_params
from upper_planner.lower_policy import FrozenLowerPolicy
from upper_planner.target_interface import UpperFootholdTargetInterface


def csv_floats(text):
    return [float(value) for value in text.split(",")]


def arguments():
    custom = [
        {"name": "--steps", "type": int, "default": 500},
        {"name": "--seed", "type": int, "default": 43},
        {"name": "--forward_m", "type": str, "default": "0.08,0.12,0.16"},
        {"name": "--lateral_m", "type": str, "default": "0.18,0.21,0.24"},
        {"name": "--yaw_deg", "type": str, "default": "-5,0,5"},
        {"name": "--enable_randomization", "action": "store_true"},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "gate2_target_grid")},
    ]
    args = gymutil.parse_arguments(
        description="explicit foothold target capability grid", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def disable_randomization(cfg):
    cfg.noise.add_noise = False
    for name in (
            "randomize_friction", "randomize_base_mass", "randomize_link_mass",
            "randomize_base_com", "randomize_Kp", "randomize_Kd", "randomize_gravity",
            "randomize_joint_damping", "randomize_joint_friction",
            "randomize_joint_armature", "kick_robots"):
        cfg.domain_rand[name] = False


def normalize(value, limits):
    return 2.0 * (value - limits[0]) / (limits[1] - limits[0]) - 1.0


def roll_pitch(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    action_cfg = json.loads((ROOT / "config" / "default.json").read_text())["action"]
    bounds = FootholdActionBounds.from_config(action_cfg)
    physical = [(f, lateral, yaw)
                for f in csv_floats(args.forward_m)
                for lateral in csv_floats(args.lateral_m)
                for yaw in csv_floats(args.yaw_deg)]

    checkpoint_path = ROOT / "checkpoints" / "lower_model_7000.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = AttrDict.from_nested(checkpoint["config"])
    cfg.asset.file = str(ROOT / "assets" / "SF_TRON1A" / "urdf" / "robot.urdf")
    cfg.env.num_envs = len(physical)
    if not args.enable_randomization:
        disable_randomization(cfg)

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    policy = FrozenLowerPolicy(checkpoint_path, env.device)
    target_interface = UpperFootholdTargetInterface(bounds)
    actions = torch.tensor([
        [normalize(f, bounds.forward_m), normalize(lat, bounds.lateral_abs_m),
         normalize(yaw, bounds.yaw_deg)]
        for f, lat, yaw in physical], dtype=torch.float32, device=env.device)

    obs, goal, _ = env.get_observations()
    initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    previous_swing = env.sampler.swing_foot.clone()
    falls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    base_height_min = torch.full((env.num_envs,), float("inf"), device=env.device)
    base_height_max = torch.full((env.num_envs,), -float("inf"), device=env.device)
    abs_roll_max = torch.zeros(env.num_envs, device=env.device)
    abs_pitch_max = torch.zeros(env.num_envs, device=env.device)
    body_contact_max = torch.zeros(env.num_envs, device=env.device)
    touchdown_errors = [[] for _ in range(env.num_envs)]
    decisions = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    for _ in range(args.steps):
        raw_action, _ = policy.infer(obs, goal)
        next_obs, _, done, extras, next_goal, _ = env.step(raw_action)

        base_height_min = torch.minimum(base_height_min, env.base_position[:, 2])
        base_height_max = torch.maximum(base_height_max, env.base_position[:, 2])
        roll, pitch = roll_pitch(env.base_quat)
        abs_roll_max = torch.maximum(abs_roll_max, roll.abs())
        abs_pitch_max = torch.maximum(abs_pitch_max, pitch.abs())
        if env.body_contact_indices.numel():
            contacts = torch.norm(env.contact_forces[:, env.body_contact_indices], dim=-1).max(dim=-1).values
            body_contact_max = torch.maximum(body_contact_max, contacts)

        switched = (env.sampler.swing_foot != previous_swing) & initialized & ~done.bool()
        ids = switched.nonzero(as_tuple=False).flatten()
        if ids.numel():
            landed = 1 - env.sampler.swing_foot[ids]
            row = torch.arange(ids.numel(), device=env.device)
            error = torch.norm(
                env.foot_positions[ids, landed, :2]
                - env.sampler.target_pos[ids, landed, :2], dim=-1)
            for env_id, value in zip(ids.tolist(), error.tolist()):
                touchdown_errors[env_id].append(value)

        time_outs = extras.get("time_outs", torch.zeros_like(done, dtype=torch.bool))
        falls += (done.bool() & ~time_outs.bool()).long()
        initialized[done.bool()] = False
        ready = ~initialized & ~env.goal_reset_pending & (env.episode_length_buf > 0)
        update = switched | ready
        update_ids = update.nonzero(as_tuple=False).flatten()
        if update_ids.numel():
            target_interface.apply(env, actions[update_ids], update_ids)
            decisions[update_ids] += 1
            initialized[update_ids] = True
        previous_swing.copy_(env.sampler.swing_foot)
        obs, goal = next_obs, next_goal

    results = []
    for index, (forward, lateral, yaw) in enumerate(physical):
        errors = np.asarray(touchdown_errors[index], dtype=np.float32)
        results.append({
            "env": index, "forward_m": forward, "lateral_abs_m": lateral,
            "yaw_deg": yaw,
            "radial_step_m": math.hypot(forward, lateral),
            "falls": int(falls[index]), "decisions": int(decisions[index]),
            "touchdowns": int(errors.size),
            "touchdown_xy_error_mean_m": float(errors.mean()) if errors.size else None,
            "touchdown_xy_error_p95_m": float(np.percentile(errors, 95)) if errors.size else None,
            "touchdown_xy_error_max_m": float(errors.max()) if errors.size else None,
            "base_height_min_m": float(base_height_min[index]),
            "base_height_max_m": float(base_height_max[index]),
            "base_abs_roll_max_deg": float(torch.rad2deg(abs_roll_max[index])),
            "base_abs_pitch_max_deg": float(torch.rad2deg(abs_pitch_max[index])),
            "nonfoot_contact_force_max_n": float(body_contact_max[index]),
        })

    valid = [row for row in results if row["falls"] == 0 and row["touchdowns"] >= 8]
    summary = {
        "seed": args.seed, "steps": args.steps,
        "simulated_seconds_per_env": args.steps * env.dt,
        "randomization_enabled": bool(args.enable_randomization),
        "num_grid_points": len(results),
        "zero_fall_points": sum(row["falls"] == 0 for row in results),
        "points_p95_below_5cm": sum(
            row["touchdown_xy_error_p95_m"] is not None
            and row["touchdown_xy_error_p95_m"] < 0.05 for row in results),
        "valid_error_mean_m": float(np.mean([
            row["touchdown_xy_error_mean_m"] for row in valid])) if valid else None,
        "training_distribution_check": {
            "radial_step_range_m": [min(math.hypot(f, lat) for f, lat, _ in physical),
                                    max(math.hypot(f, lat) for f, lat, _ in physical)],
            "lower_training_radial_range_m": list(cfg.foothold.step_distance),
            "lower_training_min_lateral_separation_m": cfg.foothold.minimum_lateral_separation,
            "lower_training_target_yaw_deg": list(cfg.foothold.target_yaw_deg),
        },
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
