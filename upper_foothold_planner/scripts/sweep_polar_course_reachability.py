"""Audit the frozen lower controller under the current polar-course interface.

This is deliberately a model-free gate.  Every vectorized environment receives
one fixed normalized upper action, repeated on a flat floor for many alternating
left/right touchdowns.  The output separates action decoding, lower-level
tracking, stability, and physical termination instead of reducing them to one
reward.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.sampler import quaternion_yaw, wrap_to_pi


def arguments():
    custom = [
        {"name": "--lower_ticks", "type": int, "default": 1200},
        {"name": "--replicas", "type": int, "default": 8},
        {"name": "--levels", "type": int, "default": 3},
        {"name": "--seed", "type": int, "default": 825},
        {"name": "--enable_randomization", "action": "store_true"},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "g0_polar_course_reachability")},
    ]
    args = gymutil.parse_arguments(
        description="polar-course lower reachable-set audit", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    args.terrain_curriculum = "obstacles"
    args.action_profile = "polar_course"
    return args


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def mean(values):
    return float(np.mean(values)) if values else None


def roll_pitch(quaternion):
    x, y, z, w = quaternion.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def main():
    args = arguments()
    if args.replicas < 1:
        raise ValueError("replicas must be positive")
    if args.levels < 2:
        raise ValueError("levels must be at least two")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    levels = tuple(float(value) for value in np.linspace(-1.0, 1.0, args.levels))
    grid = [(distance, direction, yaw)
            for distance in levels for direction in levels for yaw in levels]
    num_envs = len(grid) * args.replicas
    env, policy, interface, _, _, project_cfg = create_upper_system(
        ROOT, args, num_envs=num_envs, seed=args.seed,
        randomization=args.enable_randomization, cameras=False,
        flat_plane=True, obstacles=False, course_length_m=6.0)
    action_index = torch.arange(num_envs, device=env.device) % len(grid)
    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=env.device)
    actions = grid_tensor[action_index]

    # Explicitly expose the action transform.  This catches aliases introduced
    # by minimum lateral separation before any dynamics are involved.
    decoded = {
        str(foot): interface.bounds.decode(
            grid_tensor, torch.full((len(grid),), foot, device=env.device)
        ).detach().cpu().numpy()
        for foot in (0, 1)
    }

    obs, goal, _ = env.get_observations()
    initialized = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    previous_swing = env.sampler.swing_foot.clone()
    option_ticks = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    option_min_height = env.base_position[:, 2].clone()
    option_max_tilt = torch.zeros(num_envs, device=env.device)
    option_max_contact = torch.zeros(num_envs, device=env.device)
    records = defaultdict(lambda: defaultdict(list))
    termination = defaultdict(lambda: defaultdict(int))
    decisions = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    for _ in range(args.lower_ticks):
        lower_action, _ = policy.infer(obs, goal)
        next_obs, _, done, extras, next_goal, _ = env.step(lower_action)
        done = done.bool()
        option_ticks[initialized] += 1
        tilt = torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1.0, 1.0))
        option_min_height = torch.minimum(option_min_height, env.base_position[:, 2])
        option_max_tilt = torch.maximum(option_max_tilt, tilt)
        if env.nonfoot_indices.numel():
            contact = torch.norm(
                env.contact_forces[:, env.nonfoot_indices], dim=-1).max(dim=-1).values
            option_max_contact = torch.maximum(option_max_contact, contact)

        switched = ((env.sampler.swing_foot != previous_swing)
                    & initialized & ~done)
        ids = switched.nonzero(as_tuple=False).flatten()
        if ids.numel():
            landed = 1 - env.sampler.swing_foot[ids]
            other = env.sampler.swing_foot[ids]
            row = torch.arange(ids.numel(), device=env.device)
            landed_xy = env.foot_positions[ids, landed, :2]
            target_xy = env.sampler.target_pos[ids, landed, :2]
            xy_error = torch.norm(landed_xy - target_xy, dim=-1)
            foot_state = env.rigid_body_states[ids][:, env.feet_indices]
            landed_yaw = quaternion_yaw(foot_state[row, landed, 3:7])
            target_yaw = env.sampler.target_yaw[ids, landed]
            yaw_error = torch.abs(wrap_to_pi(landed_yaw - target_yaw))

            # Actual inter-foot displacement in the stationary foot's yaw frame.
            other_yaw = quaternion_yaw(foot_state[row, other, 3:7])
            delta = landed_xy - env.foot_positions[ids, other, :2]
            cosine, sine = torch.cos(other_yaw), torch.sin(other_yaw)
            actual_forward = cosine * delta[:, 0] + sine * delta[:, 1]
            actual_lateral = -sine * delta[:, 0] + cosine * delta[:, 1]
            for row_id, env_id in enumerate(ids.tolist()):
                key = (int(action_index[env_id]), int(landed[row_id]))
                values = {
                    "touchdown_xy_error_m": xy_error[row_id],
                    "touchdown_yaw_error_deg": torch.rad2deg(yaw_error[row_id]),
                    "actual_forward_m": actual_forward[row_id],
                    "actual_lateral_m": actual_lateral[row_id],
                    "option_duration_ticks": option_ticks[env_id],
                    "min_base_height_m": option_min_height[env_id],
                    "max_tilt_deg": torch.rad2deg(option_max_tilt[env_id]),
                    "max_nonfoot_contact_n": option_max_contact[env_id],
                }
                for name, value in values.items():
                    records[key][name].append(float(value))

        reasons = extras["termination_reasons"]
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            grid_id = int(action_index[env_id])
            if bool(extras["time_outs"][env_id]):
                reason = "timeout"
            elif bool(reasons["height"][env_id]):
                reason = "height"
            elif bool(reasons["height_above_lower_reference_limit"][env_id]):
                reason = "height_above_lower_reference_limit"
            elif bool(reasons["tilt"][env_id]):
                reason = "tilt"
            elif bool(reasons["nonfoot_contact"][env_id]):
                reason = "nonfoot_contact"
            else:
                reason = "other"
            termination[grid_id][reason] += 1

        initialized[done] = False
        ready = (~initialized & ~env.goal_reset_pending
                 & (env.episode_length_buf > 0) & ~done)
        update = switched | ready
        update_ids = update.nonzero(as_tuple=False).flatten()
        if update_ids.numel():
            interface.apply(env, actions[update_ids], update_ids)
            decisions[update_ids] += 1
            initialized[update_ids] = True
            option_ticks[update_ids] = 0
            option_min_height[update_ids] = env.base_position[update_ids, 2]
            option_max_tilt[update_ids] = tilt[update_ids]
            option_max_contact[update_ids] = 0.0

        previous_swing.copy_(env.sampler.swing_foot)
        obs, goal = next_obs, next_goal

    results = []
    for grid_id, normalized in enumerate(grid):
        row = {"grid_id": grid_id, "normalized_action": list(normalized),
               "terminations": dict(termination[grid_id]), "legs": {}}
        for foot in (0, 1):
            local = decoded[str(foot)][grid_id]
            data = records[(grid_id, foot)]
            errors = data["touchdown_xy_error_m"]
            row["legs"]["left" if foot == 0 else "right"] = {
                "decoded_local_forward_m": float(local[0]),
                "decoded_local_lateral_m": float(local[1]),
                "decoded_local_radial_m": float(math.hypot(local[0], local[1])),
                "decoded_yaw_deg": float(np.rad2deg(local[3])),
                "touchdowns": len(errors),
                "touchdown_xy_error_mean_m": mean(errors),
                "touchdown_xy_error_p95_m": percentile(errors, 95),
                "touchdown_yaw_error_mean_deg": mean(data["touchdown_yaw_error_deg"]),
                "touchdown_yaw_error_p95_deg": percentile(
                    data["touchdown_yaw_error_deg"], 95),
                "actual_forward_mean_m": mean(data["actual_forward_m"]),
                "actual_lateral_mean_m": mean(data["actual_lateral_m"]),
                "option_duration_mean_ticks": mean(data["option_duration_ticks"]),
                "min_base_height_min_m": (min(data["min_base_height_m"])
                                           if data["min_base_height_m"] else None),
                "max_tilt_max_deg": (max(data["max_tilt_deg"])
                                     if data["max_tilt_deg"] else None),
                "max_nonfoot_contact_n": (max(data["max_nonfoot_contact_n"])
                                           if data["max_nonfoot_contact_n"] else None),
            }
        results.append(row)

    # Count unique physical commands after rounding beyond simulator relevance.
    all_decoded = np.concatenate((decoded["0"], decoded["1"]), axis=0)
    unique_decoded = np.unique(np.round(all_decoded[:, [0, 1, 3]], 6), axis=0)
    leg_rows = [leg for row in results for leg in row["legs"].values()]
    summary = {
        "seed": args.seed,
        "randomization_enabled": bool(args.enable_randomization),
        "replicas_per_grid_point": args.replicas,
        "levels_per_action_dimension": args.levels,
        "num_envs": num_envs,
        "lower_ticks": args.lower_ticks,
        "simulated_seconds_per_env": args.lower_ticks * env.dt,
        "normalized_grid_points": len(grid),
        "decoded_commands_across_both_legs": len(all_decoded),
        "unique_decoded_commands_across_both_legs": len(unique_decoded),
        "leg_grid_cells_with_at_least_8_touchdowns": sum(
            item["touchdowns"] >= 8 for item in leg_rows),
        "leg_grid_cells_p95_xy_below_5cm": sum(
            item["touchdowns"] >= 8
            and item["touchdown_xy_error_p95_m"] is not None
            and item["touchdown_xy_error_p95_m"] < 0.05 for item in leg_rows),
        "physical_termination_counts": {
            name: sum(counts.get(name, 0) for counts in termination.values())
            for name in ("height", "height_above_lower_reference_limit", "tilt",
                         "nonfoot_contact", "other", "timeout")
        },
        "action_bounds": project_cfg["action_polar_course"],
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
