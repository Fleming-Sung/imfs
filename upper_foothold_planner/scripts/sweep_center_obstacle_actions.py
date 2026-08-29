"""Prove whether valid turn-and-return foothold sequences can clear a centered box."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.rollout import UpperRollout


def arguments():
    custom = [
        {"name": "--lower_ticks", "type": int, "default": 1500},
        {"name": "--seed", "type": int, "default": 501},
        {"name": "--output", "type": str, "required": True},
        {"name": "--action_profile", "type": str, "default": "polar"},
    ]
    args = gymutil.parse_arguments(
        description="center obstacle action capability sweep", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def main():
    args = arguments()
    patterns = list(itertools.product(
        (0.0, 1.0), (2, 3, 4), (4, 6, 8), (-1.0, 1.0)))
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, len(patterns), args.seed, corridor_width_m=2.5,
        randomization=False, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=2.25, obstacle_count=1, obstacle_y_m=0.0,
        reward_override={"progress": 10.0, "collision": -2.0})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    decision = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    forward = torch.tensor([p[0] for p in patterns], device=env.device)
    side_steps = torch.tensor([p[1] for p in patterns], device=env.device)
    forward_steps = torch.tensor([p[2] for p in patterns], device=env.device)
    direction = torch.tensor([p[3] for p in patterns], device=env.device)

    def scripted_actions(depth, proprio, ids):
        del depth, proprio
        step = decision[ids]
        first_end = side_steps[ids]
        straight_end = first_end + forward_steps[ids]
        return_end = straight_end + side_steps[ids]
        step_direction = torch.where(
            step < first_end, direction[ids],
            torch.where(step < straight_end, torch.zeros_like(direction[ids]),
                        torch.where(step < return_end, -direction[ids],
                                    torch.zeros_like(direction[ids]))))
        action = torch.stack((forward[ids], step_direction,
                              torch.zeros_like(step_direction)), dim=-1)
        decision[ids] += 1
        return action

    collisions = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    falls = torch.zeros_like(collisions)
    successes = torch.zeros_like(collisions)
    max_forward = torch.full((env.num_envs,), -float("inf"), device=env.device)
    max_lateral = torch.zeros(env.num_envs, device=env.device)
    for _ in range(args.lower_ticks):
        transition = rollout.lower_tick(scripted_actions)
        relative = env.base_position[:, :2] - env.env_origins[:, :2]
        max_forward = torch.maximum(max_forward, relative[:, 0])
        max_lateral = torch.maximum(max_lateral, relative[:, 1].abs())
        if transition is not None:
            ids = transition["ids"]
            collisions[ids] += transition["diagnostics"]["collision"].long()
            falls[ids] += transition["diagnostics"]["fall"].long()
            successes[ids] += transition["diagnostics"]["success"].long()

    final_distance = torch.norm(env.base_position[:, :2] - task.goals, dim=-1)
    rows = []
    for index, pattern in enumerate(patterns):
        rows.append({
            "env": index, "distance_action": pattern[0],
            "side_steps": pattern[1], "forward_steps_between": pattern[2],
            "side_direction": pattern[3],
            "successes": int(successes[index]), "falls": int(falls[index]),
            "collisions": int(collisions[index]),
            "max_forward_m": float(max_forward[index]),
            "max_abs_lateral_m": float(max_lateral[index]),
            "final_distance_m": float(final_distance[index]),
        })
    ranked = sorted(rows, key=lambda row: (
        -row["successes"], row["falls"], row["collisions"], row["final_distance_m"]))
    summary = {
        "patterns": len(patterns), "centered_obstacle_xy_m": [1.0, 0.0],
        "successful_patterns": sum(row["successes"] > 0 for row in rows),
        "top10": ranked[:10], "all": rows,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "all"}, indent=2))


if __name__ == "__main__":
    main()
