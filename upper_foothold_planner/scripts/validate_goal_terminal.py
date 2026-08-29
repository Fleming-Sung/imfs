"""Integration gate for goal-success termination and physical reset semantics."""

import json
import itertools
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout


def arguments():
    custom = [
        {"name": "--lower_ticks", "type": int, "default": 1500},
        {"name": "--seed", "type": int, "default": 501},
        {"name": "--output", "type": str, "required": True},
        {"name": "--action_profile", "type": str, "default": "polar"},
    ]
    args = gymutil.parse_arguments(
        description="goal terminal integration gate", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    patterns = list(itertools.product(
        (0.0, 1.0), (2, 3, 4), (4, 6, 8), (-1.0, 1.0)))
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, len(patterns), args.seed, corridor_width_m=2.5,
        randomization=False, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=2.25, obstacle_count=1, obstacle_y_m=0.0,
        reward_override={"progress": 10.0, "collision": -2.0})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        capacity=65536, num_envs=len(patterns),
        return_horizon=cfg["model"]["planning_horizon"],
        gamma=cfg["model"]["discount"],
        reward_scale=cfg["model"]["reward_scale"])
    decision = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    forward = torch.tensor([p[0] for p in patterns], device=env.device)
    side_steps = torch.tensor([p[1] for p in patterns], device=env.device)
    forward_steps = torch.tensor([p[2] for p in patterns], device=env.device)
    side_direction = torch.tensor([p[3] for p in patterns], device=env.device)

    def route(depth, proprio, ids):
        del depth, proprio
        # Restart each route only when a physical episode has actually reset.
        newly_reset = env.episode_length_buf[ids] <= 1
        decision[ids[newly_reset]] = 0
        step = decision[ids]
        first_end = side_steps[ids]
        straight_end = first_end + forward_steps[ids]
        return_end = straight_end + side_steps[ids]
        direction = torch.where(
            step < first_end, side_direction[ids],
            torch.where(step < straight_end, torch.zeros_like(side_direction[ids]),
                        torch.where(step < return_end, -side_direction[ids],
                                    torch.zeros_like(side_direction[ids]))))
        decision[ids] += 1
        return torch.stack((forward[ids], direction,
                            torch.zeros_like(direction)), dim=-1)

    successes = []
    successful_envs = set()
    reset_ready_seen = False
    for tick in range(args.lower_ticks):
        transition = rollout.lower_tick(route)
        if transition is None:
            continue
        replay.add_transition_batch(transition)
        success_rows = transition["diagnostics"]["success"].nonzero(
            as_tuple=False).flatten()
        for row in success_rows.tolist():
            success_id = int(transition["ids"][row])
            successful_envs.add(success_id)
            record = {
                "lower_tick": tick,
                "env": success_id,
                "pattern": list(patterns[success_id]),
                "transition_done": bool(transition["done"][row]),
                "transition_fall": bool(transition["diagnostics"]["fall"][row]),
                "terminal_distance_m": float(
                    transition["diagnostics"]["distance_to_goal_m"][row]),
                "post_reset_episode_length": int(env.episode_length_buf[success_id]),
                "post_reset_goal_pending": bool(env.goal_reset_pending[success_id]),
                "post_reset_rollout_initialized": bool(rollout.initialized[success_id]),
                "post_reset_base_relative_xyz": (
                    env.base_position[success_id] - env.env_origins[success_id]
                ).detach().cpu().tolist(),
                "post_reset_all_links_finite": bool(torch.isfinite(
                    env.rigid_body_states[success_id]).all()),
                "replay_pending_after_terminal": len(replay.pending[success_id]),
            }
            successes.append(record)
        if successes and any(
                not env.goal_reset_pending[index] and rollout.initialized[index]
                for index in successful_envs):
            reset_ready_seen = True
            break

    summary = {
        "passed": bool(successes
                       and all(item["transition_done"] for item in successes)
                       and all(not item["transition_fall"] for item in successes)
                       and all(item["post_reset_episode_length"] == 0 for item in successes)
                       and all(item["post_reset_goal_pending"] for item in successes)
                       and all(not item["post_reset_rollout_initialized"] for item in successes)
                       and all(item["post_reset_all_links_finite"] for item in successes)
                       and all(item["replay_pending_after_terminal"] == 0 for item in successes)
                       and reset_ready_seen),
        "successes": successes,
        "reset_ready_seen": reset_ready_seen,
        "replay_size": replay.size,
        "replay_valid_size": replay.valid_size,
        "ticks_executed": tick + 1,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("goal terminal integration gate failed")


if __name__ == "__main__":
    main()
