"""Closed-loop evaluation of a behaviour-cloned candidate Actor (Gate B/C).

Runs the frozen lower policy with the trained CandidateActor as the upper
planner, using only the deployable observation (depth + 36-D proprio).  The
candidate grid is read from the checkpoint so the student never touches
privileged terrain state.

Reports per-episode success / fall / timeout rates on a fixed terrain set.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # must precede torch
import torch

from upper_planner.candidate_actor import CandidateActor
from upper_planner.factory import create_upper_system
from upper_planner.rollout import UpperRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--lower_ticks", type=int, default=1500)
    parser.add_argument("--terrain_curriculum", type=str, default="research")
    parser.add_argument("--research_kind", type=str, default="random_composite")
    parser.add_argument("--course_length_m", type=float, default=3.5)
    parser.add_argument("--random_width_min_m", type=float, default=0.50)
    parser.add_argument("--random_width_max_m", type=float, default=1.30)
    parser.add_argument("--random_gap_max_m", type=float, default=0.14)
    parser.add_argument("--random_obstacle_probability", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.action_profile = "cartesian_course"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    candidates_levels = checkpoint["candidates_levels"].to(torch.float32)
    num_candidates = int(checkpoint["num_candidates"])
    model = CandidateActor(
        num_candidates, feature_dim=int(checkpoint["feature_dim"]),
        gru_hidden=int(checkpoint["gru_hidden"]))
    # Accept both BC checkpoints (state_dict) and PPO checkpoints (actor).
    state_dict = checkpoint.get("actor", checkpoint.get("state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    env, policy, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=1.5,
        randomization=False, cameras=True,
        flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(
        env, policy, interface, task, cfg["depth"], capture_depth=True)

    device = env.device
    model = model.to(device)
    candidates_levels = candidates_levels.to(device)

    outcomes = defaultdict(int)  # success / fall / timeout

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        indices, _ = model.select(depth, proprio)
        return candidates_levels[indices]

    for _ in range(args.lower_ticks):
        transitions = rollout.lower_tick(choose_actions)
        if transitions is None:
            continue
        done = transitions["done"].cpu().numpy()
        success = transitions["diagnostics"]["success"].cpu().numpy()
        timeout = transitions["next_physics"]["timeout"].cpu().numpy()
        for i in range(len(done)):
            if not done[i]:
                continue
            if success[i]:
                outcomes["success"] += 1
            elif timeout[i]:
                outcomes["timeout"] += 1
            else:
                outcomes["fall"] += 1

    total = sum(outcomes.values())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "episodes": total,
        "success": outcomes["success"],
        "fall": outcomes["fall"],
        "timeout": outcomes["timeout"],
        "success_rate": outcomes["success"] / max(total, 1),
        "fall_rate": outcomes["fall"] / max(total, 1),
        "timeout_rate": outcomes["timeout"] / max(total, 1),
    }
    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"episodes {total}: success {outcomes['success']} "
          f"({summary['success_rate']:.1%}), fall {outcomes['fall']} "
          f"({summary['fall_rate']:.1%}), timeout {outcomes['timeout']} "
          f"({summary['timeout_rate']:.1%})")
    print(f"saved to {output / 'summary.json'}")


if __name__ == "__main__":
    main()
