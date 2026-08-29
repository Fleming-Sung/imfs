"""Collect a teacher-supervised foothold dataset for Gate B/C distillation.

Runs the privileged terrain planner (teacher) with the depth camera enabled and
records, at every upper decision, the deployable observation (depth + proprio)
together with the teacher's chosen candidate index, the full per-candidate
privileged labels, and the outcome of the episode the decision belonged to.

Only decisions from *successful* episodes are retained by default, so behaviour
cloning does not imitate the teacher's failures.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # must precede torch
import torch

from upper_planner.contracts import FootholdActionBounds
from upper_planner.factory import create_upper_system
from upper_planner.privileged_planner import (
    PrivilegedPlannerConfig, PrivilegedTerrainPlanner)
from upper_planner.rollout import UpperRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--lower_ticks", type=int, default=1200)
    parser.add_argument("--terrain_curriculum", type=str, default="research")
    parser.add_argument("--research_kind", type=str, default="random_composite")
    parser.add_argument("--course_length_m", type=float, default=3.5)
    parser.add_argument("--random_width_min_m", type=float, default=0.50)
    parser.add_argument("--random_width_max_m", type=float, default=1.30)
    parser.add_argument("--random_gap_max_m", type=float, default=0.14)
    parser.add_argument("--random_obstacle_probability", type=float, default=0.0)
    parser.add_argument("--privileged_forward_levels", type=str,
                        default="-1.0,-0.818182,-0.636364,-0.454545,-0.272727,"
                                "-0.090909,0.090909,0.272727,0.454545,0.636364,"
                                "0.818182,1.0")
    parser.add_argument("--privileged_lateral_levels", type=str,
                        default="-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--privileged_yaw_levels", type=str, default="-0.5,0.0,0.5")
    parser.add_argument("--keep_failed", action="store_true",
                        help="keep decisions from failed episodes too")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _levels(text):
    return tuple(float(value) for value in text.split(","))


def main():
    args = parse_args()
    args.action_profile = "cartesian_course"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env, policy, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=1.5,
        randomization=False, cameras=True,  # camera ON for student observation
        flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(
        env, policy, interface, task, cfg["depth"], capture_depth=True)
    bounds = FootholdActionBounds.from_config(cfg["action_cartesian_course"])
    planner = PrivilegedTerrainPlanner(
        tiled, bounds, env.device,
        PrivilegedPlannerConfig(
            forward_levels=_levels(args.privileged_forward_levels),
            lateral_levels=_levels(args.privileged_lateral_levels),
            yaw_levels=_levels(args.privileged_yaw_levels)))

    pending = [deque() for _ in range(env.num_envs)]
    depths, proprios = [], []
    env_ids = []
    candidate_indices, supports, progresses, valids, scores = [], [], [], [], []
    success_labels = []  # one per decision, None until its episode ends

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        actions, diag = planner.plan(env, ids, rollout.previous_action[ids])
        ids_cpu = ids.cpu().tolist()
        batch_start = len(success_labels)
        for k, env_id in enumerate(ids_cpu):
            pending[env_id].append(batch_start + k)
        depths.append(depth.cpu().numpy())
        proprios.append(proprio.cpu().numpy())
        env_ids.append(np.asarray(ids_cpu, dtype=np.int64))
        candidate_indices.append(diag["candidate_index"].cpu().numpy())
        supports.append(diag["candidate_support_fraction"].cpu().numpy())
        progresses.append(diag["candidate_geodesic_progress_m"].cpu().numpy())
        valids.append(diag["candidate_valid"].cpu().numpy())
        scores.append(diag["candidate_score"].cpu().numpy())
        success_labels.extend([None] * len(ids_cpu))
        return actions

    for _ in range(args.lower_ticks):
        transitions = rollout.lower_tick(choose_actions)
        if transitions is None:
            continue
        done = transitions["done"].cpu().numpy()
        success = transitions["diagnostics"]["success"].cpu().numpy()
        ids = transitions["ids"].cpu().tolist()
        for i, env_id in enumerate(ids):
            if not done[i]:
                continue
            outcome = bool(success[i])
            while pending[env_id]:
                success_labels[pending[env_id].popleft()] = outcome

    labeled = [i for i, value in enumerate(success_labels) if value is not None]
    if not labeled:
        raise SystemExit("no completed episodes; increase --lower_ticks")

    depth = np.concatenate(depths, axis=0)[labeled]
    proprio = np.concatenate(proprios, axis=0)[labeled]
    env_id = np.concatenate(env_ids, axis=0)[labeled]
    candidate_index = np.concatenate(candidate_indices, axis=0)[labeled]
    support = np.concatenate(supports, axis=0)[labeled]
    progress = np.concatenate(progresses, axis=0)[labeled]
    valid = np.concatenate(valids, axis=0)[labeled]
    score = np.concatenate(scores, axis=0)[labeled]
    success = np.asarray([success_labels[i] for i in labeled], dtype=np.bool_)

    # Candidate set (normalized levels + decoded physical cartesian target).
    swing = torch.zeros(len(planner.candidates), dtype=torch.long)
    decoded = bounds.decode(planner.candidates, swing).cpu().numpy()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "dataset.npz",
        depth=np.uint8(np.clip(depth, 0.0, 1.0) * 255.0 + 0.5),
        proprio=proprio.astype(np.float32),
        env_id=env_id.astype(np.int64),
        candidate_index=candidate_index.astype(np.int64),
        candidate_support=support.astype(np.float32),
        candidate_progress=progress.astype(np.float32),
        candidate_valid=valid.astype(np.bool_),
        candidate_score=score.astype(np.float32),
        success=success,
        candidates_levels=planner.candidates.cpu().numpy().astype(np.float32),
        candidates_decoded=decoded.astype(np.float32))

    n = len(success)
    n_success = int(success.sum()) if n else 0
    print(f"collected {n} decisions, {n_success} from successful episodes "
          f"({n_success / max(n, 1):.1%}), saved to {output / 'dataset.npz'}")


if __name__ == "__main__":
    main()
