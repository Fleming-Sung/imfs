#!/usr/bin/env python3
"""Collect real frozen-lower option transitions without a teacher policy."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # noqa: F401; must precede torch
import torch

from adapters.frozen_lower_env.contracts import FootholdActionBounds
from adapters.frozen_lower_env.factory import create_upper_system
from adapters.frozen_lower_env.rollout import UpperRollout
from adapters.geometry_labels import CandidateGeometryLabeler
from cgowm.candidates import make_candidates


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--lower_ticks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--terrain_curriculum",
        choices=("research", "typical", "obstacles", "randomized"),
        default="research")
    parser.add_argument(
        "--research_kind",
        choices=("mixed", "edge_cases", "random_composite",
                 "stepping_stones", "turns", "household"),
        default="random_composite",
        help="Research terrain family for exact training-only geometry labels")
    parser.add_argument(
        "--typical_kind",
        choices=("mixed", "narrow_bridge", "irregular_support", "hurdles"),
        default="mixed")
    parser.add_argument("--difficulty_tag", default="nominal")
    parser.add_argument("--course_length_m", type=float, default=3.5)
    parser.add_argument("--corridor_width_m", type=float, default=0.90)
    parser.add_argument("--bridge_width_min_m", type=float, default=0.55)
    parser.add_argument("--bridge_width_max_m", type=float, default=0.75)
    parser.add_argument("--irregular_width_m", type=float, default=0.65)
    parser.add_argument("--hurdle_height_min_m", type=float, default=0.02)
    parser.add_argument("--hurdle_height_max_m", type=float, default=0.055)
    parser.add_argument("--random_width_min_m", type=float, default=0.50)
    parser.add_argument("--random_width_max_m", type=float, default=1.30)
    parser.add_argument("--random_gap_min_m", type=float, default=0.0)
    parser.add_argument("--random_gap_max_m", type=float, default=0.14)
    parser.add_argument("--random_obstacle_probability", type=float, default=0.0)
    parser.add_argument("--behavior", choices=("random", "safe_diverse"),
                        default="safe_diverse")
    parser.add_argument("--uniform_valid_fraction", type=float, default=0.25)
    parser.add_argument("--topk_progress", type=int, default=12)
    parser.add_argument("--unsafe_random_fraction", type=float, default=0.10,
                        help="fraction of all-grid actions for failure coverage")
    parser.add_argument("--reset_curriculum_prob", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = arguments()
    args.action_profile = "cartesian_course"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, lower, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed,
        corridor_width_m=args.corridor_width_m, randomization=True,
        cameras=True, flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    bounds = FootholdActionBounds.from_config(cfg["action_cartesian_course"])
    candidates = make_candidates(bounds, env.device)
    labeler = CandidateGeometryLabeler(
        tiled, bounds, candidates, env.device)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1009)
    rows = []
    pending_geometry = [deque() for _ in range(env.num_envs)]
    episode_id = np.zeros(env.num_envs, dtype=np.int64)
    option_index = np.zeros(env.num_envs, dtype=np.int32)

    @torch.no_grad()
    def choose(_depth, _proprio, ids):
        # Stratification is deterministic across env ids; 20% uniform indices
        # prevent the replay from containing only a few gait-compatible modes.
        base = (ids + rollout.lower_ticks // 20) % len(candidates)
        random_index = torch.randint(
            len(candidates), (len(ids),), generator=generator, device=env.device)
        random_mask = torch.rand(
            len(ids), generator=generator, device=env.device) < 0.20
        geometry = labeler.label(env, ids)
        if args.behavior == "random":
            selected = torch.where(random_mask, random_index, base)
        else:
            selected_rows = []
            for row in range(len(ids)):
                valid = geometry["candidate_valid"][row].nonzero(
                    as_tuple=False).flatten()
                if not len(valid):
                    selected_rows.append(torch.argmax(
                        geometry["candidate_support"][row]))
                    continue
                use_uniform = bool(torch.rand(
                    (), generator=generator, device=env.device)
                    < args.uniform_valid_fraction)
                if use_uniform:
                    choice = valid[torch.randint(
                        len(valid), (), generator=generator, device=env.device)]
                else:
                    count = min(int(args.topk_progress), len(valid))
                    top = valid[torch.topk(
                        geometry["candidate_progress"][row, valid], count).indices]
                    choice = top[torch.randint(
                        len(top), (), generator=generator, device=env.device)]
                selected_rows.append(choice)
            selected = torch.stack(selected_rows)
            unsafe = torch.rand(
                len(ids), generator=generator, device=env.device
            ) < args.unsafe_random_fraction
            selected = torch.where(unsafe, random_index, selected)
        for row, env_id in enumerate(ids.tolist()):
            pending_geometry[env_id].append({
                **{name: value[row].cpu().numpy()
                   for name, value in geometry.items()},
                "candidate_index": int(selected[row]),
                "episode_id": int(episode_id[env_id]),
                "option_index": int(option_index[env_id]),
                "terrain_kind": tiled.layouts[env_id].spec.kind,
                "difficulty": args.difficulty_tag,
            })
            option_index[env_id] += 1
        return candidates[selected]

    start = time.perf_counter()
    for _ in range(args.lower_ticks):
        transition = rollout.lower_tick(choose)
        if transition is None:
            continue
        diagnostics = transition["diagnostics"]
        raw_progress = transition["terms"]["progress"] / float(cfg["reward"]["progress"])
        for index, env_id in enumerate(transition["ids"].tolist()):
            if not pending_geometry[env_id]:
                raise RuntimeError("missing decision-time geometry labels")
            geometry = pending_geometry[env_id].popleft()
            rows.append({
                "depth": transition["depth"][index].cpu().numpy(),
                "proprio": transition["proprio"][index].cpu().numpy(),
                "action": transition["action"][index].cpu().numpy(),
                "reward": float(transition["reward"][index]),
                "next_depth": transition["next_depth"][index].cpu().numpy(),
                "next_proprio": transition["next_proprio"][index].cpu().numpy(),
                "done": bool(transition["done"][index]),
                "progress": float(raw_progress[index]),
                "support": float(diagnostics["support_fraction"][index]),
                "touchdown_error": float(diagnostics["touchdown_error_m"][index]),
                "fall": bool(diagnostics["fall"][index]),
                "collision": bool(diagnostics["collision"][index]),
                "success": bool(diagnostics["success"][index]),
                "duration": int(diagnostics["option_duration_ticks"][index]),
                "env_id": int(env_id),
                **geometry,
            })
            if bool(transition["done"][index]):
                episode_id[env_id] += 1
                option_index[env_id] = 0
    wall_seconds = time.perf_counter() - start
    if not rows:
        raise SystemExit("no option transitions collected")
    args.output.mkdir(parents=True, exist_ok=True)

    def array(name, dtype=None):
        result = np.asarray([row[name] for row in rows])
        return result.astype(dtype) if dtype is not None else result

    np.savez_compressed(
        args.output / "transitions.npz",
        depth=np.uint8(np.clip(array("depth"), 0, 1) * 255 + 0.5),
        proprio=array("proprio", np.float32),
        action=array("action", np.float32),
        reward=array("reward", np.float32),
        next_depth=np.uint8(np.clip(array("next_depth"), 0, 1) * 255 + 0.5),
        next_proprio=array("next_proprio", np.float32),
        done=array("done", np.bool_), progress=array("progress", np.float32),
        support=array("support", np.float32),
        touchdown_error=array("touchdown_error", np.float32),
        fall=array("fall", np.bool_), collision=array("collision", np.bool_),
        success=array("success", np.bool_), duration=array("duration", np.int16),
        env_id=array("env_id", np.int32),
        episode_id=array("episode_id", np.int64),
        option_index=array("option_index", np.int32),
        candidate_index=array("candidate_index", np.int16),
        candidate_support=array("candidate_support", np.float32),
        candidate_progress=array("candidate_progress", np.float32),
        candidate_valid=array("candidate_valid", np.bool_),
        candidate_alignment=array("candidate_alignment", np.float32),
        terrain_kind=array("terrain_kind"),
        difficulty=array("difficulty"),
        candidates=candidates.cpu().numpy())
    summary = {
        "transitions": len(rows),
        "falls": int(sum(row["fall"] for row in rows)),
        "successes": int(sum(row["success"] for row in rows)),
        "mean_progress_m": float(np.mean([row["progress"] for row in rows])),
        "rows_with_valid_candidate_fraction": float(np.mean([
            np.any(row["candidate_valid"]) for row in rows])),
        "candidate_valid_fraction": float(np.mean([
            row["candidate_valid"] for row in rows])),
        "selected_candidate_count": int(len(set(
            row["candidate_index"] for row in rows))),
        "terrain_counts": {str(kind): int(count) for kind, count in zip(
            *np.unique([row["terrain_kind"] for row in rows], return_counts=True))},
        "difficulty": args.difficulty_tag,
        "behavior": args.behavior,
        "wall_seconds": wall_seconds,
        "lower_env_steps_per_second": (
            args.lower_ticks * args.num_envs / max(wall_seconds, 1e-9)),
        "output": str(args.output / "transitions.npz"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
