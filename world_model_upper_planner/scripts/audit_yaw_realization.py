#!/usr/bin/env python3
"""Measure the frozen lower controller's realized yaw for every upper option."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # noqa: F401; must precede torch
import torch

from adapters.frozen_lower_env.contracts import FootholdActionBounds
from adapters.frozen_lower_env.factory import create_upper_system
from adapters.frozen_lower_env.rollout import UpperRollout
from cgowm.candidates import make_candidates


def yaw(root):
    x, y, z, w = root[..., 3], root[..., 4], root[..., 5], root[..., 6]
    return torch.atan2(2.0 * (w * z + x * y),
                       1.0 - 2.0 * (y * y + z * z))


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1176)
    parser.add_argument("--lower_ticks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lower_checkpoint", type=Path)
    parser.add_argument("--turn_option_adapter", action="store_true")
    parser.add_argument("--turn_curvature_gain", type=float, default=3.0)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    args = parser.parse_args()
    args.action_profile = "cartesian_course"
    args.terrain_curriculum = "obstacles"
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, randomization=False,
        cameras=False, flat_plane=True, obstacles=False, course_length_m=6.0)
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"],
                           capture_depth=False)
    bounds = FootholdActionBounds.from_config(cfg["action_cartesian_course"])
    candidates = make_candidates(bounds, env.device)
    assigned = torch.arange(env.num_envs, device=env.device) % len(candidates)
    start_yaw = torch.zeros(env.num_envs, device=env.device)
    pending = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    completed_start_yaw = torch.zeros(env.num_envs, device=env.device)
    completed_index = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device)
    rows = []

    @torch.no_grad()
    def choose(_depth, _proprio, ids):
        index = assigned[ids]
        # At a touchdown, UpperRollout asks for the next action before returning
        # the just-completed transition. Preserve the previous option identity
        # and start pose before installing the next one.
        continuing = pending[ids] >= 0
        prior_ids = ids[continuing]
        completed_start_yaw[prior_ids] = start_yaw[prior_ids]
        completed_index[prior_ids] = pending[prior_ids]
        start_yaw[ids] = yaw(env.root_states[ids, 0])
        pending[ids] = index
        return candidates[index]

    for _ in range(args.lower_ticks):
        transition = rollout.lower_tick(choose)
        if transition is None:
            continue
        ids = transition["ids"]
        has_completed = completed_index[ids] >= 0
        index = torch.where(has_completed, completed_index[ids], pending[ids])
        option_start = torch.where(
            has_completed, completed_start_yaw[ids], start_yaw[ids])
        realized = torch.rad2deg(wrap(
            yaw(transition["next_physics"]["root"]) - option_start))
        diag = transition["diagnostics"]
        for row in range(len(ids)):
            rows.append((
                int(index[row]), float(realized[row]),
                bool(diag["fall"][row]),
                float(diag["touchdown_error_m"][row]),
                int(diag["option_duration_ticks"][row]),
            ))
        completed_index[ids] = -1
        pending[ids[transition["done"].bool()]] = -1
    if not rows:
        raise RuntimeError("yaw audit produced no option transitions")
    raw = np.asarray(rows, dtype=[
        ("candidate_index", "i2"), ("realized_yaw_deg", "f4"),
        ("fall", "?"), ("touchdown_error_m", "f4"), ("duration", "i2")])
    candidate_np = candidates.cpu().numpy()
    by_command = {}
    # Group by the yaw levels actually present in the filtered candidate grid.
    # The current grid intentionally uses {-0.5, 0, 0.5}, which decodes to
    # {-6, 0, 6} degrees under the Cartesian-course bounds.
    for normalized in np.unique(candidate_np[:, 2]):
        candidate_ids = np.nonzero(np.isclose(candidate_np[:, 2], normalized))[0]
        mask = np.isin(raw["candidate_index"], candidate_ids)
        values = raw["realized_yaw_deg"][mask]
        commanded_deg = float(bounds._scale(float(normalized), bounds.yaw_deg))
        by_command[f"{commanded_deg:g}"] = {
            "options": int(mask.sum()), "mean_realized_yaw_deg": float(values.mean()),
            "median_realized_yaw_deg": float(np.median(values)),
            "p10_p90_yaw_deg": [float(np.quantile(values, 0.10)),
                                float(np.quantile(values, 0.90))],
            "correct_sign_fraction": float(np.mean(
                np.sign(values) == np.sign(normalized))) if normalized else None,
            "fall_fraction": float(raw["fall"][mask].mean()),
            "touchdown_error_mean_m": float(raw["touchdown_error_m"][mask].mean()),
        }
    detail = []
    for index in range(len(candidates)):
        mask = raw["candidate_index"] == index
        detail.append({
            "candidate_index": index, "action": candidate_np[index].tolist(),
            "options": int(mask.sum()),
            "realized_yaw_mean_deg": float(raw["realized_yaw_deg"][mask].mean()),
            "fall_fraction": float(raw["fall"][mask].mean()),
            "touchdown_error_mean_m": float(raw["touchdown_error_m"][mask].mean()),
        })
    result = {"num_envs": env.num_envs, "lower_ticks": args.lower_ticks,
              "transitions": len(raw), "by_commanded_yaw_deg": by_command,
              "candidates": detail}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "yaw_realization.npz", rows=raw,
                        candidates=candidate_np)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
