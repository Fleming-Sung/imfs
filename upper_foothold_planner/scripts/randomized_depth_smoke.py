#!/usr/bin/env python3
"""Inspect onboard depth on many random courses and optionally model predictions."""

import json
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.depth_diagnostics import (depth_prediction_sequence,
                                             depth_prediction_metrics,
                                             save_depth_prediction)
from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import make_world_model


def arguments():
    custom = [
        {"name": "--num_envs", "type": int, "default": 8},
        {"name": "--transitions", "type": int, "default": 128},
        {"name": "--sequence_horizon", "type": int, "default": 5},
        {"name": "--seed", "type": int, "default": 501},
        {"name": "--course_length_m", "type": float, "default": 2.5},
        {"name": "--random_width_min_m", "type": float, "default": 0.55},
        {"name": "--random_width_max_m", "type": float, "default": 1.20},
        {"name": "--random_gap_min_m", "type": float, "default": 0.0},
        {"name": "--random_gap_max_m", "type": float, "default": 0.10},
        {"name": "--random_obstacle_probability", "type": float, "default": 0.45},
        {"name": "--hurdle_height_min_m", "type": float, "default": 0.025},
        {"name": "--hurdle_height_max_m", "type": float, "default": 0.05},
        {"name": "--checkpoint", "type": str},
        {"name": "--action_profile", "type": str, "default": "polar_course"},
        {"name": "--output", "type": str, "required": True},
    ]
    args = gymutil.parse_arguments(
        description="random terrain depth and prediction smoke", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    args.terrain_curriculum = "randomized"
    return args


def save_sensor_grid(path, replay, layouts, count=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = replay.valid_indices[:count]
    figure, axes = plt.subplots(2, len(valid), figsize=(2.4 * len(valid), 5.0),
                                squeeze=False)
    for column, index in enumerate(valid):
        env_id = int(replay.env_id[index])
        axes[0, column].imshow(replay.depth[index, 0], cmap="viridis", vmin=0, vmax=255)
        axes[0, column].set_title("env {} depth".format(env_id))
        axes[1, column].imshow(layouts[env_id].support_mask, cmap="gray", origin="lower")
        for cx, cy, sx, sy, _ in layouts[env_id].obstacle_rectangles:
            x0 = (cx - 0.5 * sx) / layouts[env_id].spec.resolution_m
            x1 = (cx + 0.5 * sx) / layouts[env_id].spec.resolution_m
            y0 = (cy - 0.5 * sy + 0.5 * layouts[env_id].spec.width_m) / layouts[env_id].spec.resolution_m
            y1 = (cy + 0.5 * sy + 0.5 * layouts[env_id].spec.width_m) / layouts[env_id].spec.resolution_m
            axes[1, column].add_patch(plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, color="red", alpha=0.8))
        axes[1, column].set_title("env {} top view".format(env_id))
        for row in range(2):
            axes[row, column].set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env, lower, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, randomization=False, cameras=True,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        max(2048, args.transitions * 2), num_envs=args.num_envs,
        return_horizon=cfg["model"]["planning_horizon"],
        gamma=cfg["model"]["discount"], reward_scale=cfg["model"]["reward_scale"])
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1)

    def random_actions(depth, proprio, ids):
        del depth, proprio
        return 2.0 * torch.rand(
            ids.numel(), 3, generator=generator, device=env.device) - 1.0

    while (replay.valid_size < args.transitions
           or len(replay.sequence_indices(args.sequence_horizon)) < 1):
        replay.add_transition_batch(rollout.lower_tick(random_actions))

    save_sensor_grid(output / "sensor_and_terrain.png", replay, tiled.layouts)
    valid_depth = replay.depth[replay.valid_indices].astype(np.float32) / 255.0
    summary = {
        "num_envs": args.num_envs,
        "unique_support_masks": len({layout.support_mask.tobytes()
                                     for layout in tiled.layouts}),
        "support_box_counts": [len(layout.support_rectangles) for layout in tiled.layouts],
        "obstacle_box_counts": [len(layout.obstacle_rectangles) for layout in tiled.layouts],
        "depth_min": float(valid_depth.min()),
        "depth_max": float(valid_depth.max()),
        "depth_mean": float(valid_depth.mean()),
        "depth_std": float(valid_depth.std()),
        "depth_zero_fraction": float((valid_depth <= 1.0 / 255.0).mean()),
        "depth_near_fraction": float((valid_depth >= 254.0 / 255.0).mean()),
        "valid_transitions": replay.valid_size,
        "valid_sequences": len(replay.sequence_indices(args.sequence_horizon)),
        "lower_ticks": rollout.lower_ticks,
    }
    if args.checkpoint:
        checkpoints = [Path(item.strip()) for item in args.checkpoint.split(",")]
        rows = replay.sequence_indices(args.sequence_horizon)
        metric_batch = replay.sample_sequence(
            min(256, len(rows)), args.sequence_horizon, env.device)
        image_batch = {key: value[:1] for key, value in metric_batch.items()}
        summary["predictions"] = {}
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=env.device)
            variant = checkpoint.get("args", {}).get("model_variant", "compact")
            model = make_world_model(
                latent_dim=cfg["upper_observation"]["latent_dim"],
                hidden_dim=cfg["model"]["hidden_dim"], variant=variant).to(env.device)
            load = model.load_state_dict(checkpoint["model"], strict=False)
            model.eval()
            arrays = depth_prediction_sequence(model, image_batch)
            # Checkpoints from different runs commonly share names such as
            # model_050000.pt; include the run directory to avoid silent
            # result/image overwrites in architecture comparisons.
            label = "{}__{}".format(checkpoint_path.parent.parent.name,
                                      checkpoint_path.stem)
            image_summary = save_depth_prediction(
                output / ("depth_prediction_{}.png".format(label)), arrays)
            result = {"image_sample": image_summary}
            for mode in ("normal", "shuffled", "zero"):
                result[mode] = depth_prediction_metrics(model, metric_batch, mode)
            result["checkpoint_missing"] = list(load.missing_keys)
            result["checkpoint_unexpected"] = list(load.unexpected_keys)
            summary["predictions"][label] = result
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
