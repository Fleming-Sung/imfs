"""Validate tiled heightfield alignment and synchronized onboard depth sensors."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil  # must precede torch
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.config import AttrDict
from upper_planner.contracts import FootholdActionBounds, preprocess_isaac_depth
from upper_planner.env import FootholdEnv, make_sim_params
from upper_planner.lower_policy import FrozenLowerPolicy
from upper_planner.target_interface import UpperFootholdTargetInterface
from upper_planner.terrain import TerrainSpec, build_tiled_heightfield


def arguments():
    custom = [
        {"name": "--steps", "type": int, "default": 100},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "gate3_terrain_camera_smoke")},
    ]
    args = gymutil.parse_arguments(
        description="terrain and camera synchronization smoke test",
        headless=True, custom_parameters=custom)
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


def save_depth_grid(path, proximity):
    images = np.uint8(np.clip(proximity, 0.0, 1.0) * 255)
    height, width = images.shape[-2:]
    canvas = np.zeros((2 * height, 2 * width), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, 2)
        canvas[row * height:(row + 1) * height,
               column * width:(column + 1) * width] = image
    Image.fromarray(canvas, "L").resize((8 * width, 8 * height)).save(path)


def main():
    args = arguments()
    torch.manual_seed(51)
    np.random.seed(51)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    project_cfg = json.loads((ROOT / "config" / "default.json").read_text())
    checkpoint_path = ROOT / "checkpoints" / "lower_model_7000.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = AttrDict.from_nested(checkpoint["config"])
    cfg.asset.file = str(ROOT / "assets" / "SF_TRON1A" / "urdf" / "robot.urdf")
    cfg.env.num_envs = 4
    cfg.env.env_spacing_xy = [8.0, 4.0]
    cfg.init.spawn_xy = [0.20, 0.0]
    disable_randomization(cfg)

    kinds = ("straight", "s_curve", "fork", "random")
    terrain = build_tiled_heightfield([
        TerrainSpec(kind=kind, corridor_width_m=0.70, seed=51 + index)
        for index, kind in enumerate(kinds)])
    cfg.terrain.height_samples = terrain.height_samples
    cfg.terrain.horizontal_scale = terrain.horizontal_scale_m
    cfg.terrain.vertical_scale = terrain.vertical_scale_m
    cfg.terrain.heightfield_origin_xy = terrain.origin_xy_m.tolist()
    cfg.camera = AttrDict.from_nested(dict(project_cfg["depth"], enabled=True))

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    if env.viewer is not None:
        env.gym.viewer_camera_look_at(
            env.viewer, None, gymapi.Vec3(-1.5, 1.5, 1.0), gymapi.Vec3(0.8, 0.0, 0.2))
    origin_error = float(torch.max(torch.abs(
        env.env_origins[:, :2]
        - torch.as_tensor(terrain.env_origins_xy_m, device=env.device))))
    policy = FrozenLowerPolicy(checkpoint_path, env.device)
    bounds = FootholdActionBounds.from_config(project_cfg["action"])
    interface = UpperFootholdTargetInterface(bounds)
    neutral = torch.zeros(env.num_envs, 3, device=env.device)
    obs, goal, _ = env.get_observations()
    initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    previous_swing = env.sampler.swing_foot.clone()
    depths = []
    base_trace, foot_trace, target_trace, done_trace = [], [], [], []
    fall_count = 0
    start = time.perf_counter()
    for step in range(args.steps):
        action, _ = policy.infer(obs, goal)
        next_obs, _, done, _, next_goal, _ = env.step(action)
        base_trace.append(env.root_states[:, 0].detach().cpu().numpy().copy())
        foot_trace.append(env.foot_positions.detach().cpu().numpy().copy())
        target_trace.append(env.sampler.target_pos.detach().cpu().numpy().copy())
        done_trace.append(done.detach().cpu().numpy().copy())
        fall_count += int(done.sum())
        switched = (env.sampler.swing_foot != previous_swing) & initialized & ~done.bool()
        initialized[done.bool()] = False
        ready = ~initialized & ~env.goal_reset_pending & (env.episode_length_buf > 0)
        ids = (switched | ready).nonzero(as_tuple=False).flatten()
        if ids.numel():
            interface.apply(env, neutral[ids], ids)
            initialized[ids] = True
        previous_swing.copy_(env.sampler.swing_foot)
        obs, goal = next_obs, next_goal
        if step in (1, args.steps - 1):
            depths.append(env.capture_depth().cpu().numpy())
            if env.viewer is not None:
                env.gym.write_viewer_image_to_file(
                    env.viewer, str(output / ("overview_start.png" if step == 1
                                               else "overview_end.png")))
    wall_seconds = time.perf_counter() - start

    depth_cfg = project_cfg["depth"]
    proximity = [preprocess_isaac_depth(
        depth, depth_cfg["near_m"], depth_cfg["far_m"]) for depth in depths]
    save_depth_grid(output / "depth_start.png", proximity[0])
    save_depth_grid(output / "depth_end.png", proximity[-1])
    np.save(output / "depth_raw_start.npy", depths[0])
    np.save(output / "depth_raw_end.npy", depths[-1])
    np.savez_compressed(output / "trajectory.npz", root=np.stack(base_trace),
                        foot_pos=np.stack(foot_trace), target_pos=np.stack(target_trace),
                        done=np.stack(done_trace))
    finite_distance = [-depth[np.isfinite(depth)] for depth in depths]
    metrics = {
        "terrain_kinds": list(kinds),
        "steps": args.steps,
        "num_envs": env.num_envs,
        "origin_alignment_max_error_m": origin_error,
        "falls": fall_count,
        "base_height_m": env.base_position[:, 2].detach().cpu().tolist(),
        "depth_shape": list(depths[-1].shape),
        "depth_finite_fraction_start": float(np.isfinite(depths[0]).mean()),
        "depth_finite_fraction_end": float(np.isfinite(depths[-1]).mean()),
        "depth_distance_range_start_m": [float(min(item.min() for item in finite_distance[:1])),
                                          float(max(item.max() for item in finite_distance[:1]))],
        "depth_distance_range_end_m": [float(finite_distance[-1].min()),
                                        float(finite_distance[-1].max())],
        "depth_mean_abs_change": float(np.mean(np.abs(proximity[-1] - proximity[0]))),
        "control_steps_per_wall_second": args.steps * env.num_envs / wall_seconds,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
