"""Run model_7000 in the independent Isaac Gym lower-controller snapshot.

This is a validation tool, not the upper-planner task. It records the complete
rigid-body and joint trajectory so stability is never inferred from reward alone.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil  # Isaac Gym must be imported before torch.
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.config import AttrDict
from upper_planner.contracts import FootholdActionBounds
from upper_planner.env import FootholdEnv, make_sim_params
from upper_planner.lower_policy import FrozenLowerPolicy
from upper_planner.target_interface import UpperFootholdTargetInterface


def parse_args():
    custom = [
        {"name": "--checkpoint", "type": str,
         "default": str(ROOT / "checkpoints" / "lower_model_7000.pt")},
        {"name": "--steps", "type": int, "default": 1000},
        {"name": "--seed", "type": int, "default": 42},
        {"name": "--num_envs", "type": int, "default": 1},
        {"name": "--output_dir", "type": str,
         "default": str(ROOT / "experiments" / "gate1_lower_rollout")},
        {"name": "--disable_randomization", "action": "store_true"},
        {"name": "--record_video", "action": "store_true"},
        {"name": "--video_fps", "type": int, "default": 25},
        {"name": "--upper_forward_m", "type": float},
        {"name": "--upper_lateral_m", "type": float},
        {"name": "--upper_yaw_deg", "type": float},
    ]
    args = gymutil.parse_arguments(
        description="independent frozen lower-controller evaluation",
        headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def disable_randomization(cfg):
    cfg.noise.add_noise = False
    for name in (
            "randomize_friction", "randomize_base_mass", "randomize_link_mass",
            "randomize_base_com", "randomize_Kp", "randomize_Kd",
            "randomize_gravity", "randomize_joint_damping",
            "randomize_joint_friction", "randomize_joint_armature",
            "kick_robots"):
        cfg.domain_rand[name] = False


def quaternion_roll_pitch(q):
    x, y, z, w = [q[..., i] for i in range(4)]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, torch.asin(pitch_arg)


def normalized(value, limits):
    return 2.0 * (value - limits[0]) / (limits[1] - limits[0]) - 1.0


def set_rear_side_camera(env):
    """Track env 0 from about 2 m behind-left at base-link height."""
    root = env.root_states[0, 0]
    x, y, z, w = root[3:7]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
    left = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
    xy = root[:2] - 1.72 * forward + 1.02 * left
    camera = gymapi.Vec3(float(xy[0]), float(xy[1]), float(root[2] + 0.05))
    target = gymapi.Vec3(float(root[0]), float(root[1]), float(root[2]))
    env.gym.viewer_camera_look_at(env.viewer, None, camera, target)


def main():
    args = parse_args()
    upper_values = (args.upper_forward_m, args.upper_lateral_m, args.upper_yaw_deg)
    if any(value is not None for value in upper_values) and not all(
            value is not None for value in upper_values):
        raise SystemExit("explicit target needs forward, lateral and yaw together")
    if args.record_video and args.headless:
        raise SystemExit("--record_video requires viewer rendering; omit --headless")
    if args.record_video and args.num_envs != 1:
        raise SystemExit("--record_video currently requires --num_envs 1")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = AttrDict.from_nested(checkpoint["config"])
    cfg.asset.file = str(ROOT / "assets" / "SF_TRON1A" / "urdf" / "robot.urdf")
    cfg.env.num_envs = args.num_envs
    if args.disable_randomization:
        disable_randomization(cfg)

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    policy = FrozenLowerPolicy(args.checkpoint, env.device)
    obs, goal, _ = env.get_observations()
    target_interface = None
    explicit_action = None
    if all(value is not None for value in upper_values):
        action_cfg = json.loads((ROOT / "config" / "default.json").read_text())["action"]
        bounds = FootholdActionBounds.from_config(action_cfg)
        explicit_action = torch.tensor([
            normalized(args.upper_forward_m, bounds.forward_m),
            normalized(args.upper_lateral_m, bounds.lateral_abs_m),
            normalized(args.upper_yaw_deg, bounds.yaw_deg),
        ], dtype=torch.float32, device=env.device)
        if (explicit_action.abs() > 1.00001).any():
            raise SystemExit("explicit target is outside configured upper action bounds")
        target_interface = UpperFootholdTargetInterface(bounds)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frames"
    if args.record_video:
        frame_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, int(round(1.0 / (env.dt * args.video_fps))))
    frame_index = 0

    names = (
        "root", "rigid_body", "contact_force", "dof_pos", "dof_vel", "torque",
        "foot_pos", "foot_vel", "target_pos", "target_yaw", "swing_foot",
        "phase", "raw_obs", "raw_goal", "raw_action", "action", "reward", "done")
    traces = {name: [] for name in names}
    falls = 0
    switch_errors = []
    previous_swing = env.sampler.swing_foot.clone()
    initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(args.steps):
        step_index = len(traces["root"])
        if args.record_video and step_index % frame_stride == 0:
            set_rear_side_camera(env)
        raw_action, action = policy.infer(obs, goal)
        next_obs, reward, done, extras, next_goal, _ = env.step(raw_action)
        if args.record_video and step_index % frame_stride == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / "{:06d}.png".format(frame_index)))
            frame_index += 1

        traces["root"].append(cpu(env.root_states[:, 0]))
        traces["rigid_body"].append(cpu(env.rigid_body_states))
        traces["contact_force"].append(cpu(env.contact_forces))
        traces["dof_pos"].append(cpu(env.dof_pos))
        traces["dof_vel"].append(cpu(env.dof_vel))
        traces["torque"].append(cpu(env.torques))
        traces["foot_pos"].append(cpu(env.foot_positions))
        traces["foot_vel"].append(cpu(env.foot_velocities))
        traces["target_pos"].append(cpu(env.sampler.target_pos))
        traces["target_yaw"].append(cpu(env.sampler.target_yaw))
        traces["swing_foot"].append(cpu(env.sampler.swing_foot))
        traces["phase"].append(cpu(env.sampler.phase))
        traces["raw_obs"].append(cpu(obs))
        traces["raw_goal"].append(cpu(goal))
        traces["raw_action"].append(cpu(raw_action))
        traces["action"].append(cpu(action))
        traces["reward"].append(cpu(reward))
        traces["done"].append(cpu(done))

        # The first post-reset sampler initialization is not a touchdown.
        switched = (env.sampler.swing_foot != previous_swing) & initialized & ~done.bool()
        if switched.any():
            ids = switched.nonzero(as_tuple=False).flatten()
            landed = 1 - env.sampler.swing_foot[ids]
            row = torch.arange(ids.numel(), device=env.device)
            error = torch.norm(
                env.foot_positions[ids, landed, :2]
                - env.sampler.target_pos[ids, landed, :2], dim=-1)
            switch_errors.extend(cpu(error).tolist())
        initialized[done.bool()] = False
        if target_interface is not None:
            ready = ~initialized & ~env.goal_reset_pending & (env.episode_length_buf > 0)
            update_ids = (switched | ready).nonzero(as_tuple=False).flatten()
            if update_ids.numel():
                target_interface.apply(env, explicit_action, update_ids)
                initialized[update_ids] = True
        else:
            initialized |= env.episode_length_buf > 1
        previous_swing.copy_(env.sampler.swing_foot)

        time_outs = extras.get("time_outs", torch.zeros_like(done, dtype=torch.bool))
        falls += int((done.bool() & ~time_outs.bool()).sum().item())
        obs, goal = next_obs, next_goal

    arrays = {name: np.stack(values) for name, values in traces.items()}
    root = torch.as_tensor(arrays["root"])
    roll, pitch = quaternion_roll_pitch(root[..., 3:7])
    contact_norm = np.linalg.norm(arrays["contact_force"], axis=-1)
    body_contact_indices = cpu(env.body_contact_indices).astype(np.int64)
    nonfoot_contact_max = (float(contact_norm[..., body_contact_indices].max())
                           if body_contact_indices.size else 0.0)
    finite = all(np.isfinite(value).all() for value in arrays.values())

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "seed": args.seed,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "simulated_seconds_per_env": args.steps * env.dt,
        "randomization_disabled": bool(args.disable_randomization),
        "explicit_upper_target": list(upper_values) if target_interface is not None else None,
        "falls": falls,
        "finite_all_traces": finite,
        "base_height_min_m": float(arrays["root"][..., 2].min()),
        "base_height_max_m": float(arrays["root"][..., 2].max()),
        "base_abs_roll_max_deg": float(torch.rad2deg(roll.abs()).max()),
        "base_abs_pitch_max_deg": float(torch.rad2deg(pitch.abs()).max()),
        "nonfoot_contact_force_max_n": nonfoot_contact_max,
        "touchdown_xy_error_mean_m": float(np.mean(switch_errors)) if switch_errors else None,
        "touchdown_xy_error_p95_m": float(np.percentile(switch_errors, 95)) if switch_errors else None,
        "touchdowns": len(switch_errors),
        "reward_mean": float(arrays["reward"].mean()),
        "body_names": list(env.gym.get_actor_rigid_body_names(
            env.envs[0], env.actor_handles[0])),
        "dof_names": list(env.dof_names),
    }

    np.savez_compressed(str(output / "trajectory.npz"), **arrays)
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    if args.record_video:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.video_fps),
            "-i", str(frame_dir / "%06d.png"), "-c:v", "mpeg4", "-q:v", "4",
            "-pix_fmt", "yuv420p",
            str(output / "rollout.mp4"),
        ]
        subprocess.run(command, check=True)
        metrics["video"] = str(output / "rollout.mp4")
        metrics["video_frames"] = frame_index
        with (output / "metrics.json").open("w") as stream:
            json.dump(metrics, stream, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
