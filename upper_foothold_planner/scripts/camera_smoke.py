#!/usr/bin/env python3
"""Render one attached depth frame and an external audit image in Isaac Gym."""

import argparse
import json
from pathlib import Path

from isaacgym import gymapi  # must precede torch
import numpy as np
from PIL import Image

from upper_planner.contracts import preprocess_isaac_depth


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "camera_smoke")
    parser.add_argument("--pitch-deg", type=float, help="diagnostic override of base-to-camera pitch")
    return parser.parse_args()


def save_rgba(path, pixels, height, width):
    rgba = np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)
    Image.fromarray(rgba, "RGBA").convert("RGB").save(path)


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config" / "default.json").read_text())
    lower_cfg = json.loads((ROOT / "checkpoints" / "lower_training_config.json").read_text())

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 0.001
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("failed to create Isaac Gym simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    options = gymapi.AssetOptions()
    options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
    options.collapse_fixed_joints = True
    options.fix_base_link = True
    robot_path = ROOT / "assets" / "SF_TRON1A" / "urdf" / "robot.urdf"
    robot_asset = gym.load_asset(sim, str(robot_path.parent), robot_path.name, options)
    box_asset = gym.create_box(sim, 0.24, 0.35, 0.20, gymapi.AssetOptions())

    env = gym.create_env(
        sim, gymapi.Vec3(-3.0, -3.0, 0.0), gymapi.Vec3(3.0, 3.0, 3.0), 1)
    pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, 0.75))
    robot = gym.create_actor(env, robot_asset, pose, "SF_TRON1A", 0, 0)
    dof_names = gym.get_asset_dof_names(robot_asset)
    dof_state = np.zeros(len(dof_names), dtype=gymapi.DofState)
    for index, name in enumerate(dof_names):
        dof_state["pos"][index] = lower_cfg["init"]["reset_joint_angles"][name]
    gym.set_actor_dof_states(env, robot, dof_state, gymapi.STATE_ALL)

    for index, (x, y) in enumerate(((0.8, -0.35), (1.15, 0.25), (1.55, -0.05))):
        obstacle_pose = gymapi.Transform(p=gymapi.Vec3(x, y, 0.10))
        obstacle = gym.create_actor(env, box_asset, obstacle_pose, f"obstacle_{index}", index + 1, 0)
        gym.set_rigid_body_color(env, obstacle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                 gymapi.Vec3(0.75, 0.12, 0.08))

    depth_cfg = cfg["depth"]
    sensor_props = gymapi.CameraProperties()
    sensor_props.width = int(depth_cfg["width"])
    sensor_props.height = int(depth_cfg["height"])
    sensor_props.horizontal_fov = float(depth_cfg["horizontal_fov_deg"])
    sensor = gym.create_camera_sensor(env, sensor_props)
    base = gym.find_actor_rigid_body_handle(env, robot, "base_Link")
    xyz = depth_cfg["mount_xyz_base_m"]
    roll, pitch, yaw = depth_cfg["mount_rpy_base_rad"]
    if args.pitch_deg is not None:
        pitch = np.deg2rad(args.pitch_deg)
    mount = gymapi.Transform(
        p=gymapi.Vec3(*xyz),
        r=gymapi.Quat.from_euler_zyx(yaw, pitch, roll),
    )
    gym.attach_camera_to_body(sensor, env, base, mount, gymapi.FOLLOW_TRANSFORM)

    overview_props = gymapi.CameraProperties()
    overview_props.width, overview_props.height = 960, 720
    overview = gym.create_camera_sensor(env, overview_props)
    gym.set_camera_location(
        overview, env, gymapi.Vec3(-1.7, 1.5, 1.1), gymapi.Vec3(0.6, 0.0, 0.55))

    gym.prepare_sim(sim)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    raw = np.asarray(gym.get_camera_image(sim, env, sensor, gymapi.IMAGE_DEPTH), dtype=np.float32)
    raw = raw.reshape(sensor_props.height, sensor_props.width)
    proximity = preprocess_isaac_depth(raw, depth_cfg["near_m"], depth_cfg["far_m"])
    Image.fromarray(np.uint8(np.clip(proximity, 0.0, 1.0) * 255), "L").save(
        args.output / "depth_proximity.png")
    np.save(args.output / "depth_raw.npy", raw)
    save_rgba(
        args.output / "camera_color.png",
        gym.get_camera_image(sim, env, sensor, gymapi.IMAGE_COLOR),
        sensor_props.height, sensor_props.width)
    save_rgba(
        args.output / "overview.png",
        gym.get_camera_image(sim, env, overview, gymapi.IMAGE_COLOR),
        overview_props.height, overview_props.width)

    finite_distance = -raw[np.isfinite(raw)]
    metadata = {
        "camera_mount_source": "commented d435_joint in copied SF URDF",
        "mount_xyz_base_m": xyz,
        "mount_rpy_base_rad": [roll, pitch, yaw],
        "pitch_override_deg": args.pitch_deg,
        "horizontal_fov_deg": sensor_props.horizontal_fov,
        "resolution": [sensor_props.height, sensor_props.width],
        "raw_depth_finite_fraction": float(np.isfinite(raw).mean()),
        "finite_distance_min_m": float(finite_distance.min()) if finite_distance.size else None,
        "finite_distance_max_m": float(finite_distance.max()) if finite_distance.size else None,
        "proximity_mean": float(proximity.mean()),
        "note": "orientation is an unverified URDF mount assumption; inspect both saved images",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
