"""Record a rollout video of the trained candidate Actor (Gate E PPO).

Runs the frozen lower policy with the PPO actor on a single environment with
the viewer enabled, writes per-frame PNGs with a camera that follows the
robot, and stitches them into rollout.mp4 with ffmpeg.
"""

import argparse
import subprocess
import sys
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
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--lower_ticks", type=int, default=1500)
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--course_length_m", type=float, default=3.5)
    parser.add_argument("--terrain_curriculum", type=str, default="research")
    parser.add_argument("--research_kind", type=str, default="random_composite")
    parser.add_argument("--random_width_min_m", type=float, default=0.50)
    parser.add_argument("--random_width_max_m", type=float, default=1.30)
    parser.add_argument("--random_gap_max_m", type=float, default=0.14)
    parser.add_argument("--random_obstacle_probability", type=float, default=0.0)
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
    args.headless = False  # video requires the viewer
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    num_candidates = int(checkpoint["num_candidates"])
    candidates_levels = checkpoint["candidates_levels"].to(torch.float32)
    feature_dim = int(checkpoint["feature_dim"])
    gru_hidden = int(checkpoint["gru_hidden"])
    actor = CandidateActor(num_candidates, feature_dim=feature_dim,
                           gru_hidden=gru_hidden)
    state_dict = checkpoint.get("actor", checkpoint.get("state_dict"))
    actor.load_state_dict(state_dict)
    actor.eval()

    env, policy, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, 1, args.seed, corridor_width_m=1.5,
        randomization=False, cameras=True,
        flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(
        env, policy, interface, task, cfg["depth"], capture_depth=True)
    device = env.device
    actor = actor.to(device)
    candidates_levels = candidates_levels.to(device)

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        indices, _ = actor.select(depth, proprio)
        return candidates_levels[indices]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "frames"
    frame_dir.mkdir(exist_ok=True)
    frame_stride = max(1, int(round(1.0 / (env.dt * args.video_fps))))
    frame_index = 0

    if env.viewer is not None:
        env.gym.viewer_camera_look_at(
            env.viewer, None, gymapi.Vec3(-1.5, 1.5, 1.0), gymapi.Vec3(0.8, 0.0, 0.2))

    for lower_step in range(args.lower_ticks):
        if lower_step % frame_stride == 0:
            root = env.root_states[0, 0]
            x, y, z, w = root[3:7]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
            left = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
            xy = root[:2] - 1.72 * forward + 1.02 * left
            env.gym.viewer_camera_look_at(
                env.viewer, None,
                gymapi.Vec3(float(xy[0]), float(xy[1]), float(root[2] + 0.05)),
                gymapi.Vec3(float(root[0]), float(root[1]), float(root[2])))
        rollout.lower_tick(choose_actions)
        if lower_step % frame_stride == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / "{:06d}.png".format(frame_index)))
            frame_index += 1

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.video_fps),
        "-i", str(frame_dir / "%06d.png"), "-c:v", "mpeg4", "-q:v", "4",
        "-pix_fmt", "yuv420p", str(output / "rollout.mp4")], check=True)
    print(f"saved {output / 'rollout.mp4'} ({frame_index} frames, "
          f"{frame_index / args.video_fps:.1f}s)")


if __name__ == "__main__":
    main()
