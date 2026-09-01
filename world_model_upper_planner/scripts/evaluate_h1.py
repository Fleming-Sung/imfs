#!/usr/bin/env python3
"""Closed-loop evaluation and video recording for CG-OWM planners."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaacgym import gymapi  # must precede torch
import torch

from adapters.frozen_lower_env.factory import create_upper_system
from adapters.frozen_lower_env.rollout import UpperRollout
from cgowm import (CandidateGroundedWorldModel, ModelConfig, PlannerConfig,
                   VectorizedBeamPlanner)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("model_score", "prior", "beam"),
                        default="model_score")
    parser.add_argument("--planning_horizon", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=16)
    parser.add_argument("--proposals_per_beam", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--lower_ticks", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument(
        "--terrain_curriculum",
        choices=("research", "typical", "obstacles", "randomized"),
        default="research")
    parser.add_argument(
        "--research_kind",
        choices=("mixed", "edge_cases", "random_composite",
                 "stepping_stones", "turns", "household"),
        default="random_composite")
    parser.add_argument(
        "--typical_kind",
        choices=("mixed", "narrow_bridge", "irregular_support", "hurdles"),
        default="mixed")
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
    parser.add_argument("--depth_ablation", choices=("normal", "shuffled", "zero"),
                        default="normal")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=25)
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--physx", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = arguments()
    if args.record_video and args.headless:
        raise ValueError("record_video requires a viewer; omit --headless")
    args.action_profile = "cartesian_course"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.sim_device)
    config = ModelConfig(**checkpoint["model_config"])
    model = CandidateGroundedWorldModel(checkpoint["candidates"], config).to(
        args.sim_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    planner = VectorizedBeamPlanner(model, PlannerConfig(
        horizon=args.planning_horizon, beam_width=args.beam_width,
        proposals_per_beam=args.proposals_per_beam,
        support_weight=3.0, terminal_value_weight=0.1))
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed,
        corridor_width_m=args.corridor_width_m, randomization=False,
        cameras=True, flat_plane=False, obstacles=False,
        course_length_m=args.course_length_m)
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    args.output.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output / "frames"
    if args.record_video:
        frame_dir.mkdir(exist_ok=True)
    frame_stride = max(1, int(round(1.0 / (env.dt * args.video_fps))))
    frame_index = 0
    selected_prediction = {name: [] for name in (
        "progress", "support", "fall", "collision", "uncertainty", "index")}

    @torch.no_grad()
    def choose(depth, proprio, ids):
        if args.depth_ablation == "zero":
            depth = torch.zeros_like(depth)
        elif args.depth_ablation == "shuffled":
            depth = torch.roll(depth, 1, 0)
        latent = model.encode(depth, proprio)
        prediction = model.predict_candidates(latent)
        if args.mode == "prior":
            index = prediction["policy_logits"].argmax(-1)
        elif args.mode == "beam":
            index, _ = planner.plan(latent)
        else:
            q = prediction["q"]
            uncertainty = q.std(0, unbiased=False)
            score = (
                10.0 * prediction["progress"]
                - 5.0 * torch.sigmoid(prediction["fall_logit"])
                - 2.0 * torch.sigmoid(prediction["collision_logit"])
                - 3.0 * (1.0 - prediction["support"])
                - 0.5 * uncertainty
                + 0.1 * q.min(0).values)
            index = score.argmax(-1)
        row = torch.arange(len(index), device=index.device)
        selected_prediction["progress"].append(
            prediction["progress"][row, index].cpu().numpy())
        selected_prediction["support"].append(
            prediction["support"][row, index].cpu().numpy())
        selected_prediction["fall"].append(torch.sigmoid(
            prediction["fall_logit"][row, index]).cpu().numpy())
        selected_prediction["collision"].append(torch.sigmoid(
            prediction["collision_logit"][row, index]).cpu().numpy())
        selected_prediction["uncertainty"].append(
            prediction["q"][:, row, index].std(0, unbiased=False).cpu().numpy())
        selected_prediction["index"].append(index.cpu().numpy())
        return model.candidates[index]

    successes = falls = collisions = off_support = transitions = 0
    actual_progress = actual_support = touchdown = 0.0
    terminal_steps, terminal_envs, terminal_success, terminal_fall = [], [], [], []
    root_trace, foot_trace, target_trace, action_trace = [], [], [], []
    start = time.perf_counter()
    for lower_step in range(args.lower_ticks):
        if args.record_video and lower_step % frame_stride == 0:
            root = env.root_states[0, 0]
            x, y, z, w = root[3:7]
            yaw = torch.atan2(2.0 * (w * z + x * y),
                              1.0 - 2.0 * (y * y + z * z))
            forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
            left = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
            camera = root[:2] - 1.8 * forward + 1.0 * left
            env.gym.viewer_camera_look_at(
                env.viewer, None,
                gymapi.Vec3(float(camera[0]), float(camera[1]),
                            float(root[2] + 0.15)),
                gymapi.Vec3(float(root[0]), float(root[1]), float(root[2])))
        transition = rollout.lower_tick(choose)
        if args.record_video and lower_step % frame_stride == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / f"{frame_index:06d}.png"))
            frame_index += 1
        root_trace.append(env.root_states[0, 0].cpu().numpy().copy())
        foot_trace.append(env.foot_positions[0].cpu().numpy().copy())
        target_trace.append(env.sampler.target_pos[0].cpu().numpy().copy())
        action_trace.append(rollout.previous_action[0].cpu().numpy().copy())
        if transition is None:
            continue
        count = len(transition["ids"])
        transitions += count
        diag = transition["diagnostics"]
        successes += int(diag["success"].sum())
        falls += int(diag["fall"].sum())
        collisions += int(diag["collision"].sum())
        off_support += int(diag["off_support"].sum())
        actual_progress += float(
            transition["terms"]["progress"].sum() / float(cfg["reward"]["progress"]))
        actual_support += float(diag["support_fraction"].sum())
        touchdown += float(diag["touchdown_error_m"].sum())
        done_rows = transition["done"].nonzero(as_tuple=False).flatten()
        for row in done_rows.tolist():
            terminal_steps.append(lower_step)
            terminal_envs.append(int(transition["ids"][row]))
            terminal_success.append(bool(diag["success"][row]))
            terminal_fall.append(bool(diag["fall"][row]))
    wall = time.perf_counter() - start
    episodes = successes + falls
    arrays = {name: np.concatenate(values) if values else np.zeros(0)
              for name, values in selected_prediction.items()}
    metrics = {
        "checkpoint": str(args.checkpoint), "mode": args.mode,
        "terrain_curriculum": args.terrain_curriculum,
        "terrain_kind": (args.typical_kind if args.terrain_curriculum == "typical"
                         else args.research_kind),
        "research_kind": args.research_kind,
        "course_length_m": args.course_length_m,
        "terrain_parameters": {
            "random_width_m": [args.random_width_min_m, args.random_width_max_m],
            "random_gap_m": [args.random_gap_min_m, args.random_gap_max_m],
            "obstacle_probability": args.random_obstacle_probability,
            "bridge_width_m": [args.bridge_width_min_m, args.bridge_width_max_m],
            "irregular_width_m": args.irregular_width_m,
            "hurdle_height_m": [args.hurdle_height_min_m,
                                 args.hurdle_height_max_m],
        },
        "depth_ablation": args.depth_ablation,
        "num_envs": args.num_envs, "lower_ticks": args.lower_ticks,
        "simulated_seconds_per_env": args.lower_ticks * env.dt,
        "transitions": transitions, "episodes": episodes,
        "successes": successes, "falls": falls,
        "success_rate": successes / max(episodes, 1),
        "fall_rate": falls / max(episodes, 1),
        "collisions": collisions, "off_support": off_support,
        "actual_progress_mean_m": actual_progress / max(transitions, 1),
        "actual_support_mean": actual_support / max(transitions, 1),
        "touchdown_error_mean_m": touchdown / max(transitions, 1),
        "predicted": {
            name: float(value.mean()) if len(value) else None
            for name, value in arrays.items() if name != "index"},
        "unique_selected_candidates": int(len(np.unique(arrays["index"]))),
        "wall_seconds": wall,
        "lower_env_steps_per_second": (
            args.lower_ticks * args.num_envs / max(wall, 1e-9)),
    }
    np.savez_compressed(
        args.output / "trajectory_env0.npz",
        root=np.stack(root_trace), foot=np.stack(foot_trace),
        target=np.stack(target_trace), action=np.stack(action_trace), dt_s=env.dt)
    np.savez_compressed(
        args.output / "terminals.npz",
        lower_step=np.asarray(terminal_steps), env_id=np.asarray(terminal_envs),
        success=np.asarray(terminal_success), fall=np.asarray(terminal_fall))
    if args.record_video:
        video = args.output / "rollout.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate",
            str(args.video_fps), "-i", str(frame_dir / "%06d.png"),
            "-c:v", "mpeg4", "-q:v", "4", "-pix_fmt", "yuv420p",
            str(video)], check=True)
        metrics["video"] = str(video)
        metrics["video_frames"] = frame_index
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
