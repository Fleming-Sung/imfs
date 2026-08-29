#!/usr/bin/env python3
"""Run model_7000 in the flat or platform-to-plum-pile MuJoCo scenario."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from deploy.simulator import Deployment


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario", choices=("training_demo", "plum_piles", "flat"),
        default="plum_piles")
    parser.add_argument("--config", type=Path, help="custom JSON; overrides --scenario config selection")
    parser.add_argument("--seed", type=int, help="override the fixed layout seed")
    parser.add_argument("--duration", type=float, help="override rollout duration in seconds")
    parser.add_argument(
        "--step-distance", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="training_demo step-distance range in metres")
    parser.add_argument(
        "--step-angle-deg", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="training_demo per-step angle offset around the macro direction")
    parser.add_argument(
        "--movement-direction-deg", type=float, nargs=2, metavar=("MIN", "MAX"),
        help="training_demo episode-level macro direction range in world degrees")
    parser.add_argument(
        "--lateral-separation", type=float,
        help="training_demo minimum left-right target separation in the stance-foot frame")
    parser.add_argument("--headless", action="store_true", help="do not open the interactive MuJoCo viewer")
    parser.add_argument("--realtime", action="store_true", help="pace simulation at wall-clock speed")
    parser.add_argument("--record-video", action="store_true", help="save a full-duration MP4")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or ROOT / "config" / f"{args.scenario}.json"
    cfg = json.loads(config_path.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.duration is not None:
        cfg["duration_s"] = args.duration
    for name, value in (("step_distance", args.step_distance),
                        ("step_angle_deg", args.step_angle_deg),
                        ("movement_direction_deg", args.movement_direction_deg)):
        if value is not None:
            if cfg.get("scenario") != "training_demo":
                raise ValueError(f"--{name.replace('_', '-')} is only valid for training_demo")
            cfg[name] = list(value)
    if args.lateral_separation is not None:
        if cfg.get("scenario") != "training_demo":
            raise ValueError("--lateral-separation is only valid for training_demo")
        cfg["minimum_lateral_separation"] = args.lateral_separation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario = cfg.get("scenario", args.scenario)
    output = args.output_dir or ROOT / "outputs" / f"{timestamp}_{scenario}_seed{int(cfg['seed']):03d}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(cfg, indent=2))
    deployment = Deployment(ROOT, cfg, output)
    video = output / "rollout.mp4" if args.record_video else None
    deployment.run(
        cfg["duration_s"], interactive=not args.headless, realtime=args.realtime,
        video_path=video, camera_cfg=cfg)


if __name__ == "__main__":
    main()
