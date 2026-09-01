#!/usr/bin/env python3
"""Resumable large-scale CG-OWM collection, training, and evaluation.

The matrix deliberately varies terrain family, procedural seed, course length,
geometry ranges, obstacle density, and reset coverage.  Every stage writes a
manifest before moving on, so a machine restart never discards completed data.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


SCENARIOS = (
    dict(name="research_nominal", curriculum="research", kind="mixed",
         difficulty="nominal", length=4.0, width=(0.55, 1.30), gap=(0.00, 0.12), obstacle=0.45),
    dict(name="research_hard", curriculum="research", kind="mixed",
         difficulty="hard", length=5.0, width=(0.44, 1.05), gap=(0.03, 0.17), obstacle=0.65),
    dict(name="turns_nominal", curriculum="research", kind="turns",
         difficulty="nominal", length=4.0, width=(0.60, 1.10), gap=(0.00, 0.10), obstacle=0.45),
    dict(name="turns_hard", curriculum="research", kind="turns",
         difficulty="hard", length=5.0, width=(0.48, 0.82), gap=(0.04, 0.16), obstacle=0.75),
    dict(name="household_nominal", curriculum="research", kind="household",
         difficulty="nominal", length=4.0, width=(0.55, 1.20), gap=(0.00, 0.10), obstacle=0.50),
    dict(name="household_hard", curriculum="research", kind="household",
         difficulty="hard", length=5.0, width=(0.45, 0.95), gap=(0.02, 0.15), obstacle=0.80),
    dict(name="bridge_nominal", curriculum="typical", kind="narrow_bridge",
         difficulty="nominal", length=4.0, bridge=(0.58, 0.82), irregular=0.72, hurdle=(0.020, 0.050)),
    dict(name="bridge_hard", curriculum="typical", kind="narrow_bridge",
         difficulty="hard", length=5.0, bridge=(0.46, 0.64), irregular=0.55, hurdle=(0.035, 0.075)),
    dict(name="edge_hard", curriculum="research", kind="edge_cases",
         difficulty="hard", length=5.0, width=(0.42, 0.88), gap=(0.04, 0.18), obstacle=0.20),
    dict(name="stones_hard", curriculum="research", kind="stepping_stones",
         difficulty="hard", length=5.0, width=(0.42, 0.72), gap=(0.04, 0.16), obstacle=0.10),
    dict(name="irregular_nominal", curriculum="typical", kind="irregular_support",
         difficulty="nominal", length=4.0, bridge=(0.58, 0.82), irregular=0.72, hurdle=(0.020, 0.050)),
    dict(name="irregular_hard", curriculum="typical", kind="irregular_support",
         difficulty="hard", length=5.0, bridge=(0.46, 0.64), irregular=0.55, hurdle=(0.035, 0.075)),
)


def run_stage(name, command, expected, manifest):
    if expected.exists():
        manifest[name] = {"status": "reused", "artifact": str(expected)}
        return
    log = ROOT / "experiments" / "scale_v4" / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("a") as stream:
        stream.write("\nCOMMAND " + " ".join(map(str, command)) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line); stream.flush()
        code = process.wait()
    manifest[name] = {
        "status": "complete" if code == 0 else "failed",
        "artifact": str(expected), "log": str(log),
        "wall_seconds": time.time() - started, "command": list(map(str, command)),
    }
    if code or not expected.exists():
        raise SystemExit(f"stage {name} failed; inspect {log}")


def scenario_args(spec):
    result = [
        "--terrain_curriculum", spec["curriculum"],
        "--difficulty_tag", spec["difficulty"],
        "--course_length_m", str(spec["length"]),
    ]
    if spec["curriculum"] == "research":
        result += ["--research_kind", spec["kind"]]
    elif spec["curriculum"] == "typical":
        result += ["--typical_kind", spec["kind"]]
    if "width" in spec:
        result += ["--random_width_min_m", str(spec["width"][0]),
                   "--random_width_max_m", str(spec["width"][1])]
    if "gap" in spec:
        result += ["--random_gap_min_m", str(spec["gap"][0]),
                   "--random_gap_max_m", str(spec["gap"][1])]
    if "obstacle" in spec:
        result += ["--random_obstacle_probability", str(spec["obstacle"])]
    if "bridge" in spec:
        result += ["--bridge_width_min_m", str(spec["bridge"][0]),
                   "--bridge_width_max_m", str(spec["bridge"][1])]
    if "irregular" in spec:
        result += ["--irregular_width_m", str(spec["irregular"])]
    if "hurdle" in spec:
        result += ["--hurdle_height_min_m", str(spec["hurdle"][0]),
                   "--hurdle_height_max_m", str(spec["hurdle"][1])]
    if "corridor" in spec:
        result += ["--corridor_width_m", str(spec["corridor"])]
    return result


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def audit_shard(path):
    """Reject capability/interface failures before they contaminate replay."""
    with np.load(path, allow_pickle=False) as data:
        rows = len(data["env_id"])
        valid_rows = float(np.any(data["candidate_valid"], axis=1).mean())
        fall_rate = float(data["fall"].mean())
        progress = float(np.mean(data["progress"]))
        selected = int(len(np.unique(data["candidate_index"])))
        finite = bool(np.isfinite(data["progress"]).all()
                      and np.isfinite(data["candidate_progress"]).all())
    result = {
        "rows": rows, "valid_candidate_row_fraction": valid_rows,
        "fall_transition_fraction": fall_rate, "mean_progress_m": progress,
        "selected_candidate_count": selected, "finite": finite,
    }
    if not finite or valid_rows < 0.10 or fall_rate > 0.50 or selected < 12:
        raise RuntimeError(f"shard failed replay quality gate: {path}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "scale"), default="scale")
    parser.add_argument("--phase", choices=("all", "collect", "train", "eval"), default="all")
    parser.add_argument("--num_envs", type=int,
                        help="override collection parallelism after throughput probing")
    parser.add_argument("--lower_ticks", type=int,
                        help="override lower ticks per collection shard")
    parser.add_argument("--no_videos", action="store_true")
    parser.add_argument("--base_dataset", type=Path,
                        default=ROOT / "experiments/dataset_v2_replay_640env/transitions.npz")
    parser.add_argument("--init", type=Path,
                        default=ROOT / "runs/h3_v2_multiterrain_seed91/model_best.pt")
    args = parser.parse_args()
    py = sys.executable
    envs, ticks = ((16, 180) if args.profile == "smoke" else (512, 3000))
    envs = args.num_envs or envs
    ticks = args.lower_ticks or ticks
    root = ROOT / "experiments" / "scale_v4" / args.profile
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    shards = []

    if args.phase in ("all", "collect"):
        for index, spec in enumerate(SCENARIOS):
            output = root / "shards" / f"{index:02d}_{spec['name']}"
            seed = 2001 + 37 * index
            command = [
                py, "scripts/collect_random_options.py",
                "--num_envs", str(envs), "--lower_ticks", str(ticks),
                "--seed", str(seed), "--behavior", "safe_diverse",
                "--uniform_valid_fraction", "0.35",
                "--unsafe_random_fraction", "0.15",
                "--reset_curriculum_prob", "0.30",
                "--output", str(output), *scenario_args(spec),
            ]
            run_stage(f"collect_{index:02d}_{spec['name']}", command,
                      output / "transitions.npz", manifest)
            manifest[f"audit_{index:02d}_{spec['name']}"] = audit_shard(
                output / "transitions.npz")
            write_manifest(manifest_path, manifest)
            shards.append(output / "transitions.npz")
    else:
        shards = [root / "shards" / f"{i:02d}_{s['name']}" / "transitions.npz"
                  for i, s in enumerate(SCENARIOS)]

    dataset = root / f"dataset_{args.profile}_memmap"
    h1 = ROOT / "runs" / f"scale_v4_{args.profile}_h1"
    h3 = ROOT / "runs" / f"scale_v4_{args.profile}_h3"
    if args.phase in ("all", "collect", "train"):
        run_stage("merge", [py, "scripts/merge_datasets.py", "--memmap",
                  "--inputs", str(args.base_dataset), *map(str, shards),
                  "--output", str(dataset)], dataset / "manifest.json", manifest)
        write_manifest(manifest_path, manifest)
    if args.phase in ("all", "train"):
        h1_updates, h3_updates = ((30, 30) if args.profile == "smoke" else (6000, 5000))
        h1_batch, h3_batch = ((32, 16) if args.profile == "smoke" else (1024, 512))
        run_stage("train_h1", [
            py, "scripts/train_h1.py", "--dataset", str(dataset),
            "--init", str(args.init), "--output", str(h1),
            "--updates", str(h1_updates), "--batch_size", str(h1_batch),
            "--learning_rate", "0.0001", "--validate_every", "250",
            "--validation_batches", "4", "--terrain_validation_batches", "1",
            "--balanced_terrain_sampling", "--seed", "3001",
        ], h1 / "model_best.pt", manifest)
        write_manifest(manifest_path, manifest)
        run_stage("train_h3", [
            py, "scripts/train_h3.py", "--dataset", str(dataset),
            "--init", str(h1 / "model_best.pt"), "--output", str(h3),
            "--updates", str(h3_updates), "--batch_size", str(h3_batch),
            "--validate_every", "250", "--validation_batches", "4",
            "--terrain_validation_batches", "1",
            "--balanced_terrain_sampling", "--seed", "3003",
        ], h3 / "model_best.pt", manifest)
        write_manifest(manifest_path, manifest)

    if args.phase in ("all", "eval"):
        eval_envs, eval_ticks = ((4, 180) if args.profile == "smoke" else (32, 1000))
        for scenario_index, spec in enumerate(SCENARIOS):
            for seed in ((4101,) if args.profile == "smoke" else (4101, 4102, 4103)):
                name = f"eval_{scenario_index:02d}_{spec['name']}_seed{seed}"
                output = root / "eval" / name
                command = [
                    py, "scripts/evaluate_h1.py", "--checkpoint", str(h3 / "model_best.pt"),
                    "--output", str(output), "--mode", "beam",
                    "--num_envs", str(eval_envs), "--lower_ticks", str(eval_ticks),
                    "--seed", str(seed), "--headless", *scenario_args(spec),
                ]
                # difficulty_tag is collector metadata, not an evaluator argument.
                tag = command.index("--difficulty_tag")
                del command[tag:tag + 2]
                run_stage(name, command, output / "metrics.json", manifest)
                write_manifest(manifest_path, manifest)

        if not args.no_videos:
            video_indices = (1, 3, 5, 7, 8, 11)
            for scenario_index in video_indices:
                spec = SCENARIOS[scenario_index]
                name = f"video_{scenario_index:02d}_{spec['name']}_seed5101"
                output = root / "videos" / name
                command = [
                    py, "scripts/evaluate_h1.py", "--checkpoint", str(h3 / "model_best.pt"),
                    "--output", str(output), "--mode", "beam", "--num_envs", "1",
                    "--lower_ticks", "750", "--seed", "5101", "--record_video",
                    *scenario_args(spec),
                ]
                tag = command.index("--difficulty_tag")
                del command[tag:tag + 2]
                run_stage(name, command, output / "rollout.mp4", manifest)
                write_manifest(manifest_path, manifest)

    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
