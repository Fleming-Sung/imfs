#!/usr/bin/env python3
"""Resumable data -> H1 -> H3 -> closed-loop -> video experiment pipeline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(run_id, name, command, expected, manifest):
    if expected.exists():
        manifest[name] = {"status": "reused", "artifact": str(expected)}
        return
    log = ROOT / "experiments" / "pipeline_logs" / f"{run_id}_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line); stream.flush()
        code = process.wait()
    manifest[name] = {"status": "complete" if code == 0 else "failed",
                      "artifact": str(expected), "log": str(log),
                      "wall_seconds": time.time() - started, "command": command}
    if code or not expected.exists():
        raise SystemExit(f"stage {name} failed; inspect {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cgowm_auto")
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--video_seeds", type=int, nargs="*", default=[924, 42, 31])
    parser.add_argument("--no_videos", action="store_true")
    args = parser.parse_args()
    py = sys.executable
    full = args.profile == "full"
    envs, ticks = (128, 1500) if full else (16, 300)
    h1_updates, h3_updates = (1200, 800) if full else (20, 20)
    eval_envs, eval_ticks = (64, 1500) if full else (4, 300)
    dataset = ROOT / "experiments" / f"{args.name}_dataset"
    h1 = ROOT / "runs" / f"{args.name}_h1"
    h3 = ROOT / "runs" / f"{args.name}_h3"
    manifest_path = ROOT / "experiments" / f"{args.name}_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stages = [
        ("collect", [py, "scripts/collect_random_options.py", "--num_envs", str(envs),
          "--lower_ticks", str(ticks), "--seed", str(args.seed), "--output", str(dataset)],
         dataset / "transitions.npz"),
        ("train_h1", [py, "scripts/train_h1.py", "--dataset", str(dataset / "transitions.npz"),
          "--output", str(h1), "--updates", str(h1_updates), "--seed", str(args.seed)],
         h1 / "model_best.pt"),
        ("train_h3", [py, "scripts/train_h3.py", "--dataset", str(dataset / "transitions.npz"),
          "--init", str(h1 / "model_best.pt"), "--output", str(h3),
          "--updates", str(h3_updates), "--seed", str(args.seed)], h3 / "model_best.pt")]
    for name, command, expected in stages:
        run(args.name, name, command, expected, manifest); manifest_path.write_text(json.dumps(manifest, indent=2))
    for mode in ("beam", "model_score", "prior"):
        output = ROOT / "experiments" / f"{args.name}_{mode}_seed924"
        command = [py, "scripts/evaluate_h1.py", "--checkpoint", str(h3 / "model_best.pt"),
                   "--output", str(output), "--mode", mode, "--num_envs", str(eval_envs),
                   "--lower_ticks", str(eval_ticks), "--seed", "924", "--headless"]
        run(args.name, f"eval_{mode}", command, output / "metrics.json", manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if not args.no_videos:
        for seed in args.video_seeds:
            output = ROOT / "experiments" / "videos" / f"{args.name}_beam_seed{seed}"
            command = [py, "scripts/evaluate_h1.py", "--checkpoint", str(h3 / "model_best.pt"),
                       "--output", str(output), "--mode", "beam", "--num_envs", "1",
                       "--lower_ticks", "750", "--seed", str(seed), "--record_video"]
            run(args.name, f"video_seed{seed}", command, output / "rollout.mp4", manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2))
    subprocess.run([py, "scripts/summarize_results.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
