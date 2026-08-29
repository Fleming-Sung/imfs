"""Run a fixed multi-seed closed-loop suite without touching training state."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[805, 905, 1005])
    parser.add_argument("--num_envs", type=int, default=48)
    parser.add_argument("--lower_ticks", type=int, default=1200)
    parser.add_argument("--terminal_value_coef", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_one(label, checkpoint, seed, args):
    output = args.output / label / "seed_{:04d}".format(seed)
    command = [
        sys.executable, str(ROOT / "scripts" / "upper_rollout_smoke.py"),
        "--headless", "--physx", "--sim_device", "cuda:0",
        "--pipeline", "gpu", "--num_envs", str(args.num_envs),
        "--seed", str(seed), "--lower_ticks", str(args.lower_ticks),
        "--course_length_m", "3.5", "--terrain_curriculum", "research",
        "--action_profile", "polar_course", "--output", str(output),
    ]
    if checkpoint is None:
        command.append("--neutral_actions")
    else:
        command += [
            "--planner_checkpoint", str(checkpoint),
            "--planning_horizon", "5", "--cem_candidates", "128",
            "--cem_elites", "16", "--cem_iterations", "3",
            "--collision_risk", "0.2", "--fall_risk", "1.0",
            "--collision_force_risk", "0.25", "--stability_risk", "0.5",
            "--support_risk", "0.5", "--touchdown_risk", "0.15",
            "--action_l2", "0.1", "--decomposed_reward",
            "--terminal_value_coef", str(args.terminal_value_coef),
        ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((output / "metrics.json").read_text())


def aggregate(items):
    transitions = sum(item["macro_transitions"] for item in items)
    return {
        "runs": len(items),
        "macro_transitions": transitions,
        "event_rate": {
            name: sum(item[name] for item in items) / max(transitions, 1)
            for name in ("falls", "collisions", "off_support", "successes")
        },
        "event_count": {
            name: sum(item[name] for item in items)
            for name in ("falls", "collisions", "off_support", "successes")
        },
        "base_forward_progress_mean_m": sum(
            item["base_forward_progress_mean_m"] for item in items) / len(items),
        "distance_to_goal_mean_m": sum(
            item["distance_to_goal_mean_m"] for item in items) / len(items),
        "terrain": {
            kind: {
                metric: sum(item["terrain_kind_metrics"][kind][metric]
                            for item in items)
                for metric in ("transitions", "success", "fall", "collision",
                               "off_support", "progress_m")
            }
            for kind in items[0]["terrain_kind_metrics"]
        },
    }


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    systems = [("neutral", None)] + [
        (Path(checkpoint).stem, Path(checkpoint).resolve())
        for checkpoint in args.checkpoints]
    summary = {"configuration": vars(args), "systems": {}}
    for label, checkpoint in systems:
        results = [run_one(label, checkpoint, seed, args) for seed in args.seeds]
        summary["systems"][label] = aggregate(results)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
