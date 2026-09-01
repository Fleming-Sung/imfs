#!/usr/bin/env python3
"""Create a compact, evidence-linked summary from closed-loop result folders."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--output", type=Path, default=Path("docs/latest_results.md"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.experiments.glob("**/metrics.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "success_rate" not in data:
            continue
        rows.append((path.parent.name, data, path))
    lines = ["# Automatically indexed closed-loop results", "",
             "Generated from preserved `metrics.json` files. Videos and trajectories remain in the same result folders.", "",
             "| run | mode | env x seconds | success | falls | progress/option | touchdown | candidates | evidence |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, d, path in rows:
        seconds = d.get("simulated_seconds_per_env", 0)
        lines.append(
            f"| {name} | {d.get('mode')} | {d.get('num_envs')} x {seconds:.0f} | "
            f"{100*d['success_rate']:.1f}% ({d.get('successes')}/{d.get('episodes')}) | "
            f"{d.get('falls')} | {d.get('actual_progress_mean_m', 0):.3f} m | "
            f"{d.get('touchdown_error_mean_m', 0):.3f} m | "
            f"{d.get('unique_selected_candidates')} | `{path}` |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"indexed {len(rows)} evaluations in {args.output}")


if __name__ == "__main__":
    main()
