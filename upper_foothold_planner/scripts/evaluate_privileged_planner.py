"""Documented entry point for the sensor-free privileged upper-planner gate."""

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    required = [
        "--privileged_terrain_planner",
        "--action_profile", "cartesian_course",
    ]
    # Preserve all standard evaluator flags while making the two privileged
    # contract flags impossible to forget.
    sys.argv[1:1] = required
    runpy.run_path(
        str(ROOT / "scripts" / "upper_rollout_smoke.py"), run_name="__main__")
