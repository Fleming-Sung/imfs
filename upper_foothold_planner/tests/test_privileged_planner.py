import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from upper_planner.contracts import FootholdActionBounds
from upper_planner.privileged_planner import (
    PrivilegedTerrainPlanner, jump_geodesic_distance,
    obstacle_mask, sole_support_fraction)
from upper_planner.terrain import (
    TerrainSpec, build_tiled_heightfield, generate_support_layout)


ROOT = Path(__file__).resolve().parents[1]


class PrivilegedPlannerTest(unittest.TestCase):
    def setUp(self):
        cfg = json.loads((ROOT / "config" / "default.json").read_text())
        self.bounds = FootholdActionBounds.from_config(
            cfg["action_cartesian_course"])

    def test_cartesian_candidates_have_no_physical_aliases(self):
        tiled = build_tiled_heightfield([
            TerrainSpec(kind="straight", length_m=3.5, width_m=3.0, seed=3)])
        planner = PrivilegedTerrainPlanner(tiled, self.bounds, "cpu")
        for foot in (0, 1):
            swing = torch.full(
                (len(planner.candidates),), foot, dtype=torch.long)
            decoded = self.bounds.decode(planner.candidates, swing)
            unique = torch.unique(torch.round(decoded * 1.0e6), dim=0)
            self.assertEqual(len(unique), len(planner.candidates))
            self.assertGreater(float(decoded[:, 1].abs().std()), 0.02)

    def test_sole_fraction_matches_nine_point_definition(self):
        support = np.ones((9, 11), dtype=bool)
        fraction = sole_support_fraction(support, 0.05)
        self.assertEqual(float(fraction[4, 5]), 1.0)
        self.assertLess(float(fraction[0, 0]), 1.0)
        support[4, 5] = False
        fraction = sole_support_fraction(support, 0.05)
        self.assertLess(float(fraction[4, 5]), 1.0)

    def test_jump_geodesic_bridges_gap_but_not_obstacle(self):
        landing = np.ones((7, 13), dtype=bool)
        landing[:, 6] = False
        blocked = np.zeros_like(landing)
        bridged = jump_geodesic_distance(
            landing, blocked, (3, 11), 0.05, 0.16)
        self.assertTrue(np.isfinite(bridged[3, 1]))

        blocked[:, 6] = True
        blocked[-1, 6] = False
        detour = jump_geodesic_distance(
            landing, blocked, (3, 11), 0.05, 0.16)
        self.assertTrue(np.isfinite(detour[3, 1]))
        self.assertGreater(float(detour[3, 1]), float(bridged[3, 1]))

    def test_obstacles_are_inflated(self):
        layout = generate_support_layout(TerrainSpec(
            kind="hurdles", length_m=3.5, width_m=3.0, seed=5))
        raw = obstacle_mask(layout, 0.0)
        inflated = obstacle_mask(layout, 0.06)
        self.assertGreater(int(inflated.sum()), int(raw.sum()))

    def test_planner_selects_supported_forward_candidate(self):
        tiled = build_tiled_heightfield([
            TerrainSpec(kind="straight", length_m=3.5, width_m=3.0,
                        corridor_width_m=1.0, seed=7)])
        planner = PrivilegedTerrainPlanner(tiled, self.bounds, "cpu")
        # Identity xyzw foot orientations.  Left foot is the next swing foot,
        # so the right foot at y=-0.10 is the stance origin.
        foot_positions = torch.tensor([[[0.20, 0.10, 0.0],
                                        [0.20, -0.10, 0.0]]])
        rigid = torch.zeros(1, 2, 13)
        rigid[..., 6] = 1.0
        env = SimpleNamespace(
            sampler=SimpleNamespace(swing_foot=torch.tensor([0])),
            foot_positions=foot_positions,
            rigid_body_states=rigid,
            feet_indices=torch.tensor([0, 1]),
        )
        action, diagnostics = planner.plan(
            env, torch.tensor([0]), torch.zeros(1, 3))
        decoded = self.bounds.decode(action, torch.tensor([0]))
        self.assertFalse(bool(diagnostics["fallback"][0]))
        self.assertGreaterEqual(float(
            diagnostics["chosen_support_fraction"][0]), 5.0 / 9.0)
        self.assertGreater(float(decoded[0, 0]), 0.08)
        self.assertTrue(math.isfinite(float(
            diagnostics["chosen_geodesic_progress_m"][0])))


if __name__ == "__main__":
    unittest.main()
