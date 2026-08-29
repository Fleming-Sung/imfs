import json
import unittest
from pathlib import Path

import numpy as np

from deploy.course import generate_layout


ROOT = Path(__file__).resolve().parents[1]


class CourseTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "config" / "plum_piles.json").read_text())
        self.initial_xy = np.array([[0.003, 0.105], [0.003, -0.105]])

    def test_layout_is_deterministic_alternating_and_nonoverlapping(self):
        first = generate_layout(self.cfg, self.initial_xy)
        second = generate_layout(self.cfg, self.initial_xy)
        np.testing.assert_allclose(
            np.stack([p.position for p in first.footholds]),
            np.stack([p.position for p in second.footholds]))
        self.assertEqual(
            [p.foot for p in first.footholds[:8]], [0, 1, 1, 0, 1, 0, 1, 0])
        self.assertEqual([p.support for p in first.footholds[:8]], ["platform"] * 8)
        self.assertEqual(
            [p.support for p in first.footholds[:2 + self.cfg["warmup_steps"]]],
            ["platform"] * (2 + self.cfg["warmup_steps"]))
        self.assertTrue(all(
            p.position[0] < first.platform_end_x
            for p in first.footholds if p.support == "platform"))
        self.assertTrue(all(
            p.position[0] - self.cfg["pile_radius"] > first.platform_end_x
            for p in first.footholds if p.support == "pile"))

        piles = [p for p in first.footholds if p.support == "pile"]
        positions = np.stack([p.position[:2] for p in piles])
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        required = 2.0 * self.cfg["pile_radius"] + self.cfg["minimum_pile_clearance"]
        self.assertGreaterEqual(float(distances.min()) + 1e-9, required)

    def test_height_is_fixed_within_layout_and_configurable_between_runs(self):
        low = generate_layout(self.cfg, self.initial_xy)
        high_cfg = dict(self.cfg, support_height=0.40)
        high = generate_layout(high_cfg, self.initial_xy)
        np.testing.assert_allclose(
            [p.position[2] for p in low.footholds], self.cfg["support_height"])
        np.testing.assert_allclose([p.position[2] for p in high.footholds], 0.40)

    def test_flat_layout_has_only_flat_targets(self):
        cfg = json.loads((ROOT / "config" / "flat.json").read_text())
        layout = generate_layout(cfg, self.initial_xy)
        self.assertEqual(layout.scenario, "flat")
        self.assertEqual(len(layout.footholds), cfg["num_footholds"])
        self.assertTrue(all(point.support == "flat" for point in layout.footholds))

    def test_regional_pile_experiment_has_clean_transition_geometry(self):
        cfg = json.loads((ROOT / "config" / "experimental_regional_piles.json").read_text())
        layout = generate_layout(cfg, self.initial_xy)
        self.assertEqual([point.foot for point in layout.footholds[2:6]], [0, 1, 0, 1])
        self.assertEqual(
            [point.support for point in layout.footholds[10:14]],
            ["transition", "transition", "pile", "pile"])
        physical_piles = [point for point in layout.footholds if point.support == "pile"]
        self.assertTrue(all(
            point.position[0] - cfg["pile_radius"] >= layout.platform_end_x
            + cfg["pile_entry_gap"][0] - 1e-9
            for point in physical_piles))


if __name__ == "__main__":
    unittest.main()
