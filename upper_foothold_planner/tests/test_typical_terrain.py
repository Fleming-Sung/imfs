import numpy as np
import unittest

from upper_planner.terrain import (TerrainSpec, build_static_boxes,
                                   generate_support_layout)


def _layout(kind, **kwargs):
    return generate_support_layout(TerrainSpec(
        kind=kind, length_m=4.5, width_m=3.0, resolution_m=0.025,
        seed=17, **kwargs))


def _supported(layout, xy):
    ix = int(np.argmin(np.abs(layout.x_m - xy[0])))
    iy = int(np.argmin(np.abs(layout.y_m - xy[1])))
    return bool(layout.support_mask[iy, ix])


class TypicalTerrainTest(unittest.TestCase):
    def test_narrow_bridge_has_supported_endpoints_and_real_side_drop(self):
        layout = _layout("narrow_bridge", corridor_width_m=0.60)
        self.assertTrue(_supported(layout, layout.start_xy))
        self.assertTrue(_supported(layout, layout.goal_xy))
        self.assertTrue(_supported(layout, (2.25, 0.25)))
        self.assertFalse(_supported(layout, (2.25, 0.40)))
        self.assertEqual(layout.obstacle_rectangles, ())

    def test_hurdles_have_full_ground_and_low_fences(self):
        layout = _layout("hurdles", corridor_width_m=1.60)
        self.assertTrue(layout.support_mask.all(axis=1).any())
        self.assertTrue(_supported(layout, layout.start_xy))
        self.assertTrue(_supported(layout, layout.goal_xy))
        self.assertEqual(len(layout.obstacle_rectangles), 3)
        for _, _, thickness, width, height in layout.obstacle_rectangles:
            self.assertTrue(np.isclose(thickness, 0.055))
            self.assertTrue(np.isclose(width, 1.60))
            self.assertTrue(0.035 <= height <= 0.085)

    def test_irregular_support_is_not_full_or_straight(self):
        layout = _layout("irregular_support", corridor_width_m=0.65)
        self.assertTrue(_supported(layout, layout.start_xy))
        self.assertTrue(_supported(layout, layout.goal_xy))
        self.assertTrue(layout.support_mask.any() and (~layout.support_mask).any())
        centers_y = np.asarray([r[1] for r in layout.support_rectangles[1:-1]])
        widths_y = np.asarray([r[3] for r in layout.support_rectangles[1:-1]])
        self.assertGreater(np.ptp(centers_y), 0.10)
        self.assertGreater(np.ptp(widths_y), 0.05)

    def test_static_boxes_have_exact_zero_top_or_bottom(self):
        layouts = (
            _layout("narrow_bridge", corridor_width_m=0.60),
            _layout("hurdles", corridor_width_m=1.60),
            _layout("irregular_support", corridor_width_m=0.65),
        )
        boxes = build_static_boxes(layouts)
        self.assertEqual(boxes.centers_xyz_m.shape[:2], boxes.active.shape)
        self.assertEqual(boxes.active.sum(axis=1).tolist(), [3, 4, 9])
        for env_id in range(len(layouts)):
            active = boxes.active[env_id]
            centers_z = boxes.centers_xyz_m[env_id, active, 2]
            sizes_z = boxes.sizes_xyz_m[env_id, active, 2]
            obstacles = boxes.obstacle[env_id, active]
            np.testing.assert_allclose(
                centers_z[~obstacles] + 0.5 * sizes_z[~obstacles], 0.0, atol=1e-7)
            np.testing.assert_allclose(
                centers_z[obstacles] - 0.5 * sizes_z[obstacles], 0.0, atol=1e-7)

    def test_random_composite_is_deterministic_varied_and_supported(self):
        layouts = [generate_support_layout(TerrainSpec(
            kind="random_composite", length_m=2.5, seed=seed,
            hurdle_height_min_m=0.025, hurdle_height_max_m=0.05))
            for seed in range(8)]
        repeated = generate_support_layout(TerrainSpec(
            kind="random_composite", length_m=2.5, seed=0,
            hurdle_height_min_m=0.025, hurdle_height_max_m=0.05))
        np.testing.assert_array_equal(layouts[0].support_mask, repeated.support_mask)
        self.assertGreater(len({layout.support_mask.tobytes() for layout in layouts}), 5)
        self.assertGreater(sum(len(layout.obstacle_rectangles) for layout in layouts), 0)
        for layout in layouts:
            self.assertTrue(_supported(layout, layout.start_xy))
            self.assertTrue(_supported(layout, layout.goal_xy))
            self.assertTrue(layout.support_mask.any())
            self.assertTrue((~layout.support_mask).any())

    def test_research_families_are_varied_and_have_supported_goals(self):
        for kind in ("edge_cases", "stepping_stones", "turns", "household"):
            layout = generate_support_layout(TerrainSpec(
                kind=kind, length_m=3.5, width_m=3.2, seed=23))
            repeated = generate_support_layout(TerrainSpec(
                kind=kind, length_m=3.5, width_m=3.2, seed=23))
            np.testing.assert_array_equal(layout.support_mask, repeated.support_mask)
            self.assertEqual(layout.obstacle_rectangles, repeated.obstacle_rectangles)
            self.assertTrue(_supported(layout, layout.start_xy), kind)
            self.assertTrue(_supported(layout, layout.goal_xy), kind)
        household = generate_support_layout(TerrainSpec(
            kind="household", length_m=3.5, width_m=3.2, seed=23))
        self.assertGreater(len(household.obstacle_rectangles), 1)


if __name__ == "__main__":
    unittest.main()
