import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from deploy.course import build_training_demo_scene
from deploy.training_sampler import TrainingGoalSampler, quaternion_yaw


ROOT = Path(__file__).resolve().parents[1]


class TrainingSamplerTest(unittest.TestCase):
    def _make(self, directory, seed=7):
        scene = Path(directory) / "scene.xml"
        layout = build_training_demo_scene(
            ROOT / "assets" / "SF_TRON1A" / "xml" / "robot_deploy.xml", scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cfg = json.loads((ROOT / "checkpoints" / "training_config.json").read_text())
        demo_cfg = json.loads((ROOT / "config" / "training_demo.json").read_text())
        sampler = TrainingGoalSampler(model, data, cfg, demo_cfg, seed)
        sampler.reset()
        return layout, model, data, sampler

    def test_flat_scene_and_sampler_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self._make(first_dir)
            second = self._make(second_dir)
            self.assertEqual(first[0].scenario, "training_demo")
            self.assertEqual(first[0].support_height, 0.0)
            for _ in range(100):
                switched_a = first[3].advance(0.02)
                switched_b = second[3].advance(0.02)
                self.assertEqual(switched_a, switched_b)
                np.testing.assert_allclose(first[3].target_positions(), second[3].target_positions())
                np.testing.assert_allclose(
                    first[3].target_quaternions(), second[3].target_quaternions())

    def test_sampled_targets_obey_training_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, data, sampler = self._make(directory)
            for _ in range(500):
                if not sampler.advance(0.02):
                    continue
                swing = sampler.swing_foot
                stance = 1 - swing
                stance_pos = data.site_xpos[sampler.foot_site_ids[stance]]
                distance = np.linalg.norm(sampler.target_pos[swing, :2] - stance_pos[:2])
                separation = sampler.effective_settings()["minimum_lateral_separation"]
                self.assertLessEqual(distance, np.hypot(0.35, separation) + 1e-9)
                stance_yaw = quaternion_yaw(data.xquat[sampler.foot_body_ids[stance]])
                delta = sampler.target_pos[swing] - stance_pos
                local_y = -np.sin(stance_yaw) * delta[0] + np.cos(stance_yaw) * delta[1]
                if swing == 0:
                    self.assertGreaterEqual(local_y, separation - 1e-9)
                else:
                    self.assertLessEqual(local_y, -separation + 1e-9)
                self.assertAlmostEqual(sampler.target_pos[swing, 2], 0.0)
                relative_yaw = quaternion_yaw(sampler.target_quat[swing]) - quaternion_yaw(
                    data.xquat[sampler.foot_body_ids[stance]])
                self.assertLessEqual(abs(relative_yaw), np.pi / 2 + 1e-9)

    def test_demo_ranges_override_checkpoint_sampling_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            build_training_demo_scene(
                ROOT / "assets" / "SF_TRON1A" / "xml" / "robot_deploy.xml", scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            cfg = json.loads((ROOT / "checkpoints" / "training_config.json").read_text())
            demo = {"step_distance": [0.08, 0.12], "step_angle_deg": [-5.0, 5.0],
                    "movement_direction_deg": [0.0, 0.0],
                    "minimum_lateral_separation": 0.20}
            sampler = TrainingGoalSampler(model, data, cfg, demo, 11)
            sampler.reset()
            settings = sampler.effective_settings()
            self.assertEqual(settings["step_distance"], [0.08, 0.12])
            self.assertEqual(settings["step_angle_deg"], [-5.0, 5.0])
            self.assertAlmostEqual(settings["sampled_movement_yaw_deg"], 0.0)
            self.assertAlmostEqual(settings["minimum_lateral_separation"], 0.20)


if __name__ == "__main__":
    unittest.main()
