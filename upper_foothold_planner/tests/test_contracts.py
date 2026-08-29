import json
import unittest
from pathlib import Path

import numpy as np
import torch

from upper_planner.cem import plan, plan_anchored_ensemble, plan_ensemble
from upper_planner.contracts import (FootholdActionBounds, PolarFootholdActionBounds,
                                     preprocess_isaac_depth)
from upper_planner.lower_policy import FrozenLowerPolicy
from upper_planner.terrain import TerrainSpec, build_tiled_heightfield, generate_support_layout
from upper_planner.world_model import LatentWorldModel, make_world_model
from upper_planner.ensemble import make_ensemble, one_step_predictions
from upper_planner.replay import ReplayBuffer
from upper_planner.world_model_trainer import WorldModelTrainer
from upper_planner.task_world_model_trainer import TaskWorldModelTrainer
from upper_planner.option_world_model_trainer import OptionWorldModelTrainer
from upper_planner.depth_diagnostics import depth_prediction_sequence


ROOT = Path(__file__).resolve().parents[1]


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT / "config" / "default.json").read_text())

    def test_depth_preprocessing_handles_no_hit_and_near_far(self):
        raw = np.array([[-0.1, -3.0, -np.inf, np.nan]], dtype=np.float32)
        out = preprocess_isaac_depth(raw, 0.1, 3.0)
        np.testing.assert_allclose(out[0, :3], [1.0, 0.0, 0.0])
        self.assertEqual(float(out[0, 3]), 0.0)

    def test_action_has_correct_lateral_side(self):
        bounds = FootholdActionBounds.from_config(self.cfg["action"])
        neutral = np.zeros(3, dtype=np.float32)
        left, right = bounds.decode(neutral, 0), bounds.decode(neutral, 1)
        self.assertGreater(left[1], 0.0)
        self.assertLess(right[1], 0.0)
        self.assertAlmostEqual(left[0], 0.12, places=6)
        self.assertAlmostEqual(abs(left[1]), 0.21, places=6)

    def test_action_decode_supports_batched_swing_feet(self):
        bounds = FootholdActionBounds.from_config(self.cfg["action"])
        actions = torch.zeros(2, 3)
        decoded = bounds.decode(actions, torch.tensor([0, 1]))
        self.assertEqual(tuple(decoded.shape), (2, 4))
        self.assertGreater(float(decoded[0, 1]), 0.0)
        self.assertLess(float(decoded[1, 1]), 0.0)

    def test_polar_action_stays_in_lower_training_support(self):
        bounds = PolarFootholdActionBounds.from_config(self.cfg["action_polar"])
        actions = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0]])
        decoded = bounds.decode(actions, torch.tensor([0, 1]))
        distance = torch.norm(decoded[:, :2], dim=-1)
        self.assertTrue((distance <= 0.35).all())
        self.assertGreaterEqual(float(decoded[0, 1]), 0.10)
        self.assertLessEqual(float(decoded[1, 1]), -0.10)

    def test_course_action_is_a_strict_subset_of_polar_support(self):
        broad = self.cfg["action_polar"]
        course = self.cfg["action_polar_course"]
        self.assertGreaterEqual(course["distance_m"][0], broad["distance_m"][0])
        self.assertLessEqual(course["distance_m"][1], broad["distance_m"][1])
        self.assertGreaterEqual(course["direction_deg"][0], broad["direction_deg"][0])
        self.assertLessEqual(course["direction_deg"][1], broad["direction_deg"][1])
        self.assertGreaterEqual(course["yaw_deg"][0], broad["yaw_deg"][0])
        self.assertLessEqual(course["yaw_deg"][1], broad["yaw_deg"][1])
        bounds = PolarFootholdActionBounds.from_config(course)
        decoded = bounds.decode(torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0]))
        self.assertLessEqual(float(torch.norm(decoded[0, :2])), 0.25)

    def test_world_model_and_cem_shapes(self):
        torch.manual_seed(0)
        model = LatentWorldModel()
        z = model.encode(torch.zeros(2, 1, 64, 64), torch.zeros(2, 36))
        self.assertEqual(tuple(z.shape), (2, 128))
        collision, fall = model.predict_event_logits(z, torch.zeros(2, 3))
        self.assertEqual(tuple(collision.shape), (2,))
        self.assertEqual(tuple(fall.shape), (2,))
        self.assertEqual(tuple(model.reconstruct_depth(z).shape), (2, 1, 16, 16))
        self.assertEqual(tuple(model.predict_next_depth(
            z, torch.zeros(2, 3)).shape), (2, 1, 16, 16))
        components = model.predict_task_components(z, torch.zeros(2, 3))
        self.assertEqual(set(components), {
            "progress", "goal_logit", "collision_logit", "fall_logit",
            "off_support_logit"})
        action, info = plan(model, z[:1], candidates=32, elites=4, iterations=2)
        self.assertEqual(tuple(action.shape), (3,))
        self.assertEqual(tuple(info["mean"].shape), (1, 5, 3))
        batched_action, _ = plan(model, z, candidates=16, elites=4, iterations=1)
        self.assertEqual(tuple(batched_action.shape), (2, 3))

    def test_spatial_world_model_predicts_32_pixel_depth(self):
        model = make_world_model(variant="spatial")
        z = model.encode(torch.zeros(2, 1, 64, 64), torch.zeros(2, 36))
        self.assertEqual(tuple(model.reconstruct_depth(z).shape), (2, 1, 32, 32))
        self.assertEqual(tuple(model.predict_next_depth(
            z, torch.zeros(2, 3)).shape), (2, 1, 32, 32))

    def test_depth_diagnostic_uses_decoder_resolution(self):
        batch = {
            "depth": torch.zeros(1, 2, 1, 64, 64),
            "next_depth": torch.zeros(1, 2, 1, 64, 64),
            "proprio": torch.zeros(1, 2, 36),
            "action": torch.zeros(1, 2, 3),
        }
        for variant, size in (("compact", 16), ("spatial", 32)):
            arrays = depth_prediction_sequence(
                make_world_model(variant=variant), batch)
            self.assertEqual(tuple(arrays["predicted_16"].shape), (3, size, size))
            self.assertEqual(tuple(arrays["real_16"].shape), (3, size, size))

    def test_ensemble_predictions_keep_member_axis(self):
        models = make_ensemble(3, 128, 256, "cpu")
        prediction = one_step_predictions(
            models, torch.zeros(2, 1, 64, 64), torch.zeros(2, 36),
            torch.zeros(2, 3), self.cfg["reward"],
            self.cfg["model"]["reward_scale"])
        self.assertEqual(tuple(prediction["reward"].shape), (3, 2))
        self.assertEqual(tuple(prediction["next_depth"].shape), (3, 2, 1, 16, 16))
        latents = [model.encode(torch.zeros(2, 1, 64, 64), torch.zeros(2, 36))
                   for model in models]
        action, info = plan_ensemble(
            models, latents, horizon=2, candidates=16, elites=4, iterations=1,
            reward_cfg=self.cfg["reward"], uncertainty_coef=0.5)
        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertEqual(tuple(info["candidate_return_std"].shape), (2, 16))
        anchored_action, anchored_info = plan_anchored_ensemble(
            models[0], latents[0], models, latents, horizon=2,
            candidates=16, elites=4, iterations=1,
            reward_cfg=self.cfg["reward"], uncertainty_coef=0.5)
        self.assertEqual(tuple(anchored_action.shape), (2, 3))
        self.assertEqual(tuple(anchored_info["candidate_task_return_std"].shape), (2, 16))

    def test_frozen_lower_checkpoint_contract(self):
        policy = FrozenLowerPolicy(ROOT / "checkpoints" / "lower_model_7000.pt")
        raw, clipped = policy.infer(torch.zeros(1, 30), torch.zeros(1, 16))
        self.assertEqual(tuple(raw.shape), (1, 8))
        self.assertTrue(torch.isfinite(raw).all())
        self.assertLessEqual(float(clipped.abs().max()), 1.0)

    def test_replay_sequences_follow_env_links_and_stop_at_done(self):
        replay = ReplayBuffer(32, depth_shape=(1, 4, 4), num_envs=2,
                              return_horizon=2)
        for step in range(3):
            ids = torch.tensor([0, 1])
            done = torch.tensor([step == 1, step == 2])
            transition = {
                "ids": ids,
                "depth": torch.full((2, 1, 4, 4), step / 10.0),
                "next_depth": torch.full((2, 1, 4, 4), (step + 1) / 10.0),
                "proprio": torch.zeros(2, 36), "next_proprio": torch.zeros(2, 36),
                "action": torch.zeros(2, 3), "reward": torch.ones(2), "done": done,
                "diagnostics": {name: torch.zeros(2, dtype=torch.bool)
                                for name in ("collision", "fall", "success", "off_support")},
                "terms": {"progress": torch.zeros(2)},
            }
            replay.add_transition_batch(transition)
        rows = replay.sequence_indices(2)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(replay.env_id[row[0]] == replay.env_id[row[1]] for row in rows))
        self.assertTrue(all(not replay.done[row[0]] for row in rows))
        batch = replay.sample_sequence(2, 2, "cpu")
        self.assertEqual(tuple(batch["action"].shape), (2, 2, 3))
        padded_rows, padded_valid = replay.padded_sequence_indices(3)
        terminal_positions = {
            int(np.flatnonzero(replay.done[row] & mask)[0])
            for row, mask in zip(padded_rows, padded_valid)
            if (replay.done[row] & mask).any()
        }
        self.assertEqual(terminal_positions, {0, 1, 2})
        padded = replay.sample_sequence(
            2, 3, "cpu", allow_terminal_padding=True)
        self.assertEqual(tuple(padded["valid"].shape), (2, 3))
        self.assertTrue(torch.all(padded["valid"][:, 1:] <= padded["valid"][:, :-1]))

    def test_option_return_uses_elapsed_low_level_time(self):
        replay = ReplayBuffer(
            8, depth_shape=(1, 4, 4), num_envs=1, return_horizon=3,
            gamma=0.5, reward_scale=1.0, duration_aware_returns=True,
            nominal_option_ticks=25.0)
        for duration in (25.0, 50.0, 25.0):
            transition = {
                "ids": torch.tensor([0]),
                "depth": torch.zeros(1, 1, 4, 4),
                "next_depth": torch.zeros(1, 1, 4, 4),
                "proprio": torch.zeros(1, 36),
                "next_proprio": torch.zeros(1, 36),
                "action": torch.zeros(1, 3), "reward": torch.ones(1),
                "done": torch.tensor([False]),
                "diagnostics": {
                    **{name: torch.tensor([False]) for name in (
                        "collision", "fall", "success", "off_support")},
                    "option_duration_ticks": torch.tensor([duration]),
                },
                "terms": {"progress": torch.zeros(1)},
            }
            replay.add_transition_batch(transition)
        # 1 + 0.5^1 + 0.5^(1+2) = 1.625.  The third duration affects only
        # rewards after this three-option target and is therefore not included.
        self.assertAlmostEqual(float(replay.return_target[0]), 1.625, places=6)

    def test_multistep_trainer_unroll_is_finite(self):
        model = LatentWorldModel()
        trainer = WorldModelTrainer(model)
        batch = {
            "depth": torch.zeros(2, 3, 1, 64, 64),
            "next_depth": torch.zeros(2, 3, 1, 64, 64),
            "proprio": torch.zeros(2, 3, 36),
            "next_proprio": torch.zeros(2, 3, 36),
            "action": torch.zeros(2, 3, 3),
            "reward": torch.zeros(2, 3), "return_target": torch.zeros(2, 3),
            "collision": torch.zeros(2, 3), "fall": torch.zeros(2, 3),
            "success": torch.zeros(2, 3), "progress": torch.zeros(2, 3),
            "off_support": torch.zeros(2, 3),
            "done": torch.zeros(2, 3),
        }
        metrics = trainer.train_step_sequence(batch)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("loss_future_depth", metrics)
        self.assertIn("future_depth_mae", metrics)

    def test_task_world_model_has_no_depth_decoder_and_trains_multistep(self):
        model = make_world_model(variant="task")
        self.assertFalse(hasattr(model, "reconstruct_depth"))
        trainer = TaskWorldModelTrainer(model)
        shape = (4, 3)
        batch = {
            "depth": torch.rand(*shape, 1, 64, 64),
            "next_depth": torch.rand(*shape, 1, 64, 64),
            "proprio": torch.rand(*shape, 36),
            "next_proprio": torch.rand(*shape, 36),
            "action": 2.0 * torch.rand(*shape, 3) - 1.0,
            "reward": torch.zeros(*shape), "return_target": torch.zeros(*shape),
            "collision": torch.zeros(*shape), "fall": torch.zeros(*shape),
            "success": torch.zeros(*shape), "off_support": torch.zeros(*shape),
            "progress": torch.zeros(*shape), "heading_progress": torch.zeros(*shape),
            "collision_force": torch.zeros(*shape),
            "stability_margin": torch.ones(*shape),
            "support_fraction": torch.ones(*shape),
            "touchdown_error": torch.zeros(*shape), "done": torch.zeros(*shape),
            "macro_state": torch.zeros(*shape, 21),
        }
        batch["collision"][0, 0] = 1.0
        batch["fall"][1, 1] = 1.0
        batch["off_support"][2, 2] = 1.0
        metrics = trainer.train_step_sequence(batch)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("loss_task_margin", metrics)
        self.assertIn("prediction_support_fraction_mae", metrics)

    def test_option_world_model_uses_simnorm_twin_q_and_measured_returns(self):
        model = make_world_model(variant="option")
        trainer = OptionWorldModelTrainer(model)
        shape = (4, 3)
        batch = {
            "depth": torch.rand(*shape, 1, 64, 64),
            "next_depth": torch.rand(*shape, 1, 64, 64),
            "proprio": torch.rand(*shape, 36),
            "next_proprio": torch.rand(*shape, 36),
            "action": 2.0 * torch.rand(*shape, 3) - 1.0,
            "reward": torch.randn(*shape), "return_target": torch.zeros(*shape),
            "collision": torch.zeros(*shape), "fall": torch.zeros(*shape),
            "success": torch.zeros(*shape), "off_support": torch.zeros(*shape),
            "progress": torch.zeros(*shape), "heading_progress": torch.zeros(*shape),
            "collision_force": torch.zeros(*shape),
            "stability_margin": torch.ones(*shape),
            "support_fraction": torch.ones(*shape),
            "touchdown_error": torch.zeros(*shape), "done": torch.zeros(*shape),
            "option_duration_ticks": torch.full(shape, 25.0),
            "macro_state": torch.zeros(*shape, 21),
            "scene_id": torch.zeros(*shape, dtype=torch.long),
            "counterfactual_group": torch.full(shape, -1, dtype=torch.long),
            "valid": torch.tensor([
                [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        }
        batch["done"][0, 1] = 1.0
        latent = model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        grouped = latent.view(4, -1, 8)
        torch.testing.assert_close(grouped.sum(-1), torch.ones_like(grouped.sum(-1)))
        self.assertEqual(tuple(model.predict_q(latent, batch["action"][:, 0]).shape),
                         (2, 4))
        metrics = trainer.train_step_sequence(batch)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertIn("loss_q", metrics)
        self.assertIn("prediction_option_duration_mae_ticks", metrics)

    def test_option_model_policy_prior_bounded(self):
        model = make_world_model(latent_dim=128, hidden_dim=64, variant="option")
        latent = torch.zeros(5, 128)
        with torch.no_grad():
            action = model.policy_action(latent)
        self.assertEqual(tuple(action.shape), (5, 3))
        self.assertTrue((action >= -1.0).all())
        self.assertTrue((action <= 1.0).all())

    def test_option_trainer_policy_loss_uses_from_planner_mask(self):
        model = make_world_model(latent_dim=128, hidden_dim=64, variant="option")
        trainer = OptionWorldModelTrainer(model)
        shape = (2, 3)
        batch = {
            "depth": torch.rand(*shape, 1, 64, 64),
            "next_depth": torch.rand(*shape, 1, 64, 64),
            "proprio": torch.rand(*shape, 36),
            "next_proprio": torch.rand(*shape, 36),
            "action": 2.0 * torch.rand(*shape, 3) - 1.0,
            "reward": torch.randn(*shape), "return_target": torch.zeros(*shape),
            "collision": torch.zeros(*shape), "fall": torch.zeros(*shape),
            "success": torch.zeros(*shape), "off_support": torch.zeros(*shape),
            "progress": torch.zeros(*shape), "heading_progress": torch.zeros(*shape),
            "collision_force": torch.zeros(*shape),
            "stability_margin": torch.ones(*shape),
            "support_fraction": torch.ones(*shape),
            "touchdown_error": torch.zeros(*shape), "done": torch.zeros(*shape),
            "option_duration_ticks": torch.full(shape, 25.0),
            "macro_state": torch.zeros(*shape, 21),
            "scene_id": torch.zeros(*shape, dtype=torch.long),
            "counterfactual_group": torch.full(shape, -1, dtype=torch.long),
            "from_planner": torch.tensor(
                [[True, True, False], [True, False, False]]),
            "valid": torch.ones(*shape),
        }
        metrics = trainer.train_step_sequence(batch)
        self.assertIn("loss_policy_prior", metrics)
        self.assertTrue(np.isfinite(metrics["loss_policy_prior"]))
        self.assertTrue(np.isfinite(metrics["prediction_policy_mae"]))

    def test_terrain_families_are_deterministic_and_start_goal_supported(self):
        for kind in ("straight", "s_curve", "fork", "random"):
            first = generate_support_layout(TerrainSpec(kind=kind, seed=7))
            second = generate_support_layout(TerrainSpec(kind=kind, seed=7))
            np.testing.assert_array_equal(first.support_mask, second.support_mask)
            for point in (first.start_xy, first.goal_xy):
                ix = int(round(point[0] / first.spec.resolution_m))
                iy = int(round((point[1] + 0.5 * first.spec.width_m) / first.spec.resolution_m))
                self.assertTrue(first.support_mask[iy, ix], (kind, point))

    def test_tiled_heightfield_places_each_start_on_zero_height(self):
        specs = [TerrainSpec(kind=kind, seed=3)
                 for kind in ("straight", "s_curve", "fork", "random")]
        tiled = build_tiled_heightfield(specs)
        for origin, layout in zip(tiled.env_origins_xy_m, tiled.layouts):
            point = origin + layout.start_xy
            ix = int(round((point[0] - tiled.origin_xy_m[0]) / tiled.horizontal_scale_m))
            iy = int(round((point[1] - tiled.origin_xy_m[1]) / tiled.horizontal_scale_m))
            self.assertEqual(int(tiled.height_samples[ix, iy]), 0)


if __name__ == "__main__":
    unittest.main()
