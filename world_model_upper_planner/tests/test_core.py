import unittest

import torch

from cgowm import (BeamPlanner, CandidateGroundedWorldModel, ModelConfig,
                   PlannerConfig, VectorizedBeamPlanner, WorldModelTrainer)


class CoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.candidates = torch.rand(12, 3) * 2.0 - 1.0
        config = ModelConfig(geometry_dim=16, dynamics_dim=16, hidden_dim=32)
        self.model = CandidateGroundedWorldModel(self.candidates, config)

    def test_candidate_outputs_and_transition_shapes(self):
        depth = torch.rand(2, 1, 64, 64)
        proprio = torch.rand(2, 36)
        latent = self.model.encode(depth, proprio)
        output = self.model.predict_candidates(latent)
        self.assertEqual(latent.shape, (2, 32))
        self.assertEqual(output["reward"].shape, (2, 12))
        self.assertEqual(output["q"].shape, (2, 2, 12))
        self.assertEqual(self.model.next(latent, self.candidates[:2]).shape, (2, 32))

    def test_paper_candidate_grid_has_294_unique_actions(self):
        from adapters.frozen_lower_env.contracts import FootholdActionBounds
        from cgowm.candidates import make_candidates
        bounds = FootholdActionBounds(
            forward_m=(0.08, 0.30), lateral_abs_m=(0.10, 0.26),
            yaw_deg=(-12.0, 12.0), z_m=0.0)
        candidates = make_candidates(bounds)
        self.assertEqual(len(candidates), 294)
        self.assertEqual(len(torch.unique(candidates, dim=0)), 294)

    def test_planner_returns_only_static_reachable_candidates(self):
        latent = self.model.encode(torch.rand(2, 1, 64, 64), torch.rand(2, 36))
        mask = torch.zeros(2, 12, dtype=torch.bool)
        mask[:, [2, 7]] = True
        planner = BeamPlanner(self.model, PlannerConfig(
            horizon=2, beam_width=4, proposals_per_beam=12,
            feasibility_threshold=0.0))
        selected, diagnostics = planner.plan(latent, mask)
        self.assertTrue(all(int(value) in (2, 7) for value in selected))
        self.assertTrue(all(not item["fallback"] for item in diagnostics))

    def test_sequence_training_is_finite(self):
        batch_size, horizon = 2, 2
        batch = {
            "depth": torch.rand(batch_size, horizon + 1, 1, 64, 64),
            "proprio": torch.rand(batch_size, horizon + 1, 36),
            "action": torch.rand(batch_size, horizon, 3) * 2 - 1,
            "reward": torch.rand(batch_size, horizon),
            "progress": torch.rand(batch_size, horizon),
            "support": torch.rand(batch_size, horizon),
            "touchdown_error": torch.rand(batch_size, horizon),
            "fall": torch.zeros(batch_size, horizon),
            "collision": torch.zeros(batch_size, horizon),
            "done": torch.zeros(batch_size, horizon),
            "candidate_support": torch.rand(batch_size, horizon, 12),
            "candidate_progress": torch.rand(batch_size, horizon, 12),
            "candidate_valid": torch.ones(batch_size, horizon, 12),
        }
        metrics = WorldModelTrainer(self.model).train_step(batch)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_total"])))

    def test_all_invalid_candidate_row_is_finite(self):
        batch_size, horizon = 2, 1
        batch = {
            "depth": torch.rand(batch_size, horizon + 1, 1, 64, 64),
            "proprio": torch.rand(batch_size, horizon + 1, 36),
            "action": torch.rand(batch_size, horizon, 3) * 2 - 1,
            "reward": torch.rand(batch_size, horizon),
            "progress": torch.rand(batch_size, horizon),
            "support": torch.rand(batch_size, horizon),
            "touchdown_error": torch.rand(batch_size, horizon),
            "fall": torch.zeros(batch_size, horizon),
            "collision": torch.zeros(batch_size, horizon),
            "done": torch.zeros(batch_size, horizon),
            "candidate_support": torch.rand(batch_size, horizon, 12),
            "candidate_progress": torch.rand(batch_size, horizon, 12),
            "candidate_valid": torch.zeros(batch_size, horizon, 12),
        }
        metrics = WorldModelTrainer(self.model).train_step(batch)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss_total"])))

    def test_vectorized_planner_respects_sparse_static_mask(self):
        latent = self.model.encode(torch.rand(3, 1, 64, 64), torch.rand(3, 36))
        mask = torch.zeros(3, 12, dtype=torch.bool)
        mask[0, 2] = True
        mask[1, 7] = True
        mask[2, [2, 7]] = True
        planner = VectorizedBeamPlanner(self.model, PlannerConfig(
            horizon=3, beam_width=4, proposals_per_beam=12,
            feasibility_threshold=0.0))
        selected, diagnostics = planner.plan(latent, mask)
        self.assertEqual(int(selected[0]), 2)
        self.assertEqual(int(selected[1]), 7)
        self.assertIn(int(selected[2]), (2, 7))
        self.assertEqual(diagnostics["best_score"].shape, (3,))

    def test_obstacle_terrain_geometry_labels_construct(self):
        from adapters.frozen_lower_env.contracts import FootholdActionBounds
        from adapters.frozen_lower_env.terrain import (TerrainSpec,
                                                       build_tiled_heightfield)
        from adapters.geometry_labels import CandidateGeometryLabeler
        bounds = FootholdActionBounds(
            forward_m=(0.08, 0.30), lateral_abs_m=(0.10, 0.26),
            yaw_deg=(-12.0, 12.0), z_m=0.0)
        tiled = build_tiled_heightfield([
            TerrainSpec(kind="household", length_m=3.5, seed=9)])
        labeler = CandidateGeometryLabeler(
            tiled, bounds, self.candidates, "cpu")
        self.assertEqual(labeler.distance.shape[0], 1)
        self.assertTrue(torch.isfinite(labeler.distance).any())


if __name__ == "__main__":
    unittest.main()
