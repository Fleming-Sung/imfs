#!/usr/bin/env python3
"""Fast shape/gradient/planning smoke test without Isaac Gym."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgowm import CandidateGroundedWorldModel, BeamPlanner, WorldModelTrainer


def main():
    torch.manual_seed(7)
    candidates = torch.rand(24, 3) * 2.0 - 1.0
    model = CandidateGroundedWorldModel(candidates)
    trainer = WorldModelTrainer(model)
    batch_size, horizon = 4, 3
    batch = {
        "depth": torch.rand(batch_size, horizon + 1, 1, 64, 64),
        "proprio": torch.randn(batch_size, horizon + 1, 36),
        "action": torch.rand(batch_size, horizon, 3) * 2.0 - 1.0,
        "reward": torch.randn(batch_size, horizon),
        "progress": torch.randn(batch_size, horizon) * 0.1,
        "support": torch.rand(batch_size, horizon),
        "touchdown_error": torch.rand(batch_size, horizon) * 0.1,
        "fall": torch.randint(0, 2, (batch_size, horizon)).float(),
        "collision": torch.randint(0, 2, (batch_size, horizon)).float(),
        "done": torch.zeros(batch_size, horizon),
        "valid": torch.ones(batch_size, horizon),
    }
    metrics = trainer.train_step(batch)
    latent = model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
    selected, diagnostics = BeamPlanner(model).plan(latent)
    print({
        "loss": metrics["loss_total"],
        "latent_shape": tuple(latent.shape),
        "selected_shape": tuple(selected.shape),
        "fallbacks": sum(item["fallback"] for item in diagnostics),
    })


if __name__ == "__main__":
    main()

