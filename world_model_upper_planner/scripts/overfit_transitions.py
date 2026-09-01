#!/usr/bin/env python3
"""Gate 1: overfit a small real option dataset before any online training."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgowm import CandidateGroundedWorldModel, WorldModelTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    data = np.load(args.dataset)
    device = torch.device(args.device)
    model = CandidateGroundedWorldModel(data["candidates"]).to(device)
    trainer = WorldModelTrainer(model)
    count = min(len(data["reward"]), max(args.batch_size, 256))
    fixed = np.arange(count)

    def tensor(name, index, dtype=torch.float32):
        return torch.as_tensor(data[name][index], dtype=dtype, device=device)

    first_loss = last = None
    for update in range(args.updates):
        index = fixed[torch.randint(count, (args.batch_size,)).numpy()]
        depth = tensor("depth", index).float() / 255.0
        next_depth = tensor("next_depth", index).float() / 255.0
        batch = {
            "depth": torch.stack((depth, next_depth), dim=1),
            "proprio": torch.stack((tensor("proprio", index),
                                      tensor("next_proprio", index)), dim=1),
            "action": tensor("action", index).unsqueeze(1),
            "reward": tensor("reward", index).unsqueeze(1),
            "progress": tensor("progress", index).unsqueeze(1),
            "support": tensor("support", index).unsqueeze(1),
            "touchdown_error": tensor("touchdown_error", index).unsqueeze(1),
            "fall": tensor("fall", index).unsqueeze(1),
            "collision": tensor("collision", index).unsqueeze(1),
            "done": tensor("done", index).unsqueeze(1),
        }
        if "candidate_support" in data.files:
            batch.update({
                "candidate_support": tensor(
                    "candidate_support", index).unsqueeze(1),
                "candidate_progress": tensor(
                    "candidate_progress", index).unsqueeze(1),
                "candidate_valid": tensor(
                    "candidate_valid", index).unsqueeze(1),
            })
        last = trainer.train_step(batch)
        first_loss = last["loss_total"] if first_loss is None else first_loss
    print({
        "samples": count, "updates": args.updates,
        "first_loss": first_loss, "last_loss": last["loss_total"],
        "loss_ratio": last["loss_total"] / max(first_loss, 1e-9),
        "consistency_h1": last["consistency_h1"],
        "q_h1": last["q_h1"], "reward_h1": last["reward_h1"],
        "support_h1": last["support_h1"],
        "candidate_support_h1": last["candidate_support_h1"],
        "candidate_progress_h1": last["candidate_progress_h1"],
    })


if __name__ == "__main__":
    main()
