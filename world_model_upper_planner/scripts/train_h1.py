#!/usr/bin/env python3
"""Train and validate the H=1 candidate-grounded option world model."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgowm import (CandidateGroundedWorldModel, ModelConfig,
                   TrainerConfig, WorldModelTrainer)
from cgowm.data import load_arrays, sampling_groups


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path,
                        help="optional CG-OWM checkpoint for low-LR adaptation")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--val_env_fraction", type=float, default=0.20)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--validation_batches", type=int, default=8)
    parser.add_argument("--terrain_validation_batches", type=int, default=2)
    parser.add_argument("--balanced_terrain_sampling", action="store_true")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


class Dataset:
    def __init__(self, path, device, val_fraction, seed):
        self.data = load_arrays(path)
        self.device = torch.device(device)
        rng = np.random.default_rng(seed)
        envs = np.unique(self.data["env_id"])
        rng.shuffle(envs)
        val_count = max(1, int(round(len(envs) * val_fraction)))
        val_envs = envs[:val_count]
        is_val = np.isin(self.data["env_id"], val_envs)
        self.train = np.nonzero(~is_val)[0]
        self.val = np.nonzero(is_val)[0]
        terrain = (self.data["terrain_kind"] if "terrain_kind" in self.data
                   else np.full(len(self.data["env_id"]), "unknown"))
        self.train_groups = sampling_groups(self.data, self.train)
        self.val_groups = {str(kind): self.val[terrain[self.val] == kind]
                           for kind in np.unique(terrain[self.val])}
        if not len(self.train) or not len(self.val):
            raise ValueError("terrain-disjoint split produced an empty partition")

    def batch(self, indices, depth_mode="normal"):
        d = self.data
        depth = torch.as_tensor(
            d["depth"][indices], dtype=torch.float32, device=self.device) / 255.0
        next_depth = torch.as_tensor(
            d["next_depth"][indices], dtype=torch.float32,
            device=self.device) / 255.0
        if depth_mode == "zero":
            depth.zero_(); next_depth.zero_()
        elif depth_mode == "shuffled":
            order = torch.roll(torch.arange(len(indices), device=self.device), 1)
            depth = depth[order]; next_depth = next_depth[order]

        def tensor(name):
            return torch.as_tensor(
                d[name][indices], dtype=torch.float32, device=self.device)
        return {
            "depth": torch.stack((depth, next_depth), dim=1),
            "proprio": torch.stack(
                (tensor("proprio"), tensor("next_proprio")), dim=1),
            "action": tensor("action").unsqueeze(1),
            "reward": tensor("reward").unsqueeze(1),
            "progress": tensor("progress").unsqueeze(1),
            "support": tensor("support").unsqueeze(1),
            "touchdown_error": tensor("touchdown_error").unsqueeze(1),
            "fall": tensor("fall").unsqueeze(1),
            "collision": tensor("collision").unsqueeze(1),
            "done": tensor("done").unsqueeze(1),
            "candidate_support": tensor("candidate_support").unsqueeze(1),
            "candidate_progress": tensor("candidate_progress").unsqueeze(1),
            "candidate_valid": tensor("candidate_valid").unsqueeze(1),
            **({"candidate_alignment": tensor("candidate_alignment").unsqueeze(1)}
               if "candidate_alignment" in d else {}),
        }


@torch.no_grad()
def validate(trainer, dataset, batch_size, batches, mode="normal", pool=None):
    trainer.model.eval()
    sums = {}
    selected_valid = selected_progress = regret = selected_support = 0.0
    prior_valid = prior_progress = prior_regret = 0.0
    count = 0
    for offset in range(batches):
        pool = dataset.val if pool is None else pool
        start = (offset * batch_size) % len(pool)
        indices = np.resize(pool, start + batch_size)[start:start + batch_size]
        batch = dataset.batch(indices, mode)
        _, metrics = trainer.loss(batch)
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        latent = trainer.model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        prediction = trainer.model.predict_candidates(latent)
        score = 10.0 * prediction["progress"] - 3.0 * (1.0 - prediction["support"])
        index = score.argmax(-1)
        prior_index = prediction["policy_logits"].argmax(-1)
        row = torch.arange(len(index), device=index.device)
        target_valid = batch["candidate_valid"][:, 0].bool()
        target_progress = batch["candidate_progress"][:, 0]
        target_support = batch["candidate_support"][:, 0]
        best = target_progress.masked_fill(~target_valid, -10.0).max(-1).values
        selected_valid += float(target_valid[row, index].float().sum())
        selected_progress += float(target_progress[row, index].sum())
        selected_support += float(target_support[row, index].sum())
        regret += float((best - target_progress[row, index]).clamp_min(0).sum())
        prior_valid += float(target_valid[row, prior_index].float().sum())
        prior_progress += float(target_progress[row, prior_index].sum())
        prior_regret += float(
            (best - target_progress[row, prior_index]).clamp_min(0).sum())
        count += len(index)
    result = {name: value / batches for name, value in sums.items()}
    result.update({
        "selection_valid_rate": selected_valid / count,
        "selection_progress_m": selected_progress / count,
        "selection_support": selected_support / count,
        "selection_regret_m": regret / count,
        "prior_valid_rate": prior_valid / count,
        "prior_progress_m": prior_progress / count,
        "prior_regret_m": prior_regret / count,
    })
    trainer.model.train()
    return result


def save(path, model, trainer, args, update, validation):
    torch.save({
        "format_version": 1,
        "model": model.state_dict(), "target": trainer.target.state_dict(),
        "model_config": asdict(model.config),
        "trainer_config": asdict(trainer.config),
        "candidates": model.candidates.cpu(), "update": update,
        "args": vars(args), "validation": validation,
    }, path)


def main():
    args = arguments()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        args.dataset, args.device, args.val_env_fraction, args.seed)
    candidates = dataset.data["candidates"]
    if args.init:
        initial = torch.load(args.init, map_location=args.device)
        if not np.allclose(initial["candidates"].cpu().numpy(), candidates):
            raise ValueError("initial checkpoint candidate grid differs from dataset")
        model = CandidateGroundedWorldModel(
            candidates, ModelConfig(**initial["model_config"])).to(args.device)
        model.load_state_dict(initial["model"])
    else:
        initial = None
        model = CandidateGroundedWorldModel(
            candidates, ModelConfig()).to(args.device)
    trainer = WorldModelTrainer(model, TrainerConfig(
        learning_rate=args.learning_rate, q_coef=0.5,
        policy_coef=0.1, privileged_geometry_coef=2.0))
    if initial is not None:
        trainer.target.load_state_dict(initial["target"])
    rng = np.random.default_rng(args.seed + 1)
    log_path = args.output / "metrics.jsonl"
    best_score = float("inf")
    start = time.perf_counter()
    for update in range(1, args.updates + 1):
        if args.balanced_terrain_sampling:
            groups = [value for value in dataset.train_groups.values() if len(value)]
            per_group = int(np.ceil(args.batch_size / len(groups)))
            indices = np.concatenate([
                rng.choice(group, size=per_group, replace=True) for group in groups
            ])[:args.batch_size]
            rng.shuffle(indices)
        else:
            indices = rng.choice(dataset.train, size=args.batch_size, replace=True)
        train_metrics = trainer.train_step(dataset.batch(indices))
        if update == 1 or update % args.validate_every == 0:
            validation = validate(
                trainer, dataset, args.batch_size,
                args.validation_batches, "normal")
            terrain_validation = {
                kind: validate(trainer, dataset, args.batch_size,
                               args.terrain_validation_batches, "normal", pool)
                for kind, pool in dataset.val_groups.items() if len(pool)}
            record = {
                "update": update, "wall_seconds": time.perf_counter() - start,
                "train": train_metrics, "validation": validation,
                "terrain_validation": terrain_validation,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            score = float(np.mean([
                value["selection_regret_m"]
                + 0.25 * (1.0 - value["selection_valid_rate"])
                for value in terrain_validation.values()]))
            if score < best_score:
                best_score = score
                save(args.output / "model_best.pt", model, trainer,
                     args, update, validation)
    final_validation = validate(
        trainer, dataset, args.batch_size, args.validation_batches, "normal")
    ablations = {
        mode: validate(trainer, dataset, args.batch_size,
                       args.validation_batches, mode)
        for mode in ("normal", "shuffled", "zero")}
    save(args.output / "model_final.pt", model, trainer,
         args, args.updates, final_validation)
    summary = {
        "train_samples": len(dataset.train), "val_samples": len(dataset.val),
        "best_score": best_score, "final_validation": final_validation,
        "depth_ablations": ablations,
        "terrain_validation": {
            kind: validate(trainer, dataset, args.batch_size,
                           args.terrain_validation_batches, "normal", pool)
            for kind, pool in dataset.val_groups.items() if len(pool)},
        "wall_seconds": time.perf_counter() - start,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
