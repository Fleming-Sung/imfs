#!/usr/bin/env python3
"""Train three-step option dynamics from episode-linked transitions."""

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


class SequenceDataset:
    def __init__(self, path, device, horizon, val_fraction, seed):
        self.data = load_arrays(path)
        self.device = torch.device(device)
        self.horizon = int(horizon)
        lookup = {
            (int(env), int(episode), int(option)): row
            for row, (env, episode, option) in enumerate(zip(
                self.data["env_id"], self.data["episode_id"],
                self.data["option_index"]))}
        sequences = []
        for key, row in lookup.items():
            env, episode, option = key
            rows = [lookup.get((env, episode, option + step))
                    for step in range(self.horizon)]
            if all(value is not None for value in rows):
                sequences.append(rows)
        self.sequences = np.asarray(sequences, dtype=np.int64)
        rng = np.random.default_rng(seed)
        envs = np.unique(self.data["env_id"])
        rng.shuffle(envs)
        val_envs = envs[:max(1, int(round(len(envs) * val_fraction)))]
        sequence_envs = self.data["env_id"][self.sequences[:, 0]]
        self.val = np.nonzero(np.isin(sequence_envs, val_envs))[0]
        self.train = np.nonzero(~np.isin(sequence_envs, val_envs))[0]
        terrain = (self.data["terrain_kind"] if "terrain_kind" in self.data
                   else np.full(len(self.data["env_id"]), "unknown"))
        sequence_terrain = terrain[self.sequences[:, 0]]
        transition_groups = sampling_groups(self.data, self.sequences[:, 0])
        self.train_groups = {}
        first_rows = self.sequences[:, 0]
        for name, rows_for_group in transition_groups.items():
            self.train_groups[name] = self.train[np.isin(
                first_rows[self.train], rows_for_group)]
        self.val_groups = {
            str(kind): self.val[sequence_terrain[self.val] == kind]
            for kind in np.unique(sequence_terrain[self.val])}

    def batch(self, sequence_indices, depth_mode="normal"):
        rows = self.sequences[sequence_indices]
        d = self.data
        initial = rows[:, 0]
        depth0 = torch.as_tensor(
            d["depth"][initial], dtype=torch.float32, device=self.device) / 255.0
        next_depth = torch.as_tensor(
            d["next_depth"][rows], dtype=torch.float32, device=self.device) / 255.0
        if depth_mode == "zero":
            depth0.zero_(); next_depth.zero_()
        elif depth_mode == "shuffled":
            order = torch.roll(torch.arange(len(rows), device=self.device), 1)
            depth0 = depth0[order]; next_depth = next_depth[order]
        def tensor(name, index=rows):
            return torch.as_tensor(
                d[name][index], dtype=torch.float32, device=self.device)
        return {
            "depth": torch.cat((depth0[:, None], next_depth), dim=1),
            "proprio": torch.cat((
                tensor("proprio", initial)[:, None], tensor("next_proprio")), dim=1),
            "action": tensor("action"), "reward": tensor("reward"),
            "progress": tensor("progress"), "support": tensor("support"),
            "touchdown_error": tensor("touchdown_error"),
            "fall": tensor("fall"), "collision": tensor("collision"),
            "done": tensor("done"),
            "candidate_support": tensor("candidate_support"),
            "candidate_progress": tensor("candidate_progress"),
            "candidate_valid": tensor("candidate_valid"),
            **({"candidate_alignment": tensor("candidate_alignment")}
               if "candidate_alignment" in d else {}),
        }


@torch.no_grad()
def validate(trainer, dataset, batch_size, batches, mode="normal", pool=None):
    trainer.model.eval()
    sums = {}
    for offset in range(batches):
        pool = dataset.val if pool is None else pool
        indices = np.resize(pool, (offset + 1) * batch_size)[
            offset * batch_size:(offset + 1) * batch_size]
        batch = dataset.batch(indices, mode)
        _, metrics = trainer.loss(batch)
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        latent = trainer.model.encode(batch["depth"][:, 0], batch["proprio"][:, 0])
        row = torch.arange(len(latent), device=latent.device)
        for step in range(dataset.horizon):
            prediction = trainer.model.predict_candidates(latent)
            score = 10 * prediction["progress"] - 3 * (1 - prediction["support"])
            selected = score.argmax(-1)
            prior_selected = prediction["policy_logits"].argmax(-1)
            valid = batch["candidate_valid"][:, step].bool()
            progress = batch["candidate_progress"][:, step]
            best = progress.masked_fill(~valid, -10).max(-1).values
            prefix = f"selection_h{step + 1}"
            sums[prefix + "_valid"] = sums.get(prefix + "_valid", 0) + float(
                valid[row, selected].float().mean())
            sums[prefix + "_progress"] = sums.get(prefix + "_progress", 0) + float(
                progress[row, selected].mean())
            sums[prefix + "_regret"] = sums.get(prefix + "_regret", 0) + float(
                (best - progress[row, selected]).clamp_min(0).mean())
            sums[prefix + "_prior_valid"] = sums.get(
                prefix + "_prior_valid", 0) + float(
                    valid[row, prior_selected].float().mean())
            sums[prefix + "_prior_progress"] = sums.get(
                prefix + "_prior_progress", 0) + float(
                    progress[row, prior_selected].mean())
            sums[prefix + "_prior_regret"] = sums.get(
                prefix + "_prior_regret", 0) + float(
                    (best - progress[row, prior_selected]).clamp_min(0).mean())
            latent = trainer.model.next(latent, batch["action"][:, step])
    trainer.model.train()
    return {name: value / batches for name, value in sums.items()}


def save(path, model, trainer, args, update, validation):
    torch.save({
        "format_version": 1, "model": model.state_dict(),
        "target": trainer.target.state_dict(),
        "model_config": asdict(model.config),
        "trainer_config": asdict(trainer.config),
        "candidates": model.candidates.cpu(), "update": update,
        "args": vars(args), "validation": validation,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--validation_batches", type=int, default=4)
    parser.add_argument("--terrain_validation_batches", type=int, default=2)
    parser.add_argument("--balanced_terrain_sampling", action="store_true")
    parser.add_argument("--policy_only", action="store_true",
                        help="freeze world model and adapt only proposal policy")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = SequenceDataset(args.dataset, args.device, args.horizon, 0.2, args.seed)
    initial = torch.load(args.init, map_location=args.device)
    model = CandidateGroundedWorldModel(
        initial["candidates"], ModelConfig(**initial["model_config"])).to(args.device)
    model.load_state_dict(initial["model"])
    if args.policy_only:
        model.requires_grad_(False)
        model.policy_head.requires_grad_(True)
    trainer = WorldModelTrainer(model, TrainerConfig(
        learning_rate=1e-4, temporal_decay=0.8, q_coef=0.5,
        policy_coef=0.1, privileged_geometry_coef=2.0))
    trainer.target.load_state_dict(initial["target"])
    rng = np.random.default_rng(args.seed + 4)
    best = float("inf"); start = time.perf_counter()
    log_path = args.output / "metrics.jsonl"
    for update in range(1, args.updates + 1):
        if args.balanced_terrain_sampling:
            groups = [value for value in data.train_groups.values() if len(value)]
            per_group = int(np.ceil(args.batch_size / len(groups)))
            indices = np.concatenate([
                rng.choice(group, per_group, replace=True) for group in groups
            ])[:args.batch_size]
            rng.shuffle(indices)
        else:
            indices = rng.choice(data.train, args.batch_size, replace=True)
        train = trainer.train_step(data.batch(indices))
        if update == 1 or update % args.validate_every == 0:
            val = validate(trainer, data, args.batch_size,
                           args.validation_batches)
            terrain_val = {
                kind: validate(trainer, data, args.batch_size,
                               args.terrain_validation_batches, "normal", pool)
                for kind, pool in data.val_groups.items() if len(pool)}
            record = {"update": update, "wall_seconds": time.perf_counter() - start,
                      "train": train, "validation": val,
                      "terrain_validation": terrain_val}
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if args.policy_only:
                score = float(np.mean([
                    value[f"selection_h{s}_prior_regret"]
                    + 0.25 * (1 - value[f"selection_h{s}_prior_valid"])
                    for value in terrain_val.values()
                    for s in range(1, args.horizon + 1)]))
            else:
                score = float(np.mean([
                    value[f"selection_h{s}_regret"]
                    + 0.25 * (1 - value[f"selection_h{s}_valid"])
                    for value in terrain_val.values()
                    for s in range(1, args.horizon + 1)]))
            if score < best:
                best = score
                save(args.output / "model_best.pt", model, trainer, args,
                     update, val)
    ablations = {mode: validate(
        trainer, data, args.batch_size, args.validation_batches, mode)
        for mode in ("normal", "shuffled", "zero")}
    save(args.output / "model_final.pt", model, trainer, args,
         args.updates, ablations["normal"])
    summary = {"sequences": len(data.sequences), "train": len(data.train),
               "val": len(data.val), "best_score": best,
               "depth_ablations": ablations,
               "terrain_validation": {
                   kind: validate(trainer, data, args.batch_size,
                                  args.terrain_validation_batches, "normal", pool)
                   for kind, pool in data.val_groups.items() if len(pool)},
               "wall_seconds": time.perf_counter() - start}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
