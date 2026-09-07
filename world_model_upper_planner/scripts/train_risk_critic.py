#!/usr/bin/env python3
"""Train a three-option risk-to-go critic on a frozen CG-OWM representation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgowm import (CandidateGroundedWorldModel, ModelConfig,
                   OptionRiskCritic, RiskConfig)
from scripts.train_h3 import SequenceDataset


@torch.no_grad()
def build_latent_cache(base, data, cache, device, batch_size=2048):
    rows = data.sequences[:, 0]
    expected = (len(rows), base.latent_dim)
    if cache.exists():
        value = np.load(cache, mmap_mode="r")
        if value.shape == expected:
            return value
        raise ValueError(f"stale latent cache shape {value.shape}, expected {expected}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    value = np.lib.format.open_memmap(
        cache, mode="w+", dtype=np.float16, shape=expected)
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        depth = torch.as_tensor(
            np.asarray(data.data["depth"][selected]), device=device,
            dtype=torch.float32) / 255.0
        proprio = torch.as_tensor(
            np.asarray(data.data["proprio"][selected]), device=device,
            dtype=torch.float32)
        value[start:start + len(selected)] = base.encode(
            depth, proprio).cpu().numpy().astype(np.float16)
    value.flush()
    return np.load(cache, mmap_mode="r")


def labels(data):
    rows = data.sequences
    return {
        "action": np.asarray(data.data["action"][rows[:, 0]], dtype=np.float32),
        "fall": np.asarray(data.data["fall"][rows], dtype=np.bool_).any(1).astype(np.float32),
        "collision": np.asarray(
            data.data["collision"][rows], dtype=np.bool_).any(1).astype(np.float32),
    }


@torch.no_grad()
def validate(risk, latent, target, indices, candidates, device, batch_size=2048):
    risk.eval(); sums = {"fall_brier": 0.0, "collision_brier": 0.0,
                         "fall_probability": 0.0, "collision_probability": 0.0,
                         "fall_rate": 0.0, "collision_rate": 0.0}
    count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        z = torch.as_tensor(np.asarray(latent[idx]), device=device,
                            dtype=torch.float32)
        action = torch.as_tensor(target["action"][idx], device=device)
        output = risk.predict_action(z, action)
        for event in ("fall", "collision"):
            probability = torch.sigmoid(output[event + "_logit"]).mean(0)
            truth = torch.as_tensor(target[event][idx], device=device)
            sums[event + "_brier"] += float((probability - truth).square().sum())
            sums[event + "_probability"] += float(probability.sum())
            sums[event + "_rate"] += float(truth.sum())
        count += len(idx)
    result = {name: value / max(count, 1) for name, value in sums.items()}
    probe = indices[:min(2048, len(indices))]
    z = torch.as_tensor(np.asarray(latent[probe]), device=device,
                        dtype=torch.float32)
    all_risk = risk.predict_candidates(z, candidates)
    for event in ("fall", "collision"):
        probability = torch.sigmoid(all_risk[event + "_logit"]).mean(0)
        result[event + "_candidate_std"] = float(
            probability.std(-1, unbiased=False).mean())
        result[event + "_candidate_range"] = float(
            (probability.max(-1).values - probability.min(-1).values).mean())
    risk.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--world_model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = SequenceDataset(args.dataset, args.device, 3, 0.2, 3003)
    checkpoint = torch.load(args.world_model, map_location=args.device)
    base = CandidateGroundedWorldModel(
        checkpoint["candidates"], ModelConfig(**checkpoint["model_config"])).to(args.device)
    base.load_state_dict(checkpoint["model"]); base.eval().requires_grad_(False)
    latent = build_latent_cache(
        base, data, args.output / "latent_h3.npy", args.device)
    target = labels(data)
    config = RiskConfig(latent_dim=base.latent_dim)
    risk = OptionRiskCritic(config).to(args.device)
    optimizer = torch.optim.AdamW(risk.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-5)
    rng = np.random.default_rng(args.seed + 1)
    validation_indices = rng.choice(
        data.val, min(20000, len(data.val)), replace=False)
    best = float("inf"); started = time.perf_counter()
    log = args.output / "metrics.jsonl"
    candidates = base.candidates
    for update in range(1, args.updates + 1):
        groups = [group for group in data.train_groups.values() if len(group)]
        per_group = int(np.ceil(args.batch_size / len(groups)))
        indices = np.concatenate([
            rng.choice(group, per_group, replace=True) for group in groups
        ])[:args.batch_size]
        rng.shuffle(indices)
        z = torch.as_tensor(np.asarray(latent[indices]), device=args.device,
                            dtype=torch.float32)
        action = torch.as_tensor(target["action"][indices], device=args.device)
        output = risk.predict_action(z, action)
        loss = z.new_zeros(())
        train_metrics = {}
        for event in ("fall", "collision"):
            truth = torch.as_tensor(target[event][indices], device=args.device)
            event_loss = F.binary_cross_entropy_with_logits(
                output[event + "_logit"], truth[None].expand_as(
                    output[event + "_logit"]))
            loss = loss + event_loss
            train_metrics[event + "_loss"] = float(event_loss.detach())
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(risk.parameters(), 10.0)
        optimizer.step()
        if update == 1 or update % args.validate_every == 0:
            val = validate(risk, latent, target, validation_indices,
                           candidates, args.device)
            record = {"update": update,
                      "wall_seconds": time.perf_counter() - started,
                      "loss": float(loss.detach()), "grad_norm": float(grad),
                      "train": train_metrics, "validation": val}
            with log.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            score = val["fall_brier"] + val["collision_brier"]
            if score < best:
                best = score
                torch.save({"format_version": 1, "risk": risk.state_dict(),
                            "risk_config": asdict(config),
                            "world_model": str(args.world_model),
                            "update": update, "validation": val},
                           args.output / "risk_best.pt")
    summary = {"sequences": len(data.sequences), "train": len(data.train),
               "val": len(data.val), "best_score": best,
               "final_validation": validate(
                   risk, latent, target, validation_indices,
                   candidates, args.device),
               "wall_seconds": time.perf_counter() - started}
    torch.save({"format_version": 1, "risk": risk.state_dict(),
                "risk_config": asdict(config), "world_model": str(args.world_model),
                "update": args.updates, "validation": summary["final_validation"]},
               args.output / "risk_final.pt")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
