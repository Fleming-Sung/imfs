#!/usr/bin/env python3
"""Train frozen-lower landing residual distributions on real option outcomes."""

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

from cgowm import (CandidateGroundedWorldModel, LandingConfig,
                   LandingDistributionCritic, ModelConfig)
from cgowm.data import load_arrays, sampling_groups


@torch.no_grad()
def encode_cache(model, data, rows, path, device, batch_size=2048):
    expected = (len(rows), model.latent_dim)
    if path.exists():
        cached = np.load(path, mmap_mode="r")
        if cached.shape == expected:
            return cached
        raise ValueError(f"stale latent cache {cached.shape}, expected {expected}")
    value = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=expected)
    for start in range(0, len(rows), batch_size):
        index = rows[start:start + batch_size]
        depth = torch.as_tensor(np.asarray(data["depth"][index]),
                                device=device, dtype=torch.float32) / 255.0
        proprio = torch.as_tensor(np.asarray(data["proprio"][index]),
                                  device=device, dtype=torch.float32)
        value[start:start + len(index)] = model.encode(
            depth, proprio).cpu().numpy().astype(np.float16)
    value.flush()
    return np.load(path, mmap_mode="r")


def gaussian_nll(output, truth):
    error = (truth[None] - output["mean"]) / output["std"]
    return (0.5 * error.square() + torch.log(output["std"])).mean()


@torch.no_grad()
def validate(model, latent, action, residual, indices, device, batch_size=2048):
    model.eval(); errors = []; normalized = []; nll = 0.0; count = 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        z = torch.as_tensor(np.asarray(latent[idx]), device=device,
                            dtype=torch.float32)
        a = torch.as_tensor(action[idx], device=device)
        y = torch.as_tensor(residual[idx], device=device)
        output = model.predict_action(z, a)
        mean = output["mean"].mean(0)
        # Total uncertainty includes aleatoric and ensemble disagreement.
        variance = (output["std"].square() + output["mean"].square()).mean(0) \
                   - mean.square()
        std = variance.clamp_min(1e-6).sqrt()
        errors.append((mean - y).cpu().numpy())
        normalized.append(((mean - y) / std).cpu().numpy())
        nll += float(gaussian_nll(output, y)) * len(idx); count += len(idx)
    error = np.concatenate(errors); zscore = np.concatenate(normalized)
    result = {
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "mae_m": float(np.mean(np.abs(error))),
        "bias_xy_m": np.mean(error, axis=0).tolist(),
        "gaussian_nll": nll / max(count, 1),
        "coverage_1sigma": float(np.mean(np.abs(zscore) <= 1.0)),
        "coverage_2sigma": float(np.mean(np.abs(zscore) <= 2.0)),
    }
    model.train(); return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--world_model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7601)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data = load_arrays(args.dataset)
    rows = np.flatnonzero(np.asarray(data["touchdown_residual_valid"]))
    checkpoint = torch.load(args.world_model, map_location=args.device)
    base = CandidateGroundedWorldModel(
        checkpoint["candidates"], ModelConfig(**checkpoint["model_config"])).to(args.device)
    base.load_state_dict(checkpoint["model"]); base.eval().requires_grad_(False)
    latent = encode_cache(base, data, rows, args.output / "latent.npy", args.device)
    action = np.asarray(data["action"][rows], dtype=np.float32)
    residual = np.asarray(data["touchdown_residual"][rows], dtype=np.float32)
    env = np.asarray(data["env_id"][rows]); rng = np.random.default_rng(args.seed + 1)
    envs = np.unique(env); rng.shuffle(envs); val_envs = envs[:max(1, len(envs) // 5)]
    val = np.flatnonzero(np.isin(env, val_envs)); train = np.flatnonzero(~np.isin(env, val_envs))
    original_groups = sampling_groups(data, rows[train])
    row_to_valid = {int(row): i for i, row in enumerate(rows)}
    groups = [np.asarray([row_to_valid[int(row)] for row in group
                          if int(row) in row_to_valid], dtype=np.int64)
              for group in original_groups.values()]
    groups = [group for group in groups if len(group)]
    config = LandingConfig(latent_dim=base.latent_dim)
    model = LandingDistributionCritic(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-5)
    best = float("inf"); started = time.perf_counter(); log = args.output / "metrics.jsonl"
    for update in range(1, args.updates + 1):
        per_group = int(np.ceil(args.batch_size / len(groups)))
        idx = np.concatenate([rng.choice(g, per_group, replace=True)
                              for g in groups])[:args.batch_size]
        rng.shuffle(idx)
        z = torch.as_tensor(np.asarray(latent[idx]), device=args.device,
                            dtype=torch.float32)
        a = torch.as_tensor(action[idx], device=args.device)
        y = torch.as_tensor(residual[idx], device=args.device)
        output = model.predict_action(z, a); loss = gaussian_nll(output, y)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if update == 1 or update % 100 == 0:
            chosen = rng.choice(val, min(20000, len(val)), replace=False)
            metrics = validate(model, latent, action, residual, chosen, args.device)
            record = {"update": update, "loss": float(loss.detach()),
                      "grad_norm": float(grad), "validation": metrics,
                      "wall_seconds": time.perf_counter() - started}
            with log.open("a") as stream: stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if metrics["rmse_m"] < best:
                best = metrics["rmse_m"]
                torch.save({"format_version": 1, "landing": model.state_dict(),
                            "landing_config": asdict(config), "update": update,
                            "validation": metrics, "world_model": str(args.world_model)},
                           args.output / "landing_best.pt")
    final = validate(model, latent, action, residual, val, args.device)
    summary = {"valid_samples": len(rows), "train": len(train), "val": len(val),
               "best_rmse_m": best, "final_validation": final,
               "wall_seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
