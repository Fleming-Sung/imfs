#!/usr/bin/env python3
"""Held-out calibration and candidate-action sensitivity audit."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cgowm import CandidateGroundedWorldModel, ModelConfig
from cgowm.data import load_arrays


def event_metrics(probability, target):
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.bool_)
    eps = 1e-7
    order = np.argsort(probability)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    positives, negatives = target.sum(), (~target).sum()
    auroc = ((ranks[target].sum() - positives * (positives + 1) / 2)
             / max(positives * negatives, 1))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    calibration = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = ((probability >= lower) &
                (probability < upper if upper < 1.0 else probability <= upper))
        if not mask.any():
            continue
        predicted = float(probability[mask].mean())
        observed = float(target[mask].mean())
        ece += float(mask.mean()) * abs(predicted - observed)
        calibration.append({"lower": lower, "upper": upper,
                            "count": int(mask.sum()),
                            "predicted": predicted, "observed": observed})
    thresholds = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, 201)))
    best = (0.0, 0.5)
    for threshold in thresholds:
        predicted = probability >= threshold
        tpr = (predicted & target).sum() / max(positives, 1)
        tnr = ((~predicted) & (~target)).sum() / max(negatives, 1)
        if 0.5 * (tpr + tnr) > best[0]:
            best = (float(0.5 * (tpr + tnr)), float(threshold))
    top = probability >= np.quantile(probability, 0.90)
    return {
        "positive_rate": float(target.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(np.mean((probability - target) ** 2)),
        "nll": float(np.mean(-(target * np.log(probability.clip(eps, 1 - eps))
                               + (~target) * np.log((1 - probability).clip(eps, 1 - eps))))),
        "ece_10": float(ece), "auroc": float(auroc),
        "best_balanced_accuracy": best[0], "best_threshold": best[1],
        "top_decile_positive_recall": float((top & target).sum() / max(positives, 1)),
        "calibration_bins": calibration,
    }


def fit_platt(probability, target):
    """Fit a two-parameter held-out logit calibrator."""
    eps = 1e-7
    x = torch.as_tensor(
        np.log(np.asarray(probability).clip(eps, 1 - eps)
               / (1 - np.asarray(probability).clip(eps, 1 - eps))),
        dtype=torch.float64)
    y = torch.as_tensor(np.asarray(target), dtype=torch.float64)
    scale = torch.ones((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([scale, bias], max_iter=100,
                                  tolerance_grad=1e-10,
                                  tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scale * x + bias, y)
        loss.backward()
        return loss
    optimizer.step(closure)
    calibrated = torch.sigmoid(scale.detach() * x + bias.detach()).numpy()
    return {"scale": float(scale.detach()), "bias": float(bias.detach()),
            "metrics": event_metrics(calibrated, np.asarray(target))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3003)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data = load_arrays(args.dataset)
    envs = np.unique(data["env_id"])
    rng = np.random.default_rng(args.seed)
    rng.shuffle(envs)
    val_envs = envs[:max(1, int(round(0.20 * len(envs))))]
    pool = np.nonzero(np.isin(data["env_id"], val_envs))[0]
    indices = rng.choice(pool, min(args.samples, len(pool)), replace=False)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = CandidateGroundedWorldModel(
        checkpoint["candidates"], ModelConfig(**checkpoint["model_config"])).to(args.device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    predictions = {"fall": [], "collision": []}
    sensitivity = {"fall_std": [], "fall_range": [],
                   "collision_std": [], "collision_range": []}
    with torch.no_grad():
        for start in range(0, len(indices), args.batch_size):
            row = indices[start:start + args.batch_size]
            depth = torch.as_tensor(
                np.asarray(data["depth"][row]), device=args.device,
                dtype=torch.float32) / 255.0
            proprio = torch.as_tensor(
                np.asarray(data["proprio"][row]), device=args.device,
                dtype=torch.float32)
            action = torch.as_tensor(
                np.asarray(data["action"][row]), device=args.device,
                dtype=torch.float32)
            latent = model.encode(depth, proprio)
            chosen = model.predict_action(latent, action)
            all_candidates = model.predict_candidates(latent)
            for event in ("fall", "collision"):
                probability = torch.sigmoid(chosen[event + "_logit"])
                candidate_probability = torch.sigmoid(
                    all_candidates[event + "_logit"])
                predictions[event].append(probability.cpu().numpy())
                sensitivity[event + "_std"].append(
                    candidate_probability.std(-1, unbiased=False).cpu().numpy())
                sensitivity[event + "_range"].append(
                    (candidate_probability.max(-1).values
                     - candidate_probability.min(-1).values).cpu().numpy())
    fall_probability = np.concatenate(predictions["fall"])
    collision_probability = np.concatenate(predictions["collision"])
    fall_target = np.asarray(data["fall"][indices])
    collision_target = np.asarray(data["collision"][indices])
    fall_metrics = event_metrics(fall_probability, fall_target)
    collision_metrics = event_metrics(collision_probability, collision_target)
    fall_metrics["platt"] = fit_platt(fall_probability, fall_target)
    collision_metrics["platt"] = fit_platt(
        collision_probability, collision_target)
    result = {
        "checkpoint": str(args.checkpoint), "dataset": str(args.dataset),
        "samples": len(indices), "held_out_environments": len(val_envs),
        "fall": fall_metrics,
        "collision": collision_metrics,
        "candidate_sensitivity": {
            name: {"mean": float(np.concatenate(value).mean()),
                   "p90": float(np.quantile(np.concatenate(value), 0.90))}
            for name, value in sensitivity.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
