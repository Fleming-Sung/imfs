"""Behaviour-cloning pre-training for the candidate foothold Actor (Gate B).

Trains CandidateActor on a teacher dataset produced by collect_teacher_dataset.py:

    L = CE(student_logits, teacher_candidate_index)
      + BCE(candidate_feasible, candidate_valid)
      + Huber(candidate_progress, geodesic_progress)

The train/validation split is by terrain instance (env_id) so that validation
measures generalisation to unseen terrains rather than replay memorisation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upper_planner.candidate_actor import CandidateActor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--gru_hidden", type=int, default=0)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--huber_weight", type=float, default=0.5)
    parser.add_argument("--success_only", action="store_true",
                        help="train only on decisions from successful episodes")
    parser.add_argument("--soft_target", action="store_true",
                        help="distill the full teacher score distribution "
                             "instead of the one-hot argmax")
    parser.add_argument("--score_temperature", type=float, default=0.25)
    parser.add_argument("--val_env_fraction", type=float, default=0.2,
                        help="fraction of terrain instances held out for val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_dataset(path, success_only, val_env_fraction, seed, device):
    data = np.load(path)
    depth = data["depth"].astype(np.float32) / 255.0  # back to [0,1]
    proprio = data["proprio"].astype(np.float32)
    candidate_index = data["candidate_index"].astype(np.int64)
    candidate_valid = data["candidate_valid"].astype(np.float32)
    candidate_progress = data["candidate_progress"].astype(np.float32)
    candidate_score = (data["candidate_score"].astype(np.float32)
                       if "candidate_score" in data.files else None)
    success = data["success"].astype(bool)
    env_id = data["env_id"].astype(np.int64) if "env_id" in data.files else None

    mask = success if success_only else np.ones(len(depth), dtype=bool)

    rng = np.random.default_rng(seed)
    if env_id is not None:
        unique = np.unique(env_id)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * val_env_fraction)))
        val_envs = set(unique[:n_val].tolist())
        is_val = np.array([e in val_envs for e in env_id])
    else:
        perm = rng.permutation(len(depth))
        n_val = int(round(len(depth) * val_env_fraction))
        is_val = np.zeros(len(depth), dtype=bool)
        is_val[perm[:n_val]] = True

    train = mask & ~is_val
    val = mask & is_val

    def to_tensor(idx):
        score = (torch.as_tensor(candidate_score[idx], dtype=torch.float32,
                                 device=device)
                 if candidate_score is not None else None)
        return (
            torch.as_tensor(depth[idx], dtype=torch.float32, device=device),
            torch.as_tensor(proprio[idx], dtype=torch.float32, device=device),
            torch.as_tensor(candidate_index[idx], dtype=torch.long, device=device),
            torch.as_tensor(candidate_valid[idx], dtype=torch.float32, device=device),
            torch.as_tensor(candidate_progress[idx], dtype=torch.float32, device=device),
            score,
        )

    num_candidates = int(candidate_index.max()) + 1
    candidates_levels = (data["candidates_levels"].astype(np.float32)
                         if "candidates_levels" in data.files else None)
    if candidates_levels is not None and candidates_levels.shape[0] != num_candidates:
        raise SystemExit("candidates_levels row count does not match num_candidates")
    return (num_candidates, candidates_levels,
            to_tensor(np.where(train)[0]), to_tensor(np.where(val)[0]))


@torch.no_grad()
def evaluate(model, depth, proprio, candidate_index, candidate_valid,
             candidate_progress, batch_size):
    model.eval()
    total = len(depth)
    top1 = top3 = 0
    valid_rate = 0
    bce_correct = 0
    progress_abs = 0.0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        (logits, feasible, progress), _ = model(
            depth[start:end], proprio[start:end])
        pred = logits.argmax(dim=-1)
        topk = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        top1 += (pred == candidate_index[start:end]).sum().item()
        top3 += (topk == candidate_index[start:end, None]).any(dim=-1).sum().item()
        valid_rate += candidate_valid[start:end][
            torch.arange(end - start, device=logits.device), pred].sum().item()
        bce_correct += ((feasible > 0) ==
                        (candidate_valid[start:end] > 0.5)).sum().item()
        progress_abs += (progress - candidate_progress[start:end]).abs().sum().item()
    n = float(total)
    return {
        "top1_acc": top1 / n,
        "top3_acc": top3 / n,
        "chosen_valid_rate": valid_rate / n,
        "feasible_acc": bce_correct / (n * candidate_valid.shape[1]),
        "progress_mae_m": progress_abs / (n * candidate_valid.shape[1]),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    num_candidates, candidates_levels, (tr_d, tr_p, tr_i, tr_v, tr_g, tr_s), (
        va_d, va_p, va_i, va_v, va_g, _) = load_dataset(
        args.dataset, args.success_only, args.val_env_fraction, args.seed, device)

    if args.soft_target and tr_s is None:
        raise SystemExit("--soft_target requires candidate_score in the dataset")

    model = CandidateActor(num_candidates, feature_dim=args.feature_dim,
                           gru_hidden=args.gru_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"train decisions {len(tr_d)}, val decisions {len(va_d)}, "
          f"candidates {num_candidates}")

    n_train = len(tr_d)
    best_top1 = -1.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            (logits, feasible, progress), _ = model(tr_d[idx], tr_p[idx])
            if args.soft_target:
                temperature = max(float(args.score_temperature), 1e-3)
                target = F.softmax(tr_s[idx] / temperature, dim=-1)
                ce = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            else:
                ce = F.cross_entropy(logits, tr_i[idx])
            bce = F.binary_cross_entropy_with_logits(feasible, tr_v[idx])
            huber = F.smooth_l1_loss(progress, tr_g[idx], beta=1.0)
            loss = (args.ce_weight * ce + args.bce_weight * bce
                    + args.huber_weight * huber)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            n_batches += 1

        metrics = evaluate(model, va_d, va_p, va_i, va_v, va_g, args.batch_size)
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / max(n_batches, 1)
        history.append(metrics)
        print(f"epoch {epoch:3d} loss {metrics['train_loss']:.4f} "
              f"top1 {metrics['top1_acc']:.3f} top3 {metrics['top3_acc']:.3f} "
              f"valid {metrics['chosen_valid_rate']:.3f} "
              f"feas {metrics['feasible_acc']:.3f} "
              f"mae {metrics['progress_mae_m']:.3f}m")
        if metrics["top1_acc"] > best_top1:
            best_top1 = metrics["top1_acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if candidates_levels is not None:
        candidates_levels = torch.as_tensor(candidates_levels)
    if best_state is not None:
        torch.save({"state_dict": best_state,
                    "num_candidates": num_candidates,
                    "candidates_levels": candidates_levels,
                    "feature_dim": args.feature_dim,
                    "gru_hidden": args.gru_hidden,
                    "config": vars(args)},
                   output / "model_best.pt")
    torch.save({"state_dict": model.state_dict(),
                "num_candidates": num_candidates,
                "candidates_levels": candidates_levels,
                "feature_dim": args.feature_dim,
                "gru_hidden": args.gru_hidden,
                "config": vars(args)},
               output / "model_last.pt")
    with open(output / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved to {output} (best top1 {best_top1:.3f})")


if __name__ == "__main__":
    main()
