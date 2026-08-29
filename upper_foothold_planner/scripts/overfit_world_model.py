"""Gate 6: collect real closed-loop data and overfit the small world model."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import LatentWorldModel
from upper_planner.world_model_trainer import WorldModelTrainer


def arguments():
    custom = [
        {"name": "--num_envs", "type": int, "default": 16},
        {"name": "--transitions", "type": int, "default": 256},
        {"name": "--updates", "type": int, "default": 500},
        {"name": "--batch_size", "type": int, "default": 64},
        {"name": "--seed", "type": int, "default": 81},
        {"name": "--course_length_m", "type": float, "default": 2.25},
        {"name": "--obstacle_count", "type": int, "default": 1},
        {"name": "--action_profile", "type": str, "default": "polar"},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "gate6_world_model_overfit")},
    ]
    args = gymutil.parse_arguments(
        description="world model small-data overfit", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


@torch.no_grad()
def evaluate(trainer, replay, indices, device, repeats=10):
    trainer.model.eval()
    values = {}
    batch_size = min(128, len(indices))
    for _ in range(repeats):
        _, metrics = trainer.losses(replay.sample(batch_size, device, indices))
        for name, value in metrics.items():
            values.setdefault(name, []).append(float(value))
    return {name: float(np.mean(items)) for name, items in values.items()}


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=2.5,
        randomization=False, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=args.course_length_m, obstacle_count=args.obstacle_count,
        reward_override={"progress": 10.0, "collision": -2.0})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        max(1024, args.transitions * 2), num_envs=args.num_envs,
        return_horizon=cfg["model"]["planning_horizon"],
        gamma=cfg["model"]["discount"], reward_scale=cfg["model"]["reward_scale"])
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1)

    def random_actions(depth, proprio, ids):
        del depth, proprio
        return 2.0 * torch.rand(
            ids.numel(), 3, generator=generator, device=env.device) - 1.0

    collection_start = time.perf_counter()
    reward_values, fall_count, collision_count = [], 0, 0
    while replay.valid_size < args.transitions:
        transition = rollout.lower_tick(random_actions)
        if transition is not None:
            replay.add_transition_batch(transition)
            reward_values.extend(transition["reward"].detach().cpu().tolist())
            fall_count += int(transition["diagnostics"]["fall"].sum())
            collision_count += int(transition["diagnostics"]["collision"].sum())
    collection_seconds = time.perf_counter() - collection_start

    indices = np.random.permutation(replay.valid_indices)
    split = int(0.8 * len(indices))
    train_indices, test_indices = indices[:split], indices[split:]
    model = LatentWorldModel(
        latent_dim=cfg["model"]["latent_dim"] if "latent_dim" in cfg["model"] else 128,
        hidden_dim=cfg["model"]["hidden_dim"]).to(env.device)
    trainer = WorldModelTrainer(
        model, gamma=cfg["model"]["discount"],
        reward_scale=cfg["model"]["reward_scale"])
    before_train = evaluate(trainer, replay, train_indices, env.device)
    before_test = evaluate(trainer, replay, test_indices, env.device)

    snapshots = []
    train_start = time.perf_counter()
    for update in range(1, args.updates + 1):
        metrics = trainer.train_step(
            replay.sample(args.batch_size, env.device, train_indices))
        if update == 1 or update % 50 == 0 or update == args.updates:
            snapshots.append(dict(update=update, **metrics))
            print(json.dumps(snapshots[-1]))
    training_seconds = time.perf_counter() - train_start
    after_train = evaluate(trainer, replay, train_indices, env.device)
    after_test = evaluate(trainer, replay, test_indices, env.device)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "target": trainer.target.state_dict(),
        "optimizer": trainer.optimizer.state_dict(), "updates": trainer.updates,
        "config": cfg,
    }, output / "overfit_model.pt")
    summary = {
        "replay_size": replay.size, "valid_replay_size": replay.valid_size,
        "lower_ticks": rollout.lower_ticks,
        "collection_seconds": collection_seconds,
        "collection_transitions_per_second": replay.valid_size / collection_seconds,
        "reward_mean": float(np.mean(reward_values)),
        "reward_std": float(np.std(reward_values)),
        "fall_transitions": fall_count,
        "collision_transitions": collision_count,
        "done_fraction": float(replay.done[:replay.size].mean()),
        "train_size": len(train_indices), "test_size": len(test_indices),
        "updates": args.updates, "training_seconds": training_seconds,
        "before_train": before_train, "after_train": after_train,
        "before_test": before_test, "after_test": after_test,
        "snapshots": snapshots,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "snapshots"}, indent=2))


if __name__ == "__main__":
    main()
