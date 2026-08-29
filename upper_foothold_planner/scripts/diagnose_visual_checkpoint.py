"""Counterfactual visual diagnostic on one fixed set of real transitions."""

import json
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import LatentWorldModel


def arguments():
    custom = [
        {"name": "--checkpoint", "type": str, "required": True},
        {"name": "--num_envs", "type": int, "default": 32},
        {"name": "--transitions", "type": int, "default": 1024},
        {"name": "--seed", "type": int, "default": 401},
        {"name": "--output", "type": str, "required": True},
        {"name": "--action_profile", "type": str, "default": "polar"},
    ]
    args = gymutil.parse_arguments(
        description="fixed-transition depth counterfactual", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def balanced_accuracy(prediction, target):
    positive = target > 0.5
    negative = ~positive
    tpr = (prediction[positive] == positive[positive]).float().mean()
    tnr = (prediction[negative] == positive[negative]).float().mean()
    return float(0.5 * (tpr + tnr))


@torch.no_grad()
def evaluate(model, depth, proprio, action, reward, progress, collision, fall,
             success, reward_cfg, reward_scale):
    latent = model.encode(depth, proprio)
    component = model.predict_task_components(latent, action)
    collision_probability = torch.sigmoid(component["collision_logit"])
    fall_probability = torch.sigmoid(component["fall_logit"])
    predicted_reward = model.predict_task_reward(
        latent, action, reward_cfg, reward_scale)
    collision_prediction = collision_probability > 0.5
    fall_prediction = fall_probability > 0.5
    positive = collision > 0.5
    return {
        "decomposed_reward_mae_physical": float(
            (predicted_reward - reward / reward_scale).abs().mean() * reward_scale),
        "progress_mae_physical": float(
            (component["progress"] - progress / reward_scale).abs().mean() * reward_scale),
        "collision_bce": float(F.binary_cross_entropy(collision_probability, collision)),
        "collision_accuracy": float((collision_prediction == positive).float().mean()),
        "collision_balanced_accuracy": balanced_accuracy(collision_prediction, collision),
        "collision_probability_positive": float(collision_probability[positive].mean()),
        "collision_probability_negative": float(collision_probability[~positive].mean()),
        "fall_bce": float(F.binary_cross_entropy(fall_probability, fall)),
        "fall_balanced_accuracy": balanced_accuracy(fall_prediction, fall),
        "goal_bce": float(F.binary_cross_entropy_with_logits(
            component["goal_logit"], success)),
        "depth_latent_abs_mean": float(latent.abs().mean()),
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=2.5,
        randomization=True, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=2.25, obstacle_count=1,
        reward_override={"progress": 10.0, "collision": -2.0})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        max(2048, args.transitions * 2), num_envs=args.num_envs,
        return_horizon=cfg["model"]["planning_horizon"],
        gamma=cfg["model"]["discount"], reward_scale=cfg["model"]["reward_scale"])
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1)

    def random_actions(depth, proprio, ids):
        del depth, proprio
        return 2.0 * torch.rand(
            ids.numel(), 3, generator=generator, device=env.device) - 1.0

    while replay.valid_size < args.transitions:
        replay.add_transition_batch(rollout.lower_tick(random_actions))

    checkpoint = torch.load(args.checkpoint, map_location=env.device)
    model = LatentWorldModel(
        latent_dim=cfg["upper_observation"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"]).to(env.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    index = replay.valid_indices[:args.transitions]
    tensor = lambda array: torch.as_tensor(array[index], dtype=torch.float32, device=env.device)
    depth = tensor(replay.depth) / 255.0
    values = {
        "proprio": tensor(replay.proprio), "action": tensor(replay.action),
        "reward": tensor(replay.reward), "progress": tensor(replay.progress),
        "collision": tensor(replay.collision), "fall": tensor(replay.fall),
        "success": tensor(replay.success),
    }
    permutation = torch.randperm(len(index), device=env.device)
    results = {}
    for name, image in (
            ("normal", depth), ("shuffled", depth[permutation]),
            ("zero", torch.zeros_like(depth))):
        results[name] = evaluate(
            model, image, reward_cfg=cfg["reward"],
            reward_scale=cfg["model"]["reward_scale"], **values)
    summary = {
        "checkpoint": args.checkpoint, "seed": args.seed,
        "transitions": len(index), "lower_ticks": rollout.lower_ticks,
        "collision_fraction": float(values["collision"].mean()),
        "fall_fraction": float(values["fall"].mean()),
        "success_fraction": float(values["success"].mean()),
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
