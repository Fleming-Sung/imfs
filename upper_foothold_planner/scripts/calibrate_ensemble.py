"""Measure whether ensemble disagreement predicts one-step model error."""

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

from upper_planner.cem import plan
from upper_planner.ensemble import load_ensemble_checkpoint, one_step_predictions
from upper_planner.factory import create_upper_system
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import LatentWorldModel


def arguments():
    custom = [
        {"name": "--checkpoint", "type": str, "required": True},
        {"name": "--behavior_checkpoint", "type": str, "required": True},
        {"name": "--num_envs", "type": int, "default": 64},
        {"name": "--lower_ticks", "type": int, "default": 1000},
        {"name": "--seed", "type": int, "default": 314},
        {"name": "--course_length_m", "type": float, "default": 2.25},
        {"name": "--obstacle_count", "type": int, "default": 1},
        {"name": "--obstacle_y_m", "type": float, "default": 999.0},
        {"name": "--cem_candidates", "type": int, "default": 128},
        {"name": "--cem_elites", "type": int, "default": 16},
        {"name": "--cem_iterations", "type": int, "default": 3},
        {"name": "--planning_horizon", "type": int, "default": 5},
        {"name": "--action_profile", "type": str, "default": "polar"},
        {"name": "--output", "type": str, "required": True},
    ]
    args = gymutil.parse_arguments(
        description="ensemble calibration on held-out closed-loop transitions",
        headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def rank_correlation(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def calibration_summary(uncertainty, error):
    uncertainty = np.asarray(uncertainty)
    error = np.asarray(error)
    order = np.argsort(uncertainty)
    quarter = max(1, len(order) // 4)
    low = float(error[order[:quarter]].mean())
    high = float(error[order[-quarter:]].mean())
    return {
        "samples": int(len(error)),
        "uncertainty_mean": float(uncertainty.mean()),
        "error_mean": float(error.mean()),
        "spearman_rank": rank_correlation(uncertainty, error),
        "low_uncertainty_quartile_error": low,
        "high_uncertainty_quartile_error": high,
        "high_to_low_error_ratio": high / max(low, 1e-12),
    }


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=2.5,
        randomization=False, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=args.course_length_m, obstacle_count=args.obstacle_count,
        reward_override={"progress": 10.0, "collision": -2.0},
        obstacle_y_m=args.obstacle_y_m if abs(args.obstacle_y_m) < 900 else None)
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    models, checkpoint = load_ensemble_checkpoint(args.checkpoint, env.device)
    behavior_checkpoint = torch.load(args.behavior_checkpoint, map_location=env.device)
    behavior = LatentWorldModel(
        cfg["upper_observation"]["latent_dim"], cfg["model"]["hidden_dim"]).to(env.device)
    behavior.load_state_dict(behavior_checkpoint["model"])
    behavior.eval()

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        del ids
        latent = behavior.encode(depth, proprio)
        action, _ = plan(
            behavior, latent, horizon=args.planning_horizon,
            candidates=args.cem_candidates, elites=args.cem_elites,
            iterations=args.cem_iterations, discount=cfg["model"]["discount"],
            min_std=cfg["model"]["cem_min_std"], collision_risk=0.2,
            fall_risk=1.0, reward_cfg=cfg["reward"],
            reward_scale=cfg["model"]["reward_scale"])
        return action

    modes = ("normal", "shuffled", "zero")
    raw = {mode: {name: [] for name in (
        "reward_uncertainty", "reward_error", "depth_uncertainty", "depth_error",
        "collision_uncertainty", "collision_error", "fall_uncertainty", "fall_error",
        "goal_uncertainty", "goal_error")} for mode in modes}
    counts = {"transitions": 0, "success": 0, "fall": 0, "collision": 0, "done": 0}
    sequences = [[] for _ in range(env.num_envs)]
    with torch.no_grad():
        for _ in range(args.lower_ticks):
            transition = rollout.lower_tick(choose_actions)
            if transition is None:
                continue
            count = int(transition["ids"].numel())
            counts["transitions"] += count
            for name in ("success", "fall", "collision"):
                counts[name] += int(transition["diagnostics"][name].sum())
            counts["done"] += int(transition["done"].sum())
            for row, env_id in enumerate(transition["ids"].tolist()):
                sequences[env_id].append({
                    "depth": transition["depth"][row].cpu(),
                    "proprio": transition["proprio"][row].cpu(),
                    "action": transition["action"][row].cpu(),
                    "reward": float(transition["reward"][row]),
                    "done": bool(transition["done"][row]),
                })
            target_reward = transition["reward"] / cfg["model"]["reward_scale"]
            target_depth = F.adaptive_avg_pool2d(transition["next_depth"], (16, 16))
            targets = {
                "collision": transition["diagnostics"]["collision"].float(),
                "fall": transition["diagnostics"]["fall"].float(),
                "goal": transition["diagnostics"]["success"].float(),
            }
            for mode in modes:
                depth = transition["depth"]
                if mode == "shuffled":
                    depth = torch.roll(depth, shifts=1, dims=0)
                elif mode == "zero":
                    depth = torch.zeros_like(depth)
                prediction = one_step_predictions(
                    models, depth, transition["proprio"], transition["action"],
                    cfg["reward"], cfg["model"]["reward_scale"])
                reward_mean = prediction["reward"].mean(0)
                raw[mode]["reward_uncertainty"].extend(
                    prediction["reward"].std(0, unbiased=False).cpu().tolist())
                raw[mode]["reward_error"].extend(
                    (reward_mean - target_reward).abs().cpu().tolist())
                depth_mean = prediction["next_depth"].mean(0)
                raw[mode]["depth_uncertainty"].extend(
                    prediction["next_depth"].std(0, unbiased=False)
                    .flatten(1).mean(1).cpu().tolist())
                raw[mode]["depth_error"].extend(
                    (depth_mean - target_depth).abs().flatten(1).mean(1).cpu().tolist())
                for event in ("collision", "fall", "goal"):
                    values = prediction[event + "_probability"]
                    mean = values.mean(0)
                    raw[mode][event + "_uncertainty"].extend(
                        values.std(0, unbiased=False).cpu().tolist())
                    raw[mode][event + "_error"].extend(
                        (mean - targets[event]).abs().cpu().tolist())

    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_updates": checkpoint.get("updates"),
        "ensemble_size": len(models),
        "seed": args.seed,
        "obstacle_y_m": args.obstacle_y_m if abs(args.obstacle_y_m) < 900 else None,
        **counts,
        "modes": {},
    }
    for mode in modes:
        result["modes"][mode] = {
            quantity: calibration_summary(
                raw[mode][quantity + "_uncertainty"],
                raw[mode][quantity + "_error"])
            for quantity in ("reward", "depth", "collision", "fall", "goal")
        }
    result["multistep_return"] = {}
    discount = float(cfg["model"]["discount"])
    reward_scale = float(cfg["model"]["reward_scale"])
    for horizon in (1, 2, 3, 5):
        windows = []
        for sequence in sequences:
            for start in range(len(sequence) - horizon + 1):
                window = sequence[start:start + horizon]
                if any(item["done"] for item in window[:-1]):
                    continue
                windows.append(window)
        predicted_chunks, target_values = [], []
        for offset in range(0, len(windows), 256):
            batch_windows = windows[offset:offset + 256]
            depth = torch.stack([item[0]["depth"] for item in batch_windows]).to(env.device)
            proprio = torch.stack([item[0]["proprio"] for item in batch_windows]).to(env.device)
            actions = torch.stack([
                torch.stack([step["action"] for step in item])
                for item in batch_windows]).to(env.device)
            targets = torch.tensor([
                sum((discount ** step) * item[step]["reward"] / reward_scale
                    for step in range(horizon))
                for item in batch_windows], device=env.device)
            member_returns = []
            with torch.no_grad():
                for model in models:
                    latent = model.encode(depth, proprio)
                    total = torch.zeros(len(batch_windows), device=env.device)
                    for step in range(horizon):
                        action = actions[:, step]
                        total += (discount ** step) * model.predict_task_reward(
                            latent, action, cfg["reward"], reward_scale)
                        latent = model.next(latent, action)
                    member_returns.append(total)
            predicted_chunks.append(torch.stack(member_returns).cpu())
            target_values.append(targets.cpu())
        prediction = torch.cat(predicted_chunks, dim=1).numpy()
        target = torch.cat(target_values).numpy()
        uncertainty = prediction.std(axis=0)
        error = np.abs(prediction.mean(axis=0) - target)
        result["multistep_return"][str(horizon)] = calibration_summary(
            uncertainty, error)
    planning_horizons = (1, 3, 5)
    task_gate_passed = all(
        (result["multistep_return"][str(h)]["spearman_rank"] or -1) > 0.15
        and result["multistep_return"][str(h)]["high_to_low_error_ratio"] > 1.25
        for h in planning_horizons)
    normal = result["modes"]["normal"]
    result["calibration_gate"] = {
        "visual_dynamics_criterion": "one-step reward and depth rank correlation >0.15 and high/low error ratio >1.25",
        "visual_dynamics_passed": bool(
            (normal["reward"]["spearman_rank"] or -1) > 0.15
            and normal["reward"]["high_to_low_error_ratio"] > 1.25
            and (normal["depth"]["spearman_rank"] or -1) > 0.15
            and normal["depth"]["high_to_low_error_ratio"] > 1.25),
        "task_return_criterion": "1/3/5-step return rank correlation >0.15 and high/low error ratio >1.25",
        "task_return_passed": task_gate_passed,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        output / "calibration_raw.npz",
        **{"{}_{}".format(mode, name): np.asarray(value)
           for mode, values in raw.items() for name, value in values.items()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
