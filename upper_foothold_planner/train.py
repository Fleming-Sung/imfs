"""Train the upper latent world model and CEM planner on frozen lower control."""

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from isaacgym import gymutil  # must precede torch
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.cem import plan
from upper_planner.depth_diagnostics import (depth_prediction_sequence,
                                             save_depth_prediction)
from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import make_world_model
from upper_planner.world_model_trainer import WorldModelTrainer
from upper_planner.task_world_model_trainer import TaskWorldModelTrainer
from upper_planner.option_world_model_trainer import OptionWorldModelTrainer


def arguments():
    custom = [
        {"name": "--num_envs", "type": int, "default": 64},
        {"name": "--seed", "type": int, "default": 101},
        {"name": "--max_updates", "type": int, "default": 10000},
        {"name": "--warmup_transitions", "type": int, "default": 2048},
        {"name": "--pretrain_updates", "type": int, "default": 500},
        {"name": "--batch_size", "type": int, "default": 128},
        {"name": "--replay_capacity", "type": int, "default": 50000},
        {"name": "--updates_per_transition", "type": float, "default": 1.0},
        {"name": "--learning_rate", "type": float, "default": 3e-4},
        {"name": "--depth_coef", "type": float, "default": 0.25},
        {"name": "--future_depth_coef", "type": float, "default": 1.0},
        {"name": "--checkpoint_interval", "type": int, "default": 1000},
        {"name": "--log_interval", "type": int, "default": 100},
        {"name": "--depth_image_interval", "type": int, "default": 1000},
        {"name": "--cem_candidates", "type": int, "default": 128},
        {"name": "--cem_elites", "type": int, "default": 16},
        {"name": "--cem_iterations", "type": int, "default": 3},
        {"name": "--planning_horizon", "type": int, "default": 2},
        {"name": "--sequence_horizon", "type": int, "default": 1},
        {"name": "--temporal_decay", "type": float, "default": 0.8},
        {"name": "--balanced_events", "action": "store_true"},
        {"name": "--balanced_replay", "action": "store_true"},
        {"name": "--max_event_pos_weight", "type": float, "default": 20.0},
        {"name": "--safety_heads_only", "action": "store_true"},
        {"name": "--course_length_m", "type": float, "default": 2.25},
        {"name": "--terrain_curriculum", "type": str, "default": "obstacles"},
        {"name": "--research_kind", "type": str, "default": "mixed"},
        {"name": "--reset_curriculum_prob", "type": float, "default": 0.5},
        {"name": "--typical_kind", "type": str, "default": "mixed"},
        {"name": "--bridge_width_min_m", "type": float, "default": 0.55},
        {"name": "--bridge_width_max_m", "type": float, "default": 0.75},
        {"name": "--irregular_width_m", "type": float, "default": 0.65},
        {"name": "--hurdle_height_min_m", "type": float, "default": 0.035},
        {"name": "--hurdle_height_max_m", "type": float, "default": 0.085},
        {"name": "--random_width_min_m", "type": float, "default": 0.55},
        {"name": "--random_width_max_m", "type": float, "default": 1.20},
        {"name": "--random_gap_min_m", "type": float, "default": 0.00},
        {"name": "--random_gap_max_m", "type": float, "default": 0.10},
        {"name": "--random_obstacle_probability", "type": float, "default": 0.45},
        {"name": "--obstacle_count", "type": int, "default": 1},
        {"name": "--reward_progress", "type": float, "default": 10.0},
        {"name": "--reward_collision", "type": float, "default": -2.0},
        {"name": "--collision_risk", "type": float, "default": 0.2},
        {"name": "--fall_risk", "type": float, "default": 1.0},
        {"name": "--collision_force_risk", "type": float, "default": 0.0},
        {"name": "--stability_risk", "type": float, "default": 0.0},
        {"name": "--support_risk", "type": float, "default": 0.0},
        {"name": "--touchdown_risk", "type": float, "default": 0.0},
        {"name": "--action_l2", "type": float, "default": 0.0},
        {"name": "--terminal_value_coef", "type": float, "default": 0.0},
        {"name": "--planning_disable_goal", "action": "store_true"},
        {"name": "--planning_base_safety_scale", "type": float, "default": 1.0},
        {"name": "--decomposed_reward", "action": "store_true"},
        {"name": "--action_profile", "type": str, "default": "polar"},
        {"name": "--collection_policy", "type": str, "default": "cem"},
        {"name": "--mixed_planner_fraction", "type": float, "default": 0.60},
        {"name": "--mixed_perturb_fraction", "type": float, "default": 0.20},
        {"name": "--policy_coef", "type": float, "default": 0.1},
        {"name": "--use_policy_prior", "action": "store_true"},
        {"name": "--policy_rollout", "action": "store_true"},
        {"name": "--policy_std", "type": float, "default": 0.3},
        {"name": "--policy_q_coef", "type": float, "default": 0.0},
        {"name": "--policy_safe_only", "action": "store_true"},
        {"name": "--model_variant", "type": str, "default": "compact"},
        {"name": "--output", "type": str},
        {"name": "--init_checkpoint", "type": str},
    ]
    args = gymutil.parse_arguments(
        description="upper foothold world-model training", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def cpu(value):
    return value.detach().cpu().numpy().copy()


def save_checkpoint(path, model, trainer, rollout, replay, args, cfg):
    torch.save({
        "format_version": 1,
        "model": model.state_dict(), "target": trainer.target.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "updates": trainer.updates, "lower_ticks": rollout.lower_ticks,
        "replay_size": replay.size, "args": vars(args), "config": cfg,
    }, path)


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output) if args.output else (
        ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_seed{:03d}".format(args.seed)))
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "state_windows").mkdir(exist_ok=True)
    (output / "depth_predictions").mkdir(exist_ok=True)
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    writer = SummaryWriter(str(output / "tensorboard"))

    env, lower, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=2.5,
        randomization=True, cameras=True,
        flat_plane=args.terrain_curriculum == "obstacles",
        obstacles=args.terrain_curriculum == "obstacles",
        course_length_m=args.course_length_m, obstacle_count=args.obstacle_count,
        reward_override={"progress": args.reward_progress,
                         "collision": args.reward_collision})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        args.replay_capacity, num_envs=args.num_envs,
        return_horizon=cfg["model"]["planning_horizon"],
        gamma=cfg["model"]["discount"], reward_scale=cfg["model"]["reward_scale"],
        duration_aware_returns=args.model_variant == "option")
    model = make_world_model(
        latent_dim=cfg["upper_observation"]["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        variant=args.model_variant).to(env.device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=env.device)
        init_load = model.load_state_dict(initial["model"], strict=False)
        print(json.dumps({
            "event": "initial_checkpoint_loaded",
            "missing": list(init_load.missing_keys),
            "unexpected": list(init_load.unexpected_keys),
        }), flush=True)
    if args.safety_heads_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for head in (model.collision, model.fall, model.off_support):
            for parameter in head.parameters():
                parameter.requires_grad_(True)
    if args.model_variant == "option":
        if args.safety_heads_only:
            raise ValueError("safety_heads_only is not supported by the option model")
        trainer = OptionWorldModelTrainer(
            model, learning_rate=args.learning_rate,
            gamma=cfg["model"]["discount"],
            reward_scale=cfg["model"]["reward_scale"],
            policy_coef=args.policy_coef,
            policy_q_coef=args.policy_q_coef,
            policy_safe_only=args.policy_safe_only,
            max_event_pos_weight=max(args.max_event_pos_weight, 50.0))
    elif args.model_variant == "task":
        if args.safety_heads_only:
            raise ValueError("safety_heads_only is not supported by the task model")
        trainer = TaskWorldModelTrainer(
            model, learning_rate=args.learning_rate,
            gamma=cfg["model"]["discount"],
            reward_scale=cfg["model"]["reward_scale"],
            balanced_events=True,
            max_event_pos_weight=args.max_event_pos_weight)
    else:
        trainer = WorldModelTrainer(
            model, learning_rate=args.learning_rate,
            gamma=cfg["model"]["discount"],
            depth_coef=args.depth_coef, future_depth_coef=args.future_depth_coef,
            reward_scale=cfg["model"]["reward_scale"],
            balanced_events=args.balanced_events,
            max_event_pos_weight=args.max_event_pos_weight)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1)
    update_budget = 0.0
    pretrain_complete = False
    last_checkpoint = 0
    last_log_update = 0
    last_depth_image = 0
    start = time.perf_counter()
    total_transitions = 0
    accum = {name: 0.0 for name in (
        "reward", "progress", "time", "goal", "collision", "fall", "off_support",
        "distance", "done", "transitions")}
    terrain_kinds = tuple(layout.spec.kind for layout in tiled.layouts)
    scene_names = tuple(sorted(set(terrain_kinds)))
    scene_lookup = {name: index for index, name in enumerate(scene_names)}
    scene_ids = torch.as_tensor(
        [scene_lookup[name] for name in terrain_kinds],
        dtype=torch.long, device=env.device)
    kind_accum = {kind: {name: 0.0 for name in (
        "transitions", "reward", "distance", "done", "collision", "fall",
        "off_support", "goal")}
                  for kind in sorted(set(terrain_kinds))}
    recent_losses = {}
    state_window = {name: deque(maxlen=500) for name in (
        "root", "rigid_body", "contact_force", "dof_pos", "dof_vel", "foot_pos",
        "target_pos", "upper_action")}

    def enough_training_data():
        if replay.valid_size < max(args.batch_size, args.warmup_transitions):
            return False
        if args.sequence_horizon <= 1:
            return True
        rows = (replay.padded_sequence_indices(args.sequence_horizon)[0]
                if args.model_variant == "option"
                else replay.sequence_indices(args.sequence_horizon))
        return len(rows) >= args.batch_size

    def train_step():
        if args.sequence_horizon <= 1:
            batch = replay.sample(args.batch_size, env.device)
            return trainer.train_step(batch)
        batch = replay.sample_sequence(
            args.batch_size, args.sequence_horizon, env.device,
            balanced=(args.balanced_replay or args.model_variant == "option"),
            allow_terminal_padding=args.model_variant == "option")
        return trainer.train_step_sequence(batch, args.temporal_decay)

    def stratified_actions(ids):
        templates = torch.tensor([
            [0.0, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.8, 0.0, 0.0],
            [0.0, -0.8, 0.0], [0.0, 0.8, 0.0],
            [0.0, -0.4, -0.8], [0.0, 0.4, 0.8], [0.8, 0.8, 0.0],
        ], dtype=torch.float32, device=env.device)
        actions = templates[ids.remainder(len(templates))]
        noise = 0.12 * torch.randn(
            actions.shape, generator=generator, device=env.device)
        actions = (actions + noise).clamp(-1.0, 1.0)
        uniform = 2.0 * torch.rand(
            actions.shape, generator=generator, device=env.device) - 1.0
        random_mask = torch.rand(
            actions.shape[0], 1, generator=generator,
            device=env.device) < 0.20
        return torch.where(random_mask, uniform, actions)

    @torch.no_grad()
    def choose_actions(depth, proprio, ids):
        if args.collection_policy not in ("random", "stratified", "cem", "mixed"):
            raise ValueError(
                "collection_policy must be random, stratified, cem, or mixed")
        if (args.collection_policy == "random"
                or replay.valid_size < args.warmup_transitions):
            actions = 2.0 * torch.rand(
                depth.shape[0], 3, generator=generator, device=env.device) - 1.0
            return actions, torch.zeros(
                depth.shape[0], dtype=torch.bool, device=env.device)
        if args.collection_policy == "stratified":
            return stratified_actions(ids), torch.zeros(
                ids.shape[0], dtype=torch.bool, device=env.device)
        model.eval()
        latent = model.encode(depth, proprio)
        planner_reward_cfg = dict(cfg["reward"])
        for name in ("collision", "fall", "off_support"):
            planner_reward_cfg[name] *= args.planning_base_safety_scale
        if args.planning_disable_goal:
            planner_reward_cfg["goal"] = 0.0
        planned, _ = plan(
            model, latent, horizon=args.planning_horizon,
            candidates=args.cem_candidates, elites=args.cem_elites,
            iterations=args.cem_iterations, discount=cfg["model"]["discount"],
            min_std=cfg["model"]["cem_min_std"],
            collision_risk=args.collision_risk, fall_risk=args.fall_risk,
            collision_force_risk=args.collision_force_risk,
            stability_risk=args.stability_risk,
            support_risk=args.support_risk,
            touchdown_risk=args.touchdown_risk,
            action_l2=args.action_l2,
            terminal_value_coef=args.terminal_value_coef,
            reward_cfg=(planner_reward_cfg
                        if args.decomposed_reward
                        or args.model_variant in ("task", "option")
                        else None),
            reward_scale=cfg["model"]["reward_scale"],
            policy=(model.policy_action if args.use_policy_prior else None),
            policy_rollout=args.policy_rollout,
            policy_std=args.policy_std)
        if planned.ndim == 1:
            planned = planned.unsqueeze(0)
        if args.collection_policy == "mixed":
            planner_fraction = float(args.mixed_planner_fraction)
            perturb_fraction = float(args.mixed_perturb_fraction)
            if (planner_fraction < 0.0 or perturb_fraction < 0.0
                    or planner_fraction + perturb_fraction > 1.0):
                raise ValueError(
                    "mixed collection fractions must be nonnegative and sum <= 1")
            perturbed = (planned + 0.18 * torch.randn(
                planned.shape, generator=generator,
                device=env.device)).clamp(-1.0, 1.0)
            coverage = stratified_actions(ids)
            selector = torch.rand(
                planned.shape[0], 1, generator=generator, device=env.device)
            actions = torch.where(
                selector < planner_fraction, planned,
                torch.where(selector < planner_fraction + perturb_fraction,
                            perturbed, coverage))
            return actions, (selector < planner_fraction).squeeze(1)
        epsilon = max(0.10, 0.50 - 0.40 * trainer.updates / max(args.max_updates, 1))
        random = 2.0 * torch.rand(
            depth.shape[0], 3, generator=generator, device=env.device) - 1.0
        mask = torch.rand(depth.shape[0], 1, generator=generator, device=env.device) < epsilon
        actions = torch.where(mask, random, planned)
        return actions, ~mask.squeeze(1)

    print(json.dumps({
        "event": "training_started", "output": str(output),
        "num_envs": args.num_envs, "warmup_transitions": args.warmup_transitions,
        "max_updates": args.max_updates, "init_checkpoint": args.init_checkpoint,
    }))
    while trainer.updates < args.max_updates:
        transition = rollout.lower_tick(choose_actions)
        state_window["root"].append(cpu(env.root_states[0, 0]))
        state_window["rigid_body"].append(cpu(env.rigid_body_states[0]))
        state_window["contact_force"].append(cpu(env.contact_forces[0]))
        state_window["dof_pos"].append(cpu(env.dof_pos[0]))
        state_window["dof_vel"].append(cpu(env.dof_vel[0]))
        state_window["foot_pos"].append(cpu(env.foot_positions[0]))
        state_window["target_pos"].append(cpu(env.sampler.target_pos[0]))
        state_window["upper_action"].append(cpu(rollout.previous_action[0]))
        if transition is None:
            continue

        count = transition["ids"].numel()
        total_transitions += count
        transition["scene_id"] = scene_ids[transition["ids"]]
        replay.add_transition_batch(transition)
        if pretrain_complete:
            update_budget += replay.last_new_valid * args.updates_per_transition
        accum["reward"] += float(transition["reward"].sum())
        accum["distance"] += float(transition["diagnostics"]["distance_to_goal_m"].sum())
        accum["done"] += float(transition["done"].float().sum())
        accum["transitions"] += count
        for name, value in transition["terms"].items():
            accum[name] += float(value.sum())
        for kind, values in kind_accum.items():
            mask = torch.as_tensor(
                [terrain_kinds[int(env_id)] == kind for env_id in transition["ids"]],
                dtype=torch.bool, device=env.device)
            if not mask.any():
                continue
            values["transitions"] += int(mask.sum())
            values["reward"] += float(transition["reward"][mask].sum())
            values["distance"] += float(
                transition["diagnostics"]["distance_to_goal_m"][mask].sum())
            values["done"] += int(transition["done"][mask].sum())
            for event in ("collision", "fall", "off_support", "success"):
                target = "goal" if event == "success" else event
                values[target] += int(transition["diagnostics"][event][mask].sum())

        if (not pretrain_complete and enough_training_data()
                and trainer.updates < args.max_updates):
            target_updates = min(args.pretrain_updates, args.max_updates)
            while trainer.updates < target_updates:
                recent_losses = train_step()
            pretrain_complete = True
            print(json.dumps({
                "event": "pretrain_complete", "updates": trainer.updates,
                "replay_size": replay.size,
            }), flush=True)

        while (update_budget >= 1.0 and enough_training_data()
               and trainer.updates < args.max_updates):
            recent_losses = train_step()
            update_budget -= 1.0

        if trainer.updates and trainer.updates - last_log_update >= args.log_interval:
            n = max(accum["transitions"], 1.0)
            elapsed = time.perf_counter() - start
            log = {
                "event": "train", "updates": trainer.updates,
                "replay_size": replay.size, "lower_ticks": rollout.lower_ticks,
                "sequence_horizon": args.sequence_horizon,
                "valid_sequences": (len(replay.sequence_indices(args.sequence_horizon))
                                    if args.sequence_horizon > 1
                                    and args.model_variant != "option"
                                    else (len(replay.padded_sequence_indices(
                                        args.sequence_horizon)[0])
                                          if args.sequence_horizon > 1
                                          else replay.valid_size)),
                "transitions": int(accum["transitions"]),
                "reward_mean": accum["reward"] / n,
                "distance_mean_m": accum["distance"] / n,
                "done_fraction": accum["done"] / n,
                "transitions_per_second": total_transitions / elapsed,
                "total_transitions": total_transitions,
            }
            for name in ("progress", "time", "goal", "collision", "fall", "off_support"):
                log["reward_" + name] = accum[name] / n
            log.update(recent_losses)
            for kind, values in kind_accum.items():
                kind_n = max(values["transitions"], 1.0)
                for metric in ("reward", "distance", "done", "collision", "fall",
                               "off_support", "goal"):
                    log["terrain/{}/{}".format(kind, metric)] = values[metric] / kind_n
            print(json.dumps(log), flush=True)
            for name, value in log.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(name, value, trainer.updates)
            writer.add_histogram("action/all", transition["action"], trainer.updates)
            for key in accum:
                accum[key] = 0.0
            for values in kind_accum.values():
                for key in values:
                    values[key] = 0.0
            last_log_update = trainer.updates

        if (trainer.updates and hasattr(model, "reconstruct_depth")
                and args.depth_image_interval > 0
                and trainer.updates - last_depth_image >= args.depth_image_interval
                and enough_training_data()):
            if args.sequence_horizon > 1:
                depth_batch = replay.sample_sequence(
                    1, args.sequence_horizon, env.device)
            else:
                depth_batch = replay.sample(1, env.device)
            model.eval()
            arrays = depth_prediction_sequence(model, depth_batch)
            image_path = output / "depth_predictions" / "update_{:06d}.png".format(
                trainer.updates)
            depth_summary = save_depth_prediction(image_path, arrays)
            for horizon, mae in enumerate(depth_summary["mae_by_horizon"]):
                writer.add_scalar("depth_prediction/mae_h{}".format(horizon),
                                  mae, trainer.updates)
            print(json.dumps({
                "event": "depth_prediction", "updates": trainer.updates,
                "path": str(image_path), **depth_summary,
            }), flush=True)
            last_depth_image = trainer.updates

        if trainer.updates and trainer.updates - last_checkpoint >= args.checkpoint_interval:
            checkpoint = output / "checkpoints" / "model_{:06d}.pt".format(trainer.updates)
            save_checkpoint(checkpoint, model, trainer, rollout, replay, args, cfg)
            np.savez_compressed(
                output / "state_windows" / "window_{:06d}.npz".format(trainer.updates),
                **{name: np.stack(values) for name, values in state_window.items()})
            writer.flush()
            print(json.dumps({
                "event": "checkpoint", "updates": trainer.updates,
                "path": str(checkpoint), "replay_size": replay.size,
            }), flush=True)
            last_checkpoint = trainer.updates

    if trainer.updates != last_checkpoint:
        checkpoint = output / "checkpoints" / "model_{:06d}.pt".format(trainer.updates)
        save_checkpoint(checkpoint, model, trainer, rollout, replay, args, cfg)
        np.savez_compressed(
            output / "state_windows" / "window_{:06d}.npz".format(trainer.updates),
            **{name: np.stack(values) for name, values in state_window.items()})
        print(json.dumps({
            "event": "final_checkpoint", "updates": trainer.updates,
            "path": str(checkpoint), "replay_size": replay.size,
        }), flush=True)
    writer.flush()
    writer.close()
    print(json.dumps({"event": "training_complete", "updates": trainer.updates,
                      "output": str(output)}))


if __name__ == "__main__":
    main()
