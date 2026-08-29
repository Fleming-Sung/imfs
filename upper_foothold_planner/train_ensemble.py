"""Train independent bootstrap world models under one frozen behavior policy."""

import json
import sys
import time
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
from upper_planner.ensemble import make_ensemble
from upper_planner.factory import create_upper_system
from upper_planner.replay import ReplayBuffer
from upper_planner.rollout import UpperRollout
from upper_planner.world_model import LatentWorldModel
from upper_planner.world_model_trainer import WorldModelTrainer


def arguments():
    custom = [
        {"name": "--num_envs", "type": int, "default": 64},
        {"name": "--seed", "type": int, "default": 113},
        {"name": "--ensemble_size", "type": int, "default": 3},
        {"name": "--max_updates", "type": int, "default": 600},
        {"name": "--pretrain_updates", "type": int, "default": 300},
        {"name": "--warmup_transitions", "type": int, "default": 2048},
        {"name": "--batch_size", "type": int, "default": 128},
        {"name": "--replay_capacity", "type": int, "default": 50000},
        {"name": "--updates_per_transition", "type": float, "default": 0.5},
        {"name": "--max_lower_ticks", "type": int, "default": 30000},
        {"name": "--log_interval", "type": int, "default": 50},
        {"name": "--checkpoint_interval", "type": int, "default": 300},
        {"name": "--behavior_checkpoint", "type": str, "required": True},
        {"name": "--init_checkpoint", "type": str},
        {"name": "--course_length_m", "type": float, "default": 2.25},
        {"name": "--obstacle_count", "type": int, "default": 1},
        {"name": "--cem_candidates", "type": int, "default": 128},
        {"name": "--cem_elites", "type": int, "default": 16},
        {"name": "--cem_iterations", "type": int, "default": 3},
        {"name": "--planning_horizon", "type": int, "default": 5},
        {"name": "--action_profile", "type": str, "default": "polar"},
        {"name": "--output", "type": str},
    ]
    args = gymutil.parse_arguments(
        description="bootstrap upper world-model ensemble", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def save(path, models, trainers, rollout, replay, args, cfg):
    torch.save({
        "format_version": 2,
        "models": [model.state_dict() for model in models],
        "targets": [trainer.target.state_dict() for trainer in trainers],
        "optimizers": [trainer.optimizer.state_dict() for trainer in trainers],
        "updates": trainers[0].updates,
        "lower_ticks": rollout.lower_ticks,
        "replay_size": replay.size,
        "args": vars(args),
        "config": cfg,
    }, path)


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output) if args.output else (
        ROOT / "runs" / datetime.now().strftime(
            "%Y%m%d_%H%M%S_ensemble_seed{:03d}".format(args.seed)))
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str))
    writer = SummaryWriter(str(output / "tensorboard"))

    env, lower, interface, task, _, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, corridor_width_m=2.5,
        randomization=True, cameras=True, flat_plane=True, obstacles=True,
        course_length_m=args.course_length_m, obstacle_count=args.obstacle_count,
        reward_override={"progress": 10.0, "collision": -2.0})
    rollout = UpperRollout(env, lower, interface, task, cfg["depth"])
    replay = ReplayBuffer(
        args.replay_capacity, num_envs=args.num_envs,
        return_horizon=args.planning_horizon,
        gamma=cfg["model"]["discount"], reward_scale=cfg["model"]["reward_scale"])

    behavior_state = torch.load(args.behavior_checkpoint, map_location=env.device)
    behavior = LatentWorldModel(
        cfg["upper_observation"]["latent_dim"], cfg["model"]["hidden_dim"]).to(env.device)
    behavior.load_state_dict(behavior_state["model"])
    behavior.eval()
    initial_state = None
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=env.device)
        initial_state = initial.get("model", initial.get("models", [None])[0])
    models = make_ensemble(
        args.ensemble_size, cfg["upper_observation"]["latent_dim"],
        cfg["model"]["hidden_dim"], env.device, initial_state)
    trainers = [WorldModelTrainer(
        model, gamma=cfg["model"]["discount"],
        reward_scale=cfg["model"]["reward_scale"]) for model in models]
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 1000)

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
        random = 2.0 * torch.rand(
            depth.shape[0], 3, generator=generator, device=env.device) - 1.0
        explore = torch.rand(
            depth.shape[0], 1, generator=generator, device=env.device) < 0.20
        return torch.where(explore, random, action)

    def update_all():
        member_metrics = []
        # Independent bootstrap batches are the only source of member diversity
        # when all members are initialized from the same known-good checkpoint.
        for trainer in trainers:
            member_metrics.append(trainer.train_step(
                replay.sample(args.batch_size, env.device)))
        return member_metrics

    start = time.perf_counter()
    update_budget = 0.0
    pretrain_complete = False
    last_log = last_checkpoint = 0
    recent = []
    events = {"transitions": 0, "success": 0, "fall": 0, "collision": 0, "done": 0}
    print(json.dumps({
        "event": "ensemble_training_started", "output": str(output),
        "ensemble_size": args.ensemble_size,
        "behavior_checkpoint": args.behavior_checkpoint,
        "init_checkpoint": args.init_checkpoint,
    }), flush=True)

    while trainers[0].updates < args.max_updates:
        if rollout.lower_ticks >= args.max_lower_ticks:
            raise RuntimeError("max_lower_ticks reached before requested ensemble updates")
        transition = rollout.lower_tick(choose_actions)
        if transition is None:
            continue
        replay.add_transition_batch(transition)
        count = int(transition["ids"].numel())
        events["transitions"] += count
        events["success"] += int(transition["diagnostics"]["success"].sum())
        events["fall"] += int(transition["diagnostics"]["fall"].sum())
        events["collision"] += int(transition["diagnostics"]["collision"].sum())
        events["done"] += int(transition["done"].sum())
        if pretrain_complete:
            update_budget += replay.last_new_valid * args.updates_per_transition

        if not pretrain_complete and replay.valid_size >= args.warmup_transitions:
            target = min(args.pretrain_updates, args.max_updates)
            while trainers[0].updates < target:
                recent = update_all()
            pretrain_complete = True
            print(json.dumps({"event": "pretrain_complete", "updates": target,
                              "replay_valid_size": replay.valid_size}), flush=True)

        while (pretrain_complete and update_budget >= 1.0
               and trainers[0].updates < args.max_updates):
            recent = update_all()
            update_budget -= 1.0

        updates = trainers[0].updates
        if updates and updates - last_log >= args.log_interval:
            keys = recent[0].keys()
            mean_metrics = {key: float(np.mean([item[key] for item in recent])) for key in keys}
            spread_metrics = {key: float(np.std([item[key] for item in recent])) for key in keys}
            log = {
                "event": "ensemble_train", "updates": updates,
                "lower_ticks": rollout.lower_ticks, "replay_size": replay.size,
                "replay_valid_size": replay.valid_size,
                "wall_seconds": time.perf_counter() - start,
                **events,
                **{"mean/" + key: value for key, value in mean_metrics.items()},
                **{"members_std/" + key: value for key, value in spread_metrics.items()},
            }
            print(json.dumps(log), flush=True)
            for key, value in log.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, updates)
            events = {key: 0 for key in events}
            last_log = updates

        if updates and updates - last_checkpoint >= args.checkpoint_interval:
            path = output / "checkpoints" / "ensemble_{:06d}.pt".format(updates)
            save(path, models, trainers, rollout, replay, args, cfg)
            writer.flush()
            print(json.dumps({"event": "checkpoint", "path": str(path),
                              "updates": updates}), flush=True)
            last_checkpoint = updates

    if trainers[0].updates != last_checkpoint:
        path = output / "checkpoints" / "ensemble_{:06d}.pt".format(trainers[0].updates)
        save(path, models, trainers, rollout, replay, args, cfg)
    writer.close()
    print(json.dumps({"event": "ensemble_training_complete",
                      "updates": trainers[0].updates, "output": str(output)}))


if __name__ == "__main__":
    main()
