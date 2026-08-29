"""Validate macro transitions, rewards, terminations and logging before training."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from isaacgym import gymapi, gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system
from upper_planner.rollout import UpperRollout
from upper_planner.cem import plan, plan_anchored_ensemble, plan_ensemble
from upper_planner.ensemble import load_ensemble_checkpoint
from upper_planner.privileged_planner import (
    PrivilegedPlannerConfig, PrivilegedTerrainPlanner)
from upper_planner.world_model import LatentWorldModel, make_world_model


def arguments():
    custom = [
        {"name": "--num_envs", "type": int, "default": 16},
        {"name": "--lower_ticks", "type": int, "default": 300},
        {"name": "--seed", "type": int, "default": 61},
        {"name": "--corridor_width_m", "type": float, "default": 0.90},
        {"name": "--randomization", "action": "store_true"},
        {"name": "--neutral_actions", "action": "store_true"},
        {"name": "--flat_plane", "action": "store_true"},
        {"name": "--obstacles", "action": "store_true"},
        {"name": "--course_length_m", "type": float, "default": 2.25},
        {"name": "--terrain_curriculum", "type": str, "default": "obstacles"},
        {"name": "--research_kind", "type": str, "default": "mixed"},
        {"name": "--typical_kind", "type": str, "default": "mixed"},
        {"name": "--bridge_width_min_m", "type": float, "default": 0.55},
        {"name": "--bridge_width_max_m", "type": float, "default": 0.75},
        {"name": "--irregular_width_m", "type": float, "default": 0.65},
        {"name": "--hurdle_height_min_m", "type": float, "default": 0.035},
        {"name": "--hurdle_height_max_m", "type": float, "default": 0.085},
        {"name": "--random_width_min_m", "type": float, "default": 0.50},
        {"name": "--random_width_max_m", "type": float, "default": 1.30},
        {"name": "--random_gap_min_m", "type": float, "default": 0.00},
        {"name": "--random_gap_max_m", "type": float, "default": 0.14},
        {"name": "--random_obstacle_probability", "type": float, "default": 0.55},
        {"name": "--obstacle_count", "type": int, "default": 1},
        {"name": "--obstacle_y_m", "type": float, "default": 999.0,
         "help": "set below 900 to force the same obstacle y in every env"},
        {"name": "--reward_progress", "type": float, "default": 10.0},
        {"name": "--reward_collision", "type": float, "default": -2.0},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "gate4_upper_rollout_smoke")},
        {"name": "--planner_checkpoint", "type": str},
        {"name": "--ensemble_checkpoint", "type": str},
        {"name": "--anchor_checkpoint", "type": str},
        {"name": "--uncertainty_coef", "type": float, "default": 0.0},
        {"name": "--cem_candidates", "type": int, "default": 128},
        {"name": "--cem_elites", "type": int, "default": 16},
        {"name": "--cem_iterations", "type": int, "default": 3},
        {"name": "--planning_horizon", "type": int, "default": 2},
        {"name": "--depth_ablation", "type": str, "default": "none",
         "help": "none, shuffled, or zero"},
        {"name": "--collision_risk", "type": float, "default": 0.0},
        {"name": "--fall_risk", "type": float, "default": 0.0},
        {"name": "--collision_force_risk", "type": float, "default": 0.0},
        {"name": "--stability_risk", "type": float, "default": 0.0},
        {"name": "--support_risk", "type": float, "default": 0.0},
        {"name": "--touchdown_risk", "type": float, "default": 0.0},
        {"name": "--action_l2", "type": float, "default": 0.0},
        {"name": "--terminal_value_coef", "type": float, "default": 1.0},
        {"name": "--terminal_q_aggregation", "type": str, "default": "min"},
        {"name": "--freeze_latent_rollout", "action": "store_true",
         "help": "diagnostic: score every horizon step at the current latent"},
        {"name": "--future_action_mode", "type": str, "default": "free",
         "help": "free, neutral, or repeat; constrains imagined future actions"},
        {"name": "--use_policy_prior", "action": "store_true",
         "help": "warm-start CEM from the learned policy prior"},
        {"name": "--policy_rollout", "action": "store_true",
         "help": "generate imagined future actions from the policy prior"},
        {"name": "--policy_std", "type": float, "default": 0.3},
        {"name": "--planning_progress_only", "action": "store_true",
         "help": "diagnostic: CEM scores predicted progress only"},
        {"name": "--planning_add_component", "type": str, "default": "none",
         "help": "with progress-only, add one of none/goal/collision/fall/off_support"},
        {"name": "--planning_disable_goal", "action": "store_true",
         "help": "set the learned goal-probability planning weight to zero"},
        {"name": "--planning_base_safety_scale", "type": float, "default": 1.0,
         "help": "scale collision/fall/off-support terms inside task reward"},
        {"name": "--decomposed_reward", "action": "store_true"},
        {"name": "--action_profile", "type": str, "default": "legacy"},
        {"name": "--privileged_terrain_planner", "action": "store_true",
         "help": "plan from exact support/obstacle geometry without cameras"},
        {"name": "--privileged_forward_levels", "type": str,
         "default": "-1,-0.818182,-0.636364,-0.454545,-0.272727,-0.090909,0.090909,0.272727,0.454545,0.636364,0.818182,1"},
        {"name": "--privileged_lateral_levels", "type": str,
         "default": "-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1"},
        {"name": "--privileged_yaw_levels", "type": str,
         "default": "-1,0,1"},
        {"name": "--record_video", "action": "store_true"},
        {"name": "--video_fps", "type": int, "default": 25},
    ]
    args = gymutil.parse_arguments(
        description="upper transition smoke test", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    return args


def main():
    args = arguments()
    if args.depth_ablation not in ("none", "shuffled", "zero"):
        raise SystemExit("--depth_ablation must be none, shuffled, or zero")
    if args.terminal_q_aggregation not in ("min", "mean"):
        raise SystemExit("--terminal_q_aggregation must be min or mean")
    if args.future_action_mode not in ("free", "neutral", "repeat"):
        raise SystemExit("--future_action_mode must be free, neutral, or repeat")
    if args.planning_add_component not in (
            "none", "goal", "collision", "fall", "off_support"):
        raise SystemExit("invalid --planning_add_component")
    if args.planning_add_component != "none" and not args.planning_progress_only:
        raise SystemExit("--planning_add_component requires --planning_progress_only")
    if args.record_video and (args.headless or args.num_envs != 1):
        raise SystemExit("video requires viewer rendering and exactly one env")
    if args.privileged_terrain_planner:
        if args.action_profile != "cartesian_course":
            raise SystemExit(
                "privileged terrain planning requires --action_profile cartesian_course")
        if args.planner_checkpoint or args.ensemble_checkpoint:
            raise SystemExit(
                "privileged terrain planning cannot be combined with learned checkpoints")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env, policy, interface, task, tiled, cfg = create_upper_system(
        ROOT, args, args.num_envs, args.seed, args.corridor_width_m,
        randomization=args.randomization,
        cameras=not args.privileged_terrain_planner,
        flat_plane=args.flat_plane,
        obstacles=args.obstacles, course_length_m=args.course_length_m,
        obstacle_count=args.obstacle_count,
        reward_override={"progress": args.reward_progress,
                         "collision": args.reward_collision},
        obstacle_y_m=args.obstacle_y_m if abs(args.obstacle_y_m) < 900.0 else None)
    rollout = UpperRollout(
        env, policy, interface, task, cfg["depth"],
        capture_depth=not args.privileged_terrain_planner)
    def levels(text):
        return tuple(float(value) for value in text.split(","))

    privileged_config = (PrivilegedPlannerConfig(
        forward_levels=levels(args.privileged_forward_levels),
        lateral_levels=levels(args.privileged_lateral_levels),
        yaw_levels=levels(args.privileged_yaw_levels))
        if args.privileged_terrain_planner else None)
    privileged_planner = (PrivilegedTerrainPlanner(
        tiled, interface.bounds, env.device, privileged_config)
        if args.privileged_terrain_planner else None)
    privileged_diagnostics = {
        name: [] for name in (
            "candidate_valid_count", "chosen_support_fraction",
            "chosen_geodesic_progress_m", "chosen_heading_error_rad",
            "fallback")}
    planner_model = None
    ensemble_models = None
    anchor_model = None
    if args.planner_checkpoint:
        checkpoint = torch.load(args.planner_checkpoint, map_location=env.device)
        checkpoint_args = checkpoint.get("args", {})
        planner_variant = checkpoint_args.get("model_variant", "compact")
        planner_model = make_world_model(
            latent_dim=cfg["upper_observation"]["latent_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            variant=planner_variant).to(env.device)
        load_result = planner_model.load_state_dict(checkpoint["model"], strict=False)
        planner_load_missing = list(load_result.missing_keys)
        planner_load_unexpected = list(load_result.unexpected_keys)
        planner_model.eval()
    if args.ensemble_checkpoint:
        if planner_model is not None:
            raise SystemExit("choose either --planner_checkpoint or --ensemble_checkpoint")
        ensemble_models, ensemble_state = load_ensemble_checkpoint(
            args.ensemble_checkpoint, env.device)
        if args.anchor_checkpoint:
            anchor_state = torch.load(args.anchor_checkpoint, map_location=env.device)
            anchor_model = LatentWorldModel(
                latent_dim=cfg["upper_observation"]["latent_dim"],
                hidden_dim=cfg["model"]["hidden_dim"]).to(env.device)
            anchor_model.load_state_dict(anchor_state["model"])
            anchor_model.eval()
    # Model construction consumes RNG. Re-seed CEM after all checkpoints load
    # so beta ablations use identical candidate randomness.
    torch.manual_seed(args.seed + 5000)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if env.viewer is not None:
        env.gym.viewer_camera_look_at(
            env.viewer, None, gymapi.Vec3(-1.5, 1.5, 1.0), gymapi.Vec3(0.8, 0.0, 0.2))
    frame_dir = output / "frames"
    if args.record_video:
        frame_dir.mkdir(exist_ok=True)
    frame_stride = max(1, int(round(1.0 / (env.dt * args.video_fps))))
    frame_index = 0

    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 100)
    planner_reward_cfg = dict(cfg["reward"])
    if args.planning_base_safety_scale < 0.0:
        raise SystemExit("--planning_base_safety_scale must be nonnegative")
    for name in ("collision", "fall", "off_support"):
        planner_reward_cfg[name] *= args.planning_base_safety_scale
    if args.planning_disable_goal:
        planner_reward_cfg["goal"] = 0.0
    if args.planning_progress_only:
        for name in ("time", "goal", "collision", "fall", "off_support"):
            planner_reward_cfg[name] = 0.0
        if args.planning_add_component != "none":
            name = args.planning_add_component
            planner_reward_cfg[name] = cfg["reward"][name]

    def random_actions(depth, proprio, ids):
        if privileged_planner is not None:
            del depth, proprio
            actions, diagnostics = privileged_planner.plan(
                env, ids, rollout.previous_action[ids])
            for name in privileged_diagnostics:
                privileged_diagnostics[name].append(
                    diagnostics[name].detach().cpu().numpy().copy())
            return actions
        if ensemble_models is not None:
            with torch.no_grad():
                if args.depth_ablation == "shuffled":
                    depth = torch.roll(depth, shifts=1, dims=0)
                elif args.depth_ablation == "zero":
                    depth = torch.zeros_like(depth)
                latents = [model.encode(depth, proprio) for model in ensemble_models]
                kwargs = dict(
                    horizon=args.planning_horizon, candidates=args.cem_candidates,
                    elites=args.cem_elites, iterations=args.cem_iterations,
                    discount=cfg["model"]["discount"],
                    min_std=cfg["model"]["cem_min_std"],
                    collision_risk=args.collision_risk, fall_risk=args.fall_risk,
                    collision_force_risk=args.collision_force_risk,
                    stability_risk=args.stability_risk,
                    support_risk=args.support_risk,
                    touchdown_risk=args.touchdown_risk,
                    reward_cfg=(cfg["reward"]
                                if args.decomposed_reward
                                or planner_variant == "task" else None),
                    reward_scale=cfg["model"]["reward_scale"],
                    uncertainty_coef=args.uncertainty_coef,
                    action_l2=args.action_l2)
                if anchor_model is not None:
                    anchor_latent = anchor_model.encode(depth, proprio)
                    action, _ = plan_anchored_ensemble(
                        anchor_model, anchor_latent, ensemble_models, latents, **kwargs)
                else:
                    action, _ = plan_ensemble(ensemble_models, latents, **kwargs)
                return action
        if planner_model is not None:
            with torch.no_grad():
                if args.depth_ablation == "shuffled":
                    depth = torch.roll(depth, shifts=1, dims=0)
                elif args.depth_ablation == "zero":
                    depth = torch.zeros_like(depth)
                latent = planner_model.encode(depth, proprio)
                action, _ = plan(
                    planner_model, latent, horizon=args.planning_horizon,
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
                    terminal_q_aggregation=args.terminal_q_aggregation,
                    freeze_latent_rollout=args.freeze_latent_rollout,
                    future_action_mode=args.future_action_mode,
                    policy=(planner_model.policy_action
                            if args.use_policy_prior else None),
                    policy_rollout=args.policy_rollout,
                    policy_std=args.policy_std,
                    reward_cfg=(planner_reward_cfg
                                if args.decomposed_reward else None),
                    reward_scale=cfg["model"]["reward_scale"])
                return action
        del depth, proprio
        if args.neutral_actions:
            return torch.zeros(ids.numel(), 3, device=env.device)
        return 2.0 * torch.rand(
            ids.numel(), 3, generator=generator, device=env.device) - 1.0

    term_sum = {name: 0.0 for name in cfg["reward"]}
    transition_count = falls = off_support = collisions = successes = 0
    root_trace, body_trace, contact_trace, foot_trace = [], [], [], []
    dof_position_trace, dof_velocity_trace, torque_trace = [], [], []
    target_position_trace, target_yaw_trace = [], []
    phase_trace, swing_foot_trace, lower_action_trace, upper_action_tick_trace = [], [], [], []
    action_trace, decoded_action_trace = [], []
    collision_trace, fall_trace, progress_trace = [], [], []
    terminal_trace = {name: [] for name in (
        "lower_step", "env_id", "done", "success", "fall", "root", "rigid_body",
        "contact_force", "dof_pos", "dof_vel", "torque")}
    for name in ("fail_count", "timeout", "termination_height", "termination_tilt",
                 "height_above_lower_reference_limit", "termination_nonfoot_contact"):
        terminal_trace[name] = []
    terminal_reason_counts = {
        "height_low_or_nonfinite": 0, "tilt": 0,
        "nonfoot_contact": 0, "timeout": 0, "goal_success": 0}
    terrain_kinds = tuple(layout.spec.kind for layout in tiled.layouts)
    kind_sums = {kind: {name: 0.0 for name in (
        "transitions", "success", "fall", "collision", "off_support", "progress_m")}
                 for kind in sorted(set(terrain_kinds))}
    base_min = torch.full((env.num_envs,), float("inf"), device=env.device)
    base_max = torch.full((env.num_envs,), -float("inf"), device=env.device)
    start = time.perf_counter()
    for lower_step in range(args.lower_ticks):
        if args.record_video and lower_step % frame_stride == 0:
            root = env.root_states[0, 0]
            x, y, z, w = root[3:7]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            forward = torch.stack((torch.cos(yaw), torch.sin(yaw)))
            left = torch.stack((-torch.sin(yaw), torch.cos(yaw)))
            xy = root[:2] - 1.72 * forward + 1.02 * left
            env.gym.viewer_camera_look_at(
                env.viewer, None,
                gymapi.Vec3(float(xy[0]), float(xy[1]), float(root[2] + 0.05)),
                gymapi.Vec3(float(root[0]), float(root[1]), float(root[2])))
        transition = rollout.lower_tick(random_actions)
        if args.record_video and lower_step % frame_stride == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / "{:06d}.png".format(frame_index)))
            frame_index += 1
        base_min = torch.minimum(base_min, env.base_position[:, 2])
        base_max = torch.maximum(base_max, env.base_position[:, 2])
        root_trace.append(env.root_states[0, 0].detach().cpu().numpy().copy())
        body_trace.append(env.rigid_body_states[0].detach().cpu().numpy().copy())
        contact_trace.append(env.contact_forces[0].detach().cpu().numpy().copy())
        foot_trace.append(env.foot_positions[0].detach().cpu().numpy().copy())
        dof_position_trace.append(env.dof_pos[0].detach().cpu().numpy().copy())
        dof_velocity_trace.append(env.dof_vel[0].detach().cpu().numpy().copy())
        torque_trace.append(env.torques[0].detach().cpu().numpy().copy())
        target_position_trace.append(env.sampler.target_pos[0].detach().cpu().numpy().copy())
        target_yaw_trace.append(env.sampler.target_yaw[0].detach().cpu().numpy().copy())
        phase_trace.append(float(env.sampler.phase[0]))
        swing_foot_trace.append(int(env.sampler.swing_foot[0]))
        lower_action_trace.append(env.policy_actions[0].detach().cpu().numpy().copy())
        upper_action_tick_trace.append(
            rollout.previous_action[0].detach().cpu().numpy().copy())
        if transition is None:
            continue
        count = transition["ids"].numel()
        transition_count += count
        falls += int(transition["diagnostics"]["fall"].sum())
        off_support += int(transition["diagnostics"]["off_support"].sum())
        collisions += int(transition["diagnostics"]["collision"].sum())
        successes += int(transition["diagnostics"]["success"].sum())
        action_trace.append(transition["action"].detach().cpu().numpy().copy())
        decoded_action_trace.append(interface.bounds.decode(
            transition["action"], transition["diagnostics"]["landed_foot"]
        ).detach().cpu().numpy().copy())
        collision_trace.append(
            transition["diagnostics"]["collision"].detach().cpu().numpy().copy())
        fall_trace.append(transition["diagnostics"]["fall"].detach().cpu().numpy().copy())
        progress_trace.append(
            (transition["terms"]["progress"] / float(cfg["reward"]["progress"]))
            .detach().cpu().numpy().copy())
        transition_progress = (transition["terms"]["progress"]
                               / float(cfg["reward"]["progress"]))
        for kind, sums in kind_sums.items():
            mask = torch.as_tensor(
                [terrain_kinds[int(env_id)] == kind for env_id in transition["ids"]],
                dtype=torch.bool, device=env.device)
            if not mask.any():
                continue
            sums["transitions"] += int(mask.sum())
            sums["success"] += int(transition["diagnostics"]["success"][mask].sum())
            sums["fall"] += int(transition["diagnostics"]["fall"][mask].sum())
            sums["collision"] += int(transition["diagnostics"]["collision"][mask].sum())
            sums["off_support"] += int(transition["diagnostics"]["off_support"][mask].sum())
            sums["progress_m"] += float(transition_progress[mask].sum())
        terminal_rows = transition["done"].nonzero(as_tuple=False).flatten()
        for row in terminal_rows.tolist():
            terminal_trace["lower_step"].append(lower_step)
            terminal_trace["env_id"].append(int(transition["ids"][row]))
            terminal_trace["done"].append(bool(transition["done"][row]))
            terminal_trace["success"].append(bool(
                transition["diagnostics"]["success"][row]))
            terminal_trace["fall"].append(bool(
                transition["diagnostics"]["fall"][row]))
            terminal_reason_counts["height_low_or_nonfinite"] += int(
                transition["next_physics"]["termination_height"][row])
            terminal_reason_counts["tilt"] += int(
                transition["next_physics"]["termination_tilt"][row])
            terminal_reason_counts["nonfoot_contact"] += int(
                transition["next_physics"]["termination_nonfoot_contact"][row])
            terminal_reason_counts["timeout"] += int(
                transition["next_physics"]["timeout"][row])
            terminal_reason_counts["goal_success"] += int(
                transition["diagnostics"]["success"][row])
            for name in ("root", "rigid_body", "contact_force", "dof_pos", "dof_vel", "torque",
                         "fail_count", "timeout", "termination_height", "termination_tilt",
                         "height_above_lower_reference_limit", "termination_nonfoot_contact"):
                terminal_trace[name].append(
                    transition["next_physics"][name][row].detach().cpu().numpy().copy())
        for name, value in transition["terms"].items():
            term_sum[name] += float(value.sum())
    wall = time.perf_counter() - start

    if env.viewer is not None:
        env.gym.write_viewer_image_to_file(env.viewer, str(output / "overview_end.png"))
    np.savez_compressed(
        output / "trajectory_env0.npz", root=np.stack(root_trace),
        rigid_body=np.stack(body_trace), contact_force=np.stack(contact_trace),
        foot_pos=np.stack(foot_trace), dof_pos=np.stack(dof_position_trace),
        dof_vel=np.stack(dof_velocity_trace), torque=np.stack(torque_trace),
        target_pos=np.stack(target_position_trace), target_yaw=np.stack(target_yaw_trace),
        phase=np.asarray(phase_trace), swing_foot=np.asarray(swing_foot_trace),
        lower_action=np.stack(lower_action_trace),
        upper_action=np.stack(upper_action_tick_trace), dt_s=np.asarray(env.dt))
    if terminal_trace["lower_step"]:
        np.savez_compressed(
            output / "terminal_physics.npz",
            **{name: np.asarray(values) for name, values in terminal_trace.items()})
    all_actions = np.concatenate(action_trace) if action_trace else np.zeros((0, 3))
    all_decoded_actions = (np.concatenate(decoded_action_trace)
                           if decoded_action_trace else np.zeros((0, 4)))
    all_collision = np.concatenate(collision_trace).astype(bool) if collision_trace else np.zeros(0, bool)
    all_fall = np.concatenate(fall_trace).astype(bool) if fall_trace else np.zeros(0, bool)
    all_progress = np.concatenate(progress_trace) if progress_trace else np.zeros(0)

    def selected_action_mean(mask):
        return all_actions[mask].mean(0).tolist() if mask.any() else None

    metrics = {
        "num_envs": env.num_envs, "lower_ticks": args.lower_ticks,
        "simulated_seconds_per_env": args.lower_ticks * env.dt,
        "macro_transitions": transition_count,
        "transitions_per_env": transition_count / env.num_envs,
        # `falls` is retained for old result readers; it historically meant any
        # non-timeout physical terminal, including sustained body collision.
        "falls": falls, "physical_terminals_non_timeout": falls,
        "terminal_reason_counts": terminal_reason_counts,
        "off_support": off_support,
        "collisions": collisions, "successes": successes,
        "reward_term_mean": {
            name: value / max(transition_count, 1) for name, value in term_sum.items()},
        "base_height_min_m": float(base_min.min()),
        "base_height_max_m": float(base_max.max()),
        "lower_env_steps_per_wall_second": args.lower_ticks * env.num_envs / wall,
        "camera_render_rate_hz": (transition_count / env.num_envs) / wall,
        "finite_trajectory_env0": bool(all(np.isfinite(value).all() for value in (
            root_trace, body_trace, contact_trace, foot_trace))),
        "planner_checkpoint": args.planner_checkpoint,
        "privileged_terrain_planner": bool(args.privileged_terrain_planner),
        "uses_depth_sensor": not bool(args.privileged_terrain_planner),
        "ensemble_checkpoint": args.ensemble_checkpoint,
        "anchor_checkpoint": args.anchor_checkpoint,
        "ensemble_size": len(ensemble_models) if ensemble_models is not None else None,
        "uncertainty_coef": args.uncertainty_coef if ensemble_models is not None else None,
        "planning_horizon": (args.planning_horizon
                             if args.planner_checkpoint or args.ensemble_checkpoint else None),
        "depth_ablation": args.depth_ablation,
        "collision_risk": args.collision_risk,
        "fall_risk": args.fall_risk,
        "action_l2": args.action_l2,
        "terminal_value_coef": args.terminal_value_coef,
        "terminal_q_aggregation": args.terminal_q_aggregation,
        "freeze_latent_rollout": bool(args.freeze_latent_rollout),
        "future_action_mode": args.future_action_mode,
        "use_policy_prior": bool(args.use_policy_prior),
        "policy_rollout": bool(args.policy_rollout),
        "policy_std": args.policy_std,
        "planning_progress_only": bool(args.planning_progress_only),
        "planning_add_component": args.planning_add_component,
        "planning_disable_goal": bool(args.planning_disable_goal),
        "planning_base_safety_scale": args.planning_base_safety_scale,
        "decomposed_reward": args.decomposed_reward,
        "planner_load_missing": planner_load_missing if planner_model is not None else [],
        "planner_load_unexpected": planner_load_unexpected if planner_model is not None else [],
        "obstacle_y_m": args.obstacle_y_m if abs(args.obstacle_y_m) < 900.0 else None,
        "action_normalized": {
            "mean": all_actions.mean(0).tolist() if len(all_actions) else None,
            "std": all_actions.std(0).tolist() if len(all_actions) else None,
            "min": all_actions.min(0).tolist() if len(all_actions) else None,
            "max": all_actions.max(0).tolist() if len(all_actions) else None,
            "mean_collision": selected_action_mean(all_collision),
            "mean_no_collision": selected_action_mean(~all_collision),
            "mean_fall": selected_action_mean(all_fall),
        },
        "progress_by_event_m": {
            "collision": float(all_progress[all_collision].mean()) if all_collision.any() else None,
            "no_collision": float(all_progress[~all_collision].mean()) if (~all_collision).any() else None,
            "fall": float(all_progress[all_fall].mean()) if all_fall.any() else None,
        },
        "distance_to_goal_mean_m": float(torch.norm(
            env.base_position[:, :2] - task.goals, dim=-1).mean()),
        # Sum of reset-safe task-frame distance reductions divided by the
        # number of environments.  Unlike world-x displacement this remains
        # meaningful for turns and offset goals.
        "goal_distance_reduction_per_env_m": float(all_progress.sum() / env.num_envs),
        "base_forward_progress_mean_m": float(
            (env.base_position[:, 0] - env.env_origins[:, 0] - 0.20).mean()),
        "decoded_foothold_local_mean": {
            "forward_m": (float(all_decoded_actions[:, 0].mean())
                          if len(all_decoded_actions) else None),
            "lateral_abs_m": (float(np.abs(all_decoded_actions[:, 1]).mean())
                              if len(all_decoded_actions) else None),
            "radial_m": (float(np.linalg.norm(
                all_decoded_actions[:, :2], axis=1).mean())
                         if len(all_decoded_actions) else None),
            "yaw_deg": (float(np.rad2deg(all_decoded_actions[:, 3]).mean())
                        if len(all_decoded_actions) else None),
        },
        "terrain_kind_counts": {
            kind: sum(layout.spec.kind == kind for layout in tiled.layouts)
            for kind in ("straight", "s_curve", "fork", "random",
                         "narrow_bridge", "hurdles", "irregular_support")},
        "terrain_kind_metrics": {
            kind: dict(sums, progress_mean_m=(
                sums["progress_m"] / max(sums["transitions"], 1.0)))
            for kind, sums in kind_sums.items()},
    }
    terminal_episodes = successes + falls + terminal_reason_counts["timeout"]
    metrics["episode_outcomes"] = {
        "episodes": terminal_episodes,
        "successes": successes,
        "falls": falls,
        "timeouts": terminal_reason_counts["timeout"],
        "success_rate": successes / max(terminal_episodes, 1),
        "fall_rate": falls / max(terminal_episodes, 1),
        "timeout_rate": (terminal_reason_counts["timeout"]
                         / max(terminal_episodes, 1)),
    }
    if privileged_planner is not None:
        arrays = {
            name: (np.concatenate(values) if values else np.zeros(0))
            for name, values in privileged_diagnostics.items()}
        metrics["privileged_planner"] = {
            "candidate_count": int(privileged_planner.candidates.shape[0]),
            "forward_levels": list(privileged_config.forward_levels),
            "lateral_levels": list(privileged_config.lateral_levels),
            "yaw_levels": list(privileged_config.yaw_levels),
            "decisions": int(len(arrays["fallback"])),
            "fallbacks": int(arrays["fallback"].sum()),
            "fallback_rate": (float(arrays["fallback"].mean())
                              if len(arrays["fallback"]) else 0.0),
            "valid_candidates_mean": (float(
                arrays["candidate_valid_count"].mean())
                if len(arrays["candidate_valid_count"]) else None),
            "chosen_support_fraction_mean": (float(
                arrays["chosen_support_fraction"].mean())
                if len(arrays["chosen_support_fraction"]) else None),
            "chosen_geodesic_progress_mean_m": (float(
                arrays["chosen_geodesic_progress_m"].mean())
                if len(arrays["chosen_geodesic_progress_m"]) else None),
            "chosen_heading_error_mean_deg": (float(np.rad2deg(
                arrays["chosen_heading_error_rad"].mean()))
                if len(arrays["chosen_heading_error_rad"]) else None),
        }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if args.record_video:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.video_fps),
            "-i", str(frame_dir / "%06d.png"), "-c:v", "mpeg4", "-q:v", "4",
            "-pix_fmt", "yuv420p", str(output / "rollout.mp4")], check=True)
        metrics["video"] = str(output / "rollout.mp4")
        metrics["video_frames"] = frame_index
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
