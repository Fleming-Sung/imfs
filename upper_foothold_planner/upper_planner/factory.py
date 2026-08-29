"""One readable construction path for upper-planner Isaac Gym experiments."""

import json
from pathlib import Path

import numpy as np
import torch

from .config import AttrDict
from .contracts import FootholdActionBounds, PolarFootholdActionBounds
from .env import FootholdEnv, make_sim_params
from .lower_policy import FrozenLowerPolicy
from .target_interface import UpperFootholdTargetInterface
from .terrain import (TerrainSpec, build_full_triangle_mesh, build_static_boxes,
                      build_tiled_heightfield)
from .upper_state import UpperTaskDiagnostics


def disable_domain_randomization(cfg):
    cfg.noise.add_noise = False
    for name in (
            "randomize_friction", "randomize_base_mass", "randomize_link_mass",
            "randomize_base_com", "randomize_Kp", "randomize_Kd", "randomize_gravity",
            "randomize_joint_damping", "randomize_joint_friction",
            "randomize_joint_armature", "kick_robots"):
        cfg.domain_rand[name] = False


def _sample_midcourse_spawns(layouts, seed, candidates=4, min_x_m=0.5,
                             goal_margin_m=0.6):
    """Per-env supported mid-course spawn candidates for reverse-curriculum reset."""
    rng = np.random.default_rng(seed + 777)
    spawns = np.zeros((len(layouts), candidates, 2), dtype=np.float32)
    for i, layout in enumerate(layouts):
        x = np.asarray(layout.x_m, dtype=np.float32)
        y = np.asarray(layout.y_m, dtype=np.float32)
        support = np.asarray(layout.support_mask, dtype=bool)
        goal_x = float(layout.goal_xy[0])
        x_ok = (x >= min_x_m) & (x <= goal_x - goal_margin_m)
        valid = support & x_ok[None, :]
        # Erode by one cell so a spawned foot never overhangs a gap edge.
        padded = np.pad(valid, 1, mode="constant", constant_values=False)
        valid = (valid & padded[:-2, 1:-1] & padded[2:, 1:-1]
                 & padded[1:-1, :-2] & padded[1:-1, 2:])
        jy, jx = np.nonzero(valid)
        if len(jx) == 0:
            spawns[i] = np.asarray(layout.start_xy, dtype=np.float32)
            continue
        for k in range(candidates):
            idx = rng.integers(len(jx))
            spawns[i, k] = (x[jx[idx]], y[jy[idx]])
    return spawns


def create_upper_system(root, args, num_envs, seed, corridor_width_m=0.90,
                        randomization=True, cameras=True, flat_plane=False,
                        obstacles=False, course_length_m=6.0, obstacle_count=6,
                        reward_override=None, obstacle_y_m=None):
    root = Path(root)
    project_cfg = json.loads((root / "config" / "default.json").read_text())
    checkpoint_path = root / "checkpoints" / "lower_model_7000.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    cfg = AttrDict.from_nested(checkpoint["config"])
    cfg.asset.file = str(root / "assets" / "SF_TRON1A" / "urdf" / "robot.urdf")
    cfg.env.num_envs = int(num_envs)
    cfg.env.episode_length_s = 30.0
    cfg.env.env_spacing_xy = [8.0, 4.0]
    cfg.env.termination_mode = "upper_joint"
    cfg.env.max_tilt_deg = 60.0
    cfg.env.nonfoot_terminal_force_n = 40.0
    cfg.env.fail_to_terminal_time_s = 0.10
    cfg.init.spawn_xy = [0.20, 0.0]
    if not randomization:
        disable_domain_randomization(cfg)

    curriculum = getattr(args, "terrain_curriculum", "obstacles")
    if curriculum == "typical":
        typical_kind = getattr(args, "typical_kind", "mixed")
        valid_typical_kinds = ("narrow_bridge", "hurdles", "irregular_support")
        if typical_kind == "mixed":
            kinds = valid_typical_kinds
        elif typical_kind in valid_typical_kinds:
            kinds = (typical_kind,)
        else:
            raise ValueError(
                "typical_kind must be mixed, narrow_bridge, hurdles, or irregular_support")
        specs = []
        bridge_width_min = float(getattr(args, "bridge_width_min_m", 0.55))
        bridge_width_max = float(getattr(args, "bridge_width_max_m", 0.75))
        if bridge_width_min <= 0.0 or bridge_width_max < bridge_width_min:
            raise ValueError("bridge width range must be positive and ordered")
        bridge_widths = np.linspace(bridge_width_min, bridge_width_max, 3)
        irregular_width = float(getattr(args, "irregular_width_m", 0.65))
        hurdle_height_min = float(getattr(args, "hurdle_height_min_m", 0.035))
        hurdle_height_max = float(getattr(args, "hurdle_height_max_m", 0.085))
        if (irregular_width <= 0.0 or hurdle_height_min <= 0.0
                or hurdle_height_max < hurdle_height_min):
            raise ValueError("typical terrain difficulty ranges are invalid")
        for index in range(num_envs):
            kind = kinds[index % len(kinds)]
            if kind == "narrow_bridge":
                width = float(bridge_widths[(index // len(kinds)) % 3])
            elif kind == "hurdles":
                width = 1.60
            else:
                width = irregular_width
            specs.append(TerrainSpec(
                kind=kind, corridor_width_m=width,
                length_m=float(course_length_m), seed=seed * 1000 + index,
                hurdle_height_min_m=hurdle_height_min,
                hurdle_height_max_m=hurdle_height_max))
    elif curriculum == "randomized":
        width_min = float(getattr(args, "random_width_min_m", 0.55))
        width_max = float(getattr(args, "random_width_max_m", 1.20))
        gap_min = float(getattr(args, "random_gap_min_m", 0.0))
        gap_max = float(getattr(args, "random_gap_max_m", 0.10))
        obstacle_probability = float(
            getattr(args, "random_obstacle_probability", 0.45))
        hurdle_height_min = float(getattr(args, "hurdle_height_min_m", 0.025))
        hurdle_height_max = float(getattr(args, "hurdle_height_max_m", 0.05))
        specs = [TerrainSpec(
            kind="random_composite", length_m=float(course_length_m),
            seed=seed * 1000 + index,
            support_width_min_m=width_min,
            support_width_max_m=width_max,
            support_gap_min_m=gap_min,
            support_gap_max_m=gap_max,
            obstacle_probability=obstacle_probability,
            hurdle_height_min_m=hurdle_height_min,
            hurdle_height_max_m=hurdle_height_max)
            for index in range(num_envs)]
    elif curriculum == "research":
        # Weighted procedural mixture: local feasibility boundaries, composite
        # support, stepping stones, turns, and household-like obstacle layouts.
        research_kind = getattr(args, "research_kind", "mixed")
        valid_research_kinds = (
            "edge_cases", "random_composite", "stepping_stones",
            "turns", "household")
        if research_kind == "mixed":
            schedule = (
                "edge_cases", "edge_cases", "edge_cases",
                "random_composite", "random_composite",
                "stepping_stones", "stepping_stones",
                "turns", "turns", "household")
        elif research_kind in valid_research_kinds:
            schedule = (research_kind,)
        else:
            raise ValueError(
                "research_kind must be mixed, edge_cases, random_composite, "
                "stepping_stones, turns, or household")
        specs = []
        for index in range(num_envs):
            kind = schedule[index % len(schedule)]
            specs.append(TerrainSpec(
                kind=kind, length_m=float(course_length_m),
                width_m=3.2, resolution_m=0.05,
                seed=seed * 1000 + index,
                support_width_min_m=float(getattr(
                    args, "random_width_min_m", 0.50)),
                support_width_max_m=float(getattr(
                    args, "random_width_max_m", 1.30)),
                support_gap_min_m=float(getattr(
                    args, "random_gap_min_m", 0.0)),
                support_gap_max_m=float(getattr(
                    args, "random_gap_max_m", 0.14)),
                obstacle_probability=float(getattr(
                    args, "random_obstacle_probability", 0.55)),
                hurdle_height_min_m=float(getattr(
                    args, "hurdle_height_min_m", 0.02)),
                hurdle_height_max_m=float(getattr(
                    args, "hurdle_height_max_m", 0.055))))
    elif curriculum == "obstacles":
        kinds = ("straight", "s_curve", "fork", "random")
        specs = [TerrainSpec(
            kind=kinds[index % len(kinds)], corridor_width_m=corridor_width_m,
            length_m=float(course_length_m), seed=seed * 1000 + index)
            for index in range(num_envs)]
    else:
        raise ValueError(
            "terrain_curriculum must be obstacles, typical, randomized, or research")
    tiled = build_tiled_heightfield(specs)
    cfg.terrain.height_samples = None
    cfg.terrain.ground_height_m = 0.0 if flat_plane else -float(specs[0].pit_depth_m)
    cfg.terrain.pit_depth_m = float(specs[0].pit_depth_m)
    cfg.terrain.add_ground_plane = bool(flat_plane)
    cfg.terrain.mesh_vertices = None
    cfg.terrain.mesh_triangles = None
    if curriculum in ("typical", "randomized", "research"):
        if obstacles:
            raise ValueError("typical curriculum owns its static geometry")
        cfg.terrain.add_ground_plane = True
        cfg.terrain.ground_height_m = -float(specs[0].pit_depth_m)
        cfg.terrain.static_boxes = build_static_boxes(tiled.layouts)
        # Per-env gravity forces are allocated for robot bodies only. Static
        # terrain actors add bodies, so keep the exact same safety rule used by
        # the earlier box-obstacle curriculum.
        cfg.domain_rand.randomize_gravity = False
    else:
        # Corridor curricula (straight/s_curve/fork/random) render the whole
        # tiled terrain as one triangle mesh.  The box-based curricula above
        # never use it, so building it there would only waste memory and time.
        mesh = build_full_triangle_mesh(tiled)
        cfg.terrain.mesh_vertices = None if flat_plane else mesh.vertices
        cfg.terrain.mesh_triangles = None if flat_plane else mesh.triangles
    if obstacles:
        if not flat_plane:
            raise ValueError("first obstacle curriculum requires the validated flat plane")
        rng = np.random.default_rng(seed + 991)
        if int(obstacle_count) < 1:
            raise ValueError("obstacle_count must be positive when obstacles are enabled")
        final_obstacle_x = float(course_length_m) - 0.75
        if final_obstacle_x < 1.0:
            raise ValueError("course_length_m must be at least 1.75")
        x = np.linspace(1.0, final_obstacle_x, int(obstacle_count), dtype=np.float32)
        obstacle_xy = np.zeros((num_envs, len(x), 2), dtype=np.float32)
        obstacle_xy[:, :, 0] = x[None]
        if obstacle_y_m is None:
            obstacle_xy[:, :, 1] = rng.uniform(-0.65, 0.65, size=(num_envs, len(x)))
        else:
            obstacle_xy[:, :, 1] = float(obstacle_y_m)
        cfg.terrain.obstacles_xy = obstacle_xy
        cfg.terrain.obstacle_size_xyz_m = [0.35, 0.35, 0.50]
        # Extra static actors make the lower per-body gravity-force tensor invalid.
        cfg.domain_rand.randomize_gravity = False
    cfg.camera = AttrDict.from_nested(dict(project_cfg["depth"], enabled=bool(cameras)))

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)
    reset_prob = float(getattr(args, "reset_curriculum_prob", 0.0))
    if reset_prob > 0.0 and curriculum in ("typical", "randomized", "research"):
        env.configure_reset_curriculum(
            _sample_midcourse_spawns(tiled.layouts, seed), reset_prob)
    policy = FrozenLowerPolicy(checkpoint_path, env.device)
    action_profile = getattr(args, "action_profile", "legacy")
    if action_profile == "legacy":
        bounds = FootholdActionBounds.from_config(project_cfg["action"])
    elif action_profile == "polar":
        bounds = PolarFootholdActionBounds.from_config(project_cfg["action_polar"])
    elif action_profile == "polar_course":
        bounds = PolarFootholdActionBounds.from_config(
            project_cfg["action_polar_course"])
    elif action_profile == "cartesian_course":
        # Direct stance-frame x / |y| / yaw coordinates.  Unlike the polar
        # profile, minimum lateral separation does not collapse most of one
        # normalized action dimension onto the same physical command.
        bounds = FootholdActionBounds.from_config(
            project_cfg["action_cartesian_course"])
    else:
        raise ValueError(
            "action_profile must be legacy, polar, polar_course, "
            "or cartesian_course")
    project_cfg["action_profile"] = action_profile
    interface = UpperFootholdTargetInterface(bounds)
    if reward_override:
        project_cfg["reward"].update(reward_override)
    diagnostics = UpperTaskDiagnostics(env, tiled, project_cfg["reward"])
    diagnostics.off_support_enabled = (
        curriculum in ("typical", "randomized", "research") or not flat_plane)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return env, policy, interface, diagnostics, tiled, project_cfg
