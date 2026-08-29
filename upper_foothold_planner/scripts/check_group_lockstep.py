"""Validate natural lockstep replicas before counterfactual collection.

Isaac Gym cannot restore the complete PhysX solver/contact state.  This test
therefore creates K deterministic replicas per group, synchronizes only the
simulator-side gait sampler before control begins, and lets all replicas evolve
naturally under identical commands.  Candidate branching is allowed only if
the identical-action negative control stays within the declared tolerances.
"""

import json
import sys
from pathlib import Path

import numpy as np
from isaacgym import gymtorch, gymutil  # must precede torch
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from upper_planner.factory import create_upper_system


def arguments():
    custom = [
        {"name": "--groups", "type": int, "default": 8},
        {"name": "--candidates", "type": int, "default": 8},
        {"name": "--seed", "type": int, "default": 829},
        {"name": "--max_ticks", "type": int, "default": 80},
        {"name": "--output", "type": str,
         "default": str(ROOT / "experiments" / "g1_group_lockstep")},
    ]
    args = gymutil.parse_arguments(
        description="natural lockstep consistency gate", headless=True,
        custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.compute_device_id)
    args.terrain_curriculum = "obstacles"
    args.action_profile = "polar_course"
    return args


def group_error(value, groups, candidates):
    shaped = value.reshape(groups, candidates, *value.shape[1:])
    leader = shaped[:, :1]
    return float((shaped - leader).abs().max())


def local_root(env):
    root = env.root_states[:, 0].clone()
    root[:, :3] -= env.env_origins
    return root


def local_feet(env):
    return env.foot_positions - env.env_origins[:, None, :]


def broadcast_sampler(env, groups, candidates):
    sampler = env.sampler
    origins = env.env_origins.reshape(groups, candidates, 3)
    leader_origin = origins[:, :1]
    for name in ("phase", "frequency", "swing_foot", "movement_yaw", "feet_yaw",
                 "hold", "stance_reward", "step_count", "num_gaits",
                 "target_yaw", "target_quat"):
        value = getattr(sampler, name)
        shaped = value.reshape(groups, candidates, *value.shape[1:])
        shaped[:] = shaped[:, :1]
    target = sampler.target_pos.reshape(groups, candidates, 2, 3)
    target[:] = (target[:, :1] - leader_origin[:, :, None, :]
                 + origins[:, :, None, :])
    sampler.last_switch_ids = torch.empty(
        0, dtype=torch.long, device=env.device)
    env.goal_buf[:] = sampler.observation(
        env.foot_positions,
        env.rigid_body_states[:, env.feet_indices, 3:7])


def broadcast_tensor(value, groups, candidates):
    shaped = value.reshape(groups, candidates, *value.shape[1:])
    shaped[:] = shaped[:, :1]


def synchronize_writable_state(env, groups, candidates):
    """Remove reset-settle root/DOF variance without claiming full restore."""
    origins = env.env_origins.reshape(groups, candidates, 3)
    root = env.root_states[:, 0].reshape(groups, candidates, 13)
    local = root.clone()
    local[:, :, :3] -= origins
    local[:] = local[:, :1]
    root[:] = local
    root[:, :, :3] += origins
    broadcast_tensor(env.dof_state, groups, candidates)
    ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    actor_ids = env.robot_actor_indices[ids]
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(actor_ids), ids.numel())
    env.gym.set_dof_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(actor_ids), ids.numel())
    for name in ("actions", "policy_actions", "torques", "last_actions",
                 "last_dof_vel", "last_root_velocity", "dof_acc",
                 "root_acceleration", "base_lin_vel", "base_ang_vel",
                 "projected_gravity", "fail_buf"):
        broadcast_tensor(getattr(env, name), groups, candidates)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)


def main():
    args = arguments()
    if args.groups < 1 or args.candidates < 2:
        raise ValueError("groups >= 1 and candidates >= 2 are required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    num_envs = args.groups * args.candidates
    env, policy, interface, _, _, _ = create_upper_system(
        ROOT, args, num_envs=num_envs, seed=args.seed,
        randomization=False, cameras=False, flat_plane=True,
        obstacles=False, course_length_m=3.5)

    # Initialize the sampler before the first controlled tick.  The ordinary
    # env path defers this to _post_step, but its pre-initialization target is a
    # world-frame zero and therefore is not equivalent across env origins.
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env._compute_foot_state()
    pre_sampler_error = {
        "root_position": group_error(
            local_root(env)[:, :3], args.groups, args.candidates),
        "root_quaternion": group_error(
            local_root(env)[:, 3:7], args.groups, args.candidates),
        "root_velocity": group_error(
            local_root(env)[:, 7:13], args.groups, args.candidates),
        "dof_pos": group_error(env.dof_pos, args.groups, args.candidates),
        "dof_vel": group_error(env.dof_vel, args.groups, args.candidates),
        "feet": group_error(local_feet(env), args.groups, args.candidates),
    }
    all_ids = torch.arange(num_envs, device=env.device)
    env.sampler.reset(
        all_ids, env.foot_positions,
        env.rigid_body_states[:, env.feet_indices, 3:7])
    env.goal_reset_pending[:] = False
    synchronize_writable_state(env, args.groups, args.candidates)
    broadcast_sampler(env, args.groups, args.candidates)
    env._compute_observations()
    obs, goal, _ = env.get_observations()

    identical_action = torch.zeros(num_envs, 3, device=env.device)
    ids = all_ids
    interface.apply(env, identical_action, ids)
    obs, goal, _ = env.get_observations()
    # The writable-state injection does not update public rigid-body/contact
    # tensors.  Broadcast the first policy inputs only; all later inputs are
    # recomputed naturally and included in the lockstep error measurement.
    broadcast_tensor(obs, args.groups, args.candidates)
    broadcast_tensor(goal, args.groups, args.candidates)
    pre_control_error = dict(pre_sampler_error)
    pre_control_error.update({
        "goal": group_error(goal, args.groups, args.candidates),
        "observation": group_error(obs, args.groups, args.candidates),
    })
    initial_swing = env.sampler.swing_foot.clone()

    maxima = {name: 0.0 for name in (
        "root", "dof_pos", "dof_vel", "feet", "policy_action",
        "phase", "target_observation")}
    switch_tick = torch.full(
        (num_envs,), -1, dtype=torch.long, device=env.device)
    terminal_count = 0
    for tick in range(1, args.max_ticks + 1):
        lower_action, _ = policy.infer(obs, goal)
        next_obs, _, done, _, next_goal, _ = env.step(lower_action)
        switched = (env.sampler.swing_foot != initial_swing) & (switch_tick < 0)
        switch_tick[switched] = tick
        terminal_count += int(done.sum())
        maxima["root"] = max(maxima["root"], group_error(
            local_root(env), args.groups, args.candidates))
        maxima["dof_pos"] = max(maxima["dof_pos"], group_error(
            env.dof_pos, args.groups, args.candidates))
        maxima["dof_vel"] = max(maxima["dof_vel"], group_error(
            env.dof_vel, args.groups, args.candidates))
        maxima["feet"] = max(maxima["feet"], group_error(
            local_feet(env), args.groups, args.candidates))
        maxima["policy_action"] = max(maxima["policy_action"], group_error(
            env.policy_actions, args.groups, args.candidates))
        maxima["phase"] = max(maxima["phase"], group_error(
            env.sampler.phase, args.groups, args.candidates))
        maxima["target_observation"] = max(
            maxima["target_observation"],
            group_error(env.goal_buf, args.groups, args.candidates))
        obs, goal = next_obs, next_goal
        if (switch_tick >= 0).all():
            break

    switch = switch_tick.reshape(args.groups, args.candidates)
    switch_consistent = bool(((switch - switch[:, :1]) == 0).all())
    all_switched = bool((switch >= 0).all())
    tolerances = {
        "root": 1e-5, "dof_pos": 1e-5, "dof_vel": 1e-4,
        "feet": 1e-5, "policy_action": 1e-5, "phase": 1e-7,
        "target_observation": 1e-5,
    }
    passed = (all_switched and switch_consistent and terminal_count == 0
              and all(maxima[name] <= limit
                      for name, limit in tolerances.items()))
    summary = {
        "seed": args.seed, "groups": args.groups,
        "candidates_per_group": args.candidates,
        "ticks_executed": tick, "all_switched": all_switched,
        "switch_tick_consistent": switch_consistent,
        "switch_ticks": switch.detach().cpu().tolist(),
        "terminal_count": terminal_count,
        "pre_control_group_error": pre_control_error,
        "max_group_error": maxima,
        "tolerances": tolerances,
        "passed": passed,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
