"""Event-driven two-level rollout: lower ticks, upper transitions at foot switches."""

import torch

from .contracts import preprocess_isaac_depth
from .upper_state import build_proprio


class UpperRollout:
    def __init__(self, env, lower_policy, target_interface, task, depth_cfg,
                 capture_depth=True):
        self.env = env
        self.lower_policy = lower_policy
        self.target_interface = target_interface
        self.task = task
        self.depth_cfg = depth_cfg
        self.capture_depth_enabled = bool(capture_depth)
        self.obs, self.goal, _ = env.get_observations()
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_swing = env.sampler.swing_foot.clone()
        self.previous_action = torch.zeros(env.num_envs, 3, device=env.device)
        self.previous_from_planner = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
        # Duration of the currently executing upper-level option in 50 Hz
        # lower-policy ticks. An option starts when a foothold intent is issued
        # and ends on touchdown, fall, timeout, or task success.
        self.option_ticks = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device)
        self.state_depth = torch.zeros(
            env.num_envs, 1, int(depth_cfg["height"]), int(depth_cfg["width"]),
            device=env.device)
        self.state_proprio = torch.zeros(env.num_envs, 36, device=env.device)
        self.lower_ticks = 0

    @torch.no_grad()
    def _sense(self):
        if self.capture_depth_enabled:
            raw = self.env.capture_depth()
            depth = preprocess_isaac_depth(
                raw, self.depth_cfg["near_m"], self.depth_cfg["far_m"]).unsqueeze(1)
        else:
            # Privileged geometry planning deliberately bypasses the camera.
            # Keep the transition contract intact so the same closed-loop
            # diagnostics and evaluator can be reused without sensor cost.
            depth = torch.zeros(
                self.env.num_envs, 1, int(self.depth_cfg["height"]),
                int(self.depth_cfg["width"]), device=self.env.device)
        proprio = build_proprio(self.env, self.task.goals, self.previous_action)
        return depth, proprio

    @torch.no_grad()
    def lower_tick(self, choose_actions):
        """Advance 20 ms and return zero or more completed macro transitions."""
        lower_action, _ = self.lower_policy.infer(self.obs, self.goal)
        next_obs, _, done, extras, next_goal, _ = self.env.step(lower_action)
        self.task.observe_lower_tick()
        self.lower_ticks += 1
        self.option_ticks[self.initialized] += 1

        fall = done.bool() & ~extras.get(
            "time_outs", torch.zeros_like(done, dtype=torch.bool)).bool()
        switched = (self.env.sampler.swing_foot != self.previous_swing) & self.initialized & ~done.bool()
        event = switched | done.bool()
        self.initialized[done.bool()] = False
        ready = (~self.initialized & ~self.env.goal_reset_pending
                 & (self.env.episode_length_buf > 0) & ~done.bool())
        sense_ids = (event | ready).nonzero(as_tuple=False).flatten()
        transitions = None
        if sense_ids.numel():
            sensed_depth, sensed_proprio = self._sense()
            event_ids = event.nonzero(as_tuple=False).flatten()
            success_ids = torch.empty(0, dtype=torch.long, device=self.env.device)
            if event_ids.numel():
                landed = 1 - self.env.sampler.swing_foot[event_ids]
                reward, terms, diagnostics = self.task.macro_reward(
                    event_ids, fall[event_ids], landed,
                    self.option_ticks[event_ids].clone())
                # This is the foot targeted by the completed upper action.  It
                # is needed to decode normalized actions back to physical local
                # footholds without guessing from the post-switch phase.
                diagnostics["landed_foot"] = landed.clone()
                diagnostics["option_duration_ticks"] = self.option_ticks[
                    event_ids].clone().clamp_min(1)
                success = diagnostics["success"].bool()
                success_ids = event_ids[success]
                next_physics = {
                    "root": self.env.root_states[event_ids, 0].clone(),
                    "rigid_body": self.env.rigid_body_states[event_ids].clone(),
                    "contact_force": self.env.contact_forces[event_ids].clone(),
                    "dof_pos": self.env.dof_pos[event_ids].clone(),
                    "dof_vel": self.env.dof_vel[event_ids].clone(),
                    "torque": self.env.torques[event_ids].clone(),
                    "fail_count": self.env.fail_buf[event_ids].clone(),
                    "timeout": extras["time_outs"][event_ids].clone(),
                    "termination_height": extras["termination_reasons"]["height"][event_ids].clone(),
                    "height_above_lower_reference_limit": extras[
                        "termination_reasons"]["height_above_lower_reference_limit"][event_ids].clone(),
                    "termination_tilt": extras["termination_reasons"]["tilt"][event_ids].clone(),
                    "termination_nonfoot_contact": extras[
                        "termination_reasons"]["nonfoot_contact"][event_ids].clone(),
                }
                terminal_physics = extras.get("terminal_physics")
                if terminal_physics is not None:
                    terminal_ids = terminal_physics["ids"]
                    done_rows = done[event_ids].bool().nonzero(
                        as_tuple=False).flatten()
                    if not torch.equal(event_ids[done_rows], terminal_ids):
                        raise RuntimeError("terminal physics ids do not match done transition ids")
                    for name in next_physics:
                        next_physics[name][done_rows] = terminal_physics[name]
                transitions = {
                    "ids": event_ids,
                    "depth": self.state_depth[event_ids].clone(),
                    "proprio": self.state_proprio[event_ids].clone(),
                    "action": self.previous_action[event_ids].clone(),
                    "from_planner": self.previous_from_planner[event_ids].clone(),
                    "reward": reward,
                    "next_depth": sensed_depth[event_ids].clone(),
                    "next_proprio": sensed_proprio[event_ids].clone(),
                    # Goal completion is a real episodic boundary.  Continuing
                    # from the goal would leak post-success behavior into the
                    # replay return and value targets.
                    "done": done[event_ids].bool() | success,
                    "terms": terms,
                    "diagnostics": diagnostics,
                    "next_physics": next_physics,
                }
            terminal_success = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device)
            terminal_success[success_ids] = True
            self.initialized[success_ids] = False
            action_ids = ((switched | ready) & ~terminal_success).nonzero(
                as_tuple=False).flatten()
            if action_ids.numel():
                chosen = choose_actions(
                    sensed_depth[action_ids], sensed_proprio[action_ids], action_ids)
                if isinstance(chosen, tuple):
                    new_actions, planner_mask = chosen
                    self.previous_from_planner[action_ids] = torch.as_tensor(
                        planner_mask, dtype=torch.bool, device=self.env.device)
                else:
                    new_actions = chosen
                    self.previous_from_planner[action_ids] = False
                new_actions = torch.as_tensor(
                    new_actions, dtype=torch.float32, device=self.env.device).clamp(-1.0, 1.0)
                self.target_interface.apply(self.env, new_actions, action_ids)
                self.previous_action[action_ids] = new_actions
                # Store the decision state with its previous-action field, then
                # execute new_actions. This is the conventional (s_t,a_t) pair.
                self.state_depth[action_ids] = sensed_depth[action_ids]
                self.state_proprio[action_ids] = sensed_proprio[action_ids]
                self.initialized[action_ids] = True
                self.option_ticks[action_ids] = 0
                self.task.reset(ready.nonzero(as_tuple=False).flatten())

            # Preserve the sensed terminal frame in `transitions`, then perform
            # an ordinary physical reset.  No new upper action is issued at the
            # successful terminal state.
            if success_ids.numel():
                self.env.reset_task_episodes(success_ids)

        self.previous_swing.copy_(self.env.sampler.swing_foot)
        if transitions is not None and transitions["diagnostics"]["success"].any():
            self.obs, self.goal, _ = self.env.get_observations()
        else:
            self.obs, self.goal = next_obs, next_goal
        return transitions
