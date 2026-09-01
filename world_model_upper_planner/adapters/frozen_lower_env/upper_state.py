"""Upper-level deployable observation and simulator-only task diagnostics."""

import math

import torch

from .sampler import quaternion_yaw, rotate_inverse, world_to_yaw_frame, wrap_to_pi


@torch.no_grad()
def build_proprio(env, task_goal_world_xy, previous_upper_action):
    """Build the documented 36-D upper observation with explicit scaling."""
    goal_xy = torch.as_tensor(task_goal_world_xy, dtype=torch.float32, device=env.device)
    previous = torch.as_tensor(previous_upper_action, dtype=torch.float32, device=env.device)
    base_yaw = quaternion_yaw(env.base_quat)
    goal_xyz = torch.cat((goal_xy, torch.zeros(env.num_envs, 1, device=env.device)), dim=-1)
    relative_goal = world_to_yaw_frame(goal_xyz, env.base_position, base_yaw)
    goal_yaw_error = wrap_to_pi(torch.atan2(relative_goal[:, 1], relative_goal[:, 0])) / math.pi

    foot_delta = env.foot_positions - env.base_position[:, None]
    base_quat = env.base_quat[:, None].expand(-1, 2, -1)
    feet_base = rotate_inverse(base_quat, foot_delta).reshape(env.num_envs, 6)
    phase = torch.stack((torch.cos(2.0 * math.pi * env.sampler.phase),
                         torch.sin(2.0 * math.pi * env.sampler.phase)), dim=-1)
    proprio = torch.cat((
        env.projected_gravity,                 # 3, already [-1,1]
        0.5 * env.base_lin_vel[:, :2],         # 2, 2 m/s maps to 1
        0.25 * env.base_ang_vel[:, 2:3],       # 1, 4 rad/s maps to 1
        env.dof_pos,                           # 8, rad
        0.1 * env.dof_vel,                     # 8, lower-policy scale
        feet_base,                             # 6, m in full base frame
        phase,                                 # 2
        previous,                              # 3, normalized upper action
        torch.stack((relative_goal[:, 0] / 6.0,
                     relative_goal[:, 1] / 3.0,
                     goal_yaw_error), dim=-1), # 3
    ), dim=-1)
    if proprio.shape[1] != 36:
        raise RuntimeError("upper proprio contract must remain 36-D")
    return proprio


class UpperTaskDiagnostics:
    """GPU support queries and macro reward state; never exposed to the Actor."""

    def __init__(self, env, tiled_terrain, reward_cfg):
        self.env = env
        self.reward_cfg = reward_cfg
        self.height_samples = torch.as_tensor(
            tiled_terrain.height_samples, device=env.device)
        self.origin_xy = torch.as_tensor(
            tiled_terrain.origin_xy_m, dtype=torch.float32, device=env.device)
        self.scale = float(tiled_terrain.horizontal_scale_m)
        self.goals = torch.stack([
            env.env_origins[index, :2] + torch.as_tensor(
                layout.goal_xy, dtype=torch.float32, device=env.device)
            for index, layout in enumerate(tiled_terrain.layouts)])
        self.previous_distance = torch.norm(env.base_position[:, :2] - self.goals, dim=-1)
        self.previous_heading_error = self._heading_error(
            torch.arange(env.num_envs, device=env.device))
        self.collision_since_decision = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
        self.max_collision_force = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device)
        self.min_base_height = env.base_position[:, 2].clone()
        self.max_tilt = self._tilt().clone()
        self.goal_reached = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
        self.off_support_enabled = True

    def _tilt(self):
        # Upright projected gravity is [0, 0, -1].
        return torch.acos(torch.clamp(-self.env.projected_gravity[:, 2], -1.0, 1.0))

    def _heading_error(self, ids):
        delta = self.goals[ids] - self.env.base_position[ids, :2]
        target_yaw = torch.atan2(delta[:, 1], delta[:, 0])
        base_yaw = quaternion_yaw(self.env.base_quat[ids])
        return torch.abs(wrap_to_pi(target_yaw - base_yaw))

    def is_supported(self, world_xy):
        index = torch.round((world_xy - self.origin_xy) / self.scale).long()
        valid = ((index[..., 0] >= 0) & (index[..., 0] < self.height_samples.shape[0])
                 & (index[..., 1] >= 0) & (index[..., 1] < self.height_samples.shape[1]))
        ix = index[..., 0].clamp(0, self.height_samples.shape[0] - 1)
        iy = index[..., 1].clamp(0, self.height_samples.shape[1] - 1)
        return valid & (self.height_samples[ix, iy] == 0)

    def observe_lower_tick(self):
        force = torch.norm(
            self.env.contact_forces[:, self.env.nonfoot_indices], dim=-1).max(dim=-1).values
        threshold = float(self.reward_cfg.get("collision_force_threshold_n", 5.0))
        self.collision_since_decision |= force > threshold
        self.max_collision_force = torch.maximum(self.max_collision_force, force)
        self.min_base_height = torch.minimum(
            self.min_base_height, self.env.base_position[:, 2])
        self.max_tilt = torch.maximum(self.max_tilt, self._tilt())

    def support_fraction(self, ids, landed_foot):
        """Fraction of a 3x3 sole stencil supported by the terrain mask."""
        foot_xy = self.env.foot_positions[ids, landed_foot, :2]
        foot_quat = self.env.rigid_body_states[
            ids, self.env.feet_indices[landed_foot], 3:7]
        yaw = quaternion_yaw(foot_quat)
        local = torch.tensor(
            [[x, y] for x in (-0.08, 0.0, 0.08)
             for y in (-0.035, 0.0, 0.035)],
            dtype=torch.float32, device=self.env.device)
        cosine, sine = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        x = cosine * local[None, :, 0] - sine * local[None, :, 1]
        y = sine * local[None, :, 0] + cosine * local[None, :, 1]
        samples = foot_xy[:, None, :] + torch.stack((x, y), dim=-1)
        return self.is_supported(samples).float().mean(dim=-1)

    def macro_state(self, ids):
        """21-D macro motion state at the end of an option (next decision state).

        Base height, full linear/angular velocity, orientation, feet in base
        frame, gait phase and relative goal.  This is the grounding signal for
        the latent dynamics; the full next observation is NOT reconstructed.
        """
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env.device)
        base_yaw = quaternion_yaw(self.env.base_quat[ids])
        goal_xyz = torch.cat(
            (self.goals[ids], torch.zeros(ids.numel(), 1, device=self.env.device)),
            dim=-1)
        relative_goal = world_to_yaw_frame(
            goal_xyz, self.env.base_position[ids], base_yaw)
        goal_yaw_error = wrap_to_pi(
            torch.atan2(relative_goal[:, 1], relative_goal[:, 0])) / math.pi
        base_quat = self.env.base_quat[ids, None].expand(-1, 2, -1)
        foot_delta = self.env.foot_positions[ids] - self.env.base_position[ids, None]
        feet_base = rotate_inverse(base_quat, foot_delta).reshape(ids.numel(), 6)
        phase = torch.stack((
            torch.cos(2.0 * math.pi * self.env.sampler.phase[ids]),
            torch.sin(2.0 * math.pi * self.env.sampler.phase[ids])), dim=-1)
        return torch.cat((
            self.env.base_position[ids, 2:3] / 0.8,     # base height (1)
            0.5 * self.env.base_lin_vel[ids],            # linear velocity (3)
            0.25 * self.env.base_ang_vel[ids],           # angular velocity (3)
            self.env.projected_gravity[ids],             # orientation (3)
            feet_base,                                   # feet in base frame (6)
            phase,                                       # gait phase (2)
            torch.stack((relative_goal[:, 0] / 6.0,
                         relative_goal[:, 1] / 3.0,
                         goal_yaw_error), dim=-1),       # relative goal (3)
        ), dim=-1)

    def reset(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env.device)
        self.previous_distance[ids] = torch.norm(
            self.env.base_position[ids, :2] - self.goals[ids], dim=-1)
        self.previous_heading_error[ids] = self._heading_error(ids)
        self.collision_since_decision[ids] = False
        self.max_collision_force[ids] = 0.0
        self.min_base_height[ids] = self.env.base_position[ids, 2]
        self.max_tilt[ids] = self._tilt()[ids]
        self.goal_reached[ids] = False

    def macro_reward(self, ids, fall, landed_foot, option_ticks=None):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.env.device)
        distance = torch.norm(self.env.base_position[ids, :2] - self.goals[ids], dim=-1)
        progress = self.previous_distance[ids] - distance
        heading_error = self._heading_error(ids)
        heading_progress = self.previous_heading_error[ids] - heading_error
        landed_position = self.env.foot_positions[ids, landed_foot, :2]
        support_fraction = self.support_fraction(ids, landed_foot)
        # 9-point stencil: a mostly-supported foot is not off-support, unlike
        # the old single-cell check that flagged feet landing near gap edges.
        off_support = (support_fraction < 0.5).float()
        target_position = self.env.sampler.target_pos[ids, landed_foot, :2]
        touchdown_error = torch.norm(landed_position - target_position, dim=-1)
        collision_force = self.max_collision_force[ids]
        healthy_lower = float(self.env.cfg.env.healthy_height_range[0])
        max_tilt_rad = math.radians(float(self.env.cfg.env.max_tilt_deg))
        height_margin = (self.min_base_height[ids] - healthy_lower) / 0.15
        tilt_margin = (max_tilt_rad - self.max_tilt[ids]) / max(max_tilt_rad, 1e-6)
        stability_margin = torch.minimum(height_margin, tilt_margin).clamp(-2.0, 2.0)
        if not self.off_support_enabled:
            off_support[:] = False
        at_goal = distance < 0.30
        success = at_goal & ~self.goal_reached[ids]
        collision = self.collision_since_decision[ids]
        # The lower environment has already auto-reset physical terminal states.
        # Do not manufacture progress/off-support from that new reset pose.
        distance = torch.where(fall, self.previous_distance[ids], distance)
        progress = torch.where(fall, torch.zeros_like(progress), progress)
        heading_progress = torch.where(
            fall, torch.zeros_like(heading_progress), heading_progress)
        off_support = torch.where(fall, torch.zeros_like(off_support), off_support)
        # Graded unsupported fraction for the reward: a foot with 70% support
        # is penalised lightly instead of the full binary -off_support hit.
        unsupported_fraction = torch.where(
            fall, torch.zeros_like(support_fraction), 1.0 - support_fraction)
        success = success & ~fall
        self.goal_reached[ids] |= at_goal & ~fall
        cfg = self.reward_cfg
        terms = {
            "progress": float(cfg["progress"]) * progress,
            "time": (float(cfg["time"]) * option_ticks / 25.0
                     if option_ticks is not None
                     else torch.full_like(progress, float(cfg["time"]))),
            "goal": float(cfg["goal"]) * success.float(),
            "collision": float(cfg["collision"]) * collision.float(),
            "fall": float(cfg["fall"]) * fall.float(),
            "off_support": float(cfg["off_support"]) * unsupported_fraction,
        }
        reward = sum(terms.values())
        self.previous_distance[ids] = distance
        self.previous_heading_error[ids] = heading_error
        self.collision_since_decision[ids] = False
        self.max_collision_force[ids] = 0.0
        self.min_base_height[ids] = self.env.base_position[ids, 2]
        self.max_tilt[ids] = self._tilt()[ids]
        return reward, terms, {
            "distance_to_goal_m": distance, "off_support": off_support,
            "success": success, "collision": collision, "fall": fall,
            "heading_progress_rad": heading_progress,
            "collision_force_n": collision_force,
            "stability_margin": stability_margin,
            "support_fraction": support_fraction,
            "touchdown_error_m": touchdown_error,
            "macro_state": self.macro_state(ids),
        }
