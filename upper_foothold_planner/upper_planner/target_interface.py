"""Translate upper actions into the frozen lower controller's exact goal state."""

import torch

from .sampler import quaternion_yaw, wrap_to_pi, yaw_quaternion


class UpperFootholdTargetInterface:
    """Write only the next swing-foot target at a genuine gait boundary.

    The training sampler draws horizontal targets relative to the stance-foot
    origin and stance-foot yaw, then fixes world z to the terrain height. It does
    not rotate an xyz vector by the stance foot's roll and pitch. This class keeps
    that same convention and leaves phase generation inside the lower sampler.
    """

    def __init__(self, bounds, ground_height_m=0.0):
        self.bounds = bounds
        self.ground_height_m = float(ground_height_m)

    @torch.no_grad()
    def apply(self, env, normalized_actions, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=env.device)
        if ids.numel() == 0:
            return None
        actions = torch.as_tensor(
            normalized_actions, dtype=torch.float32, device=env.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).expand(ids.numel(), -1)
        if actions.shape != (ids.numel(), 3):
            raise ValueError("actions must have shape (len(ids), 3)")

        swing = env.sampler.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(ids.numel(), device=env.device)
        foot_state = env.rigid_body_states[ids][:, env.feet_indices]
        stance_pos = env.foot_positions[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_state[row, stance, 3:7])
        local = self.bounds.decode(actions, swing)

        cosine, sine = torch.cos(stance_yaw), torch.sin(stance_yaw)
        target = stance_pos.clone()
        target[:, 0] += cosine * local[:, 0] - sine * local[:, 1]
        target[:, 1] += sine * local[:, 0] + cosine * local[:, 1]
        target[:, 2] = self.ground_height_m + local[:, 2]
        target_yaw = wrap_to_pi(stance_yaw + local[:, 3])

        env.sampler.target_pos[ids, swing] = target
        env.sampler.target_yaw[ids, swing] = target_yaw
        env.sampler.target_quat[ids, swing] = yaw_quaternion(target_yaw)
        env.goal_buf[ids] = env.sampler.observation(
            env.foot_positions,
            env.rigid_body_states[:, env.feet_indices, 3:7])[ids]
        return {
            "ids": ids, "swing": swing, "stance": stance,
            "local_xyz_yaw": local, "world_position": target,
            "world_yaw": target_yaw,
        }
