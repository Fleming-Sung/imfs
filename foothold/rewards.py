"""rewards.py —— 论文落足跟踪奖励函数（逐项实现、逐项记录）。

跟踪项 r_sw = w1·exp(-ξ1·‖Δp_xy‖²) + w2·exp(-ξ2·Δp_z²) + w3·exp(-ξ3·Δψ²)，
支撑脚项在触地瞬间锁存、整段支撑相复用。其余为风格惩罚项。
公式与 mind_steps/rewards.py 一致，每项返回 (num_envs,) 张量。
"""

import torch

from .sampler import quaternion_yaw, wrap_to_pi


class Rewards:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.scales = cfg.rewards.scales
        self.sharp = cfg.rewards.sharpness
        # 上一步摆动脚的原始（无 dt）跟踪质量，供 stance 项锁存
        self._last_swing_reward = torch.zeros(env.num_envs, 3, device=env.device)
        self.names = []
        self.fns = []
        for name in self.scales:
            fn = getattr(self, f"_reward_{name}", None)
            if self.scales[name] != 0 and fn is not None:
                self.names.append(name)
                self.fns.append(fn)

    def compute(self):
        env = self.env
        rew = torch.zeros(env.num_envs, device=env.device)
        raw, weighted = {}, {}
        for name, fn in zip(self.names, self.fns):
            r = fn()
            w = self.scales[name] * r
            if self.cfg.rewards.clip_single_reward is not None:
                w = torch.clip(w, -self.cfg.rewards.clip_single_reward, self.cfg.rewards.clip_single_reward)
            rew = rew + w
            raw[name] = r.float().mean().item()
            weighted[name] = w
        if self.cfg.rewards.clip_reward is not None:
            rew = torch.clip(rew, -self.cfg.rewards.clip_reward, self.cfg.rewards.clip_reward)
        self._store_swing_reward()
        return rew, raw, weighted

    def _store_swing_reward(self):
        """存储本步摆动脚原始跟踪质量（exp(-sharp·err²)，无 dt），供下一步 stance 锁存。"""
        delta, _, yaw, _ = self._errors()
        row, swing, _ = self._swing_stance()
        self._last_swing_reward[:, 0] = torch.exp(-self.sharp.xy * torch.sum(delta[:, :2] ** 2, dim=1))
        self._last_swing_reward[:, 1] = torch.exp(-self.sharp.z * delta[:, 2] ** 2)
        self._last_swing_reward[:, 2] = torch.exp(-self.sharp.yaw * yaw ** 2)

    # ---- 工具 ----

    def _swing_stance(self):
        row = torch.arange(self.env.num_envs, device=self.env.device)
        swing = self.env.sampler.swing_foot
        return row, swing, 1 - swing

    def _foot_yaw(self):
        return quaternion_yaw(self.env.rigid_body_states[:, self.env.feet_indices, 3:7])

    def _errors(self):
        pos = self.env.foot_positions
        yaw = self._foot_yaw()
        row, swing, stance = self._swing_stance()
        s = self.env.sampler
        return (pos[row, swing] - s.target_pos[row, swing],
                pos[row, stance] - s.target_pos[row, stance],
                wrap_to_pi(yaw[row, swing] - s.target_yaw[row, swing]),
                wrap_to_pi(yaw[row, stance] - s.target_yaw[row, stance]))

    def _exp(self, error_sq, sharpness):
        return torch.exp(-sharpness * error_sq) * self.env.dt

    # ---- 跟踪项 ----

    def _reward_survival(self):
        return torch.ones(self.env.num_envs, device=self.env.device) * self.env.dt

    def _reward_swing_xy(self):
        delta, _, _, _ = self._errors()
        return self._exp(torch.sum(delta[:, :2] ** 2, dim=1), self.sharp.xy)

    def _reward_swing_z(self):
        row, swing, _ = self._swing_stance()
        half = torch.remainder(self.env.sampler.phase, 0.5) / 0.5
        desired_z = self.env.sampler.target_pos[row, swing, 2] + torch.where(
            half <= 0.5, self.cfg.foothold.swing_height, 0.0)
        err = self.env.foot_positions[row, swing, 2] - desired_z
        return self._exp(err ** 2, self.sharp.z)

    def _reward_swing_yaw(self):
        _, _, yaw, _ = self._errors()
        return self._exp(yaw ** 2, self.sharp.yaw)

    def _update_stance_latch(self):
        ids = self.env.sampler.last_switch_ids
        if ids.numel() == 0:
            return
        # 论文：支撑脚奖励锁存摆动脚落地瞬间（上一步）的跟踪质量
        self.env.sampler.stance_reward[ids] = self._last_swing_reward[ids]

    def _reward_stance_xy(self):
        self._update_stance_latch()
        return self.env.sampler.stance_reward[:, 0] * self.env.dt

    def _reward_stance_z(self):
        return self.env.sampler.stance_reward[:, 1] * self.env.dt

    def _reward_stance_yaw(self):
        return self.env.sampler.stance_reward[:, 2] * self.env.dt

    def _reward_feet_swing(self):
        phase = self.env.sampler.phase
        row, swing, _ = self._swing_stance()
        center = torch.where(swing == 0, 0.25, 0.75)
        in_window = torch.abs(phase - center) < self.cfg.foothold.swing_window_half_width
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0
        correct_air = ~contact[row, swing]
        hold_still = self.env.sampler.hold_still
        hold_ok = hold_still & contact.all(dim=1)
        return ((in_window & correct_air & ~hold_still) | hold_ok).float() * self.env.dt

    def _reward_gait_height(self):
        row, swing, _ = self._swing_stance()
        half = torch.remainder(self.env.sampler.phase, 0.5) / 0.5
        w = torch.where(half < 0.5, 2.0 * half, 2.0 * (1.0 - half))
        desired_z = self.env.sampler.target_pos[row, swing, 2] + self.cfg.foothold.swing_height
        deficit = torch.clamp(desired_z - self.env.foot_positions[row, swing, 2], min=0.0)
        # 论文用原始 deficit（不除以 swing_height），hold_still 时关闭
        return torch.exp(-self.sharp.gait_height * deficit ** 2 * w *
                         (~self.env.sampler.hold_still).float()) * self.env.dt

    def _reward_knee_height(self):
        return torch.zeros(self.env.num_envs, device=self.env.device)

    def _reward_nominal_joint_pos(self):
        # 论文：行走时只跟踪躯干关节（SF 无躯干 → 空集 → exp(0)=1）；hold_still 时跟踪全部关节
        err = torch.sum((self.env.dof_pos - self.env.default_dof_pos) ** 2, dim=1)
        hold_reward = torch.exp(-4.0 * err)
        return torch.where(self.env.sampler.hold_still, hold_reward,
                           torch.ones_like(hold_reward)) * self.env.dt

    # ---- 惩罚项 ----

    def _reward_base_height(self):
        return (self.env.base_position[:, 2] - self.cfg.rewards.base_height_target) ** 2 * self.env.dt

    def _reward_body_contacts(self):
        # 膝盖/大腿等身体部件接触地面时惩罚（净接触力 z 分量超过阈值即计数）
        if self.env.body_contact_indices.numel() == 0:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        threshold = getattr(self.cfg.rewards, "body_contacts_threshold", 1.0)
        contact = (self.env.contact_forces[:, self.env.body_contact_indices, 2]
                   > threshold).float()
        return contact.sum(dim=1) * self.env.dt

    def _reward_action_rate(self):
        # 论文用 ||a_t - a_{t-1}||²，last_actions[:, :, 0] 存的是上一步 policy action
        # hold_still 时系数翻倍
        r = torch.sum((self.env.policy_actions - self.env.last_actions[:, :, 0]) ** 2, dim=1)
        r = torch.where(self.env.sampler.hold_still, 2.0 * r, r)
        return r * self.env.dt

    def _reward_feet_slip(self):
        contact = (self.env.contact_forces[:, self.env.feet_indices, 2] > 1.0).float()
        return torch.sum(self.env.foot_velocities ** 2 * contact.unsqueeze(-1), dim=(1, 2)) * self.env.dt

    def _reward_feet_roll(self):
        q = self.env.rigid_body_states[:, self.env.feet_indices, 3:7]
        x, y, z, w = q.unbind(-1)
        roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return torch.sum(roll ** 2, dim=1) * self.env.dt

    def _reward_orientation(self):
        return torch.sum(self.env.projected_gravity[:, :2] ** 2, dim=1) * self.env.dt

    def _reward_torques(self):
        return torch.sum(self.env.torques ** 2, dim=1) * self.env.dt

    def _reward_energy(self):
        return torch.sum(torch.clamp(self.env.torques * self.env.dof_vel, min=0.0), dim=1) * self.env.dt

    def _reward_ang_vel_xy(self):
        return torch.sum(self.env.base_ang_vel[:, :2] ** 2, dim=1) * self.env.dt

    def _reward_dof_vel(self):
        return torch.sum(self.env.dof_vel ** 2, dim=1) * self.env.dt

    def _reward_dof_acc(self):
        return torch.sum(self.env.dof_acc ** 2, dim=1) * self.env.dt

    def _reward_root_acc(self):
        return torch.sum(self.env.root_acceleration ** 2, dim=1) * self.env.dt

    def _reward_dof_pos_limits(self):
        # 论文：关节角超出范围中间 98% 的计数
        scale = self.cfg.rewards.joint_position_limit_scale
        joint_range = self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]
        margin = 0.5 * (1.0 - scale) * joint_range
        lower = self.env.dof_pos_limits[:, 0] + margin
        upper = self.env.dof_pos_limits[:, 1] - margin
        outside = (self.env.dof_pos < lower) | (self.env.dof_pos > upper)
        return outside.float().sum(dim=1) * self.env.dt
