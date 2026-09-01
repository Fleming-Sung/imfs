"""sampler.py —— 四元数工具 + 落足目标采样器（论文 GoalDoubleFootPlacement）。

目标在支撑脚坐标系中表示；采样发生在每次步态相位切换。只跑在仿真侧，策略只消费
其输出（目标 + 相位）。逻辑与论文发布代码 mind_steps/foothold.py 一致。
"""

import math

import torch


# --------------------------------------------------------------------------
# 四元数工具（scalar-last: [x, y, z, w]）
# --------------------------------------------------------------------------

def wrap_to_pi(x):
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def yaw_quaternion(yaw):
    half = 0.5 * yaw
    z = torch.zeros_like(half)
    return torch.stack((z, z, torch.sin(half), torch.cos(half)), dim=-1)


def quaternion_yaw(q):
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:4]), dim=-1)


def quaternion_multiply(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ), dim=-1)


def rotate_inverse(q, vector):
    pure = torch.cat((vector, torch.zeros_like(vector[..., :1])), dim=-1)
    return quaternion_multiply(
        quaternion_multiply(quaternion_conjugate(q), pure), q)[..., :3]


def rotate_vector(q, vector):
    pure = torch.cat((vector, torch.zeros_like(vector[..., :1])), dim=-1)
    return quaternion_multiply(quaternion_multiply(q, pure),
                               quaternion_conjugate(q))[..., :3]


def rigid_body_site_state(body_state, local_offset):
    """刚体上固定 local 偏移处的世界位置/速度。body_state: (..., 13)。"""
    world_offset = rotate_vector(body_state[..., 3:7], local_offset)
    position = body_state[..., :3] + world_offset
    velocity = body_state[..., 7:10] + torch.cross(body_state[..., 10:13], world_offset, dim=-1)
    return position, velocity


def world_to_yaw_frame(vector, origin, yaw):
    delta = vector - origin
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * delta[..., 0] + s * delta[..., 1]
    y = -s * delta[..., 0] + c * delta[..., 1]
    return torch.stack((x, y, delta[..., 2]), dim=-1)


# --------------------------------------------------------------------------
# 落足目标采样器
# --------------------------------------------------------------------------

class FootholdSampler:
    """论文 GoalDoubleFootPlacement 的纯 PyTorch 实现（paper_goal_schedule）。"""

    def __init__(self, num_envs, cfg, device):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device

        z2 = torch.zeros(num_envs, 2, device=device)
        self.target_pos = torch.zeros(num_envs, 2, 3, device=device)
        self.target_yaw = z2.clone()
        self.target_quat = torch.zeros(num_envs, 2, 4, device=device)
        self.target_quat[..., 3] = 1.0
        self.phase = torch.zeros(num_envs, device=device)
        self.frequency = torch.ones(num_envs, device=device)
        self.swing_foot = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.movement_yaw = torch.zeros(num_envs, device=device)
        self.feet_yaw = torch.zeros(num_envs, device=device)
        self.hold = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stance_reward = torch.zeros(num_envs, 3, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.num_gaits = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_switch_ids = torch.empty(0, dtype=torch.long, device=device)

    def _uniform(self, limits, n, degrees=False):
        lo, hi = float(limits[0]), float(limits[1])
        out = lo + (hi - lo) * torch.rand(n, device=self.device)
        return torch.deg2rad(out) if degrees else out

    def reset(self, ids, foot_pos, foot_quat):
        if ids.numel() == 0:
            return
        n = ids.numel()
        c = self.cfg
        self.target_pos[ids] = foot_pos[ids]
        self.target_yaw[ids] = quaternion_yaw(foot_quat[ids])
        self.target_quat[ids] = foot_quat[ids]
        self.movement_yaw[ids] = self._uniform(c.movement_direction_deg, n, degrees=True)
        self.feet_yaw[ids] = self._uniform(c.feet_direction_deg, n, degrees=True)
        self.frequency[ids] = self._uniform(c.gait_frequency, n)

        rel_dir = wrap_to_pi(self.movement_yaw[ids] - self.feet_yaw[ids])
        random_phase = torch.randint(0, 2, (n,), device=self.device).float() * 0.5
        self.phase[ids] = torch.where(
            rel_dir > 0.0, torch.zeros_like(random_phase),
            torch.where(rel_dir < 0.0, torch.full_like(random_phase, 0.5), random_phase))
        self.swing_foot[ids] = (self.phase[ids] >= 0.5).long()

        self.hold[ids] = False
        self.num_gaits[ids] = 1          # 论文 sample_goal(reset=True) 初始化为 1
        self.stance_reward[ids] = 1.0
        self.step_count[ids] = 0

        if self.last_switch_ids.numel():
            resetting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            resetting[ids] = True
            self.last_switch_ids = self.last_switch_ids[~resetting[self.last_switch_ids]]

    @property
    def hold_still(self):
        # 论文：hold_still = still_phase & (num_gaits >= 2)
        return self.hold & (self.num_gaits >= 2)

    def step(self, dt, foot_pos, foot_quat):
        old_half = torch.floor(self.phase * 2.0)
        self.phase = torch.remainder(self.phase + dt * self.frequency, 1.0)
        new_half = torch.floor(self.phase * 2.0)
        switched = new_half != old_half
        ids = switched.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.swing_foot[ids] = (self.phase[ids] >= 0.5).long()
            self.step_count[ids] += 1
            resample = self.num_gaits[ids] == 0
            if resample.any():
                rs = ids[resample]
                self.hold[rs] = torch.rand(rs.numel(), device=self.device) < self.cfg.hold_probability
            self._sample_next(ids, foot_pos, foot_quat)
            self.num_gaits[ids] = torch.remainder(self.num_gaits[ids] + 1, int(self.cfg.max_num_gaits))
        self.last_switch_ids = ids
        return ids

    def observation(self, foot_pos, foot_quat):
        """返回 16 维：左右目标位置(3)+四元数(wxyz)(4)（支撑脚系），cos/sin(2πφ)。"""
        stance = 1 - self.swing_foot
        batch = torch.arange(self.num_envs, device=self.device)
        origin = foot_pos[batch, stance]
        stance_quat = foot_quat[batch, stance]
        expanded = stance_quat[:, None].expand(-1, 2, -1)
        rel_pos = rotate_inverse(expanded, self.target_pos - origin[:, None])
        rel_quat = quaternion_multiply(quaternion_conjugate(expanded), self.target_quat)
        rel_quat = rel_quat * torch.where(rel_quat[..., 3:4] < 0.0, -1.0, 1.0)
        rel_quat = rel_quat[..., [3, 0, 1, 2]]   # -> scalar-first [w,x,y,z]
        phase = torch.stack((torch.cos(2 * math.pi * self.phase),
                             torch.sin(2 * math.pi * self.phase)), dim=-1)
        steady_hold = self.hold & (self.num_gaits >= 2)
        phase[steady_hold] = 0.0
        return torch.cat((rel_pos[:, 0], rel_quat[:, 0], rel_pos[:, 1], rel_quat[:, 1], phase), dim=-1)

    def _sample_candidate(self, ids_sub, swing, stance_pos, stance_yaw):
        """对子集 ids_sub 采样一个候选目标（对应论文 _get_candidate_target）。"""
        c = self.cfg
        n = ids_sub.numel()
        d = self._uniform(c.step_distance, n)
        alpha = self._uniform(c.step_angle_deg, n, degrees=True)
        angle = self.movement_yaw[ids_sub] + alpha
        target = stance_pos.clone()
        target[:, 0] += d * torch.cos(angle)
        target[:, 1] += d * torch.sin(angle)
        target[:, 2] = 0.0   # 论文：平地目标 z = 地面高度 0

        # 防交叉：在支撑脚 yaw 系下做单侧半平面裁切
        local = world_to_yaw_frame(target, stance_pos, stance_yaw)
        sep = c.minimum_lateral_separation
        local[:, 1] = torch.where(
            swing == 0,
            torch.maximum(local[:, 1], torch.full_like(local[:, 1], sep)),
            torch.minimum(local[:, 1], torch.full_like(local[:, 1], -sep)))
        cs, sn = torch.cos(stance_yaw), torch.sin(stance_yaw)
        target[:, 0] = stance_pos[:, 0] + cs * local[:, 0] - sn * local[:, 1]
        target[:, 1] = stance_pos[:, 1] + sn * local[:, 0] + cs * local[:, 1]

        # hold：保持名义脚间距
        holds = self.hold[ids_sub]
        if holds.any():
            side = torch.where(swing[holds] == 0, 1.0, -1.0)
            local[holds] = 0.0
            local[holds, 1] = side * c.hold_feet_distance
            target[holds, 0] = stance_pos[holds, 0] + cs[holds] * local[holds, 0] - sn[holds] * local[holds, 1]
            target[holds, 1] = stance_pos[holds, 1] + sn[holds] * local[holds, 0] + cs[holds] * local[holds, 1]
            target[holds, 2] = 0.0

        # 目标 yaw
        yaw = self.feet_yaw[ids_sub] + self._uniform(c.target_yaw_deg, n, degrees=True)
        yaw = stance_yaw + torch.clamp(wrap_to_pi(yaw - stance_yaw), -math.pi / 2, math.pi / 2)
        yaw[holds] = stance_yaw[holds]

        return target, yaw

    def _sample_next(self, ids, foot_pos, foot_quat):
        n = ids.numel()
        swing = self.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(n, device=self.device)
        stance_pos = foot_pos[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_quat[ids][row, stance])
        swing_pos = foot_pos[ids][row, swing]
        holds = self.hold[ids]

        # 论文：只更新摆动脚目标；支撑脚目标保留为上一落足目标（落地误差进入观测）。
        # 拒绝采样：候选目标与摆动脚当前位置距离必须 >= min_swing_distance
        #（原论文 _check_valid 的 dist_swing 条件，平地无柱时阈值为 0.05m）。
        min_swing_dist = float(getattr(self.cfg, "min_swing_distance", 0.05))
        max_attempts = int(getattr(self.cfg, "max_rejection_attempts", 10))

        target = torch.zeros(n, 3, device=self.device)
        yaw = torch.zeros(n, device=self.device)

        # hold 目标不做拒绝采样（论文对 hold_still 走 _hold_still_proc，跳过 _check_valid）
        active = ~holds
        for _ in range(max_attempts):
            if not active.any():
                break
            this_round = active.clone()
            a_ids = ids[this_round]
            tgt, yw = self._sample_candidate(
                a_ids, swing[this_round], stance_pos[this_round], stance_yaw[this_round])
            target[this_round] = tgt
            yaw[this_round] = yw
            d_swing = torch.norm(target[this_round, :2] - swing_pos[this_round, :2], dim=1)
            active[this_round] = d_swing < min_swing_dist   # 仍不合法的进入下一轮

        if holds.any():
            tgt, yw = self._sample_candidate(ids[holds], swing[holds], stance_pos[holds], stance_yaw[holds])
            target[holds] = tgt
            yaw[holds] = yw

        self.target_pos[ids, swing] = target
        self.target_yaw[ids, swing] = wrap_to_pi(yaw)
        self.target_quat[ids, swing] = yaw_quaternion(wrap_to_pi(yaw))

