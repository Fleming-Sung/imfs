"""MuJoCo version of the training goal sampler with explicit demo overrides."""

import numpy as np
import mujoco

from .course import FOOT_SITE_NAMES
from .math_utils import canonicalize, quat_conjugate, quat_multiply, rotate_inverse, yaw_quaternion


def wrap_to_pi(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_yaw(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TrainingGoalSampler:
    """Single-environment translation of foothold.sampler.FootholdSampler.

    New targets are sampled from the live stance foot at every half-cycle.  The
    random-number generator differs from PyTorch. State transitions match training;
    selected sampling bounds may be overridden by the deployment configuration.
    """

    def __init__(self, model, data, train_cfg, demo_cfg, seed):
        self.model = model
        self.data = data
        self.cfg = dict(train_cfg["foothold"])
        self.training_minimum_lateral_separation = float(
            train_cfg["foothold"]["minimum_lateral_separation"])
        for name in ("step_distance", "step_angle_deg", "movement_direction_deg", "target_yaw_deg"):
            if name in demo_cfg:
                self.cfg[name] = list(map(float, demo_cfg[name]))
        if "minimum_lateral_separation" in demo_cfg:
            self.cfg["minimum_lateral_separation"] = float(
                demo_cfg["minimum_lateral_separation"])
        for name in ("step_distance", "step_angle_deg", "movement_direction_deg", "target_yaw_deg"):
            if len(self.cfg[name]) != 2 or float(self.cfg[name][0]) > float(self.cfg[name][1]):
                raise ValueError(f"{name} must be [MIN, MAX] with MIN <= MAX")
        if float(self.cfg["step_distance"][0]) <= 0.0:
            raise ValueError("step_distance MIN must be positive")
        if float(self.cfg["minimum_lateral_separation"]) <= 0.0:
            raise ValueError("minimum_lateral_separation must be positive")
        self.rng = np.random.default_rng(int(seed))
        self.foot_site_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITE_NAMES
        ])
        self.foot_body_ids = model.site_bodyid[self.foot_site_ids]
        self.target_pos = np.zeros((2, 3), dtype=np.float64)
        self.target_quat = np.tile(yaw_quaternion(0.0), (2, 1))
        self.target_index = [0, 1]
        self.next_target_index = 2
        self.phase = 0.0
        self.frequency = 1.0
        self.swing_foot = 0
        self.movement_yaw = 0.0
        self.feet_yaw = 0.0
        self.hold = False
        self.num_gaits = 1
        self.target_history = []
        self.movement_origin = np.zeros(3, dtype=np.float64)

    def _uniform(self, limits, degrees=False):
        value = self.rng.uniform(float(limits[0]), float(limits[1]))
        return np.deg2rad(value) if degrees else value

    @property
    def hold_still(self):
        return self.hold and self.num_gaits >= 2

    def reset(self):
        self.target_pos[:] = self.data.site_xpos[self.foot_site_ids]
        self.target_quat[:] = self.data.xquat[self.foot_body_ids]
        self.target_index[:] = [0, 1]
        self.next_target_index = 2
        self.movement_yaw = self._uniform(self.cfg["movement_direction_deg"], degrees=True)
        self.feet_yaw = self._uniform(self.cfg["feet_direction_deg"], degrees=True)
        self.frequency = self._uniform(self.cfg["gait_frequency"])
        relative_direction = wrap_to_pi(self.movement_yaw - self.feet_yaw)
        random_phase = 0.5 * int(self.rng.integers(0, 2))
        self.phase = 0.0 if relative_direction > 0.0 else (
            0.5 if relative_direction < 0.0 else random_phase)
        self.swing_foot = int(self.phase >= 0.5)
        self.hold = False
        self.num_gaits = 1
        self.movement_origin = self.target_pos.mean(axis=0)
        self.movement_origin[2] = 0.018
        self.target_history = []
        for foot in (0, 1):
            self._record_target(foot)

    def _record_target(self, foot):
        self.target_history.append({
            "time_s": float(self.data.time),
            "index": int(self.target_index[foot]),
            "foot": "left" if foot == 0 else "right",
            "position": self.target_pos[foot].tolist(),
            "quaternion_wxyz": self.target_quat[foot].tolist(),
            "hold": bool(self.hold),
        })

    def _sample_candidate(self, swing, stance_pos, stance_yaw):
        distance = self._uniform(self.cfg["step_distance"])
        alpha = self._uniform(self.cfg["step_angle_deg"], degrees=True)
        angle = self.movement_yaw + alpha
        target = stance_pos.copy()
        target[0] += distance * np.cos(angle)
        target[1] += distance * np.sin(angle)
        target[2] = 0.0

        c, s = np.cos(stance_yaw), np.sin(stance_yaw)
        delta = target - stance_pos
        local = np.array([c * delta[0] + s * delta[1],
                          -s * delta[0] + c * delta[1], delta[2]])
        separation = float(self.cfg["minimum_lateral_separation"])
        local[1] = max(local[1], separation) if swing == 0 else min(local[1], -separation)
        if self.hold:
            local[:] = [0.0, float(self.cfg["hold_feet_distance"]) * (1.0 if swing == 0 else -1.0), 0.0]
        target[0] = stance_pos[0] + c * local[0] - s * local[1]
        target[1] = stance_pos[1] + s * local[0] + c * local[1]
        target[2] = 0.0

        target_yaw = self.feet_yaw + self._uniform(self.cfg["target_yaw_deg"], degrees=True)
        target_yaw = stance_yaw + np.clip(wrap_to_pi(target_yaw - stance_yaw), -np.pi / 2, np.pi / 2)
        if self.hold:
            target_yaw = stance_yaw
        return target, wrap_to_pi(target_yaw)

    def _sample_next(self):
        swing = self.swing_foot
        stance = 1 - swing
        stance_pos = self.data.site_xpos[self.foot_site_ids[stance]].copy()
        stance_yaw = quaternion_yaw(self.data.xquat[self.foot_body_ids[stance]])
        swing_pos = self.data.site_xpos[self.foot_site_ids[swing]]
        attempts = int(self.cfg.get("max_rejection_attempts", 10))
        minimum = float(self.cfg.get("min_swing_distance", 0.05))
        target = yaw = None
        for _ in range(attempts):
            target, yaw = self._sample_candidate(swing, stance_pos, stance_yaw)
            if self.hold or np.linalg.norm(target[:2] - swing_pos[:2]) >= minimum:
                break
        self.target_pos[swing] = target
        self.target_quat[swing] = yaw_quaternion(yaw)
        self.target_index[swing] = self.next_target_index
        self.next_target_index += 1
        self._record_target(swing)

    def advance(self, policy_dt):
        old_half = int(np.floor(self.phase * 2.0))
        self.phase = (self.phase + float(policy_dt) * self.frequency) % 1.0
        new_half = int(np.floor(self.phase * 2.0))
        if new_half == old_half:
            return False
        self.swing_foot = int(self.phase >= 0.5)
        if self.num_gaits == 0:
            self.hold = bool(self.rng.random() < float(self.cfg["hold_probability"]))
        self._sample_next()
        self.num_gaits = (self.num_gaits + 1) % int(self.cfg["max_num_gaits"])
        return True

    def foothold_positions(self):
        return self.target_pos.copy()

    def target_positions(self):
        return self.target_pos.copy()

    def target_quaternions(self):
        return self.target_quat.copy()

    def target_supports(self):
        return ["flat", "flat"]

    def movement_quaternion(self):
        return yaw_quaternion(self.movement_yaw)

    def effective_settings(self):
        return {
            "step_distance": list(map(float, self.cfg["step_distance"])),
            "step_angle_deg": list(map(float, self.cfg["step_angle_deg"])),
            "movement_direction_deg": list(map(float, self.cfg["movement_direction_deg"])),
            "target_yaw_deg": list(map(float, self.cfg["target_yaw_deg"])),
            "minimum_lateral_separation": float(self.cfg["minimum_lateral_separation"]),
            "training_minimum_lateral_separation": self.training_minimum_lateral_separation,
            "sampled_movement_yaw_deg": float(np.degrees(self.movement_yaw)),
        }

    def observation(self):
        stance = 1 - self.swing_foot
        origin = self.data.site_xpos[self.foot_site_ids[stance]]
        stance_quat = self.data.xquat[self.foot_body_ids[stance]]
        parts = []
        for foot in (0, 1):
            rel_pos = rotate_inverse(stance_quat, self.target_pos[foot] - origin)
            rel_quat = canonicalize(quat_multiply(
                quat_conjugate(stance_quat), self.target_quat[foot]))
            parts.extend((rel_pos, rel_quat))
        phase = np.zeros(2) if self.hold_still else np.array([
            np.cos(2.0 * np.pi * self.phase), np.sin(2.0 * np.pi * self.phase)])
        return np.concatenate((*parts, phase)).astype(np.float32)
