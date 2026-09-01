"""Explicit sensor and action contracts shared by training and deployment."""

from dataclasses import dataclass

import numpy as np
import torch


PROPRIO_DIM = 36
UPPER_ACTION_DIM = 3
# Macro motion state used to supervise the latent dynamics: base height (1),
# base linear velocity (3), base angular velocity (3), projected gravity (3),
# feet in base frame (6), gait phase (2), relative goal + yaw error (3).
MACRO_STATE_DIM = 21


@dataclass(frozen=True)
class FootholdActionBounds:
    forward_m: tuple
    lateral_abs_m: tuple
    yaw_deg: tuple
    z_m: float = 0.0

    @classmethod
    def from_config(cls, cfg):
        return cls(tuple(cfg["forward_m"]), tuple(cfg["lateral_abs_m"]),
                   tuple(cfg["yaw_deg"]), float(cfg["z_m"]))

    @staticmethod
    def _scale(value, limits):
        low, high = limits
        return low + 0.5 * (value + 1.0) * (high - low)

    def decode(self, normalized_action, swing_foot):
        """Map [-1,1]^3 to stance-yaw-frame xyz/yaw; left=0, right=1."""
        if torch.is_tensor(normalized_action):
            action = normalized_action.clamp(-1.0, 1.0)
            swing = torch.as_tensor(swing_foot, device=action.device)
            side = torch.where(swing == 0, 1.0, -1.0).to(action.dtype)
            xyz_yaw = torch.stack((
                self._scale(action[..., 0], self.forward_m),
                side * self._scale(action[..., 1], self.lateral_abs_m),
                torch.zeros_like(action[..., 0]) + self.z_m,
                torch.deg2rad(self._scale(action[..., 2], self.yaw_deg)),
            ), dim=-1)
            return xyz_yaw
        action = np.clip(np.asarray(normalized_action, dtype=np.float32), -1.0, 1.0)
        swing = np.asarray(swing_foot)
        side = np.where(swing == 0, 1.0, -1.0).astype(np.float32)
        return np.stack((
            self._scale(action[..., 0], self.forward_m),
            side * self._scale(action[..., 1], self.lateral_abs_m),
            np.zeros_like(action[..., 0]) + self.z_m,
            np.deg2rad(self._scale(action[..., 2], self.yaw_deg)),
        ), axis=-1)


@dataclass(frozen=True)
class PolarFootholdActionBounds:
    """Distance/direction/yaw action inside the frozen lower sampler support."""

    distance_m: tuple
    direction_deg: tuple
    yaw_deg: tuple
    minimum_lateral_abs_m: float
    z_m: float = 0.0

    @classmethod
    def from_config(cls, cfg):
        return cls(tuple(cfg["distance_m"]), tuple(cfg["direction_deg"]),
                   tuple(cfg["yaw_deg"]), float(cfg["minimum_lateral_abs_m"]),
                   float(cfg["z_m"]))

    @staticmethod
    def _scale(value, limits):
        low, high = limits
        return low + 0.5 * (value + 1.0) * (high - low)

    def decode(self, normalized_action, swing_foot):
        action = torch.as_tensor(normalized_action).clamp(-1.0, 1.0)
        swing = torch.as_tensor(swing_foot, device=action.device)
        distance = self._scale(action[..., 0], self.distance_m)
        direction = torch.deg2rad(self._scale(action[..., 1], self.direction_deg))
        lateral = distance * torch.sin(direction)
        minimum = torch.full_like(lateral, self.minimum_lateral_abs_m)
        lateral = torch.where(swing == 0, torch.maximum(lateral, minimum),
                              torch.minimum(lateral, -minimum))
        return torch.stack((
            distance * torch.cos(direction), lateral,
            torch.zeros_like(distance) + self.z_m,
            torch.deg2rad(self._scale(action[..., 2], self.yaw_deg)),
        ), dim=-1)


def preprocess_isaac_depth(raw_depth, near_m, far_m):
    """Convert Isaac Gym negative view-axis depth to normalized proximity [0,1]."""
    if torch.is_tensor(raw_depth):
        distance = -raw_depth
        distance = torch.nan_to_num(distance, nan=far_m, posinf=far_m, neginf=far_m)
        distance = distance.clamp(float(near_m), float(far_m))
        return (float(far_m) - distance) / (float(far_m) - float(near_m))
    distance = -np.asarray(raw_depth, dtype=np.float32)
    distance = np.nan_to_num(distance, nan=far_m, posinf=far_m, neginf=far_m)
    distance = np.clip(distance, near_m, far_m)
    return (far_m - distance) / (far_m - near_m)
