"""Training-only exact geometry labels for every foothold candidate.

This module does not choose actions. It annotates the state-action grid with
support and geodesic progress. Vertical obstacle footprints are inflated into
a training-only traversability mask; deployment still receives only depth and
proprioception and never reads this exact map.
"""

import heapq
import math

import numpy as np
import torch

from .frozen_lower_env.sampler import quaternion_yaw


def _shift(mask, dy, dx):
    result = np.zeros_like(mask)
    y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
    x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
    sy0, sy1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
    sx0, sx1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
    result[y0:y1, x0:x1] = mask[sy0:sy1, sx0:sx1]
    return result


def sole_support_fraction(mask, resolution_m):
    x_offset = max(1, int(round(0.08 / resolution_m)))
    y_offset = max(1, int(round(0.035 / resolution_m)))
    samples = [_shift(mask, dy, dx).astype(np.float32)
               for dy in (-y_offset, 0, y_offset)
               for dx in (-x_offset, 0, x_offset)]
    return np.mean(samples, axis=0, dtype=np.float32)


def jump_distance(landing, goal_yx, resolution_m, maximum_jump_m=0.35):
    """Dijkstra field over landing cells with physically reachable gap edges."""
    landing = np.asarray(landing, dtype=bool)
    valid_y, valid_x = np.nonzero(landing)
    distance = np.full(landing.shape, np.inf, dtype=np.float64)
    if not len(valid_y):
        return distance.astype(np.float32)
    gy, gx = map(int, goal_yx)
    if not (0 <= gy < landing.shape[0] and 0 <= gx < landing.shape[1]) \
            or not landing[gy, gx]:
        nearest = np.argmin((valid_y - gy) ** 2 + (valid_x - gx) ** 2)
        gy, gx = int(valid_y[nearest]), int(valid_x[nearest])
    radius = int(math.floor(maximum_jump_m / resolution_m + 1e-6))
    neighbours = [(dy, dx, math.hypot(dy, dx) * resolution_m)
                  for dy in range(-radius, radius + 1)
                  for dx in range(-radius, radius + 1)
                  if 0 < math.hypot(dy, dx) * resolution_m
                  <= maximum_jump_m + 1e-9]
    distance[gy, gx] = 0.0
    queue = [(0.0, gy, gx)]
    while queue:
        value, y, x = heapq.heappop(queue)
        if value != distance[y, x]:
            continue
        for dy, dx, cost in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < landing.shape[0] and 0 <= nx < landing.shape[1]):
                continue
            if not landing[ny, nx]:
                continue
            candidate = value + cost
            if candidate < distance[ny, nx]:
                distance[ny, nx] = candidate
                heapq.heappush(queue, (candidate, ny, nx))
    return distance.astype(np.float32)


class CandidateGeometryLabeler:
    def __init__(self, tiled, bounds, candidates, device,
                 obstacle_clearance_m=0.18):
        self.device = torch.device(device)
        self.bounds = bounds
        self.candidates = torch.as_tensor(
            candidates, dtype=torch.float32, device=self.device)
        self.resolution = float(tiled.horizontal_scale_m)
        fractions, distances, route_cosines, route_sines, goals = [], [], [], [], []
        for layout in tiled.layouts:
            fraction = sole_support_fraction(
                np.asarray(layout.support_mask, dtype=bool), self.resolution)
            blocked = np.zeros_like(layout.support_mask, dtype=bool)
            for cx, cy, sx, sy, _height in layout.obstacle_rectangles:
                blocked |= (
                    np.abs(layout.x_m[None, :] - cx)
                    <= 0.5 * sx + obstacle_clearance_m) & (
                    np.abs(layout.y_m[:, None] - cy)
                    <= 0.5 * sy + obstacle_clearance_m)
            goal_x = int(round((layout.goal_xy[0] - layout.x_m[0]) / self.resolution))
            goal_y = int(round((layout.goal_xy[1] - layout.y_m[0]) / self.resolution))
            fractions.append(fraction)
            distance = jump_distance(
                (fraction >= 5.0 / 9.0) & ~blocked,
                (goal_y, goal_x), self.resolution)
            distances.append(distance)
            finite = np.isfinite(distance)
            fill = float(distance[finite].max() + 2.0) if finite.any() else 2.0
            gradient_y, gradient_x = np.gradient(
                np.where(finite, distance, fill), self.resolution,
                self.resolution)
            direction_x, direction_y = -gradient_x, -gradient_y
            norm = np.hypot(direction_x, direction_y)
            route_cosine = np.ones_like(norm)
            route_sine = np.zeros_like(norm)
            np.divide(direction_x, norm, out=route_cosine, where=norm > 1e-6)
            np.divide(direction_y, norm, out=route_sine, where=norm > 1e-6)
            route_cosines.append(route_cosine)
            route_sines.append(route_sine)
            goals.append(layout.goal_xy)
        self.support = torch.as_tensor(
            np.stack(fractions), dtype=torch.float32, device=self.device)
        self.distance = torch.as_tensor(
            np.stack(distances), dtype=torch.float32, device=self.device)
        self.route_cosine = torch.as_tensor(
            np.stack(route_cosines), dtype=torch.float32, device=self.device)
        self.route_sine = torch.as_tensor(
            np.stack(route_sines), dtype=torch.float32, device=self.device)
        self.goals = torch.as_tensor(
            np.stack(goals), dtype=torch.float32, device=self.device)
        first = tiled.layouts[0]
        self.local_origin = torch.tensor(
            [float(first.x_m[0]), float(first.y_m[0])],
            dtype=torch.float32, device=self.device)
        self.env_origins = torch.as_tensor(
            tiled.env_origins_xy_m, dtype=torch.float32, device=self.device)

    def _sample(self, field, ids, xy):
        ix = torch.round((xy[..., 0] - self.local_origin[0]) / self.resolution).long()
        iy = torch.round((xy[..., 1] - self.local_origin[1]) / self.resolution).long()
        inside = ((ix >= 0) & (ix < field.shape[2])
                  & (iy >= 0) & (iy < field.shape[1]))
        ix = ix.clamp(0, field.shape[2] - 1)
        iy = iy.clamp(0, field.shape[1] - 1)
        expanded = ids.view(ids.shape + (1,) * (xy.ndim - 2)).expand_as(ix)
        return field[expanded, iy, ix], inside

    @torch.no_grad()
    def label(self, env, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        swing = env.sampler.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(len(ids), device=self.device)
        foot_state = env.rigid_body_states[ids][:, env.feet_indices]
        stance_position = env.foot_positions[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_state[row, stance, 3:7])
        actions = self.candidates.unsqueeze(0).expand(len(ids), -1, -1)
        local = self.bounds.decode(
            actions, swing[:, None].expand(-1, actions.shape[1]))
        cosine, sine = torch.cos(stance_yaw)[:, None], torch.sin(stance_yaw)[:, None]
        world = torch.empty(len(ids), actions.shape[1], 2, device=self.device)
        world[..., 0] = (stance_position[:, None, 0]
                         + cosine * local[..., 0] - sine * local[..., 1])
        world[..., 1] = (stance_position[:, None, 1]
                         + sine * local[..., 0] + cosine * local[..., 1])
        target_local = world - self.env_origins[ids, None]
        support, inside = self._sample(self.support, ids, target_local)
        candidate_distance, _ = self._sample(self.distance, ids, target_local)
        route_cosine, _ = self._sample(self.route_cosine, ids, target_local)
        route_sine, _ = self._sample(self.route_sine, ids, target_local)
        target_yaw = stance_yaw[:, None] + local[..., 3]
        alignment = (torch.cos(target_yaw) * route_cosine
                     + torch.sin(target_yaw) * route_sine)
        stance_local = stance_position[:, :2] - self.env_origins[ids]
        current_distance, current_inside = self._sample(
            self.distance, ids, stance_local)
        direct_current = torch.linalg.norm(stance_local - self.goals[ids], dim=-1)
        current_distance = torch.where(
            current_inside & torch.isfinite(current_distance),
            current_distance, direct_current)
        valid = inside & torch.isfinite(candidate_distance) & (support >= 7.0 / 9.0)
        direct_candidate = torch.linalg.norm(
            target_local - self.goals[ids, None], dim=-1)
        candidate_distance = torch.where(
            torch.isfinite(candidate_distance), candidate_distance,
            direct_candidate + 2.0)
        return {
            "candidate_support": support,
            "candidate_progress": current_distance[:, None] - candidate_distance,
            "candidate_valid": valid,
            "candidate_alignment": alignment,
        }
