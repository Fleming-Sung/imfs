"""Privileged terrain planner for validating the upper foothold interface.

This module never reads a camera image.  It uses the simulator's exact support
mask and obstacle rectangles to build a jump-aware geodesic distance field,
then scores a small set of direct stance-frame foothold candidates.  The
selected target is still executed by the frozen lower controller, so success
remains a real closed-loop robot result rather than ideal point kinematics.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np
import torch

from .contracts import FootholdActionBounds
from .sampler import quaternion_yaw


@dataclass(frozen=True)
class PrivilegedPlannerConfig:
    # Two-centimetre physical spacing under the V6 Cartesian bounds.  A coarse
    # grid created artificial no-candidate states at support edges and made the
    # controller repeat destabilising minimum-length fallback steps.
    forward_levels: tuple = (
        -1.0, -0.818182, -0.636364, -0.454545, -0.272727, -0.090909,
        0.090909, 0.272727, 0.454545, 0.636364, 0.818182, 1.0)
    lateral_levels: tuple = (
        -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    yaw_levels: tuple = (-1.0, 0.0, 1.0)
    minimum_radial_m: float = 0.12
    maximum_radial_m: float = 0.35
    # A looser mask keeps the global stepping-stone graph connected; executed
    # candidates require a larger support margin because the frozen lower has
    # non-zero touchdown error.
    minimum_support_fraction: float = 5.0 / 9.0
    candidate_support_fraction: float = 7.0 / 9.0
    maximum_route_jump_m: float = 0.35
    # Inflating thin across-corridor hurdles disconnects an otherwise valid
    # route.  Exact footprints are hard constraints; physical clearance is
    # deliberately left for the frozen-lower closed-loop gate to measure.
    obstacle_margin_m: float = 0.0
    candidate_obstacle_margin_m: float = 0.04
    body_path_obstacle_margin_m: float = 0.12
    # Thin, low, corridor-spanning rectangles are hurdles: a foothold cannot
    # be placed on them, but a swing trajectory is allowed to cross them.
    # Wider rectangles remain hard route/body obstacles and must be bypassed.
    maximum_step_over_length_m: float = 0.10
    maximum_step_over_height_m: float = 0.09
    geodesic_weight: float = 10.0
    support_weight: float = 3.0
    heading_weight: float = 0.25
    action_change_weight: float = 0.04


def _shifted(mask, dy, dx):
    """Shift a 2-D mask without wraparound."""
    result = np.zeros_like(mask)
    y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
    x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
    source_y0, source_y1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
    source_x0, source_x1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
    result[y0:y1, x0:x1] = mask[source_y0:source_y1, source_x0:source_x1]
    return result


def sole_support_fraction(support_mask, resolution_m):
    """Nine-point sole support fraction at every terrain cell."""
    x_offset = max(1, int(round(0.08 / float(resolution_m))))
    y_offset = max(1, int(round(0.035 / float(resolution_m))))
    samples = [
        _shifted(support_mask, dy, dx).astype(np.float32)
        for dy in (-y_offset, 0, y_offset)
        for dx in (-x_offset, 0, x_offset)
    ]
    return np.mean(samples, axis=0, dtype=np.float32)


def obstacle_mask(layout, margin_m=0.0, include=None):
    """Rasterize obstacle footprints, optionally inflated in xy."""
    mask = np.zeros_like(layout.support_mask, dtype=bool)
    x, y = layout.x_m, layout.y_m
    margin = float(margin_m)
    for cx, cy, sx, sy, _ in layout.obstacle_rectangles:
        if include is not None and not include((cx, cy, sx, sy, _)):
            continue
        mask |= ((np.abs(x[None, :] - float(cx)) <= 0.5 * float(sx) + margin)
                 & (np.abs(y[:, None] - float(cy)) <= 0.5 * float(sy) + margin))
    return mask


def _line_offsets(dy, dx):
    steps = max(abs(int(dy)), abs(int(dx)))
    if steps <= 1:
        return ()
    values = []
    for index in range(1, steps):
        fraction = index / float(steps)
        values.append((int(round(fraction * dy)), int(round(fraction * dx))))
    return tuple(dict.fromkeys(values))


def jump_geodesic_distance(landing_mask, blocked_mask, goal_yx,
                           resolution_m, maximum_jump_m,
                           return_heading=False):
    """Distance-to-goal over valid landing cells with short gap-crossing edges.

    Edges may cross unsupported gaps up to ``maximum_jump_m`` but never cross
    an obstacle footprint.  This represents a sequence of feasible foothold
    targets rather than requiring the ground between two targets to be solid.
    """
    landing = np.asarray(landing_mask, dtype=bool)
    blocked = np.asarray(blocked_mask, dtype=bool)
    if landing.shape != blocked.shape:
        raise ValueError("landing and blocked masks must have equal shape")
    gy, gx = (int(goal_yx[0]), int(goal_yx[1]))
    valid_y, valid_x = np.nonzero(landing)
    if not len(valid_y):
        empty = np.full(landing.shape, np.inf, dtype=np.float32)
        return ((empty, np.full_like(empty, np.nan))
                if return_heading else empty)
    if not (0 <= gy < landing.shape[0] and 0 <= gx < landing.shape[1]) \
            or not landing[gy, gx]:
        nearest = np.argmin((valid_y - gy) ** 2 + (valid_x - gx) ** 2)
        gy, gx = int(valid_y[nearest]), int(valid_x[nearest])

    radius = max(1, int(math.floor(
        float(maximum_jump_m) / float(resolution_m) + 1e-6)))
    neighbours = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cells = math.hypot(dy, dx)
            if cells == 0.0 or cells * resolution_m > maximum_jump_m + 1e-9:
                continue
            neighbours.append((dy, dx, cells * resolution_m,
                               _line_offsets(dy, dx)))

    distance = np.full(landing.shape, np.inf, dtype=np.float64)
    heading = np.full(landing.shape, np.nan, dtype=np.float32)
    distance[gy, gx] = 0.0
    queue = [(0.0, gy, gx)]
    height, width = landing.shape
    while queue:
        value, y, x = heapq.heappop(queue)
        if value != distance[y, x]:
            continue
        for dy, dx, cost, line in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width) or not landing[ny, nx]:
                continue
            if any(blocked[y + ly, x + lx] for ly, lx in line):
                continue
            candidate = value + cost
            if candidate < distance[ny, nx]:
                distance[ny, nx] = candidate
                # Relaxation runs outward from the goal.  From the newly
                # reached cell, (-dx,-dy) points back toward the lower-cost
                # predecessor and therefore along the route to the goal.
                heading[ny, nx] = math.atan2(-dy, -dx)
                heapq.heappush(queue, (candidate, ny, nx))
    distance = distance.astype(np.float32)
    return (distance, heading) if return_heading else distance


class PrivilegedTerrainPlanner:
    """Batched exact-terrain candidate planner for the frozen lower policy."""

    def __init__(self, tiled, bounds, device, config=None):
        if not isinstance(bounds, FootholdActionBounds):
            raise TypeError(
                "privileged planner requires direct cartesian foothold bounds")
        self.tiled = tiled
        self.bounds = bounds
        self.device = torch.device(device)
        self.config = config or PrivilegedPlannerConfig()
        self.resolution_m = float(tiled.horizontal_scale_m)
        self.env_origins = torch.as_tensor(
            tiled.env_origins_xy_m, dtype=torch.float32, device=self.device)

        candidate = [
            (forward, lateral, yaw)
            for forward in self.config.forward_levels
            for lateral in self.config.lateral_levels
            for yaw in self.config.yaw_levels
        ]
        self.candidates = torch.tensor(
            candidate, dtype=torch.float32, device=self.device)
        decoded = self.bounds.decode(
            self.candidates,
            torch.zeros(len(self.candidates), dtype=torch.long,
                        device=self.device))
        radial = torch.linalg.norm(decoded[:, :2], dim=-1)
        reachable = ((radial >= self.config.minimum_radial_m)
                     & (radial <= self.config.maximum_radial_m + 1e-6))
        self.candidates = self.candidates[reachable]

        fractions, landing_masks, obstacle_masks, body_path_masks = [], [], [], []
        distances, headings = [], []
        goals = []
        for layout in tiled.layouts:
            fraction = sole_support_fraction(
                np.asarray(layout.support_mask, dtype=bool), self.resolution_m)
            all_obstacles = obstacle_mask(layout, self.config.obstacle_margin_m)
            hard_obstacle = lambda item: not (
                float(item[2]) <= self.config.maximum_step_over_length_m
                and float(item[4]) <= self.config.maximum_step_over_height_m)
            route_blockers = obstacle_mask(
                layout, self.config.obstacle_margin_m, include=hard_obstacle)
            route_landing = ((fraction >= self.config.minimum_support_fraction)
                             & ~all_obstacles)
            candidate_obstacles = obstacle_mask(
                layout, self.config.candidate_obstacle_margin_m)
            landing = ((fraction >= self.config.candidate_support_fraction)
                       & ~candidate_obstacles)
            body_path = obstacle_mask(
                layout, self.config.body_path_obstacle_margin_m,
                include=hard_obstacle)
            goal_x = int(round(
                (float(layout.goal_xy[0]) - float(layout.x_m[0]))
                / self.resolution_m))
            goal_y = int(round(
                (float(layout.goal_xy[1]) - float(layout.y_m[0]))
                / self.resolution_m))
            distance, heading = jump_geodesic_distance(
                route_landing, route_blockers, (goal_y, goal_x), self.resolution_m,
                self.config.maximum_route_jump_m, return_heading=True)
            fractions.append(fraction)
            landing_masks.append(landing)
            obstacle_masks.append(all_obstacles)
            body_path_masks.append(body_path)
            distances.append(distance)
            headings.append(heading)
            goals.append(layout.goal_xy)
        self.support_fraction = torch.as_tensor(
            np.stack(fractions), dtype=torch.float32, device=self.device)
        self.landing_mask = torch.as_tensor(
            np.stack(landing_masks), dtype=torch.bool, device=self.device)
        self.obstacle_mask = torch.as_tensor(
            np.stack(obstacle_masks), dtype=torch.bool, device=self.device)
        self.body_path_obstacle_mask = torch.as_tensor(
            np.stack(body_path_masks), dtype=torch.bool, device=self.device)
        self.distance = torch.as_tensor(
            np.stack(distances), dtype=torch.float32, device=self.device)
        self.route_heading = torch.as_tensor(
            np.stack(headings), dtype=torch.float32, device=self.device)
        self.goals = torch.as_tensor(
            np.stack(goals), dtype=torch.float32, device=self.device)
        first = tiled.layouts[0]
        self.local_origin = torch.tensor(
            [float(first.x_m[0]), float(first.y_m[0])],
            dtype=torch.float32, device=self.device)

    def _grid_indices(self, local_xy):
        ix = torch.round(
            (local_xy[..., 0] - self.local_origin[0])
            / self.resolution_m).long()
        iy = torch.round(
            (local_xy[..., 1] - self.local_origin[1])
            / self.resolution_m).long()
        valid = ((ix >= 0) & (ix < self.distance.shape[2])
                 & (iy >= 0) & (iy < self.distance.shape[1]))
        return (iy.clamp(0, self.distance.shape[1] - 1),
                ix.clamp(0, self.distance.shape[2] - 1), valid)

    def _sample(self, field, env_ids, local_xy):
        iy, ix, valid = self._grid_indices(local_xy)
        expanded_ids = env_ids.view(
            env_ids.shape + (1,) * (local_xy.ndim - 2)).expand_as(ix)
        value = field[expanded_ids, iy, ix]
        return value, valid

    def geodesic_distance(self, env, ids, stance_world_xy):
        """Privileged geodesic distance from each stance position to its goal,
        with the same Euclidean fallback used by plan()."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        stance_world_xy = torch.as_tensor(
            stance_world_xy, dtype=torch.float32, device=self.device)
        local = stance_world_xy - self.env_origins[ids]
        distance, inside = self._sample(self.distance, ids, local)
        direct = torch.linalg.norm(local - self.goals[ids], dim=-1)
        return torch.where(inside & torch.isfinite(distance), distance, direct)

    def support_fraction_at(self, env, ids, world_xy):
        """Privileged 9-point support fraction at the given world xy position."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        world_xy = torch.as_tensor(
            world_xy, dtype=torch.float32, device=self.device)
        local = world_xy - self.env_origins[ids]
        fraction, _ = self._sample(self.support_fraction, ids, local)
        return fraction

    @torch.no_grad()
    def plan(self, env, ids, previous_action=None):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return self.candidates.new_zeros((0, 3)), {}
        swing = env.sampler.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(ids.numel(), device=self.device)
        foot_state = env.rigid_body_states[ids][:, env.feet_indices]
        stance_position = env.foot_positions[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_state[row, stance, 3:7])

        actions = self.candidates[None].expand(ids.numel(), -1, -1)
        swing_batch = swing[:, None].expand(-1, actions.shape[1])
        local_target = self.bounds.decode(actions, swing_batch)
        cosine, sine = torch.cos(stance_yaw)[:, None], torch.sin(stance_yaw)[:, None]
        target_world = torch.empty(
            ids.numel(), actions.shape[1], 2, device=self.device)
        target_world[..., 0] = (stance_position[:, None, 0]
                                + cosine * local_target[..., 0]
                                - sine * local_target[..., 1])
        target_world[..., 1] = (stance_position[:, None, 1]
                                + sine * local_target[..., 0]
                                + cosine * local_target[..., 1])
        target_local = target_world - self.env_origins[ids, None]

        support, inside = self._sample(self.support_fraction, ids, target_local)
        landing, _ = self._sample(self.landing_mask, ids, target_local)
        candidate_distance, _ = self._sample(self.distance, ids, target_local)
        route_heading, _ = self._sample(
            self.route_heading, ids, target_local)

        stance_local = stance_position[:, :2] - self.env_origins[ids]
        current_distance, current_inside = self._sample(
            self.distance, ids, stance_local)
        direct_current = torch.linalg.norm(
            stance_local - self.goals[ids], dim=-1)
        current_distance = torch.where(
            current_inside & torch.isfinite(current_distance),
            current_distance, direct_current)
        direct_candidate = torch.linalg.norm(
            target_local - self.goals[ids, None], dim=-1)
        candidate_distance = torch.where(
            torch.isfinite(candidate_distance), candidate_distance,
            direct_candidate + 2.0)
        geodesic_progress = current_distance[:, None] - candidate_distance

        # Reject a first step whose straight swing-target segment crosses an
        # obstacle footprint, even if both endpoint cells are valid.
        path_blocked = torch.zeros_like(landing)
        for fraction in (0.25, 0.50, 0.75):
            sample = (stance_local[:, None]
                      + fraction * (target_local - stance_local[:, None]))
            blocked, sample_inside = self._sample(
                self.body_path_obstacle_mask, ids, sample)
            path_blocked |= blocked | ~sample_inside

        target_yaw = stance_yaw[:, None] + local_target[..., 3]
        heading_error = torch.atan2(
            torch.sin(target_yaw - route_heading),
            torch.cos(target_yaw - route_heading)).abs()
        heading_error = torch.where(
            torch.isfinite(heading_error), heading_error,
            torch.zeros_like(heading_error))
        score = (self.config.geodesic_weight * geodesic_progress
                 + self.config.support_weight * support
                 - self.config.heading_weight * heading_error)
        if previous_action is not None:
            previous = torch.as_tensor(
                previous_action, dtype=torch.float32, device=self.device)
            score -= (self.config.action_change_weight
                      * (actions - previous[:, None]).square().mean(dim=-1))
        valid = inside & landing & ~path_blocked
        score = torch.where(valid, score, torch.full_like(score, -1.0e9))
        chosen_index = score.argmax(dim=1)
        chosen = actions[row, chosen_index]
        no_valid = ~valid.any(dim=1)
        if no_valid.any():
            # Stay within the direct action contract while selecting the least
            # unsupported candidate.  The diagnostic exposes this condition;
            # it must not be silently counted as successful planning.
            fallback = support.argmax(dim=1)
            chosen[no_valid] = actions[row[no_valid], fallback[no_valid]]
            chosen_index[no_valid] = fallback[no_valid]
        return chosen, {
            "candidate_index": chosen_index,
            "candidate_valid_count": valid.sum(dim=1),
            "chosen_support_fraction": support[row, chosen_index],
            "chosen_geodesic_progress_m": geodesic_progress[row, chosen_index],
            "chosen_heading_error_rad": heading_error[row, chosen_index],
            "chosen_target_world_xy": target_world[row, chosen_index],
            "fallback": no_valid,
            # Full per-candidate privileged labels for Gate B/C distillation.
            "candidate_support_fraction": support,
            "candidate_geodesic_progress_m": geodesic_progress,
            "candidate_valid": valid,
            "candidate_score": score,
        }
