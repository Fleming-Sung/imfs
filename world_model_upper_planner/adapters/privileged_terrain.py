"""Fast training-only observation from the simulator's true height field."""

import numpy as np
import torch

from adapters.frozen_lower_env.sampler import quaternion_yaw


class PrivilegedTerrainObserver:
    """Render an ego-centric metric height map without camera sensors.

    The returned tensor deliberately keeps the existing [N,1,64,64] contract,
    so model and planner code are unchanged. Values encode metric height in
    [-0.40, +0.40] m as [0,1]. This is privileged training/evaluation input,
    never a claim about deployable perception.
    """

    def __init__(self, tiled, device, height=64, width=64,
                 forward_range_m=(-0.50, 2.65), lateral_range_m=(-1.575, 1.575)):
        self.device = torch.device(device)
        self.height = int(height)
        self.width = int(width)
        # `height_samples` contains the support surface, pits and heightfield
        # hurdles. Household/research obstacles are separate static-box actors,
        # so explicitly rasterize their true footprints and heights into the
        # privileged map. Without this overlay the planner is blind to exactly
        # the obstacles it is expected to avoid.
        samples = np.asarray(tiled.height_samples).copy()
        scale = float(tiled.horizontal_scale_m)
        vertical = float(tiled.vertical_scale_m)
        origin = np.asarray(tiled.origin_xy_m, dtype=np.float32)
        obstacle_cells = 0
        for env_origin, layout in zip(tiled.env_origins_xy_m, tiled.layouts):
            for cx, cy, sx, sy, obstacle_height in layout.obstacle_rectangles:
                world_cx = float(env_origin[0]) + float(cx)
                world_cy = float(env_origin[1]) + float(cy)
                ix0 = max(0, int(np.floor(
                    (world_cx - 0.5 * float(sx) - origin[0]) / scale)))
                ix1 = min(samples.shape[0], int(np.ceil(
                    (world_cx + 0.5 * float(sx) - origin[0]) / scale)) + 1)
                iy0 = max(0, int(np.floor(
                    (world_cy - 0.5 * float(sy) - origin[1]) / scale)))
                iy1 = min(samples.shape[1], int(np.ceil(
                    (world_cy + 0.5 * float(sy) - origin[1]) / scale)) + 1)
                value = int(round(float(obstacle_height) / vertical))
                samples[ix0:ix1, iy0:iy1] = np.maximum(
                    samples[ix0:ix1, iy0:iy1], value)
                obstacle_cells += max(0, ix1 - ix0) * max(0, iy1 - iy0)
        self.samples = torch.as_tensor(samples, device=self.device)
        self.obstacle_cells = int(obstacle_cells)
        self.origin = torch.as_tensor(
            tiled.origin_xy_m, dtype=torch.float32, device=self.device)
        self.horizontal_scale = float(tiled.horizontal_scale_m)
        self.vertical_scale = float(tiled.vertical_scale_m)
        forward = torch.linspace(*forward_range_m, self.height, device=self.device)
        lateral = torch.linspace(*lateral_range_m, self.width, device=self.device)
        self.forward, self.lateral = torch.meshgrid(
            forward, lateral, indexing="ij")

    @torch.no_grad()
    def __call__(self, env):
        yaw = quaternion_yaw(env.base_quat)
        cosine = torch.cos(yaw)[:, None, None]
        sine = torch.sin(yaw)[:, None, None]
        x = (env.base_position[:, 0, None, None]
             + cosine * self.forward - sine * self.lateral)
        y = (env.base_position[:, 1, None, None]
             + sine * self.forward + cosine * self.lateral)
        ix = torch.round((x - self.origin[0]) / self.horizontal_scale).long()
        iy = torch.round((y - self.origin[1]) / self.horizontal_scale).long()
        inside = ((ix >= 0) & (ix < self.samples.shape[0])
                  & (iy >= 0) & (iy < self.samples.shape[1]))
        ix = ix.clamp(0, self.samples.shape[0] - 1)
        iy = iy.clamp(0, self.samples.shape[1] - 1)
        height_m = self.samples[ix, iy].float() * self.vertical_scale
        height_m = torch.where(inside, height_m, height_m.new_full((), -0.40))
        return ((height_m.clamp(-0.40, 0.40) + 0.40) / 0.80).unsqueeze(1)

    def _sample_world(self, xy):
        ix = torch.round((xy[..., 0] - self.origin[0]) /
                         self.horizontal_scale).long()
        iy = torch.round((xy[..., 1] - self.origin[1]) /
                         self.horizontal_scale).long()
        inside = ((ix >= 0) & (ix < self.samples.shape[0])
                  & (iy >= 0) & (iy < self.samples.shape[1]))
        ix = ix.clamp(0, self.samples.shape[0] - 1)
        iy = iy.clamp(0, self.samples.shape[1] - 1)
        height = self.samples[ix, iy].float() * self.vertical_scale
        return height, inside

    @torch.no_grad()
    def candidate_mask(self, env, ids, candidates, bounds,
                       support_threshold=7.0 / 9.0,
                       height_tolerance_m=0.045,
                       step_up_m=0.14, step_down_m=0.20):
        """Current-observation foothold feasibility, without layout metadata.

        Candidate targets are decoded through the public stance-foot contract.
        A 3x3 sole stencil is sampled from the same true terrain map delivered
        to the model.  This method does not read goals, support masks, future
        simulator state, or terminal labels.
        """
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        candidates = torch.as_tensor(
            candidates, dtype=torch.float32, device=self.device)
        swing = env.sampler.swing_foot[ids]
        stance = 1 - swing
        row = torch.arange(len(ids), device=self.device)
        foot_state = env.rigid_body_states[ids][:, env.feet_indices]
        stance_position = env.foot_positions[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_state[row, stance, 3:7])
        actions = candidates.unsqueeze(0).expand(len(ids), -1, -1)
        local = bounds.decode(
            actions, swing[:, None].expand(-1, actions.shape[1]))
        cosine, sine = torch.cos(stance_yaw)[:, None], torch.sin(stance_yaw)[:, None]
        target = torch.empty(len(ids), actions.shape[1], 2, device=self.device)
        target[..., 0] = (stance_position[:, None, 0]
                          + cosine * local[..., 0] - sine * local[..., 1])
        target[..., 1] = (stance_position[:, None, 1]
                          + sine * local[..., 0] + cosine * local[..., 1])
        target_yaw = stance_yaw[:, None] + local[..., 3]

        # Approximate the physical sole using its center, longitudinal +/-8 cm
        # and lateral +/-3.5 cm samples, rotated by candidate foot yaw.
        longitudinal = torch.tensor(
            [-0.08, 0.0, 0.08], device=self.device)
        lateral = torch.tensor([-0.035, 0.0, 0.035], device=self.device)
        sole_x, sole_y = torch.meshgrid(longitudinal, lateral, indexing="ij")
        sole_x, sole_y = sole_x.flatten(), sole_y.flatten()
        foot_cos = torch.cos(target_yaw)[..., None]
        foot_sin = torch.sin(target_yaw)[..., None]
        sample_xy = target[..., None, :].expand(-1, -1, 9, -1).clone()
        sample_xy[..., 0] += foot_cos * sole_x - foot_sin * sole_y
        sample_xy[..., 1] += foot_sin * sole_x + foot_cos * sole_y
        heights, inside = self._sample_world(sample_xy)
        center_height, center_inside = self._sample_world(target)
        stance_height, stance_inside = self._sample_world(stance_position[:, :2])
        coplanar = torch.abs(heights - center_height[..., None]) <= height_tolerance_m
        support = (coplanar & inside).float().mean(-1)
        reachable_height = (
            center_inside & stance_inside[:, None]
            & (center_height <= stance_height[:, None] + step_up_m)
            & (center_height >= stance_height[:, None] - step_down_m))
        valid = reachable_height & (support >= support_threshold)

        # Keep planning finite if discretization leaves a row empty. The most
        # supported reachable-height candidate is a safer explicit fallback
        # than allowing torch.topk to select an arbitrary -inf entry.
        empty = ~valid.any(-1)
        if empty.any():
            quality = support.masked_fill(~reachable_height, -1.0)
            fallback = quality.argmax(-1)
            valid[empty, fallback[empty]] = True
        return valid, support

    @torch.no_grad()
    def landing_support_probability(self, env, ids, candidates, bounds,
                                    landing_prediction, sigma_scale=1.5,
                                    support_threshold=7.0 / 9.0,
                                    height_tolerance_m=0.045,
                                    step_up_m=0.14, step_down_m=0.20):
        """Chance support after frozen-lower landing residual uncertainty."""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        candidates = torch.as_tensor(candidates, dtype=torch.float32,
                                     device=self.device)
        swing = env.sampler.swing_foot[ids]; stance = 1 - swing
        row = torch.arange(len(ids), device=self.device)
        foot_state = env.rigid_body_states[ids][:, env.feet_indices]
        stance_position = env.foot_positions[ids][row, stance]
        stance_yaw = quaternion_yaw(foot_state[row, stance, 3:7])
        actions = candidates[None].expand(len(ids), -1, -1)
        local = bounds.decode(actions, swing[:, None].expand(-1, len(candidates)))
        stance_cosine, stance_sine = torch.cos(stance_yaw)[:, None], torch.sin(stance_yaw)[:, None]
        target = torch.empty(len(ids), len(candidates), 2, device=self.device)
        target[..., 0] = (stance_position[:, None, 0]
                          + stance_cosine * local[..., 0]
                          - stance_sine * local[..., 1])
        target[..., 1] = (stance_position[:, None, 1]
                          + stance_sine * local[..., 0]
                          + stance_cosine * local[..., 1])
        target_yaw = stance_yaw[:, None] + local[..., 3]
        foot_cosine = torch.cos(target_yaw); foot_sine = torch.sin(target_yaw)

        ensemble_mean = landing_prediction["mean"]
        ensemble_std = landing_prediction["std"]
        # Moment-match the ensemble mixture before terrain sampling. This keeps
        # both aleatoric and epistemic variance while avoiding E times more GPU
        # height queries at every upper decision.
        mean = ensemble_mean.mean(0, keepdim=True)
        variance = (ensemble_std.square() + ensemble_mean.square()).mean(
            0, keepdim=True) - mean.square()
        std = variance.clamp_min(1e-6).sqrt()
        zero = torch.zeros_like(std[..., 0])
        sigma = torch.stack((
            torch.stack((zero, zero), -1),
            torch.stack((sigma_scale * std[..., 0], zero), -1),
            torch.stack((-sigma_scale * std[..., 0], zero), -1),
            torch.stack((zero, sigma_scale * std[..., 1]), -1),
            torch.stack((zero, -sigma_scale * std[..., 1]), -1)), dim=-2)
        residual = mean[..., None, :] + sigma  # [E,B,C,S,2], target frame
        centers = target[None, ..., None, :].expand(
            mean.shape[0], -1, -1, sigma.shape[-2], -1).clone()
        centers[..., 0] += (foot_cosine[None, ..., None] * residual[..., 0]
                            - foot_sine[None, ..., None] * residual[..., 1])
        centers[..., 1] += (foot_sine[None, ..., None] * residual[..., 0]
                            + foot_cosine[None, ..., None] * residual[..., 1])

        longitudinal = torch.tensor([-0.08, 0.0, 0.08], device=self.device)
        lateral = torch.tensor([-0.035, 0.0, 0.035], device=self.device)
        sole_x, sole_y = torch.meshgrid(longitudinal, lateral, indexing="ij")
        sole_x, sole_y = sole_x.flatten(), sole_y.flatten()
        sample_xy = centers[..., None, :].expand(*centers.shape[:-1], 9, 2).clone()
        sample_xy[..., 0] += (foot_cosine[None, ..., None, None] * sole_x
                              - foot_sine[None, ..., None, None] * sole_y)
        sample_xy[..., 1] += (foot_sine[None, ..., None, None] * sole_x
                              + foot_cosine[None, ..., None, None] * sole_y)
        heights, inside = self._sample_world(sample_xy)
        center_height, center_inside = self._sample_world(centers)
        stance_height, stance_inside = self._sample_world(stance_position[:, :2])
        coplanar = torch.abs(heights - center_height[..., None]) <= height_tolerance_m
        sole_support = (coplanar & inside).float().mean(-1)
        reachable_height = (
            center_inside & stance_inside[None, :, None, None]
            & (center_height <= stance_height[None, :, None, None] + step_up_m)
            & (center_height >= stance_height[None, :, None, None] - step_down_m))
        successful = reachable_height & (sole_support >= support_threshold)
        return successful.float().mean(dim=(0, 3))
