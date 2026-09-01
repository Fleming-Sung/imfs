"""Self-contained deterministic loader for the frozen model_7000 Actor."""

from pathlib import Path

import torch
import torch.nn as nn


def _actor(input_dim, hidden_dims, output_dim):
    layers = []
    current = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), nn.ELU()))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class FrozenLowerPolicy:
    actor_obs_dim = 30
    goal_dim = 16
    action_dim = 8

    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 2:
            raise ValueError("lower checkpoint must use format_version=2")
        cfg = checkpoint["config"]
        if cfg["asset"]["name"] != "SF_TRON1A":
            raise ValueError("lower checkpoint is not for SF_TRON1A")
        self.config = cfg
        self.device = torch.device(device)
        self.actor = _actor(
            self.actor_obs_dim + self.goal_dim,
            cfg["policy"]["actor_hidden_dims"], self.action_dim).to(self.device)
        actor_state = {
            key[len("actor."):]: value
            for key, value in checkpoint["actor_critic"].items()
            if key.startswith("actor.")
        }
        self.actor.load_state_dict(actor_state)
        self.actor.eval()

        normalizer = checkpoint["normalizer"]
        self.obs_mean = normalizer["actor_obs"]["mean"].to(self.device)
        self.obs_var = normalizer["actor_obs"]["var"].to(self.device)
        self.goal_mean = normalizer["goal"]["mean"].to(self.device)
        self.goal_var = normalizer["goal"]["var"].to(self.device)
        self.normalized_clip = float(normalizer["obs_clip"])
        self.action_clip = float(cfg["normalization"]["clip_actions"])

    @staticmethod
    def _normalize(value, mean, variance, clip):
        return ((value - mean) / torch.sqrt(variance + 1e-8)).clamp(-clip, clip)

    @torch.no_grad()
    def infer(self, raw_actor_obs, raw_goal):
        obs = torch.as_tensor(raw_actor_obs, dtype=torch.float32, device=self.device)
        goal = torch.as_tensor(raw_goal, dtype=torch.float32, device=self.device)
        if obs.shape[-1] != self.actor_obs_dim or goal.shape[-1] != self.goal_dim:
            raise ValueError(
                f"expected obs/goal dims {self.actor_obs_dim}/{self.goal_dim}, "
                f"got {obs.shape[-1]}/{goal.shape[-1]}")
        norm_obs = self._normalize(obs, self.obs_mean, self.obs_var, self.normalized_clip)
        norm_goal = self._normalize(goal, self.goal_mean, self.goal_var, self.normalized_clip)
        raw_action = self.actor(torch.cat((norm_obs, norm_goal), dim=-1))
        return raw_action, raw_action.clamp(-self.action_clip, self.action_clip)
