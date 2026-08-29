"""Exact deterministic Actor and frozen training-time observation normalization."""

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(46, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.net(x)


class FrozenRMS:
    def __init__(self, state, expected_dim, clip):
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32).reshape(-1)
        self.var = torch.as_tensor(state["var"], dtype=torch.float32).reshape(-1)
        self.clip = None if clip is None else float(clip)
        if self.mean.shape != (expected_dim,) or self.var.shape != (expected_dim,):
            raise ValueError(
                f"RMS shape mismatch: mean={tuple(self.mean.shape)}, var={tuple(self.var.shape)}, "
                f"expected={(expected_dim,)}")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.var).all() or (self.var < 0).any():
            raise ValueError("checkpoint RMS contains invalid values")

    def __call__(self, values):
        values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        out = (values - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(out, -self.clip, self.clip) if self.clip is not None else out


class FootholdPolicy:
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 2:
            raise ValueError("deployment requires a format-version-2 checkpoint")
        if int(checkpoint.get("iteration", -1)) != 6999:
            raise ValueError(f"expected model_7000 iteration 6999, got {checkpoint.get('iteration')}")
        state = checkpoint["actor_critic"]
        actor_state = {}
        layer_map = {"actor.0": "net.0", "actor.2": "net.2", "actor.4": "net.4", "actor.6": "net.6"}
        for source, destination in layer_map.items():
            actor_state[f"{destination}.weight"] = state[f"{source}.weight"]
            actor_state[f"{destination}.bias"] = state[f"{source}.bias"]
        self.actor = Actor().eval()
        self.actor.load_state_dict(actor_state, strict=True)

        normalizer = checkpoint["normalizer"]
        if normalizer.get("version") != 1:
            raise ValueError("unsupported normalizer version")
        clip = normalizer["obs_clip"]
        self.obs_rms = FrozenRMS(normalizer["actor_obs"], 30, clip)
        self.goal_rms = FrozenRMS(normalizer["goal"], 16, clip)
        self.checkpoint_config = checkpoint["config"]

    @torch.no_grad()
    def infer(self, observation, goal):
        obs_n = self.obs_rms(observation)
        goal_n = self.goal_rms(goal)
        actor_input = torch.cat((obs_n, goal_n)).unsqueeze(0)
        action = self.actor(actor_input).squeeze(0)
        return action.numpy().copy(), obs_n.numpy().copy(), goal_n.numpy().copy()
