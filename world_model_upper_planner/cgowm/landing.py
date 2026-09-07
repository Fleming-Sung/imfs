"""Closed-loop landing distribution for a frozen lower controller."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import mlp


@dataclass(frozen=True)
class LandingConfig:
    latent_dim: int = 128
    action_dim: int = 3
    hidden_dim: int = 256
    ensemble: int = 5
    minimum_std_m: float = 0.01
    maximum_std_m: float = 0.25


class LandingDistributionCritic(nn.Module):
    """Predict target-frame XY residual mean and aleatoric uncertainty."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or LandingConfig()
        self.action_encoder = mlp(self.config.action_dim, 64, 32, layers=1)
        self.heads = nn.ModuleList([
            mlp(self.config.latent_dim + 32, self.config.hidden_dim, 4)
            for _ in range(self.config.ensemble)])

    def predict(self, latent, action):
        if action.ndim == 2:
            action = action[:, None]
        action_feature = self.action_encoder(action)
        repeated = latent[:, None].expand(-1, action.shape[1], -1)
        feature = torch.cat((repeated, action_feature), -1)
        raw = torch.stack([head(feature) for head in self.heads], 0)
        mean = raw[..., :2]
        span = self.config.maximum_std_m - self.config.minimum_std_m
        std = self.config.minimum_std_m + span * torch.sigmoid(raw[..., 2:])
        return {"mean": mean, "std": std}

    def predict_action(self, latent, action):
        return {name: value[..., 0, :] for name, value in
                self.predict(latent, action).items()}

    def predict_candidates(self, latent, candidates):
        if candidates.ndim == 2:
            candidates = candidates[None].expand(latent.shape[0], -1, -1)
        return self.predict(latent, candidates)
