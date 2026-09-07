"""Finite-horizon option risk critic for constrained latent planning."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import mlp


@dataclass(frozen=True)
class RiskConfig:
    latent_dim: int = 128
    action_dim: int = 3
    hidden_dim: int = 256
    ensemble: int = 3


class OptionRiskCritic(nn.Module):
    """Predict failure within a short option horizon, not only next-step done."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or RiskConfig()
        self.action_encoder = mlp(self.config.action_dim, 64, 32, layers=1)
        self.heads = nn.ModuleList([
            mlp(self.config.latent_dim + 32, self.config.hidden_dim, 2)
            for _ in range(self.config.ensemble)])

    def _features(self, latent, action):
        if action.ndim == 2:
            action = action[:, None]
        action_feature = self.action_encoder(action)
        repeated = latent[:, None].expand(-1, action.shape[1], -1)
        return torch.cat((repeated, action_feature), -1)

    def predict(self, latent, action):
        """Return ensemble logits shaped (E,B,C) for fall and collision."""
        features = self._features(latent, action)
        raw = torch.stack([head(features) for head in self.heads], 0)
        return {"fall_logit": raw[..., 0], "collision_logit": raw[..., 1]}

    def predict_action(self, latent, action):
        return {name: value[..., 0]
                for name, value in self.predict(latent, action).items()}

    def predict_candidates(self, latent, candidates):
        if candidates.ndim == 2:
            candidates = candidates[None].expand(latent.shape[0], -1, -1)
        return self.predict(latent, candidates)
