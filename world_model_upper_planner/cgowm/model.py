"""Structured latent model for frozen-lower foothold options.

The representation is split into geometry and robot-dynamics latents.  Exact
terrain and physics are training labels only; inference consumes depth and
deployable proprioception.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    def __init__(self, group_dim=8):
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value):
        if value.shape[-1] % self.group_dim:
            raise ValueError("latent dimension must be divisible by group_dim")
        shape = value.shape
        return torch.softmax(
            value.view(*shape[:-1], -1, self.group_dim), dim=-1).view(shape)


def mlp(input_dim, hidden_dim, output_dim, layers=2):
    modules = []
    width = int(input_dim)
    for _ in range(int(layers)):
        modules.extend((nn.Linear(width, hidden_dim), nn.ELU()))
        width = int(hidden_dim)
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


@dataclass(frozen=True)
class ModelConfig:
    proprio_dim: int = 36
    action_dim: int = 3
    geometry_dim: int = 64
    dynamics_dim: int = 64
    hidden_dim: int = 256
    action_embed_dim: int = 32
    q_ensemble: int = 2
    simnorm_group_dim: int = 8


class CandidateGroundedWorldModel(nn.Module):
    """Predict all candidate outcomes and an option-level latent transition."""

    outcome_names = (
        "reward", "progress", "support", "touchdown_error",
        "fall_logit", "collision_logit", "continuation_logit",
    )

    def __init__(self, candidates, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32)
        if candidate_tensor.ndim != 2 or candidate_tensor.shape[1] != self.config.action_dim:
            raise ValueError("candidates must have shape (C, action_dim)")
        self.register_buffer("candidates", candidate_tensor)

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 160), nn.ELU())
        self.proprio_encoder = mlp(self.config.proprio_dim, 128, 96)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(256, self.config.geometry_dim),
            nn.LayerNorm(self.config.geometry_dim),
            SimNorm(self.config.simnorm_group_dim))
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(96, self.config.dynamics_dim),
            nn.LayerNorm(self.config.dynamics_dim),
            SimNorm(self.config.simnorm_group_dim))
        self.action_encoder = mlp(
            self.config.action_dim, 64, self.config.action_embed_dim, layers=1)

        latent_dim = self.config.geometry_dim + self.config.dynamics_dim
        transition_input = latent_dim + self.config.action_embed_dim
        self.geometry_transition = mlp(
            transition_input, self.config.hidden_dim, self.config.geometry_dim)
        self.dynamics_transition = mlp(
            transition_input, self.config.hidden_dim, self.config.dynamics_dim)
        self.geometry_norm = SimNorm(self.config.simnorm_group_dim)
        self.dynamics_norm = SimNorm(self.config.simnorm_group_dim)

        candidate_input = latent_dim + self.config.action_embed_dim
        self.outcome_trunk = mlp(candidate_input, self.config.hidden_dim,
                                 self.config.hidden_dim)
        self.outcome_head = nn.Linear(
            self.config.hidden_dim, len(self.outcome_names))
        self.q_heads = nn.ModuleList([
            mlp(candidate_input, self.config.hidden_dim, 1)
            for _ in range(self.config.q_ensemble)])
        self.policy_head = mlp(latent_dim, self.config.hidden_dim,
                               candidate_tensor.shape[0])

    @property
    def latent_dim(self):
        return self.config.geometry_dim + self.config.dynamics_dim

    def encode(self, depth, proprio):
        depth_feature = self.depth_encoder(depth)
        proprio_feature = self.proprio_encoder(proprio)
        geometry = self.geometry_encoder(torch.cat((depth_feature, proprio_feature), -1))
        dynamics = self.dynamics_encoder(proprio_feature)
        return torch.cat((geometry, dynamics), -1)

    def split_latent(self, latent):
        return torch.split(
            latent, (self.config.geometry_dim, self.config.dynamics_dim), dim=-1)

    def next(self, latent, action):
        geometry, dynamics = self.split_latent(latent)
        action_feature = self.action_encoder(action)
        feature = torch.cat((latent, action_feature), -1)
        next_geometry = self.geometry_norm(geometry + self.geometry_transition(feature))
        next_dynamics = self.dynamics_norm(dynamics + self.dynamics_transition(feature))
        return torch.cat((next_geometry, next_dynamics), -1)

    def _candidate_features(self, latent, candidates=None):
        candidates = self.candidates if candidates is None else candidates
        if candidates.ndim == 2:
            candidates = candidates.unsqueeze(0).expand(latent.shape[0], -1, -1)
        if candidates.ndim != 3 or candidates.shape[0] != latent.shape[0]:
            raise ValueError("batched candidates must have shape (B,C,A)")
        action_feature = self.action_encoder(candidates)
        repeated = latent.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        return torch.cat((repeated, action_feature), -1)

    def predict_candidates(self, latent, candidates=None):
        features = self._candidate_features(latent, candidates)
        raw = self.outcome_head(self.outcome_trunk(features))
        result = {name: raw[..., index]
                  for index, name in enumerate(self.outcome_names)}
        result["support"] = torch.sigmoid(result["support"])
        result["touchdown_error"] = F.softplus(result["touchdown_error"])
        result["q"] = torch.stack(
            [head(features).squeeze(-1) for head in self.q_heads], dim=0)
        result["policy_logits"] = self.policy_head(latent)
        return result

    def predict_action(self, latent, action):
        output = self.predict_candidates(latent, action.unsqueeze(1))
        return {name: value[..., 0] if name != "q" else value[..., 0]
                for name, value in output.items()}

