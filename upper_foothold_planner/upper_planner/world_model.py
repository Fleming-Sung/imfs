"""Small readable latent world model; no external TD-MPC implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import PROPRIO_DIM, UPPER_ACTION_DIM, MACRO_STATE_DIM


def mlp(in_dim, hidden_dim, out_dim, final_activation=None):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ELU(),
              nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, out_dim)]
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class SimNorm(nn.Module):
    """Group-wise simplex normalization used to prevent latent saturation."""

    def __init__(self, group_dim=8):
        super().__init__()
        self.group_dim = int(group_dim)

    def forward(self, value):
        if value.shape[-1] % self.group_dim:
            raise ValueError("latent dimension must be divisible by SimNorm group_dim")
        shape = value.shape
        grouped = value.view(*shape[:-1], -1, self.group_dim)
        return torch.softmax(grouped, dim=-1).view(shape)


class LatentWorldModel(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ELU(),
        )
        self.proprio_encoder = mlp(PROPRIO_DIM, 128, 64)
        self.fuse = nn.Sequential(nn.Linear(192, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.dynamics = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, latent_dim)
        self.reward = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.value = mlp(latent_dim, hidden_dim, 1)
        # These small auxiliary heads make safety events and visual geometry
        # identifiable instead of allowing the scalar reward loss to ignore them.
        self.collision = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.fall = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.progress = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.goal = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.off_support = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, 1)
        self.depth_decoder = nn.Linear(latent_dim, 16 * 16)

    def encode(self, depth, proprio):
        return self.fuse(torch.cat((self.depth_encoder(depth), self.proprio_encoder(proprio)), -1))

    def next(self, latent, action):
        delta = self.dynamics(torch.cat((latent, action), -1))
        return torch.tanh(latent + delta)

    def predict_reward(self, latent, action):
        return self.reward(torch.cat((latent, action), -1)).squeeze(-1)

    def predict_value(self, latent):
        return self.value(latent).squeeze(-1)

    def predict_event_logits(self, latent, action):
        state_action = torch.cat((latent, action), -1)
        return (self.collision(state_action).squeeze(-1),
                self.fall(state_action).squeeze(-1))

    def predict_task_components(self, latent, action):
        state_action = torch.cat((latent, action), -1)
        collision, fall = self.predict_event_logits(latent, action)
        return {
            "progress": self.progress(state_action).squeeze(-1),
            "goal_logit": self.goal(state_action).squeeze(-1),
            "collision_logit": collision,
            "fall_logit": fall,
            "off_support_logit": self.off_support(state_action).squeeze(-1),
        }

    def predict_task_reward(self, latent, action, reward_cfg, reward_scale):
        """Expected normalized macro reward assembled from explicit task terms."""
        component = self.predict_task_components(latent, action)
        scale = float(reward_scale)
        return (component["progress"] + float(reward_cfg["time"]) / scale
                + float(reward_cfg["goal"]) / scale
                * torch.sigmoid(component["goal_logit"])
                + float(reward_cfg["collision"]) / scale
                * torch.sigmoid(component["collision_logit"])
                + float(reward_cfg["fall"]) / scale
                * torch.sigmoid(component["fall_logit"])
                + float(reward_cfg["off_support"]) / scale
                * torch.sigmoid(component["off_support_logit"]))

    def reconstruct_depth(self, latent):
        return torch.sigmoid(self.depth_decoder(latent)).view(-1, 1, 16, 16)

    def predict_next_depth(self, latent, action):
        """Decode the depth expected after applying one upper-level action."""
        return self.reconstruct_depth(self.next(latent, action))


class SpatialLatentWorldModel(LatentWorldModel):
    """Higher-resolution variant for terrain edges and small obstacles."""

    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__(latent_dim, hidden_dim)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ELU(),
        )
        self.depth_decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4), nn.ELU())
        self.depth_decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )
        # Remove the compact linear decoder so it is not optimized or saved.
        del self.depth_decoder

    def reconstruct_depth(self, latent):
        feature = self.depth_decoder_input(latent).view(-1, 64, 4, 4)
        return self.depth_decoder_conv(feature)


class TaskLatentWorldModel(nn.Module):
    """Task-sufficient latent dynamics with no observation decoder.

    Depth remains an essential input.  The latent is trained to preserve only
    information needed to predict the outcome of candidate footholds: progress,
    safety events, continuous safety margins, and the next compact task state.
    """

    # The latent dynamics is grounded by the next macro motion state (see
    # MACRO_STATE_DIM), not by reconstructing the full next observation.

    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 160), nn.ELU(),
        )
        self.proprio_encoder = mlp(PROPRIO_DIM, 128, 96)
        self.fuse = nn.Sequential(
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())
        self.dynamics = mlp(latent_dim + UPPER_ACTION_DIM, hidden_dim, latent_dim)
        self.value = mlp(latent_dim, hidden_dim, 1)

        state_action_dim = latent_dim + UPPER_ACTION_DIM
        self.progress = mlp(state_action_dim, hidden_dim, 1)
        self.heading_progress = mlp(state_action_dim, hidden_dim, 1)
        self.collision = mlp(state_action_dim, hidden_dim, 1)
        self.fall = mlp(state_action_dim, hidden_dim, 1)
        self.goal = mlp(state_action_dim, hidden_dim, 1)
        self.off_support = mlp(state_action_dim, hidden_dim, 1)
        self.collision_force = mlp(state_action_dim, hidden_dim, 1)
        self.stability_margin = mlp(state_action_dim, hidden_dim, 1)
        self.support_fraction = mlp(state_action_dim, hidden_dim, 1)
        self.touchdown_error = mlp(state_action_dim, hidden_dim, 1)
        self.task_state = mlp(latent_dim, hidden_dim, MACRO_STATE_DIM)

    def encode(self, depth, proprio):
        features = torch.cat(
            (self.depth_encoder(depth), self.proprio_encoder(proprio)), dim=-1)
        return self.fuse(features)

    def next(self, latent, action):
        delta = self.dynamics(torch.cat((latent, action), dim=-1))
        return torch.tanh(latent + delta)

    def predict_value(self, latent):
        return self.value(latent).squeeze(-1)

    def predict_event_logits(self, latent, action):
        state_action = torch.cat((latent, action), dim=-1)
        return (self.collision(state_action).squeeze(-1),
                self.fall(state_action).squeeze(-1))

    def predict_task_components(self, latent, action):
        state_action = torch.cat((latent, action), dim=-1)
        collision, fall = self.predict_event_logits(latent, action)
        return {
            "progress": self.progress(state_action).squeeze(-1),
            "heading_progress": self.heading_progress(state_action).squeeze(-1),
            "goal_logit": self.goal(state_action).squeeze(-1),
            "collision_logit": collision,
            "fall_logit": fall,
            "off_support_logit": self.off_support(state_action).squeeze(-1),
            "collision_force": F.softplus(
                self.collision_force(state_action).squeeze(-1)),
            "stability_margin": self.stability_margin(state_action).squeeze(-1),
            "support_fraction": torch.sigmoid(
                self.support_fraction(state_action).squeeze(-1)),
            "touchdown_error": F.softplus(
                self.touchdown_error(state_action).squeeze(-1)),
        }

    def predict_task_reward(self, latent, action, reward_cfg, reward_scale):
        component = self.predict_task_components(latent, action)
        scale = float(reward_scale)
        return (component["progress"] + float(reward_cfg["time"]) / scale
                + float(reward_cfg["goal"]) / scale
                * torch.sigmoid(component["goal_logit"])
                + float(reward_cfg["collision"]) / scale
                * torch.sigmoid(component["collision_logit"])
                + float(reward_cfg["fall"]) / scale
                * torch.sigmoid(component["fall_logit"])
                + float(reward_cfg["off_support"]) / scale
                * torch.sigmoid(component["off_support_logit"]))

    def predict_reward(self, latent, action):
        raise RuntimeError(
            "task world model has no redundant scalar reward head; "
            "call predict_task_reward with the reward configuration")

    def predict_task_state(self, latent):
        return self.task_state(latent)


class OptionTaskWorldModel(TaskLatentWorldModel):
    """Semi-Markov task world model with twin Q critics.

    It keeps the decoder-free, task-sufficient observation contract of the
    task model, but replaces the saturating tanh latent with SimNorm and adds
    option duration, continuation, and action-conditioned Q predictions.
    """

    def __init__(self, latent_dim=128, hidden_dim=256, q_ensemble=2):
        super().__init__(latent_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim), SimNorm(8))
        self.latent_norm = SimNorm(8)
        state_action_dim = latent_dim + UPPER_ACTION_DIM
        self.option_duration = mlp(state_action_dim, hidden_dim, 1)
        self.continuation = mlp(state_action_dim, hidden_dim, 1)
        self.q_functions = nn.ModuleList([
            mlp(state_action_dim, hidden_dim, 1)
            for _ in range(int(q_ensemble))])
        self.policy = mlp(latent_dim, hidden_dim, UPPER_ACTION_DIM)

    def next(self, latent, action):
        delta = self.dynamics(torch.cat((latent, action), dim=-1))
        return self.latent_norm(latent + delta)

    def policy_action(self, latent):
        """Deterministic policy prior: latent -> normalized action in [-1, 1]."""
        return torch.tanh(self.policy(latent))

    def predict_task_components(self, latent, action):
        component = super().predict_task_components(latent, action)
        state_action = torch.cat((latent, action), dim=-1)
        component.update({
            # Normalized by the nominal 25 lower ticks during training.
            "option_duration": F.softplus(
                self.option_duration(state_action).squeeze(-1)),
            "continuation_logit": self.continuation(
                state_action).squeeze(-1),
        })
        return component

    def predict_q(self, latent, action):
        state_action = torch.cat((latent, action), dim=-1)
        return torch.stack([
            function(state_action).squeeze(-1)
            for function in self.q_functions], dim=0)

    def predict_min_q(self, latent, action):
        return self.predict_q(latent, action).min(dim=0).values

    def predict_terminal_value(self, latent, num_actions=16, aggregation="min"):
        """Approximate max_a aggregate_i Q_i(z,a) at an MPC terminal state."""
        batch = latent.shape[0]
        action = 2.0 * torch.rand(
            batch, int(num_actions), UPPER_ACTION_DIM,
            dtype=latent.dtype, device=latent.device) - 1.0
        repeated = latent[:, None].expand(-1, int(num_actions), -1)
        q_ensemble = self.predict_q(
            repeated.reshape(-1, latent.shape[-1]),
            action.reshape(-1, UPPER_ACTION_DIM))
        if aggregation == "min":
            q = q_ensemble.min(dim=0).values
        elif aggregation == "mean":
            q = q_ensemble.mean(dim=0)
        else:
            raise ValueError("terminal Q aggregation must be min or mean")
        q = q.view(batch, int(num_actions))
        return q.max(dim=1).values


def make_world_model(latent_dim=128, hidden_dim=256, variant="compact"):
    if variant == "compact":
        return LatentWorldModel(latent_dim, hidden_dim)
    if variant == "spatial":
        return SpatialLatentWorldModel(latent_dim, hidden_dim)
    if variant == "task":
        return TaskLatentWorldModel(latent_dim, hidden_dim)
    if variant == "option":
        return OptionTaskWorldModel(latent_dim, hidden_dim)
    raise ValueError("model_variant must be compact, spatial, task, or option")
