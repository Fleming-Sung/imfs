"""Privileged critic for asymmetric PPO fine-tuning (Gate E).

The Actor sees only the deployable observation (depth + 36-D proprio).  The
Critic additionally sees privileged information: the exact geodesic distance
to the goal, the true support fraction under the stance foot, and absolute
stance/base world positions.
"""

import torch
import torch.nn as nn

from .contracts import PROPRIO_DIM

# proprio (36) + geodesic (1) + support (1) + stance_xy (2) + base_xy (2)
PRIVILEGED_DIM = PROPRIO_DIM + 6


@torch.no_grad()
def make_privileged(proprio, env, planner, ids):
    """Append privileged extras to the deployable proprio for the critic."""
    ids = torch.as_tensor(ids, dtype=torch.long, device=proprio.device)
    swing = env.sampler.swing_foot[ids]
    stance = 1 - swing
    row = torch.arange(ids.numel(), device=proprio.device)
    stance_xy = env.foot_positions[ids][row, stance, :2]
    geodesic = planner.geodesic_distance(env, ids, stance_xy)
    support = planner.support_fraction_at(env, ids, stance_xy)
    base_xy = env.base_position[ids, :2]
    extra = torch.cat((geodesic[:, None], support[:, None], stance_xy, base_xy),
                      dim=-1)
    return torch.cat((proprio, extra), dim=-1)


class PrivilegedCritic(nn.Module):
    def __init__(self, input_dim=PRIVILEGED_DIM, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, privileged):
        return self.net(privileged).squeeze(-1)
