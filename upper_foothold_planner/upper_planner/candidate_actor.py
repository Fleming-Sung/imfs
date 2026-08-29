"""Candidate foothold Actor for Gate B/C distillation.

Input is the deployable observation:

    depth      (B, 1, 64, 64)  normalized proximity [0, 1]
    proprio    (B, 36)         36-D upper observation (already embeds the
                               previous upper action and the relative route
                               goal + yaw error, see upper_state.build_proprio)

Output, all shaped (B, C) for the fixed C-candidate set:

    candidate_logits      classification logits (teacher argmax supervision)
    candidate_feasible    per-candidate feasibility logits (BCE on valid mask)
    candidate_progress    per-candidate geodesic progress (Huber)

An optional GRU turns the single-frame encoder into a history belief; it is
kept off by default so Gate B can be validated as plain behaviour cloning
before adding temporal belief.
"""

import torch
import torch.nn as nn

from .contracts import PROPRIO_DIM


def mlp(in_dim, hidden_dim, out_dim, final_activation=None):
    layers = [nn.Linear(in_dim, hidden_dim), nn.ELU(),
              nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
              nn.Linear(hidden_dim, out_dim)]
    if final_activation is not None:
        layers.append(final_activation())
    return nn.Sequential(*layers)


class CandidateActor(nn.Module):
    def __init__(self, num_candidates, proprio_dim=PROPRIO_DIM,
                 feature_dim=256, depth_feature_dim=128,
                 proprio_feature_dim=64, gru_hidden=0):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, depth_feature_dim), nn.ELU(),
        )
        self.proprio_encoder = mlp(proprio_dim, 128, proprio_feature_dim)
        encoder_dim = depth_feature_dim + proprio_feature_dim

        self.gru_hidden = int(gru_hidden)
        if self.gru_hidden > 0:
            self.gru = nn.GRU(encoder_dim, self.gru_hidden, batch_first=True)
            head_in = self.gru_hidden
        else:
            self.gru = None
            head_in = encoder_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, feature_dim), nn.ELU(),
            nn.Linear(feature_dim, feature_dim), nn.ELU())
        self.candidate_logits = nn.Linear(feature_dim, self.num_candidates)
        self.candidate_feasible = nn.Linear(feature_dim, self.num_candidates)
        self.candidate_progress = nn.Linear(feature_dim, self.num_candidates)

    def encode(self, depth, proprio):
        return torch.cat((self.depth_encoder(depth),
                          self.proprio_encoder(proprio)), dim=-1)

    def forward(self, depth, proprio, hidden=None):
        encoded = self.encode(depth, proprio)
        if self.gru is not None:
            encoded, hidden = self.gru(encoded.unsqueeze(1), hidden)
            encoded = encoded.squeeze(1)
        feature = self.head(encoded)
        return (self.candidate_logits(feature),
                self.candidate_feasible(feature),
                self.candidate_progress(feature)), hidden

    @torch.no_grad()
    def select(self, depth, proprio, hidden=None):
        (logits, _, _), hidden = self.forward(depth, proprio, hidden)
        return logits.argmax(dim=-1), hidden
