"""networks.py —— Actor-Critic 网络（论文 ELU MLP [512,256,128]，正交初始化）。"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ACTIVATIONS = {"elu": nn.ELU, "selu": nn.SELU, "relu": nn.ReLU, "tanh": nn.Tanh}


def _mlp(in_dim, hidden_dims, out_dim, activation, orthogonal_init):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if orthogonal_init:
            nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
            nn.init.constant_(layers[-1].bias, 0.0)
        layers.append(ACTIVATIONS[activation]())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if orthogonal_init:
        nn.init.orthogonal_(layers[-1].weight, 0.01)
        nn.init.constant_(layers[-1].bias, 0.0)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """对角高斯策略 + 状态价值。std 为可学习参数，限制在 [min_std, max_std]。"""

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, cfg):
        super().__init__()
        p = cfg.policy
        self.actor = _mlp(num_actor_obs, p.actor_hidden_dims, num_actions,
                          p.activation, p.orthogonal_init)
        self.critic = _mlp(num_critic_obs, p.critic_hidden_dims, 1,
                           p.activation, p.orthogonal_init)
        self.logstd = nn.Parameter(torch.full((num_actions,), float(np.log(p.init_noise_std))))
        self.min_std = p.min_std
        self.max_std = p.max_std
        self.distribution = None

    def _update_distribution(self, obs):
        mean = self.actor(obs)
        std = torch.clamp(torch.exp(self.logstd), self.min_std, self.max_std)
        self.distribution = Normal(mean, std)

    def act(self, obs):
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_obs):
        return self.critic(critic_obs)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
