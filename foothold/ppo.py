"""ppo.py —— 手写 PPO（GAE、clipped surrogate、clipped value、自适应学习率）+ 归一化。

损失与学习率调度与论文发布代码（mind_steps/ppo.py、rollout.py、normalization.py）一致：
  - 近似 KL：mean((r - 1) - log(r))，r 为新旧策略概率比
  - 价值损失：0.5 * max((V - R)², (V_clipped - R)²)
  - 自适应学习率：KL > desired_kl*kl_margin 时 lr/=kl_scale，KL < desired_kl/kl_margin 时 lr*=kl_scale
  - 奖励归一化：按 gamma 折扣回报的运行方差归一
"""

import torch
import torch.nn as nn
import torch.optim as optim


def approximate_kl(ratio):
    eps = 1e-7
    return torch.mean((ratio - 1.0 + eps) - torch.log(ratio + eps))


def clipped_value_loss(value, old_value, target, clip_param):
    value_clipped = old_value + (value - old_value).clamp(-clip_param, clip_param)
    return 0.5 * torch.max((value - target) ** 2, (value_clipped - target) ** 2).mean()


# --------------------------------------------------------------------------
# 运行归一化
# --------------------------------------------------------------------------

class RunningMeanStd:
    def __init__(self, shape, device, epsilon=1e-4, batch_var_epsilon=0.0):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(epsilon, device=device)
        self.batch_var_epsilon = batch_var_epsilon

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, *self.mean.shape)
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False) + self.batch_var_epsilon
        n = values.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        new_mean = self.mean + delta * n / total
        old_m2 = self.var * self.count
        batch_m2 = batch_var * n
        new_var = (old_m2 + batch_m2 + delta ** 2 * self.count * n / total) / total
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(new_var, min=1e-8))
        self.count.copy_(total)

    def normalize(self, values, clip=None):
        out = (values - self.mean) / torch.sqrt(self.var + 1e-8)
        if clip is not None:
            out = torch.clamp(out, -clip, clip)
        return out


class Normalizer:
    """归一化 actor 观测、goal、critic 观测，以及按折扣回报归一奖励。"""

    def __init__(self, obs_dim, goal_dim, critic_dim, gamma, device, obs_clip=10.0):
        self.actor_obs = RunningMeanStd((obs_dim,), device, epsilon=1e-6, batch_var_epsilon=1e-6)
        self.goal = RunningMeanStd((goal_dim,), device, epsilon=1e-6, batch_var_epsilon=1e-6)
        self.critic_obs = RunningMeanStd((critic_dim,), device, epsilon=1e-6, batch_var_epsilon=1e-6)
        self.return_rms = RunningMeanStd((), device)
        self.returns = None
        self.gamma = gamma
        self.obs_clip = obs_clip

    @torch.no_grad()
    def observations(self, obs, goal, critic_obs, update=True):
        if update:
            self.actor_obs.update(obs)
            self.goal.update(goal)
            self.critic_obs.update(critic_obs)
        return (self.actor_obs.normalize(obs, self.obs_clip),
                self.goal.normalize(goal, self.obs_clip),
                self.critic_obs.normalize(critic_obs, self.obs_clip))

    @torch.no_grad()
    def rewards(self, rewards, dones, update=True):
        if self.returns is None or self.returns.shape != rewards.shape:
            self.returns = torch.zeros_like(rewards)
        self.returns = self.gamma * self.returns * (~dones.bool()).float() + rewards
        if update:
            self.return_rms.update(self.returns)
        return rewards / torch.sqrt(self.return_rms.var + 1e-8)


# --------------------------------------------------------------------------
# Rollout 存储
# --------------------------------------------------------------------------

class RolloutStorage:
    def __init__(self, num_envs, num_steps, obs_dim, critic_dim, goal_dim, action_dim, device):
        self.num_envs, self.num_steps, self.device = num_envs, num_steps, device
        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.critic_obs = torch.zeros(num_steps, num_envs, critic_dim, device=device)
        self.goal = torch.zeros(num_steps, num_envs, goal_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(num_steps, num_envs, 1, device=device).byte()
        self.absorbing = torch.zeros(num_steps, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_steps, num_envs, 1, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, 1, device=device)
        self.mu = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.sigma = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.returns = torch.zeros(num_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, 1, device=device)
        self.step = 0

    def add(self, obs, critic_obs, goal, action, reward, done, absorbing, value, log_prob, mu, sigma):
        self.obs[self.step].copy_(obs)
        self.critic_obs[self.step].copy_(critic_obs)
        self.goal[self.step].copy_(goal)
        self.actions[self.step].copy_(action)
        self.rewards[self.step, :, 0] = reward
        self.dones[self.step, :, 0] = done
        self.absorbing[self.step, :, 0] = absorbing
        self.values[self.step].copy_(value)
        self.log_probs[self.step].copy_(log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        for t in reversed(range(self.num_steps)):
            next_values = last_values if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t].float()
            not_absorbing = 1.0 - self.absorbing[t].float()
            delta = self.rewards[t] + not_absorbing * gamma * next_values - self.values[t]
            advantage = delta + not_done * gamma * lam * advantage
            self.returns[t] = advantage + self.values[t]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std(unbiased=False) + 1e-8)

    def generator(self, num_mini_batches, num_epochs):
        flat = lambda x: x.flatten(0, 1)
        batch_size = self.num_envs * self.num_steps
        mini_size = batch_size // num_mini_batches
        for _ in range(num_epochs):
            idx = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                b = idx[i * mini_size:(i + 1) * mini_size]
                yield (flat(self.obs)[b], flat(self.critic_obs)[b], flat(self.goal)[b],
                       flat(self.actions)[b], flat(self.values)[b], flat(self.returns)[b],
                       flat(self.advantages)[b], flat(self.log_probs)[b])

    def clear(self):
        self.step = 0


# --------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------

class PPO:
    def __init__(self, actor_critic, cfg, device):
        p = cfg.ppo
        self.ac = actor_critic
        self.device = device
        self.clip_param = p.clip_param
        self.num_learning_epochs = p.num_learning_epochs
        self.num_mini_batches = p.num_mini_batches
        self.value_loss_coef = p.value_loss_coef
        self.entropy_coef = p.entropy_coef
        self.gamma = p.gamma
        self.lam = p.lam
        self.max_grad_norm = p.max_grad_norm
        self.use_clipped_value_loss = p.use_clipped_value_loss
        self.schedule = p.schedule
        self.desired_kl = p.desired_kl
        self.min_learning_rate = p.min_learning_rate
        self.max_learning_rate = p.max_learning_rate
        self.kl_margin = p.kl_margin
        self.kl_scale = p.kl_scale
        self.learning_rate = p.learning_rate

        optimizer_cls = optim.AdamW if p.optimizer == "adamw" else optim.Adam
        self.optimizer = optimizer_cls(self.ac.parameters(), lr=p.learning_rate,
                                       eps=p.adam_epsilon, weight_decay=p.weight_decay)
        self.storage = None

    def init_storage(self, num_envs, num_steps, obs_dim, critic_dim, goal_dim, action_dim):
        self.storage = RolloutStorage(num_envs, num_steps, obs_dim, critic_dim,
                                      goal_dim, action_dim, self.device)

    def act(self, obs, goal, critic_obs):
        actor_input = torch.cat((obs, goal), dim=-1)
        critic_input = torch.cat((critic_obs, goal), dim=-1)
        with torch.no_grad():
            action = self.ac.act(actor_input)
            value = self.ac.evaluate(critic_input)
            log_prob = self.ac.get_actions_log_prob(action)
            mu = self.ac.action_mean
            sigma = self.ac.action_std
        return action, value, log_prob, mu, sigma

    def compute_returns(self, last_critic_obs, last_goal):
        with torch.no_grad():
            last_values = self.ac.evaluate(torch.cat((last_critic_obs, last_goal), dim=-1))
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, adapt_learning_rate=True):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl = 0.0
        n = 0
        for (obs_b, critic_obs_b, goal_b, actions_b, values_b, returns_b,
             advantages_b, old_log_prob_b) in self.storage.generator(
                self.num_mini_batches, self.num_learning_epochs):
            actor_input = torch.cat((obs_b, goal_b), dim=-1)
            critic_input = torch.cat((critic_obs_b, goal_b), dim=-1)
            self.ac._update_distribution(actor_input)
            log_prob_b = self.ac.get_actions_log_prob(actions_b)
            value_b = self.ac.evaluate(critic_input)
            entropy_b = self.ac.entropy.mean()

            ratio = torch.exp(log_prob_b - old_log_prob_b.squeeze(-1))
            kl = approximate_kl(ratio)
            surrogate = -advantages_b.squeeze(-1) * ratio
            surrogate_clipped = -advantages_b.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_loss = clipped_value_loss(value_b, values_b, returns_b, self.clip_param)
            else:
                value_loss = 0.5 * (returns_b - value_b) ** 2

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_b

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if adapt_learning_rate and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    if kl > self.desired_kl * self.kl_margin:
                        self.learning_rate = max(self.min_learning_rate, self.learning_rate / self.kl_scale)
                    elif kl < self.desired_kl / self.kl_margin:
                        self.learning_rate = min(self.max_learning_rate, self.learning_rate * self.kl_scale)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            n += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl.item()

        n = max(n, 1)
        self.storage.clear()
        return mean_value_loss / n, mean_surrogate_loss / n, mean_kl / n
