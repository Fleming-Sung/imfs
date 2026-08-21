"""train.py —— 训练入口（论文 PPO + 步长 curriculum）。

用法（在 workspace 根目录）：
    python -m foothold.train                # 默认 128 环境、开仿真窗口
    python -m foothold.train --headless     # 无渲染
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
from isaacgym import gymutil
import torch

from .config import get_flat_config
from .env import FootholdEnv, make_sim_params
from .networks import ActorCritic
from .ppo import PPO, Normalizer


CHECKPOINT_FORMAT_VERSION = 2


def checkpoint_state(iteration, actor_critic, normalizer, cfg):
    """构造可正确用于 eval 的完整策略 checkpoint。"""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "iteration": iteration,
        "actor_critic": actor_critic.state_dict(),
        "normalizer": normalizer.state_dict(),
        # 转为普通 dict/list/scalar，避免 checkpoint 依赖 AttrDict 的 pickle 类型。
        "config": json.loads(json.dumps(cfg)),
    }


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 128},
        {"name": "--max_iterations", "type": int, "default": 30000},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--run_name", "type": str, "default": ""},
        {"name": "--save_interval", "type": int, "default": 100},
        {"name": "--log_dir", "type": str, "default": None},
    ]
    args = gymutil.parse_arguments(description="foothold training (paper)",
                                   headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def make_log_dir(args):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = time.strftime("%b%d_%H-%M-%S")
    name = f"{stamp}_{args.run_name}" if args.run_name else stamp
    d = args.log_dir or os.path.join(root, "logs", name)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    cfg.runner.max_iterations = args.max_iterations
    cfg.runner.save_interval = args.save_interval

    log_dir = make_log_dir(args)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, args.headless)

    num_goal = cfg.foothold.goal_dim
    ac = ActorCritic(env.num_obs + num_goal, env.num_critic_obs + num_goal, env.num_dof, cfg).to(env.device)
    ppo = PPO(ac, cfg, env.device)
    normalizer = Normalizer(env.num_obs, num_goal, env.num_critic_obs, cfg.ppo.gamma, env.device,
                            cfg.normalization.running_obs_clip)
    ppo.init_storage(cfg.env.num_envs, cfg.runner.num_steps_per_env,
                     env.num_obs, env.num_critic_obs, num_goal, env.num_dof)

    obs, goal, critic_obs = env.get_observations()
    obs, goal, critic_obs = normalizer.observations(obs, goal, critic_obs, update=True)

    rewbuf, lenbuf = deque(maxlen=100), deque(maxlen=100)
    cur_reward = torch.zeros(cfg.env.num_envs, device=env.device)
    cur_len = torch.zeros(cfg.env.num_envs, device=env.device)

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
    except Exception:
        pass

    start_time = time.time()

    for it in range(cfg.runner.max_iterations):
        # ---- 步长 curriculum（论文：前 1000 轮从短步长线性涨到完整范围）----
        cur = cfg.foothold.step_distance_curriculum
        t = min(1.0, it / float(cur.ramp_iterations))
        cfg.foothold.step_distance = [cur.start[0] + (cur.end[0] - cur.start[0]) * t,
                                      cur.start[1] + (cur.end[1] - cur.start[1]) * t]

        t0 = time.time()
        ep_ret, ep_len, ep_done = 0.0, 0.0, 0
        step_reward = 0.0
        term_means = {}

        with torch.no_grad():
            for _ in range(cfg.runner.num_steps_per_env):
                action, value, log_prob, mu, sigma = ppo.act(obs, goal, critic_obs)
                next_obs, reward, done, extras, next_goal, next_critic_obs = env.step(action)
                absorbing = extras.get("absorbing", done.bool()).bool()

                norm_reward = normalizer.rewards(reward, done, update=True)
                ppo.storage.add(obs, critic_obs, goal, action, norm_reward,
                                done.bool(), absorbing, value, log_prob, mu, sigma)

                for k, v in extras.get("reward_terms_raw", {}).items():
                    term_means[f"reward_raw/{k}"] = term_means.get(f"reward_raw/{k}", 0.0) + v
                for k, v in extras.get("reward_terms_weighted", {}).items():
                    term_means[f"reward_w/{k}"] = term_means.get(f"reward_w/{k}", 0.0) + v

                cur_reward += reward
                step_reward += float(reward.mean().item())
                cur_len += 1
                done_ids = done.nonzero(as_tuple=False).flatten()
                if done_ids.numel() > 0:
                    ep_ret += cur_reward[done_ids].sum().item()
                    ep_len += cur_len[done_ids].sum().item()
                    ep_done += done_ids.numel()
                    rewbuf.extend(cur_reward[done_ids].cpu().tolist())
                    lenbuf.extend(cur_len[done_ids].cpu().tolist())
                    cur_reward[done_ids] = 0
                    cur_len[done_ids] = 0

                obs, goal, critic_obs = normalizer.observations(
                    next_obs, next_goal, next_critic_obs, update=True)

        # ---- GAE + PPO 更新 ----
        ppo.compute_returns(critic_obs, goal)
        vf, sur, kl = ppo.update()

        # ---- 日志 ----
        fps = int(cfg.runner.num_steps_per_env * cfg.env.num_envs / (time.time() - t0))
        elapsed = time.time() - start_time
        eta = elapsed / (it + 1) * (cfg.runner.max_iterations - it - 1)
        mean_std = float(torch.exp(ac.logstd).mean().item())

        if writer is not None:
            writer.add_scalar("loss/value", vf, it)
            writer.add_scalar("loss/surrogate", sur, it)
            writer.add_scalar("loss/kl", kl, it)
            writer.add_scalar("loss/lr", ppo.learning_rate, it)
            writer.add_scalar("policy/mean_std", mean_std, it)
            writer.add_scalar("perf/fps", fps, it)
            for k, v in term_means.items():
                writer.add_scalar(k, v / cfg.runner.num_steps_per_env, it)
            if ep_done > 0:
                writer.add_scalar("episode/mean_return", ep_ret / ep_done, it)
                writer.add_scalar("episode/mean_length", ep_len / ep_done, it)
            if rewbuf:
                writer.add_scalar("train/mean_reward", float(np.mean(rewbuf)), it)
                writer.add_scalar("train/mean_length", float(np.mean(lenbuf)), it)

        if it % cfg.runner.log_interval == 0:
            ep_r = f"{np.mean(rewbuf):7.2f}" if rewbuf else "    N/A"
            ep_l = f"{np.mean(lenbuf):6.1f}" if lenbuf else "   N/A"
            print(f"[{it:5d}/{cfg.runner.max_iterations - 1}] "
                  f"t={elapsed:8.1f}s ETA={eta:8.1f}s fps={fps:5d} "
                  f"rew={ep_r} len={ep_l} "
                  f"step_r={step_reward / cfg.runner.num_steps_per_env:.3f} "
                  f"std={mean_std:.3f} kl={kl:.4f} lr={ppo.learning_rate:.1e}")

        # ---- 保存 ----
        if (it + 1) % cfg.runner.save_interval == 0:
            torch.save(checkpoint_state(it, ac, normalizer, cfg),
                       os.path.join(log_dir, f"model_{it + 1}.pt"))

    torch.save(checkpoint_state(cfg.runner.max_iterations, ac, normalizer, cfg),
               os.path.join(log_dir, f"model_{cfg.runner.max_iterations}.pt"))
    print(f"日志目录: {log_dir}")


if __name__ == "__main__":
    main()
