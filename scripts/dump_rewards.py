"""dump_rewards.py —— 采集各奖励项加权后的均值，用于奖励报告排序。

用法（在 workspace 根目录）：
    python -m foothold.dump_rewards --num_envs 128 --num_steps 200
结果打印到终端，按加权绝对值从大到小排序。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params
from foothold.networks import ActorCritic


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 128},
        {"name": "--num_steps", "type": int, "default": 200},
    ]
    args = gymutil.parse_arguments(description="dump rewards", headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    num_goal = cfg.foothold.goal_dim
    ac = ActorCritic(env.num_obs + num_goal, env.num_critic_obs + num_goal, env.num_dof, cfg).to(env.device)

    obs, goal, critic_obs = env.get_observations()
    acc = {}
    for _ in range(args.num_steps):
        with torch.no_grad():
            action = ac.act(torch.cat((obs, goal), dim=-1))
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        for k, v in extras.get("reward_terms_weighted", {}).items():
            acc[k] = acc.get(k, 0.0) + v

    print(f"\n=== 各奖励项加权后均值（{args.num_steps} 步 × {args.num_envs} 环境，初始随机策略）===")
    print(f"dt={env.dt:.4f}  总奖励均值={sum(acc.values()) / args.num_steps:.4f}\n")
    rows = sorted(acc.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"{'reward term':24s} {'weighted_mean':>14s}  {'scale':>8s}")
    for k, v in rows:
        scale = cfg.rewards.scales.get(k, float("nan"))
        print(f"{k:24s} {v / args.num_steps:>14.5f}  {scale:>8.4g}")


if __name__ == "__main__":
    main()
