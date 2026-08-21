"""check_actions.py —— 检查某个训练 run 的所有 checkpoint 输出动作是否在零附近。

每次进程只处理一个 run 目录（一次 create_sim，避免多次建 GPU sim 段错误）。
对 run 内每个 checkpoint：加载权重 → 跑 rollout → 统计采样动作 |a|、网络均值 |mu|、标准差 sigma。

用法（在 workspace 根目录）：
    python -m foothold.check_actions --run_dir foothold/logs/Aug20_23-03-09 \
        --num_envs 64 --num_steps 100 --threshold 0.1
结果打印到终端，并按 |a| 均值排序。
"""
import json
import os
import glob

from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import AttrDict
from foothold.env import FootholdEnv, make_sim_params
from foothold.networks import ActorCritic
from foothold.ppo import Normalizer


def parse_args():
    custom = [
        {"name": "--run_dir", "type": str, "default": None},
        {"name": "--num_envs", "type": int, "default": 64},
        {"name": "--num_steps", "type": int, "default": 100},
        {"name": "--warmup", "type": int, "default": 20},
        {"name": "--threshold", "type": float, "default": 0.1},
    ]
    args = gymutil.parse_arguments(description="check action magnitude", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = args.run_dir
    if run_dir is None:
        runs = sorted(glob.glob(os.path.join(root, "logs", "*")))
        if not runs:
            print("未找到 run 目录")
            return
        run_dir = runs[-1]
        print(f"未指定 --run_dir，使用最新的：{run_dir}")

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = AttrDict.from_nested(json.load(open(cfg_path)))
    cfg.env.num_envs = args.num_envs

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    num_goal = cfg.foothold.goal_dim
    ac = ActorCritic(env.num_obs + num_goal, env.num_critic_obs + num_goal, env.num_dof, cfg).to(env.device)
    checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")),
                         key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))

    print(f"\n=== {os.path.basename(run_dir)}：{len(checkpoints)} 个 checkpoint ===")
    print(f"num_envs={args.num_envs} num_steps={args.num_steps} warmup={args.warmup} "
          f"threshold={args.threshold}\n")
    print(f"{'iter':>6s} {'sigma':>8s} {'mean|mu|':>10s} {'mean|a|':>10s}")

    rows = []
    warmup = min(args.warmup, max(args.num_steps - 1, 0))
    for pt in checkpoints:
        it = os.path.basename(pt).split("_")[1].split(".")[0]
        ckpt = torch.load(pt, map_location=env.device)
        if "normalizer" not in ckpt:
            print(f"{it:>6s} {'N/A':>8s} {'N/A':>10s} {'N/A':>10s}  缺少 Normalizer，跳过")
            continue
        ac.load_state_dict(ckpt["actor_critic"])
        normalizer = Normalizer(env.num_obs, num_goal, env.num_critic_obs, cfg.ppo.gamma, env.device,
                                cfg.normalization.running_obs_clip)
        normalizer.load_state_dict(ckpt["normalizer"])

        obs, goal, critic_obs = env.get_observations()
        obs, goal, critic_obs = normalizer.observations(obs, goal, critic_obs, update=False)

        abs_a, abs_mu = [], []
        sigma = torch.clamp(torch.exp(ac.logstd), ac.min_std, ac.max_std).mean().item()

        with torch.no_grad():
            for step in range(args.num_steps):
                actor_in = torch.cat((obs, goal), dim=-1)
                ac._update_distribution(actor_in)
                mu = ac.action_mean
                a = ac.distribution.sample()
                if step >= warmup:
                    abs_mu.append(mu.abs().mean(dim=1))
                    abs_a.append(a.abs().mean(dim=1))
                obs, reward, done, extras, goal, critic_obs = env.step(a)
                obs, goal, critic_obs = normalizer.observations(obs, goal, critic_obs, update=False)

        abs_mu = torch.cat(abs_mu).mean().item() if abs_mu else 0.0
        abs_a = torch.cat(abs_a).mean().item() if abs_a else 0.0
        rows.append((it, sigma, abs_mu, abs_a))
        print(f"{it:>6s} {sigma:8.4f} {abs_mu:10.4f} {abs_a:10.4f}")

    rows.sort(key=lambda r: r[3])
    n_near = sum(1 for r in rows if r[3] < args.threshold)
    print(f"\n=== 结论（{os.path.basename(run_dir)}）===")
    print(f"共 {len(rows)} 个 checkpoint；mean|a| < {args.threshold} 的有 {n_near} 个")
    print("按 mean|a| 从小到大排序：")
    for it, sigma, abs_mu, abs_a in rows:
        print(f"  iter={it:>6s}  sigma={sigma:.4f}  mean|mu|={abs_mu:.4f}  mean|a|={abs_a:.4f}")


if __name__ == "__main__":
    main()
