"""diag_reset_correctness.py —— 验证高并行度下"判定需要重置的 env 是否真正在仿真器里被重置"。

零动作驱动，跑足够长的步数触发真实的摔倒→重置循环。
对每一步：记录上一步被重置( done=1 )的 env，在下一步模拟后它们的 base 高度分布：
  - 站起(z>0.5)      → 重置真正生效
  - 趴地(z<0.3)      → 重置未生效（仍趴着，说明 set_*_tensor_indexed 没写进 GPU）
并统计每个 episode 的存活帧数分布，看是否出现"重置后马上又失败"的异常短回合。
"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--num_steps", "type": int, "default": 120},
    ]
    args = gymutil.parse_arguments(description="diag reset correctness", headless=True,
                                   custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs
    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)
    lo, hi = cfg.env.healthy_height_range
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)

    print(f"num_envs={env.num_envs}  fail_to_terminal={env.fail_to_terminal}步  "
          f"healthy=[{lo},{hi}]\n")

    # 每个 env 的当前回合长度（本地跟踪，用于统计存活帧数）
    cur_len = torch.zeros(env.num_envs, device=env.device)
    ep_lens = []                     # 记录每个 episode 的长度
    total_reset = 0
    n_restand = 0                    # 重置后下一步站起的 env 数
    n_fallen = 0                     # 重置后下一步仍趴地的 env 数
    prev_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for step in range(args.num_steps):
        obs, reward, done, extras, goal, critic_obs = env.step(action)
        done_b = done.bool()
        cur_len += 1

        # 1) 检查"上一步被重置"的 env 在本次模拟后是否站起
        if prev_reset.any():
            z = env.base_position[prev_reset, 2] - env.env_origins[prev_reset, 2]
            s = int((z > 0.5).sum().item())
            f = int((z < 0.3).sum().item())
            n_restand += s
            n_fallen += f
            if (step - 1) % 20 == 0 or f > 0:
                print(f"[step {step-1:3d}→{step}] 上一步重置 {int(prev_reset.sum().item()):5d} 个，"
                      f"本次站起 {s:5d}，仍趴地 {f:5d}，z_min={z.min().item():.3f} z_max={z.max().item():.3f}")

        # 2) 统计 episode 长度
        if done_b.any():
            ids = done_b.nonzero(as_tuple=False).flatten()
            ep_lens.extend(cur_len[ids].cpu().tolist())
            cur_len[ids] = 0
            total_reset += ids.numel()

        prev_reset = done_b.clone()

    # 汇总
    import numpy as np
    ep = np.array(ep_lens) if ep_lens else np.array([0])
    print(f"\n=== 汇总 ===")
    print(f"总重置次数: {total_reset}")
    print(f"重置后下一步: 站起 {n_restand}，仍趴地 {n_fallen}（趴地=重置未生效）")
    print(f"episode 长度: min={ep.min()} max={ep.max()} mean={ep.mean():.1f} "
          f"<10步占比={float((ep < 10).mean()):.3f} <25步占比={float((ep < 25).mean()):.3f}")


if __name__ == "__main__":
    main()
