"""诊断：4096 环境下的地平面范围与冻结策略的回合长度。"""
from isaacgym import gymutil  # isaacgym 必须先于 torch 导入
import torch

from foothold.config import get_flat_config
from foothold.env import FootholdEnv, make_sim_params


def parse_args():
    custom = [
        {"name": "--num_envs", "type": int, "default": 4096},
    ]
    args = gymutil.parse_arguments(description="diag", headless=True, custom_parameters=custom)
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.compute_device_id}"
    return args


def main():
    args = parse_args()
    cfg = get_flat_config()
    cfg.env.num_envs = args.num_envs

    env = FootholdEnv(cfg, make_sim_params(cfg, args), args.sim_device, True)

    ox = env.env_origins[:, 0]
    oy = env.env_origins[:, 1]
    print(f"[origin] num_envs={args.num_envs} num_per_row={int(args.num_envs ** 0.5)}")
    print(f"[origin] x:[{ox.min().item():.1f}, {ox.max().item():.1f}] m  "
          f"y:[{oy.min().item():.1f}, {oy.max().item():.1f}] m")

    # 初始姿态（未步进）的脚高与基座高
    fp = env.foot_positions  # (N, 2, 3)
    print(f"[init] base_z={env.base_position[:, 2].mean().item():.4f} "
          f"foot_z_min={fp[:, :, 2].min().item():.4f} "
          f"foot_z_mean={fp[:, :, 2].mean().item():.4f} "
          f"foot_z_max={fp[:, :, 2].max().item():.4f}")
    print(f"[init] foot_z[0] = {fp[0, :, 2].tolist()}")

    # 冻结策略：动作恒为 0（PD 目标=名义姿态），不做任何 PPO 更新
    cur_len = torch.zeros(env.num_envs, device=env.device)
    action = torch.zeros(env.num_envs, env.num_dof, device=env.device)
    for it in range(4):
        ep_len, ep_count = 0.0, 0
        for _ in range(50):
            obs, rew, done, extras, goal, critic = env.step(action)
            cur_len += 1
            ids = done.nonzero(as_tuple=False).flatten()
            if ids.numel():
                ep_len += cur_len[ids].sum().item()
                ep_count += ids.numel()
                cur_len[ids] = 0
        # 按离原点距离分桶，看远端 env 是否有异常
        dist = env.env_origins[:, :2].norm(dim=1)
        near = dist < 50
        far = dist >= 100
        print(f"[frozen it={it}] mean_ep_len={ep_len / max(ep_count, 1):.1f} "
              f"ep_count={ep_count} "
              f"base_h_near={env.base_position[near, 2].mean().item():.3f} "
              f"base_h_far={env.base_position[far, 2].mean().item():.3f} "
              f"foot_z_near={env.foot_positions[near].min(dim=1)[0].min().item():.3f} "
              f"foot_z_far={env.foot_positions[far].min(dim=1)[0].min().item():.3f}")


if __name__ == "__main__":
    main()
